"""Post-hoc LDC/CADC transfer and prototype-only margin-risk diagnostics."""
import argparse
import ast
import csv
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.stats import pearsonr, spearmanr

from metrics import summarize_group_logits

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT/'manifest.json').read_text())
CACHE = Path(MANIFEST['cache'])
PREVIOUS = CACHE.parent


def write(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)


def load(path):
    return torch.load(path,map_location='cpu',weights_only=True)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)


def data(task):
    path=CACHE/f'task_{task:03d}.pt'
    expected=json.loads((PREVIOUS/'cache_manifest.json').read_text())[f'task_{task:03d}.json']
    assert sha(path)==expected['cache_sha256']
    a=load(path)
    assert a['task']==task and set(a['new_prototypes'])==set(range(2*task,2*task+2))
    return a


def fit_map(source,target,kind,task,device):
    """Official post-hoc Adam/MSE recipes in HOEP's unit-feature convention."""
    seed=1993+task*1000
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=nn.Linear(source.shape[1],source.shape[1],bias=kind=='cadc').to(device)
    if kind=='cadc':
        with torch.no_grad():model.weight.copy_(torch.eye(source.shape[1],device=device));model.bias.zero_()
    x=F.normalize(source.float(),dim=1).to(device)
    y=F.normalize(target.float(),dim=1).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    generator=torch.Generator().manual_seed(seed+17)
    curve=[]
    for epoch in range(20):
        order=torch.randperm(len(x),generator=generator)
        for rows in order.split(128):
            rows=rows.to(device)
            loss=F.mse_loss(model(x[rows]),y[rows])
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        with torch.no_grad():curve.append(float(F.mse_loss(model(x),y)))
    state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    assert all(torch.isfinite(v).all() for v in state.values())
    return state,curve


def compensate(prototypes,state,kind):
    prediction=F.linear(prototypes,state['weight'],state.get('bias'))
    if kind=='cadc':
        drift=(prediction-prototypes).norm(dim=1)
        alpha=torch.sigmoid(drift/(drift.max()+1e-8))
    else:alpha=torch.ones(len(prototypes))
    updated=F.normalize(prototypes+alpha[:,None]*(prediction-prototypes),dim=1)
    return updated,alpha


def run_compensation(kind):
    setup();device=torch.device('cuda:0')
    variants=['ldc'] if kind=='ldc' else ['cadc','cadc_uniform']
    prototypes={v:torch.empty(0,768) for v in variants}
    curves={v:[] for v in variants};matrices={v:[] for v in variants}
    for task in range(50):
        a=data(task)
        if task:
            state,losses=fit_map(a['source_train_features'],a['target_train_features'],kind,task,device)
            save(ROOT/'maps'/kind/f'task_{task:03d}.pt',dict(state=state,losses=losses))
        for variant in variants:
            if task:
                prototypes[variant],alpha=compensate(prototypes[variant],state,
                     'cadc' if variant=='cadc' else 'uniform')
            else:alpha=torch.empty(0)
            before=prototypes[variant].clone()
            new=torch.stack([a['new_prototypes'][c] for c in range(task*2,task*2+2)])
            prototypes[variant]=torch.cat([before,new])
            assert torch.equal(prototypes[variant][-2:],new)
            features=F.normalize(a['test_features'],dim=1).to(device)
            scores=(features@F.normalize(prototypes[variant],dim=1).to(device).T).cpu()
            labels=a['test_labels'];accuracy=100.0*float((scores.argmax(1)==labels).double().mean())
            record=summarize_group_logits(scores,labels,task*2) if task else {}
            record.update(task=task,global_top1=accuracy,alpha_mean=float(alpha.mean()) if task else None)
            out=ROOT/'experiments'/variant
            save(out/f'task_{task:03d}.pt',dict(prototypes=prototypes[variant].clone(),
                 logits=scores,labels=labels,sample_ids=a['test_ids'],alpha=alpha))
            write(out/f'task_{task:03d}.json',record)
            curves[variant].append(accuracy)
            per_task=[100.0*float((scores.argmax(1)[(labels//2)==j]==labels[(labels//2)==j]).double().mean())
                      for j in range(task+1)]
            matrices[variant].append(per_task)
            summary=dict(variant=variant,completed_tasks=task+1,complete=task==49,curve=curves[variant],
                  final_accuracy=accuracy,average_incremental_accuracy=float(np.mean(curves[variant])),
                  accuracy_matrix=matrices[variant],scope=MANIFEST['scope'])
            if task==49:
                summary['forgetting']=float(np.mean([max(matrices[variant][k][j] for k in range(j,50))-
                                                    matrices[variant][-1][j] for j in range(49)]))
            write(out/'metrics.json',summary)
            print('COMPENSATION',variant,task+1,accuracy,flush=True)


def correlation(x,y):
    x=np.asarray(x);y=np.asarray(y)
    if np.std(x)<1e-12 or np.std(y)<1e-12:return dict(pearson=None,spearman=None)
    return dict(pearson=float(pearsonr(x,y).statistic),spearman=float(spearmanr(x,y).statistic))


def grouped_predictions(rows,keys,outcome,scheme):
    x=np.array([[r[k] for k in keys] for r in rows],dtype=float)
    y=np.array([r[outcome] for r in rows],dtype=float)
    group=np.array([(r['class_id']//2)%5 if scheme=='class' else (r['task']-1)//10 for r in rows])
    prediction=np.empty(len(rows))
    for g in np.unique(group):
        train=group!=g;test=~train
        mean=x[train].mean(0);std=x[train].std(0);std[std<1e-10]=1
        xt=np.column_stack([np.ones(train.sum()),(x[train]-mean)/std])
        xv=np.column_stack([np.ones(test.sum()),(x[test]-mean)/std])
        penalty=np.eye(xt.shape[1]);penalty[0,0]=0
        coef=np.linalg.solve(xt.T@xt+penalty,xt.T@y[train])
        prediction[test]=xv@coef
    error=prediction-y
    return dict(mae=float(np.abs(error).mean()),rmse=float(np.sqrt(np.square(error).mean())),
                prediction_correlation=correlation(prediction,y))


def summarize_rows(rows):
    output={}
    baseline=['age','task','reference_centroid_margin']
    models={'controls':baseline,'cosine':baseline+['cosine_error'],
            'projection':baseline+['projected_pair_error'],
            'margin_risk':baseline+['margin_risk_ratio'],
            'cosine_and_margin':baseline+['cosine_error','margin_risk_ratio']}
    for variant in sorted(set(r['variant'] for r in rows)):
        selected=[r for r in rows if r['variant']==variant]
        output[variant]=dict(rows=len(selected),correlations={},grouped_cv={})
        for outcome in ['net_accuracy_gain_pp','fix_rate_pp','old_accuracy_loss_pp']:
            output[variant]['correlations'][outcome]={key:correlation([r[key] for r in selected],
                 [r[outcome] for r in selected]) for key in ['cosine_error','projected_pair_error',
                   'margin_risk_ratio','r_rec','r_weighted','age']}
            output[variant]['grouped_cv'][outcome]={scheme:{name:grouped_predictions(selected,keys,outcome,scheme)
                  for name,keys in models.items()} for scheme in ['class','task_block']}
    return output


def run_margins():
    setup();device=torch.device('cuda:0')
    variants=['none','current','half','reliable','ldc','cadc','cadc_uniform']
    rows=[];exact_errors=[];boundary_rows=[]
    source=Path(MANIFEST['diagnosis_source'])
    previous_accuracy={}
    for variant in variants:
        folder=(PREVIOUS if variant in ['none','current','half','reliable'] else ROOT)/'experiments'/variant
        first=load(folder/'task_000.pt')
        previous_accuracy[variant]={c:100*float((first['logits'].argmax(1)[first['labels']==c]==c).double().mean()) for c in range(2)}
    for task in range(1,50):
        a=data(task);reference=load(source/f'task_{task:03d}.pt')
        details=json.loads((source/f'task_{task:03d}.json').read_text())
        oracle=dict(reference['offline_refreshed_old_prototypes']);oracle.update(a['new_prototypes'])
        q=F.normalize(torch.stack([oracle[c] for c in range(2*task+2)]),dim=1)
        z=F.normalize(a['test_features'],dim=1).to(device);labels=a['test_labels']
        qo=q.to(device);oracle_scores=z@qo.T;oracle_prediction=oracle_scores.argmax(1)
        competitors=oracle_scores.clone();competitors.scatter_(1,labels.to(device)[:,None],float('-inf'))
        j_oracle=competitors.argmax(1)
        margin_oracle=oracle_scores.gather(1,labels.to(device)[:,None]).squeeze(1)-competitors.max(1).values
        similarity=q@q.T;similarity.fill_diagonal_(float('-inf'))
        neighbours=similarity.topk(min(5,len(q)-1),dim=1).indices
        aggregate=details['aggregate']
        r_rec=float(aggregate.get('r_rec',aggregate.get('normalized_irrecoverable_residual',0)))
        r_weighted=float(aggregate.get('r_weighted',0))
        for variant in variants:
            folder=(PREVIOUS if variant in ['none','current','half','reliable'] else ROOT)/'experiments'/variant
            saved=load(folder/f'task_{task:03d}.pt')
            assert torch.equal(saved['sample_ids'],a['test_ids']) and torch.equal(saved['labels'],labels)
            p=F.normalize(saved['prototypes'],dim=1)
            assert torch.equal(p[-2:],F.normalize(q[-2:],dim=1)) or torch.allclose(p[-2:],q[-2:],atol=1e-7)
            scores=z@p.to(device).T
            assert float((scores.cpu()-saved['logits']).abs().max())<1e-5
            pred=scores.argmax(1);y=labels.to(device);error=p-q;eg=error.to(device)
            masked=scores.clone();masked.scatter_(1,y[:,None],float('-inf'));j_saved=masked.argmax(1)
            margin_saved=scores.gather(1,y[:,None]).squeeze(1)-masked.max(1).values
            error_term=(z*(eg[y]-eg[j_saved])).sum(1)
            competitor_switch=(z*(qo[j_oracle]-qo[j_saved])).sum(1)
            max_error=float(((margin_saved-margin_oracle)-error_term-competitor_switch).abs().max())
            assert max_error<1e-5;exact_errors.append(max_error)
            saved_correct=pred==y;oracle_correct=oracle_prediction==y
            fix=~saved_correct & oracle_correct;harm=saved_correct & ~oracle_correct
            sample=dict(sample_ids=a['test_ids'],labels=labels,
                 exact_margin_change=(margin_saved-margin_oracle).cpu(),error_projection=error_term.cpu(),
                 competitor_switch=competitor_switch.cpu(),saved_correct=saved_correct.cpu(),
                 oracle_correct=oracle_correct.cpu(),fix=fix.cpu(),harm=harm.cpu())
            save(ROOT/'margin_samples'/variant/f'task_{task:03d}.pt',sample)
            for c in range(task*2):
                js=neighbours[c];axis=F.normalize(q[c][None]-q[js],dim=1)
                pair_error=error[c][None]-error[js]
                projected=float((pair_error*axis).sum(1).square().mean())
                ref_margins=(q[c][None]*(q[c][None]-q[js])).sum(1)
                negative=(-(q[c][None]*pair_error).sum(1)).clamp_min(0)
                risk=float(negative.max());ratio=float((negative/ref_margins.clamp_min(1e-6)).max())
                mask=y==c
                current_acc=100*float(saved_correct[mask].double().mean())
                row=dict(variant=variant,task=task,class_id=c,age=task-c//2,
                    cosine_error=float((1-(p[c]*q[c]).sum()).clamp_min(0)),
                    projected_pair_error=projected,margin_risk=risk,margin_risk_ratio=ratio,
                    reference_centroid_margin=float(ref_margins.min()),
                    r_rec=r_rec,r_weighted=r_weighted,
                    saved_accuracy=current_acc,
                    previous_accuracy=previous_accuracy[variant][c],
                    old_accuracy_loss_pp=previous_accuracy[variant][c]-current_acc,
                    oracle_accuracy=100*float(oracle_correct[mask].double().mean()),
                    net_accuracy_gain_pp=100*float((oracle_correct[mask].double()-saved_correct[mask].double()).mean()),
                    fix_rate_pp=100*float(fix[mask].double().mean()),harm_rate_pp=100*float(harm[mask].double().mean()))
                rows.append(row)
            previous_accuracy[variant]={c:100*float(saved_correct[y==c].double().mean()) for c in range(2*task+2)}
            old=y<2*task
            boundary_rows.append(dict(variant=variant,task=task,r_rec=r_rec,r_weighted=r_weighted,
                old_net_gain_pp=100*float((oracle_correct[old].double()-saved_correct[old].double()).mean()),
                oracle_global=100*float(oracle_correct.double().mean()),saved_global=100*float(saved_correct.double().mean())))
        write(ROOT/'margin_progress.json',dict(completed_boundaries=task,total_boundaries=49,
                                               class_boundary_rows=len(rows),updated=time.time()))
        print('MARGIN_BOUNDARY',task,'rows',len(rows),flush=True)
    with (ROOT/'class_boundary_metrics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(ROOT/'boundary_metrics.json',boundary_rows)
    summary=summarize_rows(rows)
    boundary_correlations={}
    for variant in variants:
        grouped=[[r for r in rows if r['variant']==variant and r['task']==task] for task in range(1,50)]
        loss=[np.mean([r['old_accuracy_loss_pp'] for r in group]) for group in grouped]
        boundary_correlations[variant]={key:correlation([np.mean([r[key] for r in group]) for group in grouped],loss)
            for key in ['cosine_error','margin_risk_ratio','r_rec','r_weighted']}
    write(ROOT/'margin_summary.json',dict(complete=True,scope=MANIFEST['diagnostic'],variants=summary,
         boundary_correlations_to_old_accuracy_loss=boundary_correlations,
         exact_margin_decomposition_max_abs=max(exact_errors),
         note='All oracle-derived predictors are diagnostic only. Repeated classes, samples and task boundaries are dependent; grouped CV is exploratory, not a causal or seed-level test.'))


def checks():
    """Compare transplanted CADC update to the actual upstream method body."""
    setup();torch.manual_seed(23)
    prototypes=F.normalize(torch.randn(7,12),dim=1)
    state=dict(weight=torch.eye(12)+0.01*torch.randn(12,12),bias=torch.randn(12)*0.001)
    ours,alpha=compensate(prototypes,state,'cadc')
    path=ROOT/'vendor/CADC/models/CADC.py'
    tree=ast.parse(path.read_text())
    method=next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='CADC'
                for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='_update_memory')
    module=ast.Module(body=[method],type_ignores=[])
    namespace=dict(torch=torch,np=np,F=F,logging=logging)
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace)
    class Mapper(nn.Module):
        def forward(self,x):return dict(logits=F.linear(x,state['weight'],state['bias']))
    class Wrapper:
        drift_estimator=Mapper()
    class Fake:
        _network=Wrapper();_device=torch.device('cpu');_known_classes=7
        drift_scale=1.0;drift_weight=True;drift_dist='euclidean'
        _protos=[v.numpy().copy() for v in prototypes]
        def _drift_estimator(self,loader):pass
    fake=Fake();namespace['_update_memory'](fake,None)
    expected=F.normalize(torch.from_numpy(np.array(fake._protos)),dim=1)
    assert torch.allclose(ours,expected,atol=1e-7)
    assert torch.all(alpha>=0.5) and torch.all(alpha<=torch.sigmoid(torch.tensor(1.0))+1e-7)
    uniform,_=compensate(prototypes,state,'uniform')
    assert torch.allclose(uniform,F.normalize(F.linear(prototypes,state['weight'],state['bias']),dim=1),atol=1e-7)
    # Algebraic margin identity including competitor switching.
    z=F.normalize(torch.randn(17,12),dim=1);q=F.normalize(torch.randn(7,12),dim=1)
    p=F.normalize(q+torch.randn_like(q)*0.2,dim=1);y=torch.arange(17)%7
    a=z@q.T;b=z@p.T;am=a.clone();bm=b.clone()
    am.scatter_(1,y[:,None],float('-inf'));bm.scatter_(1,y[:,None],float('-inf'))
    jo=am.argmax(1);js=bm.argmax(1);e=p-q
    lhs=(b[torch.arange(17),y]-bm.max(1).values)-(a[torch.arange(17),y]-am.max(1).values)
    rhs=(z*(e[y]-e[js])).sum(1)+(z*(q[jo]-q[js])).sum(1)
    assert torch.allclose(lhs,rhs,atol=2e-7)
    write(ROOT/'checks.json',dict(passed=True,cadc_upstream_method_max_abs=float((ours-expected).abs().max()),
                                 margin_identity_max_abs=float((lhs-rhs).abs().max())))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['check','ldc','cadc','margin'])
    action=parser.parse_args().action
    if action=='check':checks()
    elif action=='margin':run_margins()
    else:run_compensation(action)
