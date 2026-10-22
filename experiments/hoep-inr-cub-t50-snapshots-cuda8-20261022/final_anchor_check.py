"""Early terminal-only fixed-reference diagnostic using original final models.

This is a Task-49 comparison, not the 50-boundary replay. The full replay
service produces the incremental curve after snapshot training finishes.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--dataset',choices=['imagenetr','cub'],required=True)
parser.add_argument('--device',default='cuda:0')
args=parser.parse_args()
manifest=json.loads((ROOT/'manifest.json').read_text())
entry=manifest['datasets'][args.dataset]
sys.path.insert(0,entry['source'])
os.chdir(entry['source'])
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from backbone.coordinate_stability import fit_residual_orthogonal_transport,_apply_matrix
from scripts.evaluate_sa_sdlora import build_merged_backbone
from utils.data_manager import DataManager
from utils.hoep_snapshots import audit_snapshot


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,item):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');torch.save(item,tmp);tmp.replace(path)


def write(path,item):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(item,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def load(path):return torch.load(path,map_location='cpu',weights_only=True)


def set_stage(model,state):
    with torch.no_grad():
        for layer,block in enumerate(model.blocks):
            w=block.attn.qkv
            for offset,suffix in ((0,'q'),(1,'v')):
                i=layer*2+offset
                getattr(w,'a_'+suffix).copy_(state['shared_a'][i].to(args.device))
                getattr(w,'b_'+suffix).copy_(state['merged_b'][i].to(args.device))
    model.eval()


def features(model,dataset):
    loader=DataLoader(dataset,batch_size=64,shuffle=False,num_workers=2,pin_memory=True)
    ids,feats,labels=[],[],[]
    with torch.no_grad():
        for i,x,y in loader:
            ids.append(i.cpu());labels.append(y.cpu())
            feats.append(model(x.to(args.device,non_blocking=True)).float().cpu())
    return torch.cat(ids),torch.cat(feats),torch.cat(labels)


def centroids(feats,labels,classes):
    norm=F.normalize(feats.float(),dim=1)
    return torch.stack([F.normalize(norm[labels==c].mean(0),dim=0) for c in classes])


def metric(logits,labels):
    pred=logits.argmax(1)
    old=labels<196
    return dict(top1=100*float((pred==labels).double().mean()),
                old=100*float((pred[old]==labels[old]).double().mean()),
                new=100*float((pred[~old]==labels[~old]).double().mean()),
                old_to_new=100*float((pred[old]>=196).double().mean()),
                count=len(labels))


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    out=ROOT/'terminal_check'/args.dataset
    snap=Path(entry['outputs']['smoke'])/'task_snapshots/task_000'
    audit_snapshot(snap)
    cfg=json.loads((snap/'config.json').read_text())
    manager=DataManager(cfg['dataset'],cfg['shuffle'],cfg['seed'],4,4,cfg)
    task0=load(snap/'sa_merged_lora.pt')
    model=build_merged_backbone(timm.create_model('vit_base_patch16_224',pretrained=True,
                               num_classes=0),task0).to(args.device).eval()
    with torch.no_grad():probe=model(load(snap/'pre_merge.pt')['probe_inputs'].to(args.device)).cpu()
    probe_error=float((probe-load(snap/'deployed_probe_features.pt')).abs().max())
    assert probe_error<1e-4
    reference_path=out/'reference_all_train.pt'
    if reference_path.is_file():
        ref=load(reference_path)
    else:
        train=manager.get_dataset(np.arange(200),source='train',mode='test')
        ids,feat,labels=features(model,train)
        assert set(labels.tolist())==set(range(200))
        current_mask=labels>=196
        ref=dict(ids=ids,labels=labels,prototypes=centroids(feat,labels,range(200)),
                 current_train_ids=torch.arange(int(current_mask.sum())),
                 current_train_features=feat[current_mask],
                 current_train_labels=labels[current_mask])
        save(reference_path,ref)
        print('REFERENCE',args.dataset,len(ids),flush=True)
    on_dir=Path(entry['reference_output'])
    off_dir=Path('/home/hongzhijun/hongshan/hoep-crossdataset-anchor-final-20261022/inputs')/(args.dataset+'_off')
    on_state=load(on_dir/'sa_merged_lora.pt')
    off_state=load(off_dir/'sa_merged_lora.pt')
    state_diff=max(float((a-b).abs().max()) for key in ['shared_a','merged_b']
                   for a,b in zip(on_state[key],off_state[key]))
    assert state_diff==0.,state_diff
    set_stage(model,on_state)
    last=manager.get_dataset(np.arange(196,200),source='train',mode='test')
    ids,target,labels=features(model,last)
    assert torch.equal(ids,ref['current_train_ids'])
    assert torch.equal(labels,ref['current_train_labels'])
    on_proto=load(on_dir/'sa_prototypes.pt')
    off_proto=load(off_dir/'sa_prototypes.pt')
    current_new=centroids(target,labels,range(196,200))
    new_error=float((current_new-torch.stack([on_proto[c] for c in range(196,200)])).abs().max())
    assert new_error<1e-4,new_error
    mapping=fit_residual_orthogonal_transport(ref['current_train_features'],target,
            rank=cfg['sa_coordinate_transport_rank'],
            identity_reg=cfg['sa_coordinate_transport_reg'],
            min_validation_gain=cfg['sa_coordinate_transport_min_gain'])
    old_ref=ref['prototypes'][:196]
    moved=F.normalize(_apply_matrix(old_ref.float(),mapping['basis'],mapping['rotation']),dim=1)
    on=torch.stack([on_proto[c] for c in range(200)])
    off=torch.stack([off_proto[c] for c in range(200)])
    off_new_diff=float((on[196:]-off[196:]).abs().max())
    heads={'pt_on_deployed':on,'pt_off_deployed':off,
           'pt_off_fixed_new':torch.cat((off[:196],on[196:])),
           'anchor_none':torch.cat((old_ref,on[196:])),
           'anchor_half':torch.cat((F.normalize((old_ref+moved)/2,dim=1),on[196:])),
           'anchor_full':torch.cat((moved,on[196:]))}
    test=manager.get_dataset(np.arange(200),source='test',mode='test')
    test_ids,test_features,test_labels=features(model,test)
    scores={name:F.normalize(test_features,dim=1)@F.normalize(head,dim=1).T
            for name,head in heads.items()}
    metrics={name:metric(logits,test_labels) for name,logits in scores.items()}
    save(out/'paired_logits.pt',dict(scores=scores,heads=heads,labels=test_labels,
         test_ids=test_ids,mapping=mapping,reference_prototypes=ref['prototypes']))
    write(out/'result.json',dict(dataset=args.dataset,seed=cfg['seed'],task=49,
          protocol='same original on/off final backbone; Task0 reference; Task49 training data only for map; same new prototypes in all fixed-new arms',
          task0_snapshot_sha256=sha(snap/'complete.json'),
          on_state_sha256=sha(on_dir/'sa_merged_lora.pt'),
          off_state_sha256=sha(off_dir/'sa_merged_lora.pt'),
          state_max_abs=state_diff,task0_probe_max_abs=probe_error,
          new_prototype_max_abs=new_error,off_new_prototype_max_abs=off_new_diff,
          map_enabled=bool(mapping['enabled']),map_train_samples=len(ids),
          test_samples=len(test_ids),metrics=metrics,
          paired_logits_sha256=sha(out/'paired_logits.pt'),
          reference_all_train_sha256=sha(reference_path)))
    print('TERMINAL',args.dataset,metrics,flush=True)


if __name__=='__main__':main()
