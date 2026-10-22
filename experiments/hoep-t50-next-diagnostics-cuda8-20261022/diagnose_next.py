"""Fixed HOEP trajectory: oracle-gated transport and fixed-reference transport.

The oracle arms are retrospective diagnostics: their gates require old training
images represented by the current model. The anchor arms use only current-task
training images, a fixed Task-0 reference encoder, and stored class centroids.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
META = json.loads((ROOT / 'manifest.json').read_text())
PREV = Path(META['previous_runtime'])
ORACLE = Path(META['oracle_runtime']) / 'inputs/oracle_diag'
sys.path.insert(0, str(PREV))
import fourway as fw  # noqa: E402
from backbone.coordinate_stability import fit_residual_orthogonal_transport  # noqa: E402
from scripts.diagnose_hoep import mean_prototypes  # noqa: E402
from scripts.hoep_metrics import summarize_group_logits, paired_prediction_changes  # noqa: E402


def write(path, item):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(item, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def save(path, item):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(item, tmp)
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(task):
    path = PREV / 'cache' / f'task_{task:03d}.pt'
    marker = json.loads((PREV / 'cache' / f'task_{task:03d}.json').read_text())
    assert sha(path) == marker['cache_sha256']
    return fw.load(path)


def reference_old(task):
    path = ORACLE / f'task_{task:03d}.pt'
    recorded = META['oracle_input_sha256'][f'task_{task:03d}.pt']
    assert sha(path) == recorded
    return fw.load(path)['offline_refreshed_old_prototypes']


def new_rows(task, data):
    assert data['old_count'] == 2 * task
    return torch.stack([data['new_prototypes'][c] for c in (2 * task, 2 * task + 1)])


def head_scores(data, prototypes):
    return F.normalize(data['test_features'].float(), dim=1) @ F.normalize(prototypes.float(), dim=1).T


def metric_step(kind, task, data, old_rows, alpha, previous, extra=None):
    protos = torch.cat((old_rows, new_rows(task, data)))
    assert len(protos) == 2 * (task + 1)
    assert torch.equal(protos[-2:], new_rows(task, data))
    scores = head_scores(data, protos)
    labels = data['test_labels']
    assert torch.equal(labels, previous['labels'])
    assert torch.equal(data['test_ids'], previous['sample_ids'])
    accuracy = 100 * float((scores.argmax(1) == labels).double().mean())
    record = summarize_group_logits(scores, labels, 2 * task) if task else {}
    record.update(task=task, global_top1=accuracy,
                  alpha_mean=float(alpha.float().mean()) if len(alpha) else None,
                  alpha_zero=int((alpha == 0).sum()), alpha_one=int((alpha == 1).sum()),
                  paired_vs_no_transport=paired_prediction_changes(previous['logits'], scores, labels, 2 * task)
                  if task else None)
    out = ROOT / 'experiments' / kind
    save(out / f'task_{task:03d}.pt', dict(prototypes=protos, logits=scores,
         labels=labels, sample_ids=data['test_ids'], alpha=alpha, extra=extra or {}))
    write(out / f'task_{task:03d}.json', record)
    return protos, scores, record


def summarize(kind, curve, matrix, records):
    forgetting = float(np.mean([max(matrix[k][j] for k in range(j, 50)) - matrix[-1][j] for j in range(49)])) if len(curve) == 50 else None
    summary = dict(kind=kind, completed_tasks=len(curve), complete=len(curve) == 50,
                   final_accuracy=curve[-1], average_incremental_accuracy=float(np.mean(curve)),
                   curve=curve, accuracy_matrix=matrix, forgetting=forgetting,
                   final_old=records[-1].get('old'), final_new=records[-1].get('new'),
                   backbone='unchanged fixed HOEP5%+LS CIFAR50 seed1993 NCoff trajectory',
                   cache_manifest_sha256=META['cache_manifest_sha256'])
    write(ROOT / 'experiments' / kind / 'metrics.json', summary)


def oracle_choices(old, mapping, q, variant):
    old = F.normalize(old.float(), dim=1)
    q = F.normalize(q.float(), dim=1)
    n = len(old)
    choices = torch.tensor(META['oracle_alphas'])
    moved = fw.moved(old, mapping)
    candidate = F.normalize(old[None] + choices[:, None, None] * (moved[None] - old[None]), dim=2)
    if variant == 'oracle_cosine':
        risk = 1 - (candidate * q[None, :n]).sum(2)
        best = risk.argmin(0)
    elif variant == 'oracle_margin':
        sim = q @ q.T
        sim.fill_diagonal_(-float('inf'))
        js = sim.topk(min(5, len(q)-1), dim=1).indices[:n]
        qc = q[:n]
        margins = (qc[:, None] * (qc[:, None] - q[js])).sum(2).clamp_min(1e-6)
        competitor_error = old[js.clamp(max=n-1)] - q[js]
        competitor_error = torch.where((js < n)[:, :, None], competitor_error, torch.zeros_like(competitor_error))
        e = candidate - qc[None]
        signed = -(qc[None, :, None, :] * (e[:, :, None, :] - competitor_error[None])).sum(3)
        risk = (signed.clamp_min(0) / margins[None]).max(2).values
        # If positive risk ties, prefer the candidate closer to the reference;
        # argmin's stable first occurrence selects the smaller alpha last.
        cos = 1 - (candidate * qc[None]).sum(2)
        score = risk + 1e-5 * cos
        best = score.argmin(0)
    else:
        raise ValueError(variant)
    alpha = choices[best]
    chosen = candidate[best, torch.arange(n)]
    return chosen, alpha, dict(candidate_risk=risk[best, torch.arange(n)],
                               oracle_cosine_error=(1-(chosen*q[:n]).sum(1)))


def run_oracle():
    fw.setup()
    variants = ['oracle_cosine', 'oracle_margin']
    protos = {v: torch.empty((0, 768)) for v in variants}
    curves = {v: [] for v in variants}
    matrices = {v: [] for v in variants}
    records = {v: [] for v in variants}
    for task in range(50):
        data = load_data(task)
        baseline = fw.load(PREV / 'experiments/none' / f'task_{task:03d}.pt')
        q_old = reference_old(task) if task else None
        if task:
            q = F.normalize(torch.stack([q_old[c] for c in range(task * 2)] +
                  [data['new_prototypes'][c] for c in range(task * 2, task * 2 + 2)]), dim=1)
        for variant in variants:
            if task:
                old, alpha, diagnostic = oracle_choices(protos[variant], data['maps']['primary'], q, variant)
            else:
                old, alpha, diagnostic = protos[variant], torch.empty(0), {}
            protos[variant], scores, rec = metric_step(variant, task, data, old, alpha, baseline, diagnostic)
            curves[variant].append(rec['global_top1'])
            labels = data['test_labels']
            matrices[variant].append([100 * float((scores.argmax(1)[labels//2 == j] == labels[labels//2 == j]).double().mean())
                                      for j in range(task+1)])
            records[variant].append(rec)
            summarize(variant, curves[variant], matrices[variant], records[variant])
        print('ORACLE', task, *(f'{v}={curves[v][-1]:.2f}' for v in variants), flush=True)


def extract_anchor():
    fw.setup()
    import timm
    from utils.data_manager import DataManager
    from scripts.diagnose_hoep import build_stage_backbone
    from utils.hoep_snapshots import audit_snapshot
    snap = PREV / 'inputs/task_snapshots/task_000'
    audit_snapshot(snap)
    cfg = json.loads((snap / 'config.json').read_text())
    manager = DataManager(cfg['dataset'], cfg['shuffle'], cfg['seed'], 2, 2, cfg)
    state = fw.load(snap / 'sa_merged_lora.pt')
    model = build_stage_backbone(timm.create_model('vit_base_patch16_224', pretrained=True,
                                 num_classes=0), state).cuda().eval()
    with torch.no_grad():
        probe = model(fw.load(snap / 'pre_merge.pt')['probe_inputs'].cuda()).cpu()
    expected = fw.load(snap / 'deployed_probe_features.pt')
    err = float((probe-expected).abs().max())
    assert err < 1e-4, ('Task0 reference probe', err)
    write(ROOT/'anchor/reference_model.json', dict(task0_snapshot_sha256=sha(snap/'complete.json'),
          task0_model_state_sha256=sha(snap/'sa_merged_lora.pt'), probe_max_abs=err,
          class_order=list(map(int, manager._class_order))))
    for task in range(50):
        output = ROOT / 'anchor' / f'task_{task:03d}.pt'
        marker = output.with_suffix('.json')
        if output.exists() and marker.exists():
            assert sha(output) == json.loads(marker.read_text())['sha256']
            continue
        data = load_data(task)
        train = manager.get_dataset(np.arange(2*task,2*task+2),source='train',mode='test')
        ids, features, labels = fw.get_features(model, train)
        assert torch.equal(ids,data['current_train_ids'])
        assert torch.equal(labels,data['current_train_labels'])
        assert set(labels.tolist()) == {2*task,2*task+1}
        if task == 0:
            same = float((features-data['target_train_features']).abs().max())
            assert same < 1e-4, ('anchor Task0 source', same)
        if task == 1:
            same = float((features-data['source_train_features']).abs().max())
            assert same < 1e-4, ('anchor Task1 source', same)
        centroids = mean_prototypes(features, labels, [2*task,2*task+1])
        save(output, dict(task=task, ids=ids, labels=labels,
             features=features, reference_prototypes=centroids))
        write(marker, dict(task=task,sha256=sha(output),count=len(ids),
             sample_ids_equal=True,task0_model_probe_max_abs=err))
        print('ANCHOR_EXTRACT',task,len(ids),flush=True)


def run_anchor():
    fw.setup()
    cfg = json.loads((PREV / 'inputs/task_snapshots/task_001/config.json').read_text())
    variants = ['anchor_none','anchor_full','anchor_half']
    reference = torch.empty((0,768))
    curves = {v: [] for v in variants}
    matrices = {v: [] for v in variants}
    records = {v: [] for v in variants}
    for task in range(50):
        data = load_data(task)
        anchor_file = ROOT/'anchor'/f'task_{task:03d}.pt'
        assert sha(anchor_file) == json.loads(anchor_file.with_suffix('.json').read_text())['sha256']
        a = fw.load(anchor_file)
        assert torch.equal(a['ids'],data['current_train_ids'])
        assert torch.equal(a['labels'],data['current_train_labels'])
        if task:
            mapping = fit_residual_orthogonal_transport(
                a['features'], data['target_train_features'],
                rank=cfg['sa_coordinate_transport_rank'],
                identity_reg=cfg['sa_coordinate_transport_reg'],
                min_validation_gain=cfg['sa_coordinate_transport_min_gain'])
            save(ROOT/'anchor_maps'/f'task_{task:03d}.pt',mapping)
            candidate = fw.moved(reference,mapping)
        else:
            mapping = None
            candidate = reference
        baseline = fw.load(PREV / 'experiments/none'/f'task_{task:03d}.pt')
        for kind in variants:
            alpha_value = {'anchor_none':0.,'anchor_full':1.,'anchor_half':.5}[kind]
            if task:
                old = F.normalize(reference + alpha_value * (candidate-reference),dim=1)
                alpha = torch.full((len(old),),alpha_value * int(mapping['enabled']))
            else:
                old,alpha=reference,torch.empty(0)
            head, scores, rec = metric_step(kind,task,data,old,alpha,baseline,
                         dict(mapping_enabled=mapping['enabled'] if mapping else None))
            curves[kind].append(rec['global_top1'])
            labels=data['test_labels']
            matrices[kind].append([100 * float((scores.argmax(1)[labels//2 == j] == labels[labels//2 == j]).double().mean())
                                   for j in range(task+1)])
            records[kind].append(rec)
            summarize(kind,curves[kind],matrices[kind],records[kind])
            assert torch.equal(head[-2:],new_rows(task,data))
        reference=torch.cat((reference,torch.stack([a['reference_prototypes'][c]
                    for c in (2*task,2*task+1)])))
        print('ANCHOR_EVAL',task,'enabled',mapping['enabled'] if mapping else None,
              *(f'{v}={curves[v][-1]:.2f}' for v in variants),flush=True)


def preflight():
    fw.setup()
    assert sha(PREV/'cache_manifest.json') == META['cache_manifest_sha256']
    task=1;data=load_data(task)
    q_old=reference_old(task)
    q=F.normalize(torch.stack([q_old[c] for c in range(2)]+[data['new_prototypes'][c] for c in (2,3)]),dim=1)
    old=fw.load(PREV/'experiments/none/task_000.pt')['prototypes']
    for variant in ('oracle_cosine','oracle_margin'):
        selected,alpha,_=oracle_choices(old,data['maps']['primary'],q,variant)
        assert len(selected)==2 and torch.isfinite(selected).all()
        assert set(float(x) for x in alpha).issubset(set(META['oracle_alphas']))
        assert torch.allclose(selected.norm(dim=1),torch.ones(2),atol=1e-6)
    assert torch.equal(new_rows(task,data),torch.stack([data['official_prototypes'][c] for c in (2,3)]))
    write(ROOT/'preflight.json',dict(passed=True,cache_manifest_sha256=META['cache_manifest_sha256'],
          oracle_task1_sha256=META['oracle_input_sha256']['task_001.pt'],
          task1_same_new_prototypes=True,task1_oracle_gates_valid=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['preflight','oracle','anchor_extract','anchor_eval'])
    args=parser.parse_args()
    if args.action=='preflight':preflight()
    elif args.action=='oracle':run_oracle()
    elif args.action=='anchor_extract':extract_anchor()
    else:run_anchor()
