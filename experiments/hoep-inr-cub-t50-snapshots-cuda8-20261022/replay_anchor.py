"""Read-only fixed-reference prototype replay for ImageNet-R and CUB T50.

The map sees only the arriving task's training images. Every classifier arm
uses the same deployed backbone, test images, and new-class prototypes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['imagenetr', 'cub'], required=True)
parser.add_argument('--limit', type=int, default=50)
parser.add_argument('--phase', choices=['smoke', 'formal'], default='formal')
parser.add_argument('--device', default='cuda:0')
args = parser.parse_args()
manifest = json.loads((ROOT / 'manifest.json').read_text())
entry = manifest['datasets'][args.dataset]
sys.path.insert(0, entry['source'])
os.chdir(entry['source'])
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from backbone.coordinate_stability import fit_residual_orthogonal_transport, _apply_matrix  # noqa: E402
from scripts.evaluate_sa_sdlora import build_merged_backbone  # noqa: E402
from utils.data_manager import DataManager  # noqa: E402
from utils.hoep_snapshots import audit_snapshot  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, item):
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


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def set_stage(model, state):
    assert len(state['shared_a']) == len(state['merged_b']) == 2 * len(model.blocks)
    with torch.no_grad():
        for layer, block in enumerate(model.blocks):
            wrapper = block.attn.qkv
            for offset, suffix in ((0, 'q'), (1, 'v')):
                index = 2 * layer + offset
                getattr(wrapper, 'a_' + suffix).copy_(state['shared_a'][index].to(args.device))
                getattr(wrapper, 'b_' + suffix).copy_(state['merged_b'][index].to(args.device))
    model.eval()


def get_features(model, dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    ids, features, labels = [], [], []
    with torch.no_grad():
        for i, x, y in loader:
            ids.append(i.cpu())
            features.append(model(x.to(args.device, non_blocking=True)).float().cpu())
            labels.append(y.cpu())
    return torch.cat(ids), torch.cat(features), torch.cat(labels)


def centroids(features, labels, classes):
    features = F.normalize(features.float(), dim=1)
    result = []
    for class_id in classes:
        selected = features[labels == class_id]
        assert len(selected) > 0, class_id
        result.append(F.normalize(selected.mean(0), dim=0))
    return torch.stack(result)


def metric(scores, labels, old_count):
    prediction = scores.argmax(1)
    result = dict(top1=100 * float((prediction == labels).double().mean()),
                  count=len(labels))
    for name, mask in [('old', labels < old_count), ('new', labels >= old_count)]:
        if mask.any():
            result[name] = 100 * float((prediction[mask] == labels[mask]).double().mean())
            if name == 'old':
                result['old_to_new'] = 100 * float((prediction[mask] >= old_count).double().mean())
    runner = scores.clone()
    runner.scatter_(1, labels[:, None], float('-inf'))
    margins = scores.gather(1, labels[:, None]).squeeze(1) - runner.max(1).values
    result['mean_true_margin'] = float(margins.mean())
    return result


def extract(task, model, manager, task0_state, out):
    directory = Path(entry['outputs'][args.phase]) / 'task_snapshots' / f'task_{task:03d}'
    record = audit_snapshot(directory)
    cfg = json.loads((directory / 'config.json').read_text())
    pre = load(directory / 'pre_merge.pt')
    state = load(directory / 'sa_merged_lora.pt')
    official = load(directory / 'sa_prototypes.pt')
    known, total = pre['known_classes'], pre['total_classes']
    assert known == 4 * task and total == known + 4
    assert list(map(int, manager._class_order)) == list(map(int, cfg.get('class_order', manager._class_order)))
    train = manager.get_dataset(np.arange(known, total), source='train', mode='test')
    set_stage(model, task0_state)
    source_ids, source, source_labels = get_features(model, train)
    set_stage(model, state)
    with torch.no_grad():
        probe = model(pre['probe_inputs'].to(args.device)).float().cpu()
    probe_error = float((probe - load(directory / 'deployed_probe_features.pt')).abs().max())
    assert probe_error < 1e-4, (task, 'deployed model probe', probe_error)
    target_ids, target, target_labels = get_features(model, train)
    assert torch.equal(source_ids, target_ids) and torch.equal(source_labels, target_labels)
    assert set(target_labels.tolist()) == set(range(known, total))
    reference_new = centroids(source, source_labels, range(known, total))
    current_new = centroids(target, target_labels, range(known, total))
    official_new = torch.stack([official[c] for c in range(known, total)])
    new_error = float((current_new - official_new).abs().max())
    assert new_error < 1e-4, (task, 'new prototype reproduction', new_error)
    test = manager.get_dataset(np.arange(total), source='test', mode='test')
    test_ids, test_features, test_labels = get_features(model, test)
    mapping = None
    if task:
        mapping = fit_residual_orthogonal_transport(
            source, target, rank=cfg['sa_coordinate_transport_rank'],
            identity_reg=cfg['sa_coordinate_transport_reg'],
            min_validation_gain=cfg['sa_coordinate_transport_min_gain'])
    cache = dict(task=task, known=known, total=total,
                 train_ids=source_ids, train_labels=source_labels,
                 reference_train_features=source, current_train_features=target,
                 reference_new=reference_new, current_new=official_new,
                 test_ids=test_ids, test_labels=test_labels,
                 test_features=test_features, official_prototypes=official,
                 map=mapping, snapshot_sha256=digest(directory / 'complete.json'))
    save(out / 'cache' / f'task_{task:03d}.pt', cache)
    write_json(out / 'cache' / f'task_{task:03d}.json',
               dict(task=task,sha256=digest(out / 'cache' / f'task_{task:03d}.pt'),
                    snapshot_sha256=cache['snapshot_sha256'],probe_max_abs=probe_error,
                    new_prototype_max_abs=new_error,train_samples=len(source_ids),
                    test_samples=len(test_ids)))
    return cache


def evaluate(task, cache, reference, recursive_none, curve, out):
    known,total=cache['known'],cache['total']
    assert len(reference)==known and len(recursive_none)==known
    current_new=cache['current_new']
    official=torch.stack([cache['official_prototypes'][c] for c in range(total)])
    variants={'official':official,
              'recursive_none':torch.cat((recursive_none,current_new))}
    if task:
        mapping=cache['map']
        moved=F.normalize(_apply_matrix(reference.float(),mapping['basis'],mapping['rotation']),dim=1)
        variants['anchor_none']=torch.cat((reference,current_new))
        variants['anchor_half']=torch.cat((F.normalize((reference+moved)/2,dim=1),current_new))
        variants['anchor_full']=torch.cat((moved,current_new))
    else:
        for name in ['anchor_none','anchor_half','anchor_full']:
            variants[name]=torch.cat((reference,current_new))
    assert all(torch.equal(x[known:],current_new) for x in variants.values())
    scores={name:F.normalize(cache['test_features'],dim=1) @ F.normalize(rows,dim=1).T
            for name,rows in variants.items()}
    labels=cache['test_labels']
    report={name:metric(logits,labels,known) for name,logits in scores.items()}
    for name in variants:
        curve[name].append(report[name]['top1'])
    score_path=out/'scores'/f'task_{task:03d}.pt'
    save(score_path,dict(logits=scores,prototypes=variants,
                         sample_ids=cache['test_ids'],labels=labels))
    write_json(out/'scores'/f'task_{task:03d}.json',
               dict(task=task,sha256=digest(score_path),metrics=report,
                    map_enabled=None if not task else bool(cache['map']['enabled'])))
    write_json(out/'metrics.json',dict(dataset=args.dataset,seed=1995 if args.dataset=='imagenetr' else 1,
               tasks=50,completed_tasks=task+1,complete=task==49,
               curve=curve,final={name:report[name] for name in variants},
               average_incremental={name:float(np.mean(values)) for name,values in curve.items()},
               protocol='fixed trained HOEP5%+LS+PT-on NC-off trajectory; fixed Task0 reference; current task train only for maps; identical new prototypes/test images'))
    return torch.cat((reference,cache['reference_new'])),torch.cat((recursive_none,current_new))


def main():
    assert 1 <= args.limit <= 50
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    output=ROOT/'replay'/args.phase/args.dataset
    first=Path(entry['outputs'][args.phase])/'task_snapshots'/'task_000'
    audit_snapshot(first)
    cfg=json.loads((first/'config.json').read_text())
    manager=DataManager(cfg['dataset'],cfg['shuffle'],cfg['seed'],4,4,cfg)
    task0_state=load(first/'sa_merged_lora.pt')
    base=timm.create_model('vit_base_patch16_224',pretrained=True,num_classes=0)
    model=build_merged_backbone(base,task0_state).to(args.device).eval()
    reference=torch.empty((0,768)); recursive_none=reference.clone()
    curve={name:[] for name in ['official','recursive_none','anchor_none','anchor_half','anchor_full']}
    for task in range(args.limit):
        cache_path=output/'cache'/f'task_{task:03d}.pt'
        marker=cache_path.with_suffix('.json')
        if marker.exists():
            assert digest(cache_path)==json.loads(marker.read_text())['sha256']
            cache=load(cache_path)
        else:
            cache=extract(task,model,manager,task0_state,output)
        reference,recursive_none=evaluate(task,cache,reference,recursive_none,curve,output)
        print('REPLAY',args.dataset,task,*(f'{name}={curve[name][-1]:.2f}' for name in curve),flush=True)


if __name__=='__main__':
    main()
