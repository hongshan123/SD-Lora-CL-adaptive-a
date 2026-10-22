"""Fixed-trajectory transport ablation. Fits never receive test/old train data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from backbone.coordinate_stability import fit_residual_orthogonal_transport, _apply_matrix
from scripts.diagnose_hoep import build_stage_backbone, set_stage, mean_prototypes
from scripts.hoep_metrics import summarize_group_logits, paired_prediction_changes
from utils.data_manager import DataManager
from utils.hoep_snapshots import audit_snapshot


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_numerical_logits(actual, reference):
    """Permit argmax ties only after checking every logit and flipped margin."""
    assert actual.shape == reference.shape
    assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
    epsilon = float((actual-reference).abs().max())
    assert epsilon <= 1e-5, ('logits exceed numerical tolerance',epsilon)
    flips = actual.argmax(1) != reference.argmax(1)
    indices = torch.where(flips)[0]
    records = []
    for index in indices.tolist():
        a, ai = actual[index].topk(2)
        b, bi = reference[index].topk(2)
        assert float(a[0]-a[1]) <= 2*epsilon+1e-12
        assert float(b[0]-b[1]) <= 2*epsilon+1e-12
        records.append(dict(index=index,actual_classes=ai.tolist(),reference_classes=bi.tolist(),
                            actual_margin=float(a[0]-a[1]),reference_margin=float(b[0]-b[1])))
    return dict(passed=True,max_abs_logits=epsilon,argmax_changes=len(records),samples=records)


def setup():
    torch.set_num_threads(2)
    torch.manual_seed(1993)
    np.random.seed(1993)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def get_features(model, dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2,
                        pin_memory=True, persistent_workers=False)
    ids, features, labels = [], [], []
    with torch.no_grad():
        for i, x, y in loader:
            features.append(model(x.cuda()).float().cpu())
            ids.append(i)
            labels.append(y)
    return torch.cat(ids), torch.cat(features), torch.cat(labels)


def fit(source, target, cfg):
    # This function sees paired CURRENT TRAIN features only.
    return fit_residual_orthogonal_transport(
        source, target, rank=cfg['sa_coordinate_transport_rank'],
        identity_reg=cfg['sa_coordinate_transport_reg'],
        min_validation_gain=cfg['sa_coordinate_transport_min_gain'])


def build_maps(source, target, cfg, task):
    primary = fit(source, target, cfg)
    train_indices = torch.arange(len(source))[torch.arange(len(source)) % 5 != 0]
    x = F.normalize(source[train_indices].float(), dim=1)
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    basis = vh[:10].T.contiguous()
    median_support = (x @ basis).square().sum(1).median()
    maps = []
    for k in range(4):
        generator = torch.Generator().manual_seed(1993 + 1000 * task + k)
        rows = train_indices[torch.randint(len(train_indices), (len(train_indices),), generator=generator)]
        maps.append(fit(source[rows], target[rows], cfg))
    return dict(primary=primary, bootstrap=maps, support_basis=basis,
                median_support=median_support,
                fit_indices=train_indices, validation_indices=torch.arange(0, len(source), 5))


def moved(rows, mapping):
    return F.normalize(_apply_matrix(rows.float(), mapping['basis'], mapping['rotation']), dim=1)


def update(rows, variant, maps):
    count = len(rows)
    if variant == 'none':
        return rows.clone(), torch.zeros(count), {}
    target = moved(rows, maps['primary'])
    if variant == 'current':
        return target, torch.ones(count) * int(maps['primary']['enabled']), {}
    if variant == 'half':
        alpha = torch.full((count,), 0.5 * int(maps['primary']['enabled']))
        return F.normalize(rows + alpha[:, None] * (target - rows), dim=1), alpha, {}
    if variant != 'reliable':
        raise ValueError(variant)
    normalized = F.normalize(rows.float(), dim=1)
    support = ((normalized @ maps['support_basis']).square().sum(1) /
               maps['median_support'].clamp_min(1e-12)).clamp(0, 1)
    estimates = torch.stack([moved(rows, mapping) - normalized for mapping in maps['bootstrap']])
    mean_delta = estimates.mean(0)
    signal = mean_delta.square().sum(1)
    variance = (estimates - mean_delta[None]).square().sum(2).mean(0)
    agreement = signal / (signal + variance + 1e-12)
    enabled_fraction = sum(int(m['enabled']) for m in maps['bootstrap']) / len(maps['bootstrap'])
    alpha = support * agreement * enabled_fraction * int(maps['primary']['enabled'])
    result = F.normalize(normalized + alpha[:, None] * (target - normalized), dim=1)
    assert torch.isfinite(result).all() and torch.all((alpha >= 0) & (alpha <= 1))
    return result, alpha, dict(support=support, agreement=agreement, variance=variance,
                               enabled_fraction=enabled_fraction)


def extract(task):
    setup()
    output = ROOT / 'cache' / f'task_{task:03d}.pt'
    marker = output.with_suffix('.json')
    if marker.exists():
        assert digest(output) == json.loads(marker.read_text())['cache_sha256']
        return
    snapshots = ROOT / 'inputs/task_snapshots'
    current = snapshots / f'task_{task:03d}'
    audit_snapshot(current)
    cfg = json.loads((current / 'config.json').read_text())
    pre = load(current / 'pre_merge.pt')
    total, old_count = pre['total_classes'], pre['known_classes']
    assert old_count == task * 2 and total == old_count + 2
    manager = DataManager(cfg['dataset'], cfg['shuffle'], 1993, 2, 2, cfg)
    deployed = load(current / 'sa_merged_lora.pt')
    model = build_stage_backbone(
        timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0), deployed).cuda().eval()
    with torch.no_grad():
        probe = model(pre['probe_inputs'].cuda()).cpu()
    expected = load(current / 'deployed_probe_features.pt')
    probe_error = float((probe - expected).abs().max())
    assert probe_error < 1e-4, ('deployed probe mismatch', task, probe_error)
    train = manager.get_dataset(np.arange(old_count, total), source='train', mode='test')
    ids, target, labels = get_features(model, train)
    official = load(current / 'sa_prototypes.pt')
    new = mean_prototypes(target, labels, range(old_count, total))
    proto_error = max(float((new[c] - official[c]).abs().max()) for c in new)
    assert proto_error < 1e-4, ('new prototype mismatch', task, proto_error)
    test = manager.get_dataset(np.arange(total), source='test', mode='test')
    test_ids, test_features, test_labels = get_features(model, test)
    weights = F.normalize(torch.stack([official[c] for c in range(total)]), dim=1)
    scores = F.normalize(test_features, dim=1) @ weights.T
    accuracy = 100.0 * int((scores.argmax(1) == test_labels).sum()) / len(test_labels)
    if task:
        reference = json.loads((ROOT / 'inputs/reference' / f'task_{task:03d}.json').read_text())
        expected_acc = reference['metrics']['formal_new_prototypes']['global_top1']
        if abs(accuracy - expected_acc) >= 1e-6:
            reference_arrays = load(ROOT/'inputs/reference'/f'task_{task:03d}.pt')
            assert torch.equal(test_ids,reference_arrays['sample_ids'])
            assert torch.equal(test_labels,reference_arrays['labels'])
            audit = verify_numerical_logits(scores,reference_arrays['logits']['formal_new_prototypes'])
            audit.update(task=task,saved_head_top1=accuracy,reference_top1=expected_acc)
            write(ROOT/f'reference_audit_task_{task:03d}.json',audit)
        previous = snapshots / f'task_{task-1:03d}'
        audit_snapshot(previous)
        set_stage(model, load(previous / 'sa_merged_lora.pt'))
        source_ids, source, source_labels = get_features(model, train)
        assert torch.equal(source_ids, ids) and torch.equal(source_labels, labels)
        maps = build_maps(source, target, cfg, task)
        # Single-step reproduction before any recursive counterfactual is admitted.
        old_rows = torch.stack([pre['old_prototypes'][c] for c in range(old_count)])
        replay = moved(old_rows, maps['primary'])
        expected_old = torch.stack([official[c] for c in range(old_count)])
        transport_error = float((replay - expected_old).abs().max())
        assert transport_error < 1e-4, ('transport reproduction mismatch', task, transport_error)
    else:
        source, maps, transport_error = None, None, 0.0
    artifact = dict(task=task, old_count=old_count, total_classes=total,
                    class_order=list(map(int, manager._class_order)),
                    current_train_ids=ids, current_train_labels=labels,
                    source_train_features=source, target_train_features=target,
                    maps=maps, test_ids=test_ids, test_labels=test_labels,
                    test_features=test_features, official_prototypes=official,
                    new_prototypes={c: official[c] for c in range(old_count, total)})
    save(output, artifact)
    record = dict(task=task, cache_sha256=digest(output), probe_max_abs=probe_error,
                  new_prototype_max_abs=proto_error, transport_max_abs=transport_error,
                  original_accuracy=accuracy, completed=time.time(),
                  snapshot_sha256=digest(current / 'complete.json'))
    write(marker, record)
    print('CACHE_COMPLETE', record, flush=True)


def evaluate(variant):
    setup()
    out = ROOT / 'experiments' / variant
    out.mkdir(parents=True, exist_ok=True)
    prototypes = torch.empty(0, 768)
    curve, matrix, metrics = [], [], []
    for task in range(50):
        cache_path = ROOT / 'cache' / f'task_{task:03d}.pt'
        metadata = json.loads(cache_path.with_suffix('.json').read_text())
        assert digest(cache_path) == metadata['cache_sha256']
        data = load(cache_path)
        if task:
            prototypes, alpha, diagnostics = update(prototypes, variant, data['maps'])
        else:
            alpha, diagnostics = torch.empty(0), {}
        new = torch.stack([data['new_prototypes'][c] for c in range(2*task, 2*task+2)])
        prototypes = torch.cat([prototypes, new])
        if variant == 'current':
            official = torch.stack([data['official_prototypes'][c] for c in range(2*task+2)])
            error = float((prototypes - official).abs().max())
            assert error < 2e-4, ('recursive reproduction', task, error)
        scores = F.normalize(data['test_features'], dim=1) @ F.normalize(prototypes, dim=1).T
        labels = data['test_labels']
        acc = 100.0 * int((scores.argmax(1) == labels).sum()) / len(labels)
        reproduction_audit = None
        if variant == 'current':
            expected_scores = F.normalize(data['test_features'],dim=1) @ F.normalize(official,dim=1).T
            reproduction_audit = verify_numerical_logits(scores,expected_scores)
        if task:
            record = summarize_group_logits(scores, labels, 2*task)
        else:
            competitors = scores.clone()
            competitors.scatter_(1, labels[:, None], float('-inf'))
            margin = scores.gather(1, labels[:, None]).squeeze(1) - competitors.max(1).values
            record = dict(old=None, new=dict(count=len(labels), top1=acc,
                          mean_true_margin=float(margin.mean()), predicted_new_rate=100.0))
        record.update(task=task, global_top1=acc,
                      alpha_mean=float(alpha.mean()) if task else None,
                      alpha_min=float(alpha.min()) if task else None,
                      alpha_max=float(alpha.max()) if task else None)
        if reproduction_audit is not None:
            record['numerical_reproduction'] = reproduction_audit
        if variant != 'current':
            reference = load(ROOT/'experiments/current'/f'task_{task:03d}.pt')
            assert torch.equal(reference['sample_ids'], data['test_ids'])
            assert torch.equal(reference['labels'], labels)
            record['flips_vs_current'] = paired_prediction_changes(reference['logits'], scores, labels, 2*task)
        curve.append(acc)
        per_task = []
        for j in range(task+1):
            mask = (labels >= 2*j) & (labels < 2*j+2)
            per_task.append(100.0 * float((scores.argmax(1)[mask] == labels[mask]).double().mean()))
        matrix.append(per_task)
        metrics.append(record)
        save(out / f'task_{task:03d}.pt', dict(logits=scores, labels=labels,
             sample_ids=data['test_ids'], prototypes=prototypes.clone(), alpha=alpha,
             reliability=diagnostics, cache_sha256=metadata['cache_sha256']))
        write(out / f'task_{task:03d}.json', record)
        summary = dict(variant=variant, completed_tasks=task+1, complete=task==49,
                       curve=curve, final_accuracy=acc, average_incremental_accuracy=float(np.mean(curve)),
                       accuracy_matrix=matrix, seed=1993, dataset='cifar224', tasks=50,
                       scope='fixed trained backbone trajectory; recursively updated prototype history',
                       cache_manifest_sha256=digest(ROOT/'cache_manifest.json'))
        write(out/'metrics.json', summary)
        print('EVAL_COMPLETE', variant, task+1, acc, flush=True)
    forgetting = np.mean([max(matrix[k][j] for k in range(j,50))-matrix[-1][j] for j in range(49)])
    summary['forgetting'] = float(forgetting)
    write(out/'metrics.json', summary)


def unit_checks():
    setup()
    g = torch.Generator().manual_seed(17)
    source = torch.randn(160, 24, generator=g)
    q, _ = torch.linalg.qr(torch.randn(24, 24, generator=g))
    target = F.normalize(source @ q, dim=1)
    cfg = dict(sa_coordinate_transport_rank=10, sa_coordinate_transport_reg=1e-4,
               sa_coordinate_transport_min_gain=0.0)
    maps = build_maps(source, target, cfg, 1)
    assert not set(maps['fit_indices'].tolist()) & set(maps['validation_indices'].tolist())
    rows = F.normalize(torch.randn(8,24,generator=g),dim=1)
    before = rows.clone()
    none, _, _ = update(rows,'none',maps)
    assert torch.equal(none,rows)
    current, _, _ = update(rows,'current',maps)
    assert torch.equal(current,moved(rows,maps['primary']))
    for variant in ['none','current','half','reliable']:
        result, alpha, _ = update(rows,variant,maps)
        assert torch.allclose(result.norm(dim=1),torch.ones(8),atol=1e-6)
        assert torch.all((alpha>=0)&(alpha<=1))
    assert torch.equal(rows,before)
    # An identity drift must cause no movement in any arm.
    identity = build_maps(source,source,cfg,2)
    for variant in ['none','current','half','reliable']:
        result, _, _ = update(rows,variant,identity)
        assert torch.allclose(result,rows,atol=1e-6)
    write(ROOT/'unit_checks.json',dict(passed=True,checks=['identity invariance','no mutation',
          'none/current endpoints','bounded finite gates','normalized outputs','fit/validation separation']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract', type=int)
    parser.add_argument('--evaluate', choices=['none','current','half','reliable'])
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check: unit_checks()
    elif args.extract is not None: extract(args.extract)
    elif args.evaluate: evaluate(args.evaluate)
    else: parser.error('choose an action')
