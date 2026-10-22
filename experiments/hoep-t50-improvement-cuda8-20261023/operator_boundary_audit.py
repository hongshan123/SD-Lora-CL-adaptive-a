"""CPU diagnostics from completed boundaries; no images or model writes."""
import hashlib
import json
import math
from pathlib import Path
import time

import torch

ROOT = Path(__file__).resolve().parent
POINTS = {'cub': [1, 10, 20], 'imagenetr': [1, 4, 8]}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load_marked(folder, name):
    marker = read(folder / 'complete.json')
    path = folder / name
    assert sha(path) == marker['sha256'][name], path
    return torch.load(path, map_location='cpu', weights_only=True)


def normalized(a):
    a = a.double()
    return a / (torch.linalg.vector_norm(a) + 1e-8)


def energy(u, v, moment=None):
    gram = (v @ moment) @ v.T if moment is not None else v @ v.T
    return max(0., float(((u.T @ u) * gram).sum()))


def difference(u, v, other_u, other_v, moment=None):
    return energy(torch.cat((u, -other_u), dim=1),
                  torch.cat((v, other_v), dim=0), moment)


def relative(numerator, denominator):
    return math.sqrt(max(0., numerator) / max(denominator, 1e-30))


def analyze(job, task):
    out = Path(job['output'])
    d = out / 'task_snapshots' / f'task_{task:03d}'
    prev = out / 'task_snapshots' / f'task_{task-1:03d}'
    snapshot = load_marked(d, 'pre_merge.pt')
    deployed = load_marked(d, 'sa_merged_lora.pt')
    previous = load_marked(prev, 'sa_merged_lora.pt')
    assert snapshot['task_id'] == task and snapshot['normalize_current_branch'] is False
    assert snapshot['absorb_mode'] == 'operator_preserving_absorb'
    provenance = {str(d / name): sha(d / name)
                  for name in ('pre_merge.pt', 'sa_merged_lora.pt')}
    provenance[str(prev / 'sa_merged_lora.pt')] = sha(prev / 'sa_merged_lora.pt')
    moment_path = out / 'adaptation_state' / f'task_{task-1:03d}_moments.pt'
    moments = None
    if moment_path.exists():
        record = torch.load(moment_path, map_location='cpu', weights_only=True)
        assert record['task'] == task - 1 and len(record['moments']) == 12
        moments = record['moments']
        provenance[str(moment_path)] = sha(moment_path)
    totals = {k: 0. for k in ('old', 'a_drift', 'current_b', 'total_drift',
                             'recoverability', 'ls_change', 'deployment', 'rebuild_error',
                             'previous_error', 'weighted_old', 'weighted_projection_residual',
                             'weighted_a_drift', 'weighted_current_b', 'weighted_total_drift',
                             'weighted_ls_change')}
    branches = []
    for i, branch in enumerate(snapshot['branches']):
        a = branch['down'].double()
        abar = normalized(a)
        old_a = normalized(snapshot['alignment_previous']['shared_a'][i])
        old_g = snapshot['alignment_previous']['aggregate_up'][i].double()
        train_g = branch['historical_up'].double()
        current_up = (snapshot['current_scale'].double()
                      * (torch.linalg.vector_norm(a) + 1e-8)
                      * branch['current_up'].double())
        singular = torch.linalg.svdvals(abar)
        assert singular.min() > singular.max() * 1e-6, ('rank deficiency', i)
        old_basis = torch.linalg.qr(old_a.T, mode='reduced').Q
        new_basis = torch.linalg.qr(abar.T, mode='reduced').Q
        cosines = torch.linalg.svdvals(old_basis.T @ new_basis).clamp(0., 1.)
        sine_squared = (1. - cosines.square()).clamp_min(0.)
        aligned = torch.linalg.solve(
            abar @ abar.T, (old_g @ (old_a @ abar.T)).T).T
        deploy_a = normalized(deployed['shared_a'][i])
        deploy_g = deployed['aggregate_up'][i].double()
        prev_a = normalized(previous['shared_a'][i])
        prev_g = previous['aggregate_up'][i].double()
        item = dict(old=energy(old_g, old_a),
                    a_drift=difference(train_g, abar, old_g, old_a),
                    current_b=energy(current_up, abar),
                    total_drift=difference(train_g + current_up, abar, old_g, old_a),
                    recoverability=difference(aligned, abar, old_g, old_a),
                    ls_change=energy(aligned - train_g, abar),
                    deployment=energy(deploy_g, deploy_a),
                    rebuild_error=difference(deploy_g, deploy_a, aligned + current_up, abar),
                    previous_error=difference(prev_g, prev_a, old_g, old_a),
                    weighted_old=0., weighted_projection_residual=0.,
                    weighted_a_drift=0., weighted_current_b=0., weighted_total_drift=0.,
                    weighted_ls_change=0.)
        if moments is not None:
            covariance = moments[i // 2].double()
            item['weighted_old'] = energy(old_g, old_a, covariance)
            item['weighted_projection_residual'] = difference(
                aligned, abar, old_g, old_a, covariance)
            item['weighted_a_drift'] = difference(train_g, abar, old_g, old_a, covariance)
            item['weighted_current_b'] = energy(current_up, abar, covariance)
            item['weighted_total_drift'] = difference(
                train_g + current_up, abar, old_g, old_a, covariance)
            item['weighted_ls_change'] = energy(aligned - train_g, abar, covariance)
        for key, value in item.items():
            totals[key] += value
        branches.append(dict(layer=i // 2, branch='q' if i % 2 == 0 else 'v',
                             condition=float(singular.max() / singular.min()),
                             a_subspace_mean_sine_squared=float(sine_squared.mean()),
                             a_subspace_max_principal_sine=float(sine_squared.max().sqrt()),
                             energies=item))
    assert len(branches) == 24
    rebuild = relative(totals['rebuild_error'], totals['deployment'])
    reference = relative(totals['previous_error'], totals['old'])
    assert rebuild < 5e-6, ('deployment formula mismatch', job['name'], task, rebuild)
    assert reference < 5e-6, ('previous deployment mismatch', job['name'], task, reference)
    cross = .5 * (totals['total_drift'] - totals['a_drift'] - totals['current_b'])
    denom = math.sqrt(totals['a_drift'] * totals['current_b'])
    return dict(job=job['name'], dataset=job['dataset'], seed=job['seed'], task_id=task,
                current_scale=float(snapshot['current_scale']),
                a_subspace_rms_sine=math.sqrt(sum(
                    b['a_subspace_mean_sine_squared'] for b in branches) / len(branches)),
                a_subspace_max_principal_sine=max(
                    b['a_subspace_max_principal_sine'] for b in branches),
                a_only_relative_drift=relative(totals['a_drift'], totals['old']),
                current_b_relative_norm=relative(totals['current_b'], totals['old']),
                total_training_relative_drift=relative(totals['total_drift'], totals['old']),
                frobenius_projection_residual_ratio=totals['recoverability'] / max(totals['old'], 1e-30),
                ls_relative_operator_change=relative(totals['ls_change'], totals['old']),
                a_drift_current_b_cosine=None if denom < 1e-25 else max(-1., min(1., cross / denom)),
                deployment_formula_relative_error=rebuild, previous_anchor_relative_error=reference,
                weighted_frobenius_projection_residual_ratio=None if moments is None else
                    totals['weighted_projection_residual'] / max(totals['weighted_old'], 1e-30),
                weighted_a_only_relative_drift=None if moments is None else
                    relative(totals['weighted_a_drift'], totals['weighted_old']),
                weighted_current_b_relative_norm=None if moments is None else
                    relative(totals['weighted_current_b'], totals['weighted_old']),
                weighted_total_training_relative_drift=None if moments is None else
                    relative(totals['weighted_total_drift'], totals['weighted_old']),
                weighted_ls_relative_operator_change=None if moments is None else
                    relative(totals['weighted_ls_change'], totals['weighted_old']),
                weighted_statistics_source=None if moments is None else str(moment_path),
                energies=totals, branches=branches, sha256=provenance)


def audit(root=ROOT):
    root = Path(root)
    rows = []
    suites = [root, root.parent / 'hoep-bmetric-t50-cuda8-20261023']
    a_step = root.parent / 'hoep-a-step-t50-cuda8-20261023'
    if (a_step / 'manifest.json').exists():
        suites.append(a_step)
    for suite in suites:
        for job in read(suite / 'manifest.json')['jobs']:
            if job['phase'] != 'formal':
                continue
            method = job.get('arm', 'bmetric_' + job.get('kind', ''))
            if method not in ('frozen', 'history', 'bmetric_current', 'a_gain10', 'a_gain100'):
                continue
            for task in POINTS[job['dataset']]:
                marker = Path(job['output']) / 'task_snapshots' / f'task_{task:03d}' / 'complete.json'
                if not marker.exists():
                    continue
                row = analyze(job, task)
                row['method'] = method
                rows.append(row)
    result = dict(updated=time.time(), code_sha256=sha(Path(__file__)), diagnostic_only=True, rows=rows,
                  interpretation='Selected completed boundaries, not T50 results. Operator norms are not accuracy bounds. Weighted residual uses the Frobenius-optimal projector and each model own historical input moments; it is not a weighted-optimal reconstruction error.')
    p = root / 'records/operator_boundary_audit.json'
    tmp = p.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)
    return result


if __name__ == '__main__':
    torch.set_num_threads(1)
    result = audit()
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}))
    for row in result['rows']:
        print(json.dumps({k: v for k, v in row.items() if k not in ('energies', 'branches', 'sha256')}))
