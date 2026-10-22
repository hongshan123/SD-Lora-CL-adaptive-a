"""Read-only audits of completed formal boundaries while GPU training continues."""
import hashlib
import json
from pathlib import Path
import time

import torch

ROOT = Path(__file__).resolve().parent
HASH_KEYS = ('post_train_model_hash', 'post_train_rng_hash',
             'post_eval_model_hash', 'post_eval_rng_hash')


def read(path):
    # The training process may be replacing its live, cumulative hash file.
    for attempt in range(3):
        try:
            return json.loads(Path(path).read_text())
        except json.JSONDecodeError:
            if attempt == 2:
                raise
            time.sleep(0.1)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(root=ROOT):
    root = Path(root)
    suites = [root, root.parent / 'hoep-bmetric-t50-cuda8-20261023']
    a_step = root.parent / 'hoep-a-step-t50-cuda8-20261023'
    if (a_step / 'manifest.json').exists() and (a_step / 'status.json').exists():
        suites.append(a_step)
    manifests = {suite: read(suite / 'manifest.json') for suite in suites}
    frozen = {(j['dataset'], j['seed']): j for j in manifests[root]['jobs']
              if j['phase'] == 'formal' and j['arm'] == 'frozen'}
    rows = []
    for suite, manifest in manifests.items():
        status = read(suite / 'status.json')
        for job in manifest['jobs']:
            if job['phase'] != 'formal':
                continue
            out = Path(job['output'])
            markers = sorted((out / 'task_snapshots').glob('task_*/complete.json'))
            if not markers:
                continue
            config_path = Path(job['config'])
            assert sha(config_path) == manifest['file_sha256'][str(config_path.relative_to(suite))]
            config = read(config_path)
            reference = frozen[(job['dataset'], job['seed'])]
            current0 = read(out / 'p0_hashes.json')['0']
            reference0 = read(Path(reference['output']) / 'p0_hashes.json')['0']
            assert all(current0[key] == reference0[key] for key in HASH_KEYS), job['name']
            row = dict(job=job['name'], dataset=job['dataset'], seed=job['seed'],
                       method=job.get('arm', 'bmetric_' + job.get('kind', '')),
                       completed_snapshot_count=len(markers), common_formal_task0_exact=True,
                       queue_state=status['jobs'][job['name']]['state'])
            if not config['sa_train_a_all_tasks']:
                first = None
                tasks = []
                for expected, marker_path in enumerate(markers):
                    marker = read(marker_path)
                    assert marker['task_id'] == expected
                    path = marker_path.parent / 'pre_merge.pt'
                    assert sha(path) == marker['sha256']['pre_merge.pt']
                    branches = torch.load(path, map_location='cpu', weights_only=True)['branches']
                    current = [branch['down'] for branch in branches]
                    assert len(current) == 24
                    if first is None:
                        first = current
                    assert all(torch.equal(a, b) for a, b in zip(first, current)), (job['name'], expected)
                    tasks.append(expected)
                row.update(frozen_a_bitwise_unchanged=True, verified_task_ids=tasks)
            rows.append(row)
    result = dict(updated=time.time(), state='partial_formal_audit', rows=rows,
                  interpretation='Only completed snapshots audited; this is not full T50 performance evidence')
    path = root / 'records/formal_boundary_audit.json'
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)
    return result


if __name__ == '__main__':
    torch.set_num_threads(1)
    print(json.dumps(audit(), indent=2, allow_nan=False))
