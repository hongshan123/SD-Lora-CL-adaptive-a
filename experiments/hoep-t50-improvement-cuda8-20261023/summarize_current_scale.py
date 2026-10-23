"""Summarize previously audited scales; no new model or image evaluation."""
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    full = ROOT / 'records/long_sequence_recoverability_audit.json'
    partial = ROOT / 'records/operator_boundary_audit.json'
    output = ROOT / 'records/current_branch_scale_audit.json'
    hashes = {str(path): sha(path) for path in (full, partial)}
    assert hashes[str(full)] == '6be016f5ab4e1bcda51ae11dead32f1cdc2a61e081e083bb45386e95eeb3ffe1'
    if output.exists():
        previous = json.loads(output.read_text())
        assert previous['input_sha256'] == hashes
        print(json.dumps(previous, allow_nan=False))
        return
    complete = json.loads(full.read_text())
    early = json.loads(partial.read_text())
    source_hashes = {str(ROOT / 'source' / path): sha(ROOT / 'source' / path)
                     for path in ('backbone/sa_lora.py', 'backbone/lora.py', 'models/sdlora.py')}
    rows = complete['rows'] + [row for row in early['rows'] if row['dataset'] == 'imagenetr']
    summaries = []
    for job in sorted({row['job'] for row in rows}):
        group = sorted((row for row in rows if row['job'] == job), key=lambda row: row['task_id'])
        tasks = [row['task_id'] for row in group]
        assert tasks == (list(range(1, 50)) if group[0]['dataset'] == 'cub' else [1, 4, 8])
        scales = [row['current_scale'] for row in group]
        assert all(math.isfinite(value) and value > 0 for value in scales)
        endpoints = []
        for row in (group[0], group[-1]):
            old = math.sqrt(row['energies']['old'])
            current = math.sqrt(row['energies']['current_b'])
            assert math.isclose(current / old, row['current_b_relative_norm'], rel_tol=1e-12)
            endpoints.append(dict(task_id=row['task_id'], current_scale=row['current_scale'],
                                  historical_operator_frobenius=old,
                                  current_operator_frobenius=current,
                                  current_over_historical_operator_norm=current / old))
        summaries.append(dict(job=job, dataset=group[0]['dataset'], seed=group[0]['seed'],
                              tasks=tasks, count=len(group), minimum_scale=min(scales),
                              maximum_scale=max(scales), scale_initialization=0.8,
                              maximum_relative_deviation_from_initialization=max(abs(s / .8 - 1) for s in scales),
                              endpoints=endpoints,
                              historical_norm_growth=endpoints[-1]['historical_operator_frobenius'] / endpoints[0]['historical_operator_frobenius'],
                              current_norm_ratio=endpoints[-1]['current_operator_frobenius'] / endpoints[0]['current_operator_frobenius']))
    result = dict(updated=time.time(), observer_sha256=sha(__file__), input_sha256=hashes,
                  source_sha256=source_hashes, rows=summaries,
                  scope='Aggregation of existing audited training boundaries: CUB seed1 Tasks1-49 and ImageNet-R seed1995 Tasks1,4,8; no new inference or training.',
                  implementation='One trainable scalar shared by Q/V branches, initialized to0.8 in every task; raw current branch s*B*A; no additional alpha/r in this branch.',
                  interpretation='No observed scalar collapse in these states. A declining current/history operator norm ratio is not itself a vanishing scalar or an accuracy bound; no claim that0.8 is optimal, or that early ImageNet-R states establish its full-horizon behavior.')
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(output)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
