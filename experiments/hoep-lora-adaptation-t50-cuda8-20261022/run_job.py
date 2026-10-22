"""Run one fresh experiment, then validate all boundaries and write metrics."""
import ast
import hashlib
import json
import logging
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))


def main():
    import torch
    from trainer import train
    config = json.loads(Path(sys.argv[1]).read_text())
    config['config'] = str(Path(sys.argv[1]).resolve())
    output = Path(config['filepath'])
    if output.exists():
        raise FileExistsError(f'fresh output required: {output}')
    started = time.time()
    train(config)
    logfiles = [Path(h.baseFilename) for h in logging.getLogger().handlers
                if isinstance(h, logging.FileHandler)]
    text = logfiles[0].read_text()
    curves = re.findall(r'CNN top1 curve: (\[[^\n]+\])', text)
    curve = ast.literal_eval(curves[-1])
    expected = config['max_tasks']
    assert len(curve) == expected
    losses = []
    snapshots = []
    for task in range(expected):
        d = output / 'task_snapshots' / f'task_{task:03d}'
        record = json.loads((d / 'complete.json').read_text())
        assert record['task_id'] == task
        for filename, digest in record['sha256'].items():
            assert hashlib.sha256((d / filename).read_bytes()).hexdigest() == digest
        snapshots.append(hashlib.sha256((d / 'complete.json').read_bytes()).hexdigest())
        sidecar = output / 'adaptation_state' / f'task_{task:03d}.json'
        if config.get('sa_adaptation_experiment'):
            item = json.loads(sidecar.read_text())
            assert item['task'] == task
            for key, flag in [('total_history','sa_total_history_weight'),
                              ('allseen_prototype','sa_allseen_proto_weight')]:
                if task and config[flag] > 0:
                    assert item['losses'][key]['steps'] > 0
                    assert item['losses'][key]['max'] > 0, f'inactive loss: {key}'
            losses.append(item)
    hashes = json.loads((output / 'p0_hashes.json').read_text())
    assert set(hashes) == set(map(str, range(expected)))
    metrics = dict(state='complete', prefix=config['prefix'], dataset=config['dataset'],
                   seed=config['seed'], tasks=expected, top1=curve, final_top1=curve[-1],
                   average_accuracy=sum(curve)/len(curve),
                   history_weight=config.get('sa_total_history_weight', 0),
                   prototype_weight=config.get('sa_allseen_proto_weight', 0),
                   transport=config['sa_coordinate_stable_transport'], normcap=False,
                   batch_size=config['batch_size'], elapsed_seconds=time.time()-started,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   snapshot_marker_sha256=snapshots, loss_diagnostics=losses)
    tmp = output / 'metrics.json.tmp'
    tmp.write_text(json.dumps(metrics, indent=2, allow_nan=False)+'\n')
    tmp.replace(output / 'metrics.json')
    print('VALIDATED_METRICS', json.dumps({k:v for k,v in metrics.items()
                                         if k not in ('loss_diagnostics','snapshot_marker_sha256')}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
