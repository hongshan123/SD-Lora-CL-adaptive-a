"""Train a fresh paired run and audit results; analysis can recover finished runs."""
import argparse
import ast
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'


def parse_curve(text):
    curves = re.findall(r'CNN top1 curve: (\[[^\n]+\])', text)
    if not curves:
        raise ValueError('no accuracy curve')
    clean = re.sub(r'np\.float(?:32|64)\(\s*('+NUMBER+r')\s*\)', r'\1', curves[-1])
    curve = ast.literal_eval(clean)
    assert isinstance(curve, list) and curve
    assert all(type(x) in (float, int) and math.isfinite(x) and 0 <= x <= 100 for x in curve)
    return [float(x) for x in curve]


def finalize(config, logfile, *, elapsed=None, peak_allocated=None, peak_reserved=None):
    output = Path(config['filepath'])
    text = Path(logfile).read_text(errors='replace')
    curve = parse_curve(text)
    expected = config['max_tasks']
    assert len(curve) == expected, ('incomplete curve', len(curve), expected)
    assert 'Forgetting (CNN):' in text, 'trainer terminal marker missing'
    snapshots, losses = [], []
    for task in range(expected):
        d = output/'task_snapshots'/f'task_{task:03d}'
        marker = json.loads((d/'complete.json').read_text())
        assert marker['task_id'] == task
        for name, digest in marker['sha256'].items():
            assert hashlib.sha256((d/name).read_bytes()).hexdigest() == digest, (task,name)
        snapshots.append(hashlib.sha256((d/'complete.json').read_bytes()).hexdigest())
        if config.get('sa_adaptation_experiment'):
            item = json.loads((output/'adaptation_state'/f'task_{task:03d}.json').read_text())
            assert item['task'] == task
            for key, flag in [('total_history','sa_total_history_weight'),('allseen_prototype','sa_allseen_proto_weight')]:
                if task and config.get(flag,0) > 0:
                    record = item['losses'][key]
                    assert record['steps'] > 0 and record['max'] > 0, ('inactive loss',key)
                    assert all(math.isfinite(record[k]) for k in ['sum','mean','max'])
            losses.append(item)
    hashes = json.loads((output/'p0_hashes.json').read_text())
    assert set(hashes) == set(map(str,range(expected)))
    forgetting = re.findall(r'Forgetting \(CNN\): ('+NUMBER+r')',text)
    frozen_a_verified = None
    if not config['sa_train_a_all_tasks']:
        import torch
        first = None
        for task in range(expected):
            snapshot = torch.load(output/'task_snapshots'/f'task_{task:03d}'/'pre_merge.pt',
                                  map_location='cpu',weights_only=True)
            current = [branch['down'] for branch in snapshot['branches']]
            if first is None:first = current
            else:assert all(torch.equal(a,b) for a,b in zip(first,current)), ('Frozen A changed',task)
        frozen_a_verified = True
    seed = config['seed'][0] if isinstance(config['seed'],list) else config['seed']
    metrics = dict(state='complete',prefix=config['prefix'],dataset=config['dataset'],seed=seed,
                   tasks=expected,top1=curve,final_top1=curve[-1],average_accuracy=sum(curve)/len(curve),
                   forgetting=float(forgetting[-1]),history_weight=config.get('sa_total_history_weight',0),
                   prototype_weight=config.get('sa_allseen_proto_weight',0),
                   frozen_a=not config['sa_train_a_all_tasks'],normcap=False,
                   transport=config['sa_coordinate_stable_transport'],batch_size=config['batch_size'],world_size=1,
                   elapsed_seconds=elapsed,peak_allocated_bytes=peak_allocated,peak_reserved_bytes=peak_reserved,
                   frozen_a_verified=frozen_a_verified,
                   snapshot_marker_sha256=snapshots,loss_diagnostics=losses,training_log=str(logfile))
    temp = output/'metrics.json.tmp'
    temp.write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n');temp.replace(output/'metrics.json')
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--analyze-log',type=Path)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text());config['config']=str(Path(args.config).resolve())
    if args.analyze_log:
        print(json.dumps(finalize(config,args.analyze_log),allow_nan=False));return
    import torch
    from trainer import train
    output=Path(config['filepath'])
    if output.exists():raise FileExistsError(f'fresh output required: {output}')
    started=time.time();train(config)
    logs=[Path(h.baseFilename) for h in logging.getLogger().handlers if isinstance(h,logging.FileHandler)]
    result=finalize(config,logs[0],elapsed=time.time()-started,
                    peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
    print('VALIDATED',result['prefix'],result['final_top1'],result['average_accuracy'],flush=True)


if __name__=='__main__':main()
