"""One durable, leased job per requested GPU; no changes to training protocol."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source'
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, '/home/hongzhijun/hongshan/cuda9-baselines-multiseed-20261020')
from gpu_leases import GpuLease
from scripts.structure_ablation_queue import parse_curve, identity, write


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(kind, gpu):
    job = ROOT / kind
    job.mkdir(exist_ok=True)
    if (job / 'started.json').exists():
        raise RuntimeError('Existing attempt retained; use a new attempt for restart')
    lease = GpuLease.acquire([gpu])
    if lease is None:
        raise RuntimeError('GPU lease is held: ' + str(gpu))
    row = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    if int(row.strip()) >= 300:
        raise RuntimeError('GPU is no longer idle: ' + str(gpu))
    config_path = ROOT / 'configs' / (kind + '.json')
    write(job / 'started.json', dict(identity=identity(os.getpid()), gpu=gpu,
          config_sha256=sha(config_path), source_hash_manifest=sha(ROOT/'source_hashes.json')))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1',
               PYTHONFAULTHANDLER='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    code = 1
    child_code = None

    def run_stage(name, command):
        nonlocal child_code
        write(job / 'status.json', dict(state='running', phase=name, gpu=gpu,
                                        updated=time.time(), identity=identity(os.getpid())))
        with (job / (name + '.log')).open('w') as log:
            child = subprocess.Popen(command, cwd=SOURCE, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, pass_fds=lease.fds)
            write(job / (name + '_process.json'), dict(identity=identity(child.pid)))
            child_code = child.wait()
        if child_code:
            raise RuntimeError(f'{name} exited with code {child_code}; preserved {name}.log')

    try:
        for relative, digest in json.loads((ROOT/'source_hashes.json').read_text()).items():
            if sha(SOURCE / relative) != digest:
                raise RuntimeError('Frozen source changed: ' + relative)
        cfg = json.loads(config_path.read_text())
        output = Path(cfg['filepath'])
        if kind == 'independent_bank':
            run_stage('resource_gate', [sys.executable, str(SOURCE/'scripts/structure_resource_gate.py'),
                      '--mode', 'independent_bank', '--output', str(job/'resource')])
            gate = json.loads((job/'resource/gate.json').read_text())
            if not gate['passed'] or gate['microbatch'] != 64:
                raise RuntimeError('Original microbatch64 did not pass; do not silently change protocol')
            run_stage('formal', [sys.executable, str(SOURCE/'main.py'), '--config', str(config_path)])
            metrics = parse_curve(job/'formal.log', 50)
            import torch
            state = torch.load(output/'sa_state.pt', map_location='cpu', weights_only=True)
            protos = torch.load(output/'sa_prototypes.pt', map_location='cpu', weights_only=True)
            assert state['structure_mode'] == 'independent_bank' and len(state['entries']) == 50
            assert set(protos) == set(range(100))
            metrics.update(tasks_completed=50, tasks=50, complete=True,
                           mode='independent_bank', seed=1993, transport=True, microbatch=64)
        else:
            script = str(SOURCE/'scripts/run_frozen_backbone_prototypes.py')
            run_stage('smoke', [sys.executable, script, '--config', str(ROOT/'configs/frozen_smoke.json'),
                                '--max-tasks', '2'])
            smoke = json.loads((job/'smoke_output/metrics.json').read_text())
            assert smoke['complete'] and smoke['backbone_unchanged']
            run_stage('formal', [sys.executable, script, '--config', str(config_path)])
            metrics = json.loads((output/'metrics.json').read_text())
            assert metrics['complete'] and metrics['tasks_completed'] == 50
            assert metrics['backbone_unchanged']
        metrics.update(config_sha256=sha(config_path), gpu=gpu)
        write(job/'metrics.json', metrics)
        code = 0
    except BaseException:
        traceback.print_exc()
    finally:
        write(job/'exit.json', dict(exit_code=code, last_child_exit_code=child_code,
              config_sha256=sha(config_path), finished=time.time()))
        write(job/'status.json', dict(state='complete' if code == 0 else 'failed',
              gpu=gpu, updated=time.time()))
        lease.close()
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['independent_bank', 'frozen_pretrained'], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    sys.exit(main(args.kind, args.gpu))
