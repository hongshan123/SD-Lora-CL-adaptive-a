#!/usr/bin/env python3
"""Run isolated DDP jobs on GPU0-3 after paired real-data smoke checks."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def verify(manifest, entry):
    source = manifest['sources'][entry['source']]
    for relative, digest in source['hashes'].items():
        assert hashlib.sha256((Path(source['path']) / relative).read_bytes()).hexdigest() == digest, relative
    assert hashlib.sha256(Path(entry['config']).read_bytes()).hexdigest() == entry['config_sha256']


def available(pair):
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
    memory = {int(line.split(',')[0]): int(line.split(',')[1]) for line in query.splitlines()}
    return all(memory[index] < 512 for index in pair)


def tensors(value, prefix=''):
    import torch
    if torch.is_tensor(value):
        return {prefix: value}
    if isinstance(value, dict):
        return {p: t for key, item in value.items() for p, t in tensors(item, prefix + '/' + str(key)).items()}
    if isinstance(value, (list, tuple)):
        return {p: t for i, item in enumerate(value) for p, t in tensors(item, prefix + '/' + str(i)).items()}
    return {}


def compare_smokes(job):
    import torch
    on = Path(job['phases']['smoke_on']['output'])
    off = Path(job['phases']['smoke_off']['output'])
    left = json.loads((on / 'p0_hashes.json').read_text())
    right = json.loads((off / 'p0_hashes.json').read_text())
    assert left['0'] == right['0'], ('Task0 model/RNG changed', left['0'], right['0'])
    for key in ['post_train_model_hash', 'post_train_rng_hash']:
        assert left['1'][key] == right['1'][key], ('Task1 training changed', key)
    counts = {}
    for artifact in ['sa_state.pt', 'sa_merged_lora.pt']:
        a = tensors(torch.load(on / artifact, map_location='cpu', weights_only=True))
        b = tensors(torch.load(off / artifact, map_location='cpu', weights_only=True))
        assert a and a.keys() == b.keys(), (artifact, 'tensor keys')
        assert all(torch.equal(value, b[key]) for key, value in a.items()), (artifact, 'adapter weights changed')
        counts[artifact] = len(a)
    off_text = Path(job['phases']['smoke_off']['log']).read_text(errors='replace')
    assert off_text.count('deployment state before prototype transport') == 2, 'Both task boundaries must rebuild'
    assert 'prototype transport task' not in off_text, 'Transport unexpectedly executed'
    proof = dict(passed=True, task0_model_and_rng_bitwise_equal=True, task1_training_model_and_rng_bitwise_equal=True, final_adapter_tensors_bitwise_equal=counts, off_rebuilt_both_boundaries=True, old_prototype_transport_disabled=True)
    write(ROOT / 'proofs' / (job['name'] + '_equivalence.json'), proof)
    return proof


def main():
    lock = (ROOT / 'queue.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    status = json.loads((ROOT / 'status.json').read_text())
    assert status['state'] == 'prepared', 'Inspect existing status before recovery; refusing duplicate starts'
    status.update(state='preflight', pid=os.getpid(), started=time.time())
    write(ROOT / 'status.json', status)
    pairs = manifest['gpu_pairs']
    # Import the original completion validator, without changing training sources.
    sys.path.insert(0, manifest['sources']['speed_reference']['path'])
    from scripts.hoep_cifar_parallel_queue import validated_result
    for stage in ['preflight', 'formal']:
        status['state'] = stage
        pending = list(manifest['jobs'])
        active = {}
        while pending or active:
            for pair in pairs:
                key = tuple(pair)
                if key in active or not pending or not available(pair):
                    continue
                job = pending.pop(0)
                phases = ['smoke_on', 'smoke_off'] if stage == 'preflight' else ['formal']
                active[key] = dict(job=job, phases=phases, phase_index=0)
            for pair, record in list(active.items()):
                job = record['job']
                phase = record['phases'][record['phase_index']]
                entry = job['phases'][phase]
                if 'process' not in record:
                    verify(manifest, entry)
                    stream = Path(entry['log']).open('x')
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, pair)), CUDA_DEVICE_ORDER='PCI_BUS_ID', CUBLAS_WORKSPACE_CONFIG=':4096:8', HF_HUB_OFFLINE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1')
                    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2', 'main.py', '--config=' + entry['config']]
                    process = subprocess.Popen(command, cwd=manifest['sources'][entry['source']]['path'], env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    record.update(process=process, stream=stream, started=time.time())
                    status['jobs'][job['name']] = dict(state='running', phase=phase, gpu_pair=list(pair), pid=process.pid, started=record['started'], log=entry['log'], output=entry['output'])
                    print('START', stage, job['name'], phase, 'GPUs', pair, 'PID', process.pid, flush=True)
                code = record['process'].poll()
                if code is None:
                    continue
                record['stream'].close()
                proof_path = ROOT / 'proofs' / (entry['name'] + '.json')
                proof = dict(phase='complete' if code == 0 else 'failed', returncode=code, gpu_pair=list(pair), proof_path=str(proof_path), started=record['started'], finished=time.time())
                try:
                    verify(manifest, entry)
                    proof['result'] = validated_result(entry, proof)
                    write(proof_path, proof)
                    if phase == 'smoke_off':
                        status['jobs'][job['name']]['equivalence'] = compare_smokes(job)
                    print('PASS', job['name'], phase, 'final', proof['result']['final'], flush=True)
                except Exception:
                    proof.update(phase='failed', error=traceback.format_exc())
                    write(proof_path, proof)
                    status['jobs'][job['name']].update(state='failed', error=proof['error'])
                    status.update(state='failed', ended=time.time())
                    write(ROOT / 'status.json', status)
                    raise
                record['phase_index'] += 1
                if record['phase_index'] == len(record['phases']):
                    if stage == 'formal':
                        status['jobs'][job['name']].update(state='complete', finished=time.time(), metrics=proof['result'])
                        write(ROOT / 'outputs' / (job['name'] + '_metrics.json'), proof['result'])
                    else:
                        status['jobs'][job['name']].update(state='preflight_passed', finished=time.time())
                    del active[pair]
                else:
                    del record['process']
                    del record['stream']
            status['updated'] = time.time()
            write(ROOT / 'status.json', status)
            time.sleep(3)
        if stage == 'preflight':
            write(ROOT / 'preflight_passed.json', dict(passed=True, datasets=[j['dataset'] for j in manifest['jobs']], finished=time.time()))
    status.update(state='complete', finished=time.time())
    write(ROOT / 'status.json', status)
    print('ALL_THREE_T50_RUNS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
