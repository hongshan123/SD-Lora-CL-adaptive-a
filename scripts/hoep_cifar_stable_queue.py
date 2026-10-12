#!/usr/bin/env python3
"""Three stable GPU pairs; reuse complete cells and archive rebooted attempts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.hoep_cifar_ablation import (
    atomic_json, completed_run_metrics, freeze_sources, verify_sources,
)
from scripts.hoep_cifar_parallel_queue import (
    gpu_measurements, process_info, ready_jobs, validated_result,
    validate_reference, verify_dispatcher, worker, worker_progress,
)

PAIRS = ((0, 1), (2, 5), (6, 7))
PATH_SETTINGS = {"prefix", "filepath", "sa_hoep_reference_counts_path"}


def pair_can_dispatch(pair, active, readings, max_jobs=1, reserved_mb=11000, budget_mb=23000):
    """Admit a second owned model only with enough measured/reserved memory."""
    count = sum(tuple(item['gpu_pair']) == tuple(pair) for item in active.values())
    if count >= max_jobs:
        return False
    if count == 0:
        return all(readings[g][0] <= 64 and readings[g][1] <= 5 for g in pair)
    return all(max(readings[g][0], count * reserved_mb) + reserved_mb <= budget_mb for g in pair)


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def verify_relocation(original, relocated):
    if {k: v for k, v in original.items() if k not in PATH_SETTINGS} != {
            k: v for k, v in relocated.items() if k not in PATH_SETTINGS}:
        raise ValueError('scientific configuration changed during relocation')


def relocate_config(original, name, output, reference=None):
    result = dict(original, prefix=name, filepath=str(output) + '/')
    if reference is not None:
        result['sa_hoep_reference_counts_path'] = str(reference)
    verify_relocation(original, result)
    return result


def archive_interrupted(entry, proof_path, attempts):
    target = Path(attempts) / (entry['name'] + '_' + uuid.uuid4().hex)
    target.mkdir(parents=True, exist_ok=False)
    for source, name in ((Path(entry['log']), 'run.log'),
                         (Path(entry['output']), 'output'),
                         (Path(proof_path), 'proof.json')):
        if source.exists():
            shutil.move(str(source), str(target / name))
    return str(target)


def launch_owned_worker(command, name, pair, proof_path):
    """Release a worker only after its durable PID identity has been saved."""
    read_fd, write_fd = os.pipe()
    process = None
    try:
        process = subprocess.Popen(command + ['--start-fd', str(read_fd)], pass_fds=(read_fd,))
        identity = process_info(process.pid)
        atomic_json(proof_path, {'name': name, 'phase': 'starting', 'gpu_pair': list(pair),
                    'worker_pid': process.pid, 'worker_start_ticks': identity['start_ticks'],
                    'proof_path': str(proof_path)})
        os.write(write_fd, b'1')
        return process
    except Exception:
        if process is not None:
            process.kill()
            process.wait()
        raise
    finally:
        os.close(read_fd)
        os.close(write_fd)


def prepare(root, old, runtime):
    root, old, runtime = map(lambda p: Path(p).resolve(), (root, old, runtime))
    verify_sources(root, old)
    if runtime.exists():
        raise RuntimeError('refusing to overwrite a restart runtime')
    previous = json.loads((old / 'manifest.json').read_text())
    events = [json.loads(s) for s in (old / 'events.jsonl').read_text().splitlines()]
    if not any(e.get('kind') == 'preflight_passed' for e in events):
        raise RuntimeError('original training equivalence preflights are missing')
    for folder in ('configs', 'results', 'logs', 'references', 'parallel/job_results'):
        (runtime / folder).mkdir(parents=True, exist_ok=True)
    shutil.copy2(old / 'references/hoep_seed1993_counts.json',
                 runtime / 'references/hoep_seed1993_counts.json')
    validate_reference(runtime / 'references/hoep_seed1993_counts.json', 1993)
    shutil.copy2(old / 'reuse_audit.json', runtime / 'reuse_audit.json')
    jobs, completions, audit = [], [], []
    for entry in previous['jobs']:
        metrics = completed_run_metrics(entry, old / 'events.jsonl')
        if metrics is not None:
            jobs.append(entry)
            completions.append({'kind': 'training_complete', 'name': entry['name'],
                                'reuse_from': str(old), 'verified_original_exit_event': True})
            continue
        name = entry['name'].replace('20261008_r2', '20261009_stable')
        output = runtime / 'results' / name
        original = json.loads(Path(entry['config']).read_text())
        reference = None
        if entry['method'] in ('random', 'high'):
            if entry['seed'] == 1993:
                reference = runtime / 'references/hoep_seed1993_counts.json'
            else:
                provider = next(j for j in jobs if j['method'] == 'hoep' and j['seed'] == entry['seed'])
                reference = Path(provider['output']) / 'hoep_partition_schedule.json'
        config = relocate_config(original, name, output, reference)
        config_path = runtime / 'configs' / (name + '.json')
        atomic_json(config_path, config)
        jobs.append({**entry, 'name': name, 'config': str(config_path),
                     'output': str(output), 'log': str(runtime / 'logs' / (name + '.log'))})
        audit.append({'original': entry, 'restarted': jobs[-1],
                      'changed_keys': [k for k in config if config[k] != original.get(k)]})
    manifest = {**previous, 'run_tag': '20261009_stable', 'jobs': jobs, 'preflight': [],
                'gpu_pairs': PAIRS, 'gpu_ids': [0, 1, 2, 5, 6, 7], 'excluded_gpus': [3, 4]}
    atomic_json(runtime / 'manifest.json', manifest)
    atomic_json(runtime / 'restart_audit.json', {'original_runtime': str(old),
                'reason': 'server reboot; user excludes unstable GPU3/4',
                'restart_from_task': 0, 'relocations': audit})
    (runtime / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in completions))
    freeze_sources(root, runtime, manifest)
    pin_path = runtime / 'frozen_sources.json'
    pin = json.loads(pin_path.read_text())
    pin['config_hashes'][str(runtime / 'restart_audit.json')] = hashlib.sha256(
        (runtime / 'restart_audit.json').read_bytes()).hexdigest()
    atomic_json(pin_path, pin)
    source = Path(__file__).resolve()
    shutil.copy2(source, runtime / 'parallel/dispatcher_snapshot.py')
    atomic_json(runtime / 'parallel/dispatcher_source.json', {
        'source': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'gpu_pairs': PAIRS, 'original_runtime': str(old)})
    return manifest


def run(root, runtime):
    root, runtime = Path(root).resolve(), Path(runtime).resolve()
    work = runtime / 'parallel'
    verify_sources(root, runtime); verify_dispatcher(work)
    manifest = json.loads((runtime / 'manifest.json').read_text())
    assert tuple(map(tuple, manifest['gpu_pairs'])) == PAIRS
    jobs = manifest['jobs']
    max_jobs = manifest.get('max_jobs_per_pair', 1)
    if isinstance(max_jobs, bool) or max_jobs not in (1, 2):
        raise ValueError('max_jobs_per_pair must be 1 or 2')
    lock = (work / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    completed, active, failures, children = {}, {}, [], {}
    streak = {pair: 0 for pair in PAIRS}
    current_boot = boot_id()
    boot_file = work / 'boot.json'
    previous_boot = json.loads(boot_file.read_text())['boot_id'] if boot_file.exists() else current_boot

    def event(kind, **detail):
        record = {'at': datetime.now(timezone.utc).isoformat(), 'kind': kind, **detail}
        with (work / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)

    def save(phase='training'):
        state = {'phase': phase, 'scheduler': 'three_stable_pairs', 'formal_total': len(jobs),
                 'reused': len(manifest['reused']), 'completed': list(completed),
                 'active': list(active.values()), 'gpu_pairs': PAIRS, 'excluded_gpus': [3, 4],
                 'max_jobs_per_pair': max_jobs,
                 'pending': [j['name'] for j in jobs if j['name'] not in completed and j['name'] not in active],
                 'failures': failures, 'boot_id': current_boot,
                 'updated_at': datetime.now(timezone.utc).isoformat()}
        results = json.loads((runtime / 'reuse_audit.json').read_text()) + list(completed.values())
        for parent in (work, runtime):
            atomic_json(parent / 'status.json', state)
            atomic_json(parent / 'results_summary.json', results)

    try:
        for entry in jobs:
            name = entry['name']
            cfg = json.loads(Path(entry['config']).read_text())
            metrics = completed_run_metrics(entry, runtime / 'events.jsonl', cfg.get('max_tasks', 10))
            if metrics is not None:
                completed[name] = {**entry, **metrics}
            proof_path = work / 'job_results' / (name + '.json')
            if not proof_path.exists():
                if name not in completed and previous_boot != current_boot and (
                        Path(entry['log']).exists() or Path(entry['output']).exists()):
                    target = archive_interrupted(entry, proof_path, work / 'interrupted_attempts')
                    event('reboot_unrecorded_attempt_archived', name=name, archive=target)
                continue
            proof = json.loads(proof_path.read_text())
            if proof['phase'] == 'complete':
                completed[name] = validated_result(entry, proof)
            elif proof['phase'] == 'failed':
                failures.append({'name': name, 'error': proof.get('error', 'worker failed')})
            elif previous_boot != current_boot:
                target = archive_interrupted(entry, proof_path, work / 'interrupted_attempts')
                event('reboot_attempt_archived', name=name, archive=target)
            else:
                running = {'name': name, 'entry': entry, 'gpu_pair': proof['gpu_pair'],
                           'pid': proof['worker_pid'], 'start_ticks': proof['worker_start_ticks']}
                terminal = worker_progress(running, proof_path)
                if terminal is None:
                    active[name] = running
                elif terminal['phase'] == 'complete':
                    completed[name] = validated_result(entry, terminal)
                else:
                    failures.append({'name': name, 'error': terminal.get('error', 'worker failed')})
        atomic_json(boot_file, {'boot_id': current_boot})
        event('stable_queue_start', completed=list(completed), gpu_pairs=PAIRS, excluded_gpus=[3, 4])
        while len(completed) < len(jobs) or active:
            for name, running in list(active.items()):
                proof_path = work / 'job_results' / (name + '.json')
                proof = worker_progress(running, proof_path)
                if proof is None:
                    continue
                try:
                    result = validated_result(running['entry'], proof)
                    completed[name] = result
                    event('training_complete', name=name, final=result['final'], aaa=result['aaa'])
                except Exception as error:
                    detail = proof.get('error') or str(error)
                    failures.append({'name': name, 'error': detail})
                    event('training_failed', name=name, error=detail)
                if name in children:
                    children.pop(name).wait()
                del active[name]
            if not failures:
                readings = gpu_measurements()
                ready = ready_jobs(jobs, set(completed), set(active))
                for pair in PAIRS:
                    idle = pair_can_dispatch(pair, active, readings, max_jobs,
                                manifest.get('reserved_gpu_mb_per_job', 11000),
                                manifest.get('gpu_memory_budget_mb', 23000))
                    streak[pair] = streak[pair] + 1 if idle else 0
                    if streak[pair] < 2 or not ready:
                        continue
                    entry = ready.pop(0)
                    verify_sources(root, runtime); verify_dispatcher(work)
                    if Path(entry['log']).exists() or Path(entry['output']).exists():
                        raise RuntimeError('refusing unclaimed existing output: ' + entry['name'])
                    if entry['method'] in ('random', 'high'):
                        cfg = json.loads(Path(entry['config']).read_text())
                        validate_reference(cfg['sa_hoep_reference_counts_path'], entry['seed'])
                    command = [sys.executable, str(Path(__file__).resolve()), '--project-root', str(root),
                               '--runtime-dir', str(runtime), '--worker', entry['name'],
                               '--gpu-pair', ','.join(map(str, pair))]
                    proof_path = work / 'job_results' / (entry['name'] + '.json')
                    process = launch_owned_worker(command, entry['name'], pair, proof_path)
                    children[entry['name']] = process
                    active[entry['name']] = {'name': entry['name'], 'entry': entry, 'gpu_pair': pair,
                        'pid': process.pid, 'start_ticks': process_info(process.pid)['start_ticks']}
                    event('training_start', name=entry['name'], gpu_pair=pair, worker_pid=process.pid)
            save('draining_failed' if failures else 'training')
            if failures and not active:
                break
            time.sleep(3)
        save('failed' if failures else 'complete')
        event('stable_queue_end', completed=len(completed), failures=failures)
        return 1 if failures else 0
    except Exception as error:
        failures.append({'error': str(error)})
        event('stable_scheduler_failed', error=str(error)); save('failed')
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--runtime-dir', required=True)
    parser.add_argument('--previous-runtime')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--worker')
    parser.add_argument('--gpu-pair')
    parser.add_argument('--start-fd', type=int)
    args = parser.parse_args()
    if args.prepare:
        if not args.previous_runtime:
            parser.error('--prepare requires --previous-runtime')
        prepare(args.project_root, args.previous_runtime, args.runtime_dir)
    elif args.worker:
        pair = tuple(map(int, args.gpu_pair.split(',')))
        if pair not in PAIRS:
            parser.error('GPU pair uses an excluded or unauthorized card')
        if args.start_fd is not None:
            token = os.read(args.start_fd, 1)
            os.close(args.start_fd)
            if token != b'1':
                raise SystemExit('coordinator exited before releasing the training gate')
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        raise SystemExit(worker(args.project_root, args.runtime_dir, args.worker, pair))
    else:
        raise SystemExit(run(args.project_root, args.runtime_dir))


if __name__ == '__main__':
    main()
