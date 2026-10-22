#!/usr/bin/env python3
"""Continue original single-GPU diagnosis runs without repeating completed work."""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import fcntl
import importlib.util
import json
from pathlib import Path
import queue
import re
import shutil
import threading
import uuid


def read_events(runtime):
    return [json.loads(line) for line in (Path(runtime) / 'pipeline_events.jsonl').read_text().splitlines()
            if line.strip()]


def training_complete(runtime, entry, events):
    runtime = Path(runtime)
    name = Path(entry['config']).stem
    kinds = {'smoke_training_complete', 'formal_training_complete'}
    if not any(e.get('kind') in kinds and e.get('name') == name for e in events):
        return False
    log = runtime / 'logs' / (name + '.log')
    if not log.is_file():
        return False
    body = log.read_text(errors='replace')
    matches = re.findall(r'CNN top1 curve:\s*(\[[^\n]+\])', body)
    if not matches or 'Forgetting (CNN):' not in body:
        return False
    encoded = re.sub(r'np\.float(?:32|64)\(([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\)',
                     r'\1', matches[-1])
    try:
        curve = ast.literal_eval(encoded)
    except (ValueError, SyntaxError):
        return False
    if not isinstance(curve, list) or len(curve) != 10:
        return False
    return not entry['snapshots'] or (
        runtime / 'results' / name / 'task_snapshots/task_009/complete.json').is_file()


def diagnosis_complete(runtime, entry, task, events):
    name = Path(entry['config']).stem
    if not any(e.get('kind') == 'diagnosis_complete' and e.get('name') == name
               and e.get('task') == task for e in events):
        return False
    folder = Path(runtime) / 'diagnostics' / name
    summary = folder / f'task_{task:03d}.json'
    if not summary.is_file():
        return False
    try:
        result = json.loads(summary.read_text())
        return result['task'] == task and Path(result['logits_file']).is_file()
    except (ValueError, KeyError):
        return False


def archive_interrupted(runtime, entry):
    runtime = Path(runtime)
    name = Path(entry['config']).stem
    archive = runtime / 'interrupted_attempts' / (name + '_' + uuid.uuid4().hex)
    archive.mkdir(parents=True, exist_ok=False)
    for source, target in ((runtime / 'logs' / (name + '.log'), archive / 'run.log'),
                           (runtime / 'results' / name, archive / 'output'),
                           (runtime / 'diagnostics' / name, archive / 'diagnostics')):
        if source.exists():
            shutil.move(str(source), str(target))
    for log in (runtime / 'logs').glob(name + '_task*_diagnosis.log'):
        shutil.move(str(log), str(archive / log.name))
    return archive


def run(runtime):
    runtime = Path(runtime).resolve()
    spec = importlib.util.spec_from_file_location('cuda6_original_pipeline', runtime / 'pipeline_cuda6.py')
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    lock = (runtime / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stopped = threading.Event()
    try:
        events = read_events(runtime)
        audit = json.loads((runtime / 'smoke_audit.json').read_text())
        if not all(audit[p]['passed'] for p in ('live', 'frozen')):
            raise RuntimeError('original snapshot equivalence audit did not pass')
        if not any(e.get('kind') == 'smoke_audit_passed' for e in events):
            raise RuntimeError('original snapshot equivalence audit exit proof is missing')
        if not all(training_complete(runtime, e, events) for e in pipeline.MANIFEST['smoke']):
            raise RuntimeError('smoke runs are incomplete; original preflight must be repeated')
        entries = [e for e in pipeline.MANIFEST['smoke'] if e['snapshots']] + pipeline.MANIFEST['formal']
        work = queue.Queue()
        reused = []
        for entry in entries:
            name = Path(entry['config']).stem
            trained = training_complete(runtime, entry, events)
            if trained and all(diagnosis_complete(runtime, entry, t, events) for t in range(1, 10)):
                reused.append(name)
                continue
            if not trained and ((runtime / 'results' / name).exists()
                                or (runtime / 'logs' / (name + '.log')).exists()):
                archive = archive_interrupted(runtime, entry)
                pipeline.event('user_resume_interrupted_archived', name=name, archive=str(archive))
            work.put((entry, trained))
        pipeline.event('user_resume_start', completed_reused=reused, pending=work.qsize(), gpu_lanes=[0, 1])

        def lane(gpu):
            while not stopped.is_set():
                try:
                    entry, trained = work.get_nowait()
                except queue.Empty:
                    return
                try:
                    name = Path(entry['config']).stem
                    run_dir = runtime / 'results' / name
                    if not trained:
                        pipeline.wait_gpu_idle(gpu)
                        pipeline.event('formal_training_start', name=name, gpu=gpu)
                        pipeline.run_logged([pipeline.TORCHRUN, '--standalone', '--nproc_per_node=1',
                                             'main.py', '--config=' + entry['config']],
                                            runtime / 'logs' / (name + '.log'), gpu)
                        if not (run_dir / 'task_snapshots/task_009/complete.json').is_file():
                            raise RuntimeError(f'formal run lacks task 9 snapshot: {name}')
                        pipeline.event('formal_training_complete', name=name)
                    output = runtime / 'diagnostics' / name
                    output.mkdir(parents=True, exist_ok=True)
                    remaining = [t for t in range(1, 10)
                                 if not diagnosis_complete(runtime, entry, t, read_events(runtime))]
                    if remaining:
                        pipeline.wait_gpu_idle(gpu)
                    for task in remaining:
                        log = runtime / 'logs' / f'{name}_task{task:03d}_diagnosis.log'
                        if log.exists():
                            shutil.move(str(log), str(log.with_name(log.name + '.interrupted_' + uuid.uuid4().hex)))
                        pipeline.event('diagnosis_start', name=name, task=task, gpu=gpu)
                        pipeline.run_logged([pipeline.PYTHON, 'scripts/diagnose_live_frozen.py',
                                             '--run-dir', str(run_dir), '--task', str(task),
                                             '--output-dir', str(output), '--device', 'cuda:0',
                                             '--batch-size', '32'], log, gpu)
                        pipeline.event('diagnosis_complete', name=name, task=task)
                except Exception:
                    stopped.set()
                    raise
                finally:
                    work.task_done()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(lane, gpu) for gpu in (0, 1)]
            for future in futures:
                future.result()
        if not work.empty():
            raise RuntimeError('recovery stopped with unfinished entries')
        log = runtime / 'summary.log'
        if log.exists():
            shutil.move(str(log), str(log.with_name('summary.before_resume_' + uuid.uuid4().hex + '.log')))
        pipeline.run_logged([pipeline.PYTHON, 'scripts/summarize_live_frozen_diagnostics.py',
                             '--experiment-dir', str(runtime)], log, 0)
        pipeline.event('all_training_and_diagnostics_complete')
    except Exception as error:
        pipeline.event('pipeline_failed', error=str(error))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-dir', required=True)
    args = parser.parse_args()
    run(args.runtime_dir)
