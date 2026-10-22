"""Wait for existing cuda8 training AND replay, then smoke-gate 8 paired runs."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from gpu_leases import GpuLease

ROOT = Path(__file__).resolve().parent


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def verify_files(manifest):
    for name, digest in manifest['file_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    assert (ROOT/'source/data/cifar-100-python/train').is_file()
    assert (ROOT/'source/data/imagenet-r/train').is_dir()


def prerequisite_state(manifest):
    upstream = Path(manifest['upstream_root'])
    records = {}
    for label, name in [('training','status.json'), ('replay','replay_status.json')]:
        path = upstream/name
        records[label] = read(path) if path.exists() else {'state':'missing'}
    return records, all(item['state'] == 'complete' for item in records.values())


def gpu_free(gpu):
    text = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used',
                                    '--format=csv,noheader,nounits'], text=True)
    used = {int(row.split(',')[0]): int(row.split(',')[1]) for row in text.splitlines()}
    return used[gpu] < 300


def compare_task0(jobs):
    groups = {}
    keys = ['post_train_model_hash','post_train_rng_hash','post_eval_model_hash','post_eval_rng_hash']
    for job in jobs:
        first = read(Path(job['output'])/'p0_hashes.json')['0']
        selected = {key:first[key] for key in keys}
        if job['dataset'] in groups:
            assert groups[job['dataset']] == selected, ('Task0 differs',job['name'])
        groups[job['dataset']] = selected
    return groups


def compare_disabled_learner(jobs):
    pair = {job['arm']:job for job in jobs if job['dataset']=='cifar100'}
    original = read(Path(pair['original']['output'])/'p0_hashes.json')
    disabled = read(Path(pair['baseline']['output'])/'p0_hashes.json')
    keys = ['post_train_model_hash','post_train_rng_hash','post_eval_model_hash','post_eval_rng_hash']
    for task in ('0','1'):
        for key in keys:
            assert original[task][key] == disabled[task][key], ('disabled learner differs',task,key)
    return dict(passed=True,tasks=[0,1],keys=keys)


def run_phase(phase, manifest, status):
    jobs = [job for job in manifest['jobs'] if job['phase']==phase]
    pending = list(jobs)
    running = {}
    finished = {}
    status['state'] = phase
    write(ROOT/'status.json',status)
    try:
        while pending or running:
            for name, item in list(running.items()):
                code = item['process'].poll()
                job = item['job']
                records = list((Path(job['output'])/'task_snapshots').glob('task_*/complete.json'))
                status['jobs'][name]['completed_tasks'] = len(records)
                if code is None:
                    continue
                item['stream'].close()
                item['lease'].close()
                finished[name] = code
                status['jobs'][name].update(state='complete' if code==0 else 'failed',
                                          returncode=code, finished=time.time())
                if code==0:
                    metrics = read(Path(job['output'])/'metrics.json')
                    status['jobs'][name].update(final_top1=metrics['final_top1'],
                                              average_accuracy=metrics['average_accuracy'],
                                              peak_reserved_bytes=metrics['peak_reserved_bytes'])
                print('EXIT', name, code, flush=True)
                del running[name]
            busy = {item['job']['gpu'] for item in running.values()}
            for job in list(pending):
                gpu = job['gpu']
                if gpu in busy:
                    continue
                lease = GpuLease.acquire([gpu])
                if lease is None:
                    continue
                if not gpu_free(gpu):
                    lease.close()
                    continue
                assert not Path(job['output']).exists(), ('output already exists',job['output'])
                stream = Path(job['log']).open('x')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                           CUBLAS_WORKSPACE_CONFIG=':4096:8', HF_HUB_OFFLINE='1',
                           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           PYTHONUNBUFFERED='1')
                process = subprocess.Popen([sys.executable,'-u',str(ROOT/'run_job.py'),job['config']],
                                           cwd=ROOT/'source',env=env,stdout=stream,stderr=subprocess.STDOUT,
                                           pass_fds=lease.fds)
                running[job['name']] = dict(process=process,stream=stream,lease=lease,job=job)
                busy.add(gpu)
                pending.remove(job)
                status['jobs'][job['name']].update(state='running',pid=process.pid,started=time.time())
                print('START',phase,job['name'],'gpu',gpu,'pid',process.pid,flush=True)
            status['updated'] = time.time()
            write(ROOT/'status.json',status)
            if pending or running:
                time.sleep(20)
    finally:
        for item in running.values():
            item['stream'].close()
            item['lease'].close()  # Child inherits the lease until exit.
    assert all(code==0 for code in finished.values()), finished
    assert len(finished)==len(jobs)
    equality = compare_task0(jobs)
    write(ROOT/'proofs'/f'{phase}_task0_equality.json',equality)
    if phase=='smoke':
        write(ROOT/'proofs'/'disabled_learner_equivalence.json', compare_disabled_learner(jobs))
    verify_files(manifest)


def main():
    lock = (ROOT/'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = read(ROOT/'manifest.json')
    status = dict(state='waiting_for_existing_training_and_replay',started=time.time(),
                  gpus=[0,1,2,3],jobs={j['name']:dict(state='pending',gpu=j['gpu'],phase=j['phase'],
                      dataset=j['dataset'],arm=j['arm'],output=j['output']) for j in manifest['jobs']})
    try:
        verify_files(manifest)
        write(ROOT/'status.json',status)
        while True:
            upstream, ready = prerequisite_state(manifest)
            status['upstream'] = upstream
            status['verification'] = 'mathematical_tests_passed; full_path_tests_queued_before_formal'
            status['updated'] = time.time()
            write(ROOT/'status.json',status)
            if ready:
                break
            time.sleep(30)
        for phase in ('smoke','formal'):
            verify_files(manifest)
            run_phase(phase,manifest,status)
        status.update(state='complete',finished=time.time())
        write(ROOT/'status.json',status)
    except BaseException:
        status.update(state='failed',error=traceback.format_exc(),updated=time.time())
        write(ROOT/'status.json',status)
        raise


if __name__=='__main__':
    main()
