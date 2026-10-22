"""Three GPU read-only extraction, then four prototype-history experiments."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, '/home/hongzhijun/hongshan/cuda9-baselines-multiseed-20261020')
from gpu_leases import GpuLease
LOCK = threading.Lock()
STATE = dict(state='starting', phase='preflight', completed_cache_tasks=0,
             gpu_workers={}, experiments={v:'pending' for v in ['none','current','half','reliable']})
STOP = threading.Event()


def write(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def publish(**changes):
    with LOCK:
        STATE.update(changes)
        STATE['completed_cache_tasks'] = len(list((ROOT/'cache').glob('task_*.json')))
        STATE['updated'] = time.time()
        write(ROOT/'status.json', STATE)


def verify_source():
    for path, expected in json.loads((ROOT/'source_hashes.json').read_text()).items():
        assert hashlib.sha256((ROOT/'source'/path).read_bytes()).hexdigest()==expected,path
    for path, expected in json.loads((ROOT/'runner_hashes.json').read_text()).items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path


def run(arguments, name, gpu=None, lease=None):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu) if gpu is not None else '',
               PYTHONUNBUFFERED='1', PYTHONFAULTHANDLER='1', OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    log = ROOT/'logs'/f'{name}.log'
    with log.open('x') as stream:
        child = subprocess.Popen([sys.executable, str(ROOT/'fourway.py'), *arguments],
            cwd=ROOT/'source', env=env, stdout=stream, stderr=subprocess.STDOUT,
            pass_fds=lease.fds if lease else ())
        if gpu is not None:
            with LOCK:
                STATE['gpu_workers'][str(gpu)]=dict(name=name,pid=child.pid,log=str(log),started=time.time())
            publish()
        code = child.wait()
    write(ROOT/'logs'/f'{name}.exit.json',dict(returncode=code,finished=time.time()))
    if code:
        STOP.set()
        raise RuntimeError(f'{name} exit={code}; inspect {log}')


def extract_worker(gpu, tasks, lease):
    for task in tasks:
        if STOP.is_set():return
        run(['--extract',str(task)],f'cache_{task:03d}',gpu,lease)
        publish()
    with LOCK:
        STATE['gpu_workers'][str(gpu)]=dict(state='finished')
    publish()


def evaluation(variant):
    with LOCK:STATE['experiments'][variant]='running'
    publish()
    run(['--evaluate',variant],f'eval_{variant}')
    with LOCK:STATE['experiments'][variant]='complete'
    publish()


def main():
    assert not (ROOT/'started.json').exists(), 'Preserve prior attempt before a new launch'
    (ROOT/'logs').mkdir(exist_ok=True)
    (ROOT/'cache').mkdir(exist_ok=True)
    verify_source()
    lease = GpuLease.acquire([1,2,3])
    assert lease is not None, 'GPU lease unavailable'
    try:
        for gpu in [1,2,3]:
            used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used',
                                              '--format=csv,noheader,nounits'],text=True).strip())
            assert used < 300, (gpu,used,'no longer idle')
        write(ROOT/'started.json',dict(pid=os.getpid(),started=time.time(),gpus=[1,2,3],host=os.uname().nodename))
        publish(state='running')
        run(['--check'],'unit_checks')
        # Real-data gates before scheduling all task states.
        run(['--extract','0'],'cache_000',1,lease)
        run(['--extract','1'],'cache_001',1,lease)
        write(ROOT/'preflight_passed.json',dict(passed=True,completed=time.time(),
              checks=['unit checks','Task0 and Task1 deployment probe','new prototype reconstruction',
                      'original Task1 Top1','Task1 original transport reproduction']))
        publish(phase='extracting_fixed_trajectory')
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(extract_worker,gpu,list(range(2+i,50,3)),lease)
                     for i,gpu in enumerate([1,2,3])]
            for future in concurrent.futures.as_completed(futures):future.result()
        assert len(list((ROOT/'cache').glob('task_*.json')))==50
        verify_source()
        write(ROOT/'cache_manifest.json',{p.name:json.loads(p.read_text()) for p in sorted((ROOT/'cache').glob('task_*.json'))})
        lease.close()
        publish(phase='prototype_rollouts',gpu_workers={})
        # Validate full baseline first; alternate arms compare their logits to it.
        evaluation('current')
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(evaluation,v) for v in ['none','half','reliable']]
            for future in concurrent.futures.as_completed(futures):future.result()
        verify_source()
        results={v:json.loads((ROOT/'experiments'/v/'metrics.json').read_text())
                 for v in ['none','current','half','reliable']}
        write(ROOT/'results.json',results)
        publish(state='complete',phase='complete')
        write(ROOT/'exit.json',dict(returncode=0,finished=time.time()))
    except BaseException:
        publish(state='failed',error=traceback.format_exc())
        write(ROOT/'exit.json',dict(returncode=1,finished=time.time()))
        raise
    finally:
        lease.close()


if __name__=='__main__':main()
