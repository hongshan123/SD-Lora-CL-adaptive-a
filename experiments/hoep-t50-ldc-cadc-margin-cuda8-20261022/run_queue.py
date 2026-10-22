import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/home/hongzhijun/hongshan/cuda9-baselines-multiseed-20261020')
from gpu_leases import GpuLease


def write(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)


def verify():
    for name,expected in json.loads((ROOT/'source_hashes.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    cfg=json.loads((ROOT/'manifest.json').read_text())
    assert hashlib.sha256((Path(cfg['cache']).parent/'cache_manifest.json').read_bytes()).hexdigest()==cfg['cache_manifest_sha256']
    for task,a in json.loads((ROOT/'oracle_provenance.json').read_text()).items():
        p=Path(cfg['diagnosis_source'])/f'task_{int(task):03d}.pt'
        assert hashlib.sha256(p.read_bytes()).hexdigest()==a['compact_pt_sha256']


def run(action,gpu,lease):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu) if gpu is not None else '',
       OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
       CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONFAULTHANDLER='1',PYTHONUNBUFFERED='1')
    with (ROOT/'logs'/f'{action}.log').open('x') as stream:
        child=subprocess.Popen([sys.executable,'-u',str(ROOT/'diagnose.py'),action],env=env,
            cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,pass_fds=lease.fds)
        write(ROOT/f'{action}_process.json',dict(pid=child.pid,gpu=gpu,started=time.time()))
        code=child.wait()
    write(ROOT/f'{action}_exit.json',dict(returncode=code,finished=time.time()))
    assert code==0,(action,code)


def main():
    lease=None
    try:
        assert not (ROOT/'started.json').exists()
        verify();lease=GpuLease.acquire([0,1,2]);assert lease is not None
        for gpu in [0,1,2]:
            used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used',
                     '--format=csv,noheader,nounits'],text=True));assert used<300,(gpu,used)
        (ROOT/'logs').mkdir(exist_ok=True)
        write(ROOT/'started.json',dict(pid=os.getpid(),started=time.time(),gpus=[0,1,2]))
        run('check',None,lease)
        write(ROOT/'status.json',dict(state='running',phase='compensation_modules',gpus={'ldc':0,'cadc':1}))
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(run,method,gpu,lease) for method,gpu in [('ldc',0),('cadc',1)]]
            for future in concurrent.futures.as_completed(futures):future.result()
        verify()
        write(ROOT/'status.json',dict(state='running',phase='margin_diagnostic',gpu=2))
        run('margin',2,lease);verify()
        previous=Path(json.loads((ROOT/'manifest.json').read_text())['cache']).parent
        results={}
        for variant in ['none','current','half','reliable','ldc','cadc','cadc_uniform']:
            parent=previous if variant in ['none','current','half','reliable'] else ROOT
            metrics=json.loads((parent/'experiments'/variant/'metrics.json').read_text())
            assert metrics['complete'] and metrics['completed_tasks']==50
            results[variant]=metrics
        write(ROOT/'results.json',results)
        write(ROOT/'status.json',dict(state='complete',phase='complete',finished=time.time()))
        write(ROOT/'exit.json',dict(returncode=0,finished=time.time()))
    except BaseException:
        write(ROOT/'status.json',dict(state='failed',error=traceback.format_exc(),updated=time.time()))
        write(ROOT/'exit.json',dict(returncode=1,finished=time.time()))
        raise
    finally:
        if lease:lease.close()


if __name__=='__main__':main()
