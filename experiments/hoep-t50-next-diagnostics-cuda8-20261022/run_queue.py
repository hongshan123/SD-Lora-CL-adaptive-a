"""Run two fixed-trajectory diagnostics on separate idle cuda8 GPUs."""
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
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def verify():
    for name,expected in json.loads((ROOT/'source_hashes.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    cfg=json.loads((ROOT/'manifest.json').read_text())
    prev=Path(cfg['previous_runtime'])
    assert hashlib.sha256((prev/'cache_manifest.json').read_bytes()).hexdigest()==cfg['cache_manifest_sha256']
    oracle=Path(cfg['oracle_runtime'])/'inputs/oracle_diag'
    for name,digest in cfg['oracle_input_sha256'].items():
        assert hashlib.sha256((oracle/name).read_bytes()).hexdigest()==digest,name


def run(action,gpu,lease):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu) if gpu is not None else '',
             OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
             CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONFAULTHANDLER='1',PYTHONUNBUFFERED='1')
    with (ROOT/'logs'/f'{action}.log').open('x') as stream:
        child=subprocess.Popen([sys.executable,'-u',str(ROOT/'diagnose_next.py'),action],
               cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,pass_fds=lease.fds)
        write(ROOT/'logs'/f'{action}_process.json',dict(pid=child.pid,gpu=gpu,started=time.time()))
        code=child.wait()
    write(ROOT/'logs'/f'{action}_exit.json',dict(returncode=code,finished=time.time()))
    assert code==0,(action,code)


def anchor(gpu,lease):
    run('anchor_extract',gpu,lease)
    run('anchor_eval',gpu,lease)


def main():
    lease=None
    try:
        assert not (ROOT/'started.json').exists()
        verify()
        lease=GpuLease.acquire([0,1]);assert lease is not None
        for gpu in (0,1):
            used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used',
                 '--format=csv,noheader,nounits'],text=True))
            assert used<300,(gpu,used)
        (ROOT/'logs').mkdir(exist_ok=True)
        write(ROOT/'started.json',dict(started=time.time(),gpus=[0,1],pid=os.getpid()))
        run('preflight',None,lease)
        write(ROOT/'status.json',dict(state='running',phase='oracle_and_anchor',started=time.time()))
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(run,'oracle',0,lease),pool.submit(anchor,1,lease)]
            for future in concurrent.futures.as_completed(futures):future.result()
        verify()
        prev=Path(json.loads((ROOT/'manifest.json').read_text())['previous_runtime'])
        names=['none','current','half','reliable','oracle_cosine','oracle_margin',
               'anchor_none','anchor_full','anchor_half']
        results={}
        for name in names:
            parent=prev if name in names[:4] else ROOT
            item=json.loads((parent/'experiments'/name/'metrics.json').read_text())
            assert item['complete'] and item['completed_tasks']==50,name
            results[name]=item
        write(ROOT/'results.json',results)
        maps=[torch_load(ROOT/'anchor_maps'/f'task_{t:03d}.pt') for t in range(1,50)]
        write(ROOT/'anchor_fit_summary.json',dict(enabled=sum(int(m['enabled']) for m in maps),
             total=49,mean_validation_gain=sum(m['validation_gain'] for m in maps)/49,
             mean_train_before=sum(m['train_before'] for m in maps)/49,
             mean_train_after=sum(m['train_after'] for m in maps)/49))
        write(ROOT/'status.json',dict(state='complete',finished=time.time()))
        write(ROOT/'exit.json',dict(returncode=0,finished=time.time()))
    except BaseException:
        write(ROOT/'status.json',dict(state='failed',error=traceback.format_exc(),updated=time.time()))
        write(ROOT/'exit.json',dict(returncode=1,finished=time.time()))
        raise
    finally:
        if lease:lease.close()


def torch_load(path):
    import torch
    return torch.load(path,map_location='cpu',weights_only=True)


if __name__=='__main__':main()
