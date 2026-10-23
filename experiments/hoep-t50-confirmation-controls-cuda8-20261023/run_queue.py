"""Four independent GPU lanes, smoke gates, signature deduplication and journal."""
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time
import traceback

from gpu_leases import GpuLease
from journal import render,write_json

ROOT=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text())


def verify(manifest):
    for name,digest in manifest['file_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, name
    for ds in ('cub','imagenet-r'):
        assert (ROOT/'source/data'/ds/'train').is_dir()
        assert (ROOT/'source/data'/ds/'test').is_dir()


def save(manifest,status):
    status['updated']=time.time();write_json(ROOT/'status.json',status)
    registry={}
    for job in manifest['jobs']:
        registry.setdefault(job['signature'],[]).append(dict(name=job['name'],**status['jobs'][job['name']]))
    write_json(ROOT/'registry.json',registry)
    render(manifest,status)


def free_gpu(gpu):
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    memory={int(line.split(',')[0]):int(line.split(',')[1]) for line in text.splitlines()}
    return memory[gpu]<300


def audit_smoke(jobs):
    keys=['post_train_model_hash','post_train_rng_hash','post_eval_model_hash','post_eval_rng_hash']
    result={}
    for ds in ('cub','imagenetr'):
        group={j['arm']:j for j in jobs if j['dataset']==ds}
        hashes={arm:read(Path(j['output'])/'p0_hashes.json') for arm,j in group.items()}
        for arm,h in hashes.items():
            for key in keys:assert h['0'][key]==hashes['hoep']['0'][key],(ds,arm,key,'Task0 changed')
        for task in ('0','1','2'):
            for key in keys:assert hashes['original'][task][key]==hashes['hoep'][task][key],(ds,task,key,'disabled loss changed baseline')
        result[ds]={'common_task0_all_arms':True,'disabled_learner_three_tasks_exact':True}
    write_json(ROOT/'proofs/smoke_equality.json',result)


def phase_run(phase,manifest,status):
    jobs=[j for j in manifest['jobs'] if j['phase']==phase]
    pending=[];running={};failed=[]
    for job in jobs:
        record=status['jobs'][job['name']]
        if record['state']=='complete':continue
        assert record['state']=='pending',('nonfresh task requires audit',job['name'],record['state'])
        assert not Path(job['output']).exists(),('existing output: do not retrain',job['name'])
        pending.append(job)
    status['state']=phase;save(manifest,status)
    while pending or running:
        for name,item in list(running.items()):
            job=item['job'];record=status['jobs'][name]
            record['completed_tasks']=len(list((Path(job['output'])/'task_snapshots').glob('task_*/complete.json')))
            code=item['process'].poll()
            if code is None:
                text=Path(job['log']).read_text(errors='replace')
                if re.search(r'Loss\s+(?:nan|[+-]?inf)(?:[,\s]|$)',text,re.I):
                    record['nonfinite_detected']=True
                    item['process'].terminate()
                continue
            item['stream'].close();item['lease'].close()
            record.update(returncode=code,ended=time.time())
            metrics_file=Path(job['output'])/'metrics.json'
            if code==0 and metrics_file.exists():
                record.update(state='complete',metrics=read(metrics_file))
            else:
                record.update(state='failed',error=Path(job['log']).read_text(errors='replace')[-2200:]);failed.append(name)
            del running[name];print('EXIT',name,code,flush=True);save(manifest,status)
        used={item['gpu'] for item in running.values()}
        for gpu in manifest['gpus']:
            if not pending or gpu in used:continue
            lease=GpuLease.acquire([gpu])
            if lease is None:continue
            if not free_gpu(gpu):lease.close();continue
            job=pending.pop(0);stream=Path(job['log']).open('x')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',
                     CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1',
                     OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            process=subprocess.Popen([sys.executable,'-u',str(ROOT/'run_job.py'),'--config',job['config']],
                                     cwd=ROOT/'source',env=env,stdout=stream,stderr=subprocess.STDOUT,
                                     pass_fds=lease.fds)
            ticks=Path(f'/proc/{process.pid}/stat').read_text().split()[21]
            status['jobs'][job['name']].update(state='running',pid=process.pid,proc_start_ticks=ticks,
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),gpu=gpu,started=time.time())
            running[job['name']]=dict(job=job,process=process,stream=stream,lease=lease,gpu=gpu)
            print('START',phase,job['name'],'GPU',gpu,'PID',process.pid,flush=True);save(manifest,status)
        if pending or running:time.sleep(20)
    assert not failed,('failed jobs; formal gate stays closed',failed)
    if phase=='smoke':audit_smoke(jobs)
    verify(manifest)


def main():
    lock=(ROOT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=read(ROOT/'manifest.json');verify(manifest)
    if (ROOT/'status.json').exists():
        status=read(ROOT/'status.json')
    else:
        signatures=[j['signature'] for j in manifest['jobs'] if j['phase']!='invalid_attempt']
        assert len(set(signatures))==len(signatures),'duplicate experiment'
        status=dict(state='prepared',started=time.time(),jobs={j['name']:dict(state='pending',completed_tasks=0) for j in manifest['jobs']})
    try:
        save(manifest,status)
        for phase in ('formal',):phase_run(phase,manifest,status)
        status.update(state='complete',finished=time.time());save(manifest,status)
    except BaseException:
        status.update(state='failed',error=traceback.format_exc());save(manifest,status);raise


if __name__=='__main__':main()
