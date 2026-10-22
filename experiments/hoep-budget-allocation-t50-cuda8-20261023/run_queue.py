"""Shared idle-GPU pool, independent numerical gates and fresh A-step trials."""
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
from priority_lease import OneShotPriority
from journal import render,write_json

ROOT=Path(__file__).resolve().parent
PRIORITY=None


def read(path):
    for attempt in range(3):
        try:return json.loads(Path(path).read_text())
        except json.JSONDecodeError:
            if attempt==2:raise
            time.sleep(.1)


def verify(manifest):
    for name,digest in manifest['file_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, name
    for path,digest in manifest['reference_smoke_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    for ds in ('cub','imagenet-r'):
        assert (ROOT/'source/data'/ds/'train').is_dir()
        assert (ROOT/'source/data'/ds/'test').is_dir()
    proof=read(ROOT/'proofs/unit_tests_passed.json')
    assert proof['passed'] and proof['tests']==14


def save(manifest,status):
    if PRIORITY is not None:status['priority_admission']=PRIORITY.status()
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


def audit_smoke(jobs,manifest,status):
    keys=['post_train_model_hash','post_train_rng_hash','post_eval_model_hash','post_eval_rng_hash']
    proof={};valid={}
    for arm in ('global_gain1','global_gain10','coverage_gain1','coverage_gain10'):
        group=[j for j in jobs if j['arm']==arm]
        error=None
        try:
            assert len(group)==2
            for job in group:
                assert status['jobs'][job['name']]['state']=='complete',job['name']
                hashes=read(Path(job['output'])/'p0_hashes.json')
                reference=read(Path(job['task0_reference'])/'p0_hashes.json')
                for key in keys:assert hashes['0'][key]==reference['0'][key],(job['name'],key,'Task0 differs')
                if arm.startswith('global_'):
                    reference=read(Path(job['three_task_reference'])/'p0_hashes.json')
                    for task in ('0','1','2'):
                        for key in keys:assert hashes[task][key]==reference[task][key],(job['name'],task,key,'unit gain differs')
                else:
                    same_gain=read(Path(job['three_task_reference'])/'p0_hashes.json')
                    for task in ('0','1','2'):
                        for key in ('post_train_rng_hash','post_eval_rng_hash'):
                            assert hashes[task][key]==same_gain[task][key],(job['name'],task,key,'RNG differs')
                    metrics=read(Path(job['output'])/'metrics.json')
                    assert metrics['budget_allocation']=='coverage'
                    assert metrics['allocation_protected_rows_verified'] is True
                    assert metrics['allocation_changed_tasks']>0,'coverage did not change a task allocation'
                    assert metrics['allocation_changed_plastic_branches']>0,'no actual plastic A change'
        except (AssertionError,OSError,ValueError) as failure:
            error=str(failure)
        valid[arm]=error is None
        proof[arm]=dict(passed=error is None,error=error)
    for job in manifest['jobs']:
        control='global_gain'+str(job['multiplier'])
        if job['phase']=='formal' and not (valid[control] and valid[job['arm']]):
            status['jobs'][job['name']].update(state='not_launched_invalid_smoke',
                error='Both dataset checks and the same-gain global control must pass')
    write_json(ROOT/'proofs/smoke_equality.json',proof)
    save(manifest,status)
    return valid


def phase_run(phase,manifest,status):
    jobs=[j for j in manifest['jobs'] if j['phase']==phase and status['jobs'][j['name']]['state']!='not_launched_invalid_smoke']
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
                if phase=='formal':
                    candidate=read(Path(job['output'])/'p0_hashes.json')['0']
                    reference=read(Path(job['matched_frozen'])/'p0_hashes.json')['0']
                    for key in ('post_train_model_hash','post_train_rng_hash','post_eval_model_hash','post_eval_rng_hash'):
                        assert candidate[key]==reference[key],(name,'formal Task0 differs',key)
            else:
                record.update(state='failed',error=Path(job['log']).read_text(errors='replace')[-2200:]);failed.append(name)
            del running[name];print('EXIT',name,code,flush=True);save(manifest,status)
        used={item['gpu'] for item in running.values()}
        for gpu in manifest['gpus']:
            if not pending or gpu in used:continue
            handled,lease=PRIORITY.try_take(gpu) if PRIORITY is not None else (False,None)
            if not handled:lease=GpuLease.acquire([gpu])
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
        save(manifest,status)
        if pending or running:time.sleep(20)
    if phase=='smoke':audit_smoke(jobs,manifest,status)
    else:assert not failed,('failed formal jobs; outputs preserved',failed)
    verify(manifest)


def main():
    global PRIORITY
    lock=(ROOT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=read(ROOT/'manifest.json');verify(manifest)
    if (ROOT/'status.json').exists():
        status=read(ROOT/'status.json')
    else:
        signatures=[j['signature'] for j in manifest['jobs'] if j['phase']!='invalid_attempt']
        assert len(set(signatures))==len(signatures),'duplicate experiment'
        status=dict(state='prepared',started=time.time(),jobs={j['name']:dict(state='pending',completed_tasks=0) for j in manifest['jobs']})
    if manifest.get('priority_wait_gpu') is not None:
        assert manifest['priority_wait_gpu'] in manifest['gpus']
        PRIORITY=OneShotPriority(manifest['priority_wait_gpu'])
    try:
        save(manifest,status)
        for phase in ('smoke','formal'):phase_run(phase,manifest,status)
        state='complete_with_invalid_attempts' if any(r['state'] in ('failed','not_launched_invalid_smoke') for r in status['jobs'].values()) else 'complete'
        status.update(state=state,finished=time.time());save(manifest,status)
    except BaseException:
        status.update(state='failed',error=traceback.format_exc());save(manifest,status);raise
    finally:
        if PRIORITY is not None:PRIORITY.close_unused()


if __name__=='__main__':main()
