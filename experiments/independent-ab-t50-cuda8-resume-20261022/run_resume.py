"""Validate the recovered boundary, then isolate each remaining task process."""
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
sys.path.insert(0, '/home/hongzhijun/hongshan/cuda9-baselines-multiseed-20261020')
from gpu_leases import GpuLease


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def verify():
    for relative, digest in json.loads((ROOT / 'source_hashes.json').read_text()).items():
        assert hashlib.sha256((SOURCE / relative).read_bytes()).hexdigest() == digest, relative


def main():
    assert not (ROOT / 'started.json').exists(), 'Existing attempt retained; inspect before restarting'
    lease = GpuLease.acquire([0])
    assert lease is not None, 'GPU0 lease unavailable'
    used = int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    assert used < 300, ('GPU0 no longer idle',used)
    write(ROOT/'started.json',dict(pid=os.getpid(),started=time.time(),gpu=0))
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1',PYTHONFAULTHANDLER='1',
               OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
               CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    base = json.loads((ROOT/'base_config.json').read_text())
    checkpoint = ROOT/'initial_boundary'
    exit_code = 1
    try:
        stages = [('validation',42,True)] + [(f'task_{task:03d}',task,False) for task in range(42,50)]
        for name,task,validation in stages:
            attempt = ROOT/'attempts'/name
            attempt.mkdir(parents=True)
            cfg = dict(base,filepath=str(attempt/'output')+'/',prefix='independent_resume_'+name,
                       sa_p0_run_id='independent_resume_'+name+'_20261022',
                       sa_resume_boundary_from=str(checkpoint),
                       sa_boundary_checkpoint_root=str(ROOT/'checkpoints'),
                       sa_resume_validate_only=validation,sa_resume_verify_eval=validation,
                       max_tasks=task+1)
            config = attempt/'config.json'
            write(config,cfg)
            verify()
            log = attempt/'train.log'
            write(ROOT/'status.json',dict(state='running',phase='boundary_validation' if validation else 'training',
                  task_id=task,completed_tasks=task,gpu=0,checkpoint=str(checkpoint),log=str(log),updated=time.time()))
            with log.open('x') as stream:
                child = subprocess.Popen([sys.executable,str(SOURCE/'main.py'),'--config',str(config)],
                    cwd=SOURCE,env=env,stdout=stream,stderr=subprocess.STDOUT,pass_fds=lease.fds)
                write(attempt/'process.json',dict(pid=child.pid,started=time.time()))
                code = child.wait()
            write(attempt/'exit.json',dict(returncode=code,finished=time.time()))
            assert code == 0, (name,'child failed',code)
            verify()
            audit = json.loads((attempt/'output/restore_audit.json').read_text())
            assert audit['passed'] and audit['model_bitwise_equal']
            if validation:
                assert audit['test_top1_matches_original'] and audit['per_task_accuracy_matches_original']
                write(ROOT/'boundary_validation.json',audit)
                print('BOUNDARY_VALIDATION_PASS',audit,flush=True)
                continue
            checkpoint = ROOT/'checkpoints'/f'task_{task+1:03d}'
            meta = json.loads((checkpoint/'boundary.json').read_text())
            assert meta['tasks_completed']==task+1
            if task==42:
                audit=json.loads((attempt/'output/replay_training_audit.json').read_text())
                assert audit['passed']
                write(ROOT/'replay_training_audit.json',audit)
            metrics=dict(tasks_completed=task+1,tasks=50,complete=task==49,
                         curve=meta['cnn_curve']['top1'],final_accuracy=meta['cnn_curve']['top1'][-1],
                         average_incremental_accuracy=sum(meta['cnn_curve']['top1'])/(task+1),
                         seed=1993,mode='independent_bank',transport=True,normcap=False,
                         resumed_from_completed_tasks=42,checkpoint=str(checkpoint),gpu=0)
            write(ROOT/'metrics.json',metrics)
            write(ROOT/'status.json',dict(state='complete' if task==49 else 'between_tasks',
                  completed_tasks=task+1,gpu=0,checkpoint=str(checkpoint),updated=time.time()))
            print('TASK_COMPLETE',task+1,metrics['final_accuracy'],flush=True)
        exit_code=0
    except BaseException:
        error=traceback.format_exc();print(error,flush=True)
        write(ROOT/'status.json',dict(state='failed',error=error,gpu=0,checkpoint=str(checkpoint),updated=time.time()))
    finally:
        write(ROOT/'exit.json',dict(exit_code=exit_code,finished=time.time()))
        lease.close()
    return exit_code


if __name__=='__main__':
    sys.exit(main())
