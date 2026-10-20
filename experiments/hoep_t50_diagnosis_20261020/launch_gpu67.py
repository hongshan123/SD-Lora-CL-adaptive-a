"""One training trajectory plus asynchronous read-only boundary evaluation."""
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'source'
OLD=ROOT.parent/'.runtime_hoep_t50_diagnosis_20261020'
PYTHON='/home/hongzhijun/miniconda3/envs/sdlora/bin/python'
CONFIG=ROOT/'configs/formal.json'
CFG=json.loads(CONFIG.read_text())
OUT=Path(CFG['filepath'])
DIAG=OUT/'diagnosis'
CHILDREN=[]

def env(gpu):
    return dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1')

def write_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)

def spawn(cmd,gpu,log):
    with Path(log).open('w') as stream:
        p=subprocess.Popen(cmd,cwd=SOURCE,env=env(gpu),stdout=stream,stderr=subprocess.STDOUT)
    CHILDREN.append(p)
    return p

def check_result(task,folder=DIAG):
    result=json.loads((folder/f'task_{task:03d}.json').read_text())
    if result['task']!=task or result['seed']!=1993:raise RuntimeError('diagnostic identity mismatch')
    if not Path(result['logits_file']).is_file():raise RuntimeError('missing logits artifact')
    for key in ['real_premerge_features','real_deployed_features','s3_s4_logits','s5_deployed_logits']:
        chk=result['checks'][key]
        if chk['max_abs']>1e-4 or chk['relative_l2']>1e-5:raise RuntimeError(f'failed reconstruction {key}')
    if len(result['branches'])!=24:raise RuntimeError('incomplete branch diagnostics')
    return result

def update(phase,**details):
    value={'phase':phase,'time':time.strftime('%Y-%m-%d %H:%M:%S %z'),'host':os.uname().nodename,
           'training_gpu':6,'diagnosis_gpus_after_training':[6,7],
           'complete_snapshots':len(list(OUT.glob('task_snapshots/task_*/complete.json'))),
           'complete_diagnostics':len(list(DIAG.glob('task_*.json'))),
           'output':str(OUT),'summary':str(DIAG/'summary.md'),**details}
    write_json(ROOT/'status.json',value)

def stop(signum,frame):
    raise KeyboardInterrupt(f'signal {signum}')

signal.signal(signal.SIGTERM,stop)
signal.signal(signal.SIGINT,stop)
try:
    lock=(ROOT/'coordinator.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for gpu in [6,7]:
        used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if used>500:raise RuntimeError(f'GPU{gpu} not idle: {used} MiB')
    if OUT.exists():raise RuntimeError('fresh training output already exists; refusing overwrite')
    files={str(p.relative_to(SOURCE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCE.rglob('*.py')) if '__pycache__' not in str(p)}
    write_json(ROOT/'configs/source_provenance.json',{'sha256':files,'parent_source':str(OLD/'source'),
               'config_sha256':hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
               'training_protocol':'same single-GPU trajectory: effective128 microbatch64 FP32; train GPU6; diagnostics isolated GPU7',
               'verification_source':str(ROOT/'verification/previous_snapshot_equivalence.json')})
    unit=spawn([PYTHON,'-m','pytest','-q','tests/test_hoep_diagnostic.py'],6,ROOT/'verification/unit_tests.log')
    if unit.wait()!=0:raise RuntimeError('unit tests failed')
    train=spawn([PYTHON,'main.py','--config',str(CONFIG)],6,ROOT/'logs/formal.log')
    # Validate inference optimizations on the exact previously audited snapshots.
    validation=ROOT/'verification/optimized_diagnostic';validation.mkdir(exist_ok=True)
    reference=OLD/'results/hoep_cifar1993_t50_nc0_diag_smoke_on_20261020'
    verifier=spawn([PYTHON,'scripts/diagnose_hoep.py','--run-dir',str(reference),'--task','1',
                    '--output-dir',str(validation),'--batch-size','64'],7,ROOT/'verification/optimized_diagnostic.log')
    active={};done=set();validated=False;last_summary_count=-1
    spec=importlib.util.spec_from_file_location('summary',SOURCE/'scripts/summarize_hoep_diagnosis.py')
    summary=importlib.util.module_from_spec(spec);spec.loader.exec_module(summary)
    while True:
        rc=train.poll()
        if rc is not None and rc!=0:raise RuntimeError(f'training exited {rc}; see logs/formal.log')
        if not validated and verifier.poll() is not None:
            if verifier.returncode!=0:raise RuntimeError('optimized diagnostic verification failed; see verification log')
            check_result(1,validation)
            checker=spawn([PYTHON,str(ROOT/'verify_optimization.py'),str(reference/'diagnosis'),str(validation)],7,ROOT/'verification/compare_optimization.log')
            if checker.wait()!=0:raise RuntimeError('optimized/reference diagnostic outputs differ')
            validated=True
            write_json(ROOT/'verification/optimization_passed.json',{'passed':True,'reference':str(reference/'diagnosis/task_001.json'),'new':str(validation/'task_001.json')})
        for gpu,(task,child) in list(active.items()):
            if child.poll() is not None:
                if child.returncode!=0:raise RuntimeError(f'GPU{gpu} boundary {task} failed: {DIAG}/task_{task:03d}.log')
                check_result(task);done.add(task);del active[gpu]
        if validated:
            for gpu in ([7] if rc is None else [7,6]):
                if gpu in active:continue
                busy={t for t,p in active.values()}
                candidates=[t for t in range(1,50) if t not in done and t not in busy and
                            (OUT/'task_snapshots'/f'task_{t:03d}'/'complete.json').exists() and
                            (OUT/'task_snapshots'/f'task_{t-1:03d}'/'complete.json').exists()]
                if candidates:
                    task=min(candidates);DIAG.mkdir(exist_ok=True)
                    child=spawn([PYTHON,'scripts/diagnose_hoep.py','--run-dir',str(OUT),'--task',str(task),
                                 '--output-dir',str(DIAG),'--batch-size','64'],gpu,DIAG/f'task_{task:03d}.log')
                    active[gpu]=(task,child)
        if len(done)!=last_summary_count:
            summary.summarize(DIAG);last_summary_count=len(done)
        update('training_and_diagnosis' if rc is None else 'draining_diagnosis',
               training_pid=train.pid,training_returncode=rc,diagnostic_verified=validated,
               active_diagnostics={str(gpu):{'task':task,'pid':p.pid} for gpu,(task,p) in active.items()},
               verification_pid=verifier.pid if not validated else None)
        if rc==0 and len(done)==49 and not active:break
        if rc==0 and not active and validated and len(done)<49:raise RuntimeError('training exited without complete 50-task snapshots')
        time.sleep(10)
    sys.path.insert(0,str(SOURCE))
    from utils.hoep_snapshots import audit_snapshot
    for task in range(50):audit_snapshot(OUT/'task_snapshots'/f'task_{task:03d}')
    hashes=json.loads((OUT/'p0_hashes.json').read_text())
    if len(hashes)!=50:raise RuntimeError('missing training/eval hashes')
    if any('post_eval_model_hash' not in hashes[str(i)] for i in range(50)):raise RuntimeError('incomplete final evaluation')
    update('complete',training_returncode=0,diagnosed_boundaries=49)
except BaseException as exc:
    update('interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc))
    traceback.print_exc()
    for child in CHILDREN:
        if child.poll() is None:child.terminate()
    for child in CHILDREN:
        if child.poll() is None:
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.kill()
    sys.exit(1)
