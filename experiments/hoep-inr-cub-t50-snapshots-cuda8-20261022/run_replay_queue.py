"""Run paired fixed-reference replay after the snapshot training passes audit."""
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


def write(path,item):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(item,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def full_hash_audit(entry):
    reference=json.loads(Path(entry['reference_p0']).read_text())
    actual=json.loads((Path(entry['outputs']['formal'])/'p0_hashes.json').read_text())
    keys=['post_train_model_hash','post_train_rng_hash',
          'post_eval_model_hash','post_eval_rng_hash']
    assert set(actual)==set(map(str,range(50)))
    for task in range(50):
        for key in keys:
            assert actual[str(task)][key]==reference[str(task)][key],(task,key)
    return dict(tasks=50,keys=keys,bitwise_equal=True)


def main():
    status={'state':'waiting_for_training','datasets':{},'started':time.time()}
    write(ROOT/'replay_status.json',status)
    lease=None
    try:
        manifest=json.loads((ROOT/'manifest.json').read_text())
        while True:
            training=json.loads((ROOT/'status.json').read_text())
            if training['state']=='complete':break
            if training['state']=='failed':raise RuntimeError('snapshot training failed; replay withheld')
            time.sleep(30)
        for ds,entry in manifest['datasets'].items():
            proof=full_hash_audit(entry)
            write(ROOT/'proofs'/f'{ds}_formal_full_p0.json',proof)
            status['datasets'][ds]={'full_p0_bitwise_equal':True,'state':'validated'}
        status['state']='waiting_for_gpu';write(ROOT/'replay_status.json',status)
        while lease is None:
            lease=GpuLease.acquire([0,2])
            if lease is None:time.sleep(30)
        while True:
            values=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used',
                      '--format=csv,noheader,nounits'],text=True)
            used={int(row.split(',')[0]):int(row.split(',')[1]) for row in values.splitlines()}
            if used[0]<300 and used[2]<300:break
            time.sleep(30)
        jobs={}
        for ds,gpu in [('imagenetr',0),('cub',2)]:
            log=ROOT/'logs'/f'{ds}_anchor_replay.log'
            stream=log.open('x')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',
                     CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',
                     OMP_NUM_THREADS='2',PYTHONUNBUFFERED='1')
            command=[sys.executable,'-u',str(ROOT/'replay_anchor.py'),
                     '--dataset',ds,'--phase','formal','--limit','50','--device','cuda:0']
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,
                                     stderr=subprocess.STDOUT,start_new_session=True,
                                     pass_fds=lease.fds)
            jobs[ds]=(process,stream)
            status['datasets'][ds].update(state='running',gpu=gpu,pid=process.pid,log=str(log))
        status['state']='running';write(ROOT/'replay_status.json',status)
        for ds,(process,stream) in jobs.items():
            code=process.wait();stream.close()
            status['datasets'][ds].update(state='complete' if code==0 else 'failed',returncode=code)
            write(ROOT/'replay_status.json',status)
            if code:raise RuntimeError(f'{ds} replay exited {code}')
        status['state']='complete';status['finished']=time.time()
        write(ROOT/'replay_status.json',status)
    except BaseException:
        status['state']='failed';status['error']=traceback.format_exc()
        write(ROOT/'replay_status.json',status)
        raise
    finally:
        if lease:lease.close()


if __name__=='__main__':main()
