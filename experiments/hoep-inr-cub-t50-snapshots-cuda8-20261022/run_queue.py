"""Paired T50 ImageNet-R/CUB replay with rank-zero read-only snapshots."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/home/hongzhijun/hongshan/cuda9-baselines-multiseed-20261020')
from gpu_leases import GpuLease


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def verify_inputs(manifest):
    for ds,entry in manifest['datasets'].items():
        src=Path(entry['source'])
        for name,expected in entry['source_hashes'].items():
            assert sha(src/name)==expected,(ds,name)
        for phase,record in entry['configs'].items():
            assert sha(record['path'])==record['sha256'],(ds,phase)
        assert sha(entry['reference_p0'])==entry['reference_p0_sha256']
        dataset='imagenet-r' if ds=='imagenetr' else 'cub'
        assert (src/'data'/dataset/'train').is_dir(),(ds,'train data absent')
        assert (src/'data'/dataset/'test').is_dir(),(ds,'test data absent')


def audit_snapshot(out,task):
    import torch
    directory=Path(out)/'task_snapshots'/f'task_{task:03d}'
    marker=directory/'complete.json'
    if not marker.exists():return None
    record=json.loads(marker.read_text())
    assert record['task_id']==task
    for name,expected in record['sha256'].items():
        assert sha(directory/name)==expected,(directory,name)
    required={'pre_merge.pt','sa_state.pt','sa_merged_lora.pt','sa_prototypes.pt',
              'deployed_probe_features.pt','config.json'}
    assert required.issubset(record['sha256']),required-set(record['sha256'])
    pre=torch.load(directory/'pre_merge.pt',map_location='cpu',weights_only=True)
    assert pre['task_id']==task and pre['known_classes']==4*task
    assert pre['total_classes']==4*(task+1) and len(pre['branches'])==24
    assert len(pre['old_prototypes'])==4*task
    deployed=torch.load(directory/'sa_merged_lora.pt',map_location='cpu',weights_only=True)
    assert deployed['task_id']==task and len(deployed['merged_b'])==24
    protos=torch.load(directory/'sa_prototypes.pt',map_location='cpu',weights_only=True)
    assert set(protos)==set(range(4*(task+1)))
    return dict(task=task,files=len(record['sha256']),sha256=sha(marker),
                known_classes=pre['known_classes'],total_classes=pre['total_classes'])


def compare_p0(entry,out,task_count):
    old=json.loads(Path(entry['reference_p0']).read_text())
    new=json.loads((Path(out)/'p0_hashes.json').read_text())
    keys=['post_train_model_hash','post_train_rng_hash','post_eval_model_hash','post_eval_rng_hash']
    for task in range(task_count):
        for key in keys:
            assert new[str(task)][key]==old[str(task)][key],(task,key,new[str(task)][key],old[str(task)][key])
    return dict(tasks=task_count,keys=keys,bitwise_equal=True)


def count_complete(out,maximum):
    directory=Path(out)/'task_snapshots'
    return sum((directory/f'task_{i:03d}/complete.json').is_file() for i in range(maximum))


def run_phase(phase,manifest,lease,status):
    jobs={}
    for ds,entry in manifest['datasets'].items():
        pair=entry['gpu_pair']
        assert len(pair)==2
        cfg=entry['configs'][phase]
        outfile=Path(entry['outputs'][phase])
        assert not outfile.exists(),('fresh-run output already exists',str(outfile))
        outfile.parent.mkdir(parents=True,exist_ok=True)
        log=ROOT/'logs'/f'{ds}_{phase}.log'
        stream=log.open('x')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,pair)),
                 CUDA_DEVICE_ORDER='PCI_BUS_ID',CUBLAS_WORKSPACE_CONFIG=':4096:8',
                 HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1')
        command=[sys.executable,'-m','torch.distributed.run','--standalone',
                 '--nproc_per_node=2','main.py','--config='+cfg['path']]
        process=subprocess.Popen(command,cwd=entry['source'],env=env,stdout=stream,
                                 stderr=subprocess.STDOUT,start_new_session=True,
                                 pass_fds=lease.fds)
        jobs[ds]=dict(process=process,stream=stream,started=time.time(),output=outfile,log=log)
        status['datasets'][ds]=dict(state='running',phase=phase,pid=process.pid,gpus=pair,
                                    output=str(outfile),log=str(log),started=jobs[ds]['started'])
        print('START',ds,phase,process.pid,pair,flush=True)
    write(ROOT/'status.json',status)
    finished={}
    while len(finished)<len(jobs):
        for ds,item in jobs.items():
            if ds in finished:continue
            code=item['process'].poll()
            entry=manifest['datasets'][ds]
            maximum=2 if phase=='smoke' else 50
            status['datasets'][ds]['completed_snapshots']=count_complete(item['output'],maximum)
            if code is None:continue
            item['stream'].close()
            finished[ds]=code
            status['datasets'][ds].update(state='phase_complete' if code==0 else 'failed',
                                          returncode=code,ended=time.time())
            write(ROOT/'status.json',status)
            print('EXIT',ds,phase,code,'snapshots',status['datasets'][ds]['completed_snapshots'],flush=True)
        status['updated']=time.time();write(ROOT/'status.json',status)
        if len(finished)<len(jobs):time.sleep(10)
    for ds,code in finished.items():assert code==0,(ds,phase,code)
    for ds,entry in manifest['datasets'].items():
        out=entry['outputs'][phase]
        limit=2 if phase=='smoke' else 50
        records=[audit_snapshot(out,task) for task in range(limit)]
        assert all(records),(ds,phase,'incomplete snapshot')
        p0=compare_p0(entry,out,2)
        write(ROOT/'proofs'/f'{ds}_{phase}_snapshots.json',
              dict(dataset=ds,phase=phase,complete_snapshots=limit,p0=p0,
                   first=records[0],last=records[-1],
                   all_snapshot_checksums_valid=True))
        status['datasets'][ds].update(state='validated',phase=phase,
                                      completed_snapshots=limit,p0_bitwise_equal=True)
        write(ROOT/'status.json',status)
    verify_inputs(manifest)


def main():
    lease=None
    try:
        manifest=json.loads((ROOT/'manifest.json').read_text())
        status=json.loads((ROOT/'status.json').read_text())
        assert status['state']=='prepared'
        verify_inputs(manifest)
        lease=GpuLease.acquire([0,1,2,3]);assert lease is not None
        gpu_text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used',
                             '--format=csv,noheader,nounits'],text=True)
        used={int(x.split(',')[0]):int(x.split(',')[1]) for x in gpu_text.splitlines()}
        assert all(used[g]<300 for g in [0,1,2,3]),used
        status.update(state='smoke',started=time.time(),gpus=[0,1,2,3])
        write(ROOT/'status.json',status)
        run_phase('smoke',manifest,lease,status)
        status.update(state='formal',smoke_passed=True,smoke_finished=time.time())
        write(ROOT/'status.json',status)
        run_phase('formal',manifest,lease,status)
        status.update(state='complete',finished=time.time())
        write(ROOT/'status.json',status)
        write(ROOT/'exit.json',dict(returncode=0,finished=time.time()))
    except BaseException:
        status=json.loads((ROOT/'status.json').read_text())
        status.update(state='failed',error=traceback.format_exc(),updated=time.time())
        write(ROOT/'status.json',status)
        write(ROOT/'exit.json',dict(returncode=1,finished=time.time()))
        raise
    finally:
        if lease:lease.close()


if __name__=='__main__':main()
