"""Validate B input metrics on idle GPUs5/7, then run paired T50 extensions."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

from gpu_leases import GpuLease

ROOT=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def save(manifest,status):
    status['updated']=time.time();write(ROOT/'status.json',status)
    write(ROOT/'registry.json',{j['signature']:dict(name=j['name'],**status['jobs'][j['name']])
                                for j in manifest['jobs']})
    lines=['# B 输入度量实验','',
           '这组实验在原四卡队列之外使用空闲 GPU5、7，延续 T50、rank10、20epoch、单卡batch128、HOEP5%、LS开、NC关、PT开。复用同协议 Frozen-A 和当前HOEP对照，不重跑基线。','',
           '## 假设与方法','',
           '- B-current：从训练批次 z=Ax 计算 Σnew=E[zzᵀ]，对当前B梯度右乘带阻尼的输入度量逆。',
           '- B-history：度量同时包含 Σnew 和 A C_old Aᵀ，两者各归一化迹、权重相等；C_old 来自各任务当时的训练输入统计，不重新读取旧图像。',
           '- 阻尼ρ=0.1；每层Q/V独立计算10×10度量。变换后恢复当前梯度Frobenius范数，以检查方向变化的作用。每任务首批安装统计钩子并保持原更新，之后逐批重算。Task0保持原训练。',
           '- SGD动量仍可改变实际参数步幅；瞬时梯度范数相同不代表参数步幅或最终有效算子范数相同。部署仍是原固定大小A/G，无新增适配器。history模式额外训练统计54 MiB。','',
           '设计参考：'+', '.join(f'[原论文]({url})' for url in manifest['references'])+'。低秩优化路径有文献依据；这里是独立的数据统计驱动变体，论文没有验证本实现或它在持续学习上的收益。不能声称一般可逆因子变换不变性；数学测试只验证正交坐标变换等变性。','',
           '## 状态与结果','',f"更新时间 {time.strftime('%F %T %Z')}；状态 `{status['state']}`。",'',
           '| 阶段 | 数据集/seed | 方法 | 状态 | GPU | 完成任务 | Final | AAA | 对Frozen差值(pp) | 对当前HOEP差值(pp) |','|---|---|---|---|---|---|---|---|---|---|']
    for job in manifest['jobs']:
        record=status['jobs'][job['name']];met=record.get('metrics',{});score=met.get('final_top1')
        gaps=[]
        for ref in ['matched_frozen','matched_hoep']:
            path=Path(job[ref])/'metrics.json'
            baseline=read(path) if path.exists() else {}
            if job['phase']=='formal' and score is not None and baseline.get('tasks')==50:
                assert baseline['dataset']==job['dataset'] and baseline['seed']==job['seed']
                assert baseline['batch_size']==128 and baseline['world_size']==1 and baseline['transport'] and not baseline['normcap']
                gaps.append(f"{score-baseline['final_top1']:+.4f}")
            else:gaps.append('—')
        lines.append('| '+' | '.join([job['phase'],f"{job['dataset']}/{job['seed']}",job['kind'],record['state'],str(job['gpu']),
            str(record.get('completed_tasks',0)),'—' if score is None else f'{score:.4f}',
            '—' if not met else f"{met['average_accuracy']:.4f}",*gaps])+' |')
    lines+=['','## 异常与验收','']
    for name,record in status['jobs'].items():
        if record.get('error'):lines.append(f"- `{name}`: {record['error'].replace(chr(10),' ')[:1400]}")
    lines+=['','先做三任务短测：Task0为20epoch，其余2epoch。短测必须与主队列Task0模型及RNG一致，且后续任务实际执行预条件、有限梯度、相对范数误差小于1e-5。通过后自动运行四项完整T50。短测准确率不用于选择系数。','',
            '每项使用源码与有效配置signature去重；拒绝覆盖输出，保留所有结果与失败。完整任务结束后核验50个边界快照、模型、原型和原始曲线。是否达标仍需完整同协议配对及其他预定seed确认。','']
    p=ROOT/'EXPERIMENTS.md';tmp=p.with_suffix('.md.tmp');tmp.write_text('\n'.join(lines));tmp.replace(p)


def verify(manifest):
    for name,digest in manifest['file_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    for ds in ['cub','imagenet-r']:
        assert (ROOT/'source/data'/ds/'train').is_dir() and (ROOT/'source/data'/ds/'test').is_dir()


def gpu_free(gpu):
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    return {int(s.split(',')[0]):int(s.split(',')[1]) for s in text.splitlines()}[gpu]<300


def smoke_audit(jobs):
    proof={}
    keys=['post_train_model_hash','post_train_rng_hash','post_eval_model_hash','post_eval_rng_hash']
    for job in jobs:
        current=read(Path(job['output'])/'p0_hashes.json')['0']
        reference=read(Path(job['task0_reference'])/'p0_hashes.json')['0']
        for key in keys:assert current[key]==reference[key],(job['name'],key,'Task0 differs')
        proof[job['name']]={'common_task0_exact':True}
    write(ROOT/'proofs/smoke_task0_equality.json',proof)


def run_phase(phase,manifest,status):
    jobs=[job for job in manifest['jobs'] if job['phase']==phase]
    pending=[];running={};failures=[]
    for job in jobs:
        record=status['jobs'][job['name']]
        if record['state']=='complete':continue
        assert record['state']=='pending' and not Path(job['output']).exists(),('audit before restart',job['name'])
        pending.append(job)
    status['state']=phase;save(manifest,status)
    while pending or running:
        for name,item in list(running.items()):
            job=item['job'];record=status['jobs'][name]
            record['completed_tasks']=len(list((Path(job['output'])/'task_snapshots').glob('task_*/complete.json')))
            code=item['process'].poll()
            if code is None:
                if re.search(r'Loss\s+(?:nan|[+-]?inf)(?:[,\s]|$)',Path(job['log']).read_text(errors='replace'),re.I):item['process'].terminate()
                continue
            item['stream'].close();item['lease'].close();record.update(returncode=code,ended=time.time())
            if code==0:
                record.update(state='complete',metrics=read(Path(job['output'])/'metrics.json'))
            else:
                record.update(state='failed',error=Path(job['log']).read_text(errors='replace')[-2200:]);failures.append(name)
            del running[name];print('EXIT',name,code,flush=True)
        busy={item['job']['gpu'] for item in running.values()}
        for job in list(pending):
            gpu=job['gpu']
            if gpu in busy:continue
            lease=GpuLease.acquire([gpu])
            if lease is None:continue
            if not gpu_free(gpu):lease.close();continue
            stream=Path(job['log']).open('x')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',CUBLAS_WORKSPACE_CONFIG=':4096:8',
                     HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            process=subprocess.Popen([sys.executable,'-u',str(ROOT/'run_job.py'),'--config',job['config']],
                                     cwd=ROOT/'source',env=env,stdout=stream,stderr=subprocess.STDOUT,pass_fds=lease.fds)
            ticks=Path(f'/proc/{process.pid}/stat').read_text().split()[21]
            status['jobs'][job['name']].update(state='running',pid=process.pid,proc_start_ticks=ticks,
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),gpu=gpu,started=time.time())
            running[job['name']]=dict(job=job,process=process,stream=stream,lease=lease)
            busy.add(gpu);pending.remove(job);print('START',job['name'],gpu,process.pid,flush=True)
        save(manifest,status)
        if pending or running:time.sleep(20)
    assert not failures,failures
    if phase=='smoke':smoke_audit(jobs)
    verify(manifest)


def main():
    lock=(ROOT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=read(ROOT/'manifest.json');verify(manifest)
    if (ROOT/'status.json').exists():status=read(ROOT/'status.json')
    else:
        assert len({j['signature'] for j in manifest['jobs']})==len(manifest['jobs'])
        status=dict(state='prepared',started=time.time(),jobs={j['name']:dict(state='pending',completed_tasks=0) for j in manifest['jobs']})
    try:
        for phase in ['smoke','formal']:run_phase(phase,manifest,status)
        status.update(state='complete',finished=time.time());save(manifest,status)
    except BaseException:
        status.update(state='failed',error=traceback.format_exc());save(manifest,status);raise


if __name__=='__main__':main()
