"""Merge the two isolated queues into one live Markdown research record."""
import json
from pathlib import Path
import shlex
import subprocess
import time

MAIN=Path(__file__).resolve().parent
EXTRA=MAIN.parent/'hoep-bmetric-t50-cuda8-20261023'
HOST='hongzhijun@192.168.10.208'


def demote(text):
    return '\n'.join('#'+line if line.startswith('#') else line for line in text.splitlines())


def mirror_extra():
    command=['rsync','-a','--timeout=25','-e','ssh -o BatchMode=yes -o ConnectTimeout=10',
             '--include=/EXPERIMENTS.md','--include=/status.json','--include=/registry.json',
             '--include=/records/***','--include=/proofs/***','--include=/outputs/',
             '--include=/outputs/*/','--include=/outputs/*/metrics.json','--include=/outputs/*/b_metric/',
             '--include=/outputs/*/b_metric/*.json','--exclude=*',HOST+':'+str(EXTRA)+'/',str(EXTRA)+'/']
    subprocess.run(command,check=True,timeout=50)


def live_progress():
    code='import sys; sys.path.insert(0,'+repr(str(MAIN))+'); import interim_audit,json; print(json.dumps(interim_audit.collect()))'
    output=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,
                                     'python3 -c '+shlex.quote(code)],text=True,timeout=25)
    return json.loads(output)


def render():
    payload=live_progress();progress=payload['progress']
    lines=['# HOEP T50 统一实验记录','',
           '目标：改进 LoRA 适配，在 CUB-200 和 ImageNet-R 的 T50 上，相对同协议 Frozen-A 的 Final Top-1 提升超过0.5个百分点。这里汇总所有已启动假设、失败、参数和最终结果；完整配对结果缺失时不宣称达标。','',
           f"镜像更新时间：{time.strftime('%F %T %Z')}。训练权威数据位于 cuda8 同名目录；原四卡队列与补充两卡队列使用共享GPU锁，部署参数量保持固定。",'',
           '## 当前进程与真实快照进度','',
           '| 实验 | GPU | PID | 进程状态 | 身份匹配 | 已完成任务 |','|---|---|---|---|---|---|']
    for item in progress:
        lines.append('| '+' | '.join(str(item[k]) for k in ['job','gpu','pid','process_state','identity_matches','completed_tasks'])+' |')
    lines+=['','进程状态R/S表示当前进程可运行/等待；Z或gone可能是等待调度器回收的已结束进程，需要结合最终状态和metrics判断。进度直接数完整任务快照，不用旧状态文件推断训练停止。','',
            '## 相同任务边界的中途比较','',
            '各行只比较同数据集、同seed、相同已完成任务数和类别集合。未到50任务的数值用于观察轨迹，不作为T50结果或停止/选择候选的依据。Frozen差值包含原HOEP和新增机制的整体影响；新增模块贡献还需与本轮HOEP基线比较。','',
            '| 数据集/seed | 方法 | 共同完成任务 | 已见类别 | Frozen Top-1 | 候选Top-1 | 差值(pp) |',
            '|---|---|---|---|---|---|---|']
    for row in payload['paired_interim']:
        lines.append('| '+' | '.join([f"{row['dataset']}/{row['seed']}",row['method'],str(row['completed_tasks']),
            str(row['seen_classes']),f"{row['frozen_top1']:.2f}",f"{row['candidate_top1']:.2f}",f"{row['difference_pp']:+.2f}"])+' |')
    lines+=['','### 旧类与当前任务新类','',
            '同样使用上表的共同完成边界。两类均下降时，不能将其概括为以新类适应换取旧类遗忘；这些仍是方法整体比较。首任务没有旧类，显示为—。','',
            '| 数据集/seed | 方法 | 共同任务 | Frozen旧类 | 候选旧类 | Frozen新类 | 候选新类 |',
            '|---|---|---|---|---|---|---|']
    for row in payload['paired_interim']:
        values=['—' if row.get(k) is None else f'{row[k]:.2f}' for k in
                ['frozen_old_top1','candidate_old_top1','frozen_new_top1','candidate_new_top1']]
        lines.append('| '+' | '.join([f"{row['dataset']}/{row['seed']}",row['method'],str(row['completed_tasks']),*values])+' |')
    lines+=['',
            '## 第一组：历史响应与原型训练目标','',demote((MAIN/'EXPERIMENTS.md').read_text()),'',
            '## 第二组：当前B的输入度量','',demote((EXTRA/'EXPERIMENTS.md').read_text()),'']
    path=MAIN/'RESEARCH_LOG.md';temporary=path.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines));temporary.replace(path)
    subprocess.run(['rsync','-a','--timeout=25',str(path),HOST+':'+str(path)],check=True,timeout=35)


def main():
    while True:
        try:
            mirror_extra();render()
            states=[json.loads((root/'status.json').read_text())['state'] for root in (MAIN,EXTRA)]
            print(time.strftime('%F %T'),states,flush=True)
            if all(state in ('complete','failed') for state in states):return
        except (OSError,ValueError,subprocess.SubprocessError) as error:
            print('RETRY',type(error).__name__,str(error),flush=True)
        time.sleep(30)


if __name__=='__main__':main()
