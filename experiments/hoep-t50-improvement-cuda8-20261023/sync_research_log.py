"""Merge isolated queues into one live Markdown research record."""
import json
from pathlib import Path
import shlex
import subprocess
import time

MAIN=Path(__file__).resolve().parent
EXTRA=MAIN.parent/'hoep-bmetric-t50-cuda8-20261023'
A_STEP=MAIN.parent/'hoep-a-step-t50-cuda8-20261023'
ALLOCATION=MAIN.parent/'hoep-budget-allocation-t50-cuda8-20261023'
CONTROLS=MAIN.parent/'hoep-t50-confirmation-controls-cuda8-20261023'
AVERAGE=MAIN.parent/'hoep-operator-average-t50-cuda8-20261023'
HOST='hongzhijun@192.168.10.208'


def demote(text):
    return '\n'.join('#'+line if line.startswith('#') else line for line in text.splitlines())


def mirror_extra():
    command=['rsync','-a','--timeout=25','-e','ssh -o BatchMode=yes -o ConnectTimeout=10',
             '--include=/EXPERIMENTS.md','--include=/status.json','--include=/registry.json',
             '--include=/records/***','--include=/proofs/***','--include=/outputs/',
             '--include=/outputs/*/','--include=/outputs/*/metrics.json','--include=/outputs/*/b_metric/',
             '--include=/outputs/*/b_metric/*.json','--include=/outputs/*/a_gain/',
             '--include=/outputs/*/a_gain/*.json','--include=/outputs/*/adaptation_state/',
             '--include=/outputs/*/adaptation_state/*.json','--include=/outputs/*/branch_average/',
             '--include=/outputs/*/branch_average/*.json','--exclude=*']
    for root in (MAIN,EXTRA,A_STEP,ALLOCATION,CONTROLS,AVERAGE):
        if not (root/'manifest.json').exists():continue
        subprocess.run(command+[HOST+':'+str(root)+'/',str(root)+'/'],check=True,timeout=50)


def live_progress():
    code='import sys; sys.path.insert(0,'+repr(str(MAIN))+'); import interim_audit,json; print(json.dumps(interim_audit.collect()))'
    output=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,
                                     'python3 -c '+shlex.quote(code)],text=True,timeout=25)
    return json.loads(output)


def component_text(root,payload):
    """Refresh displayed counters without writing the training coordinator state."""
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text())
    names={(f"{j['dataset']}/{j['seed']}",j.get('arm',j.get('kind'))):j['name']
           for j in manifest['jobs'] if j['phase']=='formal'}
    progress={row['job']:row for row in payload['progress']}
    lines=[]
    for line in (root/'EXPERIMENTS.md').read_text().splitlines():
        cells=[cell.strip() for cell in line.strip().strip('|').split('|')]
        if line.startswith('| formal |') and len(cells)==10:
            live=progress.get(names.get((cells[1],cells[2])))
            if live is not None:
                cells[3]='running'
                cells[4]=str(live['gpu'])
                cells[5]=str(live['completed_tasks'])
                line='| '+' | '.join(cells)+' |'
        if line.startswith('| 阶段 |'):
            line=line.replace('完成任务','边界快照数')
        line=line.replace('更新时间：','调度器最近写入时间：')
        lines.append(line)
    return '\n'.join(lines)


def render():
    payload=live_progress();progress=payload['progress']
    lines=['# HOEP T50 统一实验记录','',
           '目标：改进 LoRA 适配，在 CUB-200 和 ImageNet-R 的 T50 上，相对同协议 Frozen-A 的 Final Top-1 提升超过0.5个百分点。这里汇总所有已启动假设、失败、参数和最终结果；完整配对结果缺失时不宣称达标。','',
           f"镜像更新时间：{time.strftime('%F %T %Z')}。训练权威数据位于 cuda8 同名目录；各组队列共享GPU锁，新任务等待健康显卡空闲，部署适配器参数形状保持固定。",'',
           '## 当前进程与真实快照进度','',
           '| 实验 | GPU | PID | 进程状态 | 身份匹配 | 完整边界快照数 |','|---|---|---|---|---|---|']
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
    audit_path=MAIN/'records/formal_boundary_audit.json'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text())
        lines+=['','## 正式训练对照一致性核验','',
                '核验时间：'+time.strftime('%F %T %Z',time.localtime(audit['updated']))+'。独立CPU进程核验Frozen-A全部已完成快照中的24个A，并核对各方案的Task0模型/RNG哈希。候选后续快照的最终完整性另行审计；本次检查不能代替完整T50结果审计。','',
                '| 数据集/seed | 方法 | 当时完整边界快照数 | Task0模型与RNG精确一致 | Frozen A逐元素不变 |',
                '|---|---|---|---|---|']
        for row in audit['rows']:
            frozen='是' if row.get('frozen_a_bitwise_unchanged') else '不适用'
            lines.append('| '+' | '.join([f"{row['dataset']}/{row['seed']}",row['method'],
                str(row['completed_snapshot_count']),'是' if row['common_formal_task0_exact'] else '否',frozen])+' |')
    notes_path=MAIN/'records/ANALYSIS_NOTES.md'
    if notes_path.exists():
        lines+=['','## 方法差异与诊断备注','',demote(notes_path.read_text()),'']
    lines+=['',
            '## 第一组：历史响应与原型训练目标','',demote(component_text(MAIN,payload)),'',
            '## 第二组：当前B的输入度量','',demote(component_text(EXTRA,payload)),'']
    if (A_STEP/'EXPERIMENTS.md').exists():
        lines+=['','## 第三组：A步幅倍率','',demote(component_text(A_STEP,payload)),'']
    if (ALLOCATION/'EXPERIMENTS.md').exists():
        lines+=['','## 第四组：分支覆盖与A步幅','',demote(component_text(ALLOCATION,payload)),'']
    if (CONTROLS/'EXPERIMENTS.md').exists():
        lines+=['','## 第五组：追加seed的Frozen配对对照','',demote(component_text(CONTROLS,payload)),'']
    if (AVERAGE/'EXPERIMENTS.md').exists():
        lines+=['','## 第六组：当前分支的后段算子平均','',demote(component_text(AVERAGE,payload)),'']
    path=MAIN/'RESEARCH_LOG.md';temporary=path.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines));temporary.replace(path)
    subprocess.run(['rsync','-a','--timeout=25',str(path),HOST+':'+str(path)],check=True,timeout=35)


def main():
    while True:
        try:
            mirror_extra();render()
            roots=[MAIN,EXTRA]+([A_STEP] if (A_STEP/'manifest.json').exists() else [])
            if (ALLOCATION/'manifest.json').exists():roots.append(ALLOCATION)
            if (CONTROLS/'manifest.json').exists():roots.append(CONTROLS)
            if (AVERAGE/'manifest.json').exists():roots.append(AVERAGE)
            states=[json.loads((root/'status.json').read_text())['state'] if (root/'status.json').exists()
                    else 'prepared' for root in roots]
            print(time.strftime('%F %T'),states,flush=True)
            if all(state in ('complete','failed','complete_with_invalid_attempts') for state in states):return
        except (OSError,ValueError,subprocess.SubprocessError) as error:
            print('RETRY',type(error).__name__,str(error),flush=True)
        time.sleep(30)


if __name__=='__main__':main()
