"""Render every planned allocation attempt and its actual results."""
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent


def write_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def render(manifest,status):
    lines=['# 分支覆盖与A步幅实验','',
        '原全局HOEP与既定A倍率10作为已存在对照；本组只新增每分支最低完整本征空间覆盖后仍受全局5%能量上限约束的规则，并分别使用A倍率1或10。不能放入最低组时回退原全局规则。两个方案都不增加部署参数。','',
        '固定CUB seed1、ImageNet-R seed1995、T50、rank10、20epoch、单卡batch128、LS/Transport开、NormCap关。只改变分配规则与预定A倍率；分类头、原型构造和Transport算法保持原实现。','',
        '先做8项三任务检查：每数据集global/coverage×倍率1/10，Task0为20epoch、后两任务为2epoch。global的全部三任务模型/RNG应匹配既有同倍率检查；coverage应保持Task0、真实改变过分配和可塑A，同时保护未选行。短测准确率不用于筛选。','',
        '共享健康GPU租约；GPU0的一次预约只等待当前所有者自然释放。其他队列和模型继续运行。','',
        f"更新时间：{time.strftime('%F %T %Z')}；状态 `{status['state']}`。",'',
        '| 阶段 | 数据集/seed | 方法 | 状态 | GPU | 完成任务 | Final | AAA | 对Frozen差值(pp) | 对同倍率global差值(pp) |',
        '|---|---|---|---|---|---|---|---|---|---|']
    for job in manifest['jobs']:
        rec=status['jobs'][job['name']];met=rec.get('metrics',{});score=met.get('final_top1');gaps=[]
        for key in ['matched_frozen','matched_rate']:
            path=Path(job[key])/'metrics.json';base=json.loads(path.read_text()) if path.exists() else {}
            if job['phase']=='formal' and score is not None and base.get('tasks')==50:
                assert base['dataset']==job['dataset'] and base['seed']==job['seed']
                gaps.append(f"{score-base['final_top1']:+.4f}")
            else:gaps.append('—')
        lines.append('| '+' | '.join([job['phase'],f"{job['dataset']}/{job['seed']}",job['arm'],
            rec['state'],str(rec.get('gpu','—')),str(rec.get('completed_tasks',0)),
            '—' if score is None else f'{score:.4f}',
            '—' if not met else f"{met['average_accuracy']:.4f}",*gaps])+' |')
    lines+=['','## 异常与验收','']
    for name,rec in status['jobs'].items():
        if rec.get('error'):lines.append(f"- `{name}`: {rec['error'].replace(chr(10),' ')[:1400]}")
    lines+=['','完整结果需核验50段曲线、所有快照、实际分配、全局能量上限和未选A行逐元素不变；相同Task0不代表性能达标。预生成的其他seed配置不算已执行训练。所有失败和不利结果保留。','']
    path=ROOT/'EXPERIMENTS.md';tmp=path.with_suffix('.md.tmp');tmp.write_text('\n'.join(lines));tmp.replace(path)
