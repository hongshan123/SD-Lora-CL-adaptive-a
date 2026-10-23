"""Persistent results for the current-branch averaging hypothesis."""
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent


def write_json(path,record):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def render(manifest,status):
    lines=['# 当前任务LoRA分支的后段算子平均','',
           '假设：用最后25%优化更新中的当前sBA算子均值降低写入噪声；最后A和历史G保持，均值投影到最后A后按原历史LS及精确吸收写入。投影残差只诊断线性算子损失，不是Accuracy界。',
           'Task0关闭。T50/rank10/e20/单卡batch128/NCoff/LSon/PTon/global HOEP5%；首seed CUB1、INR1995，预定确认seed不变。',
           '逐步保存平均窗口内的A/B/scale，正式分数须经过独立重算与快照审计。四个三任务数值/一致性检查通过后才运行两份完整训练，不按短测准确率筛选。', '',
           f'更新时间：{time.strftime("%F %T %Z")}；队列状态：`{status["state"]}`。','',
           '| 阶段 | 数据集/seed | 方法 | 状态 | GPU | 边界快照数 | Final | AAA | 对Frozen差值(pp) | signature |',
           '|---|---|---|---|---|---|---|---|---|---|']
    for job in manifest['jobs']:
        record=status['jobs'][job['name']];metric=record.get('metrics',{});score=metric.get('final_top1')
        reference=Path(job['matched_frozen'])/'metrics.json'
        frozen=json.loads(reference.read_text())['final_top1'] if reference.exists() else None
        gap='—' if job['phase']!='formal' or score is None or frozen is None else f'{score-frozen:+.4f}'
        values=[job['phase'],f'{job["dataset"]}/{job["seed"]}',job['arm'],record['state'],
                str(record.get('gpu','—')),str(record.get('completed_tasks',0)),
                '—' if score is None else f'{score:.4f}',
                '—' if 'average_accuracy' not in metric else f'{metric["average_accuracy"]:.4f}',gap,job['signature'][:12]]
        lines.append('| '+' | '.join(values)+' |')
    for name,record in status['jobs'].items():
        if record.get('error'):lines.extend(['',f'失败 `{name}`：'+record['error'].replace('\n',' ')[:2000]])
    path=ROOT/'EXPERIMENTS.md';temporary=path.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines)+'\n');temporary.replace(path)
