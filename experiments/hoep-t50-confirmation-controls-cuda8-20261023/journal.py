"""Render additional-seed controls without changing their training state."""
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent


def write_json(path, record):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def render(manifest, status):
    lines = ['# T50 三 seed Frozen 对照补全', '',
             '复用既定追加 seed 输入：CUB 2、3；ImageNet-R 1996、1997。',
             '训练源码和 worker 与首 seed Frozen 完全相同；只改变 seed 与输出标识。',
             'T50、rank10、20epoch/task、单卡物理batch128、NC关、LS/PT开；不按分数筛选seed。',
             '使用已经空闲的cuda8 GPU6，共享显卡锁，等待已有占用自然结束，不抢占。', '',
             f'更新时间：{time.strftime("%F %T %Z")}；队列状态：`{status["state"]}`。', '',
             '| 阶段 | 数据集/seed | 方法 | 状态 | GPU | 边界快照数 | Final | AAA | 对Frozen差值(pp) | signature |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for job in manifest['jobs']:
        record = status['jobs'][job['name']]
        metric = record.get('metrics', {})
        final = metric.get('final_top1');aaa = metric.get('average_accuracy')
        values = ['formal', f'{job["dataset"]}/{job["seed"]}', 'frozen', record['state'],
                  str(record.get('gpu', '—')), str(record.get('completed_tasks', 0)),
                  '—' if final is None else f'{final:.4f}',
                  '—' if aaa is None else f'{aaa:.4f}', '—', job['signature'][:12]]
        lines.append('| ' + ' | '.join(values) + ' |')
    for name, record in status['jobs'].items():
        if record.get('error'):
            lines.extend(['', f'失败 `{name}`：' + record['error'].replace('\n', ' ')[:2000]])
    target = ROOT / 'EXPERIMENTS.md';temporary = target.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines) + '\n');temporary.replace(target)
