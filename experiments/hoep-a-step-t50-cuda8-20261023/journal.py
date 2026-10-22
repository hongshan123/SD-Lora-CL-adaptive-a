"""Render A-factor step ablations without discarding invalid attempts."""
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def render(manifest, status):
    lines = ['# A步幅倍率实验', '',
             '基于已保存边界的CPU诊断追加两个预定倍率：Task1以后A的HOEP投影梯度放大10倍或100倍；B、分类头、缩放系数保持原SGD设置。Task0保持完全相同。', '',
             '固定T50、rank10、每任务20epoch、单卡batch128、HOEP5%、LS开、NormCap关、Transport开。首轮CUB seed1、ImageNet-R seed1995；复用主队列配对Frozen和HOEP，不重复训练正式基线。', '',
             '先做六项三任务短测：每数据集倍率1、10、100；Task0为20epoch，其余为2epoch。倍率1要求三个任务的模型及RNG与原HOEP精确一致；所有倍率Task0精确一致。倍率10/100各自必须在两个数据集上通过有限性与实际启用检查，才开放各自完整训练；纯数值失败不会阻止另一有效倍率。短测准确率不用于挑选倍率。', '',
             '共享0、1、3、4、5、7号卡GPU锁，等待显卡空闲后启动。独立任务不共占同一卡，不停止既有任务。未使用不稳定的2号卡，也未占用仍有基线任务的6号卡。', '',
             'GPU3有一次阻塞式租约等待，在当前所有者释放后才接续，用于避免新短测队列被旧队列立即续跑挤到最后；首次接续后恢复普通非阻塞调度。训练源码、科学配置和输入顺序保持原冻结版本。', '',
             '倍率属于因子步幅消融。'+
             '[LoRA+原论文](https://proceedings.mlr.press/v235/hayou24a.html)讨论不同因子的不同学习率，但其ADAM/宽度缩放分析不证明当前HOEP/SGD下A倍率10/100最优，也不验证本实现的持续学习性能。这里的倍率来自本轮独立诊断假设。', '',
             '梯度变换为k*(g+wd*A)-wd*A，其中g已经包含HOEP的径向衰减抵消项；避免错误地将抵消项也扩大。对当前SGD及其动量，它等价于仅增大A的切向步幅。没有额外部署参数。', '',
             f"更新时间 {time.strftime('%F %T %Z')}；状态 `{status['state']}`。", '',
             '| 阶段 | 数据集/seed | 方法 | 状态 | GPU | 完成任务 | Final | AAA | 对Frozen差值(pp) | 对HOEP差值(pp) |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for job in manifest['jobs']:
        rec = status['jobs'][job['name']]
        met = rec.get('metrics', {})
        score = met.get('final_top1')
        gaps = []
        for key in ('matched_frozen', 'matched_hoep'):
            path = Path(job[key]) / 'metrics.json'
            baseline = json.loads(path.read_text()) if path.exists() else {}
            if job['phase'] == 'formal' and score is not None and baseline.get('tasks') == 50:
                assert baseline['dataset'] == job['dataset'] and baseline['seed'] == job['seed']
                gaps.append(f"{score-baseline['final_top1']:+.4f}")
            else:
                gaps.append('—')
        lines.append('| ' + ' | '.join([job['phase'], f"{job['dataset']}/{job['seed']}",
            job['arm'], rec['state'], str(rec.get('gpu', '—')), str(rec.get('completed_tasks', 0)),
            '—' if score is None else f'{score:.4f}',
            '—' if not met else f"{met['average_accuracy']:.4f}", *gaps]) + ' |')
    lines += ['', '## 异常与验收', '']
    for name, rec in status['jobs'].items():
        if rec.get('error'):
            lines.append(f"- `{name}`: {rec['error'].replace(chr(10),' ')[:1400]}")
    lines += ['', '所有完整结果均核对50个快照、原始曲线、A倍率真实启用和同协议对照；三任务短测不能作为T50证据。全部尝试和不利结果保留。其他预定seed确认前不宣称稳定收益。', '']
    path = ROOT / 'EXPERIMENTS.md'
    temp = path.with_suffix('.md.tmp')
    temp.write_text('\n'.join(lines))
    temp.replace(path)
