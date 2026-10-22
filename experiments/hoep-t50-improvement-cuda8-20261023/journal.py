"""One persistent experiment log; failed hypotheses remain visible."""
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent


def write_json(path,record):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def render(manifest,status,root=ROOT):
    root=Path(root)
    lines=['# HOEP T50 改进实验记录','',
           '目标：在 cuda8 对 CUB-200 和 ImageNet-R 完成 T50 改进；Final Top-1 相对同协议 Frozen-A 超过 0.5 个百分点。优先修改 LoRA 适配，记录全部尝试和失败，避免重复实验。','',
           '## 固定协议','',
           '- rank10、每任务20 epoch、单卡 batch128、SGD 原学习率；HOEP 能量预算5%、历史 LS 开、NormCap 关、Coordinate Transport 开。每项独占一张 GPU，调度范围以生成的 manifest 为准。',
           '- 首轮 CUB seed1、ImageNet-R seed1995。候选达标后在 CUB seeds1–3、ImageNet-R seeds1995–1997 做配对确认；各 seed 均报告，主目标按两数据集各自平均差值评估。',
           '- Frozen-A 与候选共享分类器、Transport、训练集预处理、任务划分和训练时长，仅冻结 Task0 后的 A。Frozen 快照检查 A 逐元素不变。',
           '- 完整训练20 epoch的旧双卡轨迹仅作背景，不复用为本轮单卡的配对对照。测试集只做评估；原型和历史输入统计仅来自任务当时可用的训练集。','',
           '## 不重复规则','']
    lines+=[
            '每项由训练源码哈希、有效配置和执行协议生成 signature；相同 signature 的完成项复用，已有输出拒绝覆盖。短测与正式实验明确区分，短测不用于挑选测试集最高配置。','',
            '## 本轮假设','',
            '| 方法 | 改动 | 训练期额外状态 |','|---|---|---|',
            '| frozen | Task0 后冻结共享A；固定形状G吸收B | 原协议 |',
            '| hoep | 当前HOEP，作为同协议训练基线 | 原协议 |',
            '| history | HOEP + 总有效算子历史输入加权约束，λ=0.001 | 12层完整输入二阶矩，float64共54 MiB，不随任务增加 |',
            '| prototype | HOEP + 全已见类原型CE，λ=0.05、温度0.1、EMA0.1 | 训练期原型副本；梯度更新LoRA |',
            '| combined | history + prototype，系数保持一致 | 上述两者 |','',
            'Task0 不启用新增损失。短测采用 Task0 20 epoch、后续2 epoch、3个任务，验证有限梯度、真实损失启用、快照及模型/RNG一致性；正式训练必须通过短测。','',
            '## 运行与结果','',
            f"更新时间：{time.strftime('%Y-%m-%d %H:%M:%S %Z')}；队列状态：`{status['state']}`。",'',
            '| 阶段 | 数据集/seed | 方法 | 状态 | GPU | 完成任务 | Final | AAA | 对Frozen差值(pp) | signature |',
            '|---|---|---|---|---|---|---|---|---|---|']
    frozen={}
    for job in manifest['jobs']:
        record=status['jobs'][job['name']]
        if job['arm']=='frozen' and record['state']=='complete':
            frozen[(job['phase'],job['dataset'],job['seed'])]=record['metrics']['final_top1']
    for job in manifest['jobs']:
        record=status['jobs'][job['name']];metrics=record.get('metrics',{})
        score=metrics.get('final_top1');reference=frozen.get((job['phase'],job['dataset'],job['seed']))
        gap='—' if score is None or reference is None else f'{score-reference:+.4f}'
        lines.append('| '+' | '.join([job['phase'],f"{job['dataset']}/{job['seed']}",job['arm'],record['state'],
            str(record.get('gpu','—')),str(record.get('completed_tasks',0)),
            '—' if score is None else f'{score:.4f}',
            '—' if 'average_accuracy' not in metrics else f"{metrics['average_accuracy']:.4f}",gap,job['signature'][:12]])+' |')
    lines+=['','## 失败及修复记录','']
    for job in manifest['jobs']:
        record=status['jobs'][job['name']]
        if record.get('error'):lines += [f"- `{job['name']}`：{record['error'].replace(chr(10),' ')[:1600]}"]
    lines+=['','## 结论规则','',
            '只有完成50个任务、原始日志和逐任务快照核验通过，才填写正式 Final/AAA。首轮优势属于单seed结果；跨seed确认前不宣称稳定收益。性能未达到目标时保留失败记录，依据历史输入响应、新旧类指标和原型兼容性继续改进，不将中途准确率作为最终成绩。','',
            f"训练源码 SHA256：`{manifest['training_source_sha256']}`。完整配置见 configs/；逐任务状态、原型、输入统计及探针保存在各 outputs/ 子目录。",'']
    path=root/'EXPERIMENTS.md';tmp=path.with_suffix('.md.tmp');tmp.write_text('\n'.join(lines));tmp.replace(path)


if __name__=='__main__':
    render(json.loads((ROOT/'manifest.json').read_text()),json.loads((ROOT/'status.json').read_text()))
