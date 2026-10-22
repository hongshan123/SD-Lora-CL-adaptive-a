"""Incremental descriptive summary; boundaries are not independent seed replicates."""
import csv
import json
import statistics
from pathlib import Path


def summarize(directory):
    directory=Path(directory)
    rows=[]
    transitions=[('S0','S1'),('S1','S2'),('S2','S3'),('S3','S4'),('S4','S5'),('S5','S6'),('S6','formal_new_prototypes'),('formal_new_prototypes','offline_refreshed_old_prototypes')]
    for path in sorted(directory.glob('task_*.json')):
        r=json.loads(path.read_text()); m=r['metrics']
        row={'task':r['task'],'known_classes':r['known_classes'],**r['aggregate']}
        for name,metric in m.items():
            row[name+'_old']=metric['old']['top1'];row[name+'_new']=metric['new']['top1'];row[name+'_global']=metric['global_top1']
        for a,b in transitions:row[a+'_to_'+b+'_old']=m[b]['old']['top1']-m[a]['old']['top1']
        rows.append(row)
    if not rows:return
    with (directory/'boundaries.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# HOEP T=50 无 NormCap 单次训练阶段诊断','',f'已完成 {len(rows)}/49 个历史边界。单 seed 描述性结果，不构成跨数据集结论。','',
           'S0 上一任务部署；S1 当前 A 与实际历史 G、关闭 B；S2 训练结束未合并；S3 历史 LS 后保留 B；S4 精确吸收；S5 实际部署（本实验与 S4 等价）；S6 仅替换 transport 后旧 prototype；随后单独重算正式新 prototype。', '',
           '固定分类头由 transport 前旧 prototype 与 S2 当前任务训练集临时新 prototype 构成。旧样本用于离线诊断，不参与训练或算法选择。', '',
           '| 阶段 | 旧类 Top-1 平均变化（百分点） |','|---|---:|']
    for a,b in transitions:
        key=a+'_to_'+b+'_old';lines.append(f'| {a} → {b} | {statistics.mean(r[key] for r in rows):+.4f} |')
    lines+=['','注意：offline_refreshed_old_prototypes 使用全部旧训练类重新计算原型，只用于检查分类头不匹配，不能作为无回放正式成绩。A/B 共同训练，关闭 B 是固定状态反事实。','',f"平均归一化 LS 不可恢复残差：{statistics.mean(r['r_rec'] for r in rows):.6e}。",f"平均历史输入加权残差：{statistics.mean(r['r_weighted'] for r in rows):.6e}。",'', '逐边界数据见 boundaries.csv，逐样本 logits、样本顺序和原型见 task_XXX.pt，重建验证和 Q/V 诊断见 task_XXX.json。']
    (directory/'summary.md').write_text('\n'.join(lines)+'\n')
