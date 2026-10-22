"""CPU-only, post-deployment associations from complete matched T50 runs.

No images, prototype changes or training mutations. These associations do not
isolate training, merge, transport or class-competition effects.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import time

import torch

from operator_boundary_audit import analyze, sha
from run_job import NUMBER, parse_curve

ROOT = Path(__file__).resolve().parent
PREDICTORS = {
    'frobenius_projection_residual_ratio': 'Frobenius R_rec squared ratio',
    'weighted_frobenius_projection_residual_ratio': 'Own-history weighted R_rec squared ratio',
    'a_subspace_rms_sine': 'A row-space principal-sine RMS',
    'current_b_relative_norm': 'Current B relative operator norm',
    'ls_relative_operator_change': 'LS relative operator change',
}
OUTCOMES = {'old_gap_pp': .01, 'excess_old_drop_pp': .02}


def read(path):
    return json.loads(Path(path).read_text())


def ranks(values):
    ordered = sorted(range(len(values)), key=values.__getitem__)
    result = [0.] * len(values)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + end - 1) / 2. + 1.
        for index in ordered[start:end]:
            result[index] = rank
        start = end
    return result


def pearson(x, y):
    if len(x) != len(y) or len(x) < 2:
        return None
    dx = [v - sum(x) / len(x) for v in x]
    dy = [v - sum(y) / len(y) for v in y]
    denominator = math.sqrt(sum(v*v for v in dx) * sum(v*v for v in dy))
    if denominator <= 1e-25:
        return None
    return max(-1., min(1., sum(a*b for a, b in zip(dx, dy)) / denominator))


def detrend(values, tasks):
    mean_t, mean_y = sum(tasks)/len(tasks), sum(values)/len(values)
    denominator = sum((t-mean_t)**2 for t in tasks)
    slope = sum((t-mean_t)*(v-mean_y) for t, v in zip(tasks, values)) / denominator
    return [v - mean_y - slope*(t-mean_t) for t, v in zip(tasks, values)]


def association(rows, predictor, outcome):
    rows = [r for r in rows if r.get(predictor) is not None]
    if not rows:
        return None
    x, y = [r[predictor] for r in rows], [r[outcome] for r in rows]
    tasks = [r['task_id'] for r in rows]
    log_x = [math.log10(max(v, 1e-30)) for v in x]
    width = OUTCOMES[outcome]
    return {
        'n': len(rows), 'predictor': predictor, 'outcome': outcome,
        'pearson': pearson(x, y), 'spearman': pearson(ranks(x), ranks(y)),
        'pearson_log10_predictor': pearson(log_x, y),
        'pearson_log10_predictor_linear_task_adjusted': pearson(
            detrend(log_x, tasks), detrend(y, tasks)),
        'unique_outcome_values': len({round(v, 10) for v in y}),
        'predictor_min': min(x), 'predictor_max': max(x),
        'outcome_min_pp': min(y), 'outcome_max_pp': max(y),
        'outcome_rounding_half_width_pp': width,
        'rounding_ambiguous_outcomes': sum(abs(v) <= width+1e-10 for v in y),
        'rounded_robust_positive_outcomes': sum(v > width+1e-10 for v in y),
        'rounded_robust_negative_outcomes': sum(v < -width-1e-10 for v in y),
    }


def groups(job):
    output = Path(job['output'])
    metrics = read(output / 'metrics.json')
    assert metrics['tasks'] == 50 and metrics['state'] == 'complete'
    log = Path(metrics['training_log'])
    text = log.read_text(errors='replace')
    assert parse_curve(text) == metrics['top1']
    result = []
    for raw in re.findall(r'CNN: (\{[^\n]+\})', text):
        clean = re.sub(r'np\.float(?:32|64)\(\s*('+NUMBER+r')\s*\)', r'\1', raw)
        row = ast.literal_eval(clean)
        assert isinstance(row, dict) and all(isinstance(v, (int, float)) and math.isfinite(v)
                                             for v in row.values())
        result.append(row)
    assert len(result) == 50
    assert [r['total'] for r in result] == metrics['top1']
    return result, {'metrics': str(output / 'metrics.json'),
                    'metrics_sha256': sha(output / 'metrics.json'),
                    'training_log': str(log), 'training_log_sha256': sha(log)}


def self_check():
    assert ranks([3, 1, 1, 2]) == [4., 1.5, 1.5, 3.]
    assert pearson([1, 2, 3], [2, 4, 6]) == 1.
    assert pearson([1, 2, 3], [3, 2, 1]) == -1.
    assert pearson([1, 1, 1], [1, 2, 3]) is None
    assert max(map(abs, detrend([3, 5, 7], [1, 2, 3]))) < 1e-12


def audit(root=ROOT):
    from goal_audit import audit as full_evidence_audit
    root = Path(root)
    self_check()
    full_evidence_audit(root)
    main_manifest = read(root / 'manifest.json')
    main_status = read(root / 'status.json')
    b_root = root.parent / 'hoep-bmetric-t50-cuda8-20261023'
    b_manifest, b_status = read(b_root / 'manifest.json'), read(b_root / 'status.json')
    chosen = []
    for manifest, status in [(main_manifest, main_status), (b_manifest, b_status)]:
        for job in manifest['jobs']:
            method = job.get('arm', 'bmetric_' + job.get('kind', ''))
            if (job['phase'] == 'formal' and job['dataset'] == 'cub' and job['seed'] == 1
                    and method in ('frozen', 'history', 'bmetric_current')):
                assert status['jobs'][job['name']]['state'] == 'complete'
                assert status['jobs'][job['name']]['returncode'] == 0
                chosen.append((method, job))
    assert {method for method, _ in chosen} == {'frozen', 'history', 'bmetric_current'}
    logs, provenance = {}, {}
    for method, job in chosen:
        logs[method], provenance[method] = groups(job)
    rows, correlations = [], []
    frozen = logs['frozen']
    for method, job in chosen:
        candidate = logs[method]
        for task in range(1, 50):
            assert set(candidate[task]) == set(frozen[task])
            item = analyze(job, task)
            item['method'] = method
            item['frozen_old_accuracy'] = frozen[task]['old']
            item['candidate_old_accuracy'] = candidate[task]['old']
            item['old_gap_pp'] = candidate[task]['old'] - frozen[task]['old']
            item['candidate_old_drop_pp'] = candidate[task-1]['total'] - candidate[task]['old']
            item['frozen_old_drop_pp'] = frozen[task-1]['total'] - frozen[task]['old']
            item['excess_old_drop_pp'] = item['candidate_old_drop_pp'] - item['frozen_old_drop_pp']
            item['old_classes'] = 4*task
            item['evaluated_classes'] = 4*(task+1)
            rows.append(item)
            if task % 10 == 0 or task == 49:
                print(json.dumps({'method': method, 'completed_boundaries': task}), flush=True)
        if method != 'frozen':
            selected = [r for r in rows if r['method'] == method]
            for predictor in PREDICTORS:
                for outcome in OUTCOMES:
                    stat = association(selected, predictor, outcome)
                    if stat is not None:
                        stat['method'] = method
                        correlations.append(stat)
    return {
        'updated': time.time(), 'scope': 'CUB seed1; three complete T50 runs; 49 transitions each',
        'script_sha256': sha(Path(__file__)),
        'operator_helper_sha256': sha(root / 'operator_boundary_audit.py'),
        'raw_log_provenance': provenance, 'rows': rows, 'associations': correlations,
        'outcomes': {
            'old_gap_pp': 'candidate old Top1 at t minus Frozen old Top1 at t; identical evaluated class space',
            'excess_old_drop_pp': '(candidate total Top1 at t-1 minus candidate old Top1 at t) minus the same Frozen drop; includes new-class competition and all deployment operations',
        },
        'interpretation': 'Exploratory within-seed associations; dependent tasks, rounded outcomes, multiple prespecified predictors. No p-values or causal claim. Linear task adjustment is descriptive. Weighted residual uses own archived historical input moments and a Frobenius-optimal projector, not a weighted-optimal reconstruction or refreshed current activations of all historical images.',
    }


def plot(result, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fields = [
        ('history', 'weighted_frobenius_projection_residual_ratio', 'History: own-input weighted residual'),
        ('history', 'frobenius_projection_residual_ratio', 'History: Frobenius residual'),
        ('bmetric_current', 'frobenius_projection_residual_ratio', 'B-current: Frobenius residual'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True, layout='constrained')
    for axis, (method, field, title) in zip(axes, fields):
        rows = [r for r in result['rows'] if r['method'] == method]
        scatter = axis.scatter([max(r[field], 1e-30) for r in rows],
                               [r['excess_old_drop_pp'] for r in rows],
                               c=[r['task_id'] for r in rows], cmap='viridis',
                               vmin=1, vmax=49, s=25, alpha=.8)
        axis.set_xscale('log')
        axis.axhline(0, color='black', linewidth=.8)
        axis.axhspan(-.02, .02, color='gray', alpha=.15)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel('Normalized squared projection residual')
        axis.grid(alpha=.2)
        stat = next(s for s in result['associations'] if s['method'] == method
                    and s['predictor'] == field and s['outcome'] == 'excess_old_drop_pp')
        axis.text(.02, .98, f"n={stat['n']}; Spearman={stat['spearman']:.3f}",
                  transform=axis.transAxes, va='top', fontsize=9)
    axes[0].set_ylabel('Extra old-class accuracy drop vs Frozen (pp)')
    fig.colorbar(scatter, ax=axes, label='Task index (dependent observations)', shrink=.8)
    fig.suptitle('CUB seed1: exploratory associations after deployment; rounded outcomes', fontsize=11)
    base = root / 'records/long_sequence_recoverability_scatter'
    fig.savefig(base.with_suffix('.png'), dpi=180)
    fig.savefig(base.with_suffix('.svg'))
    plt.close(fig)


def notes(result, root):
    def number(value):
        return '—' if value is None else f'{value:+.3f}'
    p = root / 'records/ANALYSIS_NOTES.md'
    text = p.read_text()
    heading = '## 2026-10-23：完整CUB序列的可恢复性关联诊断'
    if heading in text:
        raise FileExistsError('Preserve the prior analysis; use a new audit version for changes')
    lines = ['', heading, '',
        '读取已完整结束的CUB seed1 Frozen、history与B-current各50个边界；不读图像、不构造原型、不修改模型。所有源配置、原始50点曲线和最终部署完整性先经目标审计。对task编号1–49逐一核验当前与上一部署快照，计算147个任务边界的算子指标；每个候选的相关性仅使用49个任务级观测，不把Q/V层或分支重复当作独立样本。', '',
        '主观察量为额外旧类下降：`(候选上个边界Total−当前Old)−(Frozen上个边界Total−当前Old)`。Old只包含此前4t类，当前评估空间包含4(t+1)类。减去Frozen可控制部分共同变化，但它仍包含新增类别竞争、训练、LS、Transport和原型更新的共同作用。另在JSON保存当前同边界旧类差值；没有声称这是合并前旧类损失。', '',
        '下表对主观察量报告关联。Pearson使用原指标，Spearman使用平均秩，最后一列为对`log10(指标)`与额外下降分别去除线性任务趋势后的Pearson。所有指标、两种观察量与结果均保存；没有按相关系数大小筛选边界或训练方案。', '',
        '| 方案 | 指标 | N | Pearson | Spearman | log指标去线性任务趋势 |',
        '|---|---|---|---|---|---|']
    for stat in result['associations']:
        if stat['outcome'] != 'excess_old_drop_pp':
            continue
        lines.append('| ' + ' | '.join([stat['method'], PREDICTORS[stat['predictor']],
            str(stat['n']), number(stat['pearson']), number(stat['spearman']),
            number(stat['pearson_log10_predictor_linear_task_adjusted'])]) + ' |')
    lines += ['',
        '准确率被舍入到0.01pp：同边界旧类差值的确定性不确定范围约±0.01pp，四项构成的额外下降约±0.02pp。JSON同时记录了落入舍入不确定范围的观测数。图中灰带表示±0.02pp舍入范围，不是置信区间。', '',
        '加权残差仅使用history自己历史任务当时存档的输入二阶矩；没有用它代替B-current内部输入分布，也没有重放旧图像刷新激活。它是对Frobenius最优投影残差施加数据权重，不能称作数据加权最优重构误差。', '',
        '这些是依赖任务、单seed、同序列的探索性关联；线性去趋势也不能消除所有时间/类别组成混淆。没有p值、独立样本置信区间、准确率上界或因果结论。若后续将某指标用于控制A更新，需先在未见seed/数据集上验证其预测方向，不能把本表当作已验证的控制规律。', '',
        '完整逐分支数值、原始日志/快照/计算源码SHA256以及所有关联见 `records/long_sequence_recoverability_audit.json`；标准绘图见下图。T50提升超过Frozen 0.5pp的目标仍需真实完整配对结果验证。', '',
        '![CUB recoverability associations](records/long_sequence_recoverability_scatter.png)', '']
    text += '\n'.join(lines)
    tmp = p.with_suffix('.md.tmp')
    tmp.write_text(text)
    tmp.replace(p)


if __name__ == '__main__':
    torch.set_num_threads(1)
    result = audit()
    path = ROOT / 'records/long_sequence_recoverability_audit.json'
    if path.exists():
        raise FileExistsError('Keep the completed audit; version any repeat analysis explicitly')
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temp.replace(path)
    plot(result, ROOT)
    notes(result, ROOT)
    print(json.dumps({'completed': True, 'boundaries': len(result['rows']),
                      'associations': result['associations']}, allow_nan=False), flush=True)
