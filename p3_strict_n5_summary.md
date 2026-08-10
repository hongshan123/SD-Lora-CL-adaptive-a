# P3 严格 n=5 统计汇总（确认性主统计）

## 1. 统计口径

- 确认种子（论文主统计）：ImageNet-R seed `1,2,3,4,5`；CIFAR-100 seed `1,2,3,4,5`。
- 开发种子（不进入主统计）：ImageNet-R seed `1995`（Schedule A/B 选择）、CIFAR-100 seed `1993`（开发与阶段门槛）。
- 所有数值来自提交 `c256cc4` 的 30 个正式运行（每个运行 `exit=0`），配对按 seed 内连接。
- `p3_multiseed_stats_output.txt`（n=6，含开发种子）仅作为描述性敏感性分析，不作为论文主显著性结论。
- 完整逐 seed 明细、paired t-test、95% CI、Cohen's dz、TOST 见 `p3_strict_n5_stats_output.txt`。
- 日志/配置 SHA/产物审计：`scripts/audit_p3_runs.py` 输出 `P3 AUDIT PASS`（30/30：commit、config SHA-256、seed 一致、10 任务曲线、AAA/Forgetting、artifact、queue status=0）。

## 2. 严格 n=5 结果（mean ± sample std）

| 数据集 | 方法 | Final Top1 | AAA | Forgetting（越低越好） |
| --- | --- | ---: | ---: | ---: |
| ImageNet-R | 完整方法（Live-A Aggregate-B + Dual-B） | 79.07±0.23 | 83.06±0.52 | 7.68±1.03 |
| ImageNet-R | SD-LoRA | 78.34±0.48 | 82.96±0.87 | 7.79±1.04 |
| ImageNet-R | EXP-009 | 79.16±0.32 | 82.63±0.36 | 7.69±0.93 |
| CIFAR-100 | 完整方法 | 87.41±0.32 | 91.06±0.64 | 9.09±0.62 |
| CIFAR-100 | SD-LoRA | 86.81±0.56 | 91.45±0.63 | 6.57±0.71 |
| CIFAR-100 | EXP-009 | 87.58±0.28 | 91.12±0.54 | 9.19±0.47 |

## 3. 配对差异（完整方法 − 对照，n=5）

| 数据集 | 对照 | Final Δ（p） | AAA Δ（p） | Forgetting Δ（p） |
| --- | --- | ---: | ---: | ---: |
| ImageNet-R | SD-LoRA | +0.730（p=0.0784） | +0.103（p=0.6743） | -0.105（p=0.6984） |
| ImageNet-R | EXP-009 | -0.082（p=0.1257） | +0.434（p=0.0148） | -0.012（p=0.9092） |
| CIFAR-100 | SD-LoRA | +0.602（p=0.0411） | -0.385（p=0.0057） | +2.522（p=0.0002） |
| CIFAR-100 | EXP-009 | -0.172（p=0.0890） | -0.054（p=0.5188） | -0.100（p=0.6545） |

完整方法相对 SD-LoRA：

- ImageNet-R：Final `+0.73`（5/5 为正，配对 t 检验约 `p=0.078`），AAA `+0.10`，Forgetting `-0.11`；
- CIFAR-100：Final `+0.60`（约 `p=0.041`），AAA `-0.39`（约 `p=0.006`），Forgetting `+2.52`（约 `p<0.001`）。

## 4. 允许的严谨主张

> 在减少约 82% 至 86% 持久状态的同时，完整方法保持或提高 Final Top1；ImageNet-R 的 AAA/Forgetting 与 SD-LoRA 接近，但 CIFAR-100 存在显著的中期准确率（AAA −0.39）和遗忘（+2.52）代价。

禁止声称“所有持续学习指标全面优于 SD-LoRA”。C100 Final 优势主要来自新任务塑性和最终分类校准，历史任务保持并未全面改善（历史 90 类 +0.16、最新 10 类 +4.54、Task0 最终 −2.42，AAA 缺口集中于 Task2–Task6）。

## 5. 已知 C100 症状（严格五种子）

- 历史 90 类：完整方法 `86.71`，SD-LoRA `86.55`，仅 `+0.16`；
- 最新 10 类：完整方法 `93.68`，SD-LoRA `89.14`，为 `+4.54`；
- Task0 最终准确率：完整方法 `75.42`，SD-LoRA `77.84`，为 `-2.42`；
- AAA 缺口主要集中于 Task2 至 Task6。

## 6. 生成与复核命令

```bash
source /home/zhaoyang/miniconda3/etc/profile.d/conda.sh
conda activate sdlora
PYTHON=$(which python) bash scripts/strict_n5_stats.sh   # 生成 p3_strict_n5_stats_output.txt
python scripts/audit_p3_runs.py                          # 30 个运行审计
```

统计脚本 `scripts/multiseed_stats.py` 支持 `--seeds 1,2,3,4,5` 显式确认种子列表；落入列表之外的 seed（1995/1993）会被排除并打印，缺失任一确认种子则报错。
