# P5 第三数据集 CUB-200-2011 结果（冻结协议）

更新时间：2026-08-12 22:10（完整方法 3-seed 完成；EXP-009/SD-LoRA 队列运行中）。

## 1. 协议

- 数据集：CUB-200-2011（200 类，T=10，init 20 / increment 20，官方 train/test split）。
- 冻结协议：SGD / constant / lr 0.01 / 20 epochs / batch 32 / 4 卡 NCCL / 确定性训练（与 `p5_cub_livea_dual_b_seed1_nccl.json` 一致）。
- 方法：完整方法（Live-A Aggregate-B + Dual-B）、EXP-009（Shared-A + per-task B + prototype，无 Dual-B）、SD-LoRA（原版 per-task bank + FC）。
- 汇总：`scripts/collect_p5_cub.py`；队列：`run_p5_cub_multi_queue.sh`（commit `3cdfa7d`）。

## 2. 完整方法 3-seed（已完成）

| 方法 | Final | AAA | Forgetting | 协议 |
| --- | ---: | ---: | ---: | --- |
| 完整方法（seed1） | 77.84 | 85.85 | 16.27 | 冻结 SGD |
| 完整方法（seed2） | 71.80 | 83.22 | 17.02 | 冻结 SGD |
| 完整方法（seed3） | 76.16 | 84.04 | 17.02 | 冻结 SGD |
| 完整方法（mean±std） | **75.27±3.10** | **84.37±1.34** | **16.77±0.43** | 冻结 SGD |
| EXP-009（seed1，旧） | 71.75 | 84.93 | 23.31 | 旧 Adam |
| SD-LoRA（论文发布值，5 runs） | 77.48±0.20 | 85.59±0.44 | — | Adam/batch128 |

完整方法 mean 相对旧 EXP-009：Final +3.52、AAA -0.56、F -6.54；
seed1 相对旧 EXP-009：Final +6.09、AAA +0.92、F -7.04。

## 3. 多 seed（运行中）

队列剩余 6 runs：EXP-009 seed1-3（同冻结协议重跑）、SD-LoRA seed1-3；
全部 exit=0 后回填三方法均值±标准差与配对差。

## 4. 待办

- 队列完成后运行 `python scripts/collect_p5_cub.py` 生成正式表；
- 更新 `experiment_sd.md` / `experiment_note_sd.md` / `paper_output/final_positioning.md`；
- 结论初判：完整方法 mean Final 高于旧 EXP-009（+3.52）但低于 SD-LoRA 论文
  发布值（75.27 vs 77.48），seed 方差大（71.8-77.8）；AAA 与旧 EXP-009 接近、
  Forgetting 明显更低（-6.5）；与同协议 EXP-009/SD-LoRA 的 3-seed 配对差待回填。
