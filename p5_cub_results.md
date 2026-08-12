# P5 第三数据集 CUB-200-2011 结果（冻结协议）

更新时间：2026-08-13 02:35（全部 9 runs exit=0，三方法 3-seed 完成）。

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

### 2.1 EXP-009（同冻结协议，3-seed）

| seed | Final | AAA | Forgetting |
| --- | ---: | ---: | ---: |
| 1 | 77.89 | 86.98 | 17.08 |
| 2 | 72.20 | 83.47 | 17.10 |
| 3 | 76.01 | 84.58 | 17.28 |
| mean±std | 75.37±2.90 | 85.01±1.78 | 17.15±0.11 |

同协议配对差（完整方法 − EXP-009，3 seeds）：Final -0.10±0.28、AAA -0.64±0.45、
F -0.38±0.38 —— 两者在 CUB 上基本等价（完整方法以 O(1) 状态取得）。

### 2.2 SD-LoRA（同冻结协议，3-seed）

| seed | Final | AAA | Forgetting |
| --- | ---: | ---: | ---: |
| 1 | 71.19 | 81.71 | 12.71 |
| 2 | 69.83 | 80.75 | 8.54 |
| 3 | 72.76 | 82.54 | 11.66 |
| mean±std | 71.26±1.47 | 81.66±0.90 | 10.97±2.17 |

完整方法 − SD-LoRA 配对差：Final +4.01±2.40、AAA +2.70±1.34、F +5.80±2.49。

## 2.3 三方法汇总

| 方法 | Final | AAA | Forgetting | 持久 LoRA 状态 |
| --- | ---: | ---: | ---: | ---: |
| 完整方法 | 75.27±3.12 | 84.37±1.34 | 16.77±0.43 | O(1)，368,640 |
| EXP-009 | 75.37±2.90 | 85.01±1.79 | 17.15±0.11 | O(T) |
| SD-LoRA | 71.26±1.47 | 81.66±0.90 | 10.97±2.17 | O(T)，3,686,400（T=10） |
| SD-LoRA（论文发布值） | 77.48±0.20 | 85.59±0.44 | — | 不同协议（Adam/batch128） |

结论：冻结协议下完整方法 CUB Final/AAA 显著高于同协议 SD-LoRA（+4.0/+2.7），
但 Forgetting 更高（+5.8）；相对 EXP-009 等价（Final -0.1、AAA -0.6、F -0.4）。

## 3. 多 seed（运行中）

队列状态：全部完成（`p5_cub_remaining_queue.log`，ALL DONE 2026-08-13 02:33）。

⚠️ 2026-08-12 22:22：EXP-009 seed1 在 task 9 时 CUDA OOM（GPU2 与 LoRA-DRS
本地复现并发，15.15 GiB + 8.54 GiB > 24 GiB）。已停止并发外部基线（CL-LoRA、
LoRA-DRS 均已终止；InfLoRA 完成后 GPU 全空闲再重启队列）。失败目录已移出
（`/tmp/CUB_P5_EXP009_SEED1_NCCL.failed_oom`）；`run_p5_cub_remaining_queue.sh`
包含剩余 6 runs，并在启动前检查外部基线进程。

## 4. 待办

- 队列完成后运行 `python scripts/collect_p5_cub.py` 生成正式表；
- 更新 `experiment_sd.md` / `experiment_note_sd.md` / `paper_output/final_positioning.md`；
- 结论：完整方法 mean Final 高于同协议 SD-LoRA（+4.01）但与 EXP-009 等价；
  与 SD-LoRA 论文发布值（77.48/85.59）相差 -2.2/-1.2（不同协议，不作显著比较）；
  Forgetting 高于同协议 SD-LoRA（+5.8）但低于旧 EXP-009 Adam 协议（-6.5）。
