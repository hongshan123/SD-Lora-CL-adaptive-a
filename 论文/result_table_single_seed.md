# 单 seed 结果总表（论文准备：2026-08-06）

> 数据来源：experiment_sd.md / experiment_note_sd.md / 各实验日志。第二轮 INR 门槛为 Final ≥79.14 且（Forgetting ≤6.26 或 AvgAcc ≥82.97），C100 为 Final ≥88.22 且（Forgetting ≤7.08 或 AvgAcc ≥92.57）；新方法（cumulative+gauge）最低验收线见 method_revision_sd.md。

## ImageNet-R（seed1995，10 任务）

| 方法 | Final Top1 | AvgAcc | Forgetting | LoRA 参数 | 相对 SD-LoRA |
| --- | ---: | ---: | ---: | ---: | ---: |
| SD-LoRA 基线 | 78.76 | 83.13 | 5.61 | 3,686,400 | — |
| Shared-A + prototype（EXP-009） | 79.34 | 82.47 | 7.26 | 2,027,530 | Final +0.58 |
| SA-LoRA r10（A 冻结，EXP-004） | 77.59 | 82.82 | 6.16 | 368,640 | -1.17 |
| SA-LoRA r20（A 冻结，EXP-005） | 78.03 | 82.59 | 7.93 | 737,280 | -0.73 |
| generic affine LRPT r10（EXP-010） | 79.29 | 83.00 | 6.61 | 2,027,530 | +0.53 |
| LoRA-aware LRPT NA（EXP-011） | 79.43 | 82.68 | 6.18 | 2,027,530 | +0.67 |
| effective operator stability（EXP-012） | 79.39 | 82.20 | 6.98 | 2,027,530 | +0.63 |
| **cumulative-only（v2）** | 78.78 | 81.69 | 7.28 | 371,040 | +0.02 |
| **cumulative+gauge（主方法）** | **79.06** | 81.77 | **6.82** | **371,040** | **+0.30** |
| cumulative+gauge+LRPT | 78.49 | 82.19 | 6.38 | 371,040 | -0.27 |

## CIFAR-100（seed1993，10 任务）

| 方法 | Final Top1 | AvgAcc | Forgetting | LoRA 参数 | 相对 SD-LoRA |
| --- | ---: | ---: | ---: | ---: | ---: |
| SD-LoRA 基线 | 86.89 | 91.44 | 5.58 | 3,686,400 | — |
| Shared-A + prototype（EXP-009） | 88.42 | 92.07 | 8.08 | 2,027,530 | +1.53 |
| SA-LoRA r10（A 冻结，EXP-004） | 86.53 | 91.70 | 7.47 | 368,640 | -0.36 |
| SA-LoRA r20（A 冻结，EXP-005） | 86.35 | 91.47 | 8.68 | 737,280 | -0.54 |
| generic affine LRPT r10（EXP-010 最优） | 88.58 | 92.46 | 7.12 | 2,027,530 | +1.69 |
| LoRA-aware LRPT NA（EXP-011） | 88.25 | 92.25 | 7.54 | 2,027,530 | +1.36 |
| **cumulative+gauge（主方法）** | **87.70** | 91.83 | 8.53 | **371,040** | **+0.81** |

## 主方法参数口径（v2 单文件状态）

- LoRA：canonical_down 184,320 + cumulative_up 184,320 + triangular_r 2,400 = 371,040（基线 3,686,400 的 10.07%）。
- 含原型：INR 524,640（14.23%，减 85.77%）；C100 447,840（12.15%，减 87.85%）。
- 推理：单 canonical 模型，无 task-id/router/逐任务 adapter；恢复训练无需历史 B bank。
- 一致性审计：`verify_sa_consistency.py` 在 INR/C100 主方法产物均 PASS（feature 与 prototype logits 差 0）。

## 待补

- 多 seed mean±std 与显著性（已完成，见下）。
- 任务长度 T=5/20/40。
- FLOPs/吞吐/峰值显存。
- 强基线（InfLoRA/CL-LoRA/LoRA-DRS/DGS 可复现实现，待评估）。

## 多 seed（2026-08-07 补）

| 方法 | 数据集 | Final mean±std | AvgAcc mean±std | Forgetting mean±std |
| --- | --- | ---: | ---: | ---: |
| cumulative+gauge（4 seeds） | ImageNet-R | 78.46±0.51 | 82.28±0.47 | 7.86±1.35 |
| EXP-009（4 seeds） | ImageNet-R | 79.11±0.40 | 82.80±0.49 | 7.90±1.16 |
| cumulative+gauge（4 seeds） | CIFAR-100 | 87.83±0.13 | 91.52±0.30 | 8.68±0.20 |
| EXP-009（4 seeds） | CIFAR-100 | 88.05±0.27 | 91.63±0.36 | 8.67±0.47 |

配对 t 检验（main − EXP-009，n=4，按 seed 内连接配对；脚本 `scripts/multiseed_stats.py`）：

| 数据集 | ΔFinal (p) | ΔAvgAcc (p) | ΔForgetting (p) |
| --- | ---: | ---: | ---: |
| ImageNet-R | -0.65（0.0256） | -0.52（0.0086） | -0.04（0.8521） |
| CIFAR-100 | -0.22（0.2815） | -0.12（0.1469） | +0.01（0.9375） |

TOST（±0.5）：INR Final/AvgAcc 不等价，Forgetting 等价；C100 Final 不等价，AvgAcc/Forgetting 等价。结论：不能写“统计等价”；INR 存在约 0.5–0.65 个点的显著精度代价。LoRA 371,040 vs 2,027,530（-81.7%）。

## 额外数据集（2026-08-07 补）

| 方法 | 数据集 | Final Top1 | AvgAcc | Forgetting |
| --- | --- | ---: | ---: | ---: |
| cumulative+gauge | CUB-200（seed1） | **79.79** | **87.69** | **14.20** |
| EXP-009 | CUB-200（seed1） | 71.75 | 84.93 | 23.31 |

## 效率（GPU, batch32, 2026-08-07）

| 产物 | FLOPs/forward | 吞吐 | 峰值显存 |
| --- | ---: | ---: | ---: |
| 主方法 INR | 1.129e12 | 411.9 img/s | 579.1 MiB |
| EXP-009 INR | 1.129e12 | 414.9 img/s | 577.7 MiB |
| 主方法 C100 | 1.129e12 | 413.3 img/s | 579.1 MiB |
