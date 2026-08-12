# 强外部基线：发布值与本机复现状态（P5）

更新时间：2026-08-12。所有发布值均从官方论文 PDF 提取（arXiv / CVF / ICLR OpenReview），
不做二手转述；协议差异单独标注，最终论文以“发布值（原协议）| 本机复现（本仓库协议）”分栏报告。

## 1. 官方仓库（已克隆，位于 `external_baselines/`，不入主仓库）

| 方法 | 官方仓库 | 克隆 commit | 环境要求 | 本机可导入? |
| --- | --- | --- | --- | --- |
| InfLoRA (CVPR 2024) | github.com/liangyanshuo/InfLoRA | `e08b00e` | torch 1.10 / timm 0.6.7 | 是（bestformer: torch 1.12.1 / timm 0.6.12 / sklearn 1.3.2） |
| CL-LoRA (CVPR 2025) | github.com/JiangpengHe/CL-LoRA | `df4e74e` | torch 2.0.1 / timm 0.6.12 | 是（bestformer） |
| LoRA-Sub-DRS (CVPR 2025) | github.com/scarlet0703/LoRA-Sub-DRS | 最新 master（2026-08-12 抓取） | torch 2.0.1 / timm 0.6.7 | 是（bestformer） |

## 2. 发布值（原协议）

### 2.1 InfLoRA（CVPR 2024，arXiv:2404.00228，Table 1/2，5 trials，Adam，INR 50 ep、C100 20 ep、batch 128）

| 数据集 / 任务数 | Final (`ACC_T`) | Average (`ĀCC_T`) |
| --- | ---: | ---: |
| ImageNet-R N=10 | 75.65 ± 0.14 | 80.82 ± 0.24 |
| ImageNet-R N=5 | 77.52 ± 0.37 | 82.01 ± 0.12 |
| ImageNet-R N=20 | 71.01 ± 0.45 | 77.28 ± 0.45 |
| CIFAR-100 N=10 | 86.51 ± 0.73 | 91.70 ± 0.32 |

### 2.2 CL-LoRA（CVPR 2025，arXiv:2505.24816，Table 1，10 runs；主表任务数与本文不同）

| 数据集 / 任务数 | Final (`A_T`) | Average (`A`) |
| --- | ---: | ---: |
| CIFAR-100 T=20 | 85.32 ± 0.08 | 91.02 ± 0.12 |
| ImageNet-R T=40 | 74.51 ± 0.14 | 81.58 ± 0.59 |
| CIFAR-100 T=10（Table 2 消融，完整配置） | — | 91.85 |
| ImageNet-R T=20（Table 2 消融，完整配置） | — | 84.77 |

### 2.3 LoRA⁻ DRS（CVPR 2025，arXiv:2503.18985，Table 2/3，5 seeds，Adam，INR 50 ep、C100 20 ep、batch 128）

| 数据集 / 任务数 | Final (`ACC_T`) | Average (`ĀCC_T`) |
| --- | ---: | ---: |
| ImageNet-R N=10 | 74.74 ± 0.78 | 81.16 ± 0.59 |
| ImageNet-R N=20 | 74.80 ± 0.73 | 80.69 ± 0.75 |
| CIFAR-100 N=10 | 89.14 ± 0.23 | 92.55 ± 0.25 |
| CIFAR-100 N=20 | 88.69 ± 0.15 | 92.25 ± 0.24 |

### 2.4 SD-LoRA（ICLR 2025，arXiv:2501.13198，Table 2/6，5 runs，Adam，INR 30 ep、C100/CUB 20 ep、batch 128）

| 数据集 / 任务数 | Final (`Acc`) | AAA |
| --- | ---: | ---: |
| ImageNet-R N=10 | 77.34 ± 0.35 | 82.04 ± 0.24 |
| ImageNet-R N=5 | 79.15 ± 0.20 | 83.01 ± 0.42 |
| ImageNet-R N=20 | 75.26 ± 0.37 | 80.22 ± 0.72 |
| CIFAR-100 N=10 | 88.01 ± 0.31 | 92.54 ± 0.18 |
| CUB-200 N=10 | 77.48 ± 0.20 | 85.59 ± 0.44 |

## 3. 本机同协议对照（本仓库冻结协议）

- SD-LoRA（同协议，seed1-5）：见 `p3_strict_n5_summary.md`。
- EXP-009（同协议，seed1-5）：见 `p3_strict_n5_summary.md`。
- 完整方法（Live-A Aggregate-B + Dual-B，seed1-5）：见 `p3_strict_n5_summary.md`。
- InfLoRA / CL-LoRA / LoRA-DRS 本机复现：决定运行 CIFAR-100（T=10，seed1）官方实现，使用各自官方默认训练细节；
  结果与发布值分栏报告，且仅作为“官方实现可复现性”证据，不做跨协议显著性比较。
- 移植性修复（仅设备分配，不改变算法）：
  - CL-LoRA `backbone/vit_cllora.py`：`block_weight = torch.ones(3).cuda()`
    硬编码 cuda:0，改为 `torch.ones(3, device=x.device)`；
  - LoRA-DRS `models/zoo.py::ortho_penalty` 与 `models/sinet.py`：硬编码
    `.cuda()` 改为 `device=t.device` / `device=x.device`；
  - bestformer 环境补装 scikit-learn/ipdb/easydict。

## 4. 论文口径

外部基线只用于展示方法空间与参数效率定位；不宣称任何“全指标优于外部基线”。
主对比仍为同协议 SD-LoRA / EXP-009（seed1-5 严格配对统计）。
