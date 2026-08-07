# Evidence Bank

> 证据 ID 规则：E<数据集缩写><编号>。所有数字来自 experiment_sd.md / experiment_note_sd.md / 训练日志与产物审计，可复现（配置与 commit 见各 EXP 条目）。

## E-INR 系列（ImageNet-R, 200 类, 10 任务, seed1995 主协议；多 seed 见 E-MULTI）

| ID | 内容 | 值 | 来源 |
| --- | --- | ---: | --- |
| E-INR-BASE | SD-LoRA 基线 | 78.76 / 83.13 / 5.61 | EXP-000 |
| E-INR-009 | Shared-A + prototype（EXP-009） | 79.34 / 82.47 / 7.26 | EXP-009 |
| E-INR-010 | generic affine LRPT r10 | 79.29 / 83.00 / 6.61 | EXP-010 |
| E-INR-011 | LoRA-aware LRPT NA | 79.43 / 82.68 / 6.18 | EXP-011 |
| E-INR-012 | operator stability | 79.39 / 82.20 / 6.98 | EXP-012 |
| E-INR-CUM | cumulative-only（v2, 无 gauge） | 78.78 / 81.69 / 7.28 | EXP-016 |
| E-INR-GAUGE | **cumulative+gauge（主方法）** | **79.06 / 81.77 / 6.82** | EXP-016 |
| E-INR-GLRPT | cumulative+gauge+LRPT | 78.49 / 82.19 / 6.38 | EXP-016 |
| E-INR-DIAG | 每任务 gauge residual/rotation/preservation | 1e-8~3e-8（科学计数法） | EXP-016/017 日志 |

## E-C100 系列（CIFAR-100, 100 类, 10 任务, seed1993）

| ID | 内容 | 值 | 来源 |
| --- | --- | ---: | --- |
| E-C100-BASE | SD-LoRA 基线 | 86.89 / 91.44 / 5.58 | EXP-000 |
| E-C100-009 | EXP-009 | 88.42 / 92.07 / 8.08 | EXP-009 |
| E-C100-010 | generic affine LRPT r10 最优 | 88.58 / 92.46 / 7.12 | EXP-010 |
| E-C100-GAUGE | **cumulative+gauge（主方法）** | **87.70 / 91.83 / 8.53** | EXP-017 |

## E-MULTI（多 seed，mean±std，n=4）

| ID | 内容 | Final | AvgAcc | Forgetting |
| --- | --- | ---: | ---: | ---: |
| E-MULTI-INR-MAIN | cumulative+gauge | 78.46±0.51 | 82.28±0.47 | 7.86±1.35 |
| E-MULTI-INR-BASE | EXP-009 | 79.11±0.40 | 82.80±0.49 | 7.90±1.16 |
| E-MULTI-C100-MAIN | cumulative+gauge | 87.83±0.13 | 91.52±0.30 | 8.68±0.20 |
| E-MULTI-C100-BASE | EXP-009 | 88.05±0.27 | 91.63±0.36 | 8.67±0.47 |
| E-PAIR-INR | paired Δ（main−base, n=4，seed 内连接） | -0.65（p=0.0256；95%CI [-1.15,-0.15]；dz=-2.07） | -0.52（p=0.0086；95%CI [-0.79,-0.25]；dz=-3.09） | -0.04（p=0.8521；95%CI [-0.63,0.55]；dz=-0.10） |
| E-PAIR-C100 | paired Δ（n=4，seed 内连接） | -0.22（p=0.2815；95%CI [-0.75,0.31]；dz=-0.66） | -0.12（p=0.1469；95%CI [-0.31,0.08]；dz=-0.97） | +0.01（p=0.9375；95%CI [-0.51,0.53]；dz=+0.04） |

TOST（±0.5 margin, α=0.05）：INR Final/AvgAcc 不等价（p_upper=0.0026/0.0006，p_lower=0.7976/0.5952）；INR Forgetting 等价（p_upper=0.0315，p_lower=0.0445）；C100 Final 不等价（p_upper=0.0117，p_lower=0.0970）；C100 AvgAcc/Forgetting 等价。

## E-INR-DIAG-P0（pre-save gauge 诊断，INR seed1995，2026-08-07）

| task | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| projection residual / preservation | 4.36e-2 | 2.69e-2 | 2.63e-2 | 1.31e-2 | 2.13e-2 | 2.08e-2 | 1.49e-2 | 2.57e-2 | 1.79e-2 |
| basis rotation | 2.96e-2 | 2.87e-2 | 5.46e-2 | 1.34e-3 | 2.20e-3 | 2.83e-2 | 1.75e-3 | 2.88e-2 | 1.91e-3 |

性能复现：78.91/81.86/7.02（原 gauge 79.06/81.77/6.82，单次噪声范围内）。

## E-STAGE-A（Union-SVD vs Gauge，INR seed1995）

| 方法 | Final | AvgAcc | F | LoRA 参数 |
| --- | ---: | ---: | ---: | ---: |
| union_svd_r4 | 77.59 | 81.63 | 7.59 | 147,840 |
| gauge_r4 | 77.09 | 81.26 | 8.20 | 147,840 |
| union_svd_r8 | 78.69 | 82.31 | 6.77 | 296,448 |
| gauge_r8 | 78.24 | 81.98 | 6.88 | 296,448 |
| union_svd_r10 | 78.61 | 81.91 | 7.05 | 371,040 |
| gauge_r10（主方法） | 79.06 | 81.77 | 6.82 | 371,040 |

一致性审计：union r4/r8/r10 `verify_sa_consistency.py` 全部 PASS（diff=0）。

## E-DIAG-PROTO（离线原型漂移诊断，INR seed1995，仅诊断）

- 存储训练期原型：79.06；当前骨干重算原型：79.43（+0.37）；冻结 base ViT + base 原型：79.43；base-vs-final 原型余弦均值 1.0000。
- 判断：原型坐标过期与 backbone 干扰均不主导；不触发 P3 activation sketch。

## E-TL（任务长度，seed1995/1993，cumulative+gauge）

| ID | 数据集 | T | Final | AvgAcc | Forgetting |
| --- | --- | ---: | ---: | ---: | ---: |
| E-TL-INR5 | INR | 5 | 77.54 | 81.81 | 8.83 |
| E-TL-INR10 | INR | 10 | 79.06 | 81.77 | 6.82 |
| E-TL-INR20 | INR | 20 | 77.03 | 81.57 | 9.51 |
| E-TL-INR40 | INR | 40 | 75.31 | 80.63 | 12.34 |
| E-TL-C5 | C100 | 5 | 88.06 | 91.52 | 8.77 |
| E-TL-C10 | C100 | 10 | 87.70 | 91.83 | 8.53 |
| E-TL-C20 | C100 | 20 | 85.63 | 91.25 | 10.72 |

## E-CUB（CUB-200-2011, seed1, 10 任务）

| ID | 内容 | Final | AvgAcc | Forgetting |
| --- | --- | ---: | ---: | ---: |
| E-CUB-MAIN | cumulative+gauge | 79.79 | 87.69 | 14.20 |
| E-CUB-BASE | EXP-009 | 71.75 | 84.93 | 23.31 |

## E-EFF（GPU, batch32, 20 iters）

| ID | 内容 | FLOPs/forward | 吞吐 | 峰值显存 |
| --- | --- | ---: | ---: | ---: |
| E-EFF-INR-MAIN | 主方法 INR | 1.129e12 | 411.9 img/s | 579.1 MiB |
| E-EFF-INR-BASE | EXP-009 INR | 1.129e12 | 414.9 img/s | 577.7 MiB |
| E-EFF-C100-MAIN | 主方法 C100 | 1.129e12 | 413.3 img/s | 579.1 MiB |

## E-PARAMS（持久参数）

| ID | 内容 | 值 |
| --- | --- | ---: |
| E-PARAMS-MAIN | v2 LoRA（Q^T+H+R） | 371,040（基线 10.07%） |
| E-PARAMS-INR-P | 含 INR 原型 | 524,640（14.23%，减 85.77%） |
| E-PARAMS-C100-P | 含 C100 原型 | 447,840（12.15%，减 87.85%） |
| E-PARAMS-009 | EXP-009 LoRA | 2,027,530 |
| E-PARAMS-T | v1 随 T 增长（184,320/任务+scale），v2 恒定 | T=10 时 2,027,530 vs 371,040 |

## E-AUDIT（工程审计）

| ID | 内容 | 结果 |
| --- | --- | --- |
| E-AUDIT-VERIFY | verify_sa_consistency（INR seed1995/seed3、C100 seed1993、CUB 主方法） | PASS，feature/prototype logits diff=0 |
| E-AUDIT-UNIT | 单元测试 | 45 passed（含 Phase A/B/C 代数等价性） |
| E-AUDIT-DDP | 4 卡 DDP 1-epoch smoke + 全部完整运行 | exit=0，无逐任务 B 文件 |

## E-CORR（相关性，任务 1-9）

| ID | 内容 | 结果 |
| --- | --- | --- |
| E-CORR-OP | EXP-012 operator drift vs 遗忘 | r=+0.26（p=0.50） |
| E-CORR-CTRL | log-only 控制组 raw drift vs 遗忘 | r=-0.73（p=0.025，塑性混杂） |
| E-CORR-GAUGE | gauge 三项诊断 vs 遗忘 | 零方差，无相关 |
| E-CORR-LRPT | LRPT drift_error vs 遗忘 | r=+0.18（p=0.65） |

## E-NEG（已关闭路线，避免重复）

| ID | 内容 | 结果 |
| --- | --- | --- |
| E-NEG-RANK16 | LRPT rank16 | 未优于 rank10 |
| E-NEG-DAMP | damping 0.5/0.9 | 双数据集不同时改善 |
| E-NEG-RAW | raw-space prototype | INR task1 52.74，灾难性 |
| E-NEG-CLASSMEAN | class-mean transport | 过拟合当前类均值 |
| E-NEG-JVP | LoRA-aware JVP | INR 过、C100 不过 |
| E-NEG-CLASSWISE | class-wise JVP sensitivity | 灾难性遗忘，已回退 |
| E-NEG-OPERATOR | operator stability loss | drift 压 10-30 倍但 F 只改善 0.12 |
| E-NEG-ADAPTIVE | generic adaptive | 79.13/82.86/6.84，弱于普通 affine |
| E-NEG-CONSIST | EMA prototype consistency | 78.41/81.91/7.45 |
