# Confirmed Research Spine

## Research question

在无回放（rehearsal-free）类增量学习（CIL）中，持续更新的共享下投影 LoRA（Shared-A）如何在不保存逐任务上投影 B 的前提下，保持历史有效算子并避免旧类原型漂移，同时把持久状态压缩为 O(1)？

## Gap

- SD-LoRA / SA-LoRA 的逐任务 B bank 持久状态随任务数线性增长；SA-LoRA 冻结 A 牺牲容量，持续训练 A 则旧算子随 A 变化漂移（EXP-009 的 Task 6 回落即此现象）。
- 现有原型漂移补偿（LDC 学习通用网络、CL-LoRA 双适配器、RanPAC 冻结骨干）不利用 Shared-A 的代数结构，或引入额外网络/任务路由。
- 需要：利用 LoRA 分解的 gauge 结构做闭式累计 + 对齐，单模型推理、无回放、无额外持久网络。

## Core claim

在 Shared-A 持续训练设定下，历史 LoRA 分支可精确折叠为单一累计上投影（Phase A 等价误差 <1e-5），任务结束时用闭式 gauge alignment（`H_aligned = H_old Q_old^T Q_new`）保持历史有效算子；最终持久 LoRA 状态与任务数无关（371,040 = SD-LoRA 的 10.07%）。相对上一代方法（EXP-009，2,027,530 参数），seed 正确配对后 ImageNet-R Final/AvgAcc 有小幅显著下降（-0.65/-0.52），CIFAR-100 无显著差异，Forgetting 无显著变化；在细粒度 CUB-200 上显著更优（Final +8.04，单 seed）。

## Method summary

1. 累计折叠：`H = sum_i s_i B_i/(||A|| ||B_i||)`，固定 A 时 bank 前向 ≡ `H(Ax)`。
2. Canonical 化：`A^T = Q R`，有效算子 `H_raw A = (H_raw R^T) Q^T = H Q^T`。
3. Gauge alignment：任务保存时 `H_aligned = H_old (Q_old^T Q_new)`（闭式最小二乘），并记录投影残差/基底旋转/算子保持误差。
4. 训练/推理：历史支路固定（与 v1 训练口径一致），当前支路 `s B(Ax)`；保存后删除 B；推理为单 canonical 模型。

## Acceptance / verification (from method_revision_sd.md)

- 最低线：INR Final ≥78.76、C100 Final ≥86.89、含 prototype 状态相对 SD-LoRA 减 ≥80%、单模型推理、恢复训练无需历史 B bank。
- 实测：INR seed1995 79.06、C100 seed1993 87.70、状态减 85.77%/87.85%；多 seed 相对 EXP-009：INR 小幅显著下降，C100 无显著差异，Forgetting 无显著变化。
- 论文 §11 数据齐备度：多 seed（4×2）、任务长度（T=5/10/20/40）、额外数据集（CUB-200）、效率测量（FLOPs/吞吐/显存）、一致性审计、相关性分析。

## Status

- 单 seed 与多 seed 实验全部完成；论文初稿进行中（2026-08-07）。
- 未完成/限制：外部强基线（InfLoRA/CL-LoRA/LoRA-DRS/DGS）未复现；ImageNet-A/DomainNet 未跑；CUB 仅单 seed。
