# Section Blueprints

## Title

Online Gauge-Aligned Cumulative Shared-A LoRA for Rehearsal-Free Class-Incremental Learning（暂定）

## Abstract

Spine 压缩：问题（无回放 CIL 中 LoRA 状态线性增长 + 共享 A 漂移）→ 方法（累计折叠 + canonical QR + 闭式 gauge alignment + 单模型）→ 结果（状态减 85.8%/87.9%、INR 小幅显著代价 / C100 无显著差异、CUB +8.04、诊断待 pre-save 重跑）→ 边界（外部基线未复现）。

## Introduction

1. CIL 无回放设定与 LoRA 低秩适配（CIT-01）。
2. 现有方法三类：适配器隔离（InfLoRA/C-LoRA）、原型分类（CL-LoRA/RanPAC）、漂移补偿（LDC/E2-LoRA）；共同点：不利用 Shared-A 的累计代数结构或状态随 T 增长。
3. Gap：如何在 A 持续更新时保持历史算子且状态 O(1)。
4. 贡献：累计折叠（C1）、gauge alignment（C2）、O(1) 状态（C3）、单模型（C4）、实验证据（C5-C10）。

## Method

- 3.1 问题设定与 Shared-A 符号。
- 3.2 累计等价（C1，含 Phase A 误差）。
- 3.3 Canonical 化与 gauge alignment（C2，闭式解 + 诊断定义）。
- 3.4 训练/推理流程与 v1 口径一致性。
- 3.5 持久状态与参数量（C3/C4）。

## Experiments

- 4.1 协议：数据集/划分/优化/指标/审计（§3.7 骨架）。
- 4.2 主结果：单 seed（E-INR-GAUGE/E-C100-GAUGE vs 基线）与多 seed 配对（C5）。
- 4.3 效率：参数-任务曲线 + FLOPs/吞吐/显存（C7、E-PARAMS、E-EFF）。
- 4.4 任务长度（C8、E-TL）。
- 4.5 额外数据集 CUB（C6、E-CUB）。
- 4.6 消融：cumulative-only vs gauge、LRPT 有无、EXP-009；诊断-遗忘相关性（C9/C10）。

## Discussion

- 机制解释：A 保持行空间（诊断 ~1e-8）→ gauge 精确保持历史算子；CUB 上 v1 重归一化损害更大。
- 与 EXP-009 的关系：INR Final/AvgAcc 小幅显著下降（-0.65/-0.52），C100 无显著差异，Forgetting 不变；贡献是压缩而非精度提升（除 CUB 单 seed）。
- 与相关工作的边界（CIT 表）：不主张"首个 prototype 补偿"或"全面优于 SD-LoRA"。

## Limitations

- 外部强基线未复现；CUB 单 seed；T=40 遗忘高；诊断非遗忘预测器；单模型但训练仍需每任务临时 B（峰值显存未计训练）。

## Conclusion

O(1) 状态 + 量化到 0.5 个点以内的小幅精度代价（INR）；主张以状态效率为核心，不宣称统计等价。
