# 实验记录（experiment_sd.md）

> 每个实验一行/一节；实验前必须通读本文件，避免重复。

## 模板

```text
## EXP-XXX 实验标题
- 日期：YYYY-MM-DD
- 状态：计划中 / 运行中 / 完成 / 失败
- 目标/假设：……
- 改动：文件 + commit
- 配置与命令：……
- 理论依据/文献：……
- 结果：final Top1 / AvgAcc / Forgetting / 参数量
- 分析：……
- 下一步：……
```

## EXP-000 基线确认

- 日期：2026-08-04
- 状态：完成
- 内容：使用当前代码重跑 SD-LoRA 官方配置（proto_baseline 配置即原始 SD-LoRA，仅改了 batch 与 seed 环境）。
- 结果：
  - CIFAR-100 seed1993：final Top1=86.89，AvgAcc=91.44，Forgetting=5.58。
  - ImageNet-R seed1995：final Top1=78.76，AvgAcc=83.13，Forgetting=5.61。
  - LoRA 参数量：约 3.686M（10 任务 × 24 对 × 2 × 768×10）。
- 结论：以上作为后续所有实验的对照基线。

## 先前已完成的相关实验（避免重复，详见 experiment.md）

- E001 Task-Prototype Routed SD-LoRA：路由 Top1 46.4%（INR）/54.0%（C100），global 精度低于基线。
- E002 Class-Prototype 离线复评：类别原型把路由 Top1 提升到 67.3%/82.0%，但 global 分类精度没有相应提升。
- E003-A Class-Prototype 完整训练（INR）：class Top2/task-head joint=67.34，oracle/task-head=92.01，不优于基线。
- E003-B Class-Prototype 完整训练（C100）：Task 8 中途被 SIGTERM 终止，未完成。
- E004 归一化消融：对角标准化对 C100 路由有增益（81.96→83.16），对 INR 增益很小；完整白化有秩不足风险。
- E005 SD-LoRA 聚合为 10-task router：hard route 需要 >85.6% 路由精度才能超过基线，SD-LoRA 更适合 Top-k 候选。
- 历史 K-CMS 结果（同一代码）：C100 最佳 final Top1=86.62（k4_anchor）、INR 最佳 final Top1=78.59（k4_bal002），均略低于基线，但 LoRA 参数减少 60%（4 聚类 × rank10 vs 10 任务 × rank10）。

## EXP-001 K-CMS scale 加权方向合并（scale_weighted）

- 日期：2026-08-04
- 状态：运行中
- 目标/假设：K-CMS 合并时当前只按任务数量加权 delta 方向（`separate`），忽略了每个任务学习到的 LoRA scale；而 SD-LoRA 的设计正是“幅度与方向解耦”。假设用 task scale 加权 delta 方向、同时保留平均幅度（cluster scale 仍为可学习/平均），能提高合并保真度，使 4 聚类版本追平或超过 SD-LoRA 基线，满足验收标准 1（参数减少 60%）。
- 改动：`backbone/k_cms_lora.py` 新增 `cms_scale_merge_mode="scale_weighted"`（方向用 scale 加权，幅度仍存于 cluster scale）；新增两个配置与队列脚本。
- 配置：`exps/k4_anchor_scale_weighted_c100.json`（C100 seed1993）、`exps/seed_1995_k4_bal002_scale_weighted_inr.json`（INR seed1995）。
- 理论依据：SD-LoRA（ICLR 2025）明确将 LoRA 分解为幅度（scale）与方向（归一化低秩增量）分别学习；合并时忽略幅度会丢失任务重要性信息。文献检索将在本实验分析后补充。
- 结果：
  - C100 seed1993：final Top1=86.29，AvgAcc=91.55，Forgetting=5.96。
  - ImageNet-R seed1995：final Top1=78.56，AvgAcc=82.86，Forgetting=5.47。
  - 对照：C100 基线 86.89 / 91.44 / 5.58，历史 separate k4_anchor 86.62 / 91.57 / 5.62；INR 基线 78.76 / 83.13 / 5.61，历史 separate k4_bal002 78.59 / 82.82 / 5.66。
- 分析：scale_weighted 在两个数据集上均未达到验收标准。C100 的 final Top1 比基线和历史 separate 都低（-0.60/-0.33），INR 与历史 separate 基本持平（-0.20 vs 基线）。方向加权没有带来收益。可能原因：(1) scale 是补偿归一化的幅度，不代表任务重要性；(2) 后续任务的 low_scale 会继续学习，合并时方向的选择影响有限；(3) 真正瓶颈是固定 rank=10 聚类里任务方向随合并数量增加被压缩（rank collapse）。
- 文献对照：CT-Merging（arXiv:2607.20561）指出合并时系数幅度需要显式处理；Subspace-Boosted Model Merging（arXiv:2506.16506）证明 Task Arithmetic 类合并随专家数增加发生 rank collapse。本实验结果更支持“容量/rank 是主要瓶颈，幅度加权是次要因素”。
- 下一步：INR 结果出来后，若同样未提升，改为“提高聚类容量但保持 ≥40% 参数缩减”：k=4、cms_low_rank=15（参数 2.212M，恰为基线 60%），合并模式回退 separate。

## EXP-002 K-CMS 提高聚类容量（rank 10 → 15，参数仍减 40%）

- 日期：2026-08-05
- 状态：运行中
- 目标/假设：EXP-001 分析指向 rank collapse 是主要瓶颈；把聚类低秩记忆从 rank=10 提到 rank=15（k=4，参数 2.212M，恰为基线 3.686M 的 60%，满足 ≥40% 缩减），同时保留更多任务方向，期望 final Top1 追平/超过 SD-LoRA 基线。
- 改动：仅新增配置（无代码改动）：`exps/k4_anchor_rank15_c100.json`、`exps/seed_1995_k4_bal002_rank15_inr.json`。
- 理论依据：Subspace-Boosted Model Merging（arXiv:2506.16506）证明固定秩合并随专家数增加会发生 rank collapse，维持任务向量秩可显著提升合并效果。
- 结果（C100 已完成，INR 运行中）：
  - C100 seed1993：final Top1=85.39，AvgAcc=91.51，Forgetting=6.82。
  - 对照：SD-LoRA 基线 86.89 / 91.44 / 5.58；EXP-001 scale_weighted 86.29 / 91.55 / 5.96。
- 结果补充（INR 已完成）：
  - ImageNet-R seed1995：final Top1=78.58，AvgAcc=82.83，Forgetting=5.60。
  - 对照：SD-LoRA 基线 78.76 / 83.13 / 5.61。
- 分析：rank15 在 C100 上最终任务明显下滑（Task 9/10 为 87.54/85.39，基线 87.12/86.89），遗忘显著变差；在 INR 上与 rank10 基本持平。提高聚类 rank 没有缓解近期任务损失，可能因为：(1) 更大的 A/B 改变归一化行为，scale 需要更多训练适应；(2) 固定 4 簇本身是容量瓶颈，而不是每簇秩；(3) 任务 9/10 合并进已有簇时仍发生干扰。
- 下一步：INR 结果后，改为“保留最近任务显式 LoRA + 4 聚类”：k=4、rank=10、cms_recent_tasks=2（参数 2.212M，恰 40% 缩减），直接针对 final 任务下滑问题。

## EXP-003 K-CMS 保留最近 2 个任务显式 LoRA（参数仍减 40%）

- 日期：2026-08-05
- 状态：运行中
- 目标/假设：EXP-001/002 的共性是 final 任务（8-10）下滑；保留最近 2 个任务的显式 LoRA（不合并进簇），旧 8 个任务进入 4 个聚类。参数 = 2×368,640 + 4×368,640 = 2.212M = 基线 60%，仍满足 ≥40% 缩减。期望 final Top1 ≥ 基线。
- 改动：仅新增配置（无代码改动）：`exps/k4_anchor_recent2_c100.json`、`exps/seed_1995_k4_bal002_recent2_inr.json`。
- 理论依据：SD-LoRA/K-CMS 原始设计即“最近任务残差 + 旧任务压缩记忆”（recent tasks 保留塑性）；Subspace-Boosted 也提示合并任务越多信息损失越大，减少合并数量应降低干扰。
- 结果：待运行。
- 分析：待运行后填写。
