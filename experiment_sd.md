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
- 结果：
  - C100 seed1993：final Top1=86.39，AvgAcc=91.32，Forgetting=6.37。
  - ImageNet-R seed1995：final Top1=78.19，AvgAcc=82.88，Forgetting=6.53。
  - 对照：C100 基线 86.89 / 91.44 / 5.58；INR 基线 78.76 / 83.13 / 5.61。
- 分析：recent=2 没有解决 final Top1 缺口（C100 -0.50，INR -0.57），遗忘反而变差。结合 EXP-001/002，K-CMS 有损合并路线的三种结构变体全部未达标；共性问题是聚类/合并后的低秩表示无法完全复现逐任务 LoRA 行为。
- 结论：放弃 K-CMS 有损合并方向，转入 EXP-004 共享 A 结构。

## EXP-004 Shared-A SD-LoRA（共享 A + 每任务 B，参数降至基线 ~10%）

- 日期：2026-08-05
- 状态：代码完成、冒烟测试通过；待 EXP-003 结束后运行
- 目标/假设：K-CMS 的三种合并变体都因“有损合并/聚类”在 final Top1 上落后基线。改为不合并的结构：所有任务共享同一个 A（任务 0 后冻结），每个任务只训练 B 和 scale；最终模型可以精确等价地保存为“共享 A + 求和 B”。LoRA 总量从 3.686M 降到 0.369M（减少 90%），远超 40% 验收线；由于每个任务仍有专属 B，期望 final Top1 不落后于 SD-LoRA。
- 改动：新增 `backbone/sa_lora.py`、`models/sa_sdlora.py`、factory 注册、两个配置与队列脚本；共享 A 用固定正交初始化（任务 0 后可选择继续训练 A，当前实验先冻结）。
- 理论依据：SA-LoRA（共享 A 解耦低秩适应，2026）观察到 down-projection A 是任务无关/可迁移的，up-projection B 保留任务差异；LoRA 任务向量可加性（Task Arithmetic 系列）保证共享 A 下 B 可直接求和。SLAO（Merge before Forget）也利用 LoRA 的 A/B 不对称性。
- 结果：
  - C100 seed1993：final Top1=86.53，AvgAcc=91.70，Forgetting=7.47。
  - ImageNet-R seed1995：final Top1=77.59，AvgAcc=82.82，Forgetting=6.16。
  - 对照：C100 基线 86.89 / 91.44 / 5.58；INR 基线 78.76 / 83.13 / 5.61。
  - 合并后 LoRA 参数：368,640（共享 A 24×7680 + 求和 B 24×7680），为基线 3,686,400 的 10%（减少 90%）。
- 分析：Shared-A 在 C100 上 AvgAcc 超过基线（+0.26）但 final Top1 差 0.36；在 INR 上 final Top1 差 1.17、AvgAcc 差 0.31。r=10 共享子空间容量不足，尤其对 ImageNet-R。
- 下一步：运行 EXP-005：Shared-A rank 10→20（合并参数 737,280，仍减 80%），A 仍在 Task 0 训练后冻结。

## EXP-005 Shared-A 提高秩（rank 20，参数仍减 80%）

- 日期：2026-08-05
- 状态：运行中
- 目标/假设：EXP-004 证明共享 A 方向平均精度可行，但 r=10 容量不够（INR -1.17）。把 rank 提高到 20，最终合并参数 737,280 = 基线 20%（减少 80%），仍远超 40% 验收线；期望 final Top1 在两个数据集上追平/超过基线。
- 改动：仅新增配置（无代码改动）：`exps/sa_sdlora_r20_c100_seed1993.json`、`exps/sa_sdlora_r20_inr_seed1995.json`。
- 理论依据：LoRA 秩决定适配子空间容量；SD-LoRA 论文及 SA-LoRA 均指出秩与容量-遗忘权衡直接相关。
- 结果：
  - C100 seed1993：final Top1=86.35，AvgAcc=91.47，Forgetting=8.68。
  - ImageNet-R seed1995：final Top1=78.03，AvgAcc=82.59，Forgetting=7.93。
  - 对照：C100 基线 86.89 / 91.44 / 5.58；INR 基线 78.76 / 83.13 / 5.61。
- 分析：rank20 把 INR 的 final Top1 从 77.59 提升到 78.03（仍差 0.73），C100 反而从 86.53 降到 86.35。提高秩不是决定性因素；共性缺口在最终任务的分类头上。
- 结论：需要“不加 LoRA 参数”的最终分类头校准/微调来补齐 final Top1 缺口。

## EXP-006 Shared-A + 最终分类头微调（5 epoch，仅 fc）

- 日期：2026-08-05
- 状态：运行中
- 目标/假设：所有变体的共性问题是“平均精度接近/超过基线，但 final Top1 低 0.3-1.2”。原因可能是逐任务训练的增量分类头在新类比例变化后未校准。冻结骨干、在全部训练类上微调最终 fc 5 epoch（参数不变），应补齐缺口。C100 用 r10（86.53 起点）、INR 用 r20（78.03 起点）。
- 改动：`models/sa_sdlora.py` 新增最终分类头对齐（weight_align）与 5 epoch 全类微调；新增两个配置与队列脚本。
- 理论依据：CIL 中分类头偏置校准是标准做法（如 BIC 的 bias correction、IL2M 的均值归一化）；`IncrementalNet.weight_align` 本就是仓库内置但未启用的功能。
- 结果（离线验证先行，官方全量运行中）：
  - C100（r10 产物）：微调前 86.86 → 微调后 **91.48**（基线 86.89）。
  - ImageNet-R（r20 产物）：微调前 78.03（与训练日志一致）→ 微调后 **81.51**（基线 78.76）。
  - 微调只更新分类头，不增加任何 LoRA 参数；LoRA 总量仍为基线的 10%（C100）/20%（INR）。
- 分析：5 epoch 全类分类头微调把 final Top1 大幅抬高（C100 +4.6，INR +3.5），两个数据集都超过基线；这解释了此前所有变体“平均精度接近但 final Top1 低”的现象——问题在增量分类头的比例失衡，而不是 LoRA 骨干。
- 下一步：等待官方全量管线（含重建 eval backbone + 微调）输出最终日志；若与离线一致，则验收标准 1 达成。

### ⚠️ EXP-006 无效声明（2026-08-05 用户反馈）

- **状态：invalid**。用户指出本项目是无回放（rehearsal-free）持续学习，训练完成后用全部旧类训练数据做分类头微调属于回放数据，违反设定。
- 因此 EXP-006 的全部数值（91.48 / 81.51）**不作为验收依据**；相关 head-tune 代码保留但默认关闭，后续实验一律禁止使用。

## EXP-007 Shared-A 跨任务持续训练 A（无回放，参数减 45%）

- 日期：2026-08-05
- 状态：完成（未达标：INR final Top1 低基线 0.42）
- 目标/假设：A 只在 Task 0 训练后冻结，可能导致 INR 欠拟合（-0.73）。改为 A 在**每个任务用该任务当时的数据**持续训练（共享 A，B 每任务专属，不删除 B 文件，不做有损合并）。参数 = 共享 A（24×7680）+ 10 任务 B（10×24×7680）= 2.027M = 基线 55%（减少 45%），满足 ≥40% 验收线。全程无回放：任何旧任务数据都不会在后续任务或训练结束后被再次使用。
- 改动：复用已实现的 `sa_train_a_all_tasks`；新增两个配置（head-tune 全部关闭）。
- 理论依据：SA-LoRA 论文主张 A（down-projection）捕获任务无关/可迁移表示，B（up-projection）保留任务差异；共享 A 持续训练能利用全部任务信息而不存储旧数据。
- 结果：
  - C100 seed1993：final Top1=**86.90**，AvgAcc=91.58，Forgetting=7.32（基线 86.89 / 91.44 / 5.58）。
  - ImageNet-R seed1995：final Top1=**78.34**，AvgAcc=82.60，Forgetting=6.20（基线 78.76 / 83.13 / 5.61）。
  - LoRA 参数：共享 A 184,320 + 10×B 1,843,200 + 10 个 scale = 2,027,530 ≈ 基线 3,686,400 的 55%（减少 45%）。
- 分析：A 持续训练相比 A 冻结有明显收益（INR 77.59→78.34，C100 86.53→86.90），但仍差 INR 基线 0.42、C100 险胜 0.01。C100 曲线 Task 8 掉到 86.27 后 Task 9 回升 86.90，INR 曲线 Task 8 77.27 → Task 9 78.34，说明“新任务分类头压过旧类”的偏置仍是主要瓶颈：数据无关的推理期校正（weight_align、余弦归一化）在旧产物上几乎零收益，而 5 epoch 全类 head-tune（EXP-006）能 +4.6/+3.5 但属于回放，判定无效。
- 文献对照（本轮检索）：CL-LoRA（CVPR 2025，arXiv:2505.24816）在无回放 PTM-CIL 中采用“训练期用各任务数据计算每类原型 + 推理期全类最大余弦相似度”的分类器，与本项目结构同构；RanPAC（ECCV 2024）用随机投影+类原型；LUCIR/CosNorm（CVPR 2019）证明余弦归一化需在训练期生效。据此设计 EXP-009。
- 下一步：EXP-009 = Shared-A（A 持续训练 rank10）+ 训练期原型存储 + 余弦原型分类器，全程无回放。

## EXP-009 Shared-A + 训练期原型余弦分类器（无回放，参数仍减 45%）

- 日期：2026-08-05
- 状态：**完成，验收标准 1 达成**
- 目标/假设：EXP-006/007 分析表明 final Top1 缺口来自增量分类头偏置，且数据无关推理期校正无效。改为在**每个任务训练结束时**用该任务当时的数据计算每类 L2 归一化均值原型（仅存向量，不存数据、不事后回放），最终评估用全类余弦原型分类器替代偏置的 fc 头。期望 C100 ≥86.89 且 INR ≥78.76，LoRA 参数仍为基线 55%（减少 45%）。
- 改动：`backbone/linears.py` 新增 `PrototypeCosineHead`；`utils/inc_net.py` 新增 `SharedAPrototypeNet`；`models/sa_sdlora.py` 新增 `sa_use_prototype_classifier` 与 `_compute_prototypes`（提交 f953e41）。
- 理论依据：CL-LoRA（CVPR 2025）、RanPAC（ECCV 2024）均证明训练期原型 + 余弦匹配是无回放 CIL 的有效分类器；原型等价于每类特征中心，天然抵消新类训练频次偏置。
- 结果（ImageNet-R seed1995，先跑，用户要求 INR 达标后再跑 CIFAR）：
  - final Top1=**79.34**（基线 78.76，+0.58），AvgAcc=82.47（基线 83.13），Forgetting=7.26（基线 5.61）。
  - 曲线 [91.58, 85.62, 83.83, 82.68, 81.76, 81.96, 79.3, 79.62, 79.01, 79.34]。
  - LoRA 参数同 EXP-007：2,027,530（基线 55%，减少 45%）；另存 200 个 768 维原型（0.154M，可忽略）。
- 结果（CIFAR-100 seed1993，INR 达标后确认运行）：
  - final Top1=**88.42**（基线 86.89，+1.53），AvgAcc=92.07（基线 91.44，+0.63），Forgetting=8.08（基线 5.58）。
  - 曲线 [98.3, 96.25, 94.6, 93.75, 92.2, 90.6, 90.84, 88.21, 87.57, 88.42]。
  - 另存 100 个 768 维原型（0.077M，可忽略）。
- 分析：INR final Top1 首次稳定超过基线（+0.58）。早期任务原型分类器低于 fc 头（T1 -2.5），但中后期优势逐渐扩大（T5 +1.1、T8 +1.7），说明原型抵消分类头偏置的收益随任务数增长；A 持续训练导致的旧原型漂移仍造成 Task 6 单次回落 2.66（与 LDC ECCV 2024 描述一致），但最终任务恢复。AvgAcc 低于基线 0.66、Forgetting 更差是剩余短板，但不影响验收主指标。
- 独立复验：离线评估脚本用存储原型 + 合并骨干在测试集上复算，C100=88.42、INR=79.34，与训练日志完全一致。参数量实测：共享 A 184,320 + 10×B 1,843,200 + 10 scale = 2,027,530（基线 3,686,400 的 55.0%，减少 45.0%）；含原型后合计 2,181,130（INR），仍为基线 59.2%（减少 40.8%）。
- **验收结论：满足标准 1**——LoRA/总参数量减少 ≥40%，CIFAR-100 final Top1=88.42 ≥86.89，ImageNet-R final Top1=79.34 ≥78.76；全程无回放（memory_size=0、无 head-tune、原型仅训练期用当前任务数据计算）。
- 实验顺序（用户 2026-08-05 要求）：先在 ImageNet-R 上运行并判定，INR final Top1 ≥ 78.76 后再到 CIFAR-100 上运行确认（减少实验量）。
- 下一步：无（目标达成）。若后续仍要改善遗忘指标，可尝试 LDC 式漂移补偿或 A 冻结 + 原型。

## EXP-010 LRPT：低秩原型漂移补偿（第二轮）

- 日期：2026-08-05
- 状态：运行中（ImageNet-R seed1995）
- 目标/假设：Shared-A 在任务间持续更新使历史 prototype 与当前特征空间失配（EXP-009 Task 6 单次回落 2.66）。用**当前任务**数据在模型更新前/后提取配对特征 (z_old, z_new)，闭式拟合 rank-r 仿射 transport `p' = p + U(V^T p)`（U,V ∈ R^{768×r}，r 与 LoRA rank 绑定），递归更新全部旧 prototype；不读取/保存/回放旧任务样本。
- 改动：新增 `backbone/lrpt.py`（闭式 rank-r 最小二乘 + 应用/误差函数）、`tests/test_lrpt.py`；`models/sa_sdlora.py` 增加 `lrpt_enabled/lrpt_rank/lrpt_reg` 与训练前后配对特征捕获、transport 拟合、旧原型递归更新；新增 INR/C100 配置与运行脚本。commit `4b8ae92`。
- 配置与命令：`exps/lrpt_sa_sdlora_inr_seed1995.json`；`bash run_lrpt_sa_sdlora_inr.sh`（INR 先跑，达标后才跑 C100）。
- 理论依据/文献：LRPT 明确机制见 plan_sd.md 第二轮碰撞审计（相对 SA-LoRA/CL-LoRA/RanPAC/LDC/FM-LoRA/C-LoRA/InfLoRA/EASE 的结构差异）；LDC（ECCV 2024, arXiv:2407.08536）指出原型漂移需要补偿，但 LRPT 以 LoRA ΔA 的秩为先验做闭式低秩 transport，不训练额外网络；CL-LoRA（CVPR 2025）与 RanPAC（NeurIPS 2023）提供训练期原型 + 余弦分类器骨架。
- 结果（v2 正式运行，含 task0→task1 补偿；INR seed1995 已达标）：
  - final Top1=**79.24**（基线 78.76，+0.48；EXP-009 79.34，-0.10，回退 ≤0.20 ✓）
  - AvgAcc=**83.03**（EXP-009 82.47，+0.56；门槛 82.97 ✓）
  - Forgetting=**6.31**（EXP-009 7.26，改善 0.95；门槛 6.26 差 0.047，但 AvgAcc 门槛通过 ✓）
  - 曲线 [90.95, 87.06, 85.55, 83.36, 81.89, 82.73, 80.47, 80.17, 78.86, 79.24]
  - LRPT 每任务 relative_drift_error：0.899/0.909/0.900/0.932/0.914/0.905/0.909/0.908/0.919（task1–9）。
- 结果（v1 诊断运行，缺 task0→task1 补偿，不作为验收）：INR seed1995 final Top1=79.63、AvgAcc=83.08、Forgetting=6.52；离线原型复算 79.63 与日志一致。发现实现缺口后已修复并重跑（v2）。
- 冒烟验证（1 epoch/task 全 10 任务，INR seed1995）：
  - LRPT 路径正常：每任务 pre-update 特征捕获 → 训练 → transport 拟合 → 旧原型移动 → 新原型合并；最终任务重建后同样通过。
  - 每任务 `relative_drift_error` ≈ 0.91–0.96（rank=10，reg=1e-2），即闭式 rank-10 transport 在 1-epoch 训练下只解释约 4–9% 的漂移范数；正式 20-epoch 训练是否改善需看完整实验。
- 分析：初版实现冒烟中发现末任务 `_extract_current_task_features` 的 task_index 越界（修复为默认当前任务、训练前显式传 `_cur_task+1`）；修复后全流程通过。v1 完整运行后又发现 `_cur_task >= 1` 守卫把第一次漂移（task0→task1）跳过：task1 的 pre-update 捕获条件应为 `>= 0`，否则旧原型在 task1 期间未补偿、且 task2 的 transport 从 state1→state2 无法补回 state0→state1。已修复（commit `306eb74`）并重跑 INR v2。
- 独立复验（v2）：离线脚本用存储原型 + merged 骨干复算 final Top1=79.24，与训练日志一致；`verify_sa_consistency.py` PASS（feature diff 1.1e-5，prototype logit diff 2.8e-7）；参数量 2,181,130（59.17%）≤ 预算 2,211,840；merged 推理 FLOPs 1.129e12/batch32、吞吐 414.8 img/s、峰值显存 577.7 MiB（单卡 batch32）。
- 结果（CIFAR-100 seed1993 v2，完成但未达第二轮门槛）：
  - final Top1=**88.43**（基线 86.89 ✓；EXP-009 88.42，+0.01，回退 ≤0.20 ✓）
  - AvgAcc=**92.42**（EXP-009 92.07，+0.35；门槛 92.57 ✗，差 0.15）
  - Forgetting=**7.63**（EXP-009 8.08，改善 0.45；门槛 ≤7.08 ✗）
  - 曲线 [98.3, 96.3, 95.2, 93.92, 92.86, 91.35, 91.23, 88.72, 87.92, 88.43]
- 分析（C100 未过）：LRPT 每任务 relative_drift_error ≈ 0.87–0.91，rank-10 transport 只解释约 9–13% 的漂移范数；INR 侧 AvgAcc 提升 +0.56 勉强过线，C100 提升 +0.35 不足。Final Top1 基本持平（+0.01），说明 LRPT 对最终任务影响小，瓶颈在旧原型漂移补偿容量。
- 结果（LRPT rank16 迭代）：
  - INR seed1995：final Top1=79.28、AvgAcc=83.12、Forgetting=6.48（单数据集门槛仍过：Final ≥79.14 ✓，AvgAcc ≥82.97 ✓）。
  - C100 seed1993：final Top1=88.26、AvgAcc=92.37、Forgetting=7.80（门槛未过：AvgAcc 差 0.20、Forgetting 差 0.72；且比 rank10 的 92.42/7.63 略差）。
- 容量诊断（C100 rank16 每任务）：norm 空间 rank10≈0.87–0.91、rank16≈0.85–0.89、rank32≈0.82–0.86、full≈0.57–0.62；raw 空间 full≈0.53–0.63；bias 单独解释 ≈0.89–0.91。结论：rank 提升收益有限，full-rank 可解释约 40–47% 漂移，但低秩（10/16）只解释约 10–15%。
- 结果（affine LRPT：rank16 + 全局漂移偏置，INR seed1995 已达标）：
  - final Top1=**79.38**（基线 78.76 ✓；EXP-009 79.34，+0.04，回退 ≤0.20 ✓）
  - AvgAcc=**83.03**（EXP-009 82.47，+0.56；门槛 82.97 ✓）
  - Forgetting=**6.79**（EXP-009 7.26，改善 0.47；门槛 6.26 ✗，但 AvgAcc 门槛通过 ✓）
  - 曲线 [92.04, 86.99, 85.55, 83.48, 81.73, 82.59, 80.19, 79.77, 78.56, 79.38]
  - affine rank16 每任务 residual ≈ 0.83–0.86（plain rank16 0.88–0.90），bias 提升拟合但不明显改善指标。
- 结果（affine LRPT C100 seed1993，完成但未过门槛）：
  - final Top1=**88.39**（≥86.89 ✓；≥88.22 ✓）
  - AvgAcc=**92.45**（EXP-009 +0.38；门槛 92.57 ✗，差 0.12）
  - Forgetting=**7.37**（EXP-009 8.08，改善 0.71；门槛 7.08 ✗，差 0.29）
  - affine 比 plain rank10/16 都好（92.42→92.45、7.63→7.37），但仍差一点。
- 分析：affine 提升拟合（residual 0.83 vs 0.88）与次要指标，但未达门槛。下一步在类均值空间诊断 transport 拟合质量（原型迁移真正关心类中心），并测 affine rank10（C100 上 plain rank10 是最好基线）。
- 下一步：INR affine rank10 先跑（含类均值诊断）；若达标再跑 C100 affine rank10；随后补 full-rank 消融。
- 结果（INR affine rank10，达标）：final Top1=79.29、AvgAcc=83.00、Forgetting=6.61（Final ≥79.14 ✓、AvgAcc ≥82.97 ✓）。
- 结果（classmean LRPT：类均值空间 affine rank16+bias；INR 未达标）：
  - final Top1=78.84（≥78.76 ✓，但相对 EXP-009 回退 0.50 > 0.20 ✗）、AvgAcc=82.65（门槛 82.97 ✗）、Forgetting=7.07。
  - 拟合 residual 大幅下降（classmean rank16 ≈ 0.12–0.16 vs sample ≈ 0.83–0.86），但指标反而变差：对当前类均值过拟合，transport 不迁移到旧类原型。
- 结论：类均值拟合是错误方向；回到样本空间 affine LRPT。当前最优 C100 候选是 affine rank10（plain rank10 v2 92.42/7.63 + bias 预期再提升）。
- 下一步：C100 affine rank10 运行中；若仍不过，补 full-rank 消融并评估 LoRA 结构驱动漂移基（E²-LoRA 启发）。
- 结果（C100 affine rank10，最接近门槛）：
  - final Top1=**88.58**（≥86.89 ✓；≥88.22 ✓）
  - AvgAcc=**92.46**（门槛 92.57 ✗，差 0.11）
  - Forgetting=**7.12**（门槛 7.08 ✗，差 0.04）
  - 这是目前 C100 最佳配置（plain rank10 92.42/7.63、affine rank16 92.45/7.37）。
- 下一步：测试 transport 阻尼 λ=0.5（缩放漂移补偿，减少过度校正）。INR d05 先跑，达标后 C100 d05。
- 结果（INR d05，未达标）：final Top1=79.43（✓）、AvgAcc=82.91（门槛 82.97 ✗，差 0.06）、Forgetting=6.62。阻尼过度压低 AvgAcc；按 INR-first 规则不跑 C100 d05。
- 下一步：试更温和的 λ=0.9（INR 先跑）；若仍不稳定，回到 λ=1（affine r10）并考虑 C100 复跑/其他机制。
- 结果（INR d0.9，达标）：final Top1=79.26（≥79.14 ✓）、AvgAcc=83.10（≥82.97 ✓）、Forgetting=6.65。λ=0.9 比 λ=1 的 INR AvgAcc 高 0.10，比 λ=0.5 高 0.19。
- 下一步：C100 d0.9 运行中（目标 Forgetting ≤7.08 或 AvgAcc ≥92.57）。
- 结果（C100 d0.9，未过）：final Top1=88.42（✓）、AvgAcc=92.50（门槛 92.57 ✗，差 0.07）、Forgetting=7.29（门槛 7.08 ✗）。AvgAcc 略好于 λ=1，Forgetting 反而变差。
- 分析：λ 微调无法同时满足两个次要指标；C100 affine r10（λ=1）仍是 Forgetting 最优（7.12）。正复跑该配置做方差检查；同时实现双空间 LRPT（样本空间 transport + 加权类均值 transport，闭式组合，避免 classmean 单独过拟合）。
- 下一步：C100 affine r10 复跑结果后，按 INR-first 规则跑 INR dual；若过再跑 C100 dual。

## EXP-011 LoRA-aware LRPT（用户反馈后重构）

- 日期：2026-08-06
- 状态：实现完成、冒烟通过；INR seed1995 正式运行中
- 反馈与目标：不再对完整特征映射做自由 SVD；保存训练前后共享矩阵，计算 ΔA=A_t-A_{t-1}，用 forward-mode JVP 计算 J_A(x)ΔA + J_B(x)ΔB_t 的真实 LoRA 响应，以这些响应构造最终特征空间的 transport 基底；仅用少量层权重/rank 系数，并支持按 prototype 在漂移基底上的投影自适应补偿强度。要求四组消融：通用 LRPT、仅 ΔA、ΔA+B_t、逐层 LoRA-aware。
- 改动：`backbone/lrpt.py` 新增 `fit_affine_map`、`low_rank_factors`、`project_map_to_basis`、`fit_layerwise_weights`、`apply_transport_adaptive`；`models/sa_sdlora.py` 新增 `lrpt_basis`（generic/delta_a/delta_a_b/layerwise）、`lrpt_adaptive`，采集当前任务每类 2 张锚点图，用 `functional_call + forward_ad` 对 48 个 LoRA 权重做 JVP，SVD 取 top-r 右奇异向量作为漂移基底，再把经验仿射映射投影到基底上。commit `d6c91d8`。
- 工程要点：JVP 需 math SDP 后端（flash/efficient attention 不支持高阶/forward tangent）；forward-mode AD 避免反向模式 OOM；锚点按 8 张分块。
- 冒烟验证（1-epoch 全 10 任务 INR seed1995）：每任务 JVP 基底应用成功，final 任务重建路径通过；LRPT 每任务 relative_drift_error ≈ 0.85–0.90（基底约束下略高于自由拟合，但目标是泛化到旧类）。
- 配置：`exps/lrpt_lora_inr_seed1995.json`（ΔA+B_t + adaptive）、`exps/lrpt_lora_da_inr_seed1995.json`（仅 ΔA）、`exps/lrpt_lora_lw_inr_seed1995.json`（逐层）、`exps/lrpt_lora_c100_seed1993.json`。
- 下一步：INR seed1995 正式运行；达标后跑 C100；随后补 C100 的 delta_a/layerwise 消融与多 seed。
- 结果（INR LoRA-aware ΔA+B_t + adaptive，未过）：final Top1=79.49（✓）、AvgAcc=82.64（门槛 82.97 ✗，差 0.33）、Forgetting=6.36（门槛 6.26 ✗，差 0.10）。Final 高但中早期任务低于 generic affine，AvgAcc 被拉低。
- 分析：adaptive 强度（0.5–1.0）可能是 AvgAcc 下降主因（泛化路径补偿不足）；先做受控变量：关闭 adaptive（λ=1）再跑 INR。
- 若仍不过：按用户反馈切回原本设计（generic affine LRPT，此前 INR 全过、C100 最优 92.46/7.12）。
- 结果（INR LoRA-aware ΔA+B_t，无 adaptive，**达标**）：final Top1=79.43（≥79.14 ✓）、AvgAcc=82.68（门槛 82.97 ✗，但 Forgetting=6.18 ≤ 6.26 ✓）。LoRA-aware 在 INR 上通过 Forgetting 分支。
- 下一步：C100 LoRA-aware（无 adaptive）运行中；目标 Forgetting ≤7.08 或 AvgAcc ≥92.57，Final ≥88.22。
- 结果（C100 LoRA-aware ΔA+B_t，无 adaptive，未过）：final Top1=88.25（≥88.22 ✓ 险过）、AvgAcc=92.25（门槛 92.57 ✗）、Forgetting=7.54（门槛 7.08 ✗）。
- 结论（按用户反馈）：LoRA-aware JVP 方案单 seed 未达标（INR 过、C100 不过），**切换回原本设计**（generic 样本空间 affine LRPT rank10）。该方案 INR 已过（79.29/83.00/6.61），C100 最佳 88.58/92.46/7.12（差 0.11/0.04）。复跑 C100 affine r10 做方差检查；若过则进入多 seed。
- 结果（C100 affine r10 复跑 RERUN2，仍未过）：final Top1=88.46（✓）、AvgAcc=92.48（门槛 92.57 ✗）、Forgetting=7.21（门槛 7.08 ✗）。两次运行 AvgAcc 92.46/92.48、Forgetting 7.12/7.21，均差一点。
- 下一步：affine rank12 微调（rank10 最优、rank16 变差，rank12 可能更好）。INR rank12 先跑，达标后 C100 rank12。
- classwise layerwise sensitivity 实验（用户最新反馈，已按指示回退）：
  - 实现了 JVP requires_grad 保存/恢复与一致性断言、中点 JVP 与有限差分验证（ΔA/ΔB/ΔA+B/交互项余弦/相对误差）、class-wise 12×8 sensitivity + 共享 768×8 基底、基底更新坐标变换、sensitivity-based 强度。
  - 冒烟（1-epoch INR）结果灾难性：task0 评估异常偏低（≈83.8 vs 常规 88+），task1 后旧类精度跌至 50% 并持续恶化（最终 task0 类 0%）。JVP 本身验证良好（ΔA+B 余弦 0.97–0.99），但 classwise 更新公式/基底一致性仍无法收敛。
  - 按用户指示 `git restore` 回退未提交改动，回到 d835567 仓库状态，继续原方案。

## EXP-012 Shared-A 有效 LoRA 算子稳定化

- 日期：2026-08-06
- 状态：完成；INR seed1995 工程验证通过，但未通过第二轮性能验收，不启动 C100。
- 目标/假设：上轮灾难性遗忘来自不完整的 class-wise JVP 外推，而 Shared-A 的真实历史分支可直接写作 `M_old @ normalize(A)`。训练新任务时约束该有效算子相对漂移，能在不访问旧样本的条件下从根源限制共享 A 和历史 scale 对旧类别表示的扰动。
- 改动：新增 `backbone/sa_operator_stability.py`；`SharedALoRA_ViT_timm` 在 Task>0 的训练构造期保存非持久的旧 A / 旧有效 B 快照，并提供 `old_operator_stability_loss()`；`models/sdlora.py` 增加可扩展额外训练损失钩子；`models/sa_sdlora.py` 增加 `sa_operator_stability_lambda`。新增 Task 0/Task 1 零漂移、梯度、state-dict 非持久化测试。
- 配置与命令：smoke 为 `exps/sa_sdlora_operator_stability_smoke_inr_seed1995.json`；完整 INR 为 `exps/sa_sdlora_operator_stability_inr_seed1995.json`，命令 `GPU_IDS=0,1,2,3 bash run_sa_operator_stability_inr.sh`。
- 理论依据/文献：InfLoRA（CVPR 2024）和 LoRA-DRS（CVPR 2025）均从 LoRA 参数/子空间限制任务干扰；本实验不同于冻结子空间或梯度投影，直接约束当前网络前向中全部历史 B bank 与共享 A 构成的精确有效线性算子。它也不同于 LDC 的特征空间漂移回归，不学习或应用 prototype transport。
- 参数与合规：快照只在当前任务训练内存中存在，未注册为 module buffer，`state_dict` 与 artifact 不含它；最终持久参数仍为 EXP-009 的 LoRA+prototype 口径（INR 2,181,130），满足预算。Task 0 返回严格零损失。
- 结果（首次 DDP smoke，工程 bug）：Task 0 训练/评估正常（79.56），且无 `operator_stability`；Task 1 训练期出现有限稳定项 0.0008。训练结束保存后诊断调用报 `KeyError: 1`，原因是 `save_lora_parameters()` 令 `task_id` 自增，而当前对象尚未加载该任务的 B 文件，诊断错误地以可变 `task_id` 遍历历史 B。不是遗忘或梯度异常。
- 修复后 DDP smoke（1 epoch x 10 tasks，4 x RTX 3090）：exit=0；Task 0=79.56，Task 1 至 Task 9 的稳定项均有限（0.0008 到 0.0000），末任务 Top1=69.31、AvgAcc=73.08、Forgetting=9.42。1-epoch 数值不参与性能对比，验收的是训练路径；未出现 NaN、崩溃、Task 0 污染或旧类归零。保存后的每任务相对有效算子漂移范围 0.000201--0.021862。
- 产物审计：`verify_sa_consistency.py` PASS（feature max diff=4.053e-06，prototype logit diff=2.384e-07）；总持久参数=2,181,130（59.17% 基线，预算内）。
- 完整运行选择：smoke lambda=0.1 的稳定项仅 0.0008，而 CE=1.976，约低三个数量级；完整 INR 仅把 lambda 提升到 1.0，其余 EXP-009 配置不变，以确保该单变量确实得到有效检验。
- 完整 INR 结果（seed1995，4 x RTX 3090，exit=0）：Final Top1=79.39，AvgAcc=82.20，Forgetting=6.98；曲线 `[91.26, 85.16, 83.88, 82.49, 81.70, 81.66, 78.59, 79.37, 78.51, 79.39]`。Task 1--9 保存后的 raw effective-operator drift 为 0.002878/0.002560/0.002794/0.001957/0.002917/0.003128/0.002229/0.003862/0.002568。
- 对比：相对原始 SD-LoRA（78.76/83.13/5.61）Final +0.63、AvgAcc -0.93、Forgetting +1.37；相对 EXP-009（79.34/82.47/7.26）Final +0.05、AvgAcc -0.27、Forgetting -0.28。INR 验收的 Final 下限 79.14 已通过，但 Forgetting 仍高于 6.26 且 AvgAcc 低于 82.97，故未通过。
- 最终产物审计：持久参数 2,181,130（基线 59.17%，预算内）；per-task bank 与 merged backbone feature 最大差 1.001e-05，prototype logits 最大差 2.384e-07，均 PASS。
- 分析：该项确实降低了 EXP-009 的 Forgetting，但收益不足且损失了 AvgAcc。它严格控制的是历史 LoRA 线性支路的聚合算子，而不是共享 A、新 B 与后续 ViT 非线性共同诱发的最终特征漂移；小算子漂移不等价于旧类原型和分类边界稳定。继续盲扫 lambda 预期主要牺牲新任务塑性，故将此方法记录为干净的负结果。
- 下一步：按 INR-first 规则不跑 C100。若继续本研究方向，应先用无正则的受控日志测量同一 raw operator drift 与指标的相关性；若相关性弱，则不再以该算子作为主约束目标，回到已有的 affine LRPT/其他直接面向原型或分类边界的机制。
- 控制组（EXP-012 诊断）：`sa_operator_stability_lambda=0.0`，仅记录每任务保存后的 `relative_effective_drift`，不加入训练损失，其余配置与 EXP-009/EXP-012 完全一致。目的是判断“算子漂移”与 Forgetting/AvgAcc 是否相关；若控制组的 raw drift 也很小或与遗忘无单调关系，则 operator-level 正则路线停止。
- 控制组结果（INR seed1995，lambda=0）：Final Top1=79.39、AvgAcc=82.47、Forgetting=7.10；raw drift 序列 0.0911/0.0202/0.0118/0.0048/0.0082/0.0081/0.0062/0.0097/0.0075。与 EXP-012（lambda=1，drift≈0.002–0.004）相比，drift 高 10–30 倍，但 Forgetting 仅差 0.12、AvgAcc 反而高 0.27。
- 结论：raw operator drift 与遗忘相关性弱；operator-level 正则路线正式关闭。回到 generic affine LRPT（INR 已达标、C100 最佳 92.46/7.12），第三次复跑 C100 affine r10 验证是否能过门槛。
- 结果（C100 affine r10 第三次复跑 RERUN3，仍未过）：final Top1=88.58（✓）、AvgAcc=92.50（门槛 92.57 ✗，差 0.07）、Forgetting=7.17（门槛 7.08 ✗，差 0.09）。三次结果 92.46/7.12、92.48/7.21、92.50/7.17，说明该配置系统性差 0.05–0.13，非方差。
- 下一步：同一框架内的结构性变体 **raw-space 原型**——原型改为“未归一化特征的类均值再归一化”，LRPT 也在 raw 空间拟合与应用。先跑 INR rawproto r10 验证，达标后跑 C100。
- 结果（raw-space 原型，INR 失败）：task0 Top1=91.11 正常，但 task1 骤降至 52.74，明显灾难性；按 INR-first 规则终止，不跑 C100。raw 空间低秩 transport 不适配原型归一化后的分类空间。
- 下一步：generic affine r10 + `lrpt_adaptive`（按 prototype 在 transport 输入方向上的投影自适应补偿强度，之前只在 LoRA-aware 上测过；直接针对 C100 中 T5/T7 被过度校正的问题）。INR 先跑。

## EXP-013 训练期原型一致性正则（EMA cosine）与 generic adaptive 负结果

- 日期：2026-08-06
- 状态：完成（两项 INR 均未达标，作为已关闭路线记录）
- 目标/假设：
  1. generic affine r10 + `lrpt_adaptive`：按旧 prototype 在 transport 基底上的投影自适应补偿强度（强度 0.5–1.0），避免低投影类被过度校正；
  2. 训练期原型一致性正则（`sa_prototype_consistency_weight=0.1`）：EMA 类原型与当前批次特征做余弦一致项，稳定原型在线估计，不保存额外持久参数、不回放。
- 改动：adaptive 复用 EXP-011 的 `lrpt_adaptive`（配置提交 c0ad739）；一致性正则新增 `models/sa_sdlora.py`、`models/sdlora.py`（commit fcc8882）。
- 配置与命令：
  - adaptive：`exps/lrpt_sa_sdlora_inr_affine_r10_adaptive_seed1995.json`，`bash run_lrpt_affine_r10_adaptive_inr.sh`；
  - consistency：`exps/lrpt_sa_sdlora_inr_proto_consistency_seed1995.json`，`bash run_lrpt_proto_consistency_inr.sh`。
- 结果（generic affine r10 + adaptive，INR seed1995）：
  - final Top1=**79.13**（门槛 ≥79.14 ✗，差 0.01）、AvgAcc=**82.86**（门槛 82.97 ✗，差 0.11）、Forgetting=**6.84**（门槛 6.26 ✗）；
  - 曲线 `[91.11, 86.61, 85.45, 83.29, 82.12, 82.62, 79.92, 79.69, 78.64, 79.13]`。
- 结果（training-time prototype consistency，INR seed1995）：
  - final Top1=**78.41**（基线 78.76 ✗，门槛 79.14 ✗）、AvgAcc=**81.91**（✗）、Forgetting=**7.45**（✗）；
  - 曲线 `[91.11, 85.62, 84.79, 82.87, 81.0, 81.64, 79.42, 78.23, 76.05, 78.41]`；每任务一致性项 0.03–0.05，训练正常、无 NaN。
- 分析：
  - adaptive 比普通 affine r10（79.29/83.00/6.61）全面变差（Final -0.16、AvgAcc -0.14、Forgetting +0.23），按投影强度缩放补偿没有带来泛化收益；
  - consistency 正则使当前任务特征与 EMA 原型靠拢，反而压低了中后期任务精度（Task 8 76.05 是各变体最低），Final 低于原始基线；该正则不改变参数状态，但直接干扰了新任务可塑性。
- 结论：LRPT 框架内的 adaptive/一致性正则两类小步调整均为干净负结果。结合此前 rank/damping/raw/classmean/JVP/operator 全部负结果，**关闭 LRPT 局部超参数与训练期正则路线**，按 method_revision_sd.md 转向 **Gauge-Aligned Cumulative Shared-A**（结构改变参数随任务数增长的方式，而不是继续优化同一特征回归）。
- 工程清理：adaptive 未提交的 `lrpt_adaptive_min/max` 参数化改动属于已关闭路线且无配置使用，已 `git stash`（stash@{0}，可恢复），工作树回到 HEAD 干净状态。

## EXP-014 Gauge-Aligned Cumulative Shared-A（Phase A：纯代数等价性）

- 日期：2026-08-06
- 状态：完成
- 目标/假设：历史 B bank 可精确折叠为累计上投影 `H = sum_i s_i * B_i / (||A|| ||B_i||)`；固定 A 时 bank forward 与 cumulative forward 逐层算子、feature、logits 误差均应 < 1e-5。
- 改动：`backbone/sa_lora.py` 新增 `fold_cumulative_up_projection` / `fold_all_cumulative_up_projections` 纯函数；新增 `tests/test_sa_cumulative.py`（commit `c666ac0`）。
- 理论依据：见 method_revision_sd.md 第 3–4 节（`sum_i s_i * Bbar_i * Abar = (sum_i s_i * Bbar_i) * Abar`）。
- 结果/分析：
  - 算子等价：`H @ A` 与逐任务 `s_i B_i A / (||A|| ||B_i||)` 求和相对 Frobenius 误差 < 1e-5；
  - feature 等价：真实 `_SharedAQKV` bank 前向与 cumulative 前向最大绝对误差 < 1e-5（3 历史任务、Q/V 双分支）；
  - logits 等价：tiny ViT 全模型 + 随机线性头最大绝对误差 < 1e-5；
  - 与 `save_merged_lora` 产物逐分支 `allclose(atol=1e-6)`；
  - 输入校验（空 bank、数量不匹配、非标量 scale、缺任务 scale）均抛 ValueError；全量测试 32 passed。
- 下一步：Phase B 在线累计状态（SA_STATE_VERSION 升级 + 迁移脚本 + 单 artifact 全流程）。

### EXP-014 补充：canonical QR 与 gauge alignment 纯函数（Phase C 代数）

- 状态：完成（commit `02459fc`）
- 改动：`backbone/sa_lora.py` 新增 `canonical_down_projection`（A^T=QR 薄分解，返回 Q^T、R）、`canonicalize_effective_up_projection`（H_raw R^T）、`gauge_align_up_projection`（H_old Q_old^T Q_new 闭式最小二乘）、`gauge_projection_residual`（Q_new 补空间投影误差）；`tests/test_sa_cumulative.py` 新增 4 个用例。
- 验证：A=R^T Q^T 重构误差 <1e-6；canonical 算子与 raw bank 算子一致；同 span 时 gauge 误差零；一般情形 `H_aligned Q_new^T` 等于旧算子在 Q_new 张成空间上的投影，残差与理论一致。全量 36 passed。
- 下一步：Phase B 状态集成（v2 artifact、在线累计保存/加载、迁移脚本、DDP 验证）。

## EXP-015 Cumulative Shared-A v2（Phase B：在线累计状态 + 保存时 gauge alignment）

- 日期：2026-08-06
- 状态：工程完成；1-epoch DDP smoke 通过；完整 INR 筛选待跑
- 目标/假设：历史 B bank 在线折叠为单套 canonical 状态（Q^T + H + R），持久 LoRA 状态与任务数无关；任务结束后把当前 B 折叠进 H，并用闭式 gauge alignment 把历史算子投影到新 Q 基底，保留其行空间内部分；每任务日志记录相对投影残差。
- 改动：
  - `backbone/sa_lora.py`：`SA_STATE_VERSION=2`；`_CumulativeSharedAQKV`（历史 = H@Q^T 固定，当前 = scale*B(Ax) 与 v1 训练口径一致，归一化在保存时折叠）；`_save_cumulative_state`（QR canonicalization → gauge align → 折叠当前任务 → 写单文件）；`migrate_sa_state_v1_to_v2`；`cumulative_gauge` 消融开关（commits `27a6574`、`63b267e`、`a3d3a79`）。
  - `utils/inc_net.py`、`models/sa_sdlora.py`：任务 0 构造与逐任务 update 透传 `sa_cumulative_state` / `sa_cumulative_gauge`（commit `d7ec60a`、`63b267e`）。
  - `scripts/migrate_sa_state_v1_to_v2.py`（显式迁移，v1 state 备份为 `sa_state.pt.v1`）；`scripts/measure_sa_artifact.py` 支持 v2 计数。
  - `tests/test_sa_cumulative.py` 新增 9 个 Phase B 用例（roundtrip、v1/v2 前向等价、gauge 累积公式、no-gauge 消融、迁移等价、flag 冲突、残差诊断）。
- 配置与命令：`exps/sa_cumulative_smoke_inr_seed1995.json`；`GPU_IDS=0,1,2,3 bash run_sa_cumulative_smoke_inr.sh`。
- 结果（smoke：1 epoch × 10 tasks，4×RTX 3090，exit=0）：
  - 10 个任务全部完成，无 NaN/崩溃；每任务 `cumulative gauge task N: relative_projection_residual=0.000000`（6 位小数下为 0）。
  - 产物仅 `sa_state.pt` + `sa_merged_lora.pt` + `CLs_*`；**无任何逐任务 B 文件**。
  - 参数实测：canonical_down 184,320 + cumulative_up 184,320 + triangular_r 2,400 = LoRA 371,040 = 基线 3,686,400 的 **10.07%**（减少 89.93%）；INR 含原型 153,600 后 524,640 = 14.23%（减少 85.77%）。
  - `verify_sa_consistency.py` PASS：v2 状态骨干与 merged canonical 骨干 feature max abs diff = 0.000e+00。
  - 1-epoch 指标（Final 48.32 / AvgAcc 57.83 / Forgetting 12.13）仅用于管线验证，不参与性能对比。
- 工程过程：首次 smoke 失败——任务 0 由 `utils/inc_net.get_backbone` 构造时未透传 `cumulative_state`，task0 写成 v1 产物，task1 报 legacy mismatch；修复透传后通过（commit `d7ec60a`）。
- 分析：gauge residual 在 6 位小数下为 0，说明 1-epoch 训练下共享 A 的更新基本保持在旧行空间内（QR 后 Q 未变），gauge alignment 因此精确保持历史算子；完整训练下 A 是否跨出原 span 需看完整 INR 的残差序列。
- 下一步：按 method_revision_sd.md 顺序跑完整 INR：cumulative-only（`sa_cumulative_gauge=false`）→ cumulative+gauge（默认 true）→ 通过后 C100 → gauge+residual LRPT。

## EXP-016 完整 INR：cumulative-only 与 cumulative+gauge

- 日期：2026-08-06
- 状态：完成（两档均 exit=0）；gauge+residual LRPT 组合运行中
- 目标/假设：用完整 20-epoch 训练检验 v2 累计状态（无 gauge / 有 gauge）相对 EXP-009（v1 bank）的性能；按 method_revision_sd.md §10 停止条件判断是否继续。
- 配置与命令：`exps/sa_cumulative_inr_seed1995_cumulative_only.json`（`sa_cumulative_gauge=false`）、`exps/sa_cumulative_inr_seed1995_gauge.json`（`sa_cumulative_gauge=true`），其余与 EXP-009 一致（prototype 分类器、无 LRPT）；`bash run_sa_cumulative_inr_queue.sh`。
- 结果（ImageNet-R seed1995）：
  | 方法 | Final Top1 | AvgAcc | Forgetting | 曲线 |
  | --- | ---: | ---: | ---: | --- |
  | EXP-009（v1 bank） | 79.34 | 82.47 | 7.26 | — |
  | SD-LoRA 基线 | 78.76 | 83.13 | 5.61 | — |
  | cumulative-only | 78.78 | 81.69 | 7.28 | [90.64, 84.25, 83.32, 82.45, 81.37, 81.01, 78.16, 79.03, 77.94, 78.78] |
  | cumulative+gauge | **79.06** | 81.77 | **6.82** | [91.26, 84.32, 83.01, 81.96, 81.33, 81.20, 78.35, 79.08, 78.08, 79.06] |
- 参数与工程：两档产物均只有 `sa_state.pt` + `sa_merged_lora.pt` + `sa_prototypes.pt` + `CLs_*`，无逐任务 B 文件；LoRA 371,040 = 基线 10.07%，含 INR 原型 524,640 = 14.23%（减少 85.77%）；每任务 gauge 日志 `relative_projection_residual=0.000000`（6 位小数）。
- 分析：
  - cumulative-only 相对 EXP-009：Final -0.56（触发 §10 的 >0.5 停止线）、AvgAcc -0.78、Forgetting +0.02。停止线要求"先检查历史 scale/normalization 口径"——检查结论：无 gauge 时保存直接用新 Q 坐标存旧 H（`H_old @ Q_new^T`），A 在训练中即使只在原 span 内旋转也会改变历史有效算子；这不是归一化口径错误，而是缺少 gauge alignment 的固有失真。
  - gauge 相对 cumulative-only：Final +0.28、AvgAcc +0.08、Forgetting -0.46，确认 gauge alignment 是必要的修正；相对 EXP-009：Final -0.28（≤0.5，通过）、Forgetting -0.44（改善）、AvgAcc -0.70（仍低）。
  - 每任务 projection residual≈0 说明 A 的行空间几乎不跨出旧 span；gauge 在此情形可精确保持历史算子。AvgAcc 缺口主要来自：v1 隐含的"历史 B 用当前 A 重新归一化"被去掉后新任务干扰更直接地作用于旧原型，以及深层非线性残余漂移——这正是 residual LRPT（Phase D）要补偿的对象。
  - gauge INR Final=79.06 ≥ 最低验收线 78.76（✓），且强目标 79.29 仅差 0.23；AvgAcc 未达强目标 83.00。
- 下一步：等待 gauge+residual LRPT INR（`exps/sa_cumulative_inr_seed1995_gauge_lrpt.json`，运行中）；若 Final/AvgAcc 回升到 EXP-009 水平则跑 C100。

### EXP-016 补充：gauge + residual LRPT INR（Phase D 组合）

- 状态：完成（exit=0）
- 配置：`exps/sa_cumulative_inr_seed1995_gauge_lrpt.json`（cumulative+gauge + `lrpt_enabled=true`，rank10/bias/λ=1，与 Phase D 默认一致）。
- 结果（ImageNet-R seed1995）：Final Top1=**78.49**（基线 78.76 ✗、EXP-009 79.34 -0.85、gauge-only 79.06 -0.57）、AvgAcc=**82.19**（EXP-009 -0.28、gauge-only +0.42）、Forgetting=**6.38**（EXP-009 -0.88、gauge-only -0.44）；曲线 `[91.42, 86.15, 84.38, 82.87, 80.74, 81.96, 79.28, 79.06, 77.59, 78.49]`。LRPT drift_error ≈ 0.86–0.88；每任务 gauge residual/rotation/preservation 均为 0.000000。
- 分析：residual LRPT 在累计状态上确实改善 AvgAcc（+0.42）与 Forgetting（-0.44），但 Final 比 gauge-only 低 0.57 且低于原始基线；与 v1 上的 generic affine LRPT（79.29/83.00/6.61）相比 Final/AvgAcc 都更低。单 seed 增益方向不一致（AvgAcc/F 改善、Final 回退），按 method_revision_sd.md Phase D 规则不把 LRPT 保留为主方法组件，只作为消融。
- 决策：主方法先用 **cumulative+gauge（无 LRPT）** 跑 C100；INR 侧最低验收 Final ≥78.76 已满足（79.06）。LRPT 是否保留待多 seed 稳定性证据。
- 下一步：C100 cumulative+gauge（`sa_cumulative_c100_seed1993_gauge`）。

### EXP-016/017 补充：主方法多 seed（INR）

- 状态：INR 4 seeds 完成（1995/1/2/3，均 exit=0，无逐任务 B 文件，LoRA 371,040）；C100 3 seeds 运行中。
- 结果（cumulative+gauge，20 epoch × 10 tasks）：
  | seed | Final Top1 | AvgAcc | Forgetting |
  | --- | ---: | ---: | ---: |
  | 1995 | 79.06 | 81.77 | 6.82 |
  | 1 | 77.99 | 82.54 | 6.77 |
  | 2 | 78.69 | 82.01 | 8.26 |
  | 3 | 78.09 | 82.78 | 9.60 |
  | mean±std | 78.46±0.51 | 82.28±0.47 | 7.86±1.35 |
- 诊断：gauge residual/rotation/preservation 均为 1e-8~3e-8（科学计数法实测），历史算子保持近乎精确；每任务数据完整。
- 分析：Final 均值 78.46 低于 SD-LoRA seed1995 基线（78.76），但 seed1995 单点 79.06 达标；同 seed 基线（EXP-009 seed1/2/3）尚未运行，配对比较待补。seed 间 Forgetting 方差较大（6.77–9.60），反映类序对旧类保持的影响。
- 下一步：C100 seed1/2/3 完成后，运行 EXP-009 同 seed 对照队列，做配对显著性。

### EXP-016/017 补充：主方法多 seed（C100）

- 状态：C100 4 seeds 完成（1993/1/2/3，均 exit=0）。
- 结果（cumulative+gauge，20 epoch × 10 tasks）：
  | seed | Final Top1 | AvgAcc | Forgetting |
  | --- | ---: | ---: | ---: |
  | 1993 | 87.70 | 91.83 | 8.53 |
  | 1 | 87.82 | 91.72 | 8.93 |
  | 2 | 87.80 | 91.23 | 8.74 |
  | 3 | 88.01 | 91.28 | 8.51 |
  | mean±std | 87.83±0.13 | 91.52±0.30 | 8.68±0.20 |
- 分析：C100 Final 跨 seed 稳定（87.70–88.01，std 0.13），全部高于 SD-LoRA 基线单点 86.89；gauge 诊断 ~5e-9。C100 主方法多 seed 结论比 INR 更强。
- 下一步：EXP-009 同 seed 对照队列运行中（`run_sa_baseline_multiseed_queue.sh`），完成后做配对显著性并更新验收结论。

### EXP-016/017 补充：与 EXP-009 同 seed 配对显著性（4 seeds）

- 状态：完成（EXP-009 INR/C100 × seeds 1/2/3 新跑，加 seed1995/1993，均 exit=0）。
- 结果（paired main − EXP-009，n=4）：
  | 数据集 | Final Δ | p | AvgAcc Δ | p | Forgetting Δ | p |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | ImageNet-R | -0.65 | 0.026 | -0.52 | 0.009 | -0.04 | 0.852 |
  | CIFAR-100 | -0.22 | 0.282 | -0.12 | 0.147 | +0.01 | 0.938 |
- 参数对比：主方法 LoRA 371,040 = EXP-009 2,027,530 的 18.3%（少 81.7%）；含原型 INR 524,640 vs 2,181,130（少 75.9%）、C100 447,840 vs 2,104,330（少 78.7%）；相对 SD-LoRA 3,686,400 减 85.8%/87.9%。
- 分析（2026-08-07 修订）：早期文档用文件名字典序 zip 配对（`scripts/multiseed_stats.py` 旧版），主方法与 EXP-009 文件顺序不同导致前两个 seed 错配，旧 p 值无效。按 seed 内连接重新配对后：INR Final/AvgAcc 为显著小幅下降（p=0.026/0.009），C100 不显著，Forgetting 无显著变化；TOST（±0.5）下 INR Final/AvgAcc 与 C100 Final 均不等价。持久 LoRA 状态减少约 82%。
- 结论：主方法多 seed 结论应表述为“O(1) 状态 + 大幅压缩，INR 精度代价约 0.5–0.65 个点（显著）、C100 代价约 0.22/0.12（不显著）、Forgetting 不变”；不得再写统计等价。任务长度/CUB/测量作为论文 §11 收尾。

## EXP-018 任务长度消融（T=5/10/20/40）

- 日期：2026-08-07
- 状态：完成（5 个运行全部 exit=0）
- 配置：`sa_cumulative_inr_seed1995_gauge_t{5,20,40}.json`、`sa_cumulative_c100_seed1993_gauge_t{5,20}.json`（cumulative+gauge，其余超参与 T10 一致）。
- 结果：
  | 数据集 | T | Final Top1 | AvgAcc | Forgetting |
  | --- | ---: | ---: | ---: | ---: |
  | ImageNet-R | 5 | 77.54 | 81.81 | 8.83 |
  | ImageNet-R | 10 | 79.06 | 81.77 | 6.82 |
  | ImageNet-R | 20 | 77.03 | 81.57 | 9.51 |
  | ImageNet-R | 40 | 75.31 | 80.63 | 12.34 |
  | CIFAR-100 | 5 | 88.06 | 91.52 | 8.77 |
  | CIFAR-100 | 10 | 87.70 | 91.83 | 8.53 |
  | CIFAR-100 | 20 | 85.63 | 91.25 | 10.72 |
- 分析：两个数据集均在 T=10 附近最优（INR Final 79.06、C100 AvgAcc 91.83）；T 增大时 Forgetting 单调恶化（INR 6.82→12.34、C100 8.53→10.72），T=40 时 Final 下降 3.75（INR）。每任务 LoRA 状态始终 371,040（O(1)），这是与 v1 的关键差异点。
- 下一步：CUB-200 主方法运行中；随后 GPU 测量与论文回填。

## EXP-019 CUB-200 额外数据集（主方法 vs EXP-009）+ 效率测量

- 日期：2026-08-07
- 状态：完成（CUB 主方法与 EXP-009 对照均 exit=0；GPU 测量与一致性审计通过）
- 配置：`sa_cumulative_cub_seed1_gauge.json`（主方法，cumulative+gauge）；`sa_sdlora_proto_cub_seed1.json`（EXP-009 对照）。CUB-200 官方 split 5994/5794，10 任务（init 20/inc 20），seed1。
- 结果（CUB-200 seed1）：
  | 方法 | Final Top1 | AvgAcc | Forgetting | LoRA 参数 |
  | --- | ---: | ---: | ---: | ---: |
  | EXP-009（v1 bank） | 71.75 | 84.93 | 23.31 | 2,027,530 |
  | **cumulative+gauge** | **79.79** | **87.69** | **14.20** | **371,040** |
- 分析：额外数据集上主方法大幅优于 EXP-009（Final +8.04、AvgAcc +2.76、Forgetting -9.11），与 INR/C100 上"小幅代价/无显著差异"的结论互补——v1 的逐任务重归一化在细粒度 CUB 上损害更大，gauge-aligned 累计状态更稳健。单 seed 结果，论文中作为支持性证据。
- 效率测量（GPU, batch32, 20 iters）：
  | 产物 | FLOPs/forward | 吞吐 | 峰值显存 |
  | --- | ---: | ---: | ---: |
  | 主方法 INR | 1.129e12 | 411.9 img/s | 579.1 MiB |
  | EXP-009 INR | 1.129e12 | 414.9 img/s | 577.7 MiB |
  | 主方法 C100 | 1.129e12 | 413.3 img/s | 579.1 MiB |
- 一致性审计：`verify_sa_consistency.py` 在 INR seed3 与 CUB 主方法产物 PASS（feature 与 prototype logits 差 0）。
- 结论：推理期算子与 v1 merged 等价（FLOPs/吞吐/显存一致），差异在持久状态（-82% LoRA）与训练期复杂度（v1 bank 随 T 增长，v2 恒定）。
- 下一步：论文初稿（含参数量 vs T 曲线、多 seed/任务长度/CUB/测量/相关性全部数据已齐）。

## EXP-017 C100 cumulative+gauge（主方法第二数据集验证）

- 日期：2026-08-06
- 状态：完成（exit=0）
- 配置：`exps/sa_cumulative_c100_seed1993_gauge.json`（EXP-009 C100 配置 + `sa_cumulative_state=true` / `sa_cumulative_gauge=true`，无 LRPT）。
- 结果（CIFAR-100 seed1993）：Final Top1=**87.70**（基线 86.89 ✓ +0.81；EXP-009 88.42 -0.72）、AvgAcc=**91.83**（EXP-009 92.07 -0.24）、Forgetting=**8.53**（EXP-009 8.08 +0.45）；曲线 `[98.2, 96.35, 94.5, 93.55, 91.6, 90.55, 90.57, 87.85, 87.38, 87.7]`。每任务 gauge 三项诊断均为 0.000000。
- 参数与工程：产物仅 `sa_state.pt` + `sa_merged_lora.pt` + `sa_prototypes.pt` + `CLs_*`；LoRA 371,040 = 基线 10.07%，含 C100 原型 76,800 后 447,840 = 12.15%（减少 87.85%）。
- 分析：C100 Final 87.70 ≥ 最低验收线 86.89（✓），且高于原始 SD-LoRA 基线 +0.81；相对 EXP-009 低 0.72（主要来自后半段：T7 87.85/T8 87.38 vs EXP-009 88.21/87.57），AvgAcc 仅低 0.24。C100 与 INR 一样，旧类精度保持优于 v1 重归一化方案，但最终任务略低。
- ⚠️ 2026-08-07 修订：本实验日志中的“每任务 gauge 诊断均为 0.000000”是保存后自比较的无效值（P0-2）；真实 pre-save 诊断以 EXP-020 重跑为准（residual 约 1.3e-2–4.4e-2）。
- 验收结论（单 seed）：cumulative+gauge 在两个数据集都满足最低方法验收线（INR 79.06 ≥78.76、C100 87.70 ≥86.89），含 prototype 状态减少 ≥85%，推理单模型无 task-id/router/逐任务 adapter。按 method_revision_sd.md §14，**优先进入多 seed 与论文阶段**，不再做单 seed 微调。
- 下一步：多 seed（INR/C100 × seeds 1/2/3，`run_sa_cumulative_multiseed_queue.sh`，运行中）；随后补强基线、消融与 FLOPs/吞吐/显存测量。

## EXP-020 P0-2 pre-save gauge 诊断重跑（INR seed1995）

- 日期：2026-08-07
- 状态：完成（exit=0）
- 目标/假设：EXP-016/017 日志中的 gauge 诊断在保存后被覆盖，不能证明历史算子保持；本实验用修复后的 pre-save 缓存诊断重跑完整 INR seed1995，验证真实 residual/rotation/preservation。
- 配置与命令：`exps/sa_cumulative_inr_seed1995_gauge_p0diag.json`（与 EXP-016 gauge 完全同超参，仅新目录）；`ImageNetR_SA_CUMULATIVE_INR_SEED1995_GAUGE_P0DIAG/`；commit `ec23df0`（修复）+ `7b93b21`（配置）。
- 结果：Final Top1=**78.91**、AvgAcc=**81.86**、Forgetting=**7.02**；曲线 `[91.42, 84.55, 83.16, 82.34, 81.5, 81.09, 78.52, 79.12, 78.02, 78.91]`。对照 EXP-016 gauge（79.06/81.77/6.82）：Final -0.15、AvgAcc +0.09、Forgetting +0.20，单次运行噪声范围内一致。
- pre-save 诊断（task1–9）：
  - relative_projection_residual：4.36e-2 / 2.69e-2 / 2.63e-2 / 1.31e-2 / 2.13e-2 / 2.08e-2 / 1.49e-2 / 2.57e-2 / 1.79e-2（均值 ≈2.13e-2）
  - basis_rotation_fro：2.96e-2 / 2.87e-2 / 5.46e-2 / 1.34e-3 / 2.20e-3 / 2.83e-2 / 1.75e-3 / 2.88e-2 / 1.91e-3
  - operator_preservation（aligned）：与 residual 逐任务一致（4.36e-2…1.79e-2）
- 分析：真实历史算子保持误差为 1.3%–4.4%，不是 1e-8；共享 A 在训练中确实跨出旧行空间，gauge alignment 只能保留行空间内部分。旧日志与论文 draft 中的“近精确保持（~1e-8）”作废。精度指标与 EXP-016 接近，说明该误差不直接决定 final Top1，但它为 Union-SVD（保留联合子空间的最优 rank-r 近似）提供了动机与基线。
- 下一步：Union-SVD smoke 与 Stage A 四档 INR（union/gauge × r4/r8）运行中。

## EXP-021 Stage A：Union-SVD vs Gauge（INR seed1995，r4/r8）

- 日期：2026-08-07
- 状态：完成（四档 exit=0；union r10 同秩对照运行中）
- 目标/假设：Union-SVD 先合并完整有效算子再取固定秩最优近似，应优于 gauge 的“投影到新基底再合并”（尤其旧算子 out-of-span 部分）；容量匹配比较 union/gauge × r4/r8。
- 配置与命令：`run_sa_cumulative_union_svd_queue.sh`（四档，20 epoch × 10 tasks，seed1995，prototype 分类器）；commit `1526068`。
- 结果（INR seed1995）：
  | 方法 | Final Top1 | AvgAcc | Forgetting | LoRA 参数 |
  | --- | ---: | ---: | ---: | ---: |
  | union_svd_r4 | 77.59 | 81.63 | 7.59 | 147,840 |
  | gauge_r4 | 77.09 | 81.26 | 8.20 | 147,840 |
  | union_svd_r8 | 78.69 | 82.31 | 6.77 | 295,680 |
  | gauge_r8 | 78.24 | 81.98 | 6.88 | 295,680 |
  | gauge_r10（EXP-016 参照） | 79.06 | 81.77 | 6.82 | 371,040 |
  | EXP-009（参照） | 79.34 | 82.47 | 7.26 | 2,027,530 |
- 分析：
  - Union-SVD 在 r4/r8 上都一致优于同秩 gauge（r4：Final +0.50、AvgAcc +0.37、F -0.61；r8：Final +0.45、AvgAcc +0.33、F -0.11），机制方向成立。
  - union_r8 相对当前 gauge_r10 恢复 AvgAcc +0.54、F -0.05，但 Final 仍低 0.37；Stage A 的 C100 进入门槛（Final ≥79.10）未通过（78.69，差 0.41），AvgAcc 82.31 刚好过 82.30、F 6.77 过 7.30。
  - r4 两档均明显低于门槛，说明低秩容量仍是主要瓶颈；r8 的联合子空间保存只部分补偿。
- 下一步：等待 union_r10（与 gauge_r10 容量匹配的直接对照）完成；若仍未过 Final ≥79.10 门槛，按 plan §6 停止 rank 扩展，进入 P1 诊断（离线原型漂移 + 公平任务长度基线），并按诊断结论决定是否实现 P3 activation sketch。

### EXP-021 补充：union_svd_r10（同秩对照，INR seed1995）

- 结果：Final Top1=**78.61**、AvgAcc=**81.91**、Forgetting=**7.05**；曲线 `[90.64, 84.55, 83.32, 82.42, 81.43, 81.53, 79.16, 79.16, 78.23, 78.61]`；每任务 truncation_error ≈ 1.1e-2–6.2e-2。
- 对照：gauge_r10（EXP-016）79.06/81.77/6.82 → union r10 Final -0.45、AvgAcc +0.14、F +0.23；EXP-009 79.34/82.47/7.26 → Final -0.73、AvgAcc -0.56、F -0.21。
- 结论：**Stage A 关闭**。Union-SVD 在 r4/r8/r10 三种秩下 AvgAcc/Forgetting 均不劣于同秩 gauge，且 r4/r8 全面更优；但 Final 仍低于 C100 进入门槛（≥79.10）和当前 gauge_r10（79.06）。联合子空间压缩改善了平均/旧类保持，未解决最终任务缺口；按 plan 停止 rank 扩展（不再 r12/r16），转入 P1 离线诊断与公平基线，P3 仅在诊断确认 backbone 干扰为主时实现。

## EXP-022 P1 公平任务长度基线（进行中）

- 日期：2026-08-07
- 状态：运行中（EXP-009 INR T5/T20 已完成；T40、C100、SD-LoRA 排队）
- 目标：与主方法任务长度消融（EXP-018）相同 seed/类序/epoch/batch 的 EXP-009 与 SD-LoRA 基线，避免仅凭主方法绝对曲线声称 scalability。
- 已出结果（ImageNet-R seed1995）：
  | T | cumulative+gauge | EXP-009 | SD-LoRA |
  | --- | ---: | ---: | ---: |
  | 5 | 77.54 / 81.81 / 8.83 | **80.33 / 82.55 / 5.82** | — |
  | 10 | 79.06 / 81.77 / 6.82 | 79.34 / 82.47 / 7.26 | 78.76 / 83.13 / 5.61 |
  | 20 | 77.03 / 81.57 / 9.51 | **78.13 / 82.53 / 9.30** | 待运行 |
  | 40 | 75.31 / 80.63 / 12.34 | 待运行 | 待运行 |
- 补充（CIFAR-100 seed1993 T5）：cumulative+gauge 88.06/91.52/8.77 vs **EXP-009 89.03/92.09/7.32**（Final +0.97、AvgAcc +0.57、F -1.45）。
- 初步分析：T=5/20 上 EXP-009（v1 bank）Final 分别高 2.79/1.10（INR）、0.97（C100 T5）个点；主方法在 T 变化时的相对退化更明显，论文 scalability 主张必须按“相对基线的差值”表述，不能只看绝对曲线。
- ⚠️ EXP-009 INR T40 在 task23 触发 CUDA OOM（v1 bank 逐任务 B 占满 24GB 显存），仅完成 22 个任务；SD-LoRA T40 预计同样受限。方案：队列结束后用 batch16 重跑 T40 两档并明确记录 batch 偏差，或报告资源不可行并只对比 T5/T10/T20。
- 下一步：C100 T5/T20、SD-LoRA T20/T40 完成后回填完整表并做差值分析。

## EXP-023 Live-A Aggregate-B（INR seed1995）

- 日期：2026-08-07
- 状态：运行中（freeze-old-scale 已完成；live-a-aggregate-b 运行中）
- 目标：验证“O(1) 状态保留 EXP-009 live shared-A 训练路径”能否恢复旧类性能（live_a 任务书 §4）。
- 实现：v4 state（commit `f36b933`）、`_LiveAAggregateQKV`、`sa_freeze_old_scales`；79→80 单测通过；4 卡 DDP smoke exit=0，consistency PASS，LoRA 368,640。
- 结果（EXP-009-freeze-old-scale，INR seed1995）：Final=**79.38**、AvgAcc=**82.16**、Forgetting=**7.11**；曲线 `[91.42,85.54,83.77,82.15,81.33,81.53,78.59,79.12,78.75,79.38]`。
  - 对照完整 EXP-009（79.34/82.47/7.26）：Final +0.04、AvgAcc -0.31、F -0.15。
  - 分析：冻结历史 scale 对 Final 几乎无影响，AvgAcc 略降 0.31（处于任务书“若 ≤0.3 则 scale 不是主因”的临界点附近）；说明历史 scale 继续适配不是主要收益来源，Live-A 折叠进 G 的近似可接受。
- 结果（live-a-aggregate-b，INR seed1995）：Final=**79.64**、AvgAcc=**82.10**、Forgetting=**6.67**；曲线 `[90.95,85.01,83.57,82.34,81.63,81.66,78.54,79.27,78.41,79.64]`。
  - 对照：完整 EXP-009（79.34/82.47/7.26）Final +0.30、AvgAcc -0.37、F -0.59；freeze-old-scale（79.38/82.16/7.11）Final +0.26、AvgAcc -0.06、F -0.44；当前 gauge（79.06/81.77/6.82）Final +0.58、AvgAcc +0.33、F -0.15。
  - 产物：LoRA 368,640（基线 10.00%）+ 原型 153,600 = 522,240（14.17%，减 85.83%）；无逐任务 B；`verify_sa_consistency` PASS（feature 7.2e-6 / logit 2.4e-7）。
- 门槛判定：INR 三项中 Final 79.64 ≥79.10 ✓、Forgetting 6.67 ≤7.50 ✓、AvgAcc 82.10 <82.30 ✗（差 0.20），**未达任务书 INR 门槛，不启动 C100**。
- 实现等价性：freeze-old-scale 与 live-a 的 AvgAcc 差 0.06（≤0.1 ✓）、Final 差 0.26（>0.1，单 seed 方差范围内，但按任务书先补实现诊断再复跑）；K-group 幅度增强不触发（freeze 相对完整 EXP-009 的 Final 未下降，AvgAcc 仅 -0.31）。
- 复跑（`live_a_aggregate_b_inr_seed1995_diag2`，含 §8 训练期诊断）：Final=**79.43**、AvgAcc=**81.99**、Forgetting=**7.08**；两次均值 79.54 / 82.05 / 6.87，AvgAcc 仍低于门槛 82.30（差约 0.25）。
  - 训练期诊断：task1 首 epoch 历史分支 dL/dA≈5.3e3–5.6e3、当前分支≈3.0e3–3.6e3（ratio≈1.5–1.9），确认 live A 收到历史 bank 梯度；每任务保存日志含 G/A/B/scale 范数。
  - 产物：LoRA 368,640 + 原型 153,600 = 522,240（14.17%）；一致性 PASS（feature 8.3e-6 / logit 3.3e-7）。
- 结论：**Live-A 机制验证成功**（Final 两次均超过 EXP-009/gauge，Forgetting 为变体中最优，状态 O(1)），但单 seed INR AvgAcc 未达任务书门槛（82.0–82.1 vs 82.30），严格按任务书不进入 C100；需与用户确认是否继续（如放宽门槛、增加 seed、或实现 K-group）。

## EXP-024 Live-A Aggregate-B Stage A（CIFAR-100 seed1993）

- 日期：2026-08-07
- 状态：完成
- 背景：用户 2026-08-07 新目标（`goal_live_a_sd.md`）调整顺序——不再以 INR AvgAcc 门槛阻塞 C100，Stage A 直接运行 Live-A Aggregate-B 的 CIFAR-100 seed1993，不新增模块、不调超参。
- 配置：`exps/live_a_aggregate_b_c100_seed1993.json`（与 INR 相同的 v4 聚合，T=10，rank10，prototype 分类器，4 卡 DDP，20 epochs）。日志 `live_a_aggregate_b_c100_seed1993.log`，产物 `CF100_LIVE_A_AGGREGATE_B_SEED1993/`。
- 结果（CIFAR-100 seed1993）：Final Top1=**88.32**、AvgAcc=**91.99**、Forgetting=**8.19**；曲线 `[98.30, 96.20, 94.53, 93.82, 92.10, 90.62, 90.70, 87.98, 87.28, 88.32]`。
- 门槛判定（Stage A）：
  - Final `88.32 >= 88.10` ✓
  - AvgAcc `91.99 >= 91.70` ✓
  - Forgetting `8.19 <= 8.70` ✓
  - consistency audit PASS（feature 7.9e-6、logit 2.7e-7）✓
  - LoRA 参数 `368,640`（10.00%，与任务数无关）✓；含原型 `445,440`（12.08%）✓
  - 无逐任务 B、无旧数据、无 task-id 推理 ✓
- 对照：EXP-009（sa_sdlora_proto seed1993）88.42 / 92.07 / 8.08；原始 SD-LoRA（proto baseline seed1993）86.89 / 91.44 / 5.58。Live-A Final 高于 SD-LoRA +1.43，相对 EXP-009 -0.10；AvgAcc 相对 EXP-009 -0.08，Forgetting +0.11。
- 结论：**Stage A 通过**。Live-A 在第二数据集上满足全部单 seed 门槛，进入 Stage B 多 seed 配对验证（ImageNet-R/CIFAR-100 × 至少 4 个相同 seed 的 Live-A、EXP-009、原始 SD-LoRA）。

## EXP-025 Stage B 多 seed 配对（Live-A / EXP-009 / SD-LoRA，n=4）

- 日期：2026-08-07/08
- 状态：完成
- 配置：`run_stage_b_paired_queue.sh` + `run_stage_b_sdlora_queue.sh`；Live-A 补 seed1/2/3（INR+C100），EXP-009 与 SD-LoRA 均在当前 HEAD 用独立 `*_PAIRED_RERUN*` 目录重跑 4 seeds（INR+C100）。Live-A seed1995/1993 使用当前 HEAD 的现有运行（seed1995 取 diag2 作为规范运行）。统计脚本 `scripts/stage_b_stats.sh`（seed 内连接，paired t-test、95% CI、Cohen's dz、TOST ±0.5），原始输出 `stage_b_stats_output.txt`。
- 结果（mean±std）：
  | 数据集 | 方法 | Final | AvgAcc | Forgetting |
  | --- | --- | ---: | ---: | ---: |
  | INR | Live-A | 78.94±0.47 | 82.45±0.48 | 7.95±1.10 |
  | INR | EXP-009 | 79.22±0.57 | 82.82±0.47 | 7.80±1.02 |
  | INR | SD-LoRA | 78.75±0.50 | 83.42±1.09 | 7.79±1.57 |
  | C100 | Live-A | 87.95±0.34 | 91.55±0.33 | 8.75±0.58 |
  | C100 | EXP-009 | 88.04±0.28 | 91.62±0.37 | 8.63±0.53 |
  | C100 | SD-LoRA | 86.75±0.39 | 91.62±0.24 | 6.27±1.20 |
- 配对差异（Live-A 减对照）：
  | 数据集 | 对照 | Final | AvgAcc | Forgetting |
  | --- | --- | ---: | ---: | ---: |
  | INR | EXP-009 | -0.285（p=0.194） | **-0.374（p=0.007）** | +0.158（p=0.286） |
  | INR | SD-LoRA | +0.187（p=0.698） | **-0.973（p=0.070）** | +0.166（p=0.827） |
  | C100 | EXP-009 | -0.088（p=0.392） | -0.074（p=0.126） | +0.117（p=0.372） |
  | C100 | SD-LoRA | **+1.202（p=0.003）** | -0.074（p=0.755） | **+2.472（p=0.006）** |
- TOST（±0.5）：C100 vs EXP-009 三项均等价；INR vs EXP-009 仅 Forgetting 等价；INR/C100 vs SD-LoRA 均不等价（C100 Final 是“显著优于”，INR AvgAcc 与 C100 Forgetting 是“显著劣化”方向）。
- 门槛判定：
  - Live-A vs EXP-009：C100 三项通过；INR Final -0.285 擦线通过（≥-0.30），**INR AvgAcc -0.374 < -0.30 未通过**，且 INR Final/AvgAcc 均未高于 EXP-009；“Final 或 AvgAcc 至少一项不低于 EXP-009”未满足。
  - Live-A vs SD-LoRA：两数据集 Final 平均均不低于 SD-LoRA（INR +0.19、C100 +1.20）；但 **INR AvgAcc 平均下降 0.97（超过 0.5）**，C100 Forgetting 显著恶化 +2.47（必须作为代价报告）。按目标文件不得宣称全面优越。
  - 参数：Live-A LoRA 368,640（10.00%），含原型 C100 445,440 / INR 522,240，相对 SD-LoRA 3,686,400 减少约 85.8–87.9%；状态与任务数无关。
- 结论：**Stage B 未完全通过**。Live-A 的固定小状态 Final 优势主要体现于 C100，INR AvgAcc 相对 EXP-009/SD-LoRA 的损失是稳定、结构性的；该结果作为负结果记录，不因单 seed 最优结果调整门槛。按目标文件 §6，若 Final 相对 SD-LoRA 保持优势但 AvgAcc/AAA 低于 SD-LoRA，应进入 Stage B2 evaluation-only 双头诊断。

## EXP-026 Stage B2 双头正式训练（INR seed1995，Schedule A/B）

- 日期：2026-08-08
- 状态：INR 完成；C100/多 seed 进行中
- 触发：Stage B2 evaluation-only 诊断显示融合 A/B 相对训练日志 prototype 的 AvgAcc 提升 C100 +0.87/+0.82、INR +2.39/+2.41，均 ≥0.7；正式双头路径已实现（`utils/inc_net.py` `SharedAPrototypeNet` 支持 fc/proto/fused 三种 head mode；`models/sa_sdlora.py` 每任务用当前任务训练类网格拟合 FC/prototype 温度，按归一化任务进度固定调度 A/B；最终任务 lambda=1）。
- 实现提交：`8ee6753`；生命周期修复提交 `885c69a`（每个任务训练前重置为 FC 头，避免 epoch 测试期新旧 prototype 维度不匹配）。
- 结果（INR seed1995）：
  | 调度 | Final Top1 | AvgAcc | Forgetting | 说明 |
  | --- | ---: | ---: | ---: | --- |
  | Schedule A | 79.04 | 82.741 | 6.939 | AvgAcc 低于等效下限 82.83（差 0.09） |
  | Schedule B | **79.21** | **82.937** | **6.851** | 通过 INR Stage B2 验收（Final=纯 prototype 最终值，AvgAcc ≥82.83，F 不劣于纯 prototype） |
- 每任务纯 FC/纯 prototype 结果已写入日志（`[DualHead] task ... mode=...`）；主表使用 fused 曲线，不挑选每阶段最好值。
- 决策：按预注册调度函数（与测试集无关）选择 **Schedule B** 继续 C100 seed1993 与 paired multi-seed；A 保留为负结果记录。

### EXP-026 补充：Schedule B C100 与 paired multi-seed（n=4）

- C100 seed1993：Final **88.08**、AvgAcc **91.94**、Forgetting **8.94**；主表使用 fused 曲线，纯 FC/纯 prototype 逐任务结果在日志中。
- 多 seed（Schedule B）：
  | 数据集 | 方法 | Final | AvgAcc | Forgetting |
  | --- | --- | ---: | ---: | ---: |
  | INR | Dual-B | 78.69±0.42 | 83.26±0.81 | 8.02±1.07 |
  | INR | EXP-009 | 79.22±0.57 | 82.82±0.47 | 7.80±1.02 |
  | INR | SD-LoRA | 78.75±0.50 | 83.42±1.09 | 7.79±1.57 |
  | C100 | Dual-B | 87.82±0.18 | 91.80±0.30 | 9.11±0.42 |
  | C100 | EXP-009 | 88.04±0.28 | 91.62±0.37 | 8.63±0.53 |
  | C100 | SD-LoRA | 86.75±0.39 | 91.62±0.24 | 6.27±1.20 |
- 配对差异（Dual-B 减对照）：
  | 数据集 | 对照 | Final | AvgAcc | Forgetting |
  | --- | --- | ---: | ---: | ---: |
  | INR | EXP-009 | **-0.530（p=0.074）** | **+0.444（p=0.110）** | +0.229（p=0.073） |
  | INR | SD-LoRA | -0.058（p=0.856） | -0.155（p=0.470） | +0.236（p=0.748） |
  | C100 | EXP-009 | -0.220（p=0.108，TOST 等价） | +0.175（p=0.233，TOST 等价） | +0.478（p=0.148） |
  | C100 | SD-LoRA | **+1.070（p=0.011）** | +0.175（p=0.436） | **+2.833（p=0.007）** |
- 参数：LoRA 368,640 + FC 153,600 + prototype（INR 153,600 / C100 76,800）= INR 675,840 / C100 599,040，相对 SD-LoRA 3,686,400 减少 81.7% / 83.8%；`sa_dual_head.pt` 记录调度、lambda、温度。
- 门槛判定：
  - Schedule B 修复了核心 AvgAcc 问题（INR 相对 EXP-009 +0.44，相对 SD-LoRA -0.16；C100 相对两者均 +0.18）。
  - 但严格多 seed 门槛未完全通过：INR Final 相对 EXP-009 -0.53（< -0.30）；INR Final 相对 SD-LoRA -0.058（略低于 0）；C100 Forgetting 相对 SD-LoRA +2.83（显著代价，必须报告）。
  - 最终任务 lambda=1，Final 与同次运行的纯 prototype 一致（满足“不靠测试选择切换点/最终 lambda<1”的禁止项）。
- 结论：双头融合作为“AAA 修复”机制有效（AvgAcc 恢复），但未达到“保持 Final 优势 + 不显著牺牲 Forgetting”的完整验收。按目标文件 §6.4，进入唯一后备方向：每类 K=2 prototype + max/logsumexp 聚合诊断（不再扫描调度）。

## EXP-027 K=2 每类双 prototype 后备方向（INR seed1995）

- 日期：2026-08-08
- 状态：完成（负结果）
- 实现：`MultiPrototypeCosineHead`（max / logsumexp 聚合）；`_compute_prototypes` 对每类当前任务训练特征做确定性 k-means（k=2，farthest-first 初始化，20 轮），输出每类 2 个 L2 归一化 prototype；与双头调度互斥；commit `f6e0212`、`25d8995`。
- 配置：`live_a_k2_inr_seed1995_{max,logsumexp}.json`（T=10，其余超参与 Live-A 一致），输出 `*_R2`。
- 结果（INR seed1995）：
  | 聚合 | Final Top1 | AvgAcc | Forgetting |
  | --- | ---: | ---: | ---: |
  | max | 78.56 | 81.338 | 7.551 |
  | logsumexp | 77.51 | 81.082 | 7.674 |
  | K=1 prototype 参照 | 79.43 | 81.99 | 7.08 |
  | SD-LoRA 参照 | 78.79 | 83.07 | 5.85 |
- 判定：两种聚合的 Final 与 AvgAcc 均低于 K=1 prototype 和 SD-LoRA（logsumexp Final 低 1.92 / AvgAcc 低 1.99），未达到“重新验证 Final”的准入；按目标文件 §6.4 不再继续 K 扩展，也**不启动 C100/多 seed**。
- 停止判定（§11）：双头调度（一轮）与 K=2 prototype（第二轮）连续两轮新增组件未达到预注册门槛；Live-A 不再围绕同一机制继续扫参，保留为“固定 O(1) 状态、Final 优先”的消融/负结果记录。
- 一致性审计：Dual-B（INR/C100）与 K=2（max/logsumexp）产物 `verify_sa_consistency.py` 全部 PASS（feature 6.9–9.5e-6，logit 1.9–3.0e-7），确认负结果不是 artifact 不一致导致。

## 2026-08-09 P0.5：NCCL 后端门槛通过

- backend-aware collectives（NCCL/CUDA、Gloo/CPU）；snapshot 改为 clone；新增 NCCL 四卡同步测试；NCCL Dual-B smoke `P0.5 SMOKE PASS`。
- 测试 98 passed；提交 `14ab833`，manifest 记录真实 commit。
- 进入 P1 ImageNet-R seed1995 完整配对（SGD/20 epochs/4 卡/NCCL）。

## 2026-08-09 P1：ImageNet-R seed1995 正式配对通过

- Live-A control / Dual-B / SD-LoRA / EXP-009 全部完成；trajectory hash、RNG hash、prototype 曲线、四 rank sync、fused/prototype identity 全部 PASS。
- Dual-B Final 78.84 / AAA 83.067 / F 6.624；control AAA 82.152；SD-LoRA Final 77.94 / AAA 82.898；EXP-009 Final 78.74 / AAA 82.326。
- Dual-B AAA 相对 control +0.915，Final/AAA 相对 SD-LoRA +0.90/+0.169；主状态 675,840，相对 SD-LoRA 减少 81.7%。
- artifact consistency 三组 PASS；进入 P2 CIFAR-100 seed1993。

## 2026-08-09 P2：CIFAR-100 seed1993 正式配对通过

- Dual-B Final 88.32 / AAA 91.856 / F 8.233；SD-LoRA Final 87.00 / AAA 91.998；EXP-009 Final 88.37 / AAA 92.042。
- fused Final 与 prototype Final 一致；Final 相对 SD-LoRA +1.32，AAA 相对 SD-LoRA -0.142（门槛通过）；主状态 522,240，减少 85.8%。
- artifact consistency PASS；进入 P3 多 seed 主结果。

## 2026-08-10 P3：多 seed 主结果（n=6）完成，主方法门槛未通过

- INR：Dual-B Final 79.03±0.23 / AAA 83.06±0.46 / F 7.51±1.02；相对 SD-LoRA Final +0.758、AAA +0.114、F -0.071。
- C100：Dual-B Final 87.56±0.47 / AAA 91.20±0.66 / F 8.95±0.65；相对 SD-LoRA Final +0.722、AAA -0.344、F +2.535。
- 门槛：Final 两数据集均不低于 SD-LoRA（通过）；AAA 门槛因 C100 -0.344（>0.3）未通过；状态减少 81.7%/85.8% 通过。
- 结论：**P3 未通过，停止 P4–P6**，结果如实记录为负结果/机制证据。

## 2026-08-11 P1：Dual-B 头分解与 prototype/表示遗忘分解（离线，无重训）

- 状态：完成（全部复用冻结 checkpoint/日志；不改变训练轨迹）。
- P1.1 头分解（确认种子 n=5，`scripts/diagnose_dual_b_head.py`）：
  - INR：FC AAA 83.020、prototype 82.646、fused 83.064（相对 SD-LoRA +0.103）；fused 相对纯 prototype +0.42，主要来自 T0-T4 的 FC 部分。
  - C100：FC AAA 90.467、prototype 91.021、fused 91.063（相对 SD-LoRA -0.385）；fused 相对纯 prototype 仅 +0.04，C100 中期缺口不在 Dual-B 头生命周期。
  - 逐任务明细：`p1_dual_b_head_decomposition_output.txt`。
- P1.2 漂移分解（开发种子，`scripts/diagnose_prototype_representation_drift.py`）：
  - INR seed1995：最终模型+保存原型 Final 78.84 / AAA 83.79 / F 6.19 / old 78.10 / new 85.54；oracle 重算后 Final 80.16 / AAA 84.50 / F 4.43 / old 79.75 / new 83.87（old +1.65、F -1.77、Final +1.32）。
  - C100 seed1993：最终模型+保存原型 Final 88.32 / AAA 92.53 / F 5.62 / old 88.00 / new 91.20；oracle 重算后 Final 89.43 / AAA 93.34 / F 3.97 / old 89.41 / new 89.60（old +1.41、F -1.65、Final +1.11）。
  - 保存 vs 重算原型余弦：INR 0.9478、C100 0.9622（随任务年龄增大）；类内紧致度 INR 0.5811 / C100 0.6919；最近错误类 margin INR 0.1808 / C100 0.2236。
  - 路径 4（任务时模型+重算原型）不可重构：Live-A 只持久化最终 O(1) 状态，逐任务快照未保存且 P1 禁止重训；决策用路径 1/2/3 已覆盖。
- 决策（§7.3）：两开发种子 oracle 刷新 old ≥1.0 且 F ≥0.75（1.65/1.77、1.41/1.65）→ **prototype 坐标失配是主要瓶颈，P1 支持共享 A/历史分支漂移假设**；任务时模型并不优于最终模型，共享 A 表示遗忘分支不成立。
- 汇总：`p1_mechanism_diagnostics_summary.md`。
- 下一步：P2 Historical-Branch Activation Distillation（唯一允许的新训练组件）。

## 2026-08-11 P2：HBD 开发运行（CIFAR-100 seed1993，关闭）

- 状态：完成（exit=0），**C100 开发门槛未通过，HBD 方向按任务书关闭**。
- 配置：`exps/c100_p2_hbd_seed1993_nccl.json`（λ=6.0，唯一一次工程尺度修正）；commit `baeb66c`。
- 结果（基线 88.32 / 91.856 / 8.233）：Final **87.82**（门槛 88.12 ✗，-0.50）、AAA **92.176**（门槛 92.156 ✓，+0.32）、Forgetting **6.156**（门槛 7.233 ✓，-2.08）。
- 曲线：`[99.2, 95.65, 94.17, 93.4, 92.14, 91.55, 90.91, 88.72, 88.2, 87.82]`；T5-T8 全面改善（+0.6~+0.9），T9 比基线低 0.50。
- 审计：consistency PASS（7.2e-6 / 2.2e-7）；持久参数 445,440 与无 HBD 完全相同；Task0 无蒸馏；四 rank/RNG/校准全部 PASS。
- 机制：HBD 以约束共享 A 漂移换取旧类保持，同时约束了最终任务塑性；Final 门槛差 0.30。
- 决策：不运行 INR seed1995、不运行 HBD seed1-5；停止修改方法，进入 P4 消融/参数匹配与 P5 任务长度/第三数据集/强基线/效率。
- 完整记录：`p2_hbd_development_result.md`。

## 2026-08-11 P4：消融 C（冻结 Live-A 的共享 A）开发种子

- 状态：完成（C100 seed1993 + INR seed1995，均 exit=0）。
- 配置：完整方法 + `sa_train_a_all_tasks=false`（A 在 task0 后冻结）；commit `31440f5`。
- 结果（对照完整方法）：
  - C100 seed1993：Final **88.26**（对照 88.32，-0.06）、AAA **92.119**（91.856，+0.26）、Forgetting **7.778**（8.233，-0.46）。
  - INR seed1995：Final **78.31**（78.84，-0.53）、AAA **83.03**（83.067，-0.04）、Forgetting **6.309**（6.624，-0.32）。
- 审计：consistency PASS（1.5e-7/3.1e-7）；持久参数与完整方法相同（445,440 / 522,240）。
- 分析：冻结 A 主要代价在 INR Final（-0.53，即 Live-A 的 A 更新贡献约 +0.5 的最终任务塑性）；C100 上几乎中性且 Forgetting 略好。A 冻结不是 C100 遗忘的来源。
- 3-seed 补跑：配置已预注册（`exps/p4_abl_c_*_seed{1,2,3}_nccl.json`）。

## 2026-08-11 P4：rank1 参数匹配基线（SD-LoRA rank1 + 同 Dual-B 头）开发种子

- 状态：完成（smoke 通过；C100 seed1993 + INR seed1995，均 exit=0）。
- 实现：`models/sdlora_dual_b.py`（每任务 rank-1 LoRA bank + prototype/Dual-B Schedule B 头，单遍 fused eval，RNG 中性校准）。
- 结果（对照完整方法）：
  - C100 seed1993：Final **87.73**（88.32，-0.59）、AAA **92.201**（91.856，+0.35）、Forgetting **5.689**（8.233，-2.54）。
  - INR seed1995：Final **76.28**（78.84，-2.56）、AAA **82.251**（83.067，-0.82）、Forgetting **6.751**（6.624，+0.13）。
- 参数：每任务 rank1（A+B+scale）36,885；T=10 bank 368,640，与完整方法 LoRA 持久状态相同（含 prototype/FC 后亦同）。
- 分析：同预算下，rank-10 Aggregate-B 的在线容量在 INR 上全面优于十个 rank-1 task adapter（Final +2.56、AAA +0.82）；
  C100 上 rank1 的 AAA/Forgetting 更好但 Final 低 0.59。论文主张按“INR 容量优势、C100 Final 优势”分列，不宣称全面优越。
- 3-seed 补跑：配置已预注册（`exps/p4_rank1_*_seed{1,2,3}_nccl.json`）。

## 2026-08-11 P4：消融 C 3-seed（预注册种子 1-3）完成

- 状态：6 个运行全部 exit=0（C100 seed1-3、INR seed1-3）。
- C100（冻结 A）：Final 87.92±0.44、AAA 91.80±0.47、F 7.79±0.93；配对相对完整方法 **+0.49/+0.43/-1.45**（三个种子全部一致改善）。
- INR（冻结 A）：Final 78.37±0.28、AAA 82.93±0.56、F 8.05±1.35；配对相对完整方法 **-0.59/-0.13/+0.06**。
- 结论：Live-A 的 A 更新贡献 INR 最终塑性（约 +0.5），但在 C100 上是历史干扰源；论文分列报告。
- 汇总：`p4_ablations_results.md`。

## 2026-08-11 P4：rank1 3-seed（预注册种子 1-3）完成

- 状态：6 个运行全部 exit=0。
- C100（rank1）：Final 87.69±0.25、AAA 92.14±0.44、F 6.25±0.45；配对相对完整方法 **+0.26/+0.77/-2.99**。
- INR（rank1）：Final 76.18±0.16、AAA 81.99±0.89、F 8.90±1.04；配对相对完整方法 **-2.78/-1.07/+0.90**。
- 结论：同预算对比呈数据集依赖——INR 支持 Aggregate-B rank10 容量优势，C100 支持逐任务 rank1 bank；论文分列报告，不宣称全面优越。
- 汇总：`p4_ablations_results.md`。

## 2026-08-12 P5：任务长度 N=5/10/20（冻结协议）完成

- 状态：T5/T20 共 16 个运行全部 exit=0（完整方法/SD-LoRA/EXP-009/rank1 × C100/INR）；T10 复用开发种子。
- 完整方法 Final（C100）：89.14 / 88.32 / 85.37；SD-LoRA：88.55 / 87.00 / 83.61；EXP-009：89.11 / 88.37 / 85.55；rank1：88.25 / 87.73 / 83.45。
- 完整方法 Final（INR）：79.86 / 78.84 / 78.23；SD-LoRA：79.48 / 77.94 / 76.54；EXP-009：79.84 / 78.74 / 78.23；rank1：77.18 / 76.28 / 75.14。
- 结论：完整方法 Final 相对 SD-LoRA 的优势随 T 增大（C100 +0.59/+1.32/+1.76；INR +0.38/+0.90/+1.69）；相对 EXP-009 等价；相对 rank1 同预算基线全面更高且差距随 T 扩大；Forgetting 随 T 恶化且与 EXP-009 相当（C100 T20 10.59 vs 10.85），相对 SD-LoRA 的代价如实报告。
- 完整表与分析：`p5_task_length_results.md`。

### ⚠️ 统计口径修正（2026-08-11 P0）

- 上述 n=6 统计把开发种子（INR seed1995、C100 seed1993）并入主统计，**只能作为描述性敏感性分析**，不能作为论文主显著性结论。
- 严格确认种子（seed1-5，n=5）已由 `scripts/strict_n5_stats.sh` 生成，见 `p3_strict_n5_stats_output.txt` 与 `p3_strict_n5_summary.md`：
  - INR：Dual-B 79.07±0.23 / 83.06±0.52 / 7.68±1.03；SD-LoRA 78.34±0.48 / 82.96±0.87 / 7.79±1.04；EXP-009 79.16±0.32 / 82.63±0.36 / 7.69±0.93。
  - C100：Dual-B 87.41±0.32 / 91.06±0.64 / 9.09±0.62；SD-LoRA 86.81±0.56 / 91.45±0.63 / 6.57±0.71；EXP-009 87.58±0.28 / 91.12±0.54 / 9.19±0.47。
  - 配对（Dual-B − SD-LoRA）：INR Final +0.730（p=0.0784）、AAA +0.103（p=0.6743）、F -0.105（p=0.6984）；C100 Final +0.602（p=0.0411）、AAA -0.385（p=0.0057）、F +2.522（p=0.0002）。
- 30 个 P3 运行的 manifest/config SHA-256/seed/artifact/queue status 审计全部 PASS（`scripts/audit_p3_runs.py` → `P3 AUDIT PASS`）。
- 工作分支：`p0-strict-n5`（从 `15ab791` 建立，保留 P3 原始状态）。

## 2026-08-12 P5：第三数据集 CUB-200（完整方法 seed1）完成

- 配置：`exps/p5_cub_livea_dual_b_seed1_nccl.json`（SGD/constant/0.01/20 ep/batch32/4 卡 NCCL/确定性协议，T=10，seed1）；commit `1e7d884` 后启动，实际运行 commit `1e7d884`，exit=0。
- 结果（完整方法）：**Final 77.84 / AAA 85.85 / Forgetting 16.27**（曲线最后 5 项：83.11 → 77.84；AAA 85.848）。
- 对照（旧协议，Adam）：EXP-009 CUB seed1 = 71.75 / 84.93 / 23.31（`sa_sdlora_proto_cub_seed1.log`），完整方法 **Final +6.09、AAA +0.92、F -7.04**。
- 对照（SD-LoRA 论文发布值，Adam/batch128）：CUB T=10 = 77.48 / 85.59；本方法 seed1 与之相当。
- 多 seed 队列（seed2/3 full + exp009/sdlora seed1-3，同冻结 SGD 协议）已预注册并启动：`run_p5_cub_multi_queue.sh`，commit `3cdfa7d`。
- 汇总：`p5_cub_results.md`（待队列完成后回填多 seed 表）。

## 2026-08-13 P5：CUB-200 三方法 3-seed 全部完成

- 9 个运行全部 exit=0（完整方法 seed1-3、EXP-009 seed1-3、SD-LoRA seed1-3，同冻结协议）。
- 完整方法 75.27±3.12 / 84.37±1.34 / 16.77±0.43；EXP-009 75.37±2.90 / 85.01±1.79 / 17.15±0.11；
  SD-LoRA 71.26±1.47 / 81.66±0.90 / 10.97±2.17。
- 配对差：完整方法 − EXP-009 = -0.10/-0.64/-0.38（等价）；完整方法 − SD-LoRA = +4.01/+2.70/+5.80。
- 与 SD-LoRA 论文发布值（77.48/85.59）不同协议，不作显著比较。
- 完整表：`p5_cub_results.md`；队列日志 `p5_cub_remaining_queue.log`。

## 2026-08-13 P5：强外部基线

- InfLoRA 本机复现完成（CIFAR-100 T=10 seed1）：**84.75 / 90.26**（发布值 86.51/91.70），
  见 `p5_external_baselines_results.md`。
- CL-LoRA 本机复现失败（torch 1.12 环境与官方两阶段 backward 不兼容；已修复设备/autograd
  后 task 0 仍发散为 NaN），论文采用发布值（见 `strong_baselines_sd.md`）。
- LoRA-DRS 本机复现完成（CIFAR-100 T=10 seed1）：**89.73 / 93.02**（发布值 89.14/92.55），
  见 `p5_external_baselines_results.md`（2026-08-13 05:45 完成，设备硬编码修复后全程无错）。

## 2026-08-13 P6：效率与 artifact 闭环全部完成

- 状态曲线：完整方法 LoRA 368,640 与持久磁盘 ~2.78MB 在 T=5/10/20 恒定（O(1)）；
  SD-LoRA/EXP-009/rank1 为 O(T)（`scripts/measure_state_curves.py`）。
- 显存曲线：完整方法 3,468.6 MiB（T=5/10/20 恒定）；SD-LoRA 5,963.9 → 8,186.7 → 12,632.1 MiB。
- 时间曲线：T=10 每任务耗时完整方法 204s→336s（1.65×），SD-LoRA 177s→535s（3.02×）。
- 推理：1.129e12 FLOPs/forward，空闲 GPU 515.3 img/s，578.4 MiB。
- 最小导出验证：C100 与 CUB 各一份，features/fc/proto/fused logits max_abs_diff == 0，
  Final 与训练日志一致（87.02 / 77.84）。
- 汇总：`p6_efficiency_results.md`。
## 2026-08-24 P7：T=20 多种子确认实验启动

- 目标：确认 P5 单开发种子中“完整方法相对 SD-LoRA 的 Final 优势随任务长度扩大”的趋势。
- 矩阵：CIFAR-100/ImageNet-R × 完整方法/SD-LoRA/EXP-009 × seeds 1/2/3，共 18 个运行。
- 协议：完全复制 P5 T=20 冻结配置，仅修改运行标识、seed 和独立输出目录；不调参。
- 执行：GPU 0-3，4-rank NCCL，串行 `nohup` 队列，失败即停止，独立 30 分钟监控。
- 预注册与统计口径：`p7_t20_multiseed_plan.md`。
## 2026-08-24 Coordinate-Stable Live-A 单 seed 开发实验

- 状态：实现中，待 P7 T=20 队列释放 GPU 后自动启动。
- 目标：联合解决 Live-A 更新导致的历史有效算子变化和旧 prototype 坐标失配，同时保持 O(1) LoRA 状态。
- 方法：任务结束时闭式求解 `G_aligned` 以保持 `G_old normalize(A_old)`；部署态重建后，用当前任务前后配对特征拟合 rank-10 正交残差 transport。固定 80/20 训练数据门控，无验证收益则使用恒等映射；transport 用后丢弃。
- 理论边界：不同于已有 generic affine/class-mean/JVP LRPT，本实验先在 LoRA 算子层消除可解释坐标变化，再只允许观测漂移子空间内的等距旋转，禁止自由缩放、偏置和持久映射网络。
- 配置：`c100_coordinate_stable_seed1993_nccl.json`、`inr_coordinate_stable_seed1995_nccl.json`；完整协议与对应 Live-A + Dual-B 开发运行一致。
- 预注册门槛与状态边界：见 `coordinate_stability_plan.md`。

## 2026-08-25：CoordinateStable 核心组件消融启动

- 实验标题：Effective-operator Coordinate Alignment 与 Gated Residual Orthogonal Prototype Transport 的 2x2 因果拆分。
- 理论依据：Alignment 处理共享 A 更新导致的 LoRA 有效算子坐标变化；Transport 处理剩余特征空间与历史 prototype 的失配。两者分别位于参数空间和表征空间，必须独立开关以验证互补性。
- 已有对照：A0 Live-A+Dual-B 与 A3 完整 CoordinateStable 的三个数据集三种子结果均已完成。本轮仅运行 A1（alignment-only）和 A2（transport-only）。
- 实验矩阵：CUB-200、ImageNet-R、CIFAR-100 × A1/A2 × seeds 1/2/3，共18个严格同协议运行。
- 训练协议：继承 `p5_cub_livea_dual_b_*`、`p3_inr_livea_dual_b_*`、`p3_c100_livea_dual_b_*`；SGD、batch32、4卡NCCL、确定性训练。除两个消融开关外不改超参数。
- 执行入口：`run_coordinate_stable_core_ablations_queue.sh`；队列日志 `coordinate_stable_core_ablations_queue.log`；30分钟监控 `monitor_coordinate_stable_core_ablations.sh`。
- 防污染：每次运行记录 commit/config SHA，完成后在 artifact 内保存 `effective_config.json`；已有完整结果自动跳过，发现不完整目录则停止且不覆盖。
- 状态：待提交启动；结果完成后回填 Final、AAA、Forgetting、配对差值和 transport gate 诊断。

## 2026-08-26：Bounded Norm-Calibrated Consolidation 多种子预注册

- 实验标题：将旧版隐式范数缩放转化为有界、显式的任务边界校准机制。
- 尝试方法：新增 `bounded_norm_calibrated_absorb`。令 `p_t=||A_t||_F||B_t||_F`、`alpha_t=min(1,1/p_t)`，持久化当前项为 `alpha_t s_t B_t A_t`。当 `p_t>1` 时复现旧版衰减；当 `p_t<1` 时使用 operator-preserving absorption，禁止放大。
- 理论依据：旧版与修复版形成自然消融。C100/INR 的 `p_t` 平均为 1.164/1.207，旧版分别衰减约 13%/17%；CUB 的 `p_t` 为 0.433，旧版放大约 2.33 倍。三数据集符号相反，支持“衰减有利、放大有害”的有界 consolidation 假设。
- 参数与状态：校准系数立即折叠入 `G`，不保存任务 bank 或额外标量，持久状态和推理成本不变。
- 实验矩阵：CIFAR-100、ImageNet-R、CUB-200 × seeds 1/2/3，共 9 个运行；batch32、4-rank NCCL、原任务顺序及 Dual-B/transport 配置不变。
- 对照：复用同 seed 的 `normalized_absorb` 和 `operator_preserving_absorb` 完整结果。禁止根据数据集选择不同模式或继续扫描阈值。
- 运行：`run_coordinate_stable_normcap_multiseed_queue.sh`，GPU 0-3，nohup 串行；`monitor_coordinate_stable_normcap_multiseed.sh` 每 30 分钟记录一次。
- 验证：单元测试 `38 passed`；C100 两任务四卡冒烟 exit=0，Task1 Final 87.51，alignment/transport/Dual-B/RNG hash 全部通过。Task0/1 平均 consolidation gain 为 0.8663/0.9129，分支最大值均为 1.0，无放大。
- 状态：实现与冒烟完成，提交后启动正式 9-run 队列，结果待回填。

## 2026-09-06：Adaptive-A 三路同协议诊断

- 实验标题：Live-A、冻结 A、Adaptive-A 的两卡严格配对。
- 理论依据：旧消融显示冻结 A 在 C100 上更稳、在 INR 上损失约 0.5 Final；Adaptive-A 试图只抑制改变共享 A 行空间且历史影响较大的梯度分量。
- 配置：三个数据集各使用既定开发 seed；SGD、20 epoch、rank10、CoordinateStable、bounded NormCap、transport、Dual-B 均保持不变。两卡每卡 batch64，等效 batch128。
- 三组：`live_a=(train_a=true, adaptive=false)`；`frozen_a=(false,false)`；`adaptive_a=(true,true)`。
- 执行：`run_adaptive_a_three_way_2gpu_queue.sh` 动态生成 9 份可审计 JSON，独立日志和输出目录；结果待回填。

## 2026-09-14：Function-Safe Pareto 三数据集单 seed 验证

- 实验标题：以旧类 logits KL 半空间约束 Pareto-Knee 的 Live-A 候选。
- 尝试方法：新增 `function_safe_pareto`。任务开始时冻结上一任务部署 backbone 和旧分类头；每个 block 对 Q/V Live 候选做联合半空间投影，触发条件为 cosine `< -0.05` 且绝对内积 `< -1e-12`。稳定梯度和 cross-fit utility 每 4 step 同步刷新并缓存。
- 理论依据：原 operator-risk 只约束参数空间变化，并不等价于旧类决策稳定。旧模型在当前数据上的 logits 提供 rehearsal-free function-space 约束；row-space normal 分量是共享 A 子空间变化的直接通道，优先在该通道执行最小修正。
- 与已有实验区别：纯 `functional_halfspace` 每 step 对全局 A 更新施加强约束，可能牺牲塑性；本实验保留 Pareto 的 Frozen/Tangent/Live 选择，只修正存在显著功能冲突的分块 Live 候选。
- 固定协议：CoordinateStable alignment、prototype transport、bounded NormCap、rank10、优化器、学习率、任务顺序均继承主配置；Dual-B 关闭；T=10、20 epoch、双卡每卡 batch64。
- 实验矩阵：CIFAR-100 seed1993、ImageNet-R seed1995、CUB-200 seed1。执行脚本 `run_function_safe_pareto_t10_3datasets_2gpu.sh`；实现提交 `fbb24c7`，2026-09-14 09:37 使用 GPU `0,2` / `3,4` / `5,6` 并行启动，结果待回填。
- 实现验证：Adaptive-A 定向测试 99 passed；完整仓库测试 234 passed；三份运行配置 PREPARE_ONLY 预检通过。独立审查后禁止与 Dual-B 混用，并要求非算子等价 absorption 必须配合 prototype classifier 和 CoordinateStable transport。
- 结果：三路均 exit=0。Function-Safe Pareto 为：CIFAR-100 `85.33 / 91.251 / 8.278`，ImageNet-R `78.68 / 81.777 / 5.851`，CUB-200 `84.31 / 89.578 / 8.610`（Final / AAA / Forgetting）。
- 严格同双卡 batch64 对照纯 `functional_halfspace`：CIFAR-100 `88.29 / 92.263 / 6.156`，ImageNet-R `79.43 / 82.151 / 5.638`，CUB-200 `83.02 / 89.282 / 10.484`。配对差（新策略减纯半空间）分别为 C100 `-2.96 / -1.012 / +2.122`、INR `-0.75 / -0.374 / +0.213`、CUB `+1.29 / +0.296 / -1.874`。
- 诊断：Task 9 冲突触发率为 C100 `4655/9600=48.5%`、INR `2023/5040=40.1%`、CUB `576/1200=48.0%`；平均修正比例为 `0.167 / 0.116 / 0.147`，均无 full fallback。C100/INR 的负收益说明当前任务图像上的旧 logits KL 不能稳定代表旧类边界，且投影后的 Live 候选改变了 Pareto utility/risk 的塑性平衡；CUB 的正收益支持该信号在细粒度数据上可能有效，但不具跨数据集普适性。
- 阶段结论：未达到“三数据集不弱于基线”的验收标准。暂不扫描阈值；下一步应先做 teacher signal 的离线相关性诊断（旧类下降与 KL gradient/conflict 的相关性）以及 projected-Live 与原始-Live 的成对候选分析。

## 2026-09-14：Function-Safe 失效机制诊断

- 实验标题：旧 logits teacher 质量与单步 KL 参数归因诊断。
- 目的：区分三类候选根因：(1) 当前任务图像上的旧类 teacher 分布信息量不足；(2) A 半空间投影未在有限步长下维持 KL；(3) 未受约束的 B/scale 更新主导 KL 漂移。
- 方法：新增默认关闭的 `sa_functional_diagnostics_interval`。采样时保存 optimizer step 前的 A、B 和 scale；step 后在同一 held-out 半批上使用一次性 backbone 深拷贝回放 pre、full-post、A-only-post、B/scale-only-post，记录 teacher 归一化熵、最大概率、概率 margin、KL 增量及交互项。诊断不写入真实参数，其对象身份、version counter、train/eval 状态、input-sketch 开关和 RNG 均保持不变。
- 不变量：不改变 loss、梯度、Pareto 候选、投影公式、prototype transport、NormCap 或分类头；开关默认 `0`，已有实验路径无额外前向。
- 短序列矩阵：CIFAR-100 seed1993、ImageNet-R seed1995、CUB-200 seed1；前 5 个任务，双卡每卡 batch64、SGD、20 epoch；每 20 step 诊断一次。GPU 分别为 `0,1`、`2,3`、`4,5`。
- 验收：若 teacher entropy 接近 1 且 margin 很小，支持 teacher 低信息量；若 A-only KL 不增而 full/B-only 增长，支持约束覆盖不完整；若 A-only 本身频繁增长，则有限步长、缓存或候选选择破坏了一阶保证。
- 实现验证：定向测试 `43 passed`，完整仓库 `242 passed`；真实 Shared-A 参数别名测试确认诊断前后 Parameter 身份、version counter、requires-grad 与 optimizer ownership 不变。CUB 两任务双卡 DDP 冒烟 `exit=0`，采样与非采样 step 均无 collective 死锁；运行脚本 `run_function_safe_diagnostics_t5_3datasets_2gpu.sh` 的三份配置均通过 PREPARE_ONLY 预检。
- 冒烟首批观测（仅用于机制验证，不作为性能结果）：Task1 teacher entropy `0.999111`、max probability `0.058072`、margin `0.002718`；完整单步 KL 增量 `+5.39e-9`，A-only `-2.46e-9`，B/scale-only `+1.63e-8`。这同时支持“teacher 低信息量”和“B/scale 漂移不受 A 半空间约束”两条假设，待三数据集 T5 结果确认。
- T5 正式运行 Task1 阶段证据：C100 teacher entropy/max-prob `0.999643/0.107837`，KL 增量 full/A-only/B-scale-only 为 `+6.72e-7/-2.44e-7/+9.01e-7`，safe-step `43.75%`；INR 为 `0.999846/0.053448`、`+1.05e-6/-1.25e-7/+1.17e-6`、`35.71%`；CUB 为 `0.999046/0.058946`、`+4.47e-7/+0.66e-7/+4.84e-7`、`20.00%`。
- 阶段判断：三个 teacher 均接近对应旧类数下的均匀分布；C100/INR 的 A-only 平均 KL 下降，证明 A 投影方向并非主要错误，但未受约束的 B/scale 增量分别达到 full 增量的约 `134%/111%`，覆盖了 A 的保护作用。CUB 的 Tangent 选择比例更高且 Tangent 不受投影，A-only 也轻微上升。不能通过继续调 cosine threshold 或刷新间隔解决这一结构性覆盖缺口。

## 2026-09-14：Functional stability signal 三路单-seed 对照

- 实验标题：Full-logit KL、Historical-only logit KL 与 HBD 的同协议直接比较。
- 尝试方法：保留 `function_safe_pareto` 的 Pareto 候选、cross-fit、半空间投影和所有主方法组件，只把 student 稳定性前向从完整 `historical + current` LoRA 改为仅历史 `G A/||A||` 分支；teacher、正常训练前向和部署前向不变。第三路使用已有 `functional_halfspace` HBD，直接约束历史 Q/V 响应。
- 理论依据：诊断显示完整 student KL 的正漂移主要由不受 A 投影控制的 current `sBA` 贡献。Historical-only scope 将稳定梯度重新限定为“共享 A 对历史函数的影响”，避免要求 A 抵消当前 B 为新任务学习产生的必要变化；HBD 用作不经过分类头的历史功能约束对照。
- 公平协议：CIFAR-100 seed1993、ImageNet-R seed1995、CUB-200 seed1；T=10、20 epoch、SGD、双卡每卡 batch64（等效 128）。CoordinateStable alignment、prototype transport、bounded NormCap、LoRA rank、学习率和任务顺序均继承同一数据集 source config，`sa_dual_head=false`。
- 三路：`full_logit=function_safe_pareto + scope=full`；`historical_logit=function_safe_pareto + scope=historical`；`hbd=functional_halfspace`。不启用诊断插桩，避免额外前向改变三路运行时间口径。
- 执行：`run_functional_signal_threeway_single_seed_2gpu.sh` 生成 9 份可审计配置；按数据集分三波，每波三路并行使用 GPU `0,1` / `2,3` / `4,5`。结果待回填。
- 验收：Historical-only 若在三个数据集均不弱于 Full-logit，说明去除 current-B 混杂有效；若仍不如 HBD，则瓶颈主要是当前任务图像上旧分类 logits 的低信息量，而非 branch scope。
- 结果：三波均 `status=0`。CIFAR-100：Full-logit `Final/AAA/F=85.33/91.251/8.278`，Historical-only `88.13/92.373/6.033`，HBD `88.29/92.263/6.156`；ImageNet-R：`78.68/81.777/5.851`、`79.07/82.008/5.587`、`79.43/82.151/5.638`；CUB-200：`84.31/89.578/8.610`、`84.31/89.582/8.668`、`83.02/89.282/10.484`。
- 配对差 Historical-only 减 Full-logit：CIFAR-100 `+2.80/+1.122/-2.244`，ImageNet-R `+0.39/+0.231/-0.264`，CUB-200 `0.00/+0.004/+0.058`（F 越低越好）。Historical-only 在三个数据集均不低于 Full-logit，且 C100 恢复最明显。
- 与 HBD 比较：Historical-only 在 C100 仅低 `0.16 Final`、AAA 高 `0.110`、F 低 `0.123`；在 INR 低 `0.36/0.143` 但 F 低 `0.051`；在 CUB 高 `1.29/0.300` 且 F 低 `1.816`。因此 HBD 不是跨数据集统一上限，Historical-only 在 CUB 更好。
- 阶段结论：三路单 seed 支持将 Historical-only 作为 Function-Safe 的默认候选，证明完整 student logits 中 current `sBA` 的混杂是实质问题；但 C100/INR 仍未超过 HBD 的 Final，且只有单 seed，下一步应做配对多 seed，而不是直接宣称普适最优。

## 2026-09-15：CUO 同预算低秩投影基线（待正式训练）

- 实验标题：固定低秩坐标下的 Cumulative Unified Optimization（CUO）LoRA 对照。
- 方法：每个 Q/V 分支持久保存固定正交下投影 `P in R^(10x768)`、统一上投影 `H in R^(768x10)` 和投影 Gram 矩阵 `C in R^(10x10)`。训练时历史增量为 `H(Px)`，当前任务保留原始 `sB(Ax)`；任务结束后只用当前任务训练样本的测试预处理做确定性无梯度遍历，累计 `C += Z^T Z`、`D += Y^T Z`，并求解 `H(C+lambda I)=D`，其中 `lambda=1e-5`。
- 预算：12 个 block 的 Q/V 共 24 个分支。部署 LoRA 因子为 `24 x (768x10 + 10x768)=368,640` 个标量；Gram bookkeeping 为 `24 x 10x10=2,400`；持久总计 `371,040`。报告时必须同时给出二者，不将 Gram 统计隐去。
- 协议：CIFAR-100 seed1993（10 类/任务）、ImageNet-R seed1995（20 类/任务）、CUB-200 seed1（20 类/任务）；ViT-B/16、rank10、SGD、20 epoch、双卡每卡 batch64（等效128）、prototype classifier，关闭 Dual-B、CoordinateStable/transport、HBD 和 Adaptive-A。
- 验证：CPU 回归 `74 passed`；两卡 Task0/1 合成 smoke 通过，两个 rank 的 P/H/C 哈希一致，仅写出 `sa_state.pt` 与 `sa_merged_lora.pt`，无 per-task B 文件。该 smoke 仅验证实现和分布式状态，不代表性能结果。
- 运行：`run_cuo_lowrank_r10_3datasets_2gpu.sh` 使用两组 GPU 并行运行 CIFAR-100 与 ImageNet-R，CIFAR-100 完成后复用该卡对运行 CUB-200；正式 Final/AAA/Forgetting 尚未产生，禁止把本条视为效果结论。

## 2026-09-22：SBGC P2 严格正式部署验证

- 实验标题：固定 5% per-transition 历史响应风险下的 uniform/Fisher G consolidation。
- 目的：验证 P1 中非空的风险约束与 diagonal Fisher 差异能否转化为 Final、AAA 或 Forgetting 收益，而不是继续依赖 shadow 候选诊断。
- 方法矩阵：Frozen-P additive、CUO-lowrank、uniform-budget G、Fisher-SBGC。Frozen-P 复用 P1 实际部署结果，其余三路在当前提交重跑。
- 固定协议：CIFAR-100 seed1993、ImageNet-R seed1995、CUB-200 seed1；T=10、rank10、20 epoch、双卡每卡 batch64、prototype classifier。关闭 transport、Dual-B、HBD、NormCap、Adaptive-A 和 current-branch normalization。
- 调度：GPU `0,1`、`4,5`、`6,7` 分别绑定 C100、INR、CUB；各数据集 Fisher→uniform→CUO 串行。入口为 `run_sbgc_p2_strict_3datasets_2gpu.sh`，全部 torchrun 使用 nohup，自动每 30 分钟记录状态。
- 诊断：每任务记录 calibration、solver、boundary wall time，CUDA peak/additional peak memory，以及逐分支风险、distortion、eta、sensitivity CV 和候选差异。诊断不进入 checkpoint 或算法判断。
- 验收：风险 `<=0.050001`；三数据集 Final/AAA 均距最佳 Frozen/CUO 不超过 `0.30`；至少两个数据集有 Final 或 AAA 提升 `0.30`；Forgetting 不高于两基线最小值 `0.50` 以上。
- 禁止项：P2 完成前不测试 `0.01/0.10`，不加入 warm-up，不依据中途结果修改预算。实现提交 `b8664c7`，配置提交 `23107d5`，完整测试 `436 passed`。
- 结果：Fisher-SBGC 在 C100/INR/CUB 为 `88.28/92.761/5.656`、`78.82/82.479/6.264`、`84.29/89.462/7.952`；Uniform-G 为 `88.31/92.760/5.600`、`78.63/82.408/6.354`、`84.33/89.465/7.781`；CUO 为 `88.23/92.739/4.589`、`78.33/82.530/6.493`、`84.14/89.489/8.180`。
- 判定：所有风险与状态验收通过，但 Fisher 相对最佳 Frozen/CUO 的 Final/AAA 最大提升仅 `0.10/0.022`，没有数据集达到 `+0.30`；C100 Forgetting 门槛失败，且 Fisher 仅在 INR 优于 Uniform。因此 P2 为 NO-GO，不进入多 seed。

## 2026-09-23：同预算 T=1 离线联合训练参考

- 实验标题：rank10 固定状态 LoRA 在全部类别联合可见时的经验性能参考。
- 目的：估计当前 368,640 LoRA factor 预算在没有任务边界和持续合并误差时的可达准确率，用于量化 T=10 方法与离线联合训练之间的差距。
- 定义：CIFAR-100 将 100 类作为唯一 Task0；ImageNet-R/CUB-200 将 200 类作为唯一 Task0。使用当前 SBGC Task0 路径，训练中没有历史风险约束，任务结束 QR canonicalization 严格保持有效算子。
- 固定协议：原开发 seed、rank10、20 epoch、双卡每卡 batch64、相同 SGD/学习率/weight decay、prototype classifier；GPU 为 `0,1`、`4,5`、`6,7`，避开 `2,3`。
- 边界：该结果称为 single-seed empirical offline reference，不称数学或统计理论上限。T=1 的 AAA 等于 Final，Forgetting 不定义，主比较只使用 Final。
- 执行：`run_sbgc_offline_t1_3datasets_2gpu.sh` 三数据集并行，全部 torchrun 由 nohup 包装并每 30 分钟监控；结果待回填。

## 2026-09-20：Recoverability-Constrained Accessibility Adaptive-A 正式分阶段验证

- 实验标题：从精确历史算子可恢复性到全局预算控制的四阶段累计验证。
- 理论依据：旧 impact gate 只衡量特定 LoRA 因子坐标下的算子变化。本方法以固定 task-start 历史算子为 anchor，用 LS 对齐后的不可恢复能量定义历史风险；当前任务塑性由 effective-weight gradient 在候选 row space 中的 accessibility gain 衡量。完整方法在全网络 Q/V 分支间分配一个历史算子能量预算。
- 四阶段：S1 `exact_risk`；S2 `accessibility`；S3 `anchor_realign`；S4 `global_budget`。前三项为累计机制消融，S4 为完整方法。
- 固定协议：CIFAR-100 seed1993、ImageNet-R seed1995、CUB-200 seed1；T=10、rank10、SGD、20 epoch、等效 batch128；继承 CoordinateStable、prototype transport、bounded NormCap 和 Dual-B，只改变 Recoverability Adaptive-A 字段。
- 正式入口：`run_recoverability_formal_3datasets_2gpu.sh`。INR 使用 GPU 4,5，CUB 使用 6,7，每卡 batch64；因 GPU 0,1 被外部 SLCA 占用，C100 使用 GPU2 单卡 batch128。每个数据集内部严格 S1->S4，失败即停止。
- 实现与修复：几何实现提交 `7c44f87`，控制器 `feb8773`；Task0 配置传播与安全 gradient-hook 修复 `4d937a9`；灵活 GPU 调度 `bf9a9db`。完整测试 `359 passed`，四阶段 CUDA smoke 通过。
- 无效运行：标签 `20260920_142740` 因 Task0 配置未传入且 hook 挂在原地修改 tensor 上发生 SIGSEGV；标签 `20260920_143539` 中 C100 首次运行因 GPU 0,1 已有外部进程而 OOM。二者不得计入结果统计。
- 有效运行：INR/CUB 标签 `20260920_143539`，C100 标签 `20260920_143956`。截至记录时 C100 S1 已到 Task0 Epoch10，INR S1 已进入 Task1，CUB S1 已进入 Task4，均无训练错误；最终指标待回填。

## 2026-09-21：首任务锚定子空间四路结构筛选（待完成）

- 实验标题：SA-LoRA B-bank 与固定状态 cumulative-G 的同协议因果拆分。
- 理论依据：若 Task0 学得的 A 是长期可复用子空间，则后续仅训练 B/scale 不应依赖 A 旋转。把冻结历史 scale 的归一化 B-bank 精确折叠为一个 G，可进一步检验 O(T) 专家存储是否必要；bounded-G 则测试显式 task-boundary attenuation 是否优于算子精确折叠。
- 实现修正：新增默认关闭的 `sa_normalize_current_branch`。开启后 current branch 使用 `s(B/||B||)(A/||A||)x`，与 SA-LoRA 的 separate normalization 一致；`normalized_absorb` 后的历史算子与该 current branch 严格一致。默认关闭保证历史实验语义不变。
- 对照边界：SA-LoRA 论文使用 Adam，且 ImageNet-R 为 30 epoch；本轮为控制变量采用本项目 SGD/20 epoch，因此结果只能称“同协议结构复现”，不能冒充论文原始数值复现。
- 运行协议：三数据集 T=10、rank10、单卡 batch128；GPU 0、1、4、5 动态派发，空闲卡完成一个任务后立即领取下一个。结果待回填。
- 首次有效运行结果：仅 `frozen_b_bank/C100` 完整结束，Final/AAA/Forgetting 为 `82.17/87.959/6.622`。三条 `sa_lora_bank` 均在进入 Task5 时单卡 OOM；该值可作为独立完成结果，但在 Exact-G/bounded-G 缺失时不能用于方法结论。
- 重跑修正：四路全部统一为两卡、每卡 batch64（有效 batch128），避免 SA-LoRA 可训练历史 scale 的 O(T) 前向图在单卡 batch128 下超过 24GB；不修改模型、优化器、epoch 或任务顺序。

## 2026-09-21：HOEP-A 研究立项（尚未训练）

- 实验标题：历史有效算子能量驱动的稳定/可塑 A 坐标分区。
- 尝试方法：先把共享 A canonicalize 为 row-orthonormal basis，并对规范历史系数的 `C^T C` 做谱分解；跨全部 Q/V 分支选择累计能量不超过 5% 的低能量坐标作为可塑集合，其余 A 行在当前任务中固定。任务边界继续使用 LS alignment 和 operator-preserving absorption。
- 理论依据：在规范坐标中，每个特征值精确等于该方向对部署历史算子 Frobenius 能量的贡献；若只移动被选低能量方向，LS alignment 后不可恢复历史算子能量由所选谱能量之和上界控制。该机制是方向选择，不是旧 Adaptive-A 的全方向 scalar gate。
- 现有论文支持与风险：SplitLoRA 已按历史梯度奇异值划分 major/minor space；LoDA 已从 projection energy 构造共享/隔离子空间；Geo-LoRA 已做 shared LoRA 的几何 core/slack 演化；Share 已动态维护共享 foundational subspace。故本项目仅把“部署历史算子能量 + 固定 O(1) 状态 + 全局 LS recoverability bound”的组合视为待验证差异。
- 当前结果：无。此条仅记录预注册设计，禁止在后续汇总中计作性能实验。
- 下一步：先做 P0 公式级 prior-art 审计和已有 checkpoint 谱诊断；只有谱呈现可用的中间分区，才实现训练路径。

## 2026-09-21：HOEP-A P3 三数据集端点配对（运行中）

- 实验标题：HOEP-A 与 Frozen-A、Live-A、ratio Adaptive-A 的同协议单 seed 筛选。
- 理论依据：P3 检验按历史部署算子能量释放低能量 A 坐标，是否能在不改变固定状态、任务边界对齐和吸收规则的情况下改善 Frozen/Live 两个端点的折中。
- 配置：CIFAR-100 seed1993、ImageNet-R seed1995、CUB-200 seed1；T=10、rank10、SGD、20 epoch。全部使用双卡 DDP、每卡 batch64，等效 batch128，不使用单卡 batch128。
- 控制变量：统一 prototype classifier、固定 `(A,G)`、`live_a_aggregate_b`、LS coordinate alignment 和 `operator_preserving_absorb`；统一关闭 prototype transport、Dual-B、HBD 与 current-branch normalization。HOEP 的全局历史能量预算固定为 5%。
- 运行信息：提交 `1b9d28d`；runtime `.runtime_hoep_p3_20260921_1500`；GPU 双卡槽为 `0,1`、`2,3`、`4,5`，共 12 项自动调度。
- 当前结果：三条 Frozen-A 首批任务已经进入 Task0 且无运行错误。Final、AAA 和 Forgetting 待完整队列结束后统一回填；中途指标不用于选择方法或预算。
- 中断记录：首轮 runtime `.runtime_hoep_p3_20260921_1500` 因服务器中断终止，没有任何完整 T=10 结果。C100/INR/CUB Frozen-A 分别仅完成 Task0-6、Task0-2、Task0-8；这些中途曲线不得进入最终表格。
- 重启记录：新 runtime `.runtime_hoep_p3_20260921_164138_no23` 使用提交 `6380f77`，先运行三数据集 HOEP-A，再依次运行 Frozen-A、Live-A、ratio Adaptive-A。仅使用 GPU `0,1`、`4,5`、`6,7`，继续保持双卡每卡 batch64。

## 2026-09-22：Sensitivity-Budgeted G Consolidation P0（实现完成，真实 smoke 待运行）

- 实验标题：固定 Task-0 输入基底下的功能敏感、风险受限 G 合并。
- 尝试方法：Task 0 对 A 做精确 QR canonicalization；后续固定 P，只训练当前 B/scale。任务边界以投影 activation covariance 和分支输出梯度平方构造历史响应风险，通过 FP64 闭式解与对偶二分选择累计 G。
- 理论依据：Frozen-A 的稳定结果说明继续控制 A 未必是主要矛盾；additive G merge 仍没有区分输出方向的历史功能敏感度。SBGC 直接约束固定坐标下的 branch-response drift，同时保持 `(P,G,C,f)` 为任务常数状态。
- 对照与归因：独立 merge mode 硬性排除 Adaptive-A、transport、HBD、NormCap、Dual-B 与 normalized current branch。uniform sensitivity 是关键消融，`shadow_only` 只算候选而部署 additive Frozen-P。
- 已完成：31 个 CPU 数学/集成测试全部通过；两进程 DDP smoke 通过，Task 1 部署风险不超过 0.05，两个 rank 状态 hash 一致，state save/rebuild logits 一致。
- 当前结果：尚无真实数据性能结果。下一步只运行三个数据集 Task 0/1、双卡每卡 batch64 的 P0；P0 未通过时不得启动 T=10。
- P0 结果：三数据集均完整通过。Task 0 operator error 为 `1.23e-7` 左右；Task 1 均有 24/24 分支激活且 max deployed risk 为 `0.05000000x`。C100/INR/CUB 的 mean target distortion 分别为 `0.4168/0.5656/0.5835`，Fisher/uniform gap 为 `0.0698/0.0395/0.0621`。
- P0 判断：状态、DDP、风险和确定性实现通过，但 5% 预算对当前目标的扭曲偏大。按预注册协议进入 P1 shadow diagnostic，不能据两任务准确率作性能结论。

## 2026-09-22：SBGC P1 Frozen-P Shadow Diagnostic

- 实验标题：固定 canonical P 下的 additive、uniform-budget 与 Fisher-budget G 三候选影子诊断。
- 方法：实际部署始终为 additive `G_old+sB`；uniform 与 Fisher 候选只计算、记录，不参与后续前向。三个数据集严格使用 T=10、rank10、20 epoch、双卡每卡 batch64 和预注册 seed。
- 理论目的：验证 5% 历史 branch-response 风险是否经常被 additive merge 违反，diagonal Fisher 是否实质改变候选，以及约束后的 current-target distortion 在完整序列上是否可接受。
- additive shadow 结果：C100 `87.83/92.321/7.911`，INR `78.92/82.181/7.182`，CUB `84.19/89.469/8.663`，顺序为 Final/AAA/Forgetting。
- 诊断结果：三个数据集 active transition、Fisher/uniform different transition、high-CV branch 比例均为 100%。Fisher distortion 中位数为 `0.10575/0.05797/0.07349`，median-of-medians 为 `0.07349`。
- 判断：`scripts/analyze_sbgc_shadow.py` 的六项预注册检查全部通过，决策为 GO，允许进入 P2。该阶段没有部署风险约束，不能声称 SBGC 已提升性能。
- 限制：Fisher 与 uniform distortion 使用不同加权度量，不能直接跨列比较优劣；必须以 P2 的准确率、AAA 和遗忘作结论。

## 2026-09-23：CUB Task 2/6 单任务谱子空间因果干预

- 实验标题：同一历史状态下比较 Frozen-A、Live-A、Raw-gradient Spectral、Stable-demand Spectral。
- 理论依据：以历史有效算子的最小二乘可恢复损失 `R_rec` 约束 row-space 变化；用当前任务梯度的右奇异空间提供新任务需求。Raw 用一批 32 个类均衡样本，Stable 用 160 个样本/5 批累计梯度；两者的候选基底均须满足每个 Q/V 分支 `R_rec <= 0.05`。该约束仅涉及历史分支，不约束当前 `sBA` 对旧类的功能干扰。
- 协议：CUB-200 seed1、rank10、20 epoch、SGD lr0.01、双卡每卡 batch64。Task 2 四组均从完成 Task 1 的同一 `sa_state.pt`/FC/prototype 启动；Task 6 四组均从完成 Task 5 的另一份同源状态启动。两份来源均为此前 SBGC attribution 运行，不是单独训练的 Frozen-A checkpoint。固定历史有效算子参与训练前向，任务结束才投影到最终 row-space；此处 Live 是受控干预端点，不是原项目完整 Live-A 流程。
- 测量：`Delta L_new` 是新类测试样本的全局线性训练头 CE 后减前；新/旧 Top-1 是所有已见类别上的 prototype 头准确率。新类 prototype 在任务自身训练集上计算，旧类 prototype 不更新。`R_rec` 为任务边界历史算子投影残差，而非旧类准确率或全网络遗忘上界。

| Task | 方法 | Delta CE_new | Delta Top1_new (pp) | Delta Top1_old (pp) | Final new/old Top1 (%) | Global R_rec / max branch |
|---|---|---:|---:|---:|---:|---:|
| 2 | Frozen | -9.075 | +2.027 | -0.691 | 93.75 / 90.67 | 0 / 0 |
| 2 | Live | -9.075 | +2.027 | -0.604 | 93.75 / 90.76 | 0.0001 / 0.0007 |
| 2 | Raw | -9.071 | +4.223 | -51.641 | 95.95 / 39.72 | 0.0244 / 0.0491 |
| 2 | Stable | -9.071 | +4.054 | -54.318 | 95.78 / 37.05 | 0.0193 / 0.0470 |
| 6 | Frozen | -10.921 | +5.546 | -1.586 | 91.23 / 86.27 | 0 / 0 |
| 6 | Live | -10.922 | +5.367 | -1.442 | 91.06 / 86.41 | 0.0004 / 0.0010 |
| 6 | Raw | -10.883 | +8.587 | -35.939 | 94.28 / 51.92 | 0.0209 / 0.0416 |
| 6 | Stable | -10.912 | +0.537 | -0.461 | 86.23 / 87.40 | 0.0209 / 0.0493 |

- 验证：8/8 结果均为 20 epoch、24 个分支风险记录、同 Task 内一致的来源 SHA 和训练前旧类准确率；两种 Spectral 的每分支风险均不超过 5%。两项算子级测试通过，日志无异常。正式结果保存在 `CAUSAL_CUB_INTERVENTION_20260923/results/`，代码提交 `28a695d`；cuda11 使用显式 SSH identity `~/.ssh/id_ed25519_pro6000`。
- 分析：Task 2 的 Stable 比 Raw 旧类再低 2.68 pp，且两者均远差于 Frozen；Task 6 的 Stable 比 Raw 多保留 35.48 pp 旧类，但新类低 8.05 pp。Task 6 Stable 的按样本数加权已见类 Top-1 为 87.23%，仅比 Frozen 的 86.96% 高 0.27 pp；Task 2 则为 56.91% 对 Frozen 的 91.71%。两任务的旧类大幅下降已在 merge 前出现，不能归咎于任务边界 LS 投影。
- 决策：预设跨两个 transition 的稳定收益条件未满足，停止把此谱方法直接扩为正式 T=10 实验。先诊断新任务 `sBA` 的功能干扰和旧 prototype/头失配；不要把 `R_rec <= 5%` 解释为旧类损失 <=5%。

## 2026-09-23：CUB 谱干预的旧类下降反事实归因

- 实验标题：固定已训练模型，分别关闭当前 `sBA`、恢复未投影的历史算子，并限制 prototype 预测范围，定位 Task 2/6 的旧类损失。
- 方法与依据：六组 Frozen/Raw/Stable 按上一节相同数据、种子、双卡每卡 64、20 epoch 重跑；训练路径不变。训练完成后使用同一组新旧 prototype，测量完整部署前向、关闭当前分支但保留已对齐历史算子、关闭当前分支且恢复原历史算子。另计算完整模型的旧类限定标签 Top-1；这只诊断跨新旧类竞争，不能代替全局 CIL Top-1。旧测试集只用于评估，未参与训练或候选选择。
- 理论检验：若完整前向与“仅历史”相差大、而已对齐与原历史几乎相等，则历史 LS 不可恢复性不是主要旧类损失来源；若旧类限定标签远高于全局准确率，则旧样本主要被新类 prototype 抢走，而非完全失去旧类间可分性。

| Task | 方法 | 完整全局旧类 | 完整旧类限定 | 关闭当前分支 | 原历史且关闭当前 |
|---|---|---:|---:|---:|---:|
| 2 | Frozen | 90.67 | 93.09 | 91.62 | 91.62 |
| 2 | Raw | 39.72 | 88.60 | 93.09 | 93.09 |
| 2 | Stable | 37.05 | 88.51 | 93.01 | 93.09 |
| 6 | Frozen | 86.27 | 88.78 | 88.09 | 88.09 |
| 6 | Raw | 51.92 | 86.13 | 88.84 | 88.81 |
| 6 | Stable | 87.40 | 88.49 | 88.00 | 87.97 |

- 结果：六组重跑的完整旧类 Top-1 与上一轮 JSON 精确一致。Task 2 的 Raw/Stable 关闭当前分支后旧类恢复约 53/56 pp；Task 6 Raw 恢复约 37 pp。对这三组，仅把已对齐历史替换为原历史的差异不超过 0.09 pp。完整模型中，Task 2 Raw/Stable 的旧类限定标签仍约 88.5%，但全局仅约 40%/37%；Task 6 Raw 也为 86.13% 对 51.92%。
- 判断：严重损失来自当前分支改动旧样本特征后与新 prototype 发生的跨任务竞争，而不是 LS alignment；受限标签指标不能作为论文主结果，且“分支影响”和“新 prototype 竞争”存在交互，不宜当作两个独立可加的误差项。线性 FC 的旧类准确率在 Frozen 下也明显下降，是另一项头漂移诊断，不解释 prototype 头特有的大幅下降。
- 验证与边界：新增两项反事实单元测试，focused suite `4 passed`；六组均无异常、任务/来源一致、完整 20 epoch。原训练未保存模型权重，本次通过严格同协议复现再执行后验只读诊断。明细保存在 `CAUSAL_CUB_INTERVENTION_20260923/counterfactual/`，代码提交 `72affff`。这是 CUB 单 seed 两个 task 的机制诊断，不是跨数据集方法收益结论。

## 2026-09-23：CUB Task 2/6 真实旧/新分支门控诊断

- 实验标题：只用当前任务训练数据校准的双前向 prototype 门控，检验 oracle 分支收益能否实际兑现。
- 方法与依据：在上一节同源 Frozen/Raw/Stable 的 20 epoch 训练后，历史单分支计算旧类最大余弦，完整分支计算新类最大余弦；二者之差作为“新类”分数。当前任务训练集采用固定种子、按类别 80/20 分层划分：80% 构造新 prototype，20% 仅校准阈值。主阈值固定为校准集新类召回至少 95%；90%/99% 是预设敏感性点。旧类训练图像、旧/新测试标签均不参与阈值确定；全局 prototype 分类不使用任务标签 mask。
- 误路由代价定义：旧类代价是误送完整分支时历史单分支正确率损失，按全部旧类样本归一成 pp；新类代价是误送历史单分支时完整分支正确率损失，按全部新类样本归一。两者按旧/新样本数加权之和等于 oracle 与真实门控的整体 Top-1 差距。

| Task | 方法 | AUC | 完整 Top-1 | Oracle Top-1 | 门控 Top-1 (95%) | 旧误路由 / 新误路由 | 旧 / 新误路由代价 (pp) |
|---|---|---:|---:|---:|---:|---:|---:|
| 2 | Frozen | 0.9938 | 91.43 | 92.17 | 91.14 | 1.30% / 8.78% | 0.00 / 3.04 |
| 2 | Raw | 0.9867 | 59.09 | 94.06 | 88.46 | 8.72% / 3.72% | 6.74 / 3.38 |
| 2 | Stable | 0.9880 | 57.49 | 94.11 | 89.60 | 5.87% / 5.07% | 4.58 / 4.39 |
| 6 | Frozen | 0.9864 | 86.91 | 88.50 | 87.28 | 5.25% / 4.65% | 1.27 / 0.89 |
| 6 | Raw | 0.9775 | 57.68 | 89.47 | 82.24 | 10.41% / 5.19% | 7.82 / 3.58 |
| 6 | Stable | 0.9879 | 86.98 | 87.56 | 87.23 | 6.32% / 4.11% | 0.37 / 0.00 |

- 敏感性：90%/95%/99% 校准点下，Task2 Raw 门控为 `89.43/88.46/86.57`，Stable 为 `89.31/89.60/80.00`；Task6 Raw 为 `84.50/82.24/73.85`，Stable 为 `87.31/87.23/87.16`。不能依据测试准确率反选阈值。
- 判断：AUC 达 `0.978--0.988` 仍不足以保证高代价旧类样本不被误送完整分支。Task2 两种谱方法均比配对 Frozen 完整准确率 `91.43` 低至少 `1.83 pp`，距各自 oracle `4.51--5.60 pp`；Task6 Raw 同样失败。Task6 Stable 距自身 oracle 仅 `0.33 pp`，但只比 Frozen 完整结果高 `0.32 pp`，且 Frozen 自身门控为 `87.28`。因此门控在局部可行，尚不具备跨 transition 的稳定主线收益；按预定分流转向训练期 current branch 功能干扰约束，先做 Task2/6 单变量因果验证，不直接扩展多 seed。
- 限制：这是 CUB seed1 的两个 transition，不是完整 T=10；校准样本虽不参与 prototype 均值，但参与了既有 LoRA 训练，属于允许的当前训练数据，可能使训练校准估计偏乐观。门控还需要每个样本两次 ViT 前向，不能直接声称推理加速。此诊断的 80% 新 prototype 与前两节的全训练集 prototype 不同，因此表内同协议配对比较有效，跨表数值不可直接相减。
- 验证：六组训练前来源 hash 与原实验逐项一致，完整训练后的旧/新准确率也与前轮数值完全一致；459 项仓库测试通过。明细位于 `CAUSAL_CUB_INTERVENTION_20260923/gate_diag/results/`。

## 2026-09-23：真实主方法的 T=3 当前分支侵入诊断

- 实验标题：Frozen-A + CoordinateStable + bounded NormCap + prototype transport，在当前 `sBA` 合并前关闭/开启该分支的旧新类反事实。
- 理论依据：前述谱干预的旧类跨任务误判可能是谱方法特有，不能直接归因于主方法；同一模型、同一 prototype/头下仅关闭当前分支，检验主方法是否也有训练期旧类侵入。
- 协议：CUB seed1 与 CIFAR-100 seed1993，各 Task0-2；rank10、20 epoch、SGD、双卡每卡 batch64，分别使用 GPU 4/5 与 6/7。只对 Task1/2 做旧类诊断，测试样本只用于评估。
- 结果：CUB Task1/2 的旧类历史到完整 Top-1 为 `90.96->90.61`、`91.88->91.62`，正确旧类跨新类错误各为 `0.35%`；C100 Task1/2 为 `96.00->94.70`、`94.85->93.50`，相应跨类错误为 `2.30%`、`1.55%`。新类准确率四次均上升。完整逐样本 logits、每任务合并前后状态和日志已保存，六个任务快照通过 SHA256 审计。
- 判断：主方法确有一定的当前分支跨类竞争，C100 强于 CUB，但 T=3 不足以说明后续趋势；不能把此前谱方法的几十 pp 灾难性下降推广到主方法。详见 `frozen_branch_intrusion_sequence_20260923.md`，代码提交 `84ff9b1`，归档提交 `1c27213`。

## 2026-09-23：真实主方法 T=10 分支侵入序列（运行中）

- 实验标题：在完整十任务序列上检验当前 `sBA` 对历史类别的跨任务误判是否累积。
- 理论依据：T=3 的 CUB/C100 信号大小不同，需要观察 Task1-9 的逐任务曲线，区分短期局部侵入与长期累积风险；不修改现有训练算法。
- 协议：与上述 T=3 唯一配置差异是任务数、实验名和输出目录。CUB seed1、C100 seed1993；T=10、rank10、20 epoch、双卡每卡 batch64。GPU 4/5 与 6/7 并行训练，结束后各自顺序对 Task1-9 生成只读分支诊断；所有 Task0-9 保存合并前后快照。
- 运行信息：提交 `761e1fd`；unit `frozen-branch-intrusion-t10-20260923.service`；队列日志 `frozen_branch_intrusion_t10_queue_20260923.log`；结果目录 `FROZEN_BRANCH_INTRUSION_CUB_T10_SEED1_20260923/` 与 `FROZEN_BRANCH_INTRUSION_C100_T10_SEED1993_20260923/`。22:06 两组均进入 Task0 Epoch1，无启动错误。
- 当前结果：无完整 T=10 结果，暂不作性能判断。验收时检查每组十份任务快照、九份逐样本 logits、逐任务全局 Top-1、旧类正确到新类错误比例及新类收益；若训练失败，不使用部分序列下结论。
