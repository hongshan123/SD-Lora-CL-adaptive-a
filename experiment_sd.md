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
- 下一步：C100 affine（运行中）；若 C100 仍不过，补充 full-rank 消融并转向其他机制（如 LoRA 结构驱动的漂移基）。
