# SD-LoRA 改进研究计划（plan_sd.md）

## 1. 研究目标

在当前仓库的 SD-LoRA 及其变体（CMS-SDLoRA / K-CMS-SDLoRA / Proto-Routed SD-LoRA）基础上持续迭代改进，直到满足下述任一验收标准。

## 2. 验收标准（两个标准满足其一即成功）

### 标准 1：参数压缩 ≥ 40%
- LoRA 总量（所有任务保存的 LoRA 权重 A+B，含共享/聚类记忆）或额外引入参数量（LoRA + 可学习 scale，不含与 SD-LoRA 相同的分类头）相对 SD-LoRA 基线减少 **≥ 40%**。
- 在 **CIFAR-100（cifar224, seed1993）** 与 **ImageNet-R（imagenetr, seed1995）** 两个数据集上，**最终任务 Top1 ≥ SD-LoRA 基线**。

### 标准 2：同参数量提精度
- LoRA 数量/参数量与 SD-LoRA 基线相同（允许增加极小、可忽略的辅助参数，如原型）。
- 两个数据集上 **最终任务 Top1 均 > SD-LoRA 基线**。

### 指标口径
- 主指标：最终任务 Top1（最后一个任务评估得到的全类 Top1）。
- 辅助指标：Average Accuracy (CNN)、Forgetting (CNN)、LoRA 参数量、磁盘存储量。
- 参数量统计：SD-LoRA 基线 = 10 任务 × 24 个低秩对/任务 × (768×10 + 10×768) ≈ 3.686M；每个任务额外有 1 个 scale。

## 3. 基线（2026-08-04 使用当前代码重新跑的官方配置）

| 数据集 | seed | 配置 | 最终 Top1 | AvgAcc | Forgetting |
|---|---|---|---|---|---|
| CIFAR-100 | 1993 | exps/sdlora_c100_seed1993_proto_baseline.json | 86.89 | 91.44 | 5.58 |
| ImageNet-R | 1995 | exps/sdlora_inr_seed1995_proto_baseline.json | 78.76 | 83.13 | 5.61 |

基线为 rank=10、全部 12 层 QKV LoRA、batch=32、4 卡 DDP。

## 4. 总体路线

### 阶段 0：工程与基线（已完成）
- Git 仓库初始化、忽略规则、三份 md 文件。
- 确认 SD-LoRA 基线数值。

### 阶段 1：低成本诊断
- 优先使用已有 checkpoint 做离线评估（不改训练，只改评估/路由），快速暴露瓶颈。
- 已有发现：Proto-Routed 的冻结 ViT 原型可分性差；中心化后 margin 大幅提升，值得做离线验证。

### 阶段 2：参数压缩方向（对应标准 1）
- 降低 rank / 固定正交 A 矩阵 / 层间共享 LoRA / 单低秩记忆合并（CMS 系）等。
- 目标：LoRA 总量减少 ≥ 40%，两个数据集 Top1 不掉。

### 阶段 3：同参数提精度方向（对应标准 2）
- 改进原型路由（中心化/白化/可学习投影/多深度融合）。
- 改进 CMS/K-CMS 合并策略（加权 SVD、scale 处理）。
- 目标：两个数据集 Top1 均超过基线。

### 阶段 4：验收与收尾
- 在验收配置下重跑两个数据集。
- 对比基线、记录参数量与存储量、给出结论。

## 5. 实验原则
1. 每个实验只改一个核心变量；多变量时必须显式记录。
2. 每个实验前必须读取 experiment_sd.md，禁止重复实验。
3. 每次代码修改后立即 git commit，保证可回退。
4. 先离线/低成本验证，再跑完整训练；GPU 紧张时优先 2 卡小批量验证。
5. 每个实验完成后：分析结果 → 检索文献 → 决定下一步。
6. 全程在 experiment_sd.md 和 experiment_note_sd.md 记录。

## 6. 当前状态
- 阶段 0-3 已完成多轮迭代；K-CMS 有损合并路线（EXP-001~003）放弃；Shared-A 路线（EXP-004/005）平均精度接近/超过基线但 final Top1 低 0.4-1.2。
- EXP-006（全类 head-tune）因违反无回放设定判定无效（用户 2026-08-05 反馈）。
- 数据无关的推理期校正（weight_align、余弦归一化）已验证无效（见 experiment_note_sd.md 2026-08-05 10:00）。
- EXP-007 已完成：C100 final Top1=86.90（达标），INR=78.34（差 0.42，未达标）；参数减 45%（达标）。A 持续训练优于 A 冻结，但分类头偏置仍是瓶颈。
- **EXP-009 已完成并通过验收标准 1**：C100 final Top1=88.42（基线 86.89）、INR final Top1=79.34（基线 78.76）；LoRA 参数 2,027,530 = 基线 55%（减少 45%），含原型仍减少 40.8%。全程无回放。
- 若后续需要改善遗忘/平均精度，备选：LDC（ECCV 2024）式原型漂移补偿；A 冻结 + 原型；EXP-008 训练期余弦 fc 头。
- EXP-012（有效 LoRA 算子稳定化）已完成 INR：Final=79.39（较 SD-LoRA +0.63）、AvgAcc=82.20、Forgetting=6.98；工程与恢复审计均通过、参数不变，但未满足第二轮的 AvgAcc/Forgetting 门槛，因此不启动 C100，也不把它作为主方法。

---

# 第二轮研究目标（2026-08-05 用户新目标）

## 1. 核心研究问题

Shared-A 在任务间持续更新时（`sa_train_a_all_tasks=True`），历史类别 prototype 与当前特征空间失配（EXP-009 中 Task 6 单次回落 2.66 即此现象）。优先研究 **LoRA-aware prototype drift compensation / Low-Rank Prototype Transport (LRPT)**：
- 只使用当前任务数据在**模型更新前/后**的配对特征 (z_old, z_new)；
- 估计与共享 A 低秩变化相关的 transport（rank 与 LoRA rank 绑定）；
- 递归更新旧 prototype；不读取、不保存、不回放旧任务样本。

## 2. 碰撞审计（方法边界）

| 方法 | 官方出处 | 核心机制 | 与 LRPT 的结构差异 |
|---|---|---|---|
| SA-LoRA | J. King Saud Univ. Comput. Inf. Sci. 2026 (Springer, s44443-026-00925-x) | 共享 A + 每任务 B，非对称共享；无原型漂移补偿 | LRPT 在共享 A 持续更新时显式补偿旧原型，SA-LoRA 不处理该漂移 |
| CL-LoRA | CVPR 2025 (arXiv:2505.24816) | 共享+任务专属双适配器，KD+梯度重分配；原型余弦分类但推理按任务逐适配器匹配 | CL-LoRA 不做递归原型 transport；LRPT 保持单模型 O(1) 推理并用配对特征估计 transport |
| RanPAC | NeurIPS 2023 (arXiv:2307.02251) | 冻结骨干 + 冻结随机投影 + 类原型 + 去相关 | RanPAC 骨干不更新故无漂移；LRPT 针对持续更新的低秩适配器显式建模漂移 |
| LDC | ECCV 2024 (arXiv:2407.08536) | 学习前向投影网络（通用可学习映射）补偿旧原型，不假设漂移结构 | LDC 是梯度训练的通用的网络；LRPT 是闭式仿射 I+UV^T、rank 绑定 LoRA ΔA、最小二乘拟合、无附加网络 |
| FM-LoRA | CVPR 2025 Workshop DG-EBF (arXiv:2504.08823) | F-LoRA 共享基+任务系数、动态 rank 选择、meta-prompting | 提示/子空间分配路线，无原型 transport 机制 |
| C-LoRA | arXiv:2502.17920 | 单一 LoRA + 可学习路由矩阵 A·R·B + 正交约束 | 路由矩阵路线，无原型与漂移估计 |
| InfLoRA | CVPR 2024 (arXiv:2404.00228) | 注入参数重参数化预训练权重于固定子空间，消除任务干扰 | 参数子空间隔离路线，无原型 transport |
| EASE | CVPR 2024 | 每任务适配器子空间 + 语义引导原型补全（用语义合成旧类新特征） | EASE 用语义/合成路线且推理 O(T)；LRPT 用数据驱动的低秩 transport，推理 O(1) |

LRPT 的明确新机制：**以 LoRA 分解的结构为先验（ΔA 秩 = r ⇒ 特征漂移近似落在秩 ≤ r 子空间），用当前任务更新前后的配对特征做闭式 rank-r 最小二乘拟合，得到仿射 transport I+UV^T，递归作用于全部旧原型**。不使用任何旧任务数据、不训练额外网络、不引入语义合成。

## 3. 验收标准（第二轮）

1. 严格无回放：memory_size=0；不访问旧任务数据；不做全类 head-tune/旧数据校准/隐式 replay；原型/统计量/模型状态允许保存但计入参数口径。
2. 参数预算：最终 LoRA + prototype + 持久化补偿状态 ≤ 2,211,840（SD-LoRA 3,686,400 的 60%）。临时训练模块（如 transport 拟合中间量）单独报告训练参数与峰值显存。
3. 单 seed 技术验收（先 INR seed1995）：
   - INR Final Top1 ≥ 78.76（原始 SD-LoRA），且相对 EXP-009（79.34）回退 ≤0.20（即 ≥79.14）；
   - 同时 Forgetting ≤ 6.26（EXP-009 7.26 - 1.0）或 AvgAcc ≥ 82.97（EXP-009 82.47 + 0.5）；
   - INR 达标后才跑 CIFAR-100 seed1993：Final Top1 ≥ 86.89 且 ≥88.22（EXP-009 88.42 - 0.20），Forgetting ≤ 7.08 或 AvgAcc ≥ 92.57。
4. 多 seed：单 seed 双数据集通过后补 ≥3 seeds，报告均值/标准差与显著性；并补充与 SA-LoRA/CL-LoRA 可比任务划分的强基线。
5. 论文级审计：merged/恢复推理可独立重载且固定输入 logits 一致；参数/FLOPs/训练-推理显存与吞吐实测；消融（EXP-009、普通 transport、LRPT、无 prototype）；不以 oracle/task mask/换 seed/换协议宣称成功。

## 4. 路线

- 阶段 A：碰撞审计与文献（本节完成）。
- 阶段 B：实现 LRPT（配对特征缓存 → 闭式 rank-r transport → 递归更新原型）；单测 + 恢复一致性测试；提交（已完成，commit `4b8ae92`；冒烟通过）。
- 阶段 C（回退后）：generic affine LRPT 为主方法；INR 已过；C100 affine r10 两跑均差 0.04–0.13。迭代中：affine rank12。
- 阶段 C 附注：classwise layerwise sensitivity（用户反馈方案）已实现并冒烟，结果灾难性；按指示 git 回退，继续原方案 affine rank12。
- 阶段 D：CIFAR-100 seed1993 运行；双数据集通过后多 seed + 强基线 + 消融 + 测量。
- 阶段 E：归档三份 md、完整日志与复现测试。

## 5. 参数/存储口径（当前基线）

- EXP-009：LoRA 2,027,530 + INR 原型 153,600 = 2,181,130（C100 原型 76,800 = 2,104,330）。剩余预算（按 INR）：30,710。
- 持久 transport 若存 U,V ∈ R^{768×r_T}：r_T=16 时 24,576 ≤ 30,710；r_T=20 时 30,720 超预算。首选 r_T ≤ 16，且 transport 仅在任务内驻留、应用后即丢弃（持久化状态仅原型），训练参数与峰值显存单独报告。

## 6. 有效 LoRA 算子稳定化（EXP-012）

- class-wise JVP sensitivity 已在 1-epoch 冒烟中触发灾难性遗忘：其 class response 系数没有参数侧的完整映射，不能用来外推任意未来 `Delta A` 的原型位移。
- 新候选不再预测 prototype 漂移，而是在每个 Task t 开始时快照历史分支的精确有效算子：`M_old @ normalize(A_old)`，其中 `M_old=sum_i s_i normalize(B_i)`。训练期间最小化当前历史算子相对漂移 `||M_live normalize(A)-M_old normalize(A_old)||_F^2 / ||M_old normalize(A_old)||_F^2`。
- 该项只使用已保存的旧 B、历史 scale 和当前共享 A；不读取旧图像、不新增持久状态、不改变 Task 0，也不移动 prototype。1-epoch x 10-task 四卡 smoke 已通过：Task 0 无稳定项、Task 1 起各项有限、保存/重建/一致性审计均通过。由于 smoke 中 lambda=0.1 仅贡献约 0.0008（相对 CE=1.976 过弱），完整 INR 使用 lambda=1.0；若仍无指标增益，再以日志中的 raw drift 决定是否提高或放弃该正则。
- 完整 INR（lambda=1.0）已经完成：Final=79.39，AvgAcc=82.20，Forgetting=6.98。相对 EXP-009 只减少 Forgetting 0.28，却使 AvgAcc 降低 0.27，未满足 `F<=6.26` 或 `AvgAcc>=82.97`；此分支停止在 INR，不创建 C100 配置。
- 后续原则：先验证被约束的 raw operator drift 是否与最终旧类精度相关，再决定是否保留任何 operator-level 正则；禁止没有此诊断的 lambda 扫描。
- 控制组：lambda=0 只记录 raw drift（INR seed1995）已配置，运行中；据结果决定是否彻底关闭 operator-level 路线。
- 控制组完成：drift 相关性弱，operator-level 路线关闭；回到 affine LRPT 主线，C100 affine r10 第三次复跑。
- C100 affine r10 三次均差 0.05–0.13，排除方差；迭代中：raw-space 原型变体（INR 先跑）。
- raw-space 原型 INR 失败（task1 52.74），已终止；迭代中：generic affine r10 + adaptive。
- generic affine r10 + adaptive INR 失败（79.13/82.86/6.84，三项均差 0.01/0.11/0.58）；训练期原型一致性正则 INR 失败（78.41/81.91/7.45）。LRPT 局部微调与训练期正则路线全部关闭。

## 7. 新主线：Online Gauge-Aligned Cumulative Shared-A（2026-08-06）

- 完整方案、碰撞边界、验收线、实现计划与停止条件见 `method_revision_sd.md`。
- 核心：把历史 B bank 在线折叠为累计上投影 `H_t` + canonical 下投影 `Q_t^T`，持久状态从 `O(Tdr)` 降为 `O(dr)`（预计总状态减约 86%），再用闭式 gauge alignment 保持历史有效算子；affine LRPT 降级为剩余漂移的 residual correction。
- **Phase A（纯代数等价性）**：完成（commit `c666ac0`）——累计折叠纯函数 + 固定 A 下 bank/cumulative 的算子、feature、logits 等价测试（<1e-5，32 tests passed）。
- **Phase B（在线累计状态）**：完成（commits `27a6574`/`d7ec60a`/`63b267e`）——v2 单文件状态、`_CumulativeSharedAQKV`、保存时在线累计、v1→v2 显式迁移脚本、4 卡 DDP 1-epoch smoke 通过（exit=0，无逐任务 B 文件，LoRA 371,040 = 基线 10.07%，一致性 PASS）。
- **Phase C（gauge alignment）**：完整 INR 已跑两档——cumulative-only 78.78/81.69/7.28（触发停止线，口径检查确认为无 gauge 的固有失真）；cumulative+gauge 79.06/81.77/6.82（Final ≥78.76 ✓、Forgetting 较 EXP-009 -0.44、AvgAcc 仍低 0.70）。保存时 QR canonicalization + `H_old_aligned = H_old Q_old^T Q_new` + 每任务残差/基底旋转/算子保持日志已落地；`cumulative_gauge=false` 消融可用。
- **Phase D（residual LRPT）**：rank-10/bias/λ=1 默认值，不继续扫 rank/damping。
- **单 seed 筛选**：已完成——INR cumulative-only（78.78，停止线已查）、INR cumulative+gauge（79.06）、gauge+LRPT INR（78.49，LRPT 降为消融）、C100 cumulative+gauge（87.70）。**双数据集最低验收线达成**（INR 79.06 ≥78.76、C100 87.70 ≥86.89、状态减 ≥85%、单模型推理）。
- **多 seed**：全部完成——主方法 INR 78.46±0.51 / 82.28±0.47 / 7.86±1.35、C100 87.83±0.13 / 91.52±0.30 / 8.68±0.20；EXP-009 seed 正确配对后：INR Final/AvgAcc 显著小幅下降（p=0.026/0.009），C100 不显著，Forgetting 不变；LoRA 参数 -81.7%。下一步：任务长度/CUB/GPU 测量 → 论文回填（已完成，论文主张按“小幅代价 + O(1) 状态”表述）。

## 8. 论文准备清单（对应 method_revision_sd.md §11）

| 项目 | 状态 |
| --- | --- |
| 多 seed（≥3，mean±std + 显著性） | 运行中（INR/C100 × seeds 1/2/3，6 个完整运行） |
| 任务长度 T=5/10/20/40 | 完成——INR T5/T10/T20/T40=77.54/79.06/77.03/75.31、C100 T5/T10/T20=88.06/87.70/85.63（Final）；Forgetting 随 T 单调恶化，LoRA 状态恒 371,040 |
| 额外数据集（ImageNet-A/CUB/DomainNet） | CUB-200 完成：主方法 79.79/87.69/14.20 vs EXP-009 71.75/84.93/23.31（Final +8.04）；ImageNet-A/DomainNet 待评估 |
| 强基线 | SD-LoRA（有）、EXP-009（有，含多 seed）、SA-LoRA r10/r20（有）、generic affine LRPT（有）；InfLoRA/CL-LoRA/LoRA-DRS/DGS 未实现，列入 limitations |
| 消融 | cumulative-only（有单 seed）、gauge+LRPT（有单 seed）、无 prototype（待定）、EXP-009（有） |
| 测量（FLOPs/吞吐/峰值显存） | GPU 实测完成：主方法 INR/C100 与 EXP-009 均为 1.129e12 FLOPs、~413 img/s、~579 MiB（batch32） |
| 恢复/merged 一致性 | `verify_sa_consistency.py` 已在 INR（seed1995/seed3）、C100（seed1993）、CUB 主方法产物 PASS（diff=0） |
| 相关性分析 | 初算完成（`scripts/drift_forgetting_correlation.py`）：operator drift 弱相关（r=0.26）、控制组负相关（r=-0.73，塑性混杂）、gauge 诊断零方差、LRPT drift 弱相关；论文按此如实报告 |
| 参数量随 T 曲线 | v2 恒定 371,040（LoRA）；v1 线性增长可解析计算，论文阶段制图 |
| 论文初稿 | 首稿完成（`paper_output/first_draft/main.md`，含 spine/evidence/claims/citations/blueprints/rationale 全套构件，commit `3b9b4d6`）；待补：引用核实、训练显存表、图与 LaTeX |

## 9. Live-A Aggregate-B 状态（2026-08-08）

- 机制：v4 O(1) 聚合状态，LoRA 368,640（10%），无逐任务 B/旧数据/task-id/router；Live-A INR seed1995 Final 79.43 / AvgAcc 81.99 / F 7.08；C100 seed1993 Final 88.32 / AvgAcc 91.99 / F 8.19（Stage A 通过）。
- Stage B 多 seed：C100 相对 EXP-009 通过（TOST 等价）；INR AvgAcc 相对 EXP-009 -0.374（p=0.007）未过门槛，相对 SD-LoRA -0.973。
- Stage B2 双头：Schedule B 修复 INR AvgAcc（83.26±0.81，相对 EXP-009 +0.44），但 INR Final 相对 EXP-009 -0.53、C100 Forgetting 相对 SD-LoRA +2.83，严格门槛未过。
- K=2 prototype 后备：INR max 78.56/81.34/7.55、logsumexp 77.51/81.08/7.67，均低于 K=1 与 SD-LoRA，停止。
- 结论：Live-A 不升级为论文主方法；保留为“固定 O(1) 状态、Final 优先”的消融/负结果记录。所有实验日志、统计脚本、文档与 Git 提交已同步。

> ⚠️ 口径修正（2026-08-11）：本节数值来自 Stage B2 的 n=4 配对；其后 P3 以严格确定性协议完成 n=6/n=5 主统计（提交 `c256cc4`），完整方法（Live-A Aggregate-B + Dual-B）成为冻结主方法。主统计口径与修正见第 10 节。

## 10. 冻结起点与下一阶段（2026-08-11，ccfa_next_experiment_guide_sd.md）

- 冻结方法：**Live-A Aggregate-B + Dual-B**（rank-10 Q/V LoRA、共享 live down-projection A、历史 B 在线折叠为累计 G、每任务 fresh B、Dual-B 固定 Schedule B、最终 lambda=1、无回放/无 task-id/无 router/无随任务增长的 LoRA bank）。
- 持久状态：LoRA `368,640`；含 Dual-B 头 INR `675,840`（相对 SD-LoRA 3,686,400 减 81.7%）、C100 `522,240`（减 85.8%）。
- 严格 n=5（seed 1-5；开发 seed 1995/1993 仅作敏感性分析）：
  - INR：完整方法 79.07±0.23 / 83.06±0.52 / 7.68±1.03；SD-LoRA 78.34±0.48 / 82.96±0.87 / 7.79±1.04；EXP-009 79.16±0.32 / 82.63±0.36 / 7.69±0.93。
  - C100：完整方法 87.41±0.32 / 91.06±0.64 / 9.09±0.62；SD-LoRA 86.81±0.56 / 91.45±0.63 / 6.57±0.71；EXP-009 87.58±0.28 / 91.12±0.54 / 9.19±0.47。
  - 完整方法相对 SD-LoRA：INR Final +0.73（p≈0.078）、AAA +0.10、F -0.11；C100 Final +0.60（p≈0.041）、AAA -0.39（p≈0.006）、F +2.52（p<0.001）。
- 主门槛现状：Final 与参数目标通过；**C100 AAA −0.385 > −0.30 未通过，C100 Forgetting +2.52 显著**，因此 P3 未通过，P4–P6 未触发。
- 下一阶段（按任务书）：P0 严格统计已完成（`p0-strict-n5` 分支）；P1 离线机制诊断（Dual-B 头分解、prototype 失配/表示遗忘分解）；P2 唯一允许的新训练组件 Historical-Branch Activation Distillation（C100 seed1993 → INR seed1995，门槛见任务书 §8.4）；通过后冻结并执行 HBD seed1-5 一次性确认；之后 P4 消融/参数匹配、P5 任务长度/第三数据集/强基线/效率。
