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
- 阶段 C：INR seed1995 的 rank10/rank16/affine/d0.9 均通过；C100 affine rank10 最佳（92.46/7.12）。迭代中：C100 d0.9。
- 阶段 D：CIFAR-100 seed1993 运行；双数据集通过后多 seed + 强基线 + 消融 + 测量。
- 阶段 E：归档三份 md、完整日志与复现测试。

## 5. 参数/存储口径（当前基线）

- EXP-009：LoRA 2,027,530 + INR 原型 153,600 = 2,181,130（C100 原型 76,800 = 2,104,330）。剩余预算（按 INR）：30,710。
- 持久 transport 若存 U,V ∈ R^{768×r_T}：r_T=16 时 24,576 ≤ 30,710；r_T=20 时 30,720 超预算。首选 r_T ≤ 16，且 transport 仅在任务内驻留、应用后即丢弃（持久化状态仅原型），训练参数与峰值显存单独报告。
