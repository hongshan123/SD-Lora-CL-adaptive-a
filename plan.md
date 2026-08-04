# SD-LoRA-CL 研究计划

最后更新：2026-08-04 21:34 CST

## 1. 研究目标

在不使用测试任务 ID 的 Class-Incremental Learning（CIL）设置下，构建能够按样本选择任务专属 LoRA 的路由系统，并判断稀疏路由能否稳定超过原始 SD-LoRA。

核心问题：

1. 如何从冻结 Base ViT 或 SD-LoRA 特征中识别输入所属任务？
2. 任务专属 LoRA 是否形成了足够明显的专家差异？
3. 路由到任务头后，如何避免一次错误路由直接排除真实类别？
4. 如何校准任务相似度与任务内分类置信度，使 Top-k 候选真正发挥作用？

## 2. 实验原则

- 主结果使用全局标签空间，不使用真实任务 ID。
- `oracle/task-head` 只表示上限，不能作为 CIL 主结果。
- ImageNet-R 固定 seed `1995`、20 类/任务、10 个任务、batch size `32`。
- CIFAR-100 固定 seed `1993`、10 类/任务、10 个任务、batch size `32`。
- 首轮保持原始 SD-LoRA 的学习率、epoch、任务顺序和分类头一致。
- 路由统计只能使用训练集和确定性测试预处理，不能使用测试标签调参。
- 单 seed 仅用于筛选方向；产生稳定收益后至少补 3 个 seed。
- 同时报告 Top1、Top5、Average Accuracy、Last Accuracy、Forgetting、任务路由 Top1/Top2、专家利用率和混淆矩阵。

## 3. 当前系统

### 3.1 原始 SD-LoRA

- 历史 LoRA 与当前 LoRA 共同参与前向。
- 使用全局分类头完成 CIL 分类。
- ImageNet-R 最终 Top1 `78.76%`；CIFAR-100 最终 Top1 `86.89%`。

### 3.2 第一阶段：Task-Prototype Router

- 每个任务独立训练一套 LoRA。
- 使用冻结 Base ViT 的任务平均 prototype 做 hard Top-1 路由。
- 已完成，路由精度不足：ImageNet-R `46.43%`，CIFAR-100 `53.98%`。

### 3.3 第二阶段：Class-Prototype Top-k Router

- 保存每个类别的 Base ViT 特征和、样本数以及全局统计。
- 对特征和类别 prototype 做中心化余弦相似度。
- 任务分数取该任务内类别相似度最大值。
- 支持 Top-1、Top-2、任务分类头联合分数、完整专家恢复。
- ImageNet-R 完整训练已完成；CIFAR-100 完整训练进行中。

## 4. 分阶段计划

### 阶段 A：建立可信基线（已完成）

- [x] 同协议运行原始 SD-LoRA。
- [x] 保存完整 LoRA、分类头和日志。
- [x] 确认 ImageNet-R/CIFAR-100 基线结果。

验收：训练无报错，10 个任务检查点完整，结果与已有 SD-LoRA 水平一致。

### 阶段 B：验证独立专家与非参数路由（已完成）

- [x] 实现 task prototype hard Top-1。
- [x] 输出 global、task mask、oracle、all-LoRA 结果。
- [x] 判断瓶颈来自路由还是专家。

结论：独立专家的 oracle 上限不低，但单任务平均 prototype 无法稳定识别任务。

### 阶段 C：提高路由分辨率（进行中）

- [x] 使用类别 prototype 替代单任务 prototype。
- [x] 使用类别最大相似度生成任务分数。
- [x] 加入 Top-2 候选和任务头联合分数。
- [x] 完成 ImageNet-R 完整训练。
- [ ] 完成 CIFAR-100 完整训练。
- [ ] 检查完整训练结果与离线复评的一致性。

验收：任务路由显著高于第一阶段，并完整输出 8 种评估模式。

### 阶段 D：特征归一化消融（下一步）

- [x] 离线比较 raw、样本均值中心化、prototype 均值中心化、对角标准化。
- [ ] 将 `raw/centered/diag_standardized` 接入正式配置。
- [ ] 使用多 seed 验证对角标准化收益。
- [ ] 仅在对角标准化稳定后尝试 shrinkage/power whitening。

首选方案：

```text
z_j = (x_j - mean_j) / max(std_j, 0.1 * median(std))
```

不直接使用 768 维完整 ZCA。类别 prototype 只有 100/200 个，协方差秩不足且容易放大低方差噪声。

### 阶段 E：SD-LoRA 与 prototype 混合路由（计划中）

- [x] 评估 SD-LoRA 的 200 类 logits 聚合成 10 个任务分数。
- [ ] 统计 SD-LoRA 任务 Top-2/Top-3 召回率。
- [ ] 融合 SD-LoRA task score、prototype score 和任务头置信度。
- [ ] 比较 hard Top-1、Top-k rerank 和稀疏概率融合。

候选任务分数：

```text
score(task) = alpha * sdlora_logsumexp
            + beta  * prototype_similarity
            + gamma * task_head_confidence
```

### 阶段 F：任务头校准（计划中）

- [ ] 离线扫描 router temperature：`0.1/0.2/0.3/0.5`。
- [ ] 校准各任务分类头的 logit scale 和 bias。
- [ ] 使用 Top-k 条件归一化，避免任务 softmax 对第二候选权重压制过强。
- [ ] 比较 max、LogSumExp、energy、entropy 和 margin rerank。

### 阶段 G：论文级验证（待前述阶段产生正收益后）

- [ ] 至少运行 3 个 seed。
- [ ] 报告均值和标准差。
- [ ] 补充计算量、参数量、专家激活数和推理延迟。
- [ ] 与 SD-LoRA、K-CMS-LoRA、LAMDA-PILOT 中同协议方法比较。
- [ ] 明确区分 CIL、Task-IL 和 Oracle 结果。

## 5. 当前判断与成功标准

当前主要瓶颈不是类别 prototype 的召回能力，而是：

1. 独立 LoRA 的专家差异对 global head 的影响较弱。
2. 错误任务头会完全排除真实类别。
3. 固定 `router_temperature=0.07` 使 Top-2 实际行为接近 Top-1。
4. SD-LoRA 的 10-task 路由约 `79.78%`，仍低于使用任务头超过基线所需的约 `85.60%`。

进入多 seed 阶段前，至少满足以下一项：

- ImageNet-R global CIL Top1 超过 `78.76%`；
- CIFAR-100 global CIL Top1 超过 `86.89%`；
- 或在不使用真实任务 ID 的条件下，Top-k task-head 方法稳定超过对应 SD-LoRA 基线。
