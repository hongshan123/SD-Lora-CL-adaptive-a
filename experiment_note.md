# Agent 实时实验笔记

用途：按时间追加 Agent 的即时观察、假设、判断和下一步动作。这里允许记录尚未证实的想法；正式可复现结果应整理到 `experiment.md`，稳定计划应同步到 `plan.md`。

记录格式：

```text
## YYYY-MM-DD HH:MM CST
- 观察：
- 判断：
- 风险：
- 下一步：
```

---

## 2026-08-04 19:40 CST

- 观察：第一阶段 task prototype 在 ImageNet-R/CIFAR-100 的任务路由只有约 `46.43%/53.98%`，但 oracle/global 为 `78.54%/88.45%`。
- 判断：单个任务平均 prototype 把一个多模态任务压成一个中心，信息损失过大；应保留类别级模式。
- 风险：任务标签由随机类别集合构成，本身可能没有紧凑的任务语义簇。
- 下一步：实现 class prototype、任务内 max score 和 Top-2 候选。

## 2026-08-04 19:46 CST

- 观察：class-prototype 路由代码、完整专家恢复、任务头保存和离线复评器已完成。
- 判断：保持新模型 `class_proto_routed_sdlora` 独立，避免改变原始 `sdlora`、K-CMS 和第一阶段路由行为。
- 风险：Top-2 联合分数中的任务相似度与分类 logits 量纲不同，固定温度可能不合适。
- 下一步：先固定 `router_temperature=0.07`、`classifier_temperature=1.0`，用已有专家做无重训复评。

## 2026-08-04 19:52 CST

- 观察：ImageNet-R 离线 class route Top1/Top2 为 `67.34%/77.26%`，相比 task prototype 明显提高；class Top1/global 为 `75.41%`。
- 判断：类别 prototype 方向有效，但 global 准确率几乎没有随路由准确率提高。
- 风险：独立专家可能缺少任务专属性，global 分类头对选中专家不敏感。
- 下一步：检查 random/global、oracle/global 和 task-head 上限。

## 2026-08-04 19:54 CST

- 观察：CIFAR-100 离线路由达到 Top1 `81.96%`、Top2 `91.38%`；oracle/task-head 达到 `98.39%`，但 Top2/task-head 只有 `81.84%`。
- 判断：候选召回不是唯一瓶颈，任务概率和任务内类别概率的联合校准更加关键。
- 风险：错误 hard route 会把真实类别设为负无穷，错误代价远大于 global head。
- 下一步：完成正式增量训练并比较离线复评一致性。

## 2026-08-04 20:32 CST

- 观察：ImageNet-R 完整 class-prototype 训练成功完成，class Top1/global `75.48%`，Top2/task-head `67.34%`，oracle/task-head `92.01%`。
- 判断：离线和完整训练结果一致。路由提高约 21 个点，但 global 准确率与旧路由基本相同。
- 观察：random/global `74.69%`，只比 class Top1/global 低 `0.79`；oracle/global `78.61%`。
- 判断：专家选择对 global head 的影响弱，当前 LoRA 专家没有形成足够强的任务差异。
- 风险：如果继续只优化 prototype route，可能提高路由指标但无法提高最终 CIL 指标。
- 下一步：同时研究专家专属性和任务头校准，不把路由准确率当成最终目标。

## 2026-08-04 20:55 CST

- 观察：raw、样本均值中心化和 prototype 均值中心化的路由差异很小；对角标准化把 CIFAR-100 Top1 从约 `81.96%` 提到 `83.16%`，ImageNet-R 只提高约 `0.18`。
- 判断：当前代码已经做了样本加权全局均值中心化。仅更换公共均值不是重要方向，对角尺度校准更有价值。
- 风险：类别 prototype 数量小于 768 维，完整协方差白化秩不足。CIFAR-100 类间有效秩仅约 `25.32`。
- 下一步：优先接入 diagonal standardization；完整白化只考虑 shrinkage/power/低秩版本。

## 2026-08-04 21:15 CST

- 观察：原 SD-LoRA 没有保存训练后的 scale，使用默认 scale 恢复后，ImageNet-R 类别 Top1 为 `78.21%`，距离原日志 `78.76%` 为 `0.55`。
- 观察：将 200 类 logits 聚合成 10 个任务后，max route 为 `79.78%`，LogSumExp route 为 `79.71%`。
- 判断：SD-LoRA 的大部分错误是跨任务错误；任务空间从 200 类缩减到 10 个随机类别集合，并没有自动带来很高的任务识别率。
- 判断：以 oracle/task-head `92.01%` 估计，`79.78%` hard route 的最终结果约为 `73.41%`，仍不足以超过 SD-LoRA。
- 风险：如果用 SD-LoRA Top1 类别反推任务，再对同一 logits 做 mask，预测不会改变，没有实际收益。
- 下一步：把 SD-LoRA 用作 Top-k 候选生成器，融合 prototype 和任务头 confidence 进行 rerank。

## 2026-08-04 21:34 CST

- 观察：CIFAR-100 完整训练已完成 Task 0-4，阶段主结果为 `99.10, 95.30, 93.13, 91.05, 88.74`，Task 5 正在运行。
- 判断：训练进程健康，不能用阶段性结果宣称最终提升。
- 风险：当前主返回模式是 Top2/task-head joint，不等同于 class Top1/global；最终分析必须同时读取各评估模式。
- 下一步：CIFAR-100 结束后立即记录退出状态、8 种模式、路由曲线、Average Accuracy 和 Forgetting，并与离线结果和 `86.89%` 基线对齐。
