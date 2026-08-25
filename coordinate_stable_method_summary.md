# CoordinateStable: 面向持续学习的坐标稳定 SD-LoRA

## 1. 工作目标与核心结论

本工作基于 SD-LoRA，研究如何在类别增量学习中以更少的任务相关 LoRA 状态保持较高的最终分类性能。当前主方法为 **Live-A Aggregate-B + Dual-B**，并加入两个互补机制：

1. **Effective-operator Coordinate Alignment**：当共享的 LoRA 下投影 `A` 更新后，将历史聚合上投影转换到新的 `A` 坐标系。
2. **Gated Residual Orthogonal Prototype Transport**：利用当前任务训练前后的冻结特征估计低秩漂移，在验证集收益为正时才变换历史类别 prototype。

截至目前，CoordinateStable 已在 CIFAR-100、ImageNet-R 与 CUB-200 的严格同协议三种子实验中完成验证。相对前一版 Live-A+Dual-B，方法在 ImageNet-R 与 CUB-200 上同时提高 Final 和 AAA、降低 Forgetting；在 CIFAR-100 上保持 Final 基本不变，并改善 AAA 与 Forgetting。

## 2. 问题设定

采用 class-incremental learning（CIL）设置。冻结预训练 ViT-B/16 主干，只在每个 Transformer block 的 Q/V 投影使用 LoRA，12 个 block 共 24 个 LoRA 分支。每个分支秩为 `r=10`，输入/输出维度均为 768。

目标是在任务持续到来时：

- 避免为每个历史任务保留完整 LoRA；
- 保持历史类别的分类边界和 prototype 坐标一致性；
- 用全局类别空间评估，而非 task-aware 标签掩码；
- 以 Final Top-1、AAA 与 Forgetting 共同衡量性能。

## 3. 主体方法

### 3.1 Live-A Aggregate-B 参数化

标准任务 LoRA 的增量可写为：

`Delta W_t = s_t B_t A_t`。

本方法将当前任务的 `A_t` 作为 live（可训练）下投影，而历史任务不再保留独立的 `A_i/B_i` 对，仅以聚合算子 `G` 表示历史适配：

`y_hist = G A_live x / ||A_live||`。

当前任务分支仍为：

`y_current = s B_current (A_live x)`。

训练完成后将当前任务的归一化上投影写入历史聚合状态：

`G_new = G_old + s B_current / ||B_current||`。

因此，任务数量增长时，不需要继续保存逐任务 LoRA 矩阵。对 24 个 rank-10、768 维 Q/V 分支，持久 LoRA 矩阵状态为：

`24 * (10*768 + 768*10) = 368,640` 个参数，外加少量 scale。这里的 O(1) 指的是**任务索引 LoRA 适配器状态**；分类头和类别 prototype 仍随类别数线性增长。

### 3.2 Effective-operator Coordinate Alignment

Live-A 结构的关键风险是：每个新任务更新共享 `A_live` 后，历史 `G` 仍处于旧坐标，导致历史有效算子发生非预期改变。

对每个 LoRA 分支，记更新前后归一化下投影为 `A_old_norm`、`A_new_norm`，历史有效算子为：

`W_hist_old = G_old A_old_norm`。

我们求解最小二乘问题：

`X* = argmin_X ||X A_new_norm - G_old A_old_norm||_F^2`，

并以 `G_aligned = X*` 替换历史聚合状态。这样在新的共享下投影坐标下，尽可能保持历史有效算子不变。该机制直接处理的是 LoRA 参数分解带来的坐标漂移，而非仅在特征空间做事后补偿。

### 3.3 Gated Residual Orthogonal Prototype Transport

即使历史 LoRA 有效算子被对齐，当前任务训练仍可能使冻结评估表征发生局部变化，导致旧类别 prototype 与新特征空间失配。

具体流程如下：

1. 在训练当前任务前，用冻结评估 backbone 提取当前任务样本特征；
2. 当前任务训练与 LoRA 聚合完成后，再提取同一批样本的特征；
3. 计算配对残差并通过 SVD 提取主要漂移子空间 `U`；
4. 在该子空间中以带单位阵正则的 Procrustes 问题求旋转 `R`；
5. 对历史 prototype 施加低秩残差变换：

`p_new = p + U(R - I)U^T p`。

为降低由单一新任务估计全局漂移的风险，使用确定性 80/20 划分：仅当 held-out 对齐误差确实下降时启用 transport；否则保持原 prototype 不变。变换只用于更新已有历史 prototype，不保留可累积的任务级 transport 状态。

### 3.4 Dual-B 分类头

模型同时保留 FC head 与 prototype/cosine head。训练和任务过程中按既定调度融合二者，最终测试阶段以 prototype head 为输出。该设计服务于压缩后表征空间的类中心稳定性，而不是额外的任务路由假设。

## 4. 实现状态

当前实现在 `backbone/sa_lora.py`、`backbone/coordinate_stability.py` 与 `models/sa_sdlora.py` 中，主配置启用：

```json
{
  "sa_cumulative_state": true,
  "sa_cumulative_merge": "live_a_aggregate_b",
  "sa_live_a_history_groups": 1,
  "sa_live_a_coordinate_align": true,
  "sa_coordinate_stable_transport": true,
  "sa_coordinate_transport_rank": 10,
  "sa_coordinate_transport_reg": 0.0001,
  "sa_coordinate_transport_min_gain": 0.0,
  "sa_dual_head": true
}
```

训练均采用确定性设置。Prototype transport 的统计量、是否通过 gate 与对应误差会写入实验输出，便于后续消融与错误分析。

## 5. 已完成的严格同协议结果

下表中 CoordinateStable 与旧 Live-A+Dual-B 使用相同任务顺序、学习率、epoch、batch size、分类头和三种 seed；仅增加坐标对齐与 prototype transport。

| Dataset | Method | Final Top-1 | AAA | Forgetting |
| --- | --- | ---: | ---: | ---: |
| CIFAR-100 | Live-A+Dual-B | 87.43 | 91.37 | 9.23 |
| CIFAR-100 | CoordinateStable | 87.40 +/- 0.62 | 91.77 +/- 0.53 | 8.09 +/- 1.18 |
| ImageNet-R | Live-A+Dual-B | 78.96 | 83.06 | 7.99 |
| ImageNet-R | CoordinateStable | 79.34 +/- 0.36 | 83.71 +/- 0.45 | 7.55 +/- 1.19 |
| CUB-200 | Live-A+Dual-B (P5) | 75.27 +/- 3.12 | 84.37 +/- 1.34 | 16.77 +/- 0.43 |
| CUB-200 | CoordinateStable | 80.39 +/- 1.66 | 86.31 +/- 0.89 | 11.50 +/- 0.10 |

相对 Live-A+Dual-B 的均值变化：

| Dataset | Delta Final | Delta AAA | Delta Forgetting |
| --- | ---: | ---: | ---: |
| CIFAR-100 | -0.03 | +0.40 | -1.14 |
| ImageNet-R | +0.38 | +0.65 | -0.44 |
| CUB-200 | +5.12 | +1.94 | -5.27 |

因此，当前证据支持：CoordinateStable 对 ImageNet-R 与 CUB-200 的最终精度、全程平均精度与遗忘均有正向影响；在 CIFAR-100 上基本不牺牲 Final，并改善 AAA 与遗忘。CIFAR-100 的 Final 变化远小于 seed 波动，应描述为持平而非显著提升。

## 6. 与原始 SD-LoRA 的关系

在当前相同 seed 的已审计结果中：

- CIFAR-100：CoordinateStable 相对 SD-LoRA 的均值 Final 约 `+0.59`，AAA 约 `+0.02`，但 Forgetting 更高约 `+1.53`；
- ImageNet-R：CoordinateStable 相对 SD-LoRA 的均值 Final 约 `+0.83`，AAA 约 `+0.57`，Forgetting 更低约 `-0.74`。
- CUB-200：CoordinateStable 相对 SD-LoRA 的均值 Final 约 `+9.13`，AAA 约 `+4.65`，但 Forgetting 更高约 `+0.53`。

这说明方法的主要优势不是简单追求最低 forgetting，而是在极小的 task-indexed LoRA 状态下保持或提高最终性能，并在较复杂的 ImageNet-R 任务上表现出更一致的稳定性收益。论文中应同时报告参数状态、Final、AAA 与 Forgetting，避免单指标叙事。

## 7. CUB-200 严格协议结果与审计

历史冻结的 CUB-200 主表使用 SGD、constant scheduler、20 epoch、batch size 32、4 GPU NCCL 的三 seed 协议：

| Method | Final | AAA | Forgetting |
| --- | ---: | ---: | ---: |
| P5 Live-A+Dual-B | 75.27 +/- 3.12 | 84.37 +/- 1.34 | 16.77 +/- 0.43 |
| EXP-009 | 75.37 +/- 2.90 | 85.01 +/- 1.79 | 17.15 +/- 0.11 |
| SD-LoRA | 71.26 +/- 1.47 | 81.66 +/- 0.90 | 10.97 +/- 2.17 |
| CoordinateStable | **80.39 +/- 1.66** | **86.31 +/- 0.89** | **11.50 +/- 0.10** |

CoordinateStable 三个 seed 的单次结果为：seed 1 `81.77 / 87.30 / 11.57`，seed 2 `78.55 / 85.59 / 11.39`，seed 3 `80.86 / 86.02 / 11.55`。相对 P5 Live-A+Dual-B，均值变化为 Final `+5.12`、AAA `+1.94`、Forgetting `-5.27`。这构成当前最强的跨数据集证据。

本轮运行严格继承 P5 配置：`optimizer=sgd`、`lrate=0.01`、constant scheduler、20 epoch、batch size 32、4 GPU NCCL；仅额外启用 Coordinate Alignment 与 Prototype Transport。三组日志与输出为：

- 输出目录：`CUB_COORDINATE_STABLE_SGD_SEED{1,2,3}_NCCL/`；
- 日志：`cub_coordinate_stable_sgd_seed{1,2,3}_nccl.log`；
- 三个 seed 均以 status 0 完成。

审计中发现，早先一份名义为 `optimizer=adam` 的 CUB 队列产生了逐项相同的结果。原因是 `models/sa_sdlora.py` 的最终 FC 分类头微调固定使用 SGD，因此配置中的 optimizer 不能完整控制该阶段。论文主表只应采用本节明确记录的 P5-SGD 严格协议；该实现行为也应在复现说明中披露。

## 8. 论文可主张的内容与边界

当前最稳妥的技术叙述是：

1. 提出一种将历史 LoRA 聚合到共享下投影坐标的持续学习参数化；
2. 显式校正 shared-A 更新引起的历史有效算子漂移；
3. 以无历史数据、受 held-out gate 约束的低秩 prototype transport 缓解表征坐标漂移；
4. 在 CIFAR-100、ImageNet-R 与 CUB-200 的严格同协议三 seed实验中，展示压缩 LoRA 状态下的性能保持及稳定性收益，其中 CUB-200 取得同时提升 Final、AAA 并降低 Forgetting 的结果。

当前不应主张：

- 所有状态均为 O(1)；分类头和 prototype 状态仍随类别数增加；
- 坐标对齐与 transport 各自独立贡献，除非完成完整消融。

## 9. 论文前必须补齐的实验

1. 做组件消融：Live-A+Dual-B、仅 Coordinate Alignment、仅 Transport、完整方法；
2. 汇报每个任务的 accuracy 曲线、Final/AAA/Forgetting、参数量和训练/推理成本；
3. 对 transport gate 报告触发率、held-out gain、子空间 rank 与失败案例；
4. 用相同 backbone、rank、任务划分与 seed 对比原始 SD-LoRA 及主要强基线；
5. 说明 CIL 全局分类评估与 task-aware 设置的差异，主文仅用全局分类结果；
6. 修正或明确记录最终 FC head 固定 SGD 的实现语义，避免将 `optimizer=adam` 误写为完整训练流程的优化器。

## 10. 当前复现入口

远程仓库当前分支为 `codex/coordinate-stable-prototype`。严格 CUB SGD 对照的启动脚本为：

```bash
cd /home/zhaoyang/SD-Lora-CL-coordinate
nohup bash ./run_coordinate_stable_cub_sgd_queue.sh \
  > coordinate_stable_cub_sgd_queue.log 2>&1 &
```

进度监控脚本为：

```bash
cd /home/zhaoyang/SD-Lora-CL-coordinate
nohup bash ./monitor_coordinate_stable_cub_sgd.sh \
  > coordinate_stable_cub_sgd_monitor.log 2>&1 &
```

本文档用于论文撰写和后续实验审计。新增实验应先核对有效配置与本文件中的对照协议，避免再次混淆优化器或训练日程造成的不公平比较。
