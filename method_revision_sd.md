# SD-LoRA 后续方法修正方案

## 1. 文档目的

本文档记录当前最值得继续验证的方法修正，供后续 Agent 实现、实验和论文整理时参考。

主路线不再继续搜索 LRPT rank、damping、raw prototype、class-wise JVP sensitivity 或 operator-stability loss。下一阶段聚焦：

> **Online Gauge-Aligned Cumulative Shared-A LoRA**：把所有历史任务的上投影在线合并为一个累计矩阵，并利用 LoRA 因子空间的闭式对齐，在共享下投影 A 更新后尽可能保持历史有效算子；现有 affine LRPT 仅用于补偿剩余的非线性 prototype drift。

目标是把当前“约 40.8% 持久状态压缩”提升为“约 86% 持久状态压缩”，同时维持或超过 SD-LoRA 的最终准确率，并使方法贡献从通用特征传输转为 LoRA 特有的恒定状态持续学习。

## 2. 当前结果快照

### 2.1 主要基线

| 方法 | 数据集 | Final Top1 | AvgAcc | Forgetting |
| --- | --- | ---: | ---: | ---: |
| SD-LoRA | ImageNet-R | 78.76 | 83.13 | 5.61 |
| SD-LoRA | CIFAR-100 | 86.89 | 91.44 | 5.58 |
| Shared-A + prototype | ImageNet-R | 79.34 | 82.47 | 7.26 |
| Shared-A + prototype | CIFAR-100 | 88.42 | 92.07 | 8.08 |
| affine LRPT rank-10 | ImageNet-R | 79.29 | 83.00 | 6.61 |
| affine LRPT rank-10，3 次同 seed 均值 | CIFAR-100 | 88.54 | 92.48 | 7.17 |

当前 affine LRPT 包含 LoRA bank 和 prototype 时，ImageNet-R 持久状态为 2,181,130，相当于 SD-LoRA 3,686,400 的 59.17%，减少 40.83%。

### 2.2 已关闭路线

- 提高 rank：rank-16 未稳定优于 rank-10。
- damping：0.5、0.9 未同时改善双数据集。
- raw-space prototype：ImageNet-R Task 1 降至 52.74，灾难性失败。
- class-mean transport：拟合残差明显降低，但最终性能下降，属于当前任务过拟合。
- LoRA-aware JVP：ImageNet-R 可过单项门槛，但 CIFAR-100 未通过。
- class-wise JVP sensitivity：出现灾难性遗忘，已回退。
- effective operator stability loss：将算子漂移压低 10 至 30 倍，但 Forgetting 只改善约 0.12，且 AvgAcc 下降。
- generic adaptive transport：ImageNet-R 为 79.13 / 82.86 / 6.84，弱于普通 affine rank-10。
- training-time prototype consistency（EMA cosine，w=0.1）：ImageNet-R 为 78.41 / 81.91 / 7.45，Final 低于原始基线，训练期原型拉近牺牲新任务可塑性。

这些结果说明继续做小范围超参数微调的收益很低。下一步必须改变参数状态随任务增长的结构，而不是继续优化同一个 feature regression。

## 3. 核心观察：历史 B bank 可以精确累计

当前 Shared-A 的历史 LoRA 分支共享同一个下投影 A。把 scale 和归一化全部折叠到有效上投影后，有：

```text
sum_i s_i * Bbar_i * Abar
    = (sum_i s_i * Bbar_i) * Abar
    = H_t * Abar
```

因此，历史任务不需要永久保存 `B_0, B_1, ..., B_t`。每个 Q/V LoRA 层只需要保存：

- 一个共享下投影 `A_t`；
- 一个累计历史上投影 `H_t`；
- 当前任务训练期间的临时 `B_t`。

任务结束后，把 `B_t` 合并进 `H_t` 并删除 `B_t`。这样持续训练所需的 LoRA 状态从 `O(Tdr)` 降为 `O(dr)`。

## 4. Gauge-Aligned 累计合并

### 4.1 Canonicalization

不能直接对代码中的 raw A/B 做变换，因为当前实现包含 Frobenius normalization 和 task scale。必须先构造真实有效算子，再做规范化分解。

对某层共享下投影做 thin QR：

```text
A_t^T = Q_t R_t
A_t   = R_t^T Q_t^T
```

其中 `Q_t` 的列正交。于是：

```text
B_t A_t = (B_t R_t^T) Q_t^T
```

将 scale、A/B normalization 和 `R_t^T` 全部吸收到上投影，使用 `Q_t^T` 作为 canonical down projection。这样不同任务的 LoRA 均在明确的正交坐标系中表示，避免因 LoRA factor gauge freedom 导致数值口径不一致。

### 4.2 历史算子对齐

任务 `t-1` 完成后的历史有效算子记为：

```text
DeltaW_old = H_{t-1} Q_{t-1}^T
```

任务 `t` 训练后得到新的共享基底 `Q_t`。在新基底中寻找最接近历史算子的上投影：

```text
H_old_aligned = H_{t-1} Q_{t-1}^T Q_t
```

这是以下最小二乘问题的闭式最优解：

```text
min_H ||H_{t-1} Q_{t-1}^T - H Q_t^T||_F^2
```

历史投影误差为：

```text
E_t = ||H_{t-1} Q_{t-1}^T (I - Q_t Q_t^T)||_F
```

如果新旧 A 的行空间一致，则 `E_t = 0`，历史 LoRA 算子可以精确保持。一般情况下，误差由两个行空间的 principal angles 决定。

### 4.3 合并当前任务

把当前任务有效更新转换到 `Q_t` 坐标，记为 `H_new_task`，然后执行：

```text
H_t = H_old_aligned + H_new_task
```

保存：

```text
canonical_A = Q_t^T
cumulative_B = H_t
```

删除：

```text
per-task B_t
per-task historical B files
per-task historical scales
```

后续任务只加载 `(canonical_A, cumulative_B)`。历史 scale 已折叠进 `cumulative_B`，不再允许后续任务单独修改历史任务权重，这也是一种无回放稳定性约束。

## 5. 与 prototype transport 的关系

Gauge alignment 只处理参数空间中历史 LoRA 有效算子的变化。新任务 LoRA、ViT 非线性和深层传播仍会产生剩余 feature drift。

因此保留当前 normalized-space affine LRPT，但重新定位为 residual correction：

1. 先完成 cumulative LoRA 的 gauge alignment；
2. 用部署状态下训练前/训练后的当前任务配对特征拟合 rank-10 affine transport；
3. 仅更新旧 prototype；
4. transport 的 U/V/b 使用后立即丢弃；
5. 不保存可训练 transport 网络。

论文中必须消融：

- cumulative merge，无 gauge，无 LRPT；
- cumulative merge + gauge；
- cumulative merge + LRPT；
- cumulative merge + gauge + LRPT。

只有当 LRPT 在多 seed 下稳定改善 AvgAcc 或 Forgetting时，才把它保留为主方法组件；否则将其降为消融或移除，避免主方法过于复杂。

## 6. 可选的轻量分类校准

如果 Gauge + LRPT 后 Forgetting 仍明显高于 SD-LoRA，可为每类保存一个 vMF concentration 或类内聚合度标量：

```text
rho_c = ||mean_i(normalize(z_i))||
```

使用 `rho_c` 校准 prototype cosine logit 或温度。该状态只增加一个标量/类。

这只能作为轻量分类器校准和消融，不能作为论文主要贡献。不得使用旧训练样本重新拟合校准参数。

## 7. 参数量预估

rank=10、12 个 ViT block、每层 Q/V 两个 LoRA 分支时：

```text
shared/canonical A: 24 * 10 * 768 = 184,320
cumulative B:       24 * 768 * 10 = 184,320
LoRA total:                            368,640
```

加入 prototype：

| 数据集 | LoRA | Prototype | 预计总状态 | 相对 SD-LoRA | 减少比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ImageNet-R | 368,640 | 153,600 | 522,240 | 14.17% | 85.83% |
| CIFAR-100 | 368,640 | 76,800 | 445,440 | 12.08% | 87.92% |

最终统计必须包含所有持久 scale、classifier state、normalization state 和 metadata。临时 QR/SVD、LRPT U/V/b、优化器状态和峰值训练显存单独报告，不得混入部署参数，也不能完全忽略。

## 8. 实现计划

### Phase A：纯代数等价性

1. 在 `backbone/sa_lora.py` 增加把历史 B bank 折叠成 cumulative B 的纯函数。
2. 使用实际 scale 和 normalization 计算有效上投影。
3. 固定 A 时验证 bank forward 与 cumulative forward 一致。
4. 要求每层有效算子误差、feature 误差和 logits 误差均小于 `1e-5` 量级。

状态：完成（commit `c666ac0`，2026-08-06）。`tests/test_sa_cumulative.py` 验证固定 A 下算子/feature/logits 等价误差 < 1e-5，且累计 B 与 `save_merged_lora` 产物一致；全量 32 tests passed。

### Phase B：在线累计状态

1. 升级 `SA_STATE_VERSION`，artifact 保存 canonical A、cumulative B、task id 和必要 metadata。
2. Task 0 结束后立即累计，不再保留 task0 独立 B。
3. Task t 只加载累计状态并创建一个临时当前 B。
4. Task t 结束后再次累计并删除临时 B。
5. 增加旧 artifact 到新 artifact 的显式迁移脚本，不做静默兼容。

### Phase C：Gauge alignment

1. 每个任务训练前保存 canonical `(H_{t-1}, Q_{t-1})`。
2. 训练后对新 A 做 QR canonicalization。
3. 闭式计算 `H_old_aligned` 并合并当前任务。
4. 记录 principal angles、relative projection residual 和算子保持误差。
5. 不先加入 principal-angle regularizer；只有当投影残差与遗忘显著相关时才考虑。

代数部分完成（commit `02459fc`）：`canonical_down_projection` / `canonicalize_effective_up_projection` / `gauge_align_up_projection` / `gauge_projection_residual` 已有单测；待 Phase B 状态集成后接入训练路径。

### Phase D：Residual LRPT

1. prototype pre/post 特征必须来自真正部署的累计模型状态。
2. 在 gauge 完成后应用 normalized affine LRPT。
3. 保留现有 rank-10、bias、lambda=1 作为默认值，不继续扫 rank/damping。

### Phase E：可选 classifier calibration

只在前四阶段完成后评估每类单标量 concentration。若没有跨 seed 稳定收益，删除该组件。

## 9. 测试要求

- 固定 A 的 bank-to-cumulative 算子等价测试。
- 包含 normalization 和 scale 的 forward 等价测试。
- QR canonicalization 前后有效 `BA` 不变。
- 新旧 A 行空间相同时，gauge alignment 误差接近零。
- 构造正交行空间时，投影残差与理论值一致。
- Task 0 至 Task 9 全流程中始终只有一个 cumulative B artifact。
- 恢复训练后，task id、prototype、A、B 和 logits 一致。
- 4 卡 DDP 无重复累计、重复保存或 rank 间状态差异。
- merged/cumulative 单次推理与训练结束部署状态一致。

## 10. 实验顺序与停止条件

### 单 seed 筛选

1. ImageNet-R seed 1995：cumulative only。
2. ImageNet-R seed 1995：cumulative + gauge。
3. 通过后运行 CIFAR-100 seed 1993。
4. 再运行 gauge + residual LRPT。

### 最低方法验收线

- ImageNet-R Final Top1 >= 78.76。
- CIFAR-100 Final Top1 >= 86.89。
- 包含 prototype 的持久状态相对 SD-LoRA 减少至少 80%。
- 推理不需要 task id、router 或逐任务 adapter 前向。
- 后续训练不需要恢复历史 B bank。

### 强结果目标

- ImageNet-R Final >= 79.29，AvgAcc >= 83.00。
- CIFAR-100 Final >= 88.4，AvgAcc >= 92.4。
- Forgetting 至少不比当前 affine LRPT 更差。
- gauge projection residual 与旧类遗忘存在可解释关系。

### 停止条件

- cumulative only 相对当前 Shared-A prototype 在两个数据集下降超过 0.5：先检查历史 scale/normalization 口径，不继续加新模块。
- gauge 连续两次运行都不改善 AvgAcc/Forgetting，且 projection residual 与遗忘相关性弱：保留 cumulative merge，删除 gauge 主张。
- LRPT 多 seed 增益小于运行标准差：从最终主方法移除 LRPT。
- 禁止重新进入 rank、damping、raw-space、classmean-only、class-wise JVP 和 operator-lambda 扫描。

## 11. A 会论文所需实验

完成单 seed 方法筛选后再进行：

- 至少 3 个不同 seed，最好 5 个，报告 mean +/- std 和显著性检验。
- CIFAR-100 与 ImageNet-R 的不同任务长度，例如 T=5/10/20/40。
- 至少增加 ImageNet-A、CUB-200 或 DomainNet 中的一个数据集。
- 同协议比较 SD-LoRA、ES-LoRA、InfLoRA、CL-LoRA、LoRA-DRS、DGS 和可复现的 2026 方法。
- 同时报 Final、AvgAcc、Forgetting、持久状态、训练临时状态、FLOPs、吞吐量和峰值显存。
- 绘制参数量随任务数增长曲线，突出 cumulative state 的 O(1) LoRA 存储。
- 提供 old operator drift、principal angle、projection residual、真实 prototype drift 与分类遗忘的相关性分析。

## 12. 文献碰撞边界

- LDC 已覆盖通用 moving-backbone prototype drift compensation。
- ICLR 2026 Two-Way Alignment 已覆盖旧/新特征双向映射、cycle consistency 和 Gaussian prototype transport。
- DGS 已覆盖 LoRA gradient 与 semantic-shift prototype alignment。
- E2-LoRA 已覆盖 output feature drift 的低秩结构和能量排序。
- Janus-LoRA 已覆盖 closed-form LoRA gradient rectification。
- Balanced LoRA 已讨论 LoRA 因子表示不唯一与 canonical/balanced manifold。

因此，不能把“低秩”“闭式”“prototype transport”单独作为新颖性。需要强调以下组合性质：

1. Shared-A 使历史 task LoRA 可以代数精确累计。
2. Gauge alignment 在共享 A 改变后提供闭式最优历史算子投影。
3. 持续训练状态不随任务数增长，而不只是最终部署时合并。
4. 无回放、无 task-id、无 router、无持久 transport network。
5. residual LRPT 只处理参数空间对齐后的剩余非线性漂移。

## 13. 建议论文主张

建议主张：

> We propose an online gauge-aligned cumulative LoRA framework for rehearsal-free class-incremental learning. By exploiting the shared down-projection, all historical task updates are consolidated into a single canonical low-rank operator whose state is independent of the number of tasks. A closed-form gauge alignment preserves historical operators as the shared subspace evolves, while a disposable prototype transport corrects residual nonlinear feature drift.

不建议主张：

- 全面优于 SD-LoRA，除非 AvgAcc 和 Forgetting 也在多 seed 下成立。
- 第一个 prototype drift compensation 方法。
- 第一个 closed-form LoRA continual learning 方法。
- LRPT 是 LoRA-specific，除非完成结构推导和对应实验证明。

## 14. 最终决策原则

如果 cumulative + gauge 在两个数据集保持当前准确率，总持久状态减少约 86%，即使 Top1 不再额外提高，也优先进入多 seed 和论文阶段。这个结果的研究价值高于继续用大量试验换取 0.1% 左右的单 seed 精度变化。

如果累计状态造成明显性能下降，则保留当前 affine LRPT 作为效率 Pareto 基线，但不继续堆叠缺乏统一机制的小模块。
