# Adaptive-A CoordinateStable SD-LoRA：当前完整方法设计与实验状态

> 更新时间：2026-09-07  
> 对应分支：`codex/adaptive-a-coordinate`  
> 对应代码基线：`b31be24`  
> 当前主方法：无 AOB 的 CoordinateStable + bounded NormCap + Adaptive-A + prototype transport  
> 说明：本文不把分类头列为方法结构或创新点。现有正式配置仍沿用既定分类评估设置，因此涉及该配置的结果必须在对比实验中保持分类器口径一致。

## 1. 研究问题

原始 SD-LoRA 在每个增量任务上训练并保存一套任务 LoRA。设第 `t` 个任务的 LoRA 更新为

\[
\Delta W_t=s_tB_tA_t,
\]

其中 `A_t` 为 down projection，`B_t` 为 up projection，`s_t` 为可学习缩放。经历 `T` 个任务后，原始方法需要保存并在前向中使用 `T` 套 LoRA，LoRA 状态、前向分支数和后期训练开销均随任务数线性增长。

当前方法研究的问题是：

> 在无回放类增量学习中，能否只保留固定大小的共享 LoRA 状态，同时允许共享子空间继续适应新任务，并控制由共享坐标变化引起的历史知识漂移？

方法的核心不是路由或任务专家选择，而是把所有历史任务压缩为一套固定状态 `(A, G)`，并同时处理三个耦合问题：

1. 新任务知识如何写入固定大小的 LoRA 状态；
2. 共享 `A` 更新后，历史有效算子如何在新坐标下保持稳定；
3. 共享 `A` 应在何时改变行空间、何时保留历史坐标。

## 2. 学习设置与 LoRA 注入位置

- 场景：rehearsal-free class-incremental learning；不保存历史图像。
- Backbone：ImageNet 预训练 ViT-B/16，主干权重冻结。
- LoRA 位置：12 个 Transformer block 的 attention Q、V 投影，共 24 个分支。
- LoRA rank：`r=10`。
- 每个分支输入、输出维度：`d=768`。
- 测试：在所有已见类别上进行全局分类，不提供 task ID。

对任意 Q/V 分支：

\[
A_t\in\mathbb{R}^{r\times d},\qquad
B_t,G_{t-1}\in\mathbb{R}^{d\times r}.
\]

`sa_shared_a_orthogonal=true` 只控制共享 `A` 的正交初始化。当前实现没有在训练过程中持续强制 `A_tA_t^\top=I`，因此不能把该配置描述为正交约束训练。

## 3. Live-A Aggregate-B 固定状态参数化

### 3.1 任务内状态

训练任务 `t` 时，每个分支只包含：

- `A_t`：当前共享 down projection，可训练；
- `G_{t-1}`：此前任务折叠后的历史 aggregate up projection，任务内冻结；
- `B_t`：当前任务临时 up projection，可训练且零初始化；
- `s_t`：当前任务可学习缩放。

定义

\[
\widehat A_t=\frac{A_t}{\lVert A_t\rVert_F+\epsilon}.
\]

任务内 LoRA 前向为

\[
\Delta h_t(x)
=G_{t-1}\widehat A_tx+s_tB_tA_tx.
\]

第一项调用固定大小的历史状态，第二项学习当前任务更新。两项都依赖当前共享 `A_t`，所以 `A_t` 的普通反向梯度同时包含历史分支和当前分支的信息。

### 3.2 固定持久状态

任务结束后，当前 `B_t` 被吸收到 `G_t`，随后不再作为逐任务专家保存。部署和恢复训练只需要：

\[
\mathcal S_t=\{A_t,G_t\}.
\]

因此 LoRA 主状态不随任务数增长，推理时也不需要 task ID、router 或逐任务 LoRA bank。

## 4. Effective-Operator Coordinate Alignment

### 4.1 需要对齐的原因

历史有效算子为

\[
M_{t-1}=G_{t-1}\widehat A_{t-1}.
\]

当新任务把共享矩阵从 `A_{t-1}` 更新为 `A_t` 时，直接使用 `G_{t-1}\widehat A_t` 会改变历史算子。该变化既可能是无法避免的行空间变化，也可能只是 LoRA 因子坐标变化导致的非语义漂移。

### 4.2 实际优化目标

当前实现先归一化旧、新 `A`：

\[
\widehat A_{t-1}=\frac{A_{t-1}}{\lVert A_{t-1}\rVert_F+\epsilon},\qquad
\widehat A_t=\frac{A_t}{\lVert A_t\rVert_F+\epsilon}.
\]

然后求解

\[
G_{t-1}^{\mathrm{align}}
=\arg\min_X
\left\lVert
X\widehat A_t-G_{t-1}\widehat A_{t-1}
\right\rVert_F^2.
\]

执行代码使用

```python
torch.linalg.lstsq(A_new_hat.T, M_old.T).solution.T
```

没有在该最小二乘中加入 ridge 正则。`sa_coordinate_transport_reg=1e-4` 属于后面的 prototype transport，不用于 operator alignment。

若新旧 `A` 行空间完全相同，历史有效算子可以严格保持；若行空间已经变化，对齐只能得到在新行空间中的最小二乘投影。日志中的 alignment `before/after` 就是对齐前后相对历史算子误差。

## 5. Bounded Norm-Calibrated Absorption

当前主配置使用：

```json
"sa_live_a_absorb_mode": "bounded_norm_calibrated_absorb"
```

对任务 `t` 的当前 LoRA，定义

\[
n_A=\lVert A_t\rVert_F+\epsilon,\qquad
n_B=\lVert B_t\rVert_F+\epsilon,
\]

\[
\widehat A_t=A_t/n_A,\qquad
\widehat B_t=B_t/n_B,
\]

以及基于因子范数乘积的 consolidation gain：

\[
c_t=\min\left(1,\frac{1}{n_An_B}\right).
\]

当前任务写入系数为

\[
\gamma_t=s_tn_An_Bc_t
=s_t\min(n_An_B,1),
\]

最终聚合为

\[
G_t=G_{t-1}^{\mathrm{align}}+\gamma_t\widehat B_t.
\]

吸收后当前任务对应的有效算子为

\[
\Delta W_t^{\mathrm{commit}}
=\gamma_t\widehat B_t\widehat A_t
=c_ts_tB_tA_t.
\]

因此：

- 当 `n_A n_B <= 1` 时，`c_t=1`，任务边界前后当前算子保持等价；
- 当 `n_A n_B > 1` 时，算子按 `1/(n_A n_B)` 衰减，主动限制写入历史状态的强度；
- 日志中的 `absorption_relative_error` 在非零算子情况下近似为 `|1-c_t|`，它是该机制的实际压缩强度，不应被解释为纯数值误差。

这个机制是 factor-norm-based NormCap，不是 exact operator budget。当前主方法不包含 AOB、operator-norm EMA 或方向冲突分解。

## 6. Adaptive-A：漂移/冲突感知的共享坐标可塑性

### 6.1 动机

始终训练 `A` 可以提高新任务适应能力，但会改变历史算子的输入行空间；Task 0 后完全冻结 `A` 可以增强稳定性，却可能限制 ImageNet-R 等数据集所需的后续表示调整。

Adaptive-A 不在“始终训练”和“永久冻结”之间做全局二选一，而是在每一层、每个优化步上，只调节会改变 `A` 行空间的梯度分量。

### 6.2 梯度分解

设某个分支的共享 `A` 梯度为

\[
D=\frac{\partial\mathcal L}{\partial A}\in\mathbb{R}^{r\times d}.
\]

对 `A^T` 做 thin QR 分解，得到具有正交行的基

\[
Q\in\mathbb{R}^{r\times d},\qquad QQ^\top=I.
\]

将梯度分解为

\[
D_{\parallel}=(DQ^\top)Q,
\qquad
D_{\perp}=D-D_{\parallel}.
\]

其中：

- `D_parallel` 主要改变现有行空间内部的坐标；
- `D_perp` 改变 `A` 的行空间方向，是历史有效算子漂移的主要一阶来源。

### 6.3 当前任务收益与历史风险

对 Transformer 第 `l` 层的 Q/V 两个分支，定义当前任务影响：

\[
P_l=
\sqrt{
\lVert s_tB_{t,q}D_{\perp,q}\rVert_F^2+
\lVert s_tB_{t,v}D_{\perp,v}\rVert_F^2
}.
\]

定义历史算子影响：

\[
H_l=
\sqrt{
\left\lVert\frac{G_{t-1,q}}{\lVert A_{t,q}\rVert_F+\epsilon}
D_{\perp,q}\right\rVert_F^2+
\left\lVert\frac{G_{t-1,v}}{\lVert A_{t,v}\rVert_F+\epsilon}
D_{\perp,v}\right\rVert_F^2
}.
\]

代码通过低秩 Gram 矩阵计算这些 Frobenius norm，不显式构造 `768 x 768` 稠密算子。

### 6.4 自适应门控

瞬时门值为

\[
g_l^{\mathrm{raw}}=
\operatorname{clip}\left(
\frac{P_l}{P_l+\lambda H_l+\epsilon},
g_{\min},1
\right).
\]

当前配置为：

\[
\lambda=1.0,\qquad g_{\min}=0.05.
\]

再用任务内 EMA 平滑：

\[
g_l\leftarrow
\mu g_l^{\mathrm{prev}}+(1-\mu)g_l^{\mathrm{raw}},
\qquad \mu=0.9.
\]

最终替换 Q/V 的共享 `A` 梯度：

\[
D'_q=D_{\parallel,q}+g_lD_{\perp,q},
\qquad
D'_v=D_{\parallel,v}+g_lD_{\perp,v}.
\]

同一 Transformer 层的 Q/V 共用一个 gate，从而以层为粒度控制坐标变化。门控发生在 `loss.backward()` 之后、`optimizer.step()` 之前；DDP 场景下使用已经同步的梯度，不需要额外通信。

### 6.5 边界行为与持久状态

- Task 0 没有历史 `G`，梯度保持不变，等价于原始 trainable-A 基线；
- 若 Q/V 的 `D_perp` 都为零，则不抑制梯度；
- gate、EMA、`P_l/H_l` 仅用于当前任务训练和日志诊断；
- Adaptive-A 不增加可训练参数，也不增加任何随任务增长的持久状态；
- 保存状态仍然只有每个分支的 `(A_t,G_t)`。

## 7. Gated Low-Rank Orthogonal Prototype Transport

Coordinate alignment 处理 LoRA 有效算子的坐标稳定，但任务训练和受控吸收后，旧类别 prototype 仍可能与新部署特征空间失配。当前实现使用当前任务数据的训练前/后配对特征估计这种漂移。

### 7.1 配对特征

对当前任务相同样本，在任务训练前的旧部署状态和任务吸收、重建后的新部署状态分别提取：

\[
Z_{\mathrm{old}},Z_{\mathrm{new}}\in\mathbb{R}^{N_t\times768}.
\]

两个特征集合先逐样本 L2 归一化。该过程只访问当前任务数据，不需要历史样本。

### 7.2 低秩漂移子空间与正交变换

对训练子集的漂移矩阵

\[
D=Z_{\mathrm{new}}-Z_{\mathrm{old}}
\]

执行 SVD，取前 `k=10` 个右奇异向量组成漂移基

\[
U\in\mathbb{R}^{768\times k}.
\]

在该子空间中拟合带 identity bias 的正交 Procrustes 旋转。设

\[
C=\frac{(Z_{\mathrm{old}}U)^\top(Z_{\mathrm{new}}U)}{N}+\eta I,
\qquad \eta=10^{-4},
\]

若 `C=U_C\Sigma V_C^T`，则

\[
R=U_CV_C^T.
\]

低秩 residual transport 为

\[
T(z)=z+(zU)(R-I)U^T.
\]

### 7.3 Held-out gate

样本按固定规则划分为 80% 拟合、20% 验证。仅当 transport 降低 held-out 配对特征误差时，才把 `T` 应用于全部历史类别 prototype；否则使用恒等变换。

`U` 和 `R` 在应用后立即丢弃，不作为任务级状态保存。持久化的是被更新后的类别 prototype，因此 transport 网络参数开销为零，但 prototype 数量仍随类别数增长。

该模块应定位为受约束的 prototype drift compensation。其问题设定与 SDC/LDC 接近，论文新颖性不能建立在“首次提出 prototype transport”上；可讨论的差异是低秩、正交、held-out gate 和零持久 transport 参数。

## 8. 完整任务流程

对任务 `t`，实际顺序为：

1. 加载冻结 ViT、共享 `A_{t-1}`、历史聚合 `G_{t-1}` 和历史 prototype；
2. 若 `t>0`，用旧部署状态提取当前任务的 pre-update 特征；
3. 创建零初始化 `B_t` 和当前缩放 `s_t`；
4. 使用当前任务数据训练 `A_t`、`B_t`、`s_t` 与既定分类器参数；
5. 每次 backward 后通过 Adaptive-A 改写共享 `A` 的 Q/V 梯度，再执行 optimizer step；
6. 任务结束时先把 `G_{t-1}` 对齐到新的 `A_t` 坐标；
7. 通过 bounded NormCap 把当前 `s_tB_tA_t` 受控吸收到 `G_t`；
8. 保存 `(A_t,G_t)` 并重建部署 backbone；
9. 用新部署状态提取同一批当前任务特征，拟合并按 gate 应用 prototype transport；
10. 计算当前新类别 prototype，更新全类别 prototype 集并执行全局 CIL 评估；
11. 丢弃当前任务临时 `B_t`，进入下一任务。

最终推理只有一套 rank-10 LoRA：

\[
\Delta W_{\mathrm{deploy}}=G_T\widehat A_T.
\]

不存在逐任务 LoRA 求和、task-ID oracle 或 prototype router。

## 9. 状态复杂度与效率

### 9.1 LoRA 参数

每个 Q/V 分支保存一个 `A` 和一个 `G`：

\[
2rd=2\times10\times768=15,360.
\]

24 个分支共：

\[
24\times15,360=368,640
\]

个 LoRA 参数，与任务数 `T` 无关。原始 SD-LoRA 在 `T=10` 时保存：

\[
10\times368,640=3,686,400
\]

个 LoRA 参数，因此当前方法的 LoRA-only 状态减少 90%。

需要严格区分：

- LoRA 状态是 `O(1)`；
- 类别 prototype 和分类器仍随已见类别数增长，为 `O(C)`；
- 不能声称整个模型的所有增量状态都是 `O(1)`。

### 9.2 已完成的效率审计

| 指标 | 当前固定状态方法 | SD-LoRA |
| --- | ---: | ---: |
| T=5 LoRA 参数 | 368,640 | 1,843,200 |
| T=10 LoRA 参数 | 368,640 | 3,686,400 |
| T=20 LoRA 参数 | 368,640 | 7,372,800 |
| T=10 训练峰值显存 | 3,468.6 MiB | 8,186.7 MiB |
| T=20 训练峰值显存 | 3,468.6 MiB | 12,632.1 MiB |

Adaptive-A 只修改已有梯度，不改变上述持久参数量。

## 10. 当前主配置

当前方法的关键配置为：

```json
{
  "model_name": "sa_sdlora",
  "lora_rank": 10,
  "sa_shared_a_orthogonal": true,
  "sa_train_a_all_tasks": true,
  "sa_cumulative_state": true,
  "sa_cumulative_merge": "live_a_aggregate_b",
  "sa_live_a_history_groups": 1,
  "sa_live_a_coordinate_align": true,
  "sa_live_a_absorb_mode": "bounded_norm_calibrated_absorb",
  "sa_coordinate_stable_transport": true,
  "sa_coordinate_transport_rank": 10,
  "sa_coordinate_transport_reg": 0.0001,
  "sa_coordinate_transport_min_gain": 0.0,
  "sa_adaptive_a_enabled": true,
  "sa_adaptive_a_stability_weight": 1.0,
  "sa_adaptive_a_gate_floor": 0.05,
  "sa_adaptive_a_gate_momentum": 0.9,
  "sa_adaptive_a_eps": 1e-8
}
```

当前主方法明确不包含：AOB、exact operator budget、canonicalized persistent `A`、方向冲突分解、任务路由、K-CMS merge、回放样本或生成回放。

## 11. 实验协议

| 数据集 | 类别/任务 | 任务数 | Epoch/task | 优化器与调度 | 学习率 | 等效 batch |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| CIFAR-100 | 10 | 10 | 20 | SGD + cosine | 0.008 | 128 |
| ImageNet-R | 20 | 10 | 20 | SGD + constant | 0.01 | 128 |
| CUB-200 | 20 | 10 | 20 | SGD + constant | 0.01 | 128 |

当前 Pro6000 复现实验使用单 GPU、`batch_size=128`；此前四卡实验使用每卡 `batch_size=32`。二者名义等效 batch 相同，但 world size、样本归约顺序和浮点执行顺序不同，因此跨硬件结果仍应视为复现验证，不能默认逐位等价。

主指标：

- Final Top-1：最后任务结束后，对全部类别的 Top-1；
- AAA：每个任务边界全局 Top-1 的平均；
- Forgetting：历史任务峰值准确率到最终准确率的平均下降，越低越好。

分类器及其评估调度不是本文的方法创新点。所有方法对比必须使用完全相同的分类器设置，不能把分类器变化与 LoRA 状态管理收益混在一起。

## 12. 实验结果

### 12.1 Adaptive-A 当前已完成结果

截至 2026-09-07，Pro6000 单卡 `batch_size=128` 的 CIFAR-100 三 seed 已完成：

| Seed | Final Top-1 | AAA | Forgetting |
| ---: | ---: | ---: | ---: |
| 1993 | 88.38 | 92.268 | 6.167 |
| 1994 | 86.85 | 91.603 | 8.967 |
| 1995 | 88.04 | 91.883 | 6.478 |
| Mean +/- population std | **87.76 +/- 0.66** | **91.92 +/- 0.27** | **7.20 +/- 1.25** |

当前结论：Adaptive-A 在 CIFAR-100 上可以得到较高 Final 和 AAA，但 seed1994 的遗忘明显更高，说明自适应门控尚未消除任务顺序敏感性。应等待 ImageNet-R 与 CUB-200 多 seed 全部完成后再形成跨数据集结论。

正在运行的队列：

- ImageNet-R：seed `1995/1996/1997`；
- CUB-200：seed `1/2/3`。

未完成实验不进入最终均值表。

### 12.2 Adaptive-A 之前的 CoordinateStable 参考结果

下面结果用于说明当前方法所继承的基线能力，不应冒充 Adaptive-A 的消融结果：

| Dataset | Final Top-1 | AAA | Forgetting |
| --- | ---: | ---: | ---: |
| CIFAR-100 | 87.40 +/- 0.62 | 91.77 +/- 0.53 | 8.09 +/- 1.18 |
| ImageNet-R | 79.34 +/- 0.36 | 83.71 +/- 0.45 | 7.55 +/- 1.19 |
| CUB-200 | 80.39 +/- 1.66 | 86.31 +/- 0.89 | 11.50 +/- 0.10 |

这些结果表明固定状态 CoordinateStable 主干已经具备接近或超过 SD-LoRA 的潜力；Adaptive-A 的新增问题是，能否在三个数据集上比统一 trainable-A/frozen-A 策略取得更稳定的稳定性-可塑性折中。

## 13. 方法贡献与论文定位

### 13.1 可作为主线的贡献

1. **固定状态的持续 LoRA 聚合**：逐任务临时 `B_t` 在线吸收到单一 `G_t`，LoRA 持久状态和推理分支数不随任务增长。
2. **有效算子坐标对齐**：在共享 `A` 更新后，以闭式最小二乘把历史 `G` 转换到新坐标，显式控制因 LoRA 分解坐标变化造成的历史算子漂移。
3. **Adaptive-A 坐标可塑性**：把共享 `A` 梯度分为行空间内和行空间外分量，根据当前任务收益与历史算子风险逐层门控真实的行空间变化，不增加参数或持久状态。
4. **受控任务边界写入**：bounded NormCap 把偶发过大的任务更新转化为可解释的 consolidation gain，避免无界写入固定聚合状态。

### 13.2 辅助模块与主张边界

Prototype transport 是固定状态 LoRA 的辅助表征稳定模块。由于 SDC/LDC 已覆盖“用当前任务训练前后特征补偿旧 prototype 漂移”的核心问题，论文中应把当前实现定位为几何保持、低秩、held-out-gated 的受约束变体，而不是全新的 prototype transport 范式。

当前证据也不支持以下表述：

- 所有数据集和所有指标都稳定优于 SD-LoRA；
- 整个增量状态为 `O(1)`；
- NormCap 严格保持任务边界前后的当前算子；
- Adaptive-A 已经在三个数据集完成多 seed 验证；
- prototype transport 的基本思想此前不存在。

## 14. 论文所需的关键消融

为证明收益来自 LoRA 状态管理本身，至少需要保持相同分类器和训练协议，比较：

1. trainable-A CoordinateStable；
2. frozen-A CoordinateStable；
3. Adaptive-A CoordinateStable；
4. Adaptive-A 去掉 coordinate alignment；
5. Adaptive-A 去掉 bounded NormCap；
6. Adaptive-A 去掉 prototype transport；
7. `normalized_absorb`、`operator_preserving_absorb` 与 `bounded_norm_calibrated_absorb`；
8. gate floor、stability weight 和 EMA momentum 的敏感性；
9. SDC/LDC 式 prototype drift compensation 与当前低秩正交 transport 的同协议比较。

除 Final、AAA、Forgetting 外，还应报告：

- 每层 gate 均值与随任务变化；
- `P_l/H_l` 分布；
- alignment before/after residual；
- consolidation gain 与 absorption relative error；
- prototype transport gate 触发率和 held-out gain；
- LoRA-only、分类器和 prototype 的分项参数量；
- T=5/10/20/40 的状态、显存和耗时曲线。

## 15. 一句话总结

Adaptive-A CoordinateStable SD-LoRA 将逐任务 LoRA bank 压缩为固定大小的 `(A,G)`，通过历史有效算子坐标对齐、受控任务边界吸收和漂移/冲突感知的共享 `A` 行空间梯度门控，在不增加持久参数的前提下协调新任务可塑性与历史知识稳定性；prototype transport 作为无历史样本的辅助校准机制，进一步缓解旧类别中心与新部署特征空间的失配。
