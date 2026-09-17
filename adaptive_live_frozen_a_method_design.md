# 当前 Shared-A SD-LoRA 三种 A 策略

## 0. 文档范围

本文以当前仓库 `codex/functional-halfspace-adaptive-a` 分支的可执行代码，以及
`scripts/generate_tasklen_fla_configs.py` 生成的任务长度队列为准。这里的三种方法是：

- `Frozen-A`：Task0 训练共享下投影，后续任务冻结 `A`；
- `Live-A`：所有任务继续更新共享 `A`，不做自适应门控；
- `Adaptive-A`：所有任务更新共享 `A`，但从 Task1 开始按历史/当前算子影响比率门控
  改变 `A` 行空间的梯度，并在任务边界进行坐标对齐。

本文不是旧版 Dual-B、CoordinateStable prototype transport、CUO 或
functional-halfspace 版本的总结。当前任务长度三路队列实际启用的 Adaptive-A 策略是
`impact_ratio`，不是 `functional_halfspace` 或 Pareto 策略。

审计基准：2026-09-17。主要代码入口：

- 配置生成：[scripts/generate_tasklen_fla_configs.py](scripts/generate_tasklen_fla_configs.py)
- 模型包装：[models/sa_sdlora.py](models/sa_sdlora.py)
- Shared-A LoRA 主体：[backbone/sa_lora.py](backbone/sa_lora.py)
- 坐标对齐：[backbone/coordinate_stability.py](backbone/coordinate_stability.py)
- 原型分类头：[utils/inc_net.py](utils/inc_net.py)

## 1. 当前三路实验的实际配置

任务长度脚本在每个数据集上生成 `T=5/10/20/50`，方法为
`frozen_a/live_a/adaptive_a`。当前脚本覆盖 `CIFAR-100`、`ImageNet-R`、`CUB-200`，
以及已加入脚本的 `DomainNet`。

| 配置项 | 三种方法共同设置 | Frozen-A | Live-A | Adaptive-A |
|---|---|---:|---:|---:|
| `model_name` | `sa_sdlora` | 相同 | 相同 | 相同 |
| backbone | `vit_base_patch16_224` | 相同 | 相同 | 相同 |
| LoRA rank | `10` | 相同 | 相同 | 相同 |
| LoRA 注入位置 | 12 个 ViT block 的 Q/V | 相同 | 相同 | 相同 |
| `sa_shared_a_orthogonal` | `true` | 相同 | 相同 | 相同 |
| `sa_cumulative_state` | `true` | 相同 | 相同 | 相同 |
| `sa_cumulative_merge` | `live_a_aggregate_b` | 相同 | 相同 | 相同 |
| `sa_live_a_history_groups` | `1` | 相同 | 相同 | 相同 |
| `sa_use_prototype_classifier` | `true` | 相同 | 相同 | 相同 |
| `sa_dual_head` | `false` | 相同 | 相同 | 相同 |
| `sa_train_a_all_tasks` | - | `false` | `true` | `true` |
| `sa_live_a_coordinate_align` | - | `false` | `false` | `true` |
| `sa_adaptive_a_enabled` | - | `false` | `false` | `true` |
| `sa_adaptive_a_strategy` | - | - | - | `impact_ratio` |
| gate floor | - | - | - | `0.05` |
| gate momentum | - | - | - | `0.9` |
| stability weight | - | - | - | `1.0` |
| batch size | `64` per process | 相同 | 相同 | 相同 |

生成脚本从带有旧实验命名的源配置复制基础优化器、学习率、epoch 和数据集参数，随后
显式覆盖上表中的三路差异，并删除 `sa_dual_head_schedule`。因此不能仅凭源配置文件名
中的 `dual_b` 判断当前队列开启了 Dual-B。

当前三路队列没有显式设置 `sa_live_a_absorb_mode`，模型构造器默认使用
`operator_preserving_absorb`，见
[models/sa_sdlora.py:870-895](models/sa_sdlora.py#L870-L895) 和
[backbone/sa_lora.py:1560-1571](backbone/sa_lora.py#L1560-L1571)。

## 2. 共同的模型结构

### 2.1 注入位置和参数形状

对于每个 ViT block 的 Q 或 V 分支，设输入 token 的维度为 `d=768`，LoRA rank 为
`r=10`。代码为每个 Q/V 分支分别创建：

\[
A\in\mathbb{R}^{r\times d},\qquad
B\in\mathbb{R}^{d\times r}.
\]

这里的“共享 A”是**跨任务共享**，但每个 transformer block 的 Q 分支和 V 分支有各自
的 `A` 和 `B`，并不是全网络只有一份矩阵。每个分支还使用一个当前任务的标量
`s`；当前实现中的 `wrapped_param` 是一个标量参数，而不是每个任务永久保存一个标量。

基座 ViT 参数在构造后全部设置为不可训练：

```text
for param in vit_model.parameters():
    param.requires_grad = False
```

LoRA 只加到 QKV 输出的前 `d` 维 Q 和后 `d` 维 V，K 分支不加 LoRA。相关前向代码见
[_LiveAAggregateQKV.forward](backbone/sa_lora.py#L1489-L1503)。

### 2.2 Task 内前向

在当前 `live_a_aggregate_b` 结构中，某个 Q/V 分支的历史聚合矩阵记作 `G`。训练当前
任务时，代码实际计算：

\[
\Delta y_{\mathrm{hist}}
=G\frac{Ax}{\lVert A\rVert_F+\varepsilon},
\]

\[
\Delta y_{\mathrm{cur}}
=sB(Ax).
\]

所以该分支的 LoRA 输出为：

\[
\Delta y
=G\frac{Ax}{\lVert A\rVert_F+\varepsilon}+sBAx.
\]

Task0 没有历史聚合矩阵时，`G=0`，只有当前分支。历史分支使用归一化的共享 `A`，
当前分支保持原始 `sBA` 语义。这一差异是任务边界吸收逻辑存在的原因。

### 2.3 任务边界吸收

当前默认的 `operator_preserving_absorb` 在
[absorb_live_a_current_projection](backbone/sa_lora.py#L198-L254) 中执行：

\[
n_A=\lVert A\rVert_F+\varepsilon,\qquad
n_B=\lVert B\rVert_F+\varepsilon,
\]

\[
\hat A=A/n_A,\qquad \hat B=B/n_B,
\]

\[
\gamma=s n_A n_B,
\qquad
\Delta G=\gamma\hat B.
\]

任务边界写入：

\[
G_{\mathrm{new}}=G_{\mathrm{history}}+\Delta G.
\]

由于代码使用同一组 `n_A,n_B`，有：

\[
\Delta G\hat A
=(s n_A n_B)\frac{B}{n_B}\frac{A}{n_A}
=sBA.
\]

因此，在不考虑历史坐标对齐造成的历史算子投影误差时，当前任务的 raw 分支在吸收
前后严格保持同一个有效算子。旧的 `normalized_absorb` 仍被代码支持，但当前三路队列
没有使用它。

任务结束后，`sa_state.pt` 保存 `aggregate_up`、最新 `shared_a` 和合并模式；部署文件
中的 `merged_b` 由 `G/(||A||+eps)` 导出。当前 cumulative state 不再为每个任务保存一套
永久 B；当前任务的 B 和 scale 在边界被折入 `G`。

### 2.4 任务边界的共同生命周期

1. 从当前 cumulative state 加载共享 `A` 和历史聚合 `G`。
2. 创建当前任务的 `B`，初始化为零；创建当前任务标量 `s`，初始化值为 `0.8`。
3. 冻结基座 ViT，按方法策略决定共享 `A` 是否接收梯度。
4. 用当前任务数据训练分类头、当前 `B`、当前 `s`，并按策略处理 `A` 梯度。
5. 任务训练结束后，将当前 `B`、`s` 按上式吸收进 `G`。
6. 将 `shared_a`、`aggregate_up` 等 fixed-state 内容写入 cumulative artifact。
7. 必要时重建部署 backbone；再使用当前任务训练集计算当前类别 prototype，并和历史
   prototype 合并。
8. 评估时使用所有已见类别的 prototype cosine head，不使用任务标签 mask。

`Adaptive-A` 会在第 5 步之前先对历史 `G` 做坐标对齐；`Frozen-A` 和 `Live-A` 不做这一步。

## 3. Frozen-A

### 3.1 训练规则

Frozen-A 的配置为：

```text
sa_train_a_all_tasks = false
sa_live_a_coordinate_align = false
sa_adaptive_a_enabled = false
```

Task0 中 `A` 从固定随机正交下投影开始，并和 Task0 的 `B`、`s` 一起训练。进入
Task1 以后，构造器从 cumulative state 复制已保存的 `A`，然后执行：

```text
a_q.weight.requires_grad_(False)
a_v.weight.requires_grad_(False)
```

因此 Task1 及之后的优化变量主要是当前任务的 Q/V `B`、当前 `s` 和训练分类头；共享
`A` 不再改变。这里的 `sa_shared_a_orthogonal=true` 主要负责 Task0 初始化，不等于
每一步都执行硬正交投影。

### 3.2 前向和合并

由于 `A_t=A_0` 对所有任务成立：

\[
\Delta y_{\mathrm{hist}}=G\frac{A_0x}{\lVert A_0\rVert_F+\varepsilon},
\qquad
\Delta y_{\mathrm{cur}}=s_tB_tA_0x.
\]

任务边界不执行 coordinate alignment，直接令：

\[
G_{t}=G_{t-1}+s_t\lVert A_0\rVert_F B_t
\]

（代码通过 `gamma*B_hat` 实现同一结果）。由于共享坐标没有漂移，历史 `G` 不需要
从旧坐标系转换到新坐标系。

### 3.3 方法含义

Frozen-A 把固定 rank 的共享行空间视为整个持续学习过程的公共特征坐标：

- 优点：历史算子的坐标稳定，边界处理最简单，历史分支不因后续 `A` 更新而改变；
- 风险：长任务序列或任务分布变化较大时，Task0 学到的行空间可能无法覆盖后续任务
  的有效方向；
- 与 Live-A 的差别：不是每个任务独立拥有一套 `A`，而是整个序列只保留 Task0 之后
  的同一份共享 `A`。

## 4. Live-A

### 4.1 训练规则

Live-A 的配置为：

```text
sa_train_a_all_tasks = true
sa_live_a_coordinate_align = false
sa_adaptive_a_enabled = false
```

Task0 训练 `A`。从 Task1 开始，`A` 继续接收未处理的分类损失梯度；当前任务的 `B`、
`s` 也继续训练。代码没有对 `A` 的梯度做门控，也没有把旧类 teacher loss 加入当前
任务损失。

### 4.2 历史分支的动态坐标

Live-A 的历史分支始终使用**当前时刻的 live `A`**：

\[
\Delta y_{\mathrm{hist},t}
=G_{t-1}\frac{A_t x}{\lVert A_t\rVert_F+\varepsilon}.
\]

这意味着即使 `G_{t-1}` 不变，`A_t` 的更新也会改变历史任务的有效算子。与此同时，
当前任务使用：

\[
\Delta y_{\mathrm{cur},t}=s_tB_tA_tx.
\]

因此 Live-A 用一个能适应后续任务的共享行空间换取历史坐标漂移风险。

### 4.3 任务边界

Live-A 不执行 `align_live_a_aggregate`，直接以当前 `A_t` 为坐标吸收当前 `B_t`：

\[
G_t=G_{t-1}+\Delta G_t,
\qquad
\Delta G_t\frac{A_t}{\lVert A_t\rVert_F+\varepsilon}=s_tB_tA_t.
\]

下一任务会从保存的 `A_t,G_t` 重建网络，并将上一任务的 raw `B_t` 清除为新的当前
任务分支；因此部署状态仍然是固定大小的 `A+G`，而不是随任务数线性增长的 LoRA bank。

### 4.4 方法含义

Live-A 是“完全适应”的上界式基线：

- 优点：每个新任务都可以改变共享行空间，适合任务间差异较大或序列较长的场景；
- 风险：历史分支与当前 `A` 共享同一坐标，后续 `A` 的行空间旋转会造成 forgetting；
- 与 Adaptive-A 的差别：Live-A 不判断这次 `A` 更新对历史/当前算子的影响，且不在边界
  做 LS 坐标补偿。

## 5. Adaptive-A（当前主版本）

### 5.1 当前启用的具体版本

当前任务长度脚本为 Adaptive-A 写入：

```text
sa_train_a_all_tasks = true
sa_live_a_coordinate_align = true
sa_adaptive_a_enabled = true
sa_adaptive_a_strategy = impact_ratio
sa_adaptive_a_stability_weight = 1.0
sa_adaptive_a_gate_floor = 0.05
sa_adaptive_a_gate_momentum = 0.9
sa_adaptive_a_eps = 1e-8
```

这不是一个可训练 router，也不是按任务选择不同 LoRA 的 sparse routing。它是在共享
`A` 的梯度层面决定“本次更新保留多少改变行空间的成分”。

### 5.2 梯度的行空间分解

代码先对每个分支的 `A` 做 thin QR：

\[
A^\top=QR,
\]

并令 `Q^T` 为 canonical down projection。共享 `A` 行空间的投影矩阵为：

\[
P_A=Q Q^\top.
\]

对原始共享-A 梯度 `D`，代码执行：

\[
D_{\parallel}=D P_A,
\qquad
D_{\perp}=D(I-P_A).
\]

其中：

- `D_parallel` 只在当前行空间内部改变坐标，通常不会直接改变 row space；
- `D_perp` 才是改变、旋转或扩展共享 `A` 行空间的通道。

对应实现是
[decompose_adaptive_a_gradient](backbone/sa_lora.py#L371-L390)，实际使用
`(gradient @ q_t.T) @ q_t` 完成投影。

### 5.3 Impact-ratio 门控

对一个 block 的 Q/V 两个分支，代码分别计算：

\[
I_{\mathrm{cur}}
=\sqrt{
\lVert sB_qD_{q,\perp}\rVert_F^2
+\lVert sB_vD_{v,\perp}\rVert_F^2
},
\]

\[
I_{\mathrm{hist}}
=\sqrt{
\left\lVert\frac{G_q}{\lVert A_q\rVert_F+\varepsilon}
 D_{q,\perp}\right\rVert_F^2
+\left\lVert\frac{G_v}{\lVert A_v\rVert_F+\varepsilon}
 D_{v,\perp}\right\rVert_F^2
}.
\]

然后得到未平滑门控：

\[
g_{\mathrm{raw}}
=\operatorname{clip}\left(
\frac{I_{\mathrm{cur}}}
{I_{\mathrm{cur}}+\lambda I_{\mathrm{hist}}+\varepsilon},
g_{\min},1
\right),
\]

其中当前队列 `lambda=1.0`、`g_min=0.05`。跨 optimizer step 的门控使用 EMA：

\[
g_t=m g_{t-1}+(1-m)g_{\mathrm{raw},t},
\qquad m=0.9.
\]

最终写回共享 `A` 的梯度：

\[
D_{A}=D_{\parallel}+g_tD_{\perp}.
\]

Q/V 两个分支共享一个 block-level gate；gate 本身不是训练参数，也不写入 cumulative
artifact。实现见
[adaptive_a_layer_gradient](backbone/sa_lora.py#L733-L871) 和
[SharedALoRA_ViT_timm.apply_adaptive_a_gradients](backbone/sa_lora.py#L2433-L2554)。

### 5.4 梯度处理的执行时机

训练循环的顺序是：

```text
forward -> cross entropy -> loss.backward()
         -> _after_backward()
         -> optimizer.step()
```

DDP 梯度同步完成后，`models/sa_sdlora.py` 的 `_after_backward` 调用
`apply_adaptive_a_gradients()`。因此当前 `impact_ratio` 版本是在 optimizer step 前直接
修改已同步的 `A.grad`；`B`、`s` 和分类头不经过这套 gate。Task0 没有历史 `G`，代码
保持原始 `A` 梯度，不做历史抑制。

### 5.5 Task-boundary coordinate alignment

Adaptive-A 的第二个实际组件是任务边界的 historical aggregate coordinate alignment。
在当前任务开始时保存上一状态的 `A_old`，任务训练完成后得到 `A_new`。对每个 Q/V 分支，
代码先定义：

\[
O_{\mathrm{old}}
=G_{\mathrm{old}}
\frac{A_{\mathrm{old}}}{\lVert A_{\mathrm{old}}\rVert_F+\varepsilon},
\qquad
\hat A_{\mathrm{new}}
=\frac{A_{\mathrm{new}}}{\lVert A_{\mathrm{new}}\rVert_F+\varepsilon}.
\]

然后求：

\[
X^*
=\arg\min_X
\left\|X\hat A_{\mathrm{new}}-O_{\mathrm{old}}\right\|_F^2.
\]

实际代码通过

```text
torch.linalg.lstsq(new_down.t(), old_operator.t()).solution.t()
```

求解，而不是 `pinv`、`solve` 或持久化一个 transport matrix。得到的 `X*` 只在任务
边界使用：

\[
G_{\mathrm{history}}\leftarrow X^*.
\]

随后才执行当前 `B,s` 的 operator-preserving absorption：

\[
G_{\mathrm{new}}=X^*+\Delta G_t.
\]

因此 Adaptive-A 的顺序是：

```text
训练期 impact-ratio 梯度门控
    -> 得到 A_new
    -> 用 A_old/A_new 对历史 G 做 LS alignment
    -> 将当前 s*B 吸收到 G
    -> 保存 A_new/G_new
```

当新旧行空间完全一致时，alignment 可以保持历史有效算子；当行空间发生真实变化时，
它只能求新坐标中的最小二乘近似。对齐 residual 和 `max_condition` 只是诊断量，当前
实现不保存 `X*`。

### 5.6 方法含义

Adaptive-A 试图在 Frozen-A 与 Live-A 之间建立一个按层、按训练状态变化的连续折中：

- 行空间内部更新保留，避免过度抑制对当前任务有用的坐标重标定；
- 对历史算子影响大的行空间改变被降低，但至少保留 `5%` 的 perpendicular update；
- 当前任务影响相对大时，gate 接近 1，行为接近 Live-A；
- 历史算子影响相对大时，gate 接近 floor，行为接近 Frozen-A 的行空间约束；
- task boundary 再用 LS alignment 把已存在的历史 `G` 映射到新的 `A` 坐标。

需要严格区分：当前 `impact_ratio` 使用的是历史/当前有效算子影响代理，不直接计算旧
任务 loss，也不提供 teacher consistency 的数学保证。代码中虽然还支持
`functional_halfspace`、`risk_budgeted`、`pareto_knee` 和 `function_safe_pareto`，但它们
不是当前任务长度三路队列的主配置。

## 6. 分类头、prototype 和评估语义

三种方法都使用同一套分类头协议，因此 A 策略对比不会同时改变分类器结构：

1. 训练阶段使用增量扩展的 `SimpleLinear` FC，对当前任务交叉熵训练。
2. `sa_use_prototype_classifier=true` 时，任务结束后使用当前任务训练集、但采用 test
   preprocessing，提取 backbone feature。
3. 先对每个样本 feature L2 normalize，再按类求均值并再次 L2 normalize。
4. 历史 prototype 从 `sa_prototypes.pt` 读回，与当前任务 prototype 合并。
5. 评估阶段 `SharedAPrototypeNet` 的默认 `head_mode` 是 `proto`，因此当前三路队列的
   主 logits 是所有已见类别的 PrototypeCosineHead 输出。
6. `sa_dual_head=false`，所以不是 FC 和 prototype 的加权融合，也没有 Dual-B 头。

prototype 计算入口是
[_compute_prototypes](models/sa_sdlora.py#L2927-L2976)，分类头切换逻辑是
[SharedAPrototypeNet.forward](utils/inc_net.py#L437-L510)。当前实现不使用任务标签 mask；
所有已见类别在同一 cosine 分类空间中竞争。

## 7. 三种方法的横向比较

| 维度 | Frozen-A | Live-A | Adaptive-A |
|---|---|---|---|
| Task0 后 `A` | 冻结 | 继续更新 | 继续更新，但门控 |
| 当前 `B,s` | 训练 | 训练 | 训练 |
| 历史分支使用的 `A` | 固定 | 当前 live `A` | 当前 live `A` |
| 行空间改变 | 不允许 | 完全允许 | 按 block gate 保留 |
| 历史坐标对齐 | 不需要 | 关闭 | task boundary 开启 |
| 当前主 gate | 无 | 无 | `impact_ratio` |
| 当前 gate 是否修改 B/scale | 否 | 否 | 否 |
| merge 结构 | `A+G` | `A+G` | `A+G` |
| absorption | operator-preserving | operator-preserving | operator-preserving |
| 持久化 transport 参数 | 无 | 无 | 无，LS 解只在边界临时使用 |
| prototype/分类头 | 相同 | 相同 | 相同 |
| Dual-B | 关闭 | 关闭 | 关闭 |

从表示能力看，三者不是“每个任务一套独立 LoRA”：当前 cumulative state 始终把各任务
的更新提交到一个固定大小的 `G`，并维护跨任务共享的最新 `A`。差别只在于 `A` 如何
演化，以及历史 `G` 是否在任务边界重新对齐。

## 8. 参数量和存储量

每个 Q/V 分支包含一个 `A` 和一个 `B`，共 24 个分支。按 `d=768,r=10` 计：

- 每个矩阵参数量为 `7680`；
- 一套 24 分支的 `A` 或 `B` 约为 `184,320` 个参数；
- 训练时当前任务的 `A`、`B`、标量 `s` 和分类头可训练；
- Frozen-A 在 Task1 以后不再训练 `A`；Live-A 和 Adaptive-A 继续训练 `A`；
- cumulative deployment 只保留 24 个共享 `A` 和 24 个历史聚合 `G`，LoRA 状态大小与
  任务数量无关；
- prototype 和分类头按类别数量增长，这是类别增量学习的分类状态，不是每任务 LoRA
  因子；
- 当前脚本设置 `sa_delete_per_task_files=false`，但在 `sa_cumulative_state=true` 分支
  中保存逻辑不会为新任务写入 legacy per-task B 文件。

## 9. 当前设计的边界和论文表述建议

### 可以准确表述的内容

```text
We compare three shared-coordinate LoRA policies under the same cumulative
Aggregate-B state and the same prototype classifier. Frozen-A fixes the shared
down subspace after Task 0, Live-A updates it without intervention, and
Adaptive-A gates the row-space-changing component of the shared-A gradient by
the relative current/historical effective-operator impact and aligns the
historical aggregate at task boundaries.
```

### 当前代码不能直接声称的内容

- 不能说当前 Adaptive-A 训练了一个任务路由器；当前没有 task router。
- 不能说当前主队列使用了 function-space teacher constraint；当前主策略是
  `impact_ratio`。
- 不能说当前主队列启用了 prototype transport；`sa_coordinate_stable_transport` 没有
  被任务长度生成器打开。
- 不能说 `sa_shared_a_orthogonal=true` 保证整个训练过程中的 `A` 始终正交；代码明确的
  硬事实是 Task0 初始化使用正交矩阵。
- 不能把 coordinate alignment 的 residual 直接解释成旧类 loss 不增加；它是历史有效
  算子在新坐标下的最小二乘投影误差。

## 10. 评估指标解释

当前 `trainer.py` 的主 CNN top-1 曲线来自每个任务结束后的全已见类别预测：

\[
\mathrm{Final}=\mathrm{CNN\ top1\ curve}[-1],
\]

其中每个任务的 `top1` 是 `accuracy(...)["total"]`，不是只在新类上的 accuracy，也不是
任务标签 mask accuracy。进一步报告：

- `AAA`：各任务结束时全已见类别 top-1 的平均值；
- `Forgetting`：准确率矩阵中历史任务最佳值与最终值之差的平均；
- `Top5`：同一全类别预测空间下的 top-5；Task0 类别数不足 5 时，代码会将有效 k
  限制为可用类别数；
- prototype head、backbone、数据顺序和分类空间在 Frozen/Live/Adaptive 三路中保持一致。

因此三路差异主要应归因于共享 `A` 的训练策略、Adaptive-A 门控和 Adaptive-A 的边界
坐标对齐，而不应归因于分类头差异。

## 11. 一句话总结

当前方法族的核心不是保存多套任务 LoRA，而是在固定大小的 `A+G` cumulative state
中管理共享 LoRA 坐标：`Frozen-A` 完全固定坐标，`Live-A` 完全适应坐标，当前
`Adaptive-A` 用有效算子影响比率限制改变行空间的梯度，并在任务边界用无持久状态的
least-squares alignment 修复历史聚合坐标。

## 12. 2026-09-17 理论版：Momentum-Aware Adaptive-A

本节描述新实现及 `run_momentum_adaptive_a_t10_3datasets_2gpu.sh`，不追溯修改前已经
完成的实验。旧任务长度实验仍使用 gate floor `0.05`、gate EMA `0.9`；新理论版使用
floor `0`、gate EMA `0`，两者必须分开报告。

### 12.1 门控实际优化器方向

SGD momentum 为 `mu`、历史 buffer 为 `v` 时，优化器实际采用的方向不是当前 raw
gradient `D`，而是：

\[
D_{\mathrm{eff}}=D+\mu v.
\]

新实现对 `D_eff` 做 row-space 分解和 operator-impact gate，得到
`D_eff,gated`，随后写回：

\[
D_{\mathrm{raw,new}}=D_{\mathrm{eff,gated}}-\mu v.
\]

因此下一次 `optimizer.step()` 恢复出的实际方向严格等于门控后的方向。这个修正保证
`gamma=0` 时不会因为旧 momentum buffer 的垂直分量而继续旋转共享 `A` 的 row
space。

### 12.2 Ratio 与 Squared-Ratio

配置 `sa_adaptive_a_gate_formula` 支持：

\[
\gamma_{\mathrm{ratio}}=
\frac{I_{\mathrm{cur}}}
{I_{\mathrm{cur}}+\lambda_o I_{\mathrm{hist}}+\epsilon},
\]

以及：

\[
\gamma_{\mathrm{sq}}=
\frac{I_{\mathrm{cur}}^2}
{I_{\mathrm{cur}}^2+\lambda_o I_{\mathrm{hist}}^2+\epsilon}.
\]

其中当前实现把 `lambda_n` 固定为 `1`，`lambda_o` 对应
`sa_adaptive_a_stability_weight`。论文主实验先使用 `ratio` 和
`sa_adaptive_a_stability_weight=1`；`squared_ratio` 仅作为公式消融。

### 12.3 显式 Tangent-A

新增 `sa_adaptive_a_strategy=tangent` 作为严格端点。Task0 正常训练；Task1 以后，
对实际 optimizer direction 强制 `gamma=0`：

\[
D_{\mathrm{eff,gated}}=D_{\parallel}.
\]

在局部 SGD 步的左乘矩阵可逆时，这只改变当前 row space 内的坐标，不改变 row
space 本身。它与 Frozen-A 不同：Frozen-A 完全不更新 `A`，Tangent-A 仍更新
`D_parallel`。

### 12.4 本轮主实验协议

三数据集均为 T=10、rank 10、20 epoch、SGD、双卡每进程 batch size 64，即有效
batch size 128：

| 数据集 | seed | GPU | gate | floor | gate EMA |
|---|---:|---|---|---:|---:|
| CIFAR-100 | 1993 | 0,1 | ratio | 0 | 0 |
| ImageNet-R | 1995 | 4,5 | ratio | 0 | 0 |
| CUB-200 | 1 | 6,7 | ratio | 0 | 0 |

三者都关闭 Dual-B、HBD 和 prototype transport，保留 prototype classifier、task-boundary
least-squares alignment 与 `operator_preserving_absorb`。这样本轮变化只来自
momentum-aware operator-impact gate，而不是分类头或 transport。
