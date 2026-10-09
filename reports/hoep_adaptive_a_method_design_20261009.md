# HOEP-A：固定 LoRA 状态下的历史算子能量分区 Adaptive-A

整理日期：2026-10-09。本文根据现有实现与实际运行配置总结方法；方法简称沿用代码中的 **HOEP-A**。

> 核心思路：在固定 rank 的共享 LoRA 表示中，用历史有效算子的谱能量识别需要保护的方向，将低能量方向分配为可塑容量，在训练中保持稳定方向不变，并在任务边界通过最小二乘对齐与精确吸收更新固定大小的部署状态。

## 1. 方法解决的问题

在持续学习中，共享下投影 A 决定 LoRA 能够使用的输入子空间。

- **Frozen-A**：首任务之后固定 A，后续任务主要学习 B；历史坐标稳定，但新任务只能利用已有子空间。
- **Live-A**：后续任务持续更新 A，增加适应新任务的机会；历史分支也依赖当前 A，因而可能产生历史功能漂移及任务边界的不可恢复损失。
- **HOEP-A**：每个任务开始时重新评估历史算子的方向能量，固定承载较多历史能量的方向，允许较低能量方向更新。

HOEP-A 的设计包含两个相互配合的部分：

1. **固定大小的累计 LoRA 表示**：使用一组 A/G 汇总历史任务，不为每个新任务追加部署适配器。
2. **历史算子能量分区**：在该固定表示内分配稳定与可塑方向，限制可塑方向所涉及的历史算子能量。

这里的“共享 A”指**跨任务共享**。不同 Transformer 层以及 Q/V 分支分别拥有各自的 A；并非整个 ViT 共用一个 A。

## 2. 模型结构与符号

当前骨干为预训练 ViT-B/16。LoRA 插入每个 Transformer block 的 Q、V 投影，K 投影保持基础结构；预训练骨干参数冻结。

对一个 Q 或 V 分支，设：

| 符号 | 形状 | 含义 |
|---|---|---|
| W₀ | d×d | 冻结的预训练投影 |
| A | r×d | 跨任务共享的下投影 |
| G | d×r | 已累计的历史上投影系数 |
| Bₜ | d×r | 当前任务上投影 |
| sₜ | 标量 | 当前任务的可学习缩放系数 |
| Ā | r×d | A 的 Frobenius 范数归一化结果 |

忽略数值稳定项 ε，定义：

\[
\bar A=\frac{A}{\|A\|_F}.
\]

任务训练期间，该分支的有效增量为：

\[
\Delta W_t=G_{t-1}^{\mathrm{coord}}\bar A_t+s_tB_tA_t.
\]

`coord` 表示 G 可以随规范坐标变换而重参数化。它不是通过任务损失直接优化的历史训练参数；逐步回缩会同步变换它，以保持坐标变换前后的有效算子一致。

**两条分支的归一化不同**：历史分支使用 GĀ，当前分支使用原始 sBA。现有 HOEP 配置明确关闭当前分支的 A/B 范数归一化。

每个新任务复用当前任务分支的结构，学习新的 B 和缩放系数；部署时将其吸收到 G。当前实现的缩放系数由 Q/V 分支共享，不应写成每个谱方向都有独立门控系数。

## 3. 任务开始：建立历史算子的规范坐标

### 3.1 对 A 做薄 SVD，并同步变换 G/B

对任务开始时的满行秩 A 做：

\[
A=U\Sigma V^\top=TQ,\qquad T=U\Sigma,\quad Q=V^\top.
\]

Q 满足 QQᵀ=Iᵣ，且 ‖Q‖F=√r。同步设置：

\[
A_c=Q,\qquad
G_c=GT\frac{\|Q\|_F}{\|A\|_F},\qquad
B_c=BT.
\]

于是：

\[
G_c\frac{A_c}{\|A_c\|_F}=G\frac{A}{\|A\|_F},
\qquad B_cA_c=BA.
\]

这一步仅改变因子的表示坐标，保留历史分支与当前分支的有效算子。代码使用 float64 完成分解和关键代数；检测到 A 秩亏时直接报错。

### 3.2 对历史有效算子做能量分解

在行正交坐标中，令：

\[
C=\frac{G_c}{\sqrt r},\qquad
H=C^\top C=O\Lambda O^\top,
\quad \Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_r).
\]

特征值按从小到大排列。用正交矩阵 O 同步旋转：

\[
A_e=O^\top A_c,\qquad G_e=G_cO,\qquad B_e=B_cO.
\]

旋转仍然保持 GĀ 和 BA 不变。令 cᵢ 为 Gₑ/√r 的第 i 列、aᵢ 为 Aₑ 的第 i 行，则：

\[
M_{\mathrm{old}}=\sum_{i=1}^r c_i a_i,
\qquad \|c_i\|_2^2=\lambda_i,
\qquad \|M_{\mathrm{old}}\|_F^2=\sum_i\lambda_i.
\]

因此，λᵢ 衡量的是该方向承载的**历史有效算子能量**。它不是 A 的奇异值，也不是当前任务梯度大小、分类重要性或 Fisher 信息。

## 4. 全网络低能量预算与方向选择

ViT-B 有12层，每层 Q/V 两个分支，共24个分支。rank10 时共有240个谱方向。

将全部分支的谱能量汇总：

\[
E_{\mathrm{all}}=\sum_{\ell=1}^{24}\sum_{i=1}^{r}\lambda_{\ell,i}.
\]

选择可塑方向集合 P，使得：

\[
E_{\mathrm{plastic}}=
\sum_{(\ell,i)\in P}\lambda_{\ell,i}
\le \delta E_{\mathrm{all}},\qquad \delta=0.05.
\]

当前实现按以下规则分配：

1. 每个分支内部按特征值接近程度形成谱簇，默认相对容差为10⁻⁶。
2. 同一谱簇整体选择或整体保留，避免从退化特征空间中任意挑选一个特征向量。
3. 汇总所有分支的谱簇，按簇内平均特征值从小到大排序。
4. 逐簇加入可塑集合；若下一个完整簇超过全局预算，则停止。
5. 剩余方向作为稳定方向，在本任务中保持不变。

**5% 是能量比例，不是参数比例或可更新行数比例。** 当较多方向承载的历史能量很低时，可塑方向数可以明显超过总方向数的5%。不同层、不同 Q/V 分支以及不同任务的可塑维数也可以不同。

该选择是确定的低能量前缀规则，不是背包最优化。全局预算不要求每个分支都满足独立5%约束。历史总能量近零时，代码开放全部方向。

分区在每个任务开始时重新计算，本任务内的 mask 固定；不会在每个 minibatch 后重新计算谱分区。

## 5. 训练期间：稳定行固定，可塑行在约束空间中移动

Task0 没有历史知识，正常学习 A/B/缩放系数，跳过 HOEP 分区与逐步回缩。Task1 起执行以下约束。

### 5.1 梯度投影

将 A 分为稳定行 Aₛ 和可塑行 Aₚ，将当前梯度的可塑部分记为 Dₚ。首先去掉稳定子空间分量：

\[
D_0=D_p-(D_pA_s^\top)A_s.
\]

再投影到行正交约束的切空间：

\[
D_{\mathrm{tan}}=
D_0-\operatorname{sym}(D_0A_p^\top)A_p,
\qquad \operatorname{sym}(X)=\tfrac12(X+X^\top).
\]

稳定行梯度置零，可塑行使用 Dtan。这样同时满足与稳定行的正交约束和可塑行的 Stiefel 切空间约束。

实现中还处理 SGD 的两个细节：

- 抵消 SGD 随后加入到 A 的 coupled L2 weight decay，避免径向缩放影响规范坐标或被吸收到 B。
- 将 A 的 momentum buffer 同样投影，防止旧动量推动稳定方向或离开约束空间。

DDP 下在梯度同步之后执行这一处理。HOEP 当前实现要求使用 SGD。

### 5.2 每步更新后的回缩与同步重参数化

SGD 更新产生候选矩阵 A* 后：

1. 从任务开始时的 anchor 恢复稳定行。
2. 将候选可塑行投影到稳定行的正交补。
3. 对可塑行转置做薄 QR，获得新的行正交矩阵 A⁺。
4. 计算坐标映射 T*=A*(A⁺)ᵀ，并检查 A*=T*A⁺ 在数值容差内成立。
5. 同步变换历史系数和当前上投影：

\[
G^+=G^*T^*\frac{\|A^+\|_F}{\|A^*\|_F},
\qquad B^+=B^*T^*.
\]

因此：

\[
G^+\bar A^+=G^*\bar A^*,\qquad B^+A^+=B^*A^*.
\]

同时重新投影 A 的动量，并用 T* 变换 B 的动量。回缩还记录正交误差、行空间重建误差以及历史/当前算子误差。

这保证的是**回缩这一步不额外改变候选有效算子**。训练更新本身仍然会改变低能量方向及模型功能；高能量稳定行保持不变也不等于整个 ViT 的旧类预测保持不变。

当前训练沿用当前任务的分类交叉熵及基础训练流程。HOEP 的核心控制来自梯度投影与回缩，不添加旧样本重放或 HBD 蒸馏目标。B 的全部列仍可学习，稳定 A 行也可被当前任务的 B 利用。

## 6. 任务边界：历史 LS 对齐与当前分支精确吸收

### 6.1 将任务开始时的历史算子重新表达在最终 A 中

记任务开始时、已经规范化并旋转的历史状态为 G₀/A₀，任务训练结束的共享矩阵为 Aₜ。求解：

\[
G_{\mathrm{LS}}=
\arg\min_X\|X\bar A_t-G_0\bar A_0\|_F^2.
\]

一般写为：

\[
G_{\mathrm{LS}}=M_{\mathrm{old}}\bar A_t^\dagger.
\]

这里使用任务开始时的历史 anchor 重新对齐，不是直接把训练期间的历史系数原样写入文件。若 A 的旧、新行空间相同，LS 可精确保留历史算子；发生行空间变化时，落在新行空间外的部分无法由固定 rank 表示恢复。

### 6.2 精确吸收当前任务分支

现有 HOEP 使用 `operator_preserving_absorb`：

\[
G_t=G_{\mathrm{LS}}+s_t\|A_t\|_F B_t,
\qquad M_{\mathrm{deploy},t}=G_t\bar A_t.
\]

于是：

\[
M_{\mathrm{deploy},t}
=G_{\mathrm{LS}}\bar A_t+s_tB_tA_t.
\]

当前 B 分支被精确吸收；历史部分的误差来自 LS 对齐的表示限制。论文中应分别描述这两部分，不能把整次合并笼统称为“无损合并”。

部署骨干使用合并后的 A/G，随后由当前任务训练集构造新类 prototype，合并已有旧类 prototype。下一任务复用新的部署状态，并建立新的当前任务分支。

## 7. 可恢复性的数学依据与适用范围

设最终 Aₜ 行正交，Pₜ=AₜᵀAₜ 是其行空间的正交投影。最优 LS 历史残差为：

\[
\mathcal R_\ell=
\min_X\|X\bar A_{\ell,t}-M_{\ell,\mathrm{old}}\|_F^2
=\|M_{\ell,\mathrm{old}}(I-P_{\ell,t})\|_F^2.
\]

由于稳定行始终留在最终行空间内，其历史分量可被完整表达。残差仅来自任务开始时被选为可塑的分量。由正交投影不增加 Frobenius 范数可得：

\[
\mathcal R_\ell
\le\sum_{i\in P_\ell}\lambda_{\ell,i}.
\]

跨所有 Q/V 分支求和：

\[
\sum_\ell\mathcal R_\ell
\le E_{\mathrm{plastic}}
\le\delta\sum_\ell\|M_{\ell,\mathrm{old}}\|_F^2.
\]

因此，理想满秩、精确算术及稳定行保持条件下，HOEP 将**单次任务转换的历史算子最优 LS 不可恢复平方残差**限制在全网络历史算子能量的5%内。实现需要考虑 ε、谱簇容差和浮点误差。

这个性质不直接给出：

- 旧类 Accuracy 下降不超过5%的保证；
- 历史训练分布上的输出误差上界；
- ViT 多层非线性组合的分类遗忘上界；
- 所有任务累积遗忘保持5%的保证；
- 优于 Frozen-A 或 Live-A 的性能保证。

这里的能量保护是可恢复性机制依据，实际准确率仍受层内扰动、后续层输入变化、当前 B、分类间隔及 prototype 几何影响。

## 8. 固定参数量的准确表述

每个分支部署必需的适配张量为 A 和 G，元素数为 rd+dr。24个分支、d=768、r=10 时：

\[
N_{\mathrm{adapter}}=24\times2\times10\times768=368{,}640.
\]

该数量不随任务数增长。B 和当前缩放系数用于训练并被吸收，不需要为每个历史任务保留一个新的部署适配器。每个任务的谱、mask、anchor 和 optimizer cache 不作为新增的历史适配器持久化。

“固定参数量”应限定为**部署所需 LoRA 适配状态**：

- 分类头、类 prototype 随类别数增加，通常是 O(Cd)。
- 训练阶段包含当前 B、优化器与临时 anchor 等额外内存。
- 为复核保留的任务文件、诊断 mask schedule、checkpoint、logits 和日志会使实验目录增大。
- 因此不能声称整个分类系统或全部实验磁盘空间都不随任务增长。

## 9. 当前实际配置

以下字段来自正在使用的 CIFAR 配对消融配置，而非基础模板的默认 batch 设置：

```json
{
  "lora_rank": 10,
  "sa_train_a_all_tasks": true,
  "sa_adaptive_a_enabled": true,
  "sa_adaptive_a_strategy": "operator_energy_partition",
  "sa_hoep_partition_policy": "low",
  "sa_hoep_energy_budget": 0.05,
  "sa_hoep_eigenvalue_rtol": 1e-6,
  "sa_adaptive_a_eps": 1e-8,
  "sa_cumulative_state": true,
  "sa_cumulative_merge": "live_a_aggregate_b",
  "sa_live_a_history_groups": 1,
  "sa_live_a_coordinate_align": true,
  "sa_live_a_absorb_mode": "operator_preserving_absorb",
  "sa_normalize_current_branch": false,
  "sa_use_prototype_classifier": true,
  "sa_coordinate_stable_transport": false,
  "sa_dual_head": false,
  "sa_hbd_enabled": false,
  "optimizer": "sgd",
  "batch_size": 64,
  "init_epoch": 20,
  "epochs": 20,
  "max_tasks": 10
}
```

实际 HOEP 实验采用两卡 DDP，每卡64，总 batch128。当前 CIFAR 队列使用 seeds1993–1995；1993 的完整原实验结果复用，补齐1994/1995并进行方向选择消融。

**NormCap 和 prototype transport 当前都关闭。** HOEP 校验要求精确吸收模式；不能直接把 NormCap 当作当前 HOEP 已启用的组件。cuda6 的 Live/Frozen 边界诊断保留 NormCap/transport，属于另一个协议，其准确率和机制结果应分别引用。

## 10. 算法流程

```text
输入：冻结 ViT、固定 rank r、历史状态 {A_l, G_l}、当前任务训练集 D_t

若 t=0：
    正常训练 A、B、s 和分类训练头
否则：
    对每个 Q/V 分支规范化 A，并同步变换 G、B
    分解历史有效系数的 Gram 矩阵，旋转到能量坐标
    用全网络能量预算选择完整低能量谱簇
    保存本任务稳定行 anchor 和可塑 mask
    对每个 minibatch：
        计算当前任务分类损失并反向传播
        DDP 同步梯度后，投影 A 梯度与动量，处理 A 的 weight decay
        执行 SGD
        恢复稳定行、回缩可塑行，并同步变换 G、B 与相应动量

任务结束：
    将任务开始时的历史算子 LS 对齐到最终 A
    精确吸收 sBA 到 G，保存新的固定 A/G 状态
    在部署骨干上由当前任务训练集计算新类 prototype，保留旧 prototype
    下一任务重新计算谱分区
```

## 11. 正在验证的方向选择消融

| 对照 | A 的更新策略 | 检验目的 |
|---|---|---|
| Frozen-A | Task0 后完全固定 | 固定子空间端点 |
| Raw Live-A | 全部 A 自由学习 | 无分区约束端点 |
| Canonical all-plastic Live-A | 保留规范化/切空间/回缩，开放全部行 | 检查收益是否来自几何优化流程 |
| HOEP low | 全局预算选择低历史能量谱簇 | 当前主设计 |
| Matched random | 同 seed、同任务、每分支匹配 HOEP 可塑维数，随机选择完整谱簇 | 检查低能量排序的作用 |
| Matched high | 同样匹配维数，选择能量最大的可行完整谱簇 | 检查保护高能量方向的作用 |

Random/high 保留相同几何训练流程，但不再施加5%能量上限，否则不能形成有效的方向选择对照。Random 使用独立 CPU 随机生成器；若匹配维数无法由完整谱簇组合得到，明确失败，不拆分谱簇凑数。

每任务、每分支的开放维数和相应诊断写入 `hoep_partition_schedule.json`，属于实验复核资料，不是推理时新增的任务专属适配器。

## 12. 已有证据与当前结论

### 12.1 同协议单 seed 三数据集结果

协议：T10、rank10、每任务20epoch、双卡每卡64；LS/精确吸收开启，NormCap/transport/Dual-B/HBD关闭。数值为 Final / AAA：

| 方法 | CIFAR-100 seed1993 | ImageNet-R seed1995 | CUB-200 seed1 |
|---|---:|---:|---:|
| Frozen-A | 87.75 / 92.286 | 78.90 / 82.173 | 84.14 / 89.423 |
| Live-A | 87.31 / 91.822 | 78.72 / 82.279 | 76.54 / 87.856 |
| HOEP-A | 87.92 / 92.383 | 78.95 / 82.172 | 84.10 / 89.445 |

HOEP 在 CIFAR 优于这两个端点，在 CUB 避免了该 Live 配置的大幅性能下降，三数据集接近 Frozen/Live 的较优端点。但改进较小，只有每数据集单 seed；原预注册“至少两个数据集提升0.30”的门槛未通过，原 NO-GO 记录保留。

### 12.2 仍需验证的主张

- 同协议多 seed 的 HOEP/Frozen/Live 比较。
- Low 是否优于匹配维数的 random/high，以及几何全开放控制。
- 是否需要逐数据集预算，以及预算对准确率/遗忘的影响。
- 历史算子残差能否预测实际旧类遗忘。
- HOEP 与 NormCap/transport 的交互；这需要新的同协议配对实验。

Functional-HOEP 已完成的是 activation-aware mask 的 shadow 诊断，训练仍使用 operator-HOEP；当前归档记录中其训练 Phase B 未启动。它不能作为已经验证的新主方法结果写入上述表格。

截至2026-10-09 11:46核查，cuda6 的另一协议中 Live-A 在 ImageNet-R 三个预定 seed 均优于 Frozen-A；前两个 seed 的全局优势在 NormCap 后明显扩大。该观察支持继续分析学习与合并的相互作用，不等于 HOEP 已获同样的三 seed 确认。

## 13. 可用于论文的方法概述

> 我们采用固定秩的共享下投影与累计上投影表示历史任务。在每个任务开始时，将历史有效 LoRA 算子转换到行正交规范坐标，对其系数 Gram 矩阵进行谱分解，并通过跨层全局能量预算分配稳定与可塑方向。训练期间固定稳定方向，仅在其正交补内更新可塑方向，结合切空间梯度投影和保持有效算子的回缩重参数化。任务结束后，通过最小二乘历史对齐和当前分支精确吸收更新固定大小的适配状态。该分区为单次任务边界的最优历史算子不可恢复残差提供能量上界，实际分类收益通过配对实验检验。

当前可清楚表述的是这一**表示、能量分配、几何更新与任务边界处理的组合设计**。是否具有足够的新颖性、是否能稳定改善三个数据集，需要相关工作对照和完整消融支持；本文不把方法动机写成已证实的性能结论。

## 14. 实现与证据入口

当前 CIFAR 消融工作树中的代码：

- [谱能量、规范化、方向选择、梯度投影与回缩](https://github.com/hongshan123/SD-Lora-CL-adaptive-a/blob/codex/hoep-cifar-ablation-20261008/backbone/operator_energy_partition.py)
- [Q/V 前向、任务开始分区、逐步更新及 A/G 保存](https://github.com/hongshan123/SD-Lora-CL-adaptive-a/blob/codex/hoep-cifar-ablation-20261008/backbone/sa_lora.py)
- [历史 LS 坐标对齐](https://github.com/hongshan123/SD-Lora-CL-adaptive-a/blob/codex/hoep-cifar-ablation-20261008/backbone/coordinate_stability.py)
- [模型配置校验、训练 hook、prototype 与 mask schedule](https://github.com/hongshan123/SD-Lora-CL-adaptive-a/blob/codex/hoep-cifar-ablation-20261008/models/sa_sdlora.py)
- [配对矩阵与配置生成](https://github.com/hongshan123/SD-Lora-CL-adaptive-a/blob/codex/hoep-cifar-ablation-20261008/scripts/hoep_cifar_ablation.py)

已有记录：

- [单 seed 同协议结果与适用范围](adaptive_a_proximity_20261008.md)
- [HOEP Task0/1 真实数据与固定状态检查](../hoep_p2_real_data_smoke_results.md)
- [Functional-HOEP shadow 诊断结果](../functional_hoep_phase_a_results.md)
- [当前 CIFAR 消融启动记录](hoep_cifar15_stable_restart_20261009.md)
- [cuda6 Live/Frozen 三 seed 结果及协议区别](cuda6_live_frozen_progress_20261009_1146.md)

本文不更改训练代码、实验配置或运行队列。动态队列状态应以对应 runtime 的 status.json 与完整日志为准。
