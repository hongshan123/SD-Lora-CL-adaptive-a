# HOEP-A 先验工作与实现审计

日期：2026-09-21

## 1. 审计范围

本轮同时核对论文描述与官方 executable code。代码版本如下：

| 方法 | 官方实现 | 审计 commit |
|---|---|---|
| SplitLoRA | `qhmiao/SplitLoRA` | `1d7b48821bf001c8d9f29d5d8da07e2bbf2b94e1` |
| LoDA | `HHHLF/LoDA_ICML2026` | `a07b79b290c2534ad0322679882a66ea67334c03` |
| Share | `ankit-vaidya19/Share` | `ed4bc9c261f4389b1c2abd639e4f55ae81a1028b` |
| Geo-LoRA | arXiv source | 2026-08-31 arXiv source；未找到可核验的官方代码仓库 |

## 2. 可执行代码结论

### 2.1 SplitLoRA

关键代码为 `lora/utils.py`：

- `LoRALayer.update()` 将已完成任务的 `B @ A` 合并进 dense `old_weight`，同时为新任务追加一套 A/B；
- `update_grad_after_train()` 对训练期间累计的 `x^T x` 激活协方差做 SVD；
- 下一任务 A 从启发式选择的 minor feature subspace 初始化；
- 代码中的分解对象实际是输入激活协方差，不是逐参数历史梯度矩阵。

与 HOEP-A 的实质差别：SplitLoRA 管理当前任务的新适配器及历史 dense merge；HOEP-A 不追加任务适配器，分解的是当前部署历史算子 `G A / ||A||` 在 rank 坐标中的精确能量。

### 2.2 LoDA

关键代码为 `models/decomposed_lora.py`：

- `FrozenA_TrainableB` 固定 A、训练 B；
- `prev_matrix/cur_matrix` 累计历史和当前输入激活协方差；
- general A 由 `M_cur + M_prev` 的主奇异方向构造；
- task-specific A 通过历史协方差 Cholesky 白化后的广义能量方向构造；
- merge 系数按当前/历史激活能量计算，并将任务更新写回 qkv 权重。

与 HOEP-A 的实质差别：LoDA 依据数据激活能量重新构造 general/task-specific down projections；HOEP-A 不保存数据统计，依据已部署历史有效算子本身的能量保护坐标，并继续保持单一 `(A,G)` 固定状态。

### 2.3 Share

关键代码为 `ImageClassification/utils.py` 与 `weight_update.py`：

- 拼接历史与当前 adapter 的重构 A/B；
- 对拼接矩阵做 PCA/eigendecomposition；
- 更新共享 EigenFlux components，并把旧 adapter 重投影为新的 loadings；
- 历史任务系数/loadings 仍属于方法状态。

与 HOEP-A 的实质差别：Share 学习跨 adapter 的共享主成分表示并保留任务 loadings；HOEP-A 不拟合 adapter 集合的 PCA，只对一个累计历史算子做规范坐标分账，不增加 per-task coefficient。

### 2.4 Geo-LoRA

论文方法包含：

- SPP：相邻任务投影矩阵的 Grassmann/chordal 距离；
- ACSA：principal-angle core 与 residual slack 的联合控制；
- MCBO：深层任务分支与历史子空间的归一化 overlap 约束。

最接近的碰撞点是“区分稳定 core 与可塑 residual”。不能声称 HOEP-A 首次提出稳定/可塑子空间。可保留的区别是：Geo-LoRA 的 SPP/ACSA 主要度量未加权子空间几何，HOEP-A 用 `G^T G` 对实际部署历史算子的方向能量加权，并通过全网络统一预算给出 task-boundary LS 不可恢复能量上界。

## 3. 最终实现

策略名：`operator_energy_partition`。

Task 0 正常训练。Task `t>0` 对每个 Q/V 分支执行：

1. 对 A 做 thin-SVD canonicalization，并同步变换 G、B，保持历史和当前算子不变；
2. 计算 `C=G/||A||`，对小矩阵 `C^T C` 做 `eigh`；
3. 用特征向量执行正交 gauge rotation，得到历史能量对角坐标；
4. 汇总全部 24 个 Q/V 分支的特征值，在全局预算内选择低能量谱簇；
5. 稳定行梯度严格置零；可塑行梯度投影到稳定行正交补和 Stiefel tangent；
6. 每个 optimizer step 后回缩 A，并同步重参数化 G、B 和 SGD momentum；
7. 任务边界仍执行原有 LS coordinate alignment 与 operator-preserving absorption；
8. checkpoint 仍只保存 A、G 和既有元数据，不保存谱、mask 或历史任务状态。

主参数：

```json
{
  "sa_adaptive_a_enabled": true,
  "sa_adaptive_a_strategy": "operator_energy_partition",
  "sa_hoep_energy_budget": 0.05,
  "sa_hoep_eigenvalue_rtol": 1e-6,
  "sa_live_a_absorb_mode": "operator_preserving_absorb"
}
```

## 4. P0 离线谱诊断

使用三个已有 rank-10、T=10 live-A v4 checkpoint 做离线诊断。每个模型共 24 个 Q/V 分支、240 个谱方向。

| 数据集 checkpoint | 预算 | 可塑方向 | mixed 分支 | Frozen 分支 | Live 分支 | 实际能量比 |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-100 | 1% | 48/240 | 54.2% | 45.8% | 0.0% | 0.977% |
| CIFAR-100 | 5% | 107/240 | 87.5% | 12.5% | 0.0% | 4.987% |
| CIFAR-100 | 10% | 144/240 | 95.8% | 0.0% | 4.2% | 9.894% |
| ImageNet-R | 1% | 37/240 | 33.3% | 66.7% | 0.0% | 0.971% |
| ImageNet-R | 5% | 89/240 | 70.8% | 29.2% | 0.0% | 4.938% |
| ImageNet-R | 10% | 123/240 | 75.0% | 16.7% | 8.3% | 9.870% |
| CUB-200 | 1% | 81/240 | 58.3% | 41.7% | 0.0% | 0.991% |
| CUB-200 | 5% | 138/240 | 58.3% | 20.8% | 20.8% | 4.969% |
| CUB-200 | 10% | 166/240 | 58.3% | 8.3% | 33.3% | 9.948% |

结论：三个数据集在 5% 主预算下均有超过 30% 的分支落在 `0 < k < r`，未整体退化为 Frozen 或 Live，满足立项书 P0 的第一项 Go 条件。该诊断来自既有不同训练版本的最终 checkpoint，只证明谱分区具有非平凡工作区间，不证明训练收益。

## 5. 当前边界

- 不能把“稳定/可塑子空间”或“谱能量阈值”作为首次提出；
- 当前可主张的是 fixed-state deployed-operator energy accounting、跨层全局预算、LS recoverability bound 的组合；
- 需要用 SplitLoRA、LoDA、Geo-LoRA、Share 做正文公式级对照；
- 在两任务 smoke 和三数据集单 seed 完成前，不应声称性能改善；
- 机制筛选配置关闭 prototype transport、Dual-B、HBD 和 bounded NormCap，避免把下游补偿误当成 A 策略收益。
