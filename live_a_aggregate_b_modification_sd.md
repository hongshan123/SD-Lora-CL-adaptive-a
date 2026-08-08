# Live-A Aggregate-B：恢复 EXP-009 训练语义的固定状态改进方案

## 1. 文档目的与优先级

本文档供后续 Agent 在 cuda6 仓库 `/home/zhaoyang/SD-Lora-CL` 中执行。

本方案的实验优先级高于 `protected_union_modification_sd.md`。最新代码对照表明，当前
cumulative+gauge 相对 EXP-009 的主要损失不是普通 SVD 没有保护新任务方向，而是在线折叠
同时删除了 EXP-009 的历史分支梯度、共享 A 协同更新和历史 scale 调整。应先恢复这些训练
语义，再决定是否需要 Protected Union。

执行前必须阅读：

- `plan_sd.md`
- `experiment_sd.md`
- `experiment_note_sd.md`
- `next_improvement_sd.md`
- `protected_union_modification_sd.md`
- 本文档

当前 `run_p1_tasklen_baselines.sh` 未结束时，不得修改 `backbone/`、`models/`、`utils/` 或
队列后续进程会加载的任何代码。等待队列完全结束后再实现。

## 2. 已确认的性能现象

ImageNet-R 最终 old/new accuracy：

| 任务数 | cumulative+gauge | EXP-009 | 主要差值 |
| ---: | ---: | ---: | --- |
| T5 | 76.00 / 83.37 | 79.50 / 83.45 | old -3.50，new -0.08 |
| T10 | 78.26 / 85.54 | 78.60 / 85.39 | old -0.34，new +0.15 |
| T20 | 76.45 / 87.38 | 77.57 / 88.01 | old -1.12，new -0.63 |

下降主要来自旧类，而不是当前任务无法收敛。T5 Final `77.54 vs 80.33`，T20 Final
`77.03 vs 78.13`。当前方法在 T10 附近表现最好，但尚不能证明任务长度可扩展性优于
EXP-009。

## 3. 根因：当前方法改变了 EXP-009 的训练动力学

EXP-009 在任务 `t` 的历史分支为：

```text
y_old(x) = sum_i s_i * B_i A_t x / (||B_i|| ||A_t||)
```

其中历史 `B_i` 冻结，但共享 `A_t` 和历史 scale `s_i` 仍是训练参数。因此：

- 历史 B bank 会立即给共享 A 提供非零梯度；
- 共享 A 的更新同时受当前任务和全部历史方向影响；
- 历史 scale 能在后续任务中继续重标定任务贡献；
- 多个 B 虽然最终可合并，但在训练期间形成有用的过参数化优化路径。

当前 cumulative+gauge 使用：

```text
y_old(x) = H_old Q_old^T x
y_cur(x) = s_t B_t A_t x
```

训练期间 `H_old Q_old^T` 完全冻结，不依赖 A。当前 B 又从零初始化，因此新任务开始时
共享 A 几乎得不到 LoRA 分支梯度。保存时还需要把旧算子投影到新 A 子空间；修正后的
T10 诊断显示每次存在约 `1.3%--4.4%` 的真实保持误差。

EXP-009 最终历史 scale 为 `{2.21, 1.73, 1.31, 1.31, 0.94, 1.08, 1.00, 0.96,
1.09, 1.10}`，说明任务级幅度确实继续变化，不能假设 scale 调整完全无效。

## 4. 主方法：Live-A Aggregate-B

由于所有历史 B 共用同一个当前 A，可利用线性性将历史 bank 精确聚合：

```text
G_(t-1) = sum_i s_i * B_i / ||B_i||

y_old(x) = G_(t-1) A_t x / ||A_t||
y_cur(x) = s_t B_t A_t x
```

在历史 scale 冻结的条件下，这与 EXP-009 的历史 bank 前向及其对 A 的梯度严格等价，
但只需保存一个 `G`。任务结束后：

```text
G_t = G_(t-1) + s_t * B_t / ||B_t||
```

不要把 `||A_t||` 折入 G，因为下一任务必须使用新的、可训练的 A 及其当前范数。

### 4.1 配置接口

新增独立模式，不改动已有 gauge/union 行为：

```json
{
  "sa_cumulative_state": true,
  "sa_cumulative_merge": "live_a_aggregate_b",
  "sa_train_a_all_tasks": true,
  "sa_live_a_history_groups": 1
}
```

### 4.2 持久状态

每个 Q/V 分支只保存：

- `aggregate_up G`，形状 `[768, r]`；
- 当前共享 `A`，形状 `[r, 768]`；
- 必需的版本、rank、task_id 和归一化语义元数据；
- 当前 prototype artifact 保持不变。

不得保存逐任务 B。LoRA 状态应保持约 37 万量级，并且与任务数无关。所有参数统计必须
实际读取 artifact，计入 prototype、group scale 和额外统计，不能只写理论值。

### 4.3 前向与保存生命周期

- 历史分支必须直接使用模块中可训练的当前 A，禁止 `detach(A)`。
- 当前任务仍使用 EXP-009 的 raw branch：`s_t B_t A_t x`。
- 当前 B 仍按原逻辑零初始化。
- 非最终任务保存后，当前内存模型仍维持“旧 G + 当前 raw B”的语义，用于当前任务原型
  提取和阶段评估。
- 下一任务重建模型时，上一任务 B 才作为归一化项进入 G。
- 最终任务必须重建一次 aggregate backbone，确保最终评估与保存 artifact 完全一致。
- DDP 下只允许 rank 0 更新和保存 G，barrier 后所有 rank 加载一致状态。

## 5. 必须先做的等价性验证

### 5.1 离线 bank-to-aggregate 转换

新增迁移/诊断脚本，把已有 EXP-009 artifact 的全部 B、scale 和最终 A 转为 G：

```text
scripts/migrate_sa_v1_to_live_a_aggregate.py
```

使用固定输入验证：

- bank backbone 与 aggregate backbone 的 Q/V 输出一致；
- 最终 feature 最大绝对误差不超过 `1e-5`；
- prototype logits 最大绝对误差不超过 `1e-5`；
- INR/C100 最终 Top1 与 EXP-009 完全一致或仅有浮点舍入差异。

这一步只证明最终推理压缩可无损完成，不代表在线训练状态已经等价。

### 5.2 梯度等价测试

构造小型随机 A、多个 B 和固定 scale，对比 legacy bank 与 aggregate：

- forward 输出一致；
- `dL/dA` 一致；
- 输入梯度一致；
- `m=0`、单任务、多任务和不同 B 范数均覆盖；
- fp32 使用严格容差，AMP 使用合理相对容差。

## 6. 历史 scale 的因果消融

Live-A Aggregate-B 基础版会把每个任务保存时的 scale 固定进 G，而 EXP-009 会继续训练
全部历史 scale。必须用以下三组确定性能差来自哪里：

1. 已有完整 EXP-009：历史 A 和历史 scale 均继续训练。
2. `EXP-009-freeze-old-scale`：保留完整 B bank 和 live A，但历史 scale 保存后冻结。
3. `live-a-aggregate-b`：历史 scale 折入 G，无逐任务 B，live A。

第 2、3 组在数学上应具有相同历史前向和 A 梯度。相同 seed 下 Final/AvgAcc 差异若超过
`0.1`，优先检查实现、保存时机、当前 raw branch 和 DDP，不得直接继续调参。

若第 1、2 组差距不超过 `0.3`，说明历史 scale 适配不是主要因素，基础 aggregate 模式可
直接进入后续验证。

## 7. 可选增强：固定 K 组幅度适配

只有当 `EXP-009-freeze-old-scale` 相对完整 EXP-009 下降超过 `0.3` 时才实现。设置固定
`K=2` 或 `K=4` 个 aggregate groups：

```text
y_old = sum_k alpha_k * G_k A_t x / ||A_t||
```

- `G_k` 数量固定，与任务数无关；
- `alpha_k` 在新任务训练时可学习；
- 保存时将 alpha 折入对应 G，再把 alpha 重置为 1；
- 任务分组先采用确定性的 round-robin 或固定 recency bucket，不训练 router；
- K 的选择必须计入总参数、训练显存和 artifact 大小。

不要直接恢复逐任务 scale+B，也不要引入 task-id 推理。

## 8. 诊断日志

每任务至少记录：

- 第一个 batch 中历史分支贡献的 `||dL/dA||`；
- current branch 对 A 的梯度范数；
- `||G||`、`||A||`、当前 B 范数和 scale；
- 保存前后固定 probe 的 feature/logit 最大误差；
- old/new accuracy；
- artifact LoRA 参数、总附加参数和文件大小。

不得继续使用现有 `diagnose_prototype_drift.py` 的 base-vs-final 结论，直到其原地修改
base model 的 bug 被单独修复。stored prototype 与 final-space 重算的 `+0.37` 可以作为
有限诊断，但不能证明 base backbone 与 merged backbone 相同。

## 9. 测试要求

- bank-to-aggregate forward、feature、logit 和 A-gradient 等价测试；
- task0、task1、task2 保存/恢复测试；
- 非最终任务 raw-current 语义测试；
- 最终任务 rebuild 后与 artifact 一致；
- 多任务后持久张量数量与任务数无关；
- 旧 v1/v2/v3 artifact 不得静默按新版本加载；
- 迁移脚本保留原文件备份；
- 4 卡 Task0/Task1 DDP smoke；
- 全量现有测试通过。

## 10. 实验顺序与门槛

严格按以下顺序执行，不做大规模超参数搜索：

1. 离线迁移已有 INR/C100 EXP-009，证明最终推理无损聚合。
2. INR seed1995：`EXP-009-freeze-old-scale`。
3. INR seed1995：`live-a-aggregate-b`。
4. 第 2、3 组差异不超过 `0.1` 后，与完整 EXP-009 比较。
5. INR 达到 Final `>=79.10`、AvgAcc `>=82.30`、Forgetting `<=7.50` 后，运行 C100 seed1993。
6. C100 达到 Final `>=88.10`、AvgAcc `>=91.70` 后，进入四种子配对实验。
7. 四种子通过后再运行 T5/T20/T40 和 CUB 多 seed。

四种子目标：

- INR 相对 EXP-009 Final/AvgAcc 平均差距不超过 `0.3`；
- C100 相对 EXP-009 Final/AvgAcc 平均差距不超过 `0.3`；
- Forgetting 恶化不超过 `0.3`；
- LoRA 状态相对 EXP-009 至少减少 `60%`；
- T20/T40 下差距不得随任务数明显扩大。

## 11. 停止条件

- 离线 bank-to-aggregate 无法达到数值等价：停止训练，先修复归一化或生命周期错误。
- freeze-old-scale bank 与 aggregate 差距超过 `0.1`：判定实现不等价，不得解释为随机波动。
- live-A aggregate 在 INR seed1995 仍低于当前 gauge Final，停止该方向。
- K=4 幅度组仍不能把相对 EXP-009 的 Final 差距压到 `0.5` 内，停止增加 K。
- 任何改进依赖旧样本回放、task-id 推理或 O(T) B bank，均不作为主方法。

## 12. Git 与实验记录

- 诊断修复、基础 aggregate 实现、迁移脚本、K-group 增强必须分别提交。
- 每次实验前检查 `experiment_sd.md`，不得重复已有失败实验。
- 每个实验完成后立即更新 `experiment_sd.md`。
- 实时观察、风险和队列状态写入 `experiment_note_sd.md`。
- 不得覆盖其他 Agent 的未提交修改。
- 只有代码测试和 artifact 一致性通过后才能启动完整训练。

本阶段的核心问题不是“更聪明地压缩一个已经冻结的历史算子”，而是验证能否在 O(1)
状态下保留 EXP-009 的 live shared-A 训练路径。如果该假设成立，Live-A Aggregate-B 应当
比 Gauge/Union-SVD 更直接地恢复旧类性能，同时保留当前方法的参数与训练显存优势。
