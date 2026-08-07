# SD-LoRA 固定状态方法下一步改进与执行建议

## 1. 文档用途

本文档供后续 Agent 在 cuda6 仓库 `/home/zhaoyang/SD-Lora-CL` 中继续执行。
目标是在不重复既有失败实验的前提下，修正当前证据链，并进一步恢复固定状态压缩在
ImageNet-R 上损失的精度。

执行前必须读取：

- `plan_sd.md`
- `experiment_sd.md`
- `experiment_note_sd.md`
- `method_revision_sd.md`
- 本文档 `next_improvement_sd.md`

当前 `run_sa_cumulative_tasklen_queue.sh` 仍在运行时，不得终止、覆盖日志或修改其输出目录。
等待队列自然结束后再开始正式代码实验。

## 2. 当前方法与已确认结果

当前主方法是 **Gauge-Aligned Cumulative Shared-A LoRA**：

- 每层历史任务的上投影在线折叠为累计矩阵 `H`；
- 下投影使用 canonical basis `Q^T`；
- 任务结束时使用 QR 与 gauge alignment 更新坐标；
- 不保存逐任务 B bank，LoRA 持久状态关于任务数为 `O(1)`；
- 无 replay、router、task-id 或逐任务 adapter 选择；
- 推理使用一个累计模型和 prototype 分类器。

参数量：

- 当前 LoRA：371,040；
- EXP-009 LoRA：2,027,530，当前减少 81.7%；
- 原始 SD-LoRA：3,686,400；
- 含 prototype 后，相对原始 SD-LoRA 减少 85.8%（INR）/87.9%（C100）。

正确按 seed 对齐后的 4-seed 配对结果如下：

| 数据集 | 指标 | 当前方法 | EXP-009 | 差值 | paired p |
| --- | --- | ---: | ---: | ---: | ---: |
| ImageNet-R | Final | 78.46 | 79.11 | -0.65 | 0.0256 |
| ImageNet-R | AvgAcc | 82.28 | 82.80 | -0.52 | 0.0086 |
| ImageNet-R | Forgetting | 7.86 | 7.90 | -0.04 | 0.8521 |
| CIFAR-100 | Final | 87.83 | 88.05 | -0.22 | 0.2815 |
| CIFAR-100 | AvgAcc | 91.52 | 91.63 | -0.12 | 0.1469 |
| CIFAR-100 | Forgetting | 8.68 | 8.67 | +0.01 | 0.9375 |

ImageNet-R 的 Final 与 AvgAcc 是小幅但统计显著的下降。当前论文主张只能是
“大幅压缩并保持可控性能损失”，不能写成“无损”或“统计等价”。

## 3. P0：开始新方法前必须修复的两个问题

### 3.1 修复多 seed 配对

`scripts/multiseed_stats.py` 当前按文件名字典序直接 zip。由于：

- 主方法顺序为 `seed1995, seed1, seed2, seed3`；
- EXP-009 顺序为 `seed1, seed1995, seed2, seed3`；

前两个 seed 被错误交叉配对，导致旧文档中的 `p=0.117/0.158` 无效。

必须修改为：

1. 从配置或文件名显式解析 seed；
2. 使用 `{seed: metric}` 字典按 seed inner join；
3. 对缺失、重复或两组 seed 集不一致直接报错；
4. 输出每个 pair 的 seed、A、B、差值；
5. 新增单元测试覆盖 `1/2/3/1995` 的词典序陷阱；
6. 重新计算 paired t-test、95% CI 和标准化 effect size；
7. “等价”必须使用预先定义 margin 的 TOST，不能用 `p>0.05` 代替。

建议先报告两套实用等价 margin：

- 严格：Final ±0.5，AvgAcc ±0.5，Forgetting ±0.5；
- 宽松：Final ±1.0，AvgAcc ±1.0，Forgetting ±1.0。

margin 必须在查看新候选结果前写入实验文档。

### 3.2 修复 gauge 诊断生命周期

当前 `models/sa_sdlora.py` 在 `super().incremental_train()` 返回后调用
`cumulative_gauge_diagnostics()`；但父类训练已经调用 `save_lora_parameters()`，
`_save_cumulative_state()` 已覆盖 `cumulative_up/canonical_down`。因此日志中的约 `1e-8`
主要是新状态与自身比较，不能证明历史算子被近乎精确保留。

修复要求：

1. 在 `_save_cumulative_state()` 覆盖旧状态之前计算真实诊断；
2. 将结果缓存到非持久字段 `_last_cumulative_gauge_diagnostics`；
3. 保存结束后只打印缓存值；
4. 诊断至少包括 projection residual、basis rotation、aligned preservation error；
5. 构造已知旋转、同 span 和 out-of-span 三类单元测试；
6. 断言 out-of-span 测试必须得到显著非零 residual；
7. 修复后至少重跑一个完整 INR seed，旧日志不得继续作为机制证据。

P0 单独提交，不与新方法实现混在一个 commit 中。

## 4. P1：先完成诊断和公平基线

### 4.1 完成任务长度实验

当前已完成 INR：

| tasks | Final | AvgAcc | Forgetting |
| ---: | ---: | ---: | ---: |
| 5 | 77.54 | 81.81 | 8.83 |
| 10 | 79.06 | 81.77 | 6.82 |
| 20 | 77.03 | 81.57 | 9.51 |
| 40 | 75.31 | 80.63 | 12.34 |

固定状态可以运行到 40 tasks，但长序列下 Final 下降、Forgetting 增大。由于任务数改变时
每任务类别数也改变，不能仅凭此表宣称任务数导致退化。

队列结束后必须补：

- EXP-009 的 INR T5/T20/T40；
- 原始 SD-LoRA 至少 INR T10/T20/T40；
- EXP-009 与当前方法的 C100 T5/T10/T20；
- 每个设置使用相同 seed、类序、epoch、batch size 和分类器；
- 同时报告 LoRA 状态、总附加状态、训练峰值显存、推理吞吐和 artifact 大小。

只有相对基线随任务数的差值更稳定，才能宣称固定状态具有更好的 scalability。

### 4.2 定位精度损失来自哪里

在不改变正式 rehearsal-free 方法的前提下，可使用旧训练数据做离线诊断，但必须标注
“diagnostic only”：

1. 当前累计 backbone + 已保存旧 prototype；
2. 当前累计 backbone + 离线重算旧 prototype；
3. 冻结 Base ViT + base-space prototype；
4. 历史任务 checkpoint + 当时 prototype；
5. 分层测量旧样本 feature cosine drift 和分类 margin drift。

判断规则：

- 若 2 明显优于 1，主要问题是 prototype 坐标过期；
- 若 2 仍接近 1 且明显低于 4，主要问题是 backbone 表示干扰；
- 只有第一种情况才继续做 prototype 修正；否则优先做算子合并结构改进。

## 5. P2 主改进：Fixed-Rank Union-SVD Cumulative LoRA

### 5.1 动机

当前 gauge 更新把历史算子投影进新任务的 `Q_new` 空间。新旧下投影子空间不完全一致时，
旧算子的 out-of-span 方向会被丢弃。建议不再先把旧状态强制投影到 `Q_new`，而是先合并
完整有效算子：

```text
M_t = H_(t-1) Q_(t-1)^T + s_t B_t A_t
```

再计算固定 rank `R` 的最优近似：

```text
M_t ~= U_R Sigma_R V_R^T
H_t  = U_R Sigma_R
Q_t^T = V_R^T
```

这会把新旧方向的联合空间纳入压缩，而不是只保留当前任务 A 的空间。

### 5.2 实现约束

- 不允许显式构造每层 `768 x 768` 的稠密矩阵；
- 使用左右因子的 QR 和最高 `2R x 2R` 小核心 SVD；
- Q/V 分支分别处理，保持现有归一化与 scale 语义；
- 持久 rank 固定，状态仍为 `O(1)`；
- artifact 升级版本并提供显式 v2 到新版本迁移；
- 旧版本加载不得静默退化；
- DDP 中只允许 rank0 保存并在 barrier 后恢复；
- 参数统计必须把 `H/Q/R`、prototype 和任何额外统计全部计入。

### 5.3 必须添加的测试

- 无截断时 union factorization 与显式 `M_old + M_cur` 等价；
- 截断误差等于显式 truncated SVD 的误差；
- union-SVD rank-R 误差不高于“投影到 Q_new 后合并”的误差；
- 保存、加载和 merged forward 的 feature/logit 一致；
- 连续多个任务后持久张量数量与任务数无关；
- 4 卡 Task0/Task1 smoke 无重复累计；
- 固定输入恢复前后的 logits 一致。

## 6. P2 实验矩阵与停止条件

不要一开始扫描大量 rank。严格按以下顺序：

### Stage A：结构验证

1. `union_svd_r4`，ImageNet-R seed1995；
2. `union_svd_r8`，ImageNet-R seed1995；
3. 与当前 gauge-r4、EXP-009 同 seed 比较。

进入 C100 的最低门槛：

- Final 不低于 79.10；
- AvgAcc 不低于 82.30；
- Forgetting 不高于 7.30；
- artifact/恢复/DDP 测试全部通过。

`r8` 若相对当前方法不能恢复至少 0.3 AvgAcc，且 Final 改善小于 0.2，则停止 rank 扩展，
不再尝试 r12/r16 的盲扫。

### Stage B：第二数据集

赢家在 CIFAR-100 seed1993 上必须满足：

- Final >= 87.70；
- AvgAcc >= 91.50；
- Forgetting <= 8.70；
- 相对原始 SD-LoRA 的总附加状态仍至少减少 60%。

### Stage C：多 seed

只有同时通过 INR/C100 单 seed 门槛的一个配置进入多 seed：

- 使用与现有结果完全相同的四个 paired seeds；
- 先修复的统计脚本按 seed join；
- 报告 mean/std、paired difference、95% CI、effect size 和 TOST；
- 不根据多 seed 结果继续对这四个 seed 调参。

论文级目标：

- 相对 EXP-009，两个数据集 Final/AvgAcc 的平均损失控制在 0.3 内；
- Forgetting 平均恶化不超过 0.3；
- 相对 EXP-009 LoRA 状态至少减少 60%，相对原始 SD-LoRA 至少减少 70%；
- 在 T20/T40 下相对基线的性能差距不随任务数显著扩大。

## 7. P3 备选：Fixed-Capacity Activation Sketch

只有 Union-SVD r8 未恢复 INR 且离线诊断确认主要是 backbone 干扰时才实现。

每层保存 rank `m=4` 或 `m=8` 的历史输入激活 sketch `U_l`，通过在线 SVD 或
Frequent Directions 固定容量更新。下一任务可选择一种方式：

```text
A_t = A_raw (I - U_l U_l^T)
```

或加入功能空间约束：

```text
lambda * ||DeltaW_l U_l||_F^2
```

原则：

- sketch 来自当前任务数据，任务结束后在线压缩，不保存图像；
- sketch 容量与任务数无关；
- sketch 必须计入方法状态和显存；
- 先固定 `m=4`、单一 lambda，禁止大范围超参数搜索；
- 必须比较参数空间 operator penalty 与 activation-aware penalty，证明后者确实更相关。

理论动机可参考 InfLoRA 的干扰子空间和 LoRA-DRS 的漂移抵抗思想，但实现必须突出
固定容量 streaming sketch 与在线累计 LoRA 的结合。

## 8. Prototype 方向的使用条件

只有 P1 诊断证明“离线重算旧 prototype 可以显著恢复准确率”时，才测试：

- 冻结 Base ViT prototype；
- 当前累计 LoRA prototype；
- 两空间 cosine logits 的固定权重融合。

这会增加一次 base forward 或额外工程复杂度，因此先作为诊断/消融，不默认进入主方法。
参数、FLOPs 和吞吐必须完整报告。

## 9. 已关闭、禁止重复的实验路线

除非出现新的理论证据，不得重复：

- LRPT rank/damping 的继续微调；
- generic adaptive LRPT；
- raw-space prototype/LRPT；
- class-mean transport；
- LoRA-aware JVP、classwise/layerwise sensitivity；
- EMA prototype consistency；
- 单纯 effective-operator L2 stability 或继续扫 lambda；
- 仅调整 rank、batch size、学习率而没有结构假设的实验；
- 路由到逐任务 LoRA，因其违背当前单模型固定状态主线。

每次实验前必须搜索 `experiment_sd.md`，确认没有同配置或等价配置。

## 10. 论文对比与相关工作

后续必须在相同协议下尽可能补齐：

- SD-LoRA 及其高效变体；
- EXP-009 Shared-A + prototype；
- InfLoRA；
- LoRA-DRS；
- CL-LoRA；
- C-LoRA；
- Share / Shared LoRA Subspaces；
- EASE、L2P、DualPrompt 等标准 CIL 基线。

需要明确与 Share 的区别：本文候选方法直接对累计有效算子做固定 rank 在线压缩，不保存
每任务 coefficient；持久 LoRA 状态严格与任务数无关。

## 11. Git 与实验记录要求

每个阶段都必须：

1. 修改前检查 `git status`，不得覆盖用户或正在运行实验产生的改动；
2. 先更新 `experiment_note_sd.md`，记录假设、风险和准备动作；
3. 代码、测试、配置完成后运行单测和 4 卡 smoke；
4. 每个独立阶段单独 commit，禁止把 P0、Union-SVD 和 sketch 混成一个提交；
5. 正式实验结束后更新 `experiment_sd.md`；
6. 结果无论正负都提交，禁止删除负结果；
7. 日志和 artifact 使用新目录，不覆盖历史结果；
8. 每次提交后记录 commit id 和可回退点。

建议提交顺序：

```text
1. Fix seed-keyed paired statistics and add equivalence tests
2. Fix pre-save cumulative gauge diagnostics
3. Add fixed-rank union-SVD algebra and unit tests
4. Integrate union-SVD cumulative state and migration
5. Add DDP smoke configs and record engineering validation
6. Record INR/C100 experiments and paired analysis
7. Add activation sketch only if Union-SVD stopping rule triggers
```

## 12. 最终决策原则

优先追求可验证、可解释的 Pareto 改善，而不是单个 seed 的最高 Top1。

- 若 Union-SVD 恢复 INR 的 0.5 左右 AvgAcc 缺口并保持固定状态，它应成为新主方法；
- 若 r8 仍无法恢复且 activation sketch 也无效，停止继续堆机制，保留当前方法作为
  “约 82% LoRA 压缩、0.5--0.7 INR 精度代价”的诚实结果；
- 任何 A 会主张都必须建立在正确 seed 配对、真实 pre-save gauge 诊断、同协议长序列基线
  和完整参数/计算核算之上。

## 13. 执行状态（2026-08-07）

- **P0-1 完成**（commit `442bdc5`）：`scripts/multiseed_stats.py` 重写为 seed 内连接配对，
  输出逐 pair 明细/paired t/95%CI/Cohen's dz/TOST；新增 7 个单测（全量 52→55 passing）；
  重算结果与 `next_improvement_sd.md` §2 一致（INR Final/AvgAcc 显著下降，C100 与
  Forgetting 不显著，TOST ±0.5 不等价）。论文/实验文档中的“统计等价”表述已全部修正。
- **P0-2 代码完成**（commit `ec23df0`）：gauge 诊断改为 `_save_cumulative_state` 覆盖
  旧状态前缓存到 `_last_cumulative_gauge_diagnostics`，日志读缓存；新增同 span/out-of-span/
  保存缓存单测（全量 55→58 passing）。完整 INR seed1995 重跑进行中（配置
  `exps/sa_cumulative_inr_seed1995_gauge_p0diag.json`，目录
  `ImageNetR_SA_CUMULATIVE_INR_SEED1995_GAUGE_P0DIAG`）；首任务 pre-save
  residual≈4.36e-2、rotation≈2.96e-2、preservation≈4.36e-2，确认旧日志 1e-8 无效。
- **P2 代码完成**（commit `1526068`）：v3 union-SVD 累计状态（纯函数、保存/加载/迁移、
  日志、配置、脚本、单测 61 passing）；Stage A 四档 INR 配置与 smoke 队列就绪。
- **P1 配置完成**：EXP-009 INR T5/T20/T40、C100 T5/T20，SD-LoRA INR T20/T40 配置与
  队列脚本；离线原型漂移诊断脚本 `scripts/diagnose_prototype_drift.py` 就绪。
- **待执行**：P0DIAG 完整结果 → union_svd DDP smoke → Stage A 四档完整 INR → 门槛判定 →
  C100/多 seed → P1 公平基线队列与离线诊断 → 论文回填。
