# SD-LoRA 下一阶段修改任务书

## 一、当前结论

当前主方法为 **Gauge-Aligned Cumulative Shared-A LoRA**：

- LoRA 参数量为 `371,040`，比 EXP-009 减少 `81.7%`。
- ImageNet-R 四种子相对 EXP-009：Final `-0.65`、AvgAcc `-0.52`，下降具有统计显著性。
- CIFAR-100：Final `-0.22`、AvgAcc `-0.12`，差异不显著。
- Union-SVD 在 `r4/r8` 优于同秩 Gauge，但 `r10` 的 Final 从 `79.06` 降到 `78.61`。
- `r10` Union-SVD 的新任务精度为 `83.87`，低于 Gauge 的 `85.54`，说明普通 SVD 压缩会牺牲能量较小但对新任务重要的方向。
- 长序列 `T20` 中旧类精度仅 `76.45`、新类精度为 `87.38`，说明还存在固定秩容量不足和历史方向逐步丢失。
- 不要继续尝试 damping、LRPT 或 prototype sensitivity transport，它们已经没有稳定收益或出现过灾难性遗忘。

## 二、执行前约束

1. 阅读 `plan_sd.md`、`experiment_sd.md`、`experiment_note_sd.md` 和 `next_improvement_sd.md`，避免重复实验。
2. 当前 `run_p1_tasklen_baselines.sh` 正在运行 EXP-009 ImageNet-R T20。等待整个队列结束，不能修改训练代码，否则后续队列会加载新代码。
3. 保留当前未提交的 `experiment_note_sd.md`，不要覆盖或回退。
4. 每个独立修改完成测试后单独提交 Git。
5. 每轮实验结束都记录配置、理论依据、结果和下一步判断。

## 三、首先修复诊断错误

修复 `scripts/diagnose_prototype_drift.py`：

- `build_merged_backbone(base_model, merged_state)` 会原地修改 `base_model`。
- 当前所谓 base backbone 实际上也是 merged backbone，因此 `base=final` 和 prototype cosine `1.0` 无效。
- 分别实例化两个 ViT，或者在 merge 前执行 `copy.deepcopy(base_model)`。
- 添加断言：两个模型对象和参数存储地址不同，base 模型中不存在已合并 LoRA。
- 用修复后的脚本重新计算 stored、final-recomputed、true-base 三组结果。
- 单独提交此修复，旧诊断结论不得继续引用。

## 四、主改进：Novelty-Protected Union

新增累计合并模式：

```json
{
  "sa_cumulative_merge": "protected_union",
  "sa_cumulative_rank": 10,
  "sa_protected_novel_rank": 2
}
```

设历史算子为 `M_old = H_old Q_old^T`，当前任务更新为 `M_cur`，目标算子为：

```text
M_true = M_old + M_cur
```

合并步骤：

1. 计算当前更新相对历史右子空间的残差：

```text
N = M_cur (I - Q_old^T Q_old)
```

2. 对 `N` 做低秩 SVD，保留前 `m` 个右奇异向量作为新任务保护子空间 `V_new`。
3. 在 `V_new` 的正交补中，对 `M_true` 做 rank `R-m` 截断 SVD，得到 `V_old`。
4. 构造固定秩正交基并重新拟合真实累计算子：

```text
V = [V_new; V_old]
H = M_true V^T
M_saved = H V
```

这样固定状态既保留历史高能量方向，也保证新任务独有方向不会全部被普通 SVD 截掉。

### 实现要求

- 不能显式构造 `768 x 768` 矩阵，沿用 QR 加小核心 SVD。
- 使用现有归一化和 LoRA scale 后的真实 `M_cur`。
- Task 0 退化为普通 fixed-rank SVD。
- artifact 升级版本，旧版本必须显式迁移，不能静默加载。
- 状态仍为 `O(1)`，不能保存逐任务 LoRA。
- 记录 total truncation error、old/current capture error、novelty energy、实际保护秩和子空间主角度。

## 五、必须添加的测试

- 已知正交新方向时，保护方向必须出现在最终基底中。
- `protected_union` 的 current-update capture error 小于普通 Union-SVD。
- `m=0` 与现有 Union-SVD 数值等价。
- 最终满足 `V V^T` 近似单位阵，并验证保存、加载和 merged forward 一致。
- 多任务后持久张量数量不随任务数增长。
- 4 卡 Task 0/Task 1 DDP smoke 无重复累计。
- 固定输入在恢复前后的 logits 一致。
- 全量现有测试必须通过。

## 六、实验顺序

只运行以下有限矩阵，不做大规模搜索：

1. `protected_union_r10_m2`，ImageNet-R seed1995。
2. `protected_union_r10_m4`，ImageNet-R seed1995。
3. 若至少一个配置相对 union-r10 提升 Final `>=0.3`，且 AvgAcc/Forgetting 恶化不超过 `0.3`，运行赢家的 `r16`。
4. 为 `r16` 补同秩 `gauge_r16` 和 `union_r16`，确保提升来自保护机制而非单纯增加容量。
5. INR 达到 Final `>=79.10`、AvgAcc `>=82.20`、Forgetting `<=7.30` 后，再跑 CIFAR-100 seed1993。
6. 两个数据集通过后才进入四种子和 T20 验证。

`r16` LoRA 约为 `595,968` 参数，相对 EXP-009 仍减少约 `70.6%`，相对原始 SD-LoRA 减少约 `83.8%`，不会破坏压缩目标。正式报告必须由参数统计脚本重新测量，并计入 prototype 和所有额外状态。

## 七、停止条件

出现以下任一情况应停止该方向：

- `m2/m4` 均不能提升 Final 至少 `0.3`。
- 新任务精度提高，但 Forgetting 恶化超过 `0.5`。
- `r16 protected_union` 不优于同秩 gauge/union。
- 改进只能依靠任务 ID、replay 或逐任务 LoRA 才成立。

不要立即修改论文主张。等公平任务长度队列、修正后的 prototype 诊断和 protected-union 实验全部完成后，再统一更新论文与证据表。
