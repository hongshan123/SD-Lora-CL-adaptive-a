# P0 验证后的实验执行计划

更新时间：2026-08-09

## 1. 目标与当前起点

当前 Gloo clean smoke 已证明：

- 两次 control 的逐任务训练状态、RNG 和指标完全一致；
- Dual-B 与 control 的逐任务训练状态完全一致；
- Dual-B 在两任务 smoke 中将 AAA 从 `90.515` 提高到 `90.895`；
- Final、最终旧类/新类准确率完全不变；
- `lambda=1` 时 fused/prototype logits 差为 `0`；
- 当前测试为 `96 passed`，Git HEAD 为 `e6b7626`，工作树干净。

smoke 只验证机制，不能作为论文性能结果。后续目标是在正式训练协议下确认 Dual-B 保持 Live-A Final，同时修复 AAA，并补齐 CCF-A 所需证据。

## 2. P0.5：正式训练后端门槛

完整实验默认使用 NCCL，而当前 smoke 使用 Gloo。启动完整训练前必须先完成：

1. prototype、`lambda/tau` 和 rank consistency collectives 根据 backend 选择 device：
   - NCCL：CUDA tensor；
   - Gloo：CPU tensor。
2. eval/calibration 的 tensor snapshot 使用 `detach().cpu().clone()`，不能保存引用。
3. 新增 snapshot mutation 单元测试和 NCCL 四卡同步测试。
4. 从已提交且干净的 Git HEAD 启动一次 NCCL 两任务 Dual-B smoke。
5. manifest 必须在启动时记录真实 commit，禁止训练后回填。

验收：NCCL smoke 无报错、无非确定性 warning、四 rank 同步 PASS、Final fused/prototype 差不超过 `1e-6`。

P0.5 单独提交；不通过时不得进入 P1。

## 3. 固定实验协议

第一轮继续使用已有 paired released-code 协议，不同时修改优化器：

- 4 卡 DDP，GPU `0,1,2,3`；
- 每卡 batch size `32`，全局 batch size `128`；
- SGD、20 epochs、现有学习率；
- `sa_deterministic_training=true`；
- `dist_backend=nccl`；
- 相同预训练权重、seed、类序和数据预处理；
- Dual-B 调度固定，禁止查看测试结果后调整；
- 每次运行使用不存在的新目录和 run manifest。

若后续验证论文声明的 Adam/30 epochs 协议，应建立独立实验组，并让所有基线同时重跑，不能混入本轮结果。

## 4. P1：ImageNet-R seed1995 完整配对

### 4.1 运行顺序

1. Live-A prototype control，`N=10`、seed1995。
2. Live-A + Dual-B，完全相同训练配置，只增加 dual-head 开关。
3. 同协议 SD-LoRA seed1995。
4. 同协议 EXP-009 seed1995。

建议输出目录：

```text
INR_P1_LIVEA_CONTROL_SEED1995_NCCL/
INR_P1_LIVEA_DUALB_SEED1995_NCCL/
INR_P1_SDLORA_SEED1995_NCCL/
INR_P1_EXP009_SEED1995_NCCL/
```

### 4.2 必须输出

- 每个 task 的 FC、prototype、fused Top1/Top5；
- 每个 task 的 `post_train_hash` 和 RNG hash；
- Final、AAA、Forgetting、BWT、最终旧类准确率；
- accuracy matrix；
- LoRA、FC、prototype 和总持久参数；
- artifact consistency。

### 4.3 进入下一阶段的门槛

- control 与 Dual-B 的 task0--9 `post_train_hash` 完全一致；
- Dual-B prototype 曲线与 control 曲线完全一致；
- `fused Final == prototype Final`，容差 `1e-6`；
- Dual-B fused AAA 相对 prototype 至少提升 `0.7` 个百分点；
- Dual-B Final 不低于同协议 SD-LoRA；
- Dual-B AAA 不低于 SD-LoRA 超过 `0.3` 个百分点；
- ImageNet-R 必要总状态约 `675,840`，相对 SD-LoRA 减少至少 80%。

任一训练 hash 不同，立即停止并定位，不允许用最终准确率接近代替轨迹验收。

## 5. P2：CIFAR-100 seed1993 完整配对

只有 P1 通过后执行，配置与 P1 保持一致：

- Live-A + Dual-B；
- SD-LoRA；
- EXP-009。

P0 已证明 Dual-B 与 Live-A 训练轨迹一致，因此 P1 通过后，CIFAR-100 不必再单独重跑 Live-A control；直接使用 Dual-B 日志中的 prototype 曲线作为同轨迹 control。

验收：

- fused Final 与 prototype Final 完全一致；
- Final 不低于同协议 SD-LoRA；
- AAA 不低于 SD-LoRA 超过 `0.3`；
- 同时报告 F、BWT 和最终旧类准确率；
- C100 必要主权重按正确口径为 `522,240`，约减少 85.8%，另计 bias/temperature/lambda 标量。

Forgettting 变高但最终旧类准确率不下降时，必须解释为历史峰值效应，不能隐去该指标。

## 6. P3：多 seed 主结果

P1/P2 均通过后运行。每个数据集使用现有 seeds 加至少两个未参与开发的 seeds 4/5，建议最终 `n=6`。

每个 seed 运行：

- Live-A + Dual-B；
- SD-LoRA；
- EXP-009。

Dual-B 日志中的 prototype 结果作为同轨迹 Live-A control，避免重复训练。

报告：

- 每 seed 原始结果；
- mean、std、逐 seed 配对差；
- paired t-test、95% CI、Cohen's dz；
- TOST，预注册区间 `+/-0.5`；
- Final、AAA、F、BWT 和最终旧类准确率。

主方法门槛：两个数据集 Final 均不低于 SD-LoRA，AAA 不低于 SD-LoRA 超过 `0.3`，总状态减少至少 80%。

## 7. P4：最终方法消融

先用 seed1995/1993 完成最小消融；只有关键结论不稳定时才扩展多 seed。

| 编号 | 配置 | 证明内容 |
|---|---|---|
| A0 | SD-LoRA | 原始基线 |
| A1 | Shared-A + per-task B | 共享 A 影响 |
| A2 | Aggregate-B，无历史 A 梯度 | O(1) 聚合本身 |
| A3 | Live-A Aggregate-B | 历史梯度恢复 |
| A4 | Live-A + FC | FC 头贡献 |
| A5 | Live-A + prototype | prototype 头贡献 |
| A6 | Live-A + Dual-B | 完整方法 |
| A7 | Dual-B 无温度校准 | 温度校准贡献 |
| A8 | 固定融合或 Schedule A | Schedule B 贡献 |

不得重复已经关闭的 K=2、继续增加 K、rank/damping 扫描。

## 8. P5：任务长度与额外数据集

多 seed 主结果通过后执行：

- ImageNet-R `N=5/10/20`；
- CIFAR-100 对应任务长度；
- 至少一个额外数据集，并使用多 seed；
- 统一比较 Dual-B、SD-LoRA、EXP-009；
- 所有任务长度使用同一归一化调度函数，不单独调切换点。

先用 seed1995 做结构筛查，再补配对 seeds。某个长度失败时如实报告边界，不为该长度临时调参。

## 9. P6：效率与 artifact 闭环

1. 导出最小可恢复 artifact，仅保留一份 LoRA、最终 FC、prototype 和 Dual-B 状态。
2. 恢复后验证 logits、Final、AAA 与导出前一致。
3. 测量真实训练峰值显存、时间、推理吞吐、FLOPs 和磁盘字节数。
4. 在 `N=5/10/20` 下绘制状态、显存和时间曲线，验证 Live-A 为 `O(1)`、基线为 `O(T)`。

## 10. 执行与停止纪律

1. 每一阶段先提交代码/配置，再启动训练。
2. 运行期间不修改代码或配置，不覆盖输出目录。
3. 每项结束先更新 `experiment_sd.md` 和 `experiment_note_sd.md`，再启动下一项。
4. P1 单 seed失败时禁止直接启动多 seed队列。
5. 所有数字由脚本从日志自动汇总，禁止手工选择 task 或 seed。
6. 每阶段单独 Git commit，日志和 manifest 必须记录该 commit。

执行顺序固定为：

```text
P0.5 NCCL -> P1 ImageNet-R -> P2 CIFAR-100 -> P3 多 seed
-> P4 消融 -> P5 任务长度/额外数据集 -> P6 效率
```
