# Stage B2 当前结果与下一步执行指示

更新时间：2026-08-08

## 1. 文档目的

本文档给出当前 SD-LoRA 参数压缩研究的阶段性结论，并规定下一位 Agent 的执行顺序、验收门槛和停止条件。

当前不应继续盲目搜索 rank、damping、K 或聚合函数。下一步首先修复 Dual-head 评估流程对随机数状态和后续任务训练轨迹的干扰，再在受控条件下判断方法是否真正同时保持 Final Top-1 与 AAA。

## 2. 当前四种方法的四 seed 结果

表中均为 `mean +- std`，Final 表示最后一个任务训练完成后的全类 Top-1，AAA 表示各阶段平均准确率，F 表示遗忘度，越低越好。

| 数据集 | 方法 | Final Top-1 | AAA | F |
|---|---:|---:|---:|---:|
| ImageNet-R | Live-A | 78.94 +- 0.47 | 82.45 +- 0.48 | 7.95 +- 1.10 |
| ImageNet-R | Dual-B | 78.69 +- 0.42 | 83.26 +- 0.81 | 8.02 +- 1.07 |
| ImageNet-R | EXP-009 | 79.22 +- 0.57 | 82.82 +- 0.47 | 7.80 +- 1.02 |
| ImageNet-R | SD-LoRA | 78.75 +- 0.50 | 83.42 +- 1.09 | 7.79 +- 1.57 |
| CIFAR-100 | Live-A | 87.95 +- 0.34 | 91.55 +- 0.33 | 8.75 +- 0.58 |
| CIFAR-100 | Dual-B | 87.82 +- 0.18 | 91.80 +- 0.30 | 9.11 +- 0.42 |
| CIFAR-100 | EXP-009 | 88.04 +- 0.28 | 91.62 +- 0.37 | 8.63 +- 0.53 |
| CIFAR-100 | SD-LoRA | 86.75 +- 0.39 | 91.62 +- 0.24 | 6.27 +- 1.20 |

关键配对比较：

| 比较 | Final 差值 | AAA 差值 | 结论 |
|---|---:|---:|---|
| ImageNet-R Dual-B - SD-LoRA | -0.058 | -0.155 | 数值接近，但尚未证明等效或更优 |
| ImageNet-R Dual-B - EXP-009 | -0.530 | +0.444 | AAA 修复有效，但 Final 有下降 |
| CIFAR-100 Dual-B - SD-LoRA | +1.070 | +0.175 | Final 显著更高，AAA 不低于基线 |
| CIFAR-100 Dual-B - EXP-009 | -0.220 | +0.175 | AAA 提升，Final 有小幅代价 |

补充统计：

- CIFAR-100 Dual-B 相对 SD-LoRA 的 Final 提升具有统计显著性，`p = 0.011`。
- ImageNet-R Dual-B 相对 SD-LoRA 的 Final 差异不显著，`p = 0.856`，但“不显著”不能解释为已经证明等效。
- ImageNet-R Dual-B 相对 EXP-009 的 Final 差异为 `-0.530`，`p = 0.074`。
- CIFAR-100 Dual-B 的最终旧类平均准确率约为 `87.16`，SD-LoRA 约为 `86.44`。较高 F 部分来自 Dual-B 更高的历史峰值，不能简单解释为最终旧类保持更差。

## 3. 当前可确认的结论

1. Live-A/共享参数路线已经证明可以大幅减少 LoRA 参数，并在 CIFAR-100 上提高 Final Top-1。
2. Dual-B 确实修复了 Live-A 的 AAA 缺口。其单次 ImageNet-R 实验中，相对同一训练轨迹的纯 prototype 头，融合头 AAA 提升约 `1.178` 个百分点。
3. Dual-B 还没有在严格实验控制下证明 Final 完全不受影响。当前跨次重训结果混入了随机训练轨迹变化。
4. ImageNet-R 目前只能表述为“以约 81.7% 的方法必要状态压缩，性能数值接近 SD-LoRA”，不能表述为稳定优于 SD-LoRA。
5. CIFAR-100 当前结果较强：Final 高于 SD-LoRA，AAA 持平或略高；但 F 明显更高，论文中必须同时报告 Final 旧类准确率、BWT 和完整 accuracy matrix。
6. `K=2` prototype 已得到明确负结果，应停止该方向：
   - max：`78.56 Final / 81.338 AAA / 7.551 F`
   - logsumexp：`77.51 / 81.082 / 7.674`
   - K=1 参考：`79.43 / 81.99 / 7.08`
7. K=2 artifact 一致性测试已通过，负结果不是保存/加载错误。因此不要继续扫描 K、max/logsumexp、rank 或 damping。

## 4. Final 波动的最可能原因

Schedule B 在最后阶段令融合系数 `lambda = 1`，所以同一次运行中：

```text
fused Final == prototype Final
```

当前日志也满足这一点。因此融合公式本身不会直接降低 Final。跨次重训的 Final 变化更可能来自评估和校准流程改变了全局随机数状态：

- `_compute_prototypes` 遍历一次数据；
- `_prepare_dual_head` 再调用 `_extract_current_task_features`；
- `eval_task` 分别为 fused、FC 和 prototype 头重复遍历测试集；
- DataLoader 未使用独立的 `torch.Generator`，worker 启动会消耗全局 RNG；
- 校准主要在 rank 0 执行，可能造成不同 rank 的 RNG 状态进一步分叉；
- 随后的任务使用随机数据增强，因此额外数据遍历会改变后续训练样本和增强序列。

纯 Live-A 的同 seed 重复运行本身已有约 `0.21` 个百分点波动。现阶段不能把不同重训轨迹之间的差值全部归因于 Dual-head 方法。

## 5. P0：唯一应立即执行的代码修正

本阶段只做“RNG-neutral Dual-head audit”，它属于实验控制修复，不属于新增方法组件。

### 5.1 必须修改

1. 将 fused、FC、prototype 三种 logits 合并到同一次测试 DataLoader 遍历中计算。
2. prototype 提取与温度拟合复用同一批缓存特征，避免重复启动训练集 DataLoader。
3. 为所有校准/评估 DataLoader 提供独立、固定 seed 的 `torch.Generator`。
4. 在所有 eval-only/calibration 代码前后保存并恢复以下 RNG 状态：
   - Python `random`
   - NumPy
   - Torch CPU
   - 当前进程的全部 CUDA RNG
5. 在 DDP 中由 rank 0 拟合 `lambda/tau` 后显式 broadcast，所有 rank 使用一致值。
6. 校准前后比较 LoRA A/B、分类头和 prototype 张量，要求逐元素不变。
7. 禁止使用测试集标签选择 `lambda`、`tau` 或 Schedule；Schedule B 从现在起冻结。

### 5.2 必须增加的测试

- RNG 单元测试：执行完整校准和三头评估后，四类 RNG 状态与执行前完全一致。
- 参数不变测试：校准前后所有可训练/保存参数的 `max_abs_diff == 0`。
- 单遍评估测试：同一 batch 的单遍输出与旧版三次遍历输出在容差内一致。
- DDP 冒烟测试：4 卡完成 Task 0 和 Task 1，所有 rank 的 `lambda/tau` 一致，无未同步参数。
- 轨迹测试：纯 Live control 与启用 Dual-B 的两任务运行，在每个任务训练刚结束时比较 checkpoint hash；在确定性环境下应相同。

### 5.3 提交要求

P0 修复和测试必须形成一个独立 Git commit。提交前运行现有测试，并把命令、结果、commit hash 写入 `experiment_note_sd.md`。

## 6. P1：受控重跑顺序

必须按以下顺序执行，前一步不通过时不得启动后一步：

1. 4 GPU、两任务 smoke test：纯 Live control 与 Dual-B 使用相同 seed、配置和数据顺序。
2. 检查任务训练结束时的参数 hash/RNG 状态；确认 Dual-head 只改变评估结果，不改变下一任务训练轨迹。
3. 仅重跑 ImageNet-R seed 1995 全流程，验证同一运行中的 `fused Final == prototype Final`，容差 `1e-6`。
4. smoke 和 seed 1995 均通过后，再运行已有 seeds 1/2/3。
5. 若形成论文最终结果，新增未参与开发选择的 seeds 4/5；seed 1995 已是开发 seed，不能再视为完全独立验证。

每次实验完成后，先更新 `experiment_sd.md` 和 `experiment_note_sd.md`、提交 Git，再启动下一项。不得并行混跑不同实现版本。

## 7. P1 验收门槛

### 实验控制门槛

- eval/calibration 前后 RNG 状态一致；
- eval/calibration 前后模型参数完全一致；
- 同一轨迹中 fused 与 prototype 的 Final 差值不超过 `1e-6`；
- 重新加载最终 artifact 后 logits、Final、AAA 与导出前一致。

### 方法门槛

ImageNet-R：

- Final 均值不低于同 seed SD-LoRA；
- AAA 不低于 SD-LoRA `0.3` 个百分点以上；
- 方法必要的可恢复状态相对 SD-LoRA 减少至少 80%。

CIFAR-100：

- Final 均值不低于 SD-LoRA；
- AAA 不低于 SD-LoRA `0.3` 个百分点以上；
- 同时报告 F、BWT、最终旧类准确率和 accuracy matrix，不允许通过压低早期准确率来人为降低 F。

若 RNG 修复后的 Dual-B 仍未通过 ImageNet-R 门槛，则停止把 Live-A/Dual-B 作为主方法继续微调；保留为高压缩消融或负结果。不得以继续扫描 K、rank、damping 来绕过停止条件。

## 8. P2：artifact 与参数统计修正

当前文档曾把 CIFAR-100 的 FC 参数误记为 `153,600`。实际最终分类头 `CLs_weight9.pt` 的形状为 `[100, 768]`，即 `76,800` 个权重。

在仅保留方法必要最终状态时：

| 数据集 | LoRA | FC | Prototype | 主权重合计 | 相对 3,686,400 的减少 |
|---|---:|---:|---:|---:|---:|
| CIFAR-100 | 368,640 | 76,800 | 76,800 | 522,240 | 约 85.8% |
| ImageNet-R | 368,640 | 153,600 | 153,600 | 675,840 | 约 81.7% |

以上还需在最终报告中计入 bias、temperature、lambda 等全部标量。不能只报告理论参数量，还必须报告实际 artifact 大小。

P2 应单独提交：

1. 增加 `export_final_artifact`，仅保留最新 FC、prototype、Dual-head 状态和一种 LoRA 表示。
2. 不直接删除旧 checkpoint；先导出到新目录。
3. 从精简 artifact 重建模型，验证路由/分类 logits 与原模型一致。
4. 输出逐项参数量、总参数量和磁盘字节数，作为论文效率表的来源。

## 9. 后续完整实验的开启条件

只有 P0/P1 门槛通过后，才进入完整论文实验：

- ImageNet-R `N = 5/10/20`；
- CIFAR-100 对应多任务长度；
- 至少 5 seeds 的均值、标准差、配对检验和等效性检验；
- 训练时间、推理时间、峰值显存、实际 checkpoint 大小；
- SD-LoRA、EXP-009、Live-A、Dual-B 的同协议对比；
- 完整消融：prototype 头、FC 头、固定融合、Schedule B、参数压缩组件。

若 P1 未通过，Stage C/D 只能在用户明确授权后作为消融或负结果运行，不能据此宣称主目标已完成。

## 10. 永久停止的重复方向

下一位 Agent 在开始实验前必须读取 `experiment_sd.md`，以下方向不得重复：

- K=2 prototype 及其 max/logsumexp 聚合；
- 继续增加 K；
- 未定位问题前继续扫描 rank、damping；
- Protected Union、LRPT 的重复组合；
- 与当前主问题无关的 router/replay 扩展；
- 使用测试集结果继续调 Schedule B。

## 11. 给下一位 Agent 的立即指示

1. 先确认 GPU 上无遗留训练进程，并记录当前 commit。
2. 阅读 `plan_sd.md`、`experiment_sd.md`、`experiment_note_sd.md`、`goal_live_a_sd.md` 和本文档。
3. 只实施 P0 RNG-neutral 修复，不添加新算法组件。
4. 完成单元测试和 4 卡两任务 smoke test，单独提交 Git。
5. 通过后执行 P1；失败则先定位实验控制问题，不启动全量训练。
6. P1 通过后执行 P2 artifact 导出与恢复验证，并再次单独提交。
7. 所有数字必须来自日志或脚本自动汇总，禁止手工选择有利 task/seed。

当前最重要的问题不是再寻找一个新模块，而是确认 Dual-B 的 AAA 收益是否能在完全相同的训练轨迹上获得，同时让 Final 保持不变。只有完成这一点，后续多任务长度和论文级实验才有可信基础。
