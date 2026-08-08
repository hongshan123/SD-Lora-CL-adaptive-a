# Live-A / Dual-B CCF-A 实验补全指引

更新时间：2026-08-09

## 1. 目标

将当前 Live-A Aggregate-B + Dual-B 从“有潜力的参数压缩结果”补全为可信、可复现的论文证据。

主张应限定为：

> 在 rehearsal-free CIL 中，以 `O(1)` 持久状态和约 82%--86% 的总状态压缩，达到 SD-LoRA 级准确率。

在完成本文档 P0 前，不增加新算法组件，不扫描 K、rank、damping 或融合调度。

## 2. 当前有效结果

| 数据集 | 方法 | Final | AAA | Forgetting |
|---|---|---:|---:|---:|
| ImageNet-R | Dual-B | 78.69 | 83.26 | 8.02 |
| ImageNet-R | SD-LoRA | 78.75 | 83.42 | 7.79 |
| CIFAR-100 | Dual-B | 87.82 | 91.80 | 9.11 |
| CIFAR-100 | SD-LoRA | 86.75 | 91.62 | 6.27 |

当前 `D2` 目录及连续复用该目录得到的 smoke 均为污染结果，不得写入论文或实验汇总。

## 3. P0：先修复实验可信度

### 必须修改

1. 每次训练使用全新输出目录；目录中已有 artifact 时直接报错。
2. 只有显式设置 `resume=true` 才允许加载 `sa_state.pt`。
3. 设置完整随机种子：Python、NumPy、Torch CPU、Torch CUDA 和 DataLoader generator。
4. 使用严格确定性模式：

```python
torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
```

5. 校准和评估必须保持 RNG、模型参数及 buffer 不变。
6. 不要直接比较 `torch.save` 文件字节；按键名排序后，对 tensor 的名称、shape、dtype 和连续 CPU 数据计算 canonical hash。
7. 禁止两个进程写入同一 `filepath`；启动脚本增加运行锁或非空目录检查。

### 验收顺序

1. 两次独立 control，使用不同空目录。
2. 比较每个 task 训练结束时的 canonical tensor hash，要求完全一致。
3. 再运行一个 Dual-B，与 control 比较训练轨迹 hash。
4. Dual-B 的校准前后要求 `max_abs_diff == 0`，最终阶段要求 `fused Final == prototype Final`，容差 `1e-6`。
5. 4 卡两任务 smoke 全部通过后，才允许完整重跑。

P0 必须形成独立 Git commit，并把命令、目录、日志、hash 和测试结果写入 `experiment_note_sd.md`。

## 4. P1：冻结方法并补公平对比

P0 通过后冻结 Live-A + Dual-B，不再根据测试集修改调度和超参数。

统一以下条件：类序、seed、优化器、学习率、epoch、全局 batch size、预训练权重和代码 commit。

必须运行：

- SD-LoRA；
- EXP-009；
- Live-A；
- Live-A + Dual-B；
- 至少三个强相关外部基线，如 InfLoRA、CL-LoRA、LoRA-DRS、DGS。

ImageNet-R 和 CIFAR-100 均至少使用 5 个配对 seeds。报告每个 seed、均值、标准差、配对检验、95% CI 和 TOST，禁止只报告最好 seed。

## 5. P2：最终方法消融

使用相同 seed 和协议完成以下最小消融矩阵：

| 编号 | 配置 | 验证内容 |
|---|---|---|
| A0 | SD-LoRA | 原始基线 |
| A1 | Shared-A + per-task B | 共享 A 的影响 |
| A2 | Aggregate-B，不恢复历史 A 梯度 | O(1) 聚合本身 |
| A3 | Live-A Aggregate-B | 历史 A 梯度恢复 |
| A4 | Live-A + FC | FC 分类头贡献 |
| A5 | Live-A + prototype | prototype 分类头贡献 |
| A6 | Live-A + Dual-B | 完整方法 |
| A7 | Dual-B 无温度校准或固定融合 | 校准与调度贡献 |

主表统一报告 Final、AAA、Forgetting、BWT、最终旧类准确率、总参数和实际 artifact 大小。

## 6. P3：泛化与任务长度

完成以下实验：

- ImageNet-R `N=5/10/20`；
- CIFAR-100 对应任务长度；
- 至少增加一个数据集，并使用多 seed；
- 所有任务长度使用同一归一化 Dual-B 调度，不按测试结果单独调整。

若方法在某个任务长度下明显失败，应报告失败边界和原因，不继续为该长度单独调参。

## 7. P4：效率与 artifact 闭环

1. 导出最小可恢复 artifact，只保留一份 LoRA 状态、最终 FC、prototype 和 Dual-B 状态。
2. 从精简 artifact 恢复模型，验证 logits 和全部指标与导出前一致。
3. 报告 LoRA-only、分类器/prototype、总持久参数和磁盘字节数。
4. 在相同硬件上测量真实训练峰值显存、每 epoch 时间、总训练时间、推理吞吐和 FLOPs。
5. 绘制状态量、显存和时间随任务数变化的曲线，验证 Live-A 为 `O(1)`，基线为 `O(T)`。

## 8. 继续与停止条件

只有同时满足以下条件，才进入论文最终整理：

- P0 确定性与轨迹测试通过；
- 两个主数据集 Final 不低于 SD-LoRA，或通过预注册等效性检验；
- AAA 不低于 SD-LoRA 超过 0.3 个百分点；
- 总可恢复状态减少至少 80%；
- 多任务长度和新增数据集没有结构性失效；
- 强基线、消融和真实效率实验完整。

若 ImageNet-R 在干净的 5-seed 实验中仍未达到门槛，停止继续堆叠模块，将方法诚实定位为“高压缩、近似保持准确率”的 Pareto 方法。

## 9. 执行纪律

1. 每次实验前读取 `experiment_sd.md`，避免重复实验。
2. 每次代码修改单独 Git commit；实验日志记录对应 commit hash。
3. 不覆盖旧日志和 artifact，不在训练运行中修改其加载代码或配置。
4. 每轮结束先分析并更新 `experiment_sd.md`、`experiment_note_sd.md`，再启动下一轮。
5. 当前禁止重复 K=2、增加 K、rank/damping 扫描及已判定失败的组合。

执行优先级固定为：`P0 可复现性 -> P1 强基线 -> P2 消融 -> P3 泛化 -> P4 效率`。
