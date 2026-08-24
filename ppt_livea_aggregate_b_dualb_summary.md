# Live-A Aggregate-B + Dual-B：方法与实验结果总结

> 用途：制作阶段汇报或论文工作汇报 PPT。  
> 结果封板日期：2026-08-15。  
> 冻结方法：`Live-A Aggregate-B + Dual-B`。  
> 核心定位：无回放类增量学习中的固定状态 LoRA 聚合，以显著降低持续增长的适配器存储，同时保持 SD-LoRA 级最终准确率。

## 1. 一页结论

现有 SD-LoRA 为每个增量任务保留一套 LoRA，历史知识保存充分，但 LoRA 参数、训练显存和前向计算均随任务数线性增长。本文方法在 ViT-B/16 的每个 Transformer block 的 Q/V 分支中使用一套持续更新的共享 down-projection `A`，将每个任务训练得到的 up-projection `B_t` 在线折叠到固定大小的历史聚合矩阵 `G`，任务结束后丢弃 `B_t`。因此，无论经历多少任务，LoRA 状态始终只有一套 rank-10 `A+G`。

分类端采用固定调度的 Dual-B 头：早期任务融合线性 FC 头和类别 prototype 头，后期逐步切换到 prototype 头；最终任务严格令 `lambda=1`，因此 Final Top1 完全来自 prototype 头，不通过测试集选择最佳头。

主要结果：

- ImageNet-R 严格 5-seed：相对同协议 SD-LoRA，Final `+0.73`、AAA `+0.10`、Forgetting `-0.11`。
- CIFAR-100 严格 5-seed：Final `+0.60`，但 AAA `-0.39`、Forgetting `+2.52`，存在稳定性代价。
- CUB-200 同协议 3-seed：Final `+4.01`、AAA `+2.70`，但 Forgetting `+5.80`。
- T=10 时 LoRA-only 参数从 SD-LoRA 的 `3,686,400` 降到 `368,640`，减少 `90%`。
- T=5/10/20 时当前方法 LoRA 参数始终为 `368,640`，SD-LoRA 增长为 `1.84M/3.69M/7.37M`。
- 训练峰值显存保持约 `3.47 GiB`，SD-LoRA 为 `5.96/8.19/12.63 GiB`。

一句话结论：**该方法形成了明显的 Final-存储-扩展性 Pareto 优势，但不是在所有数据集和所有持续学习指标上全面超越 SD-LoRA。**

## 2. 问题背景

### 2.1 研究任务

- 场景：rehearsal-free class-incremental learning，训练时不保存历史样本。
- 测试：每个阶段在所有已见类别上进行全局分类，不提供 task ID。
- 主干：ImageNet 预训练 ViT-B/16，主干参数冻结，仅训练 LoRA 和分类头。
- 目标：避免每到一个任务就永久新增一套 LoRA，同时尽可能保持最终准确率和平均增量准确率。

### 2.2 原始 SD-LoRA 的扩展性问题

设每个任务保存一套 rank-`r` LoRA：

```text
Delta W_t = s_t B_t A_t
```

经历 `T` 个任务后，需要保存并参与前向的 LoRA bank 为：

```text
{A_1, B_1, ..., A_T, B_T}
```

因此：

- 持久 LoRA 状态为 `O(T)`；
- 前向中的 LoRA 分支数量为 `O(T)`；
- 训练显存和任务后期耗时随 `T` 增长；
- 长任务序列下可扩展性较差。

## 3. 完整方法

### 3.1 LoRA 注入位置

- Backbone：ViT-B/16，共 12 个 Transformer blocks。
- 在每个 block 的 attention Q 和 V 投影中注入 LoRA。
- 每个 Q/V 分支使用 rank `r=10`。
- K 分支及预训练 backbone 保持冻结。
- 共享 `A` 采用固定正交矩阵初始化，后续任务继续训练；这里是正交初始化，不是持续施加正交正则。

### 3.2 Live-A：持续更新的共享 down-projection

任务 `t` 到来时，仅保留：

```text
A_t       当前共享 down-projection，可训练
G_{t-1}   历史任务聚合后的 up-projection，冻结
B_t       当前任务新建的 up-projection，可训练，零初始化
s_t       当前任务可学习 scale
```

历史和当前任务分支共同进入 Q/V 前向：

```text
Delta h_t(x)
  = G_{t-1} A_t x / ||A_t||_F
  + s_t B_t A_t x
```

其中：

- 第一项表示已经折叠的历史知识；
- 第二项表示当前任务的新知识；
- 历史项与当前项都依赖 `A_t`，因此共享 A 同时接收历史分支和当前分支的梯度；
- `G_{t-1}` 在当前任务训练期间不更新，避免再次展开历史 LoRA bank。

这就是“Live-A”的含义：历史知识不是通过冻结的旧 A 前向，而是在当前持续更新的共享 A 坐标系中被调用。

### 3.3 Aggregate-B：任务结束后的在线折叠

任务 `t` 训练完成后，将当前 `B_t` 归一化并累加到固定大小的历史状态：

```text
G_t = G_{t-1} + s_t B_t / (||B_t||_F + eps)
```

随后：

1. 保存 `G_t` 和当前共享 `A_t`；
2. 丢弃任务专属 `B_t`；
3. 下一个任务创建新的零初始化 `B_{t+1}`；
4. `G` 的形状始终不变，不随任务数增加。

最终可进一步得到单一 merged LoRA：

```text
B_merged = G_T / (||A_T||_F + eps)
Delta W_merged = B_merged A_T
```

因此最终推理只需要每层一套共享 rank-10 LoRA，不需要 task ID、router 或逐任务专家前向。

### 3.4 类别 prototype

每个任务训练结束后：

1. 使用该任务训练集和确定性的 test transform 提取特征；
2. 对每类样本特征先做 L2 归一化；
3. 计算每类均值并再次归一化，得到一个类别 prototype；
4. 只保存 prototype，不保存历史图像或特征样本。

prototype 分类 logits 为特征与所有类别 prototype 的余弦相似度。

### 3.5 Dual-B 分类头

训练阶段使用扩展 FC 头；任务结束后同时具备：

- `z_fc`：温度校准后的 FC logits；
- `z_proto`：温度校准后的 prototype logits。

融合输出为：

```text
z = (1 - lambda_t) * z_fc / tau_fc
    + lambda_t * z_proto / tau_proto
```

温度只使用当前任务训练特征拟合，不访问测试标签。Schedule B 定义为：

```text
lambda_t = clip((t/(T-1) - 1/9) / (4/9), 0, 1)
```

当 `T=10` 时：

| Task | 0 | 1 | 2 | 3 | 4 | 5-9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lambda | 0 | 0 | 0.25 | 0.50 | 0.75 | 1.00 |

作用：

- 早期类别少时利用 FC 头较强的判别能力；
- 中期平滑切换到更适合无回放场景的 prototype 头；
- 最终 `lambda=1`，Final 与 prototype logits 完全一致；
- Dual-B 主要改善训练过程中的 AAA，不人为抬高最终阶段结果。

### 3.6 完整训练流程

```text
输入任务 D_t
  |
  |-- 加载冻结 ViT、A_{t-1}、G_{t-1} 和历史 prototypes
  |-- 新建 B_t=0 和当前分类头类别
  |-- 仅用 D_t 训练 A_t、B_t、s_t 和当前分类头
  |-- 计算并保存当前类别 prototypes
  |-- 训练数据校准 tau_fc、tau_proto，按固定 Schedule B 设置 lambda_t
  |-- 评估所有已见类别
  |-- 折叠 G_t = G_{t-1} + s_t B_t / ||B_t||
  |-- 丢弃 B_t，进入下一任务
```

### 3.7 最终方法明确不包含的组件

以下均属于探索或负结果，不是最终方法：

- task prototype LoRA router；
- HBD 历史分支蒸馏；
- K=2 多 prototype；
- LRPT prototype transport；
- operator-stability loss；
- K-CMS 多专家 merge；
- 回放样本、生成回放或 task-ID oracle。

## 4. 实验协议与指标

### 4.1 数据集和训练设置

| 数据集 | 类别数 | 任务数 | 每任务类别 | 主统计 seeds | 优化器 | LR | Epoch/task | Batch |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| ImageNet-R | 200 | 10 | 20 | 1-5 | SGD, constant | 0.01 | 20 | 32 |
| CIFAR-100 | 100 | 10 | 10 | 1-5 | SGD, cosine | 0.008 | 20 | 32 |
| CUB-200-2011 | 200 | 10 | 20 | 1-3 | SGD, constant | 0.01 | 20 | 32 |

共同设置：

- `shuffle=True`，不同 seed 对应不同类别顺序；
- 4 x RTX 3090，NCCL DDP；
- ViT-B/16 使用 `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k` 预训练权重；
- 无 rehearsal memory；
- 所有正式运行使用确定性训练和独立输出目录。

### 4.2 指标

- **Final Top1**：完成最后任务后，在全部类别上的 Top-1 准确率。
- **AAA**：每个增量阶段全局 Top-1 的平均值，反映整个持续学习过程。
- **Forgetting**：历史任务峰值准确率到最终准确率的平均下降，越低越好。
- **Persistent state**：训练结束后恢复模型所需的 LoRA、分类头和 prototype 参数。

## 5. 多数据集主结果

### 5.1 ImageNet-R：严格 5-seed

| 方法 | Final Top1 | AAA | Forgetting |
| --- | ---: | ---: | ---: |
| 当前方法 | **79.07±0.23** | **83.06±0.52** | **7.68±1.03** |
| SD-LoRA | 78.34±0.48 | 82.96±0.87 | 7.79±1.04 |
| EXP-009 task-B bank | **79.16±0.32** | 82.63±0.36 | 7.69±0.93 |

当前方法相对 SD-LoRA：

| 指标 | 配对差值 | p-value | 解释 |
| --- | ---: | ---: | --- |
| Final | +0.730 | 0.0784 | 稳定提升趋势，未达到 0.05 显著性 |
| AAA | +0.103 | - | 基本持平 |
| Forgetting | -0.105 | - | 基本持平，方向略优 |

结论：ImageNet-R 上，在总持久状态减少 `81.7%` 的情况下，Final、AAA 和 Forgetting 均达到或略优于同协议 SD-LoRA；相对 O(T) 的 EXP-009，Final/F 基本相当，AAA 更高。

### 5.2 CIFAR-100：严格 5-seed

| 方法 | Final Top1 | AAA | Forgetting |
| --- | ---: | ---: | ---: |
| 当前方法 | **87.41±0.32** | 91.06±0.64 | 9.09±0.62 |
| SD-LoRA | 86.81±0.56 | **91.45±0.63** | **6.57±0.71** |
| EXP-009 task-B bank | **87.58±0.28** | **91.12±0.54** | 9.19±0.47 |

当前方法相对 SD-LoRA：

| 指标 | 配对差值 | p-value | 解释 |
| --- | ---: | ---: | --- |
| Final | **+0.602** | **0.0411** | Final 显著提高 |
| AAA | **-0.385** | **0.0057** | 过程平均性能显著下降 |
| Forgetting | **+2.522** | **0.0002** | 历史遗忘显著增加 |

结论：CIFAR-100 上当前方法更偏向最终塑性。它在总状态减少 `85.8%` 时提高 Final，但没有保持 SD-LoRA 的历史稳定性。相对 EXP-009 基本相当，说明 O(1) 聚合可以压缩 task-B bank，而主要代价来自两者共有的共享表示漂移。

### 5.3 CUB-200：同协议 3-seed

| 方法 | Final Top1 | AAA | Forgetting | LoRA 状态 |
| --- | ---: | ---: | ---: | ---: |
| 当前方法 | **75.27±3.12** | **84.37±1.34** | 16.77±0.43 | **368,640, O(1)** |
| EXP-009 task-B bank | **75.37±2.90** | **85.01±1.79** | 17.15±0.11 | O(T) |
| SD-LoRA | 71.26±1.47 | 81.66±0.90 | **10.97±2.17** | 3,686,400, O(T) |

配对均值差：

```text
当前方法 - EXP-009 = Final -0.10 / AAA -0.64 / F -0.38
当前方法 - SD-LoRA = Final +4.01 / AAA +2.70 / F +5.80
```

结论：

- 当前方法以 O(1) LoRA 状态取得与 EXP-009 接近的 Final/AAA；
- 相对同协议 SD-LoRA，Final 和 AAA 均值更高，但遗忘明显更大；
- 只有 3 seeds，不能使用“统计显著高于”或“正式统计等价”的措辞；
- seed2 Final 为 `71.80`，说明 CUB 细粒度类别下存在较强类别顺序敏感性。

### 5.4 三个数据集汇总

| 数据集 | 当前方法相对同协议 SD-LoRA | 参数/状态结论 | 主要代价 |
| --- | --- | --- | --- |
| ImageNet-R | Final +0.73，AAA +0.10，F -0.11 | 总状态 -81.7% | Final 提升未达 p<0.05 |
| CIFAR-100 | Final +0.60，AAA -0.39，F +2.52 | 总状态 -85.8% | AAA 和遗忘显著变差 |
| CUB-200 | Final +4.01，AAA +2.70，F +5.80 | 总状态约 -81.7% | 3-seed 方差大、遗忘增加 |

## 6. 任务长度与可扩展性

> 以下 T=5/10/20 使用开发 seed：CIFAR-100 seed1993、ImageNet-R seed1995。它们适合展示趋势，不能替代多 seed 显著性结论。

### 6.1 Final Top1 随任务数变化

| 数据集 | 方法 | T=5 | T=10 | T=20 |
| --- | --- | ---: | ---: | ---: |
| CIFAR-100 | 当前方法 | **89.14** | **88.32** | **85.37** |
|  | SD-LoRA | 88.55 | 87.00 | 83.61 |
|  | EXP-009 | 89.11 | **88.37** | **85.55** |
|  | rank1 + Dual-B | 88.25 | 87.73 | 83.45 |
| ImageNet-R | 当前方法 | **79.86** | **78.84** | **78.23** |
|  | SD-LoRA | 79.48 | 77.94 | 76.54 |
|  | EXP-009 | 79.84 | 78.74 | **78.23** |
|  | rank1 + Dual-B | 77.18 | 76.28 | 75.14 |

当前方法相对 SD-LoRA 的 Final 优势：

```text
CIFAR-100: +0.59 -> +1.32 -> +1.76
ImageNet-R: +0.38 -> +0.90 -> +1.69
```

主要观察：

- 当前方法 LoRA 状态不随 T 增长，Final 优势在开发 seed 上随任务数扩大；
- 相对 EXP-009 的准确率基本相当，但 EXP-009 仍保存 O(T) task-B bank；
- T=20 时当前方法的 LoRA 参数只有 rank1 bank 的一半，但 Final 高出 C100 `1.92`、INR `3.09`；
- CIFAR-100 T=20 Forgetting 仍为 `10.59`，高于 SD-LoRA 的 `7.22`，扩展性并未自动解决历史遗忘。

## 7. 参数量与效率

### 7.1 LoRA 参数随任务数变化

| 方法 | T=5 | T=10 | T=20 | 复杂度 |
| --- | ---: | ---: | ---: | --- |
| 当前方法 | **368,640** | **368,640** | **368,640** | **O(1)** |
| SD-LoRA | 1,843,200 | 3,686,400 | 7,372,800 | O(T) |
| EXP-009 | 1,105,925 | 2,027,530 | 3,870,740 | O(T) |
| rank1 SD-LoRA | 184,320 | 368,640 | 737,280 | O(T) |

当前方法相对 SD-LoRA 的 LoRA-only 压缩率：

```text
T=5:  80%
T=10: 90%
T=20: 95%
```

### 7.2 完整持久参数

| 数据集 | LoRA | FC | Prototypes | 合计 | 相对 SD-LoRA LoRA bank |
| --- | ---: | ---: | ---: | ---: | ---: |
| CIFAR-100 | 368,640 | 76,800 | 76,800 | **522,240** | **-85.8%** |
| ImageNet-R | 368,640 | 153,600 | 153,600 | **675,840** | **-81.7%** |
| CUB-200 | 368,640 | 153,600 | 153,600 | **675,840** | **-81.7%** |

说明：FC 和 prototype 随类别数增长，但核心 LoRA 状态不随任务数增长。LoRA artifact 实测约 `2.78 MB`；包含恢复所需全部文件的最小包为 C100 `4.37 MB`、CUB `5.01 MB`。

### 7.3 训练峰值显存

| 方法 | T=5 | T=10 | T=20 |
| --- | ---: | ---: | ---: |
| 当前方法 | **3,468.6 MiB** | **3,468.6 MiB** | **3,468.6 MiB** |
| SD-LoRA | 5,963.9 MiB | 8,186.7 MiB | 12,632.1 MiB |

### 7.4 训练时间增长

ImageNet-R T=10，每任务墙钟时间包含全类评估：

| 方法 | Task0 | Task9 | 增长倍数 |
| --- | ---: | ---: | ---: |
| 当前方法 | 204 s | 336 s | **1.65x** |
| SD-LoRA | 177 s | 535 s | 3.02x |
| EXP-009 | 205 s | 488 s | 2.38x |
| rank1 | 221 s | 531 s | 2.40x |

### 7.5 推理和恢复

- merged 推理 FLOPs：`1.129e12` per batch-32 forward；
- 空闲 RTX 3090 吞吐：`515.3 images/s`；
- 推理峰值显存：`578.4 MiB`；
- C100、CUB 最小 artifact 恢复后 features/FC/prototype/fused logits 的最大绝对误差均为 `0`；
- 恢复后的 Final 与训练日志完全一致。

## 8. 消融与机制分析

### 8.1 冻结共享 A

冻结 A 相对完整方法的 3-seed 配对差：

| 数据集 | Delta Final | Delta AAA | Delta Forgetting |
| --- | ---: | ---: | ---: |
| CIFAR-100 | +0.49 | +0.43 | -1.45 |
| ImageNet-R | -0.59 | -0.13 | +0.06 |

解释：

- Live-A 更新在 ImageNet-R 上提供塑性和约 `0.5` Final；
- 在 CIFAR-100 上，共享 A 更新反而是历史干扰源；
- 共享 A 的作用具有数据集依赖性，这是当前方法稳定性-塑性权衡的直接证据。

### 8.2 rank1 同预算基线

T=10 时，rank1 task bank 和当前方法均有 `368,640` 个 LoRA 参数。

rank1 相对完整方法的 3-seed配对差：

| 数据集 | Delta Final | Delta AAA | Delta Forgetting |
| --- | ---: | ---: | ---: |
| CIFAR-100 | +0.26 | +0.77 | -2.99 |
| ImageNet-R | -2.78 | -1.07 | +0.90 |

解释：

- ImageNet-R 支持将固定预算集中为 rank-10 聚合表示；
- CIFAR-100 更适合逐任务 rank1 bank 的隔离和历史保持；
- 不能声称当前容量分配在所有数据集上全面更优。

### 8.3 Dual-B 的贡献

严格 5-seed 的 FC/prototype/fused AAA：

| 数据集 | FC | Prototype | Fused Dual-B | Fused - Prototype |
| --- | ---: | ---: | ---: | ---: |
| ImageNet-R | 83.020 | 82.646 | **83.064** | **+0.42** |
| CIFAR-100 | 90.467 | 91.021 | **91.063** | +0.04 |

解释：Dual-B 对 ImageNet-R 早期阶段有效，对 CIFAR-100 的提升很小；CIFAR-100 的主要瓶颈不在头部调度。

### 8.4 Prototype 坐标漂移诊断

使用最终模型和全部训练类离线重算 prototype，仅作为 oracle 诊断：

| 数据集 | Delta Final | Delta AAA | Delta Forgetting | 旧类提升 |
| --- | ---: | ---: | ---: | ---: |
| ImageNet-R | +1.32 | +0.71 | -1.77 | +1.65 |
| CIFAR-100 | +1.11 | +0.81 | -1.65 | +1.41 |

保存 prototype 与最终重算 prototype 的平均余弦相似度：INR `0.9478`、C100 `0.9622`。这说明共享 A/G 更新后，旧 prototype 留在历史特征坐标系，是遗忘的重要来源。

### 8.5 已验证但未进入最终方法的方向

- HBD：C100 Forgetting 从 `8.23` 降到 `6.16`，但 Final 从 `88.32` 降到 `87.82`，未过预注册门槛。
- K=2 prototype：ImageNet-R 的 Final/AAA 均低于 K=1，关闭。
- LRPT、operator stability 等探索没有形成跨数据集稳定收益，均不进入最终方法。

## 9. 外部方法位置

外部发布值使用各论文原协议，不能与本文冻结协议做配对显著性比较。

| 方法 | CIFAR-100 发布 Final / AAA | ImageNet-R 发布 Final / AAA | 说明 |
| --- | ---: | ---: | --- |
| InfLoRA | 86.51 / 91.70 | 75.65 / 80.82 | CVPR 2024 |
| CL-LoRA | T20: 85.32 / 91.02 | T40: 74.51 / 81.58 | CVPR 2025，任务数不同 |
| LoRA-DRS | 89.14 / 92.55 | 74.74 / 81.16 | CVPR 2025，准确率强但 LoRA 状态增长 |
| SD-LoRA | 88.01 / 92.54 | 77.34 / 82.04 | ICLR 2025 |

本机官方实现单 seed：InfLoRA `84.75/90.26`，LoRA-DRS `89.73/93.02`；CL-LoRA 因官方代码与本机 Torch 版本兼容问题未成功复现。外部方法结果应作为方法空间定位，不宣称当前方法在绝对准确率上全面 SOTA。

## 10. 方法优势、局限与论文口径

### 10.1 可以明确声称

1. 将 SD-LoRA/EXP-009 的任务 LoRA bank 从 `O(T)` 压缩为固定 `O(1)` LoRA 状态。
2. 在 ImageNet-R、CIFAR-100 严格同协议 5-seed 上，Final 均不低于 SD-LoRA。
3. 在 CUB 同协议 3-seed 上，Final/AAA 均值高于 SD-LoRA，并接近 EXP-009。
4. 实测 LoRA 参数、磁盘状态和训练显存不随任务数增长。
5. 开发 seed 上，T 越长，相对 SD-LoRA 的 Final 优势越大。
6. 完整 artifact 可无误差恢复，结果具有确定性和可复现性。

### 10.2 不能声称

1. 不能声称所有指标全面优于 SD-LoRA。
2. 不能声称 CIFAR-100/CUB 的遗忘得到解决。
3. 不能声称绝对准确率超过所有外部 SOTA；LoRA-DRS 在 C100 更强。
4. 不能将单开发 seed 的 T=5/T=20 趋势描述成多 seed 显著结论。
5. 不能把 CUB 3-seed 的均值差写成“统计显著”或正式“统计等价”。
6. 不能把 HBD、router、LRPT 或 K=2 prototype 写入最终方法。

### 10.3 当前最准确的论文/PPT主张

> Live-A Aggregate-B 将持续增长的任务 LoRA bank 在线折叠到一个固定 rank 的共享适配器，并通过 Dual-B 改善增量过程中的分类头生命周期。在 rehearsal-free CIL 中，它以 81.7%-85.8% 的总持久状态压缩获得 SD-LoRA 级或更高 Final，并显著改善长任务下的存储和显存扩展性；代价是细粒度或历史稳定性敏感数据集上的遗忘增加。

## 11. 推荐 PPT 结构

### 第 1 页：标题

**Fixed-State Continual LoRA via Live-A Aggregate-B**  
副标题：Rehearsal-free Class-Incremental Learning with O(1) Adapter State

### 第 2 页：研究问题

- SD-LoRA 精度较好，但每任务新增 LoRA；
- 参数、显存和前向分支随任务数线性增长；
- 目标是在不回放历史样本的情况下固定 LoRA 状态。

### 第 3 页：核心思路图

- 左侧：SD-LoRA 的 `{A_t,B_t}` bank 持续增长；
- 右侧：共享 live A + 当前 B_t + 固定历史 G；
- 任务结束箭头：`G_t = G_{t-1} + s_t B_t/||B_t||`，随后删除 B_t。

### 第 4 页：训练与推理流程

- 训练：历史 G 分支与当前 B 分支共同更新 A；
- 保存：只保留 A、G、prototype 和 FC；
- 推理：单 merged LoRA，无 task ID、无 router。

### 第 5 页：Dual-B 分类头

- 展示 FC/prototype logits 融合公式；
- 展示 T=10 的 lambda：`0,0,0.25,0.5,0.75,1,...`；
- 强调最终 lambda=1，不通过测试选择头。

### 第 6 页：ImageNet-R 主结果

- 放置 5-seed 表；
- 高亮 Final `+0.73`、AAA 持平、总状态 `-81.7%`。

### 第 7 页：CIFAR-100 主结果

- 高亮 Final `+0.60`、总状态 `-85.8%`；
- 同时用醒目颜色标注 AAA `-0.39`、Forgetting `+2.52`。

### 第 8 页：CUB-200 结果

- 当前方法与 EXP-009 准确率接近，但状态从 O(T) 降到 O(1)；
- 相对 SD-LoRA Final/AAA 均值更高，但遗忘更大；
- 标注 3-seed 和类别顺序敏感性。

### 第 9 页：任务长度扩展性

- 横轴 T=5/10/20，纵轴 Final；
- 分别绘制 C100、INR 当前方法与 SD-LoRA；
- 辅助标注 LoRA 参数：当前恒定，SD-LoRA 线性增长。

### 第 10 页：效率

- 参数量折线：`0.37M` 恒定 vs `1.84/3.69/7.37M`；
- 显存柱状图：`3.47 GiB` 恒定 vs `5.96/8.19/12.63 GiB`；
- 时间增长：`1.65x` vs `3.02x`。

### 第 11 页：消融和机制

- 冻结 A：INR 塑性下降，C100 稳定性改善；
- rank1 同预算：数据集依赖；
- oracle prototype 重算：确认 prototype 坐标漂移。

### 第 12 页：结论与局限

- 贡献：固定状态、Final-参数 Pareto、长任务效率；
- 局限：C100/CUB 遗忘、CUB seed 方差、任务长度多 seed 尚缺；
- 下一步：无历史样本的 prototype transport 或历史坐标稳定机制。

## 12. 建议制作的图表

1. **方法结构图**：历史 `G`、live `A`、fresh `B_t` 三条路径。
2. **状态复杂度图**：横轴任务数，当前方法水平线，SD-LoRA/EXP-009/rank1 斜线。
3. **Final vs 参数散点图**：三个数据集分别展示 Pareto 位置。
4. **任务长度折线图**：T=5/10/20 的 Final 和参数量双轴图。
5. **显存柱状图**：当前方法与 SD-LoRA 在 T=5/10/20 的峰值显存。
6. **旧类/新类示意图**：解释 CUB/C100 中新类塑性与历史遗忘的权衡。

## 13. 数据来源索引

- 严格 5-seed 主结果：`p3_strict_n5_summary.md`
- 统计原始输出：`p3_strict_n5_stats_output.txt`
- 机制诊断：`p1_mechanism_diagnostics_summary.md`
- HBD 负结果：`p2_hbd_development_result.md`
- 消融与 rank1：`p4_ablations_results.md`
- 任务长度：`p5_task_length_results.md`
- CUB 三方法结果：`p5_cub_results.md`
- 外部基线：`p5_external_baselines_results.md`、`strong_baselines_sd.md`
- 效率和恢复：`p6_efficiency_results.md`
- 最终定位：`paper_output/final_positioning.md`
- 完成审计：`completion_audit_sd.md`

