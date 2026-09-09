# 无 Dual-B 核心消融与三路配对结果

更新时间：2026-09-08  
运行服务器：cuda11（NVIDIA RTX PRO 6000 Blackwell Server Edition）  
队列：`nodual_core_then_threeway_queue.log`  
队列脚本：[run_nodual_core_then_threeway_single_gpu_queue.sh](./run_nodual_core_then_threeway_single_gpu_queue.sh)

## 1. 实验目的

本轮实验在去掉 Dual-B 分类头后，验证以下问题：

1. Prototype Transport、Coordinate Alignment 和 NormCap 是否仍然有效；
2. Adaptive-A 是否优于始终更新 A 的 Live-A 和冻结 A 的 Frozen-A；
3. 去掉 Dual-B 是否改变最终性能；
4. 当前方法的性能增益来自哪个模块。

## 2. 统一协议

- `sa_dual_head=false`；
- 保留全局 prototype 分类器，关闭 FC/prototype 融合头；
- ImageNet 预训练 ViT-B/16，LoRA rank=10；
- 无回放样本，`memory_size=0`；
- 每个任务训练20个 epoch；
- 单卡训练，GPU0/GPU1 并行，`batch_size=128`；
- CIFAR-100：10类/任务，10个任务；
- ImageNet-R、CUB-200：20类/任务，10个任务；
- 保持 SGD、学习率、任务顺序和 prototype 评估协议不变；
- 42个唯一实验全部正常结束，队列状态均为 `status=0`。

指标定义：

- `Final`：最后一个任务结束后的全类 Top-1；
- `AAA`：各任务边界全类 Top-1 的平均值；
- `Forgetting`：历史任务峰值准确率到最终准确率的平均下降，越低越好。

## 3. 核心组件单 seed 消融

每个数据集使用一个开发 seed：CIFAR-100 为1993，ImageNet-R为1995，CUB-200为1。数值格式为 `Final / AAA / Forgetting`。

| 配置 | CIFAR-100 | ImageNet-R | CUB-200 |
|---|---:|---:|---:|
| Adaptive-A 完整 | 88.38 / 92.439 / 5.956 | 79.03 / 82.476 / 4.183 | 83.86 / 89.370 / 9.169 |
| Adaptive-A 去 Transport | 88.09 / 92.078 / 8.256 | 78.94 / 81.864 / 6.602 | 82.67 / 89.028 / 10.753 |
| Adaptive-A 去 NormCap | 87.95 / 92.641 / 7.778 | 79.13 / 83.218 / 5.643 | 83.86 / 89.370 / 9.169 |
| Live-A 完整 | 88.36 / 92.379 / 6.411 | 79.34 / 82.781 / 4.369 | 82.52 / 88.982 / 11.218 |
| Live-A 仅 Alignment | 88.04 / 91.972 / 8.556 | 79.04 / 82.054 / 6.896 | 76.98 / 87.525 / 17.469 |
| Live-A 仅 Transport | 88.33 / 92.346 / 6.467 | 79.66 / 82.946 / 4.162 | 82.38 / 89.015 / 11.544 |
| Live-A 无 Alignment/Transport | 88.07 / 91.978 / 8.567 | 79.06 / 82.062 / 7.116 | 76.51 / 87.496 / 17.969 |

### 3.1 组件结论

- Transport 是最稳定的有效组件。
- 相比无 Transport，完整 Live-A 的变化为：
  - CIFAR-100：Final `+0.29`，AAA `+0.40`，Forgetting `-2.16`；
  - ImageNet-R：Final `+0.28`，AAA `+0.72`，Forgetting `-2.76`；
  - CUB-200：Final `+6.01`，AAA `+1.49`，Forgetting `-6.75`。
- Coordinate Alignment 的直接精度收益很小：CIFAR-100近乎不变，ImageNet-R略有负作用，CUB-200在无 Transport 时提高约 `0.47` Final。
- NormCap 具有数据集依赖性：CIFAR-100改善 Final 和遗忘，ImageNet-R降低遗忘但牺牲 AAA，CUB-200本 seed 下与完整配置完全相同，说明该约束可能没有触发。

## 4. Live-A/Frozen-A/Adaptive-A 三路配对

以下均为无 Dual-B 结果，统计为3个 seed 的均值±总体标准差。

| 数据集 | 方法 | Final | AAA | Forgetting |
|---|---|---:|---:|---:|
| CIFAR-100 | Live-A | 87.78±0.57 | 91.91±0.51 | 7.27±1.01 |
| CIFAR-100 | Frozen-A | **87.84±0.50** | **91.93±0.41** | **6.51±0.99** |
| CIFAR-100 | Adaptive-A | 87.76±0.66 | 91.93±0.49 | 6.97±1.21 |
| ImageNet-R | Live-A | **79.44±0.07** | **82.65±0.45** | 4.36±0.71 |
| ImageNet-R | Frozen-A | 78.94±0.31 | 82.06±0.36 | **4.26±0.51** |
| ImageNet-R | Adaptive-A | 79.16±0.14 | 82.33±0.37 | 4.45±0.58 |
| CUB-200 | Live-A | 80.76±1.47 | 87.82±0.83 | 10.95±0.40 |
| CUB-200 | Frozen-A | **83.95±0.26** | **88.82±0.50** | **7.07±0.89** |
| CUB-200 | Adaptive-A | 83.28±0.43 | 88.59±0.56 | 7.97±0.99 |

### 4.1 逐 seed 结果

格式为 `Final / AAA / Forgetting`。

| 数据集 | Seed | Live-A | Frozen-A | Adaptive-A |
|---|---:|---:|---:|---:|
| CIFAR-100 | 1993 | 88.36 / 92.379 / 6.411 | 88.30 / 92.398 / 5.667 | 88.38 / 92.439 / 5.956 |
| CIFAR-100 | 1994 | 87.01 / 91.199 / 8.689 | 87.15 / 91.397 / 7.900 | 86.85 / 91.262 / 8.667 |
| CIFAR-100 | 1995 | 87.97 / 92.159 / 6.711 | 88.06 / 91.989 / 5.967 | 88.04 / 92.075 / 6.278 |
| ImageNet-R | 1995 | 79.34 / 82.781 / 4.369 | 78.51 / 82.074 / 4.070 | 79.03 / 82.476 / 4.183 |
| ImageNet-R | 1996 | 79.46 / 83.127 / 5.228 | 79.16 / 82.499 / 4.956 | 79.09 / 82.691 / 5.259 |
| ImageNet-R | 1997 | 79.51 / 82.052 / 3.480 | 79.16 / 81.611 / 3.754 | 79.36 / 81.813 / 3.912 |
| CUB-200 | 1 | 82.52 / 88.982 / 11.218 | 84.31 / 89.512 / 8.221 | 83.86 / 89.370 / 9.169 |
| CUB-200 | 2 | 78.93 / 87.084 / 10.377 | 83.71 / 88.598 / 6.064 | 82.84 / 88.301 / 6.744 |
| CUB-200 | 3 | 80.82 / 87.381 / 11.241 | 83.83 / 88.357 / 6.919 | 83.14 / 88.101 / 7.986 |

### 4.2 配对差值

差值定义为 `Adaptive-A - 对照方法`；Final 和 AAA 越大越好，Forgetting 越小越好。

| 数据集 | 对照 | ΔFinal | ΔAAA | ΔForgetting |
|---|---|---:|---:|---:|
| CIFAR-100 | Live-A | -0.023 | +0.013 | -0.304 |
| CIFAR-100 | Frozen-A | -0.080 | -0.003 | +0.456 |
| ImageNet-R | Live-A | -0.277 | -0.327 | +0.093 |
| ImageNet-R | Frozen-A | +0.217 | +0.265 | +0.191 |
| CUB-200 | Live-A | +2.523 | +0.775 | -2.979 |
| CUB-200 | Frozen-A | -0.670 | -0.232 | +0.898 |

## 5. 去掉 Dual-B 的影响

此前 cuda11 的 Dual-B Adaptive-A 结果与本轮无 Dual-B Adaptive-A 结果对比：

| 数据集 | 协议 | Final | AAA | Forgetting |
|---|---|---:|---:|---:|
| CIFAR-100 | Dual-B | 87.757 | 91.918 | 7.204 |
| CIFAR-100 | 无 Dual-B | 87.757 | 91.925 | 6.967 |
| ImageNet-R | Dual-B | 79.160 | 83.012 | 5.304 |
| ImageNet-R | 无 Dual-B | 79.160 | 82.327 | 4.451 |
| CUB-200 | Dual-B | 83.280 | 87.033 | 7.555 |
| CUB-200 | 无 Dual-B | 83.280 | 88.591 | 7.966 |

Final 在三个数据集上完全不变，说明最终任务边界已经使用 prototype 主结果，Dual-B 主要影响中间任务的评估头。AAA 和 Forgetting 会因是否使用 FC/prototype 融合而变化，因此论文中必须统一分类头协议。

## 6. 当前结论与论文口径

1. **Prototype Transport** 是当前最有证据支持的模块，三个数据集的单 seed 消融均显示其可以提升稳定性，CUB-200上的收益尤其明显。
2. **Coordinate Alignment** 当前主要应作为固定状态算子的一致性机制，而不能仅凭现有结果宣称其带来普遍精度提升。
3. **NormCap** 的作用具有数据集依赖性，适合报告为稳定性-塑性折中机制。
4. **Adaptive-A** 没有在三个数据集上超过最佳固定策略：CIFAR-100和CUB-200中 Frozen-A 更好，ImageNet-R中 Live-A 更好。当前不应声称 Adaptive-A 全面优于 Live-A/Frozen-A。
5. 去掉 Dual-B 不影响 Final，但会改变 AAA 和 Forgetting；无 Dual-B 的结果可以作为更简洁的最终协议，但所有基线和消融必须使用相同分类头。
6. 结果不应通过删除表现最差的 seed 来改善均值，正式报告必须保留预注册的全部 seed。

## 7. 运行证据

- cuda11 队列最终日志：`/home/hongzhijun/SD-Lora-CL-adaptive-a/nodual_core_then_threeway_queue.log`；
- 核心消融共21条，三路配对共27条，重复配置自动跳过；
- 所有唯一实验退出状态均为0；
- 相关模型测试：`39 passed`；
- 队列脚本提交：`3405f68`。
