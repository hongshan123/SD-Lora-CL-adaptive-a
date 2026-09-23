# SBGC P2 Strict Deployment Results

日期：2026-09-23

## 协议

- CIFAR-100 seed1993、ImageNet-R seed1995、CUB-200 seed1；
- T=10、rank10、20 epoch、双卡每卡 batch64；
- 风险预算固定为 0.05，不使用 warm-up 或数据集专属调参；
- Frozen-P 复用 P1 additive deployment，CUO、Uniform-G 和 Fisher-SBGC 在当前代码上正式运行。

## 主结果

| Dataset | Method | Final | AAA | Forgetting |
|---|---|---:|---:|---:|
| CIFAR-100 | Frozen-P | 87.83 | 92.321 | 7.911 |
| CIFAR-100 | CUO | 88.23 | 92.739 | **4.589** |
| CIFAR-100 | Uniform-G | **88.31** | 92.760 | 5.600 |
| CIFAR-100 | Fisher-SBGC | 88.28 | **92.761** | 5.656 |
| ImageNet-R | Frozen-P | **78.92** | 82.181 | 7.182 |
| ImageNet-R | CUO | 78.33 | **82.530** | 6.493 |
| ImageNet-R | Uniform-G | 78.63 | 82.408 | 6.354 |
| ImageNet-R | Fisher-SBGC | 78.82 | 82.479 | **6.264** |
| CUB-200 | Frozen-P | 84.19 | 89.469 | 8.663 |
| CUB-200 | CUO | 84.14 | **89.489** | 8.180 |
| CUB-200 | Uniform-G | **84.33** | 89.465 | **7.781** |
| CUB-200 | Fisher-SBGC | 84.29 | 89.462 | 7.952 |

相对每个数据集的最佳 Frozen-P/CUO 基线，Fisher-SBGC 的 `Final/AAA` 差值为：

- CIFAR-100：`+0.05/+0.022`；
- ImageNet-R：`-0.10/-0.051`；
- CUB-200：`+0.10/-0.027`。

Fisher 相对 Uniform 的 `Final/AAA/Forgetting` 差值为：

- CIFAR-100：`-0.03/+0.001/+0.056`；
- ImageNet-R：`+0.19/+0.071/-0.090`；
- CUB-200：`-0.04/-0.003/+0.171`。

因此 diagonal Fisher 只在 ImageNet-R 上有一致的小幅优势，在 CIFAR-100/CUB-200 上没有超过 uniform 风险约束。

## 风险与开销审计

| Method | Dataset | Active branches | Mean distortion | Calibration total | Solver total | Boundary total |
|---|---|---:|---:|---:|---:|---:|
| Fisher | C100 | 206/216 | 0.248 | 223.81s | 44.74s | 269.71s |
| Fisher | INR | 191/216 | 0.193 | 220.99s | 41.21s | 263.50s |
| Fisher | CUB | 207/216 | 0.338 | 36.68s | 45.10s | 82.92s |
| Uniform | C100 | 210/216 | 0.249 | 227.76s | 43.27s | 272.62s |
| Uniform | INR | 207/216 | 0.217 | 220.62s | 41.33s | 263.23s |
| Uniform | CUB | 215/216 | 0.400 | 44.94s | 46.21s | 92.58s |

- 六条运行的最大部署风险均不超过 `0.050000010`，通过 `0.050001` 容差；
- artifact 均含 Task0--9，统计全部有限；
- 每条最终状态均为 389,520 个持久标量，未随任务增长；
- task-boundary allocator peak 约为 2,852 MiB。

## 预注册判定

- 三数据集 Final/AAA 距最佳基线不超过 0.30：**通过**；
- 至少两个数据集 Final 或 AAA 提升 0.30：**失败，零个数据集达到**；
- Forgetting 不高于最佳基线 0.50：**C100 失败**，比 CUO 高 1.067；
- Fisher 在至少两个数据集超过 Uniform：**失败，仅 ImageNet-R**；
- 风险、有限值和固定状态：**通过**。

最终判定：**NO-GO，不进入 P3 多 seed**。

## 结论

固定 5% G 风险约束是有效且非空的 regularizer，相比 Frozen-P 在三个数据集都降低了 forgetting；但它没有稳定超过 CUO，也没有把 diagonal Fisher 的方向敏感性转化为跨数据集收益。约 88%--100% 的分支被约束且 current-target distortion 达 0.19--0.40，表明方法主要在强制收缩当前写入，而不是实现更优的功能选择。

因此不继续扫描 0.01/0.10，也不进入 seeds 1--5。若保留该研究线，只能把结果作为“功能加权 G 风险未优于 uniform/CUO”的失败机制分析；论文主线应回到更简单、性能更稳定的固定基底方法。
