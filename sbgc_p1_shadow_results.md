# SBGC P1 Shadow Diagnostic Results

日期：2026-09-22

## 协议

- CIFAR-100 seed1993、ImageNet-R seed1995、CUB-200 seed1；
- T=10、rank10、20 epoch；
- 双卡 DDP、每卡 batch64，有效 batch128；
- `sa_g_shadow_only=true`：计算 additive、uniform-budget 和 Fisher-budget 候选，但实际部署 additive canonical Frozen-P；
- transport、Dual-B、HBD、NormCap、Adaptive-A 和 current-branch normalization 全部关闭。

## Additive Shadow 性能

| Dataset | Final | AAA | Forgetting |
|---|---:|---:|---:|
| CIFAR-100 | 87.83 | 92.321 | 7.911 |
| ImageNet-R | 78.92 | 82.181 | 7.182 |
| CUB-200 | 84.19 | 89.469 | 8.663 |

这些数值是实际部署的 additive Frozen-P 轨迹，不是 SBGC 候选性能。

与旧 `live_a_aggregate_b` Frozen-A 同 seed 结果相比，Final 差值仅为 `+0.08/+0.02/+0.05`，AAA 差值为 `+0.035/+0.008/+0.046`。两者不是逐位相同：SBGC 在 Task 0 后将输入基底 QR canonicalize，后续 SGD 对一般 gauge 变换不严格不变。

## 预注册诊断

| Dataset | Active transition | Fisher/uniform different transition | High-CV branch | Fisher distortion median | Uniform distortion median |
|---|---:|---:|---:|---:|---:|
| CIFAR-100 | 100% | 100% | 100% | 0.10575 | 0.09713 |
| ImageNet-R | 100% | 100% | 100% | 0.05797 | 0.05466 |
| CUB-200 | 100% | 100% | 100% | 0.07349 | 0.11092 |

三数据集 median-of-medians Fisher distortion 为 `0.07349`。分析器输出保存在 `sbgc_p1_go_no_go.json`，六项预注册检查全部通过，P1 决策为 **GO**。

注意：Fisher 与 uniform distortion 分别在不同 sensitivity metric 下计算，不能直接用两列大小宣称某个候选更优。P1 只证明约束非空、diagonal Fisher 会实质改变解，并且完整序列的总体 Fisher target distortion 没有超过预注册阈值。

## 关键观察

1. 三个数据集的 9/9 transition 都至少有一个分支需要正对偶变量，说明 additive G merge 经常超过 5% 历史响应风险预算。
2. Fisher/uniform 候选在每个 transition 的平均相对差异都超过 `1e-3`，输出 sensitivity 不是数值上的无效装饰。
3. 早期 transition 扭曲很强：Task 1 的 Fisher distortion 中位数为 C100 `0.478`、INR `0.452`、CUB `0.500`。随着累计 G 能量增大，同量新更新的相对风险下降，后期 distortion 通常明显降低。
4. 因为 P1 不部署候选，它不能证明风险约束改善准确率。真正的有效性必须由 P2 uniform/Fisher 正式部署与 Frozen-P/CUO 的 Final、AAA 和 Forgetting 决定。

## 完整性与异常记录

- 三份最终 artifact 均包含 Task 0--9、每任务 24 个 Q/V 分支、有限且为正的 covariance/sensitivity counts。
- 校准前后 tensor hash 与 RNG hash 均通过，完整测试为 `423 passed`。
- 早期无效/中断运行分别归档在 `SBGC_P1_INVALID_INITIAL_FACTORY_20260922`、`SBGC_P1_INTERRUPTED_EXTERNAL_20260922_1719` 和 `SBGC_P1_FAILED_NCCL_INR_20260922`，均不进入结果统计。

