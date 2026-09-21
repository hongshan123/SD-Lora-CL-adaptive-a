# HOEP-A P2 真实数据 Smoke 结果

日期：2026-09-21

## 协议

- 两卡 NCCL DDP；
- 每卡 batch size 64，有效 batch size 128；
- CIFAR-100 seed 1993、ImageNet-R seed 1995、CUB-200 seed 1；
- `max_tasks=2`，Task 0/1 各训练 2 epoch；
- rank 10，HOEP 全局历史算子能量预算 5%；
- 关闭 prototype transport、Dual-B、HBD 和 bounded NormCap；
- 保留 prototype classifier、LS coordinate alignment 和 operator-preserving absorption。

本轮是机制与系统 smoke，2-epoch 准确率不能作为 P3 性能结论。

## 运行结果

| 数据集 | 退出状态 | Task 1 可塑方向 | 选中历史能量 | optimizer steps | Final Top-1 | 两任务 AAA | Forgetting |
|---|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-100 | 0 | 149/240 | 4.941711% | 80 | 95.65 | 96.025 | 0.70 |
| ImageNet-R | 0 | 129/240 | 4.987046% | 42 | 80.73 | 82.34 | 2.91 |
| CUB-200 | 0 | 135/240 | 4.953234% | 10 | 92.49 | 94.68 | 6.09 |

## 数值与边界检查

| 数据集 | init equivalence | orthogonality | row-space reconstruction | historical equivalence | current equivalence | boundary residual | 预算内 |
|---|---:|---:|---:|---:|---:|---:|:---:|
| CIFAR-100 | 2.648679e-7 | 3.849935e-8 | 3.849935e-8 | 5.052598e-8 | 5.300722e-8 | 7.084959e-7 | PASS |
| ImageNet-R | 2.469243e-7 | 4.550201e-8 | 4.550201e-8 | 5.876169e-8 | 5.970918e-8 | 2.936213e-7 | PASS |
| CUB-200 | 2.994076e-7 | 4.478251e-8 | 4.478251e-8 | 5.304810e-8 | 5.193264e-8 | 7.377185e-10 | PASS |

三个数据集的 LS alignment condition number 均为 1.000。实际 task-boundary 全局平方残差均远低于预注册的 0.05 上界。

## DDP 与持久状态

- Task 0/1 的 prototype rank consistency、RNG hash 和 evaluation tensor hash 全部 PASS；
- 训练期间抽样单 rank 显存约 8.4 GiB，未出现 OOM；
- 三个最终 `sa_state.pt` 均为 `task_id=2`、version 4、24 个 Q/V 分支；
- 每个状态仅含 `shared_a` 和 `aggregate_up` 等既有固定状态字段，总适配张量元素数均为 368,640；
- HOEP 谱、mask、anchor 和 optimizer cache 均未持久化；
- 日志扫描未发现 Traceback、RuntimeError、CUDA/NCCL error 或 NaN。

## 结论

P2 通过。HOEP-A 已在三个真实数据集上完成 Task 0/1 的两卡 DDP、谱分区、逐步 retraction、任务边界 LS alignment、operator-preserving absorption、prototype 同步与重建评估。可以进入 P3 单 seed、20 epoch 的严格 Frozen-A/Live-A/ratio Adaptive-A/HOEP-A 配对实验。

## 文件

- `exps/hoep_p2_c100_seed1993_t2_e2.json`
- `exps/hoep_p2_inr_seed1995_t2_e2.json`
- `exps/hoep_p2_cub_seed1_t2_e2.json`
- `hoep_p2_c100_seed1993_t2_e2.log`
- `hoep_p2_inr_seed1995_t2_e2.log`
- `hoep_p2_cub_seed1_t2_e2.log`
