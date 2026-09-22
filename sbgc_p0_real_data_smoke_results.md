# SBGC P0 Real-Data Smoke Results

日期：2026-09-22  
代码提交：`0fe0c33`  
协议：Task 0/1、2 epoch、rank10、双卡 DDP、每卡 batch64；GPU 2、3 未使用。

| Dataset | Seed | Task0 operator error | Task1 active branches | Max deployed risk | Mean target distortion | Mean Fisher CV | Fisher/uniform gap | Final Top-1 | AAA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-100 | 1993 | 1.2301e-7 | 24/24 | 0.050000002 | 0.4168 | 1.5266 | 0.0698 | 94.90 | 95.65 |
| ImageNet-R | 1995 | 1.2316e-7 | 24/24 | 0.050000002 | 0.5656 | 1.3457 | 0.0395 | 78.45 | 81.20 |
| CUB-200 | 1 | 1.2524e-7 | 24/24 | 0.050000005 | 0.5835 | 1.4339 | 0.0621 | 92.57 | 94.72 |

## Acceptance Audit

- 三个运行均完整结束，无 Traceback、NCCL error、OOM 或非有限值；
- Task 0 QR canonicalization 均满足 `<1e-6`；
- Task 1 所有分支的 FP32 部署风险均满足 `<=0.05+1e-6`；
- tensor/RNG hash、prototype rank consistency 与 EvalTensorHash 均通过；
- v6 state 均为 `task_id=2`，包含 24 个 Q/V 分支且 covariance/sensitivity counts 全部为正；
- CPU save/rebuild logits 测试误差 `<2e-6`，全仓测试 419 passed；
- 两进程合成 DDP smoke 的所有 rank 最终状态 hash 一致。

P0 判定：**PASS**。

## Mechanism Warning

工程验收通过不等于方法有效。三个数据集 Task 1 的 24 个分支全部激活约束，且 mean current-target distortion 为 0.42--0.58，远高于 P1 预注册的 10% 中位阈值。该现象说明统一 5% historical-risk budget 可能显著偏离 additive 当前任务目标。

因此下一阶段只运行 `shadow_only=true`：部署仍为 additive Frozen-P，仅统计九次 transition 的约束活跃率、Fisher/uniform 差异和 distortion。不得依据本次两任务准确率宣称 SBGC 优于任何基线。

