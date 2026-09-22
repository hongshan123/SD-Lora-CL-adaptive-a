# Function-Sensitive G Consolidation 研究任务书

更新：2026-09-22

## 1. 研究转向

Adaptive-A 与 activation-aware HOEP 均未在三个数据集上稳定超过 Frozen-A。当前研究停止继续搜索 A 的更新门控，固定 Task 0 学得的输入基底，转而研究当前任务系数如何写入累计上投影矩阵 G。

暂定方法名：**Sensitivity-Budgeted G Consolidation (SBGC)**。

目标是保持 rehearsal-free、单一 LoRA、任务常数部署状态，同时把任务边界的 additive merge 改成对历史分支响应风险有显式约束的闭式合并。

## 2. 方法定义

每个 Q/V 分支维护固定状态：

$$
P\in\mathbb R^{r\times d},\quad
G\in\mathbb R^{d\times r},\quad
C_{\rm hist}\in\mathbb R^{r\times r},\quad
f_{\rm hist}\in\mathbb R^d.
$$

Task 0 结束时对 $A_0^T=QR$ 做精确规范化：

$$
P=Q^T,\qquad G_0=s_0B_0R^T,
$$

因此 $G_0P=s_0B_0A_0$。Task $t>0$ 固定 $P$，仅训练 $B_t,s_t$ 和分类头：

$$
\Delta y=G_{t-1}Px+s_tB_tPx.
$$

additive 目标为 $G_{\rm tar}=G_{t-1}+s_tB_t$。确定性任务边界校准收集：

$$
C_t=\mathbb E[(Px)(Px)^T],\qquad
f_{t,j}=\mathbb E[(\partial L_t/\partial y_j)^2].
$$

SBGC 最小化当前任务的 sensitivity-weighted target distortion，并约束历史响应风险：

$$
R_{\rm hist}(G)=
\frac{\sum_j f_{{\rm hist},j}(g_j-g_{{\rm old},j})C_{\rm hist}(g_j-g_{{\rm old},j})^T}
{\sum_j f_{{\rm hist},j}g_{{\rm old},j}C_{\rm hist}g_{{\rm old},j}^T+\epsilon}
\le 0.05.
$$

给定对偶变量 $\eta$，每个输出行的闭式解为：

$$
g_j(\eta)=
\left[f_{t,j}g_{{\rm tar},j}C_t+
\eta f_{{\rm hist},j}g_{{\rm old},j}C_{\rm hist}+\delta g_{{\rm tar},j}\right]
\left[f_{t,j}C_t+
\eta f_{{\rm hist},j}C_{\rm hist}+\delta I\right]^{-1}.
$$

实现使用 FP64 `torch.linalg.solve`、倍增括区间和 40 步二分，部署保存 FP32。当前任务统计只在求解完成后写入历史累计量，不能约束自己。

## 3. 状态与边界

- 部署 LoRA 仍只有 `(P,G)`，ViT-B rank10 共 368,640 个标量。
- 额外固定状态为 2,400 个 covariance 标量、18,432 个 sensitivity 标量和 48 个 count 标量。
- 精确持久状态总量为 389,520 个标量，比 LoRA factors 增加 5.66%，不随任务数增长。
- 分类头随类别扩张不计入 task-constant continual-adaptation overhead。
- 禁止与 Adaptive-A、coordinate/prototype transport、HBD、NormCap、Dual-B 和 normalized current branch 联用。
- 5% 仅是 per-transition branch-response surrogate risk，不是全局 forgetting bound。
- diagonal Fisher 不是完整 Hessian，校准样本来自当前任务，不能声称直接测得旧数据风险。

## 4. 实验阶段

### P0：数学、状态与真实 Task 0/1 smoke

- CIFAR-100 seed1993、ImageNet-R seed1995、CUB-200 seed1；
- 双卡每卡 batch64，rank10，Task 0/1，2 epoch；
- GPU 对为 `0,1`、`4,5`、`6,7`，避开 `2,3`。

验收：Task 0 算子误差 `<1e-6`，非 shadow 部署风险 `<=0.05+1e-6`，重建 logits `max_abs_diff<1e-5`，DDP 状态一致，校准不污染分类头、已有 `.grad` 或 RNG。

### P1：Frozen-P shadow diagnostic

三个数据集 T=10 单 seed、20 epoch、等效 batch128。同时计算 additive、uniform-budget 和 Fisher-budget 候选，只部署 additive。

进入 P2 的预注册条件：至少两个数据集有至少 30% transition 激活约束；Fisher/uniform 候选在至少 30% transition 相对差异大于 `1e-3`；至少 30% 分支 sensitivity CV 大于 0.1；Fisher 中位 current distortion 不超过 10%；shadow 轨迹与 Frozen-P 一致。

### P2/P3

P2 单 seed严格比较 Frozen-P additive、`cuo_lowrank`、uniform-budget G、Fisher SBGC。只有三数据集均距最佳基线不超过 0.30，且至少两个数据集 Final 或 AAA 提升 0.30，才进入 seeds 1--5 与长序列。

## 5. 当前实施状态

- [x] FP64 闭式求解、风险计算、bracketing/bisection 与状态计数内核；
- [x] 固定 P 的 Q/V wrapper、Task 0 精确 QR、Task 1+ 固定 P；
- [x] test preprocessing 的带梯度校准，DDP 非重叠分片和 all-reduce；
- [x] uniform/Fisher 双候选、shadow-only additive 部署及逐分支诊断；
- [x] v6 task-constant artifact、严格恢复校验与 merged deployment；
- [x] 31 个 CPU 数学/集成测试；
- [x] 两进程 DDP lifecycle smoke，rank 状态 hash 一致；
- [x] 三数据集真实 P0 Task 0/1 smoke，工程验收通过；
- [x] 三数据集 P1 shadow diagnostic 完成；
- [x] 预注册 Go/No-Go 六项检查全部通过，决策为 GO；
- [~] P2 正式比较 Frozen-P、CUO、uniform-budget G 和 Fisher SBGC；配置、计时诊断和队列已冻结，等待正式运行完成。

P0 数值见 `sbgc_p0_real_data_smoke_results.md`。三数据集 Task 1 均为 24/24 分支激活风险约束，但 current-target distortion 已达 0.42--0.58；P1 必须以 shadow 方式验证该现象是否贯穿完整序列。

P1 数值见 `sbgc_p1_shadow_results.md`。C100/INR/CUB 的 Fisher distortion 中位数分别为 `0.10575/0.05797/0.07349`，median-of-medians 为 `0.07349`；三个数据集的 active/different transition fraction 和 high-CV branch fraction 均为 100%。该结果只允许进入 P2，不构成 SBGC 性能提升证据。

P2 固定使用风险预算 `0.05`，不加入 warm-up 或数据集专属调参。三数据集分别使用 GPU `0,1`、`4,5`、`6,7`，每条数据集队列按 Fisher、uniform、CUO 顺序执行。Frozen-P 复用 P1 additive deployment；CUO 在当前提交上全部重跑。进入 P3 的门槛为：三数据集 Final/AAA 均距最佳 Frozen-P/CUO 不超过 `0.30`，至少两个数据集有一项提升 `0.30`，且 Forgetting 不高于两基线最小值 `0.50` 以上。
