# HOEP-A 研究立项书

> 暂定名称：Historical-Operator Energy Partitioned A（HOEP-A）  
> 中文名称：历史算子能量分区的固定状态持续 LoRA  
> 立项日期：2026-09-21  
> 当前状态：有条件立项，先做新颖性审计与离线可行性筛选，不直接启动大规模训练

## 1. 研究问题

现有 Frozen-A、Live-A 与 Adaptive-A 的差别，本质上是共享 LoRA down-projection 的 row space 是否以及如何随任务变化。已有实验表明：

- Frozen-A 在 CIFAR-100/CUB-200 上通常稳定，但可能限制新任务方向；
- Live-A 在部分 ImageNet-R 设置中具有塑性优势，但会移动全部历史坐标；
- 现有 scalar-gate Adaptive-A 对所有 normal directions 使用同一个连续系数，无法区分哪些坐标真正承载历史算子，因而常退化到某一端点。

本项目不再寻找单一全层标量 `gamma`，而是研究：

> 能否直接依据已部署历史有效算子的方向能量，把共享 A 分成必须保护的稳定坐标和允许更新的低风险坐标，并在一个可证明的全局历史算子风险预算内获得比 Frozen/Live 更好的稳定性-塑性折中？

## 2. 方法定义

### 2.1 固定状态与 canonicalization

每个 ViT Q/V LoRA 分支持久保存：

```text
A in R^(r x d_in)
G in R^(d_out x r)
```

现有历史前向为：

$$
M=G\bar A,\qquad \bar A=\frac{A}{\|A\|_F+\epsilon}.
$$

任务边界先对 A 做薄 SVD：

$$
A=U\Sigma V^\top,
$$

并重参数化为 row-orthonormal basis：

$$
A_c=V^\top,\qquad
G_c=\frac{\sqrt r}{\|A\|_F+\epsilon}GU\Sigma.
$$

于是：

$$
A_cA_c^\top=I_r,
\qquad
G_c\frac{A_c}{\sqrt r}=G\bar A.
$$

canonicalization 不改变部署历史算子，也不增加持久状态。Task 0 没有历史风险，A 正常训练；该机制从 Task 1 开始启用。

### 2.2 历史算子方向能量

定义规范化历史系数：

$$
C=\frac{G_c}{\sqrt r},
\qquad M=CA_c.
$$

对小矩阵做特征分解：

$$
S=C^\top C=V_S\Lambda V_S^\top,
\qquad
\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_r),
$$

其中特征值按从小到大排序。再执行纯正交 gauge rotation：

$$
A_e=V_S^\top A_c,
\qquad
C_e=CV_S.
$$

有：

$$
C_eA_e=CA_c=M,
\qquad
C_e^\top C_e=\Lambda.
$$

因此每个坐标的 $\lambda_i$ 是该方向对当前历史有效算子 Frobenius 能量的精确贡献，而不是 LoRA 因子范数代理。

### 2.3 全局风险预算与稳定/可塑坐标

汇总全部 Q/V 分支的方向能量，按 $\lambda_{l,i}$ 从小到大选择最大的可塑集合 $\mathcal P$：

$$
\sum_{(l,i)\in\mathcal P}\lambda_{l,i}
\le
\varepsilon\sum_{l,i}\lambda_{l,i}.
$$

主配置预注册：

$$
\boxed{\varepsilon=0.05}.
$$

其余坐标构成稳定集合 $\mathcal S$。训练 Task $t>0$ 时：

- $A_{\mathcal S}$ 固定；
- $A_{\mathcal P}$ 在 $A_{\mathcal S}$ 的正交补内更新并 retraction；
- 当前任务的 B、scale 与分类头照常训练；
- 本任务的谱分区固定，不在 step 间刷新。

特征值重根必须作为完整谱簇选择。如果一个重根簇会跨越预算边界，则整个簇留在稳定集合，避免结果依赖 `eigh` 返回的任意特征向量坐标。

### 2.4 任务边界持久化

任务结束后保持现有两步语义：

1. 用 least-squares coordinate alignment，把任务开始时的历史 anchor 投影到新 A 坐标；
2. 用 operator-preserving absorption 把当前 $sBA$ 写入 G。

canonical A 满足 $\|A\|_F=\sqrt r$，因此当前任务吸收项为：

$$
G_t=G_{\rm align}+s_t\sqrt r B_t,
$$

并严格满足：

$$
(s_t\sqrt r B_t)\frac{A_t}{\sqrt r}=s_tB_tA_t.
$$

任务结束后仍只保存 A 和 G；谱、mask、anchor 和优化器缓存均为任务内临时状态。

## 3. 可证明性质

### 命题 1：规范坐标下的算子能量分解

当 $AA^\top=I$ 且 $C^\top C=\Lambda$ 时：

$$
\|M\|_F^2=\|CA\|_F^2=\operatorname{tr}(C^\top C)=\sum_i\lambda_i.
$$

因此 $\lambda_i$ 精确度量历史有效算子在第 i 个规范坐标上的能量。

### 命题 2：历史不可恢复能量上界

若稳定坐标对应的 A 行保持不变，仅允许 $\mathcal P$ 中的行在其正交补内变化，则完成最优 LS alignment 后：

$$
\min_X\|XA_t-M_{t-1}\|_F^2
\le
\sum_{(l,i)\in\mathcal P}\lambda_{l,i}.
$$

于是全网络相对历史算子暴露满足：

$$
\frac{\sum_l\min_X\|X A_{l,t}-M_{l,t-1}\|_F^2}
{\sum_l\|M_{l,t-1}\|_F^2+\epsilon}
\le\varepsilon.
$$

该上界约束的是实际部署历史算子，不是未加权 Grassmann 距离或因子范数。

### 命题 3：端点统一

- $\varepsilon=0$：所有历史能量方向受保护，接近 Frozen-A；
- $\varepsilon=1$：全部方向可塑，退化为 canonicalized Live-A；
- $0<\varepsilon<1$：按历史算子能量进行方向选择，不是对全部方向统一缩放。

注意：这不保证分类损失或 forgetting 单调改善；它只给出 task-boundary 历史有效算子的可恢复性上界。性能主张必须由实验验证。

## 4. 与现有工作的边界

本项目不能声称首次提出稳定/可塑子空间、能量阈值或 LoRA 子空间演化。高风险近邻包括：

- SplitLoRA（ICLR 2026）：分解历史任务的平均梯度空间，并在 minor gradient subspace 中构造每任务 LoRA；
- LoDA（2026）：以 projection energy 构造 general/task-specific down-projection 子空间；
- Geo-LoRA（2026）：通过 Grassmann 几何约束 shared/task-specific LoRA 的演化，并包含 core/slack 分解；
- Share（2026）：动态更新单个共享低秩 foundational subspace，并重投影历史知识；
- InfLoRA、LoRA-DRS、BiLoRA：通过梯度、漂移或近正交参数空间控制任务干扰。

拟议方法当前可争取的窄差异是：

> 不保存历史梯度子空间或任务 LoRA bank，而是直接对固定状态部署算子 G A 的规范坐标能量做谱分账；用一个跨层全局预算选择允许移动的历史低能量坐标，并给出 LS alignment 后的历史算子不可恢复能量上界。

在完成 SplitLoRA、LoDA、Geo-LoRA 与 Share 的公式级审计前，不得在论文中宣称方法新颖。

## 5. 实现边界

首轮实现只允许改动共享 A 的坐标管理和梯度/retraction：

- 保留 CoordinateStable 的 LS alignment；
- 保留 operator-preserving absorption；
- 不引入 per-task adapter、旧样本、旧特征、task-id 或 router；
- 不修改 prototype transport、Dual-B、HBD 或 classifier 代码；
- 机制筛选时关闭 prototype transport、Dual-B、HBD 与 bounded NormCap，避免掩盖 A 策略；
- 系统验证阶段再把冻结的最佳下游组件等量加回所有对照。

复杂度：每个任务开始时，每个分支只需对 $r\times r$ 的 $C^\top C$ 做 `eigh`；谱和 mask 不持久化。除不可避免的分类头扩展外，active adapter、stored adapter 和 continual state 对任务数均为 O(1)。

## 6. 分阶段实验计划

### P0：碰撞审计与离线谱诊断

1. 逐公式审计 SplitLoRA、LoDA、Geo-LoRA、Share。
2. 从已有 Frozen/Live checkpoint 提取各层 $G^\top G$ 谱。
3. 统计 $\varepsilon=0.01/0.05/0.10$ 下每层可塑维数、跨任务变化和谱集中度。

Go 条件：Task 2 以后至少 30% 分支在主预算下满足 $0<k_l<r$，且三数据集至少两个呈现稳定的非平坦谱。否则该方法大概率退化为 Frozen 或 Live，停止代码主线。

### P1：CPU 数学内核与单元测试

- canonicalization 前后历史算子相对误差 `<1e-6`（FP32）；
- $\|AA^\top-I\|_F<10^{-6}$；
- gauge rotation 前后历史算子相对误差 `<1e-6`；
- 数值 LS residual 不超过所选谱能量上界；
- 重根旋转下选择结果与风险不变；
- operator-preserving absorption 相对误差 `<1e-6`；
- checkpoint/rebuild 输出一致；
- Task 0 跳过分区；Task 1+ 稳定行无梯度、可塑行可更新；
- 持久状态 key/numel 不随任务数增长。

### P2：两任务 smoke 与机制测量

每个数据集只跑 Task 0/1，检查 DDP 同步、显存、retraction、任务边界输出差和实际 alignment residual。任何 NaN、谱 mask 不一致或算子上界违背均先修复，不进入正式训练。

### P3：三数据集单 seed 严格筛选

数据集与 seed：

- CIFAR-100：seed 1993；
- ImageNet-R：seed 1995；
- CUB-200：seed 1。

同协议比较：Frozen-A、Live-A、旧 ratio Adaptive-A、HOEP-A。固定 T=10、rank10、20 epoch、有效 batch128、相同优化器/学习率/任务顺序；主方法只使用 `epsilon=0.05`。

报告 Final、AAA、Forgetting、每任务旧类/新类准确率、每层 k、谱能量、预测风险、实际 LS residual、任务边界 pre/post 输出差、训练时间和峰值显存。

### P4：敏感性与多 seed

仅当 P3 通过后：

- 单 seed 报告 `epsilon in {0.01, 0.05, 0.10}`，不得按数据集选择不同 epsilon；
- 固定一个全局 epsilon 后运行三数据集 seeds 1-5；
- 做 paired bootstrap/置信区间，并与每个数据集更强的 Frozen/Live 端点配对。

### P5：完整系统与长序列

在所有方法上等量恢复 prototype transport/分类头组件，再验证 T=5/10/20/50、task-order permutation、rank 4/10/20。系统结果与纯 A 机制结果分表报告。

## 7. 预注册验收与停止条件

P3 通过条件：

1. 三个数据集 Final 与 AAA 均不低于同 seed 最佳 Frozen/Live 端点超过 0.30 个百分点；
2. 至少两个数据集在 Final 或 AAA 上超过最佳端点至少 0.30 个百分点；
3. 实际 LS residual 始终满足预算上界（数值容差除外）；
4. 不增加随任务数增长的 adapter/continual state。

任一情况触发停止或降级为分析性消融：

- P0 谱近似平坦，或绝大多数层长期 `k=0`/`k=r`；
- 只能通过每数据集单独 epsilon 才取得收益；
- 两个及以上数据集低于最佳端点超过 0.30；
- 性能提升主要来自重新加入的 classifier/prototype 组件，而纯 A 机制无收益；
- 与 SplitLoRA、LoDA、Geo-LoRA 或 Share 的公式和状态语义实质等价。

## 8. 论文候选主张

只有在 P4/P5 通过后，才能主张：

1. 固定状态 continual LoRA 中，共享 A 的方向并非同等重要，其重要性可由已部署历史算子的规范坐标能量精确刻画；
2. 在全局历史算子风险预算下释放低能量坐标，可以在不保存历史数据或任务 adapter 的前提下改善 Frozen/Live 的端点折中；
3. 方法提供 task-constant adaptation state 和 LS alignment 后历史算子不可恢复能量的显式上界。

在此之前，本项目仅是待证伪假设，不替换当前已冻结主方法。

## 9. 参考近邻

- SplitLoRA, ICLR 2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/5035a409f5798e188079e236f437e522-Abstract-Conference.html
- InfLoRA, CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/html/Liang_InfLoRA_Interference-Free_Low-Rank_Adaptation_for_Continual_Learning_CVPR_2024_paper.html
- LoRA-DRS, CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/html/Liu_LoRA_Subtraction_for_Drift-Resistant_Space_in_Exemplar-Free_Continual_Learning_CVPR_2025_paper.html
- LoDA, arXiv 2026: https://arxiv.org/abs/2603.00191
- Share, arXiv 2026: https://arxiv.org/abs/2602.06043
- Geo-LoRA, arXiv 2026: https://arxiv.org/abs/2608.26960

## 10. 实施进度（2026-09-21）

- [x] 完成 SplitLoRA、LoDA、Share 官方代码和 Geo-LoRA 公式审计；
- [x] 完成三个既有 T=10 checkpoint 的 P0 最终态谱诊断；
- [x] 5% 预算下 CIFAR-100、ImageNet-R、CUB-200 的 mixed 分支比例分别为 87.5%、70.8%、58.3%，通过首项 Go 条件；
- [x] 完成 HOEP-A 数学内核、训练钩子、SGD 状态变换和固定状态集成；
- [x] 完成三数据集单 seed、关闭 transport/Dual-B/HBD/NormCap 的机制筛选配置；
- [x] 新增谱诊断脚本与 task-boundary/optimizer-step 单元测试；
- [x] 补充 LS residual 上界和简并谱任意旋转不变测试；
- [x] 验证 checkpoint 重建输出与保存的 `(A,G)` 算子一致，且持久适配状态 numel 不随任务增长；
- [x] 全量测试通过（378 passed）；
- [x] 完成两 rank NCCL task-boundary/rebuild smoke；
- [x] 完成三个真实数据集各 Task 0/1 的两卡 P2 smoke，三组均 status 0、DDP/hash/state/风险上界检查通过；
- [ ] 完成 P3 三数据集单 seed 训练并与 Frozen/Live/ratio Adaptive-A 配对（已于 2026-09-21 启动；双卡每卡 batch64，12 项自动队列运行中）。

详细审计与数值表见 `hoep_prior_art_implementation_audit.md` 和 `hoep_p2_real_data_smoke_results.md`。
