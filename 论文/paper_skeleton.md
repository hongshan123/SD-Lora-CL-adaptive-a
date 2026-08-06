# Paper Skeleton: Online Gauge-Aligned Cumulative Shared-A LoRA

> 状态：骨架（2026-08-06）。数字待多 seed/测量完成后回填；方法描述与实现一一对应（见 method_revision_sd.md 与 experiment_sd.md EXP-014~017）。

## 1. Abstract（草稿）

We propose an online gauge-aligned cumulative LoRA framework for rehearsal-free class-incremental learning. 全部历史任务的上投影被在线折叠为单一累计矩阵，持久 LoRA 状态与任务数无关（10 任务下实测 371,040 参数 = SD-LoRA 的 10.07%，含 prototype 状态减少约 86%）。共享下投影 A 更新后，用闭式 gauge alignment 保持历史有效算子；每任务投影残差/基底旋转/算子保持误差均被记录（实测 6 位小数下为 0，A 基本不跨出旧行空间）。单 seed：ImageNet-R Final 79.06（基线 78.76）、CIFAR-100 Final 87.70（基线 86.89）。多 seed 与统计显著性：待回填。

## 2. Introduction（要点）

- 问题：无回放 CIL 中低秩适配器持续更新导致旧原型漂移与遗忘；参数随任务数线性增长。
- 观察：Shared-A 结构下历史 LoRA 分支可精确写作 `(sum_i s_i Bbar_i) Abar`，因此状态可累计；A 更新后 gauge freedom 需要显式对齐。
- 与现有工作差异：LDC 通用漂移回归、CL-LoRA 双适配器、RanPAC 冻结骨干、InfLoRA 固定子空间、EASE 语义合成（详见 method_revision_sd.md §12）。
- 主张：结构（共享 A）+ 闭式累计 + gauge 对齐 + 单模型推理，不做任务路由/回放/持久 transport 网络。

## 3. Method

### 3.1 Shared-A bank 与累计等价

固定 A 时 bank 前向 = `sum_i s_i B_i(Ax)/(||A|| ||B_i||)` = `H(Ax)`，其中 `H = sum_i s_i B_i/(||A|| ||B_i||)`（Phase A，`backbone/sa_lora.py::fold_cumulative_up_projection`；等价误差 <1e-5）。

### 3.2 Canonical 化与累计状态

`A^T = Q R`（thin QR），`A = R^T Q^T`；有效算子 `H_raw A = (H_raw R^T) Q^T = H Q^T`。v2 artifact 只存 `Q^T`、`H`、`R`（rank×rank），无逐任务 B 文件（`SA_STATE_VERSION=2`）。

### 3.3 Gauge alignment

任务 t 保存时 `H_aligned = H_old (Q_old^T Q_new)`，是 `min_H ||H_old Q_old^T - H Q_new^T||_F` 的闭式解；投影残差 `||H_old Q_old^T (I - Q_new Q_new^T)||_F` 每任务记录。

### 3.4 训练与推理

- 训练：历史支路 `H@Q^T` 固定（等价于 v1 训练口径；当前任务支路 `s B(Ax)` 不归一化，与 v1 一致），任务结束折叠当前 B 后删除。
- 推理：单 canonical 模型（merged 即部署状态），无 task-id/router/逐任务 adapter。
- 消融开关：`cumulative_gauge=false`（无对齐）、`lrpt_enabled`（residual transport）。

### 3.5 完整算法（训练时单任务流程）

设任务 t 开始时已保存 `(Q_{t-1}, H_{t-1}, R_{t-1})`，当前任务参数 `(A_t, B_t, s_t)`：

1. 前向：历史支路 `H_{t-1}(Q_{t-1}^T x)`（固定、不可训练），当前支路 `s_t B_t(A_t x)`（与 v1 训练口径一致，不归一化）。
2. 训练完成后：`A_t^T = Q_t R_t`（thin QR）；`H_aligned = H_{t-1}(Q_{t-1}^T Q_t)`；当前贡献 `s_t B_t R_t^T/(||A_t|| ||B_t||)`；保存 `H_t = H_aligned + s_t B_t R_t^T/(||A_t|| ||B_t||)` 与 `(Q_t, R_t)`；删除 `B_t`。
3. 记录诊断：`residual = ||H_{t-1} Q_{t-1}^T (I - Q_t Q_t^T)||_F / ||H_{t-1} Q_{t-1}^T||_F`、`rotation = ||Q_{t-1}^T Q_t - I||_F / sqrt(r)`、`preservation = ||H_aligned Q_t^T - H_{t-1} Q_{t-1}^T||_F / ||H_{t-1} Q_{t-1}^T||_F`。
4. 无 gauge 消融：跳过第 2 步的 `H_aligned`（直接 `H_t = H_{t-1} + ...`）。

实现：`backbone/sa_lora.py`（v2 状态、`_CumulativeSharedAQKV`、`_save_cumulative_state`）；任务 0 构造透传见 `utils/inc_net.py`。

### 3.6 持久状态与参数量

- 每 Q/V 分支：`Q^T`（r×d）、`H`（d×r）、`R`（r×r）。r=10、24 分支：184,320 + 184,320 + 2,400 = 371,040（与任务数无关）。
- 原型：INR 200×768=153,600；C100 100×768=76,800。
- 与 v1 对比：v1 每任务新增 24×768×10=184,320 参数（+scale），O(T) 增长；v2 恒定。

## 3.7 实验协议

- 无回放：memory_size=0、原型仅当前任务数据、无 head-tune/旧数据校准。
- 任务划分：ImageNet-R 200 类 10 任务（init 20/inc 20）、CIFAR-100 100 类 10 任务（init 10/inc 10）；任务长度消融 T=5/10/20/40。
- 优化：Adam，lr 0.01（INR）/0.008（C100，cosine），batch 32，20 epoch/任务，rank 10，正交共享 A 初始化，4×RTX 3090 DDP。
- 指标：Final Top1（全类评估）、AvgAcc、Forgetting；参数与磁盘状态；FLOPs/吞吐/峰值显存。
- 代码与产物：commit 可回退；每运行记录 commit、配置、日志、artifact；`verify_sa_consistency.py` 审计 merged/恢复一致性。

## 4. Results（占位）

### 4.1 单 seed（已回填，见 `论文/result_table_single_seed.md`）

- INR：79.06 / 81.77 / 6.82（基线 78.76 / 83.13 / 5.61；EXP-009 79.34 / 82.47 / 7.26）。
- C100：87.70 / 91.83 / 8.53（基线 86.89 / 91.44 / 5.58；EXP-009 88.42 / 92.07 / 8.08）。

### 4.2 多 seed（待回填）

- INR seeds 1995/1/2/3：78.46±0.48 / 82.28±0.37 / 7.86±1.31（已回填）。
- C100 seeds 1993/1/2/3：87.83±0.13 / 91.52±0.28 / 8.68±0.19（已回填）。
- paired t-test vs EXP-009 同 seed：待基线队列完成后回填。

### 4.3 任务长度（待回填）

- INR T=5/10/20/40；C100 T=5/10/20。

### 4.4 效率与审计（待回填）

- FLOPs/吞吐/峰值显存（GPU 实测）；参数随 T 曲线（v2 恒定 vs v1 线性）。
- merged/恢复一致性：`verify_sa_consistency.py`（已 PASS，feature/prototype logits 差 0）。
- 相关性：operator drift 弱相关（r=0.26）、gauge 诊断零方差、LRPT drift 弱相关（r=0.18）；控制组 r=-0.73 提示塑性混杂。

## 5. Ablations（待回填多 seed）

- cumulative-only vs gauge（单 seed 已有：78.78 vs 79.06 INR；C100 待补）。
- gauge+LRPT（单 seed 已有：78.49/82.19/6.38 INR）。
- 无 prototype / EXP-009（已有单 seed；多 seed 队列待跑）。

## 6. Limitations

- 额外数据集与外部强基线（InfLoRA/CL-LoRA/LoRA-DRS/DGS）未完成；单 seed Final 低于 EXP-009（但高于 SD-LoRA），AvgAcc 仍有缺口。
- 诊断与遗忘相关性弱：历史算子保持是机制证据，不是遗忘预测器。

## 7. Figures（计划）

- 参数量 vs T：v1 线性 vs v2 恒定。
- 每任务 gauge residual/rotation/preservation 曲线。
- CIL 曲线（final Top1 随任务）。
