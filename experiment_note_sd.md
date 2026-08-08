# Agent 工作日志（experiment_note_sd.md）

> 按时间顺序记录观察、判断、风险与下一步动作。

## 2026-08-04 项目启动

- **观察**：项目不是 Git 仓库；SD-LoRA 基线在今天刚跑完（C100=86.89/91.44，INR=78.76/83.13）；Proto-Routed 路由准确率 low（full≈46-54%，block4/6≈随机）。
- **判断**：原型路由的瓶颈在冻结 ViT 任务原型可分性差；中心化后 margin 显著提升，是低成本突破口。参数压缩方向可用 rank/共享/固定正交 A。
- **风险**：GPU 被其他用户占用较多（每卡已用 ~8GB）；单次完整训练 35-70 分钟；长实验可能被打断。缓解：优先离线评估，训练用小批量/2-4 卡。
- **下一步**：初始化 Git → 建文档 → 实现离线路由评估（中心化/白化），用已有 checkpoint 验证是否能把 Proto-Routed 提升到超过基线。

## 2026-08-04 23:20 环境核查与历史实验盘点

- **观察**：发现仓库中已有另一协作方留下的 experiment.md/experiment_note.md 和 class_proto_routed 变体；E001–E005 已完成（E003-B 被 SIGTERM 中断）。GPU 0-3 均有 ~16GB 空闲。
- **判断**：路由方向（标准 2）短期难出结果：随机专家/global 只比正确路由低 0.79，说明专家任务专属性不足。转向参数压缩方向（标准 1）更可控：历史 K-CMS k=4、recent=0 已把 LoRA 减少 60%，final Top1 只比基线低 0.2–0.3。
- **风险**：直接复用历史日志判定成败不严谨；必须用当前代码+新配置重跑验收。
- **下一步**：实现 scale_weighted 合并模式（SD-LoRA 幅度/方向解耦思想的直接延伸），先在 C100/INR 上跑完整训练；期间检索 LoRA 合并/continual learning 相关论文。

## 2026-08-04 23:35 EXP-001 启动前

- **观察**：冒烟测试通过：scale_weighted 与 separate 的合并结果不同，effective_delta/scale_weighted 方向一致；cluster scale 处理区分正确。
- **判断**：代码改动范围小、风险可控；配置沿用历史最优 k4_anchor / k4_bal002，只改合并模式。
- **风险**：训练约 1.5-2 小时；GPU 被其他用户挤占可能变慢或 OOM。
- **下一步**：提交代码 → nohup 启动队列（C100 → INR）→ 轮询日志。

## 2026-08-04 23:40 EXP-001 运行中，文献检索结论

- **观察**：训练已启动（C100 Task 0），GPU 0-3 均满负载但显存充足（~11-12GB/卡）。检索到 3 篇高相关论文：
  1. CT-Merging (arXiv:2607.20561)：LoRA 合并时最终方向上的系数若直接沿用 SVD 继承值，幅度会与真实任务更新严重不匹配；作者提出 task-level RMS 系数缩放。直接支持“合并时必须显式处理任务级 scale”这一假设。
  2. Subspace-Boosted Model Merging (arXiv:2506.16506)：证明 Task Arithmetic 类合并随专家数增加发生 rank collapse（公共信息主导、任务信息被压缩）；用子空间提升维持任务向量秩。对应 K-CMS 固定 rank=10 聚类里任务越多方向越挤的风险。
  3. Fed-TaLoRA (arXiv:2505.12318)：任务无关共享低秩残差 + 聚合后校准，避免任务级参数增长。对应标准 1 的“单共享适配器 + 校准”方向。
  另有 SA-LoRA（共享 A 解耦低秩，0.20M 可训练参数）与 TMMP（按幅度显著性融合 LoRA 专家）可作后续备选。
- **判断**：scale_weighted 若失败，优先尝试“合并后幅度按任务 scale 的 RMS/范数重建”（CT-Merging 思路）和“聚类秩提升/子空间保持”（Subspace Boosting 思路）。
- **风险**：方向正确但幅度启发式（平均 scale）可能仍不准确；若 C100 提升但 INR 不提升，需要分数据集诊断。
- **下一步**：等待 C100 结果 → 分析 → 再决定是否保留 scale_weighted 或叠加幅度修正。

## 2026-08-05 00:40 EXP-001 C100 结果分析

- **观察**：C100 final Top1=86.29（基线 86.89，历史 separate 86.62），AvgAcc=91.55（基线 91.44），Forgetting=5.96（基线 5.58）。聚类分配合理（[0,9]/[1,4,5]/[2,7]/[3,6,8]），cluster scale 后续训练仍能自适应。
- **判断**：scale_weighted 未达到验收标准，甚至比历史 separate 差。结合文献，判断 rank 容量才是主要瓶颈（任务 8/9 final 下降、遗忘增大符合 rank collapse 特征）。
- **风险**：若只提高 rank，参数缩减比例会下降；cms_low_rank=15 恰好维持 40% 缩减（4 簇 × 24 对 × 2 × 768×15 = 2.212M）。
- **下一步**：INR 完成后记录结果；随后运行 EXP-002：k=4 + cms_low_rank=15 + separate（单变量：只提高聚类容量）。

## 2026-08-05 01:10 EXP-001 全部完成，启动 EXP-002

- **观察**：INR final Top1=78.56（基线 78.76，历史 separate 78.59），AvgAcc=82.86（基线 83.13），Forgetting=5.47（优于基线 5.61）。C100 final Top1=86.29（基线 86.89），AvgAcc=91.55（优于基线 91.44），Forgetting=5.96（差于基线 5.58）。
- **判断**：scale_weighted 与 separate 基本等价（INR 几乎一致，C100 略差），说明方向加权不是关键；下一步按计划提高聚类容量。
- **风险**：rank15 的集群低秩记忆仍只有 4 个；若任务语义差异大，容量提升可能不够。
- **下一步**：启动 EXP-002（rank15 + separate），C100 → INR 顺序运行。

## 2026-08-05 02:05 EXP-002 C100 结果分析

- **观察**：C100 final Top1=85.39（基线 86.89），AvgAcc=91.51（基线 91.44），Forgetting=6.82（基线 5.58）。早期阶段领先（Task 4 后 95.54 vs 95.46），后期崩（Task 9/10：87.54/85.39 vs 基线 87.12/86.89）。
- **判断**：提高聚类 rank 无效，甚至有害；问题更可能是“4 个簇的合并本身干扰近期任务”，或 rank15 的归一化/scale 需要更多轮适应。应优先保留最近任务显式 LoRA（recent_tasks=2），这也是 K-CMS 设计的初衷，且参数恰好减 40%。
- **风险**：recent=2 意味着前 8 个任务进 4 簇（每簇 2 任务），与 recent=0 的每簇 2-3 任务接近，容量压力变化不大；若仍失败，需要换结构（如共享 A）。
- **下一步**：INR 完成后记录；若 INR 也未达标，运行 EXP-003：k=4 + rank10 + recent=2。

## 2026-08-05 02:32 EXP-002 全部完成，启动 EXP-003

- **观察**：INR final Top1=78.58（基线 78.76），AvgAcc=82.83（基线 83.13），Forgetting=5.60。rank15 在两个数据集上都没有比 rank10 separate 更好。
- **判断**：聚类秩不是瓶颈；下一步测试“保留最近 2 任务显式”（recent=2，参数恰减 40%），针对性解决 final 任务下滑。
- **风险**：recent=2 会同时增大前向计算量（多 2 个显式残差），但参数量仍达标。
- **下一步**：启动 EXP-003 队列；运行期间阅读“Merge before Forget / MINGLE / CSF”等连续合并文献，准备 shared-A 备选方案。

## 2026-08-05 04:10 决定实现 Shared-A SD-LoRA（EXP-004）

- **观察**：EXP-003 C100 final Top1=86.39，仍低于基线 86.89；三种 K-CMS 结构变体均未达标。阅读 SLAO（arXiv:2512.23017）和 SA-LoRA 后确认“共享 A + 任务专属 B”是更干净的方向：无有损合并，B 可精确求和。
- **判断**：Shared-A 最终 LoRA 参数 = 共享 A（24×7680）+ 求和 B（24×7680）= 368,640，比基线少 90%，远超验收线；风险是 r=10 的共享子空间容量可能不足。
- **风险**：若正交共享 A 太受限，备选是任务 0 训练 A 后再冻结，或把 rank 提到 16-24（参数仍减 60-80%）。
- **下一步**：代码已写完并通过冒烟测试；等 EXP-003 INR 结束后立即启动 EXP-004 队列。

## 2026-08-05 04:30-04:45 EXP-004 两次启动失败与修复

- **观察**：第一次启动在 Task 0 结束后崩溃（`LoRA_ViT_timm` 没有 `save_merged_lora`）——factory 初始 backbone 未走 Shared-A；修复 `utils/inc_net.py` 后第二次启动在 `save_merged_lora` 崩溃（GPU/CPU 设备不匹配，scale 张量未 `.cpu()`）。
- **判断**：均为实现 bug，不是方法问题；Task 0 训练本身正常（20 epoch 3 分钟，train acc 94%）。
- **风险**：每次崩溃会留下半成品输出目录，重启前已移到 `*_partial*` 目录避免污染。
- **下一步**：第三次启动已验证 Task 0 → Task 1 成功；等待 C100 完成。最终合并参数已确认 368,640（基线 3,686,400 的 10%）。

## 2026-08-05 05:40 EXP-004 C100 结果分析

- **观察**：C100 final Top1=86.53（基线 86.89，-0.36），AvgAcc=91.70（基线 91.44，+0.26），Forgetting=7.47（基线 5.58）。前 8 任务领先，最后 2 任务回落。
- **判断**：Shared-A 方向正确（平均精度更高、参数减 90%），但 r=10 共享子空间在后期任务容量不足；final Top1 缺口约 0.4。
- **风险**：提高 rank 会增加参数但仍远低于 40% 缩减线；A 跨任务训练会破坏 B 精确求和，需要近似合并。
- **下一步**：等 INR 结果；然后跑 EXP-005（rank16，A 可选跨任务训练）。

## 2026-08-05 06:10 EXP-004 全部完成，启动 EXP-005

- **观察**：INR final Top1=77.59（基线 78.76，-1.17），AvgAcc=82.82（基线 83.13，-0.31）。C100 final 86.53（-0.36），AvgAcc 91.70（+0.26）。
- **判断**：r=10 太紧；直接上 rank=20（参数 737,280，减 80%）。A 仍只在 Task 0 训练，保持 B 精确求和性质。
- **风险**：若 rank20 仍差 INR >0.5，说明瓶颈不是秩而是“A 只训练一次”；下一步再开 A 跨任务训练（参数变 45% 缩减，仍达标）。
- **下一步**：启动 EXP-005 队列（C100 → INR）。

## 2026-08-05 07:40 EXP-005 全部完成，启动 EXP-006

- **观察**：r20 C100 final 86.35（比 r10 的 86.53 还差），INR final 78.03（比 r10 的 77.59 好 0.44），仍低于基线。所有 5 个实验的共性缺口都在最终任务/分类头。
- **判断**：不再调骨干结构；改用“最终分类头对齐 + 全类 5 epoch 微调”（零新增参数）。C100 用 r10、INR 用 r20。
- **风险**：全类微调可能轻微过拟合训练集，但通常能把 final Top1 抬 1-2 个点；若仍差，下一手是 A 跨任务训练。
- **下一步**：启动 EXP-006 队列（C100 → INR），重点看最终评估前的 head-tune 日志与 final Top1。

## 2026-08-05 09:00 EXP-006 离线验证成功，官方全量重跑中

- **观察**：C100 训练阶段曾因 head-tune 特征提取 OOM 中断；修复为“特征存 CPU + batch16 + empty_cache”后，改用离线脚本直接对已完成训练的产物做对齐+微调+评估：C100 微调后 final Top1=91.48（基线 86.89），INR（r20 产物）微调后=81.51（基线 78.76），微调前 78.03 与训练日志完全一致。
- **判断**：分类头 5 epoch 全类微调是决定性改进，且不增加 LoRA 参数。为消除“最后任务被重复计数”的原实现怪癖，新增最终重建干净 eval backbone 步骤，保证官方管线与离线结果一致。
- **风险**：全量重跑需约 1.7 小时；GPU 当前空闲，OOM 风险已通过 CPU 特征存储消除。
- **下一步**：轮询官方日志，拿到两个数据集的最终 Trainer 输出后做验收汇总。

## 2026-08-05 09:40 用户反馈：必须严格无回放，EXP-006 判定无效

- **观察**：用户明确：本项目是无回放持续学习，训练完成后不能再回放数据。EXP-006 的“全类 5 epoch 分类头微调”使用了所有旧类训练数据，属于回放，判定无效；已停止官方重跑。
- **判断**：head-tune 的增益不能作为验收依据。回放自由的校准（IL2M 等）需要旧类当前统计，在严格无回放下不可用。唯一合法路线是：共享 A 在每个任务用当时数据持续训练（`sa_train_a_all_tasks=True`），B 每任务专属并保留，参数减 45%，无任何事后数据使用。
- **风险**：A 跨任务训练会引入任务间干扰；若精度不足，下一手是“最终任务更多 epoch”（仅用最终任务数据，无回放）或 BIC 式仅用最终任务验证集的偏置校正。
- **下一步**：启动 EXP-007（A 持续训练 + rank10 + 关闭 head-tune）；全程记录参数与指标。

## 2026-08-05 10:00 EXP-007 已启动；离线验证两类数据无关校准全部无效

- **观察**：EXP-007（`sa_sdlora_at_*`，A 每任务持续训练、B 文件全保留、无 head-tune）09:39 启动，4 卡 DDP，C100 先跑（Task 0-1 每 epoch ~12-22s，预计 C100 ~70min、INR ~30min）。EXP-006 的文档修改已提交（56a74f0），EXP-007 配置已提交（46db103）。
- **离线校准实验（只用测试集，零训练数据）**：对 4 个旧 Shared-A 产物（r10/r20 × C100/INR）测了两种数据无关校正：
  1. `weight_align`（仅缩放最新任务分类头权重范数）：C100 r10 86.53→86.53，INR r10 77.59→77.61，C100 r20 86.35→86.35，INR r20 78.03→78.03。**几乎零收益**。
  2. 推理期余弦归一化（CosNorm，归一化特征与权重、去 bias）：C100 r10 86.53→86.84，INR r10 77.59→77.84，C100 r20 86.35→86.36，INR r20 78.03→77.88。**收益可忽略/为负**。
- **判断**：final Top1 缺口不是分类头权重范数失衡（align 无效），也不只是余弦角度偏置（CosNorm 无效）；缺口来自增量分类头训练频率失衡——EXP-006 head-tune 的 +4.6/+3.5 增益几乎全部来自“重新见过全部类数据”，这正是用户禁止的回放。数据无关的推理期修补路线已穷尽，必须转向训练期设计。
- **文献检索结论**：
  1. CL-LoRA（CVPR 2025, arXiv:2505.24816）：共享低秩适配器 + 任务专属适配器，**训练期用各任务数据算每类原型，推理期全类最大余弦相似度匹配**（rehearsal-free，无回放）。共享适配器用随机正交 down-projection + 早退 KD + 梯度重分配防遗忘；任务适配器用可学习 block-wise 权重。与本项目 Shared-A 结构几乎同构，原型分类器是最佳合法候选。
  2. RanPAC（ECCV 2024）：随机投影 + 每类原型，无回放 PTM-CIL 强基线。
  3. IL2M（ICCV 2019）：双记忆存类统计做分数校正，无回放；但其“当前统计”若在训练完成后补算属于回放，若训练期边学边存则合法。实现时只允许训练期计算。
  4. LUCIR/CosNorm（CVPR 2019）：余弦归一化训练期消除分类器失衡；离线 CosNorm 已证明不足以单用，需要训练期配合。
- **风险**：若 EXP-007 仍差，EXP-008 需要重训（~1.5h）；原型分类器与现有全模型推理的兼容性需要小心（旧类原型必须在训练期用当时模型状态计算，推理期特征用全模型时存在漂移，可参考 CL-LoRA 的 per-task adapter 匹配）。
- **下一步**：等待 EXP-007；结果出来后分析 → 若未达标，实现并运行 EXP-008（Shared-A + 训练期原型存储 + 余弦原型分类器，仍无回放）。

## 2026-08-05 10:48 EXP-007 C100 结果：恰好达标，INR 运行中

- **观察**：C100 final Top1=**86.90**（基线 86.89，+0.01），AvgAcc=91.58（基线 91.44），Forgetting=7.32（基线 5.58，更差）。曲线 [99.4, 96.55, 94.43, 93.52, 91.54, 89.57, 89.67, 87.9, 86.27, 86.90]；Task 8 掉到 86.27 后 Task 9 回升 0.63。A 持续训练的早期收益（Task 1-3 领先 0.2-0.5）被中期遗忘抵消，最终仅以 0.01 险胜。
- **参数核对**：共享 A 184,320 + 10×B 1,843,200 = 2,027,520（+10 个 scale），= 基线 3,686,400 的 55.0%，**减少 45.0%**，满足标准 1 的参数侧要求。
- **判断**：C100 侧验收成立（final Top1 ≥ 基线 且参数减 ≥40%）。成败取决于 INR final Top1 是否 ≥78.76。A 持续训练在中后期有漂移风险（Task 8 的 86.27 明显低于 A 冻结 r10 的 87.02），INR 的漂移可能更严重。
- **风险**：INR 若差 0.5+，回到 A 冻结 + 原型/余弦头方案（EXP-008/009），代码已实现并提交（f953e41、af0cacd）。
- **下一步**：等待 INR 完成（预计 11:10-11:20）；解析最终 Top1/AvgAcc/Forgetting；若达标则做参数量与复现性核对后验收，否则立即启动 EXP-008。

## 2026-08-05 19:35 EXP-007 全部完成：C100 过、INR 差 0.42，判定未达标；启动 EXP-009

- **观察**：INR final Top1=**78.34**（基线 78.76，-0.42），AvgAcc=82.60（基线 83.13），Forgetting=6.20（基线 5.61）。完整曲线 [92.36, 88.13, 85.76, 83.06, 81.56, 80.87, 79.5, 79.14, 77.27, 78.34]；与 A 冻结 r10（77.59）比整体高 0.5-0.75，说明 A 持续训练方向正确但仍差基线。
- **判断**：EXP-007 未达到验收标准（两个数据集必须同时 final Top1 ≥ 基线）。C100 86.90（+0.01）与 INR 78.34（-0.42）都呈现“Task 8 掉、Task 9 回升”，再次指向增量分类头偏置。head-tune（回放）能 +3.5/+4.6，数据无关 weight_align/余弦归一化 ~0，因此唯一合法且对症的路是训练期原型 + 余弦原型分类器（CL-LoRA/RanPAC 同构）。
- **风险**：A 持续训练下旧任务原型是用当时 A 算的，最终评估时 A 已漂移，原型可能失配；若 EXP-009 失败，退 A 冻结 + 原型，或加 CL-LoRA 式 per-task adapter 匹配。
- **下一步**：创建并提交 EXP-009 配置（`sa_use_prototype_classifier=true`、A 持续训练、rank10、B 全保留）；先离线复测 EXP-007 产物的余弦分类器空间，再启动 C100 → INR 队列。

## 2026-08-05 19:37 用户调整实验顺序：INR 优先，CIFAR 作确认；EXP-009-INR 已启动

- **观察**：用户要求“后续实验先在 ImageNet-R 上运行，运行达标后再到 CIFAR 上运行，减少实验量”。EXP-007 产物的离线余弦复测完成：C100 86.90→87.24（+0.34）、INR 78.34→78.28（-0.06），再次确认 fc 权重余弦化救不了 INR。
- **判断**：把 EXP-009 改为 INR 单独队列（`run_sa_sdlora_proto_inr.sh`），INR final Top1 ≥78.76 后才启动 C100；C100 配置保留但暂不运行。原型分类器在训练期逐任务计算并存储，全程无回放。
- **风险**：原型计算会给每个任务增加约 30-60s 开销；A 持续训练下旧原型与最终 A 的漂移是主要未知数。
- **下一步**：19:37 已启动 INR 训练；监控 Task 0 后的“prototype classifier active”日志确认原型路径生效；完成后解析 final Top1/AvgAcc/Forgetting。

## 2026-08-05 20:15 EXP-009-INR 达标（final 79.34 vs 基线 78.76），启动 CIFAR-100 确认

- **观察**：INR final Top1=**79.34**（+0.58），AvgAcc=82.47（-0.66），Forgetting=7.26（-1.65 更差）。曲线 [91.58, 85.62, 83.83, 82.68, 81.76, 81.96, 79.3, 79.62, 79.01, 79.34]。早期（T1-T4）落后 fc 头最多 -2.5，T5 起反超（+1.1），T8 达 +1.7；Task 6 单次跌 2.66 是 A 漂移特征。
- **判断**：原型余弦分类器是对症且合法的改进（训练期算原型、推理期余弦匹配，无回放），INR 已满足用户“差距 <0.2%”的门槛（实际领先 0.58）。LoRA 参数仍减 45%，原型额外 0.154M（200×768）。
- **文献补充**：Learnable Drift Compensation（ECCV 2024，arXiv:2407.08536）指出原型法在持续更新的骨干上存在语义漂移并提供无回放补偿；Resurrecting Old Classes with New Data（CVPR 2024）也做特征漂移估计。若 C100 最终任务回落，可参考 LDC 思路。
- **风险**：C100 基线更高（86.89）且 EXP-007 只有 +0.01 余量；原型早期劣势在 10 任务长序列中可能更明显。
- **下一步**：创建 C100 单独队列并启动；预计 ~70min；完成后解析 final Top1/AvgAcc/Forgetting，判定验收标准 1。

## 2026-08-05 21:22 EXP-009 全部完成：双数据集达标，验收标准 1 达成

- **观察**：C100 final Top1=**88.42**（基线 86.89，+1.53），AvgAcc=92.07（+0.63），Forgetting=8.08；曲线 [98.3, 96.25, 94.6, 93.75, 92.2, 90.6, 90.84, 88.21, 87.57, 88.42]，T2 起反超 fc 头，T8 领先 1.30。离线独立复算（存储原型 + 合并骨干 + 测试集）：C100=88.42、INR=79.34，与训练日志完全一致。
- **参数实测**：共享 A 184,320 + 10×B 1,843,200 + 10 scale = 2,027,530；基线 3,686,400。**减少 45.0%**。含原型合计（INR 200×768=153,600）后仍减少 40.8%，两种口径都满足 ≥40%。
- **无回放审计**：`memory_size=0`；原型在 `_compute_prototypes` 中于每个任务 `incremental_train` 结束时用该任务当时的数据计算并存储，训练完成后不再触碰任何训练数据；head-tune 默认关闭；离线验证只用测试集。
- **判断**：验收标准 1 满足（参数减 ≥40%，C100 88.42 ≥ 86.89，INR 79.34 ≥ 78.76）。剩余短板 AvgAcc（C100 +0.63 / INR -0.66）与 Forgetting（更差）不影响验收主指标。
- **风险**：无阻碍项。后续若追求遗忘指标，可做 LDC 漂移补偿，但目标已达成，不再增加实验量。
- **下一步**：归档三份 md、提交 git、标记目标完成。

## 2026-08-05 22:31-22:58 第二轮 LRPT 实现与冒烟

- **观察**：plan_sd.md 第二轮计划（LRPT）已提交，但没有任何实现。实现了 `backbone/lrpt.py`：闭式 rank-r 最小二乘 `W* = R X^T (X X^T + reg I)^{-1}` + SVD 截断得到 `U,V`，transport 为 `p' = p + U(V^T p)`；`models/sa_sdlora.py` 在每任务训练前用当前任务数据捕获 pre-update 特征，训练后捕获 post-update 特征，拟合后递归移动磁盘上的旧 prototype，transport 用后即弃。配置 `lrpt_rank=10`（与 LoRA rank 绑定）、`lrpt_reg=1e-2`。
- **判断**：接入点正确：pre-update 捕获在 `super().incremental_train` 之前（A 尚未更新），post-update 在训练/重建之后；只触碰当前任务训练数据，`memory_size=0`，无 head-tune。
- **风险/修复**：首次冒烟在最终任务 `_extract_current_task_features` 越界（`_cur_task+1` 超出 increments）；改为训练前显式传当前任务索引、方法内默认用 `_cur_task`。修复后 1-epoch 全 10 任务冒烟通过（exit 0）。
- **冒烟数据**：每任务 `relative_drift_error` ≈ 0.91–0.96，说明 rank-10 线性 transport 在 1-epoch 下只解释少量漂移；正式 20-epoch 训练的漂移结构未知，需要完整实验判断。
- **参数口径**：LoRA 2,027,530（同 EXP-009）；INR prototype 153,600 / C100 76,800；LRPT U,V 为任务内临时张量（2×768×10=15,360），应用后丢弃，不计入持久化参数。合计 INR 2,181,130 / C100 2,104,330，均 ≤ 2,211,840（基线 60%）。
- **下一步**：代码已提交（`4b8ae92`）；启动 INR seed1995 正式运行（20 epoch），达标后再跑 CIFAR-100 seed1993。

## 2026-08-05 23:00-23:10 审计脚本与 merged 修复

- **观察**：新写的 `scripts/verify_sa_consistency.py` 在 EXP-009 INR 产物上发现 **per-task 银行与 `sa_merged_lora.pt` 特征不一致（max diff 1.31）**；核对公式后确认最终任务重建 `_rebuild_eval_backbone()` 后用重建骨干（当前 B=0、`saved_b_tasks` 已含 0..9）再次 `save_merged_lora`，旧逻辑只循环 `range(current_task)=0..8`，把最后一任务的 B 丢掉。EXP-009 训练期评估走 per-task 路径，所以日志指标不受影响，但磁盘 merged 产物不满足恢复一致性。
- **修复**：`backbone/sa_lora.py::save_merged_lora` 改为遍历全部 `saved_b_tasks`（重建态已含当前任务），仅当当前 B 非零时追加 `s_cur * B_cur`（训练态语义），同时兼容两种状态。另修复 `scripts/measure_sa_artifact.py` 对 dict 按 key 迭代的计数 bug。
- **验证**：用修复后代码重建 EXP-009 INR/C100 的 `sa_merged_lora.pt`；`verify_sa_consistency.py` 固定输入下 feature max diff 9.5e-6 / 2.0e-5，prototype logit max diff 2.7e-7 / 3.0e-7，全部 PASS。参数审计：INR LoRA+原型 2,181,130（59.17%）、C100 2,104,330（57.08%），均 ≤ 2,211,840。
- **注意**：正在运行的 INR LRPT 进程加载的是修复前代码，跑完后需用修复后代码重建 merged 再验证。
- **下一步**：等待 INR LRPT 完成；解析 final Top1/AvgAcc/Forgetting；若达标启动 C100。

## 2026-08-05 23:35-23:45 INR v1 结果与 LRPT 起始任务缺口

- **观察**：v1 INR seed1995 完成：final Top1=79.63（基线 78.76、EXP-009 79.34）、AvgAcc=83.08（EXP-009 82.47）、Forgetting=6.52（EXP-009 7.26）；离线原型复算 79.63 一致；参数 2,181,130 ≤ 预算。LRPT 每任务 relative_drift_error ≈ 0.89–0.93。
- **判断**：指标方向正确（AvgAcc 提高 0.61、Forgetting 改善 0.74），但日志审计发现 `_cur_task >= 1` 导致 task0→task1 的第一次漂移未补偿：LRPT 从 task2 才开始（“LRPT task 2”为第一条日志），task1 的旧原型在 task1 评估期未更新。这不满足“递归更新全部旧原型”的设计，v1 不能作为验收。
- **修复**：捕获条件改为 `_cur_task >= 0`（task1 前 `_cur_task=0` 捕获 state0→state1 配对特征）；配置改用 v2 输出目录避免旧产物污染；commit `306eb74`。
- **下一步**：重跑 INR v2（23:40 启动）；确认日志出现 “LRPT task 1” 后再判门槛；达标后跑 C100 v2。

## 2026-08-06 00:17-00:30 INR v2 达标与离线审计

- **观察**：INR v2 完成，LRPT 从 task1 起共应用 9 次：final Top1=79.24（基线 78.76，EXP-009 79.34 回退 0.10）、AvgAcc=83.03（EXP-009 +0.56）、Forgetting=6.31（EXP-009 改善 0.95）。门槛判定：Final 两项均过；Forgetting 差 0.047 未达 6.26，但 AvgAcc 83.03 ≥ 82.97，**INR 达标**。
- **判断**：v1（缺第一次漂移补偿）final 79.63 更高，但 v2 才满足“递归更新全部旧原型”的定义；v2 的 task1 提升明显（87.06 vs 85.62），说明第一次补偿有效，最终序列变化符合预期。
- **离线审计**：原型+merged 复算 79.24 与日志一致；consistency PASS；参数 2,181,130 ≤ 2,211,840；单卡 batch32 推理 FLOPs 1.129e12、吞吐 414.8 img/s、峰值显存 577.7 MiB。
- **风险**：INR 通过但余量薄（AvgAcc +0.06、Forgetting 差 0.047）；C100 需 Final ≥88.22 且 Forgetting ≤7.08 或 AvgAcc ≥92.57，是更严格的确认。
- **下一步**：启动 C100 v2（00:30 左右）；期间准备多 seed 队列与消融（普通 transport / 无 LRPT / 无 prototype）。

## 2026-08-06 01:29-01:40 C100 v2 未过门槛，进入 LRPT 容量迭代

- **观察**：C100 v2 完成：final Top1=88.43（门槛 88.22 ✓）、AvgAcc=92.42（门槛 92.57 ✗，差 0.15）、Forgetting=7.63（门槛 7.08 ✗）。LRPT 9 次全部应用，relative_drift_error 0.87–0.91。
- **判断**：单 seed 双数据集验收未达成。Final Top1 两数据集都过（INR 79.24、C100 88.43），但“AvgAcc 双数据集均 +0.5 或 Forgetting 双数据集均 -1.0”的次要指标门槛没过（C100 AvgAcc +0.35、Forgetting -0.45）。根因指向 transport 容量：rank-10 只解释约 10% 漂移。
- **风险**：rank16 若仍不足，备选：(1) 调低 reg 增强拟合；(2) full-rank 普通 transport 作上限诊断（若显著更好，说明线性容量问题而非结构问题）；(3) 若 full-rank 也不够，说明单任务特征配对不足以估计跨任务漂移，需要按失败分析检索新方向。
- **下一步**：启动 INR rank16；达标后跑 C100 rank16；随后补 full-rank 消融。

## 2026-08-06 03:16-03:30 rank16 迭代结论与 affine LRPT 转向

- **观察**：INR rank16（79.28/83.12/6.48）与 rank10 v2（79.24/83.03/6.31）基本持平，仍过 INR 门槛；C100 rank16（88.26/92.37/7.80）比 rank10 v2（88.43/92.42/7.63）更差，未过 C100 门槛。
- **容量诊断**：C100 每任务 norm rank10≈0.87–0.91、rank16≈0.85–0.89、rank32≈0.82–0.86、full≈0.57–0.62；raw full≈0.53–0.63；bias_rel≈0.89。说明提高秩收益递减，普通线性 transport 的容量远大于低秩假设，但低秩+偏置可能补上 C100 的小缺口。
- **判断**：继续 rank 提升无意义；实现 affine LRPT（闭式全局漂移偏置 b + rank-r transport，仍无训练网络、无回放、U/V/b 用后即弃）作为下一候选。
- **下一步**：INR affine（rank16+bias）先跑；达标后 C100 affine；无论结果如何，再补 full-rank 消融以量化“普通 transport 上限”。

## 2026-08-06 03:54-04:10 INR affine 结果

- **观察**：INR affine（rank16+bias）final Top1=79.38、AvgAcc=83.03、Forgetting=6.79；单数据集门槛通过（Final ≥79.14 ✓、AvgAcc ≥82.97 ✓）。affine 拟合 residual 从 plain rank16 的 ~0.89 降到 ~0.85，但 AvgAcc 与 rank10 v2 相同、Forgetting 更差。
- **判断**：偏置提升拟合但不带来明确指标收益；C100 affine 是决定 affine 路线是否继续的关键。
- **注意**：同 seed 下 task0 Top1 在不同运行间有波动（90.95/91.42/92.04），说明 DDP/DataLoader 存在一定运行间方差；对余量薄的门槛需多 seed 统计判断。
- **下一步**：等待 C100 affine；同时准备 full-rank 消融。

## 2026-08-06 05:04-05:15 C100 affine 结果与类均值诊断

- **观察**：C100 affine（rank16+bias）final Top1=88.39、AvgAcc=92.45、Forgetting=7.37；仍差门槛（AvgAcc -0.12、Forgetting -0.29），但比 plain rank10/16 均更好。INR affine 已达标（79.38/83.03/6.79）。
- **判断**：bias 有正向作用但不足。下一步先做类均值空间诊断（当前 transport 拟合在样本空间，而迁移目标是类原型；类中心拟合可能更直接），并测 affine rank10。
- **下一步**：INR affine rank10（含类均值诊断）→ C100 affine rank10 → full-rank 消融。

## 2026-08-06 06:22-06:35 classmean LRPT 失败，回到样本空间 affine

- **观察**：INR classmean（类均值 affine rank16+bias）final=78.84、AvgAcc=82.65、Forgetting=7.07，**未过门槛**（Final 相对 EXP-009 回退 0.50、AvgAcc 差 0.32）。虽然 classmean 拟合 residual 仅 0.12–0.16（样本空间 0.83+），但指标全面变差。
- **判断**：对当前任务类中心过拟合，transport 不迁移到旧类；诊断指标（拟合残差）与最终指标不一致。放弃 classmean 拟合方向。
- **当前最优**：INR 侧 affine rank10（79.29/83.00/6.61）与 affine rank16（79.38/83.03/6.79）都过；C100 侧 affine rank16 最佳（88.39/92.45/7.37，差门槛 0.12/0.29）。正在跑 C100 affine rank10。
- **下一步**：若 C100 affine rank10 仍不过，跑 full-rank 消融看上限；同时检索 LoRA 输出漂移基（E²-LoRA）等结构性方案。

## 2026-08-06 07:32-07:45 C100 affine rank10 接近门槛，引入阻尼

- **观察**：C100 affine rank10 final=88.58、AvgAcc=92.46、Forgetting=7.12，与门槛仅差 0.11/0.04，是目前最优。INR affine rank10 已过（79.29/83.00/6.61）。
- **判断**：transport 可能轻微过度校正；加入阻尼 λ=0.5（应用时缩放 U,V,b 的补偿量）尝试在保持 final Top1 的同时改善 Forgetting/AvgAcc。
- **下一步**：INR d05 先跑；达标后 C100 d05。若仍差，考虑 λ=0.75/0.25 或 full-rank 上限消融。

## 2026-08-06 08:11-08:20 d0.5 未过 INR，d0.9 试验中

- **观察**：INR d0.5 final=79.43（过）、AvgAcc=82.91（差门槛 0.06）、Forgetting=6.62。λ=0.5 的 AvgAcc 低于 λ=1 的 83.00，主要来自 task0/task1 下降。
- **判断**：阻尼过度；λ=0.9 是折中。若 d0.9 INR 过线，再跑 C100 d0.9（目标：C100 Forgetting 7.12→7.08 或 AvgAcc 92.46→92.57）。
- **下一步**：等 INR d0.9；若失败，回 λ=1 并评估 C100 复跑/rank 微调/新机制。

## 2026-08-06 08:50-09:00 INR d0.9 达标，C100 d0.9 运行中

- **观察**：INR d0.9 final=79.26、AvgAcc=83.10、Forgetting=6.65，过线。λ=0.9 同时好于 λ=1（AvgAcc +0.10）与 λ=0.5（+0.19），是 INR 侧最优阻尼。
- **判断**：C100 d0.9 是当前最有希望补上 Forgetting 0.04 缺口的试验。
- **下一步**：等待 C100 d0.9；若过，开始归档/审计；若差一点，尝试 λ=0.95 或 C100 affine r10 复跑。

## 2026-08-06 10:00-10:15 C100 d0.9 未过，双空间 LRPT 准备

- **观察**：C100 d0.9 final=88.42、AvgAcc=92.50、Forgetting=7.29；AvgAcc 较 λ=1 微升、Forgetting 变差，仍未过。C100 affine r10（λ=1）的 Forgetting 7.12 仍是最接近门槛的结果。
- **判断**：单一 λ/rank 微调收益有限。实现双空间 LRPT：样本空间 affine rank10（稳定但欠拟合）+ 类均值 affine rank10（拟合好但单独用过拟合）加权组合（λ_c=0.3），闭式、无网络。
- **下一步**：C100 affine r10 复跑（方差检查）完成后，跑 INR dual → C100 dual。

## 2026-08-06 10:00-11:10 用户反馈：改为真正的 LoRA-aware transport

- **反馈**：用户要求放弃通用特征回归；保存训练前后共享矩阵，计算 ΔA，分离 shared-A 漂移与新增 B_t；用 JVP（J_A(x)ΔA）或逐层 Q/V LoRA 残差变化构造 transport 基底；只学习少量层权重/rank 系数；为每类可存 rank-10 响应统计；优先解决不同历史任务漂移方向不一致（按层/prototype 置信度自适应补偿强度）；并做通用 LRPT、仅 ΔA、ΔA+B_t、逐层四组消融。
- **实现过程**：初版用逐层 qkv 残差 SVD 做基底，发现 1536 维层残差空间与 768 维最终特征空间维度不匹配；改用 `functional_call + forward_ad` 对真实 LoRA 权重做 forward-mode JVP，直接得到最终特征空间漂移响应。修复 flash/efficient attention 高阶反向缺失（切 math SDP）、反向模式 JVP OOM（改 forward-mode + 8 张分块）、CPU/GPU 设备不一致。
- **冒烟**：1-epoch 全 10 任务通过；task1-9 每任务都应用 JVP 基底 transport；final 重建路径正常。
- **参数口径**：JVP 基底 U,V（768×10）与 bias 均任务内驻留、用后即弃；持久化仍只有 LoRA 2,027,530 + prototype（INR 153,600 / C100 76,800）。临时锚点/JVP 中间量单独计入峰值显存与训练开销。
- **下一步**：INR seed1995 正式运行（ΔA+B_t + adaptive）；达标后 C100；随后 delta_a/layerwise 消融与多 seed。

## 2026-08-06 11:43-11:55 INR LoRA-aware 未过，adaptive 受控试验

- **观察**：INR LoRA-aware（ΔA+B_t + adaptive）final=79.49、AvgAcc=82.64、Forgetting=6.36，**未过 INR 门槛**（AvgAcc 差 0.33）。final Top1 是各变体最高，但 task2–7 普遍低于 generic affine。
- **判断**：adaptive 的 0.5 下限可能过度削弱补偿；先跑 adaptive=False（全强度）的 LoRA-aware INR。若仍不过，按用户反馈切回原方案（generic affine LRPT）。
- **下一步**：INR NA 运行中。

## 2026-08-06 12:22-12:35 INR LoRA-aware（无 adaptive）达标

- **观察**：INR LoRA-aware NA final=79.43、AvgAcc=82.68、Forgetting=6.18。Final 过，Forgetting ≤6.26 过（AvgAcc 分支差 0.29 但不影响，二者取其一）。adaptive 关闭后 Forgetting 改善（6.36→6.18），AvgAcc 略升。
- **判断**：LoRA-aware 通过 INR；按规则跑 C100 LoRA-aware NA。若 C100 也过（Forgetting ≤7.08 或 AvgAcc ≥92.57），则用户反馈的新方法验收成立；若不过，切回 generic affine LRPT 完成单 seed 后做多 seed。
- **下一步**：C100 NA 运行中。

## 2026-08-06 13:33-13:45 C100 LoRA-aware 未过，切回原方案

- **观察**：C100 LoRA-aware NA final=88.25、AvgAcc=92.25、Forgetting=7.54，未过门槛。LoRA-aware 在 INR 过（Forgetting 6.18）但 C100 全面低于 generic affine。
- **判断**：按用户反馈“如果以上方法无法达标，再切换回原本设计的方法”，主方法切回 generic affine LRPT（rank10，bias，λ=1）。INR 已达标；C100 最佳 92.46/7.12，复跑该配置做方差检查。
- **下一步**：C100 affine r10 rerun 运行中；若过，进入多 seed 与归档；若仍差一点，再评估 λ=0.95 或类均值残差校准等小步调整。

## 2026-08-06 14:44-14:55 C100 affine r10 复跑仍未过，rank12 试验

- **观察**：C100 affine r10 RERUN2 final=88.46、AvgAcc=92.48、Forgetting=7.21；两次运行都差门槛 0.09–0.13（AvgAcc）/0.04–0.13（Forgetting）。INR affine r10 已过。
- **判断**：λ/复跑无解；试 rank12（介于最优 rank10 与变差的 rank16 之间）。
- **下一步**：INR affine rank12 → C100 affine rank12。

## 2026-08-06 15:00-15:40 classwise 实验失败与回退

- **实现**：按用户反馈完成 JVP 卫生（保存/恢复 requires_grad + 断言）、中点 JVP、有限差分验证日志、class-wise layerwise sensitivity（12×8/类 + 768×8 基底）、基底更新时旧 S_c 坐标变换、sensitivity-based 强度、bias 投影到基底。
- **冒烟结果**：1-epoch INR 灾难性——task0 评估异常（83.8 vs 常规 88+），task1 后旧类精度 50% 并恶化到最终 0%。JVP 验证余弦 0.97–0.99，但 classwise 更新公式（S_c × R_l 系数内积）导致原型被破坏；即便修复基底坐标变换与参数 clone，仍不收敛。
- **决策**：按用户“无法达到目标则 git 回退到当前仓库继续实验”的指示，`git restore` 丢弃未提交 classwise/JVP 改动并删除相关配置，回到 d835567。
- **下一步**：继续原方案 affine rank12（INR 先跑）。
- **补充**：15:36 的 INR affine rank12 日志在 Task 4（60-80）训练期间被 SIGTERM 中断（用户反馈打断），未产生有效结果；后续转入 operator-stability 路线，rank12 未再重跑，该配置保留在仓库中。

## 2026-08-06 15:52 有效算子稳定化取代 class-wise 外推

- **观察**：class-wise sensitivity 的 JVP/有限差分一致性高，并不代表 `S_class` 可以把未知未来 `Delta A` 映射到正确的类原型位移；参数侧映射缺失时，基底坐标变换会把很小误差放大为递归 prototype 破坏。Task 0 也异常说明该路径的隔离性不足。
- **判断**：不再恢复该 prototype transport。Shared-A 的旧分支有可精确计算的线性形式：`sum_i scale_i normalize(B_i) @ normalize(A)`。应保护这个真实历史算子，而非拟合一个不受约束的特征/类别漂移映射。
- **实现/测试**：在 cuda6 的 sdlora 环境，纯函数与极小 Shared-A 骨干测试通过：Task 0 loss=0、Task 1 初始 loss=0、扰动 A 后 loss>0、A 与历史 scale 均有有限梯度、快照不进入 state dict。GPU 0-3 空闲，准备运行 1-epoch x 10-task DDP smoke。
- **风险**：过大的 lambda 会限制 A 的塑性，导致当前任务学习不足；初始 lambda=0.1 只作为安全验证，需从日志中的 raw relative drift 与分类损失比例判断是否做完整训练或调整。

## 2026-08-06 15:55 首次 DDP smoke 的保存后诊断修复

- **观察**：Task 0=79.56，符合 1-epoch 预期且没有稳定项；Task 1 的训练日志出现 `operator_stability=0.0008`，说明损失路径、DDP 与梯度均工作。随后保存阶段报 `KeyError: 1`。
- **判断**：错误发生在训练之后的诊断，不影响已执行的反向传播。`save_lora_parameters` 会递增 backbone 的 `task_id`，但 `saved_b_tasks` 只含构造时加载的旧 B；稳定项必须固定使用构造时快照的历史任务数，不能读取可变计数。
- **动作**：增加 `_operator_reference_task_count`，将所有历史 B/scale 遍历改为该冻结计数，并在单测中模拟保存后再次调用。使用新输出目录重跑 DDP smoke，禁止复用失败产物。

## 2026-08-06 16:04 DDP smoke 通过，启动有效强度 INR

- **观察**：R2 smoke 覆盖 Task 0--9 并以 exit=0 结束。稳定项只从 Task 1 开始，训练项为 0.0008/0.0003/0.0001/0.0000...；保存后 raw relative drift 分别为 0.021862、0.007290、0.001274、0.000201、0.001541、0.000761、0.001501、0.001647、0.000942。merged/per-task 特征和 prototype logits 一致性均 PASS，参数预算不变。
- **判断**：没有重现灾难性遗忘的工程前兆，机制的隔离、DDP、artifact 与恢复路径可信。lambda=0.1 对 CE 的贡献过低，完整训练若沿用它大概率退化为 EXP-009；选择 lambda=1.0 作为最小有效放大，而非直接跳到高强度正则。
- **下一步**：完整 INR 运行中，重点记录每任务 raw drift、当前任务训练精度、最终 Top1/AvgAcc/Forgetting；若出现当前任务塑性明显受损或指标不如 EXP-009，则不再盲目扫 lambda，先比较无正则的旧算子漂移再决定。

## 2026-08-06 16:43-16:45 EXP-012 完整 INR 结果

- **观察**：4 卡完整 INR 正常 exit=0。Final Top1=79.39，AvgAcc=82.20，Forgetting=6.98；相对 EXP-009 的 79.34/82.47/7.26，final 近乎持平（+0.05）、遗忘下降 0.28，但平均准确率下降 0.27。对 SD-LoRA 的最终 Top1 有 +0.63，但 AvgAcc -0.93、Forgetting +1.37。
- **工程审计**：artifact 参数为 2,181,130（SD-LoRA 的 59.17%，预算内）；bank/merged feature 最大差 1.001e-05、prototype logits 差 2.384e-07，均 PASS。Task 1--9 raw operator drift 为约 0.002--0.004，约束没有失效；训练中无 NaN、DDP/保存错误或 Task 0 污染。
- **判断**：有效算子稳定性是一个可实现且安全的约束，但不是充分的遗忘代理。它只保持历史 `sum_i s_i B_i @ A` 的局部线性支路；新任务 B、注意力/MLP 的非线性与深层累积仍可使最终旧类特征和原型决策边界变化。把 lambda 再增大更可能削弱当前任务可塑性，不能据此期待跨过 0.72 的 Forgetting 缺口。
- **决策**：EXP-012 在 INR 未达到 `Final>=79.14` 且 `F<=6.26 或 AvgAcc>=82.97` 的联合门槛，不按 INR-first 规则运行 C100。先补一个不带正则但记录同一 raw drift 的受控诊断，确认 drift 与 Forgetting 的相关性；若弱，停止 operator-level 路线，避免重复 lambda/rank 扫描。

## 2026-08-06 17:00 启动 operator drift 控制组

- **动作**：将保存后诊断条件改为 `_cur_task > 0`（lambda=0 也记录 raw drift），新增 `exps/sa_sdlora_operator_stability_logonly_inr_seed1995.json`（lambda=0，其余同 EXP-009/EXP-012）。训练损失钩子在 lambda<=0 时不加入任何项，因此训练路径等价于 EXP-009。
- **判断**：控制组可同时验证两点：(1) EXP-009 本身的 raw operator drift 大小；(2) drift 与每任务旧类遗忘是否相关。若控制组 drift 与 EXP-012 相当而遗忘更好，说明该算子不是遗忘主因。
- **下一步**：运行 INR 控制组；完成后与 EXP-009/EXP-012 对比。

## 2026-08-06 17:29 控制组完成：operator drift 与遗忘相关性弱

- **观察**：lambda=0 控制组 Final=79.39、AvgAcc=82.47、Forgetting=7.10，基本复现 EXP-009（79.34/82.47/7.26）。raw drift 序列为 0.091/0.020/0.012/0.005/0.008/0.008/0.006/0.010/0.007；EXP-012 约束后为 0.003/0.003/0.003/0.002/0.003/0.003/0.002/0.004/0.003。
- **判断**：drift 被压低 10–30 倍，Forgetting 只改善 0.12，同时 AvgAcc 损失 0.27。该算子不是遗忘主因，operator-level 正则无继续价值。
- **下一步**：回到 generic affine LRPT，第三次复跑 C100 affine r10（前两次 92.46/7.12、92.48/7.21）。

## 2026-08-06 18:39 C100 affine r10 第三次未过；转 raw-space 原型

- **观察**：RERUN3 final=88.58、AvgAcc=92.50、Forgetting=7.17。三次 C100 结果稳定差 0.05–0.13，排除方差解释。
- **判断**：继续调 λ/rank 无意义；改试 raw-space 原型（raw 特征均值再归一化，LRPT 在 raw 空间拟合），可能改变漂移补偿的有效空间。
- **下一步**：INR rawproto r10 先跑；达标后 C100。

## 2026-08-06 18:46 raw-space 原型失败；转 generic adaptive

- **观察**：INR rawproto r10 task0=91.11，task1=52.74，灾难性；终止。离线双头融合评估因缺少每任务原型快照而口径无效，已放弃。
- **判断**：raw 空间低秩 transport 与原型归一化分类空间不匹配；回到 normalized 空间，试 generic affine r10 + adaptive（V 投影强度）。
- **下一步**：INR adaptive 先跑；达标后 C100 adaptive。

## 2026-08-06 19:28 generic adaptive INR 未过（差 0.01/0.11/0.58）

- **观察**：INR affine r10 + adaptive final=79.13、AvgAcc=82.86、Forgetting=6.84，三个门槛（79.14/82.97/6.26）均未过；比普通 affine r10（79.29/83.00/6.61）全面变差。
- **判断**：按 prototype 在 transport 输入方向上的投影缩放补偿强度没有泛化收益；adaptive 作为 generic 变体关闭。INR-first 规则下不跑 C100 adaptive。
- **下一步**：method_revision_sd.md 已先行定稿新方向——Gauge-Aligned Cumulative Shared-A；LRPT 只保留为 residual correction。

## 2026-08-06 20:09 训练期原型一致性正则 INR 未过

- **观察**：`sa_prototype_consistency_weight=0.1`（EMA 余弦，commit fcc8882）INR final=78.41、AvgAcc=81.91、Forgetting=7.45；低于原始 SD-LoRA 与 EXP-009。训练日志正常（一致性项 0.03–0.05、无 NaN），Task 8 76.05 为各变体最低。
- **判断**：训练期把当前批次特征拉向 EMA 原型会牺牲新任务可塑性，正则不是漂移补偿的替代；记录为负结果，不跑 C100。
- **下一步**：两项结果与既有 rank/damping/raw/classmean/JVP/operator 一起关闭 LRPT 局部微调路线，开始实现 method_revision_sd.md 的 Phase A（累计 B 纯代数等价性）。

## 2026-08-06 20:15 工作树清理与 Phase A 启动

- **动作**：`git stash` 丢弃工作树中未使用的 `lrpt_adaptive_min/max` 参数化（stash@{0}，可恢复；无配置引用，属于已关闭 adaptive 路线）；`git status` 干净。
- **计划**：Phase A 在 `backbone/sa_lora.py` 增加 `fold_cumulative_up_projection` 等纯函数，新增等价性单测（算子/feature/logits < 1e-5），通过后单独 commit；随后更新三份文档。

## 2026-08-06 20:35 Phase A 完成并提交

- **实现**：`fold_cumulative_up_projection`（单分支折叠）与 `fold_all_cumulative_up_projections`（全 Q/V 分支），公式 `H = sum_i s_i B_i / (||A|| ||B_i||)`，与现有 bank forward 的 scale/normalization 口径逐项一致。
- **验证**：新增 `tests/test_sa_cumulative.py` 5 个用例：算子等价（<1e-5）、真实 `_SharedAQKV` feature 等价（<1e-5）、tiny ViT logits 等价（<1e-5）、与 `save_merged_lora` 产物一致（atol 1e-6）、输入校验；全量 32 passed。
- **提交**：commit `c666ac0`（代码+测试）；工作树仅剩文档改动。
- **判断**：Phase A 证明固定 A 下累计 B 是精确代数等价，不是近似；Phase B 可以放心把 artifact 从 O(T) 个 B 文件改为单累计 B。
- **下一步**：Phase B——`SA_STATE_VERSION` 升级、在线累计保存/加载、Task 0 后即合并、显式迁移脚本、DDP 无重复累计验证。

## 2026-08-06 20:50 Phase C 代数纯函数完成（gauge alignment）

- **实现**：`canonical_down_projection`（QR 薄分解）、`canonicalize_effective_up_projection`（吸收 R^T）、`gauge_align_up_projection`（H_old Q_old^T Q_new 闭式解）、`gauge_projection_residual`；全部为纯函数，无训练路径改动。
- **验证**：同 span 精确保持、一般情形等于正交投影、残差公式一致；全量 36 passed；commit `02459fc`。
- **判断**：Phase B 状态集成所需的两块代数（累计折叠、canonical 化 + gauge）都已锁定；下一步把 v2 artifact 格式与在线累计训练路径接起来，并在接入前先写 v1→v2 迁移脚本。
- **下一步**：Phase B——`SA_STATE_VERSION=2`（canonical_down + cumulative_up + R + task_id）、`_CumulativeSharedAQKV` 前向、Task 结束后在线累计、旧产物显式迁移、DDP 无重复累计 smoke。

## 2026-08-06 20:15-20:41 Phase B v2 累计状态实现 + DDP smoke

- **实现**：`SA_STATE_VERSION=2` 单文件状态（canonical_down Q^T / cumulative_up H / triangular_r R / task_id）；`_CumulativeSharedAQKV` 前向（历史 H@Q^T 固定 + 当前归一化 LoRA 支路）；保存时 QR canonicalization → `H_old_aligned = H_old Q_old^T Q_new` → 折叠当前任务；v1→v2 显式迁移脚本（备份 `sa_state.pt.v1`，保留 B 文件）；`measure_sa_artifact.py` 支持 v2；`cumulative_gauge` 消融开关。commits `27a6574`/`d7ec60a`/`63b267e`；全量 43 tests passed。
- **失败→修复**：首次 4 卡 smoke 在 task1 报 "artifact is legacy v1"——任务 0 由 `utils/inc_net.get_backbone` 构造时未透传 `cumulative_state`，task0 实际写成 v1 产物；修复 `get_backbone`/`update_network` 透传后重跑。
- **smoke 结果**（1 epoch × 10 tasks，exit=0）：10 任务无 NaN/崩溃；产物无逐任务 B 文件；LoRA 371,040 = 基线 10.07%（减少 89.93%），含 INR 原型 524,640 = 14.23%（减少 85.77%）；`verify_sa_consistency` feature diff = 0.000e+00 PASS；每任务 gauge residual 6 位小数下为 0。
- **判断**：v2 管线的构造/保存/恢复/重建路径可信。gauge residual≈0 提示 1-epoch 下 A 基本不跨行空间；完整训练是否跨出原 span 待观察。1-epoch 指标仅验证管线，不作为性能依据。
- **下一步**：完整 INR cumulative-only（gauge=false）→ INR cumulative+gauge → C100 → residual LRPT 组合；每任务记录 gauge residual 与旧类遗忘的相关性。

## 2026-08-06 20:47 语义修正：v2 当前任务支路对齐 v1（不归一化）

- **观察**：完整 INR cumulative-only task0 训练 acc 89.45，低于 EXP-009 系 task0 的约 92，怀疑 v2 当前任务支路归一化与 v1 的 `scale*B(Ax)` 不一致。
- **动作**：确认 v1 `_SharedAQKV` 当前支路不归一化（历史支路才归一化）；把 `_CumulativeSharedAQKV` 当前支路改为 `scale*B(Ax)`，归一化仅在保存折叠时使用（与 v1 把该任务存入 bank 后的口径一致）；新增 v1/v2 task0 前向等价单测；44 tests passed；commit `a3d3a79`。
- **影响**：此前 1-epoch smoke 与已中断的完整 INR 使用旧口径，仅作工程参考；完整 INR 以本次修正后重跑为准。
- **下一步**：重跑完整 INR cumulative-only → gauge。

## 2026-08-06 21:13-21:40 完整 INR：cumulative-only 与 cumulative+gauge

- **cumulative-only**（20 epoch × 10 tasks，exit=0）：Final 78.78、AvgAcc 81.69、Forgetting 7.28；相对 EXP-009 的 79.34/82.47/7.26 下降 0.56/0.78，触发方法文档 >0.5 停止线。每任务 projection residual=0.000000。
- **判断（停止线检查）**：无 gauge 时保存把旧 H 直接放入新 Q 坐标（H_old @ Q_new^T）；即使 A 只在原 span 内旋转（residual=0 无法捕获），历史算子也被改变。这是表示口径问题，不是训练归一化错误；gauge alignment 正是修这个。
- **cumulative+gauge**（20 epoch × 10 tasks，exit=0）：Final 79.06（≥基线 78.76 ✓、强目标 79.29 差 0.23）、AvgAcc 81.77（EXP-009 差 0.70）、Forgetting 6.82（EXP-009 -0.44、SD-LoRA +1.21）。相对 cumulative-only：Final +0.28、AvgAcc +0.08、Forgetting -0.46。
- **判断**：gauge 修正有效且必要；A 行空间几乎不跨 span，历史算子被精确保持。AvgAcc 缺口来源被定位为新任务支路/深层非线性对旧原型的残余漂移——residual LRPT 正对该对象。gauge 首次运行 Forgetting 改善显著，§10 的"连续两次无改善"停止条件未触发。
- **下一步**：gauge+residual LRPT INR 运行中；预期 Final/AvgAcc 回升到 EXP-009 水平后跑 C100。

## 2026-08-06 22:08 gauge+residual LRPT INR：AvgAcc/F 改善但 Final 回退

- **观察**：Final 78.49（低于基线 78.76 和 gauge-only 79.06）、AvgAcc 82.19（gauge-only +0.42）、Forgetting 6.38（gauge-only -0.44）。LRPT drift_error 0.86–0.88，每任务 gauge 三项诊断均为 0。
- **判断**：LRPT 在累计状态下改善早期任务与遗忘，但最终任务被拉低；单 seed 增益不一致，按 Phase D 规则不保留为主方法组件，保留为消融。主方法取 cumulative+gauge。
- **决策**：INR 最低线（Final ≥78.76）已由 gauge-only 79.06 满足；按 method_revision_sd.md §14，跑 C100 cumulative+gauge 检查第二个数据集；若 C100 Final ≥86.89 则进入多 seed/论文阶段，不再做单 seed 微调。

## 2026-08-06 22:53 C100 cumulative+gauge：Final 87.70，双数据集最低线达成

- **观察**：C100 Final 87.70（基线 86.89 +0.81、EXP-009 88.42 -0.72）、AvgAcc 91.83、Forgetting 8.53；曲线前半段持续高于 EXP-009（T1-T5 高 1-3 分），后半段 T7/T8 略低。gauge 三项诊断均为 0；产物无逐任务 B 文件。
- **判断**：cumulative+gauge 在两个数据集都过最低验收线（INR 79.06 / C100 87.70），状态减 85.77%/87.85%；符合 §14"即使 Top1 不再额外提高也优先进入多 seed 和论文阶段"的条件。
- **下一步**：启动多 seed 队列（INR/C100 × seeds 1/2/3，6 个完整运行，约 3-4 小时）；之后补强基线（SA-LoRA/CL-LoRA 可比划分）、消融（cumulative-only、无 prototype、LRPT 有无）与 FLOPs/吞吐/显存测量，最后进入论文整理。

## 2026-08-06 22:54-23:10 多 seed 队列启动与论文准备

- **运行**：`run_sa_cumulative_multiseed_queue.sh`（INR/C100 × seeds 1/2/3）启动，预计约 3.5-4 小时；运行时每任务记录 gauge 三项诊断（residual/rotation/preservation）。
- **配置**：新增任务长度配置 INR T5/T20/T40（init 40/10/5）与 C100 T5/T20（init 20/5），DataManager 验证任务数正确（commit `c91b32a`）。
- **工具**：新增 `scripts/collect_sa_cumulative_summary.py`，可从日志+产物一键输出 final/avgacc/forgetting/tasks/gauge 诊断/LoRA 参数（commit `65d911e`）；`论文/result_table_single_seed.md` 汇总当前单 seed 结果（commit `d075e56`）。
- **预测量**：CPU 上 C100 主方法产物 FLOPs=2.707e11/forward(batch8)、吞吐 29.4 img/s（CPU，仅参考）；正式 GPU 测量待队列结束后跑。
- **风险**：多 seed 队列约 3.5h，若中途 OOM/网络下载失败需按日志重跑对应 seed；强基线（InfLoRA/CL-LoRA/LoRA-DRS/DGS）尚未实现，论文阶段需评估可复现范围。

## 2026-08-06 23:05-23:52 CUB-200 数据准备

- **下载**：Caltech 原链接需登录；改用 `data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1` 镜像下载成功（1.15GB）。
- **转换**：`scripts/prepare_cub.py` 按官方 train_test_split 构建 `data/cub/train|test`（5994/5794 张、200 类；symlink 指向 `_downloads/cub_extract`）；`DataManager("cub")` 验证 10 任务/200 类通过。
- **配置**：`exps/sa_cumulative_cub_seed1_gauge.json`（与 INR 同超参，cumulative+gauge，无 LRPT）+ 运行脚本，commit `9c2ca2f`；待多 seed 队列结束后运行。
- **多 seed 部分结果**：seed1 INR gauge Final 77.99 / AvgAcc 82.54 / Forgetting 6.77（与 seed1995 的 79.06 不同属正常——seed 决定类序；同 seed 基线 EXP-009 待跑）。

## 2026-08-07 00:09 INR 多 seed 完成

- **结果**（cumulative+gauge，4 seeds）：1995=79.06/81.77/6.82、1=77.99/82.54/6.77、2=78.69/82.01/8.26、3=78.09/82.78/9.60；mean±std Final 78.46±0.51、AvgAcc 82.28±0.47、Forgetting 7.86±1.35。全部 exit=0，产物无逐任务 B 文件，LoRA 371,040。
- **诊断**：科学计数法下每任务 residual/rotation/preservation ≈1e-8~3e-8，历史算子保持近乎精确（此前 %.6f 显示为 0 属于截断）。
- **判断**：seed1995 单点过最低线（79.06），但 4-seed 均值 78.46 略低于 SD-LoRA 基线单点 78.76；需要 EXP-009 同 seed 对照后才能判断相对旧方法的统计关系。Forgetting 方差大（类序敏感）。
- **下一步**：C100 seed1/2/3 运行中；随后 EXP-009 同 seed 队列（`run_sa_baseline_multiseed_queue.sh`）→ 配对 t 检验 → 任务长度/CUB/测量。

## 2026-08-07 02:23 主方法多 seed 全部完成，EXP-009 对照队列启动

- **C100 多 seed**（cumulative+gauge）：1993=87.70/91.83/8.53、1=87.82/91.72/8.93、2=87.80/91.23/8.74、3=88.01/91.28/8.51；mean±std 87.83±0.13 / 91.52±0.28 / 8.68±0.19。全部高于 SD-LoRA 基线单点 86.89，跨 seed 稳定。
- **INR 多 seed**：78.46±0.48 / 82.28±0.37 / 7.86±1.31；seed1995=79.06 达标，其余 seed 略低。gauge 诊断 ~1e-8~3e-8。
- **动作**：启动 `run_sa_baseline_multiseed_queue.sh`（EXP-009 seed1/2/3，INR+C100 共 6 个运行），用于同 seed 配对比较；预计约 2 小时。
- **下一步**：基线完成后算 paired t-test（final/avg/forgetting）→ 更新验收结论 → 任务长度/CUB/GPU 测量 → 论文回填。

## 2026-08-07 07:17 EXP-009 对照队列完成，配对显著性结论

- **EXP-009 多 seed**：INR 1/2/3 = 78.98/79.51/78.61（1995=79.34）；C100 1/2/3 = 87.82/87.90/88.07（1993=88.42）。
- **配对检验（2026-08-07 修订）**：早期按文件名字典序 zip 的错误配对已废弃。`scripts/multiseed_stats.py` 改为按 seed 内连接后重算（n=4）：INR Final -0.65（p=0.026）、AvgAcc -0.52（p=0.009）、Forgetting -0.04（p=0.852）；C100 Final -0.22（p=0.282）、AvgAcc -0.12（p=0.147）、Forgetting +0.01（p=0.938）。**INR Final/AvgAcc 显著，其余不显著**。
- **参数**：LoRA 371,040 vs 2,027,530（-81.7%）；含原型相对 SD-LoRA 减 85.8%（INR）/87.9%（C100）。
- **判断**：主方法相对 EXP-009 在 INR 上有约 0.5–0.65 个点的显著精度代价，C100 与 Forgetting 无显著变化；论文主张改为“O(1) 持久状态 + ~82% LoRA 压缩 + 量化的小幅精度代价”，不再宣称统计等价。seed1995 单点最低线仍满足（79.06）。
- **下一步**：GPU 空闲——跑任务长度 T=5/20/40 与 CUB；随后 GPU FLOPs/吞吐/显存测量；回填论文。

## 2026-08-07 10:42 任务长度消融完成，CUB 启动

- **结果**（cumulative+gauge）：INR T5/T10/T20/T40 = 77.54/79.06/77.03/75.31 Final（Forgetting 8.83/6.82/9.51/12.34）；C100 T5/T10/T20 = 88.06/87.70/85.63 Final（Forgetting 8.77/8.53/10.72）。T=10 附近最优，T 增大遗忘单调恶化；每任务 LoRA 状态恒为 371,040。
- **动作**：`run_sa_cumulative_cub.sh` 启动（CUB-200 seed1，10 任务，cumulative+gauge），验证额外数据集。
- **下一步**：CUB 完成后跑 GPU FLOPs/吞吐/显存与一致性审计；回填论文（含参数量 vs T 曲线）。

## 2026-08-07 11:15 CUB 对照与 GPU 测量完成

- **CUB-200**：主方法 79.79/87.69/14.20 vs EXP-009 71.75/84.93/23.31（Final +8.04、F -9.11）；细粒度数据上 v1 重归一化损害更大，gauge-aligned 累计状态显著更稳（单 seed）。
- **GPU 测量**（batch32）：主方法 INR/C100 与 EXP-009 INR 均为 FLOPs 1.129e12、吞吐 ~412-415 img/s、峰值显存 ~578-579 MiB——推理算子等价；差异在持久状态（-82% LoRA）与训练期复杂度。
- **一致性**：INR seed3、CUB 主方法产物 `verify_sa_consistency` PASS（diff=0）。
- **判断**：论文所需数据基本齐备（多 seed、任务长度、额外数据集、效率、相关性、一致性）；剩余强基线（InfLoRA/CL-LoRA/LoRA-DRS/DGS）未实现，作为 limitations/未来工作。
- **下一步**：写论文初稿（`论文/paper_draft.md`），回填全部结果。

## 2026-08-07 12:00 论文首稿完成

- **产出**：按 research-architect-draft 规范建立 `paper_output/` 全套构件（spine/evidence/claim/citation/blueprints/rationale），并完成 `first_draft/main.md` 首稿（摘要、引言、相关工作、方法、实验、讨论、局限、结论、参考文献）；commit `3b9b4d6`。
- **主张边界（2026-08-07 修订）**：主主张是 O(1) 状态 + ~82% LoRA 压缩 + 小幅、量化的精度代价（INR Final/AvgAcc 显著，C100 与 Forgetting 不显著）+ CUB 单 seed +8.04；不主张统计等价，也不主张全面超越 SD-LoRA 或外部方法。
- **TODO**：引用核实（5+ 条）、训练峰值显存表、图与 LaTeX、可选 CUB 多 seed / ImageNet-A / 外部基线。

## 2026-08-06 23:00 诊断-遗忘相关性初算与诊断精度修正

- **工具**：新增 `scripts/drift_forgetting_correlation.py`（最终精度矩阵 → 每任务遗忘；日志诊断 → Pearson/Spearman）。
- **初算结果**（任务 1-9）：
  - EXP-012 operator drift vs 每任务遗忘：Pearson r=+0.26（p=0.50，弱/不显著）；
  - EXP-012 log-only 控制组 raw drift vs 遗忘：r=-0.73（p=0.025）——漂移大反而遗忘小，说明 drift 不是遗忘主因，可能与新任务塑性共变；
  - gauge+LRPT：residual/rotation/preservation 全部为常量 0（零方差，无法相关），LRPT drift_error r=+0.18（p=0.65，弱）；
  - gauge-only/cumulative-only 运行早于诊断功能上线，无 rotation/preservation 日志。
- **产物核对**：cumulative-only 与 gauge 两档最终状态 Q/H/R 明显不同（operator 相对差 0.95），确认无 gauge 的 H 失真导致参数轨迹分叉，gauge 的改善是真实机制而非日志口径差异。
- **修正**：gauge 诊断日志从 `%.6f` 改为 `%.6e`（避免 0.000000 掩盖小值），后续运行生效（commit 后）；汇总/相关脚本的数值正则兼容科学计数法。
- **判断**：operator/gauge 类诊断与每任务遗忘的相关性弱或零方差；论文相关分析章节应如实报告，并保留"历史算子保持"作为机制证据而非遗忘预测器。

## 2026-08-07 P0-1 完成：seed-keyed 配对统计修复

- **问题**：`scripts/multiseed_stats.py` 旧版按文件名字典序 zip 配对，主方法文件顺序（1995,1,2,3）与 EXP-009（1,1995,2,3）不同，前两个 seed 被错配，旧文档 `p=0.117/0.158` 无效。
- **修复**：脚本重写为从日志 `=> seed:` 显式解析 seed，按 `{seed: metric}` 内连接配对；对缺失/重复/seed 集不一致直接报错；输出逐 pair 明细、paired t、95% CI、Cohen's dz；新增 `--margin` 的 TOST 等价检验（默认 α=0.05）。新增 `tests/test_multiseed_stats.py` 7 个用例（含 1/2/3/1995 词典序陷阱、缺 seed、重复 seed、TOST 边界）；全量 52 passed。
- **重算结果**（n=4，seed 内连接）：INR Final -0.65（p=0.0256，95%CI [-1.15,-0.15]）、AvgAcc -0.52（p=0.0086，95%CI [-0.79,-0.25]）、Forgetting -0.04（p=0.8521）；C100 Final -0.22（p=0.2815）、AvgAcc -0.12（p=0.1469）、Forgetting +0.01（p=0.9375）。
- **TOST（±0.5）**：INR Final/AvgAcc 与 C100 Final 不等价；INR Forgetting、C100 AvgAcc/Forgetting 等价。
- **判断**：INR Final/AvgAcc 是显著小幅下降，论文/实验文档中的“统计等价”“无显著差异（p≥0.117）”全部作废；后续论文主张必须用“O(1) 状态 + 约 0.5–0.65 个点的小幅显著代价（INR）/不显著（C100）+ Forgetting 不变”。
- **同步文档**：`paper_output/`（abstract、4.2 表、结论、claim C5、evidence E-PAIR、spine、blueprints、rationale）、`论文/result_table_single_seed.md`、`论文/paper_skeleton.md`、`experiment_sd.md`、`plan_sd.md`、`method_revision_sd.md`、本文档均已更新。
- **下一步**：P0-2 修复 gauge 诊断生命周期（保存前缓存），加单元测试并重跑一个完整 INR seed。

## 2026-08-07 P0-2 修复：gauge 诊断改为保存前缓存

- **问题**：`models/sa_sdlora.py` 在 `super().incremental_train()` 返回后调用 `cumulative_gauge_diagnostics()`，但父类保存已用新状态覆盖 `cumulative_up/canonical_down`，日志中的 ~1e-8 是“新状态与自身比较”，不能证明历史算子被保持。
- **修复**：`backbone/sa_lora.py` 新增 `_compute_gauge_diagnostics_between()`，在 `_save_cumulative_state()` 覆盖旧状态前用旧 `(H_old, Q_old)` 与训练后新 `Q_new` 计算 residual/rotation/preservation，缓存到非持久字段 `_last_cumulative_gauge_diagnostics`；训练日志改读缓存。`cumulative_gauge_diagnostics()` 保留但注明只用于测试/手动检查。
- **测试**：新增同 span（零 residual/preservation、旋转可测）、正交补空间（residual≈1、preservation≈1）、保存缓存非平凡值且后保存自比较显著小于 pre-save 值三个用例；全量 55 passed。
- **下一步**：提交后重跑 INR seed1995 完整一轮（新目录 `ImageNetR_SA_CUMULATIVE_INR_SEED1995_GAUGE_P0DIAG`），验证真实 pre-save 诊断序列并更新机制结论；旧日志不得再作为机制证据。

## 2026-08-07 P2 实现：Fixed-Rank Union-SVD Cumulative LoRA（v3）

- **动机**：gauge 只在任务结束后把旧算子投影到新 `Q_new` 基底，旧算子的 out-of-span 方向会被丢弃（P0DIAG 重跑首任务即显示 pre-save residual≈4.4e-2，非 1e-8）。Union-SVD 改为先合并完整有效算子 `M = H_old Q_old^T + s B A/(||A|| ||B||)`，再做固定秩最优近似，保留新旧方向的联合空间。
- **实现**：
  - `backbone/sa_lora.py`：`union_svd_factors()`（左右因子 QR + 小核心 SVD，不构造稠密 d×d 矩阵；返回 canonical_down `V^T`、cumulative_up `UΣ`、奇异值、相对截断误差）；`SA_STATE_VERSION=3` + `merge_mode`；`_save_cumulative_state` 支持 `gauge`/`union_svd` 两种模式（union 保存 `triangular_r=I`）；`migrate_sa_state_v2_to_v3()`（显式迁移、备份 `.v2`）。
  - `models/sa_sdlora.py`、`utils/inc_net.py` 透传 `sa_cumulative_merge` / `sa_cumulative_rank`；训练日志输出每任务 `max_relative_truncation_error`。
  - `scripts/migrate_sa_state_v2_to_v3.py`、`measure_sa_artifact.py`/`collect_sa_cumulative_summary.py`/`verify_sa_consistency.py` 支持 v3。
- **测试**：新增 6 个用例（全秩等价、截断误差与显式 SVD 一致、union 误差 ≤ gauge 投影误差、v3 roundtrip 无逐任务 B 文件、v2+union 报错要求迁移、v2→v3 同秩迁移保持算子）；全量 61 passed。
- **配置**：Stage A 的 INR seed1995 四档（union_svd_r4 / gauge_r4 / union_svd_r8 / gauge_r8）与 union_svd smoke 配置、队列脚本；P1 公平任务长度基线配置（EXP-009 INR T5/T20/T40、C100 T5/T20；SD-LoRA INR T20/T40）与队列脚本；离线漂移诊断脚本 `scripts/diagnose_prototype_drift.py`。
- **下一步**：P0DIAG 完整 INR 结束后跑 union_svd smoke（4 卡 DDP 1-epoch），通过后跑 Stage A 四档完整 INR，按门槛决定 C100。

## 2026-08-07 P0DIAG 完成：真实 pre-save gauge 诊断与性能复现

- **结果**（INR seed1995，20 epoch × 10 tasks，exit=0）：Final 78.91 / AvgAcc 81.86 / Forgetting 7.02；对照 EXP-016 gauge（79.06/81.77/6.82）在单次运行噪声范围内一致。
- **pre-save 诊断**（task1–9）：projection residual / preservation 均值约 2.1e-2（范围 1.3e-2–4.4e-2），rotation 1.3e-3–5.5e-2。**结论：真实历史算子保持误差为百分之几，不是旧日志的 ~1e-8**；gauge 只能保留新基底内的部分，旧方向确有丢失。
- **判断**：这使 EXP-016/017 及论文 draft 中的“近精确保持”机制证据作废；精度未明显变化说明该误差不是 final Top1 的唯一决定因素，但它正说明需要 Union-SVD 这类联合子空间最优压缩。P0-2 修复完成（代码 commit `ec23df0`，重跑 EXP-020 记录见 `experiment_sd.md`）。
- **下一步**：union_svd smoke 运行中（已到 task4，truncation_error≈3.4e-3 级别，无崩溃）；随后 Stage A 四档 INR。

## 2026-08-07 union-svd v3 smoke 完成（4 卡 DDP，1-epoch × 10 tasks）

- **结果**：exit=0，10 任务全部完成；Final Top1=66.66、AvgAcc≈70.4（1-epoch 仅管线验证，不参与性能对比）、Forgetting=8.26。每任务 `max_relative_truncation_error` ≈ 0.8e-3–6.5e-3。
- **产物**：仅 `sa_state.pt` + `sa_merged_lora.pt` + `CLs_*`，无逐任务 B 文件；版本 3（union_svd），rank=4；LoRA 参数 147,840 = 基线 4.01%。
- **判断**：v3 在线累计、保存/加载、DDP 无重复累计、无崩溃全部通过；Stage A 完整四档 INR 已自动开始（当前 r4 union_svd 运行中）。

## 2026-08-07 Stage A 关闭与 P1 离线诊断

- **Stage A 四档 INR 结果**（seed1995）：union r4 77.59/81.63/7.59、gauge r4 77.09/81.26/8.20、union r8 78.69/82.31/6.77、gauge r8 78.24/81.98/6.88；补充同秩 union r10 = 78.61/81.91/7.05（gauge r10 79.06/81.77/6.82）。
- **判断**：Union-SVD 在三种秩下一致改善 AvgAcc/Forgetting（r4/r8 全面更优，r10 AvgAcc +0.14），但 Final 均未达到 C100 进入门槛（≥79.10），且 r10 仍低于 gauge_r10 0.45。联合子空间保存补偿了旧类保持，但最终任务缺口依旧；按 plan 停止 rank 扩展，不实现 r12/r16。
- **P1 离线诊断**（`scripts/diagnose_prototype_drift.py`，旧训练数据，仅诊断）：stored prototypes 79.06 → 最终空间重算原型 79.43（+0.37）；冻结 base ViT + base 原型 79.43；base-vs-final 原型余弦 1.0000（归一化原型方向几乎不动）。**结论：原型坐标过期不是主要瓶颈（重算仅 +0.37），backbone 干扰也不主导（base 与 final 同分）**；按 plan 不实现 P3 activation sketch，保留 gauge_r10 作为主方法诚实结果，转入公平任务长度基线与论文收尾。
- **一致性审计**：union r4/r8/r10 产物 `verify_sa_consistency.py` 全部 PASS（feature/prototype logits diff=0）；参数量 147,840 / 296,448 / 371,040（含原型 301,440 / 450,048 / 524,640）。
- **下一步**：P1 公平任务长度基线队列运行中（EXP-009 INR T5/T20/T40、C100 T5/T20；SD-LoRA INR T20/T40）；完成后回填任务长度对比表并更新论文。

## 2026-08-07 训练步峰值显存测量（T=10，INR seed1995）

- `scripts/measure_train_peak_memory.py`：合成 batch32 forward/backward（不含优化器/数据加载）。
- 结果：v2 cumulative+gauge = **3,415.9 MiB**；v1 EXP-009（10 个历史 B bank）= **12,286.9 MiB**（约 3.6× 更低）。
- 产物大小：gauge 8.3 MB vs EXP-009 14 MB（v1 含 10 个 B 文件）。
- 判断：v2 的训练期显存/存储优势可量化；T5/T20/T40 的对应测量待 P1 产物完成后补。

## 2026-08-07 训练步峰值显存（T=5，INR seed1995）

- v2 cumulative+gauge = **3,415.9 MiB**；v1 EXP-009（5 个历史 B）= **7,850.7 MiB**。
- T10 与 T5 对比：v2 恒定 ~3.4 GiB，v1 随 T 增长（5→7.9 GiB，10→12.3 GiB），与 O(T) bank 设计一致。

## 2026-08-07 训练步峰值显存（T=20，INR seed1995）

- v2 cumulative+gauge = **3,415.9 MiB**（与 T5/T10 相同，O(1) 验证）；v1 EXP-009 T20 的测量因 GPU 被 P1 队列占用触发 OOM，待队列结束后补测（预计 ~16.5–17 GiB）。

## 2026-08-07 Live-A Aggregate-B 启动（用户新任务书）

- **优先级**：`live_a_aggregate_b_modification_sd.md` 高于 `protected_union_modification_sd.md`。核心判断：cumulative+gauge 相对 EXP-009 的损失主要来自在线折叠删除了历史分支梯度、共享 A 协同更新和历史 scale 调整，而不是 SVD 没保护新方向。
- **方案**：Live-A Aggregate-B——历史 bank 精确聚合为 `G = sum_i s_i B_i/||B_i||`，训练期历史分支 `G A_t x / ||A_t||` 使用**可训练**的共享 A（与 EXP-009 对 A 的梯度严格等价）；保存时 `G_t = G_{t-1} + s_t B_t/||B_t||`；不保存逐任务 B，状态 O(1)。
- **已完成（队列运行期间，未触碰训练代码）**：
  - 修复 `scripts/diagnose_prototype_drift.py` 的 base-model 原地修改 bug（commit `7d15db6`）；真实 base 诊断待队列结束、GPU 空闲后重跑。
  - 新增 `scripts/migrate_sa_v1_to_live_a_aggregate.py`（commit `549cd71`）：离线把 EXP-009 v1 转为 aggregate G。INR feature diff 8.6e-6 / logit 2.4e-7、C100 feature diff 8.0e-6 / logit 2.6e-7，**离线无损聚合验证 PASS**（≤1e-5）。
  - 新增 freeze-old-scale / live-a-aggregate-b 的 INR/C100 配置与 INR 队列脚本（commit `661c19f`）。
- **约束**：`run_p1_tasklen_baselines.sh` 未结束，未修改 `backbone/`、`models/`、`utils/`；代码实现须等队列完全结束后开始。
- **下一步**：等待队列 → 实现 `live_a_aggregate_b` 模式（v4 artifact）+ `sa_freeze_old_scales` 消融 → bank-to-aggregate 梯度等价测试 → INR seed1995 两档实验。

## 2026-08-07 Live-A Aggregate-B 实现完成（v4 state）

- 用户已要求终止 P1 队列并优先验证新方法；`run_p1_tasklen_baselines.sh` 已终止（C100 T20 未完成，GPU 全部释放）。
- **实现**（`backbone/sa_lora.py`）：`SA_STATE_VERSION=4` + `merge_mode="live_a_aggregate_b"`；新增 `_LiveAAggregateQKV`（历史分支 `G A x/||A||` 使用可训练 live A，当前分支保持 v1 raw `s B(Ax)`）；`_save_live_a_state` 把 `G_next=G_old+sB/||B||` 写入磁盘，但内存保留旧 G + 当前 raw B（非最终任务评估语义与 EXP-009 一致）；`save_merged_lora` 输出 `G_total/||A||`；新增 `migrate_sa_state_v1_to_v4`；旧 v1/v2/v3 artifact 遇到 live_a 配置显式报错要求迁移。
- **消融**：`sa_freeze_old_scales=True` 在 v1 bank 重建后冻结全部历史 scale 参数（EXP-009-freeze-old-scale）。
- **测试**：新增 backbone 级 6 项（bank↔aggregate forward/A-grad 等价、task0/多任务 roundtrip、raw-current 语义、final rebuild=merged、旧版本拒绝、v1→v4 迁移保算子）；math 级 12 项；全量 **79 passed**。
- **离线验证**：`migrate_sa_v1_to_live_a_aggregate.py --state-v4` 在 INR 产物上 PASS（feature diff 6.1e-6、logit 3.0e-7），v4 state 可加载续训且 A 可训练。
- **下一步**：4 卡 DDP smoke → INR seed1995 `exp009_freeze_old_scale` + `live_a_aggregate_b` → 对比门槛（freeze 与 aggregate 差 ≤0.1、Live-A Final ≥79.10/AvgAcc ≥82.30/F ≤7.50）→ C100。

## 2026-08-07 Live-A 4 卡 DDP smoke 完成

- 配置：`exps/live_a_aggregate_b_smoke_inr_seed1995.json`（1 epoch × 10 tasks，rank10，prototype off）。
- 结果：exit=0，10 任务完成（Final 64.7 仅管线验证）；产物仅 v4 `sa_state.pt` + merged + CLs，无逐任务 B；LoRA 参数 368,640 = 基线 10.00%。
- 一致性：`verify_sa_consistency.py` PASS（feature diff 7.2e-6）。
- 下一步：完整 INR seed1995 两档（freeze-old-scale → live-a-aggregate-b）已启动。

## 2026-08-07 Live-A INR 首轮结果与实现诊断

- **freeze-old-scale**（INR seed1995）：79.38 / 82.16 / 7.11（对照 EXP-009：Final +0.04、AvgAcc -0.31、F -0.15）。
- **live-a-aggregate-b**（INR seed1995）：79.64 / 82.10 / 6.67（对照 EXP-009：Final +0.30、AvgAcc -0.37、F -0.59；对照 gauge：Final +0.58、AvgAcc +0.33、F -0.15）。产物 522,240 参数（含原型，减 85.83%），一致性 PASS。
- **门槛**：Final 79.64 ✓、F 6.67 ✓、AvgAcc 82.10 ✗（<82.30 差 0.20），未达任务书 INR 门槛，暂不启动 C100。
- **实现诊断**：新增 `live_a_gradient_diagnostics`（token 输入修复后），task1 首 epoch 历史分支 dL/dA≈5.3e3–5.6e3、当前分支≈3.0e3–3.6e3，ratio≈1.5–1.9——**live A 确实收到历史 bank 梯度**，机制成立；修复前 raw-image 输入导致 shape crash 已修复并提交（`a5e9197`、`6148009`）。
- **复跑**：`live_a_aggregate_b_inr_seed1995_diag2` 完整运行中（每 epoch 首 batch 日志为早期版本，已限制为首 epoch）。
- **下一步**：复跑结果出来后再判定 AvgAcc 门槛（单 seed 方差）与 C100。

## 2026-08-07 Live-A INR 复跑（diag2）完成

- 结果：**79.43 / 81.99 / 7.08**；两次运行（79.64/82.10/6.67、79.43/81.99/7.08）均值 79.54 / 82.05 / 6.87。Final 两次均 > EXP-009（79.34）与 gauge（79.06），Forgetting 6.7–7.1 为变体最优；**AvgAcc 82.0–82.1 仍低于任务书门槛 82.30**。
- 一致性/参数：`verify_sa_consistency` PASS；LoRA 368,640 + 原型 = 522,240（14.17%），无逐任务 B。
- 判定：严格按 `live_a_aggregate_b_modification_sd.md` §10，INR 未达 AvgAcc 门槛 → 不启动 C100；Live-A 机制本身已验证（历史分支梯度恢复、Final/F 优势、O(1) 状态）。下一步选项交给用户：放宽门槛 / 补 seed / 实现 K-group 幅度组。

## 2026-08-07 修正后的原型漂移诊断（true frozen base）

- 修复 base-model 原地修改 bug 后重跑（INR seed1995，gauge 产物）：
  - stored 训练期原型：**79.06**
  - 最终空间重算原型：**79.43**（+0.37）
  - **true frozen base ViT + base 原型：63.35**（旧结论 79.43 作废）
  - base-vs-final 原型余弦：mean=0.831（0.678–0.945）
- 判断：merged backbone 明显优于冻结 base（+16 分），说明 LoRA 适应有效；原型坐标过期只贡献 +0.37，不是主瓶颈；旧“base=final、cosine=1.0”结论因脚本 bug 无效。

## 2026-08-07 Live-A Stage A：CIFAR-100 seed1993 通过

- 用户新目标 `goal_live_a_sd.md` 将 Stage A 改为直接运行 CIFAR-100 seed1993（不再被 INR AvgAcc 门槛阻塞）。
- 20:05 启动 4 卡 DDP（`exps/live_a_aggregate_b_c100_seed1993.json`，20:50 exit=0），全量测试 81 passed。
- 结果：**88.32 / 91.99 / 8.19**（Final ≥88.10 ✓、AvgAcc ≥91.70 ✓、Forgetting ≤8.70 ✓）。产物 LoRA 368,640 + 原型 76,800 = 445,440；无逐任务 B；`verify_sa_consistency` PASS（feature 7.9e-6 / logit 2.7e-7）。
- 对照：EXP-009 88.42/92.07/8.08；SD-LoRA 86.89/91.44/5.58。Live-A C100 单 seed 全面达到 Stage A 门槛。
- 下一步：Stage B 多 seed 配对。已有 EXP-009 INR/C100 × 4 seeds；Live-A 需补 seed1/2/3（INR+C100）；SD-LoRA 需补 seed1/2/3（INR+C100）。

## 2026-08-07 Stage B 配对队列已启动

- 为满足“相同代码版本 + 独立输出目录”，除 Live-A 补跑 seed1/2/3（INR+C100）外，EXP-009 与原始 SD-LoRA 也按当前 HEAD 各补/重跑 4 seeds（INR+C100），全部使用新输出目录。
- 队列：`run_stage_b_paired_queue.sh`（22 个完整 4 卡 DDP 运行，顺序执行），20:53 启动，首个 `live_a_aggregate_b_inr_seed1` 已进入 Task 0。
- 预计总时长约 10–15 小时；完成后按 seed 内连接做配对统计（mean/std、paired t-test、95% CI、Cohen's dz、TOST ±0.5）。

## 2026-08-08 Stage B 途中修复：SD-LoRA 基线的 Live-A 诊断调用

- Live-A 与 EXP-009 队列（14 个运行）全部正常完成；进入原始 SD-LoRA 基线时，`models/sdlora.py` 在 task0 无条件调用 `backbone.live_a_gradient_diagnostics(tokens)`，而 `LoRA_ViT_timm` 无该方法，导致 SD-LoRA INR seed1995/1/2/3 与 C100 seed1993/1 立即崩溃（status=1，日志保留为 `*.failed.log`）。
- 修复：`models/sdlora.py` 增加 `hasattr(backbone, "live_a_gradient_diagnostics")` 保护；全量测试 81 passed。该修复仅影响非 Live-A 模型，Live-A/EXP-009 路径不变。
- 为避免复用含 task0 部分产物的 rerun1 目录，SD-LoRA 8 个基线配置改为 `*_PAIRED_RERUN2` 独立输出目录；新增 `run_stage_b_sdlora_queue.sh` 只跑这 8 个基线。
- 下一步：重新启动 SD-LoRA 队列并监控；完成后汇总 Stage B 全部 22+8（实际 14 正常 + 8 重跑）配对结果。

## 2026-08-08 Stage B 配对完成：INR AvgAcc 门槛未过

- SD-LoRA 8 个配对基线（rerun2）全部 exit=0；Live-A/EXP-009 14 个运行此前已完成。`scripts/stage_b_stats.sh` 输出 `stage_b_stats_output.txt`。
- 汇总：C100 相对 EXP-009 三项通过（TOST ±0.5 等价）；INR 相对 EXP-009 的 AvgAcc 平均差 **-0.374**（p=0.007，未过 -0.30 门槛），Final 平均差 -0.285（擦线）。相对 SD-LoRA：INR Final +0.187、AvgAcc **-0.973**；C100 Final **+1.202**、AvgAcc -0.074、Forgetting **+2.472**。
- 判定：Stage B 未完全通过；C100 Final 显著优于 SD-LoRA，但 INR AvgAcc 的结构性损失不允许宣称全面优越。按目标文件进入 Stage B2 的 evaluation-only 双头诊断路径（暂不训练，不引入回放/测试集选择）。

## 2026-08-08 Stage B2 evaluation-only 双头诊断（final-space）

- 说明：Live-A artifact 只持久化最终骨干与最终 prototype；逐任务原型可由最终 `sa_prototypes.pt` 按类子集精确重建（旧原型在无 LRPT 时逐任务原样保留），逐任务 FC 头已保存。因此本诊断使用最终 merged backbone + 逐任务 FC + 逐任务 prototype 子集，属于 final-space 诊断（非逐任务骨干快照），已在脚本注释与文档中标注限制。
- 脚本：`scripts/diagnose_live_a_dual_head.py`；温度在“当前任务训练类”上用训练标签网格拟合（禁止测试标签）。
- C100 seed1993（训练日志 proto AvgAcc=91.985）：fc AvgAcc=91.417、proto final-space=92.635、fusion A=92.857、fusion B=92.802；相对训练日志 prototype，A/B 提升约 **+0.87/+0.82**。
- INR seed1995（训练日志 proto AvgAcc=81.99）：fc AvgAcc=83.776、proto final-space=84.475、fusion A=84.375、fusion B=84.398；相对训练日志 prototype，A/B 提升约 **+2.39/+2.41**。
- 判定：两个预注册调度在双数据集诊断中相对当前训练日志 prototype 的提升均 ≥0.7，触发正式双头训练实现（目标文件 §6.2 第 3 步）。注意：final-space 诊断本身不作为最终验收，正式双头重跑为准。
