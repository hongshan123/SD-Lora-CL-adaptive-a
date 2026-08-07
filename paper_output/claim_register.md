# Claim Register

| Claim ID | Claim | Evidence | Boundary |
| --- | --- | --- | --- |
| C1 | Shared-A bank 在固定 A 下可精确折叠为单一累计上投影（等价误差 <1e-5） | E-AUDIT-UNIT（Phase A 测试） | 固定 A、按仓库 scale/normalization 口径；不是对训练动态的声明 |
| C2 | 任务结束后闭式 gauge alignment 保持历史有效算子的行空间内部分 | E-AUDIT-UNIT（Phase C 测试）；真实训练诊断待 P0-2 修复后重跑（TODO_VERIFY） | 代数成立；旧日志中的 ~1e-8 诊断因保存后自比较无效，须以 pre-save 诊断为准 |
| C3 | 持久 LoRA 状态与任务数无关：371,040（T=5/10/20/40 均实测） | E-PARAMS-MAIN、E-PARAMS-T、E-TL | 不含优化器/训练临时量；原型计入后仍减 ≥85%（相对 SD-LoRA） |
| C4 | 单模型推理、无 task-id/router/逐任务 adapter、恢复训练无需历史 B bank | E-AUDIT-VERIFY、E-AUDIT-DDP、E-PARAMS | 在本仓库实现与协议下 |
| C5 | 主方法相对 EXP-009：INR Final/AvgAcc 显著小幅下降（-0.65/-0.52，p≈0.026/0.009），C100 无显著差异，Forgetting 无显著变化 | E-MULTI、E-PAIR | n=4，seed 内连接配对；CI 与 TOST 见 E-PAIR；不主张统计等价 |
| C6 | 主方法在 CUB-200 上显著优于 EXP-009（Final +8.04） | E-CUB | 单 seed；作为支持性证据，需多 seed 复核 |
| C7 | 推理期 FLOPs/吞吐/峰值显存与 v1 merged 等价 | E-EFF | batch32、单卡、20 iters；训练期复杂度 v2 更低（未单独计时） |
| C8 | 任务数增加时遗忘单调恶化，但状态保持 O(1) | E-TL | T=5/10/20/40；T=40 时 INR Final 75.31 |
| C9 | operator/gauge 类诊断不是遗忘的可靠预测器 | E-CORR | 任务级相关，n=9；诊断方差小 |
| C10 | LRPT 在 v2 上单 seed 增益不一致（AvgAcc/F 改善、Final 回退），不作为主方法 | E-INR-GLRPT | 单 seed；保留为消融 |
