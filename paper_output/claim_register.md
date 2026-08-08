# Claim Register

| Claim ID | Claim | Evidence | Boundary |
| --- | --- | --- | --- |
| C1 | Shared-A bank 在固定 A 下可精确折叠为单一累计上投影（等价误差 <1e-5） | E-AUDIT-UNIT（Phase A 测试） | 固定 A、按仓库 scale/normalization 口径；不是对训练动态的声明 |
| C2 | 任务结束后闭式 gauge alignment 保持历史有效算子的行空间内部分 | E-AUDIT-UNIT（Phase C 测试）、E-INR-DIAG-P0（pre-save 诊断） | 代数成立；实测 pre-save residual 1.3e-2–4.4e-2（均值约 2e-2），不是精确保持；旧日志 ~1e-8 因保存后自比较无效 |
| C3 | 持久 LoRA 状态与任务数无关：371,040（T=5/10/20/40 均实测） | E-PARAMS-MAIN、E-PARAMS-T、E-TL | 不含优化器/训练临时量；原型计入后仍减 ≥85%（相对 SD-LoRA） |
| C4 | 单模型推理、无 task-id/router/逐任务 adapter、恢复训练无需历史 B bank | E-AUDIT-VERIFY、E-AUDIT-DDP、E-PARAMS | 在本仓库实现与协议下 |
| C5 | 主方法相对 EXP-009：INR Final/AvgAcc 显著小幅下降（-0.65/-0.52，p≈0.026/0.009），C100 无显著差异，Forgetting 无显著变化 | E-MULTI、E-PAIR | n=4，seed 内连接配对；CI 与 TOST 见 E-PAIR；不主张统计等价 |
| C6 | 主方法在 CUB-200 上显著优于 EXP-009（Final +8.04） | E-CUB | 单 seed；作为支持性证据，需多 seed 复核 |
| C7 | 推理期 FLOPs/吞吐/峰值显存与 v1 merged 等价 | E-EFF | batch32、单卡、20 iters；训练期复杂度 v2 更低（未单独计时） |
| C8 | 任务数增加时遗忘单调恶化，但状态保持 O(1) | E-TL | T=5/10/20/40；T=40 时 INR Final 75.31 |
| C9 | operator/gauge 类诊断不是遗忘的可靠预测器 | E-CORR | 任务级相关，n=9；诊断方差小 |
| C10 | LRPT 在 v2 上单 seed 增益不一致（AvgAcc/F 改善、Final 回退），不作为主方法 | E-INR-GLRPT | 单 seed；保留为消融 |
| C11 | 固定秩 Union-SVD 在 r4/r8/r10 均优于/不劣于同秩 gauge（AvgAcc/Forgetting），但 Final 未达 C100 门槛 | E-STAGE-A | 单 seed INR；作为消融，非主方法 |
| C12 | Live-A 可用 O(1) 固定状态获得 INR Final 优势（seed1995 79.43 > SD-LoRA 78.76、C100 88.32 > 86.89），但多 seed 下 INR AvgAcc 相对 EXP-009/SD-LoRA 分别 -0.37/-0.97，不能宣称全面优越 | E-LIVE-INR、E-LIVE-C100、E-LIVE-PAIR-* | n=4，seed 内连接；负结果/消融 |
| C13 | 双头 FC+prototype 融合（预注册 Schedule B）可修复 Live-A 的 INR AvgAcc（相对 EXP-009 +0.44），但 INR Final 相对 EXP-009 -0.53、C100 Forgetting 相对 SD-LoRA +2.83，严格验收未通过 | E-LIVE-DUAL-INR、E-LIVE-DUAL-C100、E-LIVE-PARAMS | 最终 lambda_final=1；温度仅用当前任务训练类；不引入回放/测试集选择 |
| C14 | 每类 K=2 prototype（max/logsumexp）在 INR seed1995 上均低于 K=1 与 SD-LoRA，后备方向失败 | E-LIVE-K2-MAX、E-LIVE-K2-LSE | 单 seed INR；按目标文件未继续 C100/多 seed |
| C15 | Live-A 不作为论文主方法；论文应报告为“固定状态、Final 优先”的消融与负结果，主主张仍为 cumulative+gauge 的 O(1) 状态 + 量化小幅代价 | E-LIVE-A 全部 | 以目标文件 §11 停止条件为准 |
