# INVALID P0 RNG smoke runs

更新时间：2026-08-09

以下日志和 artifact 全部来自旧 `CONTROL`/`CONTROL_D2` 输出目录的复用、运行中修改代码/配置，或 strict determinism 尚未落地时的诊断运行。它们**不得进入实验表**，也不得作为 P0 验收证据。

| 日志 / artifact | 时间 | 失败原因 |
| --- | --- | --- |
| `CF100_LIVE_A_RNG_SMOKE_CONTROL/` | 2026-08-08 22:29-22:55 | 复用初版 `CONTROL`，无 fresh-run guard、无 run manifest；轨迹 hash 为文件字节 hash |
| `CF100_LIVE_A_RNG_SMOKE_CONTROL_D2/` | 2026-08-08 22:50-2026-08-09 01:28 | 复用 `D2`，旧 `sa_state.pt`/Aggregate-B 状态污染，运行中修改代码，strict determinism 未生效 |
| `CF100_LIVE_A_RNG_SMOKE_DUAL_B/` | 2026-08-08 22:41 | 初版 Dual-B，复用输出目录 |
| `CF100_LIVE_A_RNG_SMOKE_DUAL_B_D2/` | 2026-08-08 23:02 | 复用 `D2`，旧状态污染 |
| `live_a_rng_smoke_control.run1.log` / `run2.log` | 2026-08-08 22:29-22:49 | 复用 `CONTROL`，文件字节 hash |
| `live_a_rng_smoke_control.log` | 2026-08-08 22:50 | `CONTROL_D2` 首轮，非确定 warning、find_unused warning、旧状态污染 |
| `live_a_rng_smoke_dual_b.log` | 2026-08-08 23:04 | `DUAL_B_D2`，旧状态污染、非确定 warning |
| `live_a_rng_smoke_control.det_run1.log` / `det_run2.log` | 2026-08-08 22:50-23:11 | `D2` 复用 + 非确定 warning |
| `live_a_rng_smoke_control.det_zw1.log` / `det_zw2.log` | 2026-08-08 23:11-23:30 | `D2` 复用；诊断性日志 |
| `live_a_rng_smoke_control.det_math1.log` / `det_math2.log` | 2026-08-08 23:28-23:38 | `D2` 复用；SDP/数学后端诊断 |
| `live_a_rng_smoke_control.det_seed1.log` / `det_seed2.log` | 2026-08-08 23:45-00:05 | `D2` 复用；seed 诊断 |
| `live_a_rng_smoke_control.det_tf1.log` / `det_tf2.log` | 2026-08-09 00:01-00:12 | `D2` 复用；TF32 诊断 |
| `live_a_rng_smoke_control.det_nofind1.log` / `det_nofind2.log` | 2026-08-09 00:18-00:29 | `D2` 复用；find_unused 诊断 |
| `live_a_rng_smoke_control.det_gloo1.log` / `det_gloo2.log` | 2026-08-09 00:35-00:46 | `D2` 复用；gloo 后端诊断 |
| `live_a_rng_smoke_control.det_single1.log` / `det_single2.log` | 2026-08-09 00:52-01:03 | `D2` 复用；单卡诊断 |
| `live_a_rng_smoke_control.det_strict.log` / `det_strict2.log` / `det_strict_single1.log` | 2026-08-09 01:13-01:27 | `D2` 复用；strict determinism 落地前诊断 |
| `CF100_P0_CONTROL_CLEAN_R1_PREHASHFIX/` / `R2_PREHASHFIX/` / `DUAL_B_CLEAN_R1_PREHASHFIX/` | 2026-08-09 01:56-02:23 | 首次 clean 目录 smoke；代码在运行后修复了 post_train hash 混入 eval-only lambda/tau 标量的问题，因此该轮不作为最终验收 |
| `live_a_rng_smoke_control_r1.prehashfix.log` / `r2.prehashfix.log` / `dual_b_r1.prehashfix.log` | 2026-08-09 01:56-02:23 | 同上；R1/R2 轨迹一致、无 warning，但 Dual-B task1 的 training-state hash 因 eval 标量混入而不与 control 可比，运行后已修复 |
| `CF100_P0_CONTROL_CLEAN_R1_PRE_RANKSYNC/` / `R2_PRE_RANKSYNC/` / `DUAL_B_CLEAN_R1_PRE_RANKSYNC/` | 2026-08-09 02:25-02:52 | 第二次 clean 目录 smoke；post_train hash 已一致，但运行后补加了四 rank 一致性 all-gather 断言与 fused/proto PASS 日志，旧日志缺少最终验收行 |
| `live_a_rng_smoke_control_r1.preranksync.log` / `r2.preranksync.log` / `dual_b_r1.preranksync.log` | 2026-08-09 02:25-02:52 | 同上；训练轨迹与 RNG hash 全部一致，但缺少四 rank 日志与 fused/proto PASS 日志，按严格一一对应标准标记为 INVALID |

结论：以上运行均不满足 P0 完成定义，一律标记为 `INVALID`。P0 验收只认全新目录下的 `CF100_P0_CONTROL_CLEAN_R1/R2` 与 `CF100_P0_DUAL_B_CLEAN_R1`，并满足运行锁、canonical tensor hash、RNG hash、无 warning、全量测试等条件。
