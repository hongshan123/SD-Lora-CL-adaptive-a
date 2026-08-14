# Live-A Aggregate-B + Dual-B 实验补全最终审计

更新时间：2026-08-13 05:45（全部完成；DRS 本机复现已定案）。
分支：`p0-strict-n5`；最近提交见各阶段记录。

## 1. P0 可复现性

- [x] 每次训练新输出目录 + resume 显式开关。
- [x] 完整 seed/确定性设置；RNG/校准/评估不变量 hash PASS。
- [x] 两个独立 control 轨迹一致；Dual-B 校准前后 max_abs_diff == 0。
- [x] 30 个 P3 运行 audit PASS（`scripts/audit_p3_runs.py` → `P3 AUDIT PASS`）。
- 证据：`p3_strict_n5_stats_output.txt`、`p3_strict_n5_summary.md`、commit `dae994f`/`c256cc4`。

## 2. P1 机制诊断与冻结

- [x] 离线 oracle 原型重算：INR old +1.65 / F -1.77；C100 old +1.41 / F -1.65。
- [x] 方法冻结为 Live-A Aggregate-B + Dual-B（无 HBD）。
- 证据：`p1_mechanism_diagnostics_summary.md`、`p1_dual_b_head_decomposition_output.txt`。

## 3. P2 HBD

- [x] HBD 实现 + 单测（108 passed）。
- [x] C100 seed1993 开发运行 Final 87.82 < 88.12 → **关闭**，不进入 INR/seed1-5。
- 证据：`p2_hbd_development_result.md`、commit `baeb66c`。

## 4. P3 严格 n=5 主统计

- [x] seed 1-5（INR/C100 × 完整方法/SD-LoRA/EXP-009）。
- [x] 每 seed、mean±std、配对 t、95% CI、TOST ±0.5。
- 结果：INR 79.07±0.23 / 83.06±0.52 / 7.68±1.03 vs SD-LoRA 78.34±0.48 / 82.96±0.87 / 7.79±1.04；
  C100 87.41±0.32 / 91.06±0.64 / 9.09±0.62 vs 86.81±0.56 / 91.45±0.63 / 6.57±0.71。
- 门槛：Final 通过；C100 AAA -0.385（p=0.006）与 F +2.52（p=0.0002）未过 → Pareto 定位。

## 5. P4 消融与同预算基线

- [x] A0-A8 覆盖（C 冻结 A 3-seed、rank1 3-seed、B/D/E/F 由既有运行覆盖）。
- 结果：C100/INR 数据集依赖；rank1 同预算数据集依赖。
- 证据：`p4_ablations_results.md`、commit `ecd57c0`/`40bd2cb`。

## 6. P5 任务长度

- [x] T=5/10/20 × C100/INR × full/SD-LoRA/EXP-009/rank1（16 个 T5/T20 新跑 exit=0）。
- 结果：Final 相对 SD-LoRA 优势随 T 增大（C100 +0.59/+1.32/+1.76；INR +0.38/+0.90/+1.69）。
- 证据：`p5_task_length_results.md`、commit `1e7d884`。

## 7. P5 第三数据集 CUB-200-2011

- [x] 三方法 × 3 seeds（9 runs exit=0，同冻结协议）。
- 结果：完整方法 75.27±3.12 / 84.37±1.34 / 16.77±0.43；EXP-009 75.37±2.90 / 85.01±1.79 / 17.15±0.11；
  SD-LoRA 71.26±1.47 / 81.66±0.90 / 10.97±2.17。
- 结论：完整方法 vs EXP-009 等价（O(1) 状态）；vs SD-LoRA Final/AAA 优势、Forgetting 代价。
- 证据：`p5_cub_results.md`、`scripts/collect_p5_cub.py`。

## 8. P5 强外部基线

- [x] 发布值收集（InfLoRA/CL-LoRA/LoRA-DRS/SD-LoRA 官方论文 PDF）。
- [x] InfLoRA 本机复现完成（C100 T10 seed1：84.75/90.26）。
- [x] CL-LoRA 本机复现失败记录（torch 1.12 兼容性；采用发布值）。
- [x] LoRA-DRS 本机复现完成（C100 T10 seed1：89.73 / 93.02；发布 89.14/92.55）。
- 证据：`strong_baselines_sd.md`、`p5_external_baselines_results.md`。

## 9. P6 效率与 artifact 闭环

- [x] 状态曲线：完整方法 O(1)（368,640 参数、~2.78MB 恒定 T5/10/20）；银行式 O(T)。
- [x] 显存曲线：完整方法 3,468.6 MiB 恒定；SD-LoRA 5,963.9 / 8,186.7 / 12,632.1 MiB。
- [x] 时间曲线：T10 每任务完整方法 1.65× 增长 vs SD-LoRA 3.02×。
- [x] 推理 FLOPs 1.129e12、吞吐 515.3 img/s、峰值 578.4 MiB。
- [x] 最小导出：C100 + CUB，logits max_abs_diff == 0，Final 与日志一致。
- 证据：`p6_efficiency_results.md`、`scripts/measure_state_curves.py`、
  `scripts/export_minimal_artifact.py`、`scripts/verify_minimal_artifact.py`。

## 10. 论文定位

- [x] `paper_output/final_positioning.md`：Pareto 定位（O(1) 状态、Final-参数、C100 AAA/F 代价、
  CUB 结果、强基线分栏、限制）。
- [x] 旧草稿标记为历史证据（`paper_output/first_draft/main.md` 顶部注明）。

## 11. 执行纪律

- [x] 每阶段单独 commit；运行日志记录 commit/config SHA。
- [x] 未覆盖/覆盖旧日志与 artifact；失败目录移出保留（/tmp）。
- [x] 每阶段更新 `experiment_sd.md` / `experiment_note_sd.md`。
- [x] 未按测试结果调参（HBD 按门槛关闭；λ 工程修正仅一次并记录）。

## 12. 最终状态

- 全部 P0-P6 要求完成（详见上表，无 [~] 或未验证项）。
- 工作树干净；所有记录已 commit（最后提交见本文档所在 commit）。
