# Adaptive-A 接近最优结果核查

核查日期：2026-10-08。只读取既有配置、原始完整十任务日志及来源记录；没有新增训练或修改方法。

本报告的“接近”指 Final 或 AAA 相对指定对照组逐数据集最佳值的落后不超过 0.30 个百分点，是数值描述，不代表统计等价。最佳值仅限指定本地方法组，不是外部 SOTA。

## 1. HOEP 四路机制配对

T=10、rank10、20 epoch；C100 seed1993、INR seed1995、CUB seed1；双卡每卡64。统一 LS alignment 与精确吸收；关闭 NormCap、prototype transport、Dual-B 和 HBD。配置差异仅方法开关及对应策略参数。HOEP 训练提交6380f77，其余04eb11f；git diff确认两者只差三份Markdown文档，训练代码相同。12条日志均十任务完整且status0，队列status0。

| 方法 | C100 Final / AAA | INR Final / AAA | CUB Final / AAA |
|---|---:|---:|---:|
| frozen | 87.75 / 92.286 | 78.90 / 82.173 | 84.14 / 89.423 |
| live | 87.31 / 91.822 | 78.72 / 82.279 | 76.54 / 87.856 |
| ratio | 87.71 / 92.252 | 79.08 / 82.180 | 84.10 / 89.403 |
| hoep | 87.92 / 92.383 | 78.95 / 82.172 | 84.10 / 89.445 |

相对各数据集最佳 Frozen/Live 端点（两个指标分别取最大）：

| 策略 | C100 ΔFinal / ΔAAA | INR ΔFinal / ΔAAA | CUB ΔFinal / ΔAAA |
|---|---:|---:|---:|
| ratio | -0.040 / -0.034 | +0.180 / -0.099 | -0.040 / -0.020 |
| hoep | +0.170 / +0.097 | +0.050 / -0.107 | -0.040 / +0.022 |

两种策略都满足三数据集接近最佳端点的数值条件；HOEP 未达到原预注册“至少两个数据集提升0.30”的另一项门槛，所以原 NO-GO 判定仍然成立。它只有每数据集单seed，不能声称多seed稳定超过端点。

### Momentum-aware 参考

同基础训练/合并/分类协议，floor=0、gate EMA=0。来自较早实现版本的独立实验，不能冒充与HOEP同提交的第五臂。

| 方法 | C100 Final / AAA | INR Final / AAA | CUB Final / AAA |
|---|---:|---:|---:|
| Momentum-aware | 87.82 / 92.251 | 79.15 / 82.137 | 84.19 / 89.400 |

## 2. 保留 NormCap 与 transport 的功能约束变体

相同三开发seed、T10/rank10/20epoch、双卡每卡64；保留LS、bounded NormCap和transport，Dual-B关闭。Historical、fixed_hybrid和reliability_hybrid配置仅stability signal与输出标识不同。功能半空间HBD是不同约束策略；不同开发提交，按同协议筛选组解释，不是同提交多seed确认。

| 方法 | C100 Final / AAA | INR Final / AAA | CUB Final / AAA |
|---|---:|---:|---:|
| historical | 88.13 / 92.373 | 79.07 / 82.008 | 84.31 / 89.582 |
| fixed_hybrid | 88.17 / 92.351 | 78.87 / 81.960 | 84.31 / 89.541 |
| reliability_hybrid | 88.09 / 92.343 | 79.15 / 82.105 | 84.28 / 89.516 |
| full_logit | 85.33 / 91.251 | 78.68 / 81.777 | 84.31 / 89.578 |
| functional_halfspace_HBD | 88.29 / 92.263 | 79.43 / 82.151 | 83.02 / 89.282 |

相对上述五种功能约束变体分别取最优，Reliability-hybrid 的 ΔFinal 为 -0.200, -0.280, -0.030；ΔAAA 为 -0.030, -0.046, -0.066。三个数据集均接近组内最佳，但这里没有同时配对的完整 Frozen/Live 三数据集端点，不能扩展该结论。

## 3. 多seed与其他候选

- 原始完整Adaptive-A：原单卡batch128三seed Final为87.76 / 79.16 / 83.28；相对同表Live/Frozen最佳87.84 / 79.44 / 83.95，差值−0.08 / −0.28 / −0.67。CUB未达到0.30接近条件。见nodual_core_threeway_results.md。
- Historical-only全部13seed Final为87.918±0.430 / 78.374±0.244 / 83.674±0.642（样本标准差），AAA为91.848 / 82.222 / 88.926。这个单臂多seed结果有助于估计本身稳定性，缺少对应13seed同协议Frozen/Live，不与原单卡三seed表进行配对推断。见goal_historical_only_seed4_13_sd.md。
- 原始Pareto-Knee及RMS修正版已有单seed完整日志，但INR Final分别78.78 / 78.92，CUB为83.81 / 83.71，不能用不同协议的最高单次结果证明三数据集兼顾。
- Risk-budget/relative-risk本机找到的是T3校准，不当作T10结果。
- Recoverability S1--S4没有三数据集完整矩阵；已完整的CUB S4为76.20、AAA85.600，INR S2为76.97、AAA82.291，不能支持统一接近最优结论。
- Functional-HOEP Phase A只是operator-HOEP的shadow诊断复现；并非训练过的新functional策略，Phase B没有启动。

## 4. 本次表格的原始日志

所有路径相对本仓库：

- hoep_p3_c100_frozen_seed1993_t10_e20_bs128_2gpu_20260921_164138.log
- hoep_p3_inr_frozen_seed1995_t10_e20_bs128_2gpu_20260921_164138.log
- hoep_p3_cub_frozen_seed1_t10_e20_bs128_2gpu_20260921_164138.log
- hoep_p3_c100_live_seed1993_t10_e20_bs128_2gpu_20260921_164138.log
- hoep_p3_inr_live_seed1995_t10_e20_bs128_2gpu_20260921_164138.log
- hoep_p3_cub_live_seed1_t10_e20_bs128_2gpu_20260921_164138.log
- hoep_p3_c100_ratio_seed1993_t10_e20_bs128_2gpu_20260921_164138.log
- hoep_p3_inr_ratio_seed1995_t10_e20_bs128_2gpu_20260921_164138.log
- hoep_p3_cub_ratio_seed1_t10_e20_bs128_2gpu_20260921_164138.log
- hoep_p3_c100_hoep_seed1993_t10_e20_bs128_2gpu_20260921_164138.log
- hoep_p3_inr_hoep_seed1995_t10_e20_bs128_2gpu_20260921_164138.log
- hoep_p3_cub_hoep_seed1_t10_e20_bs128_2gpu_20260921_164138.log
- momentum_adaptive_a_c100_t10_seed1993_bs128_2gpu_20260917_214113.log
- momentum_adaptive_a_inr_t10_seed1995_bs128_2gpu_20260917_214113.log
- momentum_adaptive_a_cub_t10_seed1_bs128_2gpu_20260917_214113.log
- c100_fs_historical_seed1993_t10_i4_bs128_2gpu.log
- inr_fs_historical_seed1995_t10_i4_bs128_2gpu.log
- cub_fs_historical_seed1_t10_i4_bs128_2gpu.log
- c100_fs_fixed_hybrid_seed1993_t10_i4_bs128_2gpu.log
- inr_fs_fixed_hybrid_seed1995_t10_i4_bs128_2gpu.log
- cub_fs_fixed_hybrid_seed1_t10_i4_bs128_2gpu.log
- c100_fs_reliability_hybrid_seed1993_t10_i4_bs128_2gpu.log
- inr_fs_reliability_hybrid_seed1995_t10_i4_bs128_2gpu.log
- cub_fs_reliability_hybrid_seed1_t10_i4_bs128_2gpu.log
- c100_function_safe_pareto_seed1993_t10_i4_bs128_2gpu.log
- inr_function_safe_pareto_seed1995_t10_i4_bs128_2gpu.log
- cub_function_safe_pareto_seed1_t10_i4_bs128_2gpu.log
- c100_functional_halfspace_seed1993_t10_bs128_2gpu.log
- inr_functional_halfspace_seed1995_t10_bs128_2gpu.log
- cub_functional_halfspace_seed1_t10_bs128_2gpu.log

结论：HOEP-A和ratio Adaptive-A已有同协议单seed接近Frozen/Live端点的证据；Reliability-hybrid已有完整组件功能约束变体组内接近最佳的单seed证据。尚无某一种Adaptive-A在三个数据集的同协议多seed确认中均稳定接近各自最佳方法的完整证据。
