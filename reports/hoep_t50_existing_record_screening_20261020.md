# 无 NormCap HOEP T=50：已有记录筛查

审计日期：2026-10-20。范围为本机三个已完成 T=50 单 seed 正式运行；未启动训练。

## 当前运行的日志诊断

残差定义为全部 Q/V 分支 LS 后历史算子平方残差之和，除以历史算子能量之和；不是准确率误差。均值覆盖 Task 1–49。

| 运行 | 边界数 | 残差均值 | 残差最大值 | 最大吸收相对误差 | Transport 启用 | 当前任务验证相对 MSE 改善均值 |
|---|---:|---:|---:|---:|---:|---:|
| hoep_cub_transport_nc0_seed1_t50_e20_bs128_20261012 | 49 | 2.034592e-08 | 4.643274e-08 | 1.269820e-07 | 49/49 | 43.8701% |
| recover_hoep_cifar224_nc0_seed1993_t50_speed_20261012 | 49 | 9.873843e-07 | 3.377369e-06 | 1.299742e-07 | 49/49 | 23.5898% |
| recover_hoep_imagenetr_nc0_seed1995_t50_speed_20261012 | 49 | 3.850077e-06 | 5.830656e-05 | 1.344439e-07 | 49/49 | 13.7574% |

这些数据降低了将 LS 投影损失/吸收数值误差作为首要瓶颈的优先级，但小算子误差不能排除分类边界敏感性。Transport 验证来自当前任务样本，不能证明旧类准确率收益。

## 可用性与缺失

- 已有每任务部署准确率、旧/新类准确率、任务分组准确率、HOEP 分区记录、LS 残差、吸收误差、Transport 标量诊断。
- 三个正式输出目录只有最终 sa_state.pt、sa_merged_lora.pt 和 sa_prototypes.pt；另有每任务分类头 CLs_weight/CLs_bias。分类头不能代替逐任务 backbone 状态。
- 最终 sa_state.pt 只含当前累计 G、共享 A 和元数据，没有分离的当前 B、历史 A/G 或输入统计。
- 未保存当前三个运行的 pre_merge、Transport 前后 prototype 对、逐状态 logits。不能直接重建当前 HOEP T=50 的 S0–S6。
- 本机项目范围存在其他配置的 pre_merge 快照，但检索未发现路径标记为 T50 或 task_049 的 pre_merge.pt。
- 可以用最终模型重新提取历史训练集特征，构造仅用于离线诊断的新 prototype，与已存 prototype 比较；这衡量累计 prototype 不匹配，不能单独归因为 Transport。不得纳入正式无回放成绩。

## 已有旧协议的 S0–S6 诊断

来源是 Live-A/Frozen-A、T=10、NormCap/Transport 开启的旧协议，非当前 HOEP。以下对 Live-A 的三个 seed × 九边界等权平均，单位为准确率百分点；不能作为独立因果效应。

| 数据集 | 边界数 | A-only：S0→S1 | 加入 B：S1→S2 | LS：S2→S3 |
|---|---:|---:|---:|---:|
| imagenetr | 27 | +0.1576 | -2.2300 | -0.0143 |
| cifar224 | 27 | +0.0084 | -2.2648 | +0.0108 |
| cub | 27 | -0.2168 | -1.4063 | +0.0464 |

原始汇总：[boundary_rows.csv](/home/hongzhijun/hongshan/SD-lora-cl_2/SD-Lora-CL-adaptive-a/reports/cuda6_completed_20261010/boundary_rows.csv)。A/B 为共同训练结果，关闭 B 是固定状态反事实，不是独立训练干预。

## 下一步

先利用已有最终模型进行 prototype 诊断。完整阶段定位需补跑当前 HOEP T=50，并在同一训练轨迹保存合并前、历史部署状态和 Transport 前后 prototype；不必分别训练每个反事实状态。首先单 seed 验证，再做配对多 seed。

## 当前日志来源

- [hoep_cub_transport_nc0_seed1_t50_e20_bs128_20261012 日志](/home/hongzhijun/hongshan/SD-lora-cl_2/SD-Lora-CL-hoep-taskgrid-20261012/.runtime_hoep_taskgrid_no_normcap_1seed_20261012/logs/hoep_cub_transport_nc0_seed1_t50_e20_bs128_20261012.log)；[Transport 诊断](/home/hongzhijun/hongshan/SD-lora-cl_2/SD-Lora-CL-hoep-taskgrid-20261012/.runtime_hoep_taskgrid_no_normcap_1seed_20261012/results/hoep_cub_transport_nc0_seed1_t50_e20_bs128_20261012/sa_coordinate_diagnostics.json)
- [recover_hoep_cifar224_nc0_seed1993_t50_speed_20261012 日志](/home/hongzhijun/hongshan/SD-lora-cl_2/SD-Lora-CL-hoep-normcap-speed-20261012/.runtime_hoep_normcap_speed_20261012/logs/recover_hoep_cifar224_nc0_seed1993_t50_speed_20261012.log)；[Transport 诊断](/home/hongzhijun/hongshan/SD-lora-cl_2/SD-Lora-CL-hoep-normcap-speed-20261012/.runtime_hoep_normcap_speed_20261012/results/recover_hoep_cifar224_nc0_seed1993_t50_speed_20261012/sa_coordinate_diagnostics.json)
- [recover_hoep_imagenetr_nc0_seed1995_t50_speed_20261012 日志](/home/hongzhijun/hongshan/SD-lora-cl_2/SD-Lora-CL-hoep-normcap-speed-20261012/.runtime_hoep_normcap_speed_20261012/logs/recover_hoep_imagenetr_nc0_seed1995_t50_speed_20261012.log)；[Transport 诊断](/home/hongzhijun/hongshan/SD-lora-cl_2/SD-Lora-CL-hoep-normcap-speed-20261012/.runtime_hoep_normcap_speed_20261012/results/recover_hoep_imagenetr_nc0_seed1995_t50_speed_20261012/sa_coordinate_diagnostics.json)
