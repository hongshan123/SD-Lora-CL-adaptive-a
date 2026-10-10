# HOEP 10 seed 扩展与实验恢复（2026-10-09）

## 用户指令与实验范围

恢复停止的实验，将当前 CIFAR-100 消融队列中 **HOEP 主方法**扩展至 10 个 seed，其余方法沿用原来的 seed。cuda6 的 Live-A / Frozen-A 收益来源诊断恢复原队列，保持数据集和 seed 不变。

| 队列 | 方法 | seed | 实验格子数 |
| --- | --- | --- | --- |
| cuda7 CIFAR-100 | HOEP / low-energy | 1993–2002 | 10 |
| cuda7 CIFAR-100 | Frozen、Live、canonical_live、random、high | 每种 1993–1995 | 每种 3，共 15 |
| cuda6 诊断 | ImageNet-R Live / Frozen | 每种 1995–1997 | 6 |
| cuda6 诊断 | CIFAR-100 Live / Frozen | 每种 1993–1995 | 6 |
| cuda6 诊断 | CUB-200 Live / Frozen | 每种 1–3 | 6 |

HOEP 的 7 个新增 seed 预先固定为 1996–2002，不筛选。HOEP 与其余消融的配对比较仍只使用共同的 1993–1995；10 seed 的 HOEP 结果另行汇总，不能当作 10 seed 配对比较。

## cuda7 配置与恢复策略

- 继续使用原双卡协议：T=10、rank=10、每任务 20 epoch、每卡 batch64，全局 batch128。
- HOEP 的 energy budget 仍为 0.05；LS 对齐和精确吸收开启；NormCap、prototype transport、Dual-B、HBD 关闭。
- 三组训练并行：物理 GPU **0/1、2/5、6/7**；排除 3、4。
- 源代码保持与停止前相同，运行前通过原源代码哈希校验。原方法配置仅修改运行目录；新增 HOEP 仅额外修改 `seed` 和 `sa_hoep_partition_seed`。
- 原 seed1993 的 Frozen / Live / HOEP 三个完整结果复用；Frozen1994 完整结果复用。共 25 个格子，复用 4 个结果，需执行 21 个完整训练。
- 原先中断的运行全部留在原目录。新运行使用新目录，从 Task 0 开始；没有声称恢复了中断时的 optimizer / RNG 状态。
- random / high 的 seed1994、1995 继续等待同 seed 的 HOEP 完成，并校验全部 9 个任务、24 个 Q/V 分支的方向数。

新运行目录：

`/home/hongzhijun/hongshan/SD-lora-cl_2/SD-Lora-CL-hoep-cifar-ablation-20261008/.runtime_hoep_cifar25_20261009_10seed`

配置与审计：`manifest.json`、`prepare_10seed.py`、`restart_audit.json`、`frozen_sources.json`。

调度服务：`codex-hoep-cifar25-cuda7-10seed.service`；已启用开机启动，用户 linger 已开启。停止服务会终止整个实验进程组。

启动核查时三组均已进入 Task 0：HOEP seed1994 使用 0/1，seed1995 使用 2/5，seed1996 使用 6/7。调度器无失败记录。

## cuda6 恢复策略

- 保持原单卡 batch128 协议，在物理 GPU0、GPU1 上各运行一个独立任务。
- 复用要求同时满足：训练成功退出事件、完整 10 任务准确率日志、Task9 快照；诊断还要求对应完成事件、JSON 和逐样本 logits 文件。
- 已完成的 12 个“训练及全部诊断”格子跳过。CIFAR Live1993 已完成训练，仅补 Task1–9 诊断；CIFAR Live1994、1995 和 CUB Live1、2、3 继续训练并诊断。
- 已通过的 snapshot on/off smoke 一致性审计复用，原配置和训练实现均保持不变。
- 新恢复脚本：[resume_live_frozen_cuda6.py](../scripts/resume_live_frozen_cuda6.py)。原 `pipeline_cuda6.py` 保留；恢复脚本复用其中的训练调用、环境设置、GPU 空闲检查和日志事件格式。
- 服务：`codex-live-frozen-cuda6-resume-20261009.service`，已启用开机启动。原停止服务不重新启用。

远端运行目录：`/home/zhaoyang/SD-Lora-CL-live-frozen-20261008/.runtime_live_frozen_diagnosis_20261008`。

恢复审计：`resume_20261009_audit.json`；队列日志：`resume_queue_20261009.log`；训练与诊断日志仍写入原 `logs/`。

## 验证

- cuda7 调度相关测试：14 passed。
- 新恢复脚本测试：本机及 cuda6 各 4 passed，覆盖错误复用、缺少最后快照、缺少 logits 及中断归档。
- 当前诊断仓库 CPU 测试套件：619 passed、1 skipped；通过 `CUDA_VISIBLE_DEVICES=''` 运行，避免占用实验 GPU。
- 两个新服务均已核查为 active / enabled。训练最终结果需要待各运行完成后再汇总。
