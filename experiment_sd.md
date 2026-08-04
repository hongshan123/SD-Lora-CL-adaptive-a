# 实验记录（experiment_sd.md）

> 每个实验一行/一节；实验前必须通读本文件，避免重复。

## 模板

```text
## EXP-XXX 实验标题
- 日期：YYYY-MM-DD
- 状态：计划中 / 运行中 / 完成 / 失败
- 目标/假设：……
- 改动：文件 + commit
- 配置与命令：……
- 理论依据/文献：……
- 结果：final Top1 / AvgAcc / Forgetting / 参数量
- 分析：……
- 下一步：……
```

## EXP-000 基线确认

- 日期：2026-08-04
- 状态：完成
- 内容：使用当前代码重跑 SD-LoRA 官方配置（proto_baseline 配置即原始 SD-LoRA，仅改了 batch 与 seed 环境）。
- 结果：
  - CIFAR-100 seed1993：final Top1=86.89，AvgAcc=91.44，Forgetting=5.58。
  - ImageNet-R seed1995：final Top1=78.76，AvgAcc=83.13，Forgetting=5.61。
  - LoRA 参数量：约 3.686M（10 任务 × 24 对 × 2 × 768×10）。
- 结论：以上作为后续所有实验的对照基线。
