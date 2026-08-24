# P7 T=20 多种子确认实验

## 目标

验证 P5 单开发种子观察到的长任务趋势是否能跨类别顺序复现：完整方法相对
SD-LoRA 的 Final 优势是否在 T=20 仍然成立，以及完整方法能否以 O(1) LoRA
状态取得接近 O(T) EXP-009 的准确率。

## 预注册矩阵

- 数据集：CIFAR-100、ImageNet-R。
- 任务数：T=20。
- 方法：冻结完整方法、原始 SD-LoRA、EXP-009。
- 确认种子：1、2、3。
- 总运行数：`2 x 3 x 3 = 18`。
- 配对单位：同一数据集、同一 seed、同一类别顺序。

## 冻结协议

配置仅从已完成的 P5 T=20 开发种子配置复制，并修改 `prefix`、`seed`、
`filepath` 和显式 `sa_resume=false`。不得依据中间或最终准确率修改：

- 模型结构、rank、LoRA 注入位置；
- 优化器、学习率、scheduler；
- epoch、batch size；
- Dual-B schedule 或 prototype 计算；
- 类别顺序生成规则。

CIFAR-100：5 类/任务，SGD cosine，lr 0.008，20 epochs，batch 32。

ImageNet-R：10 类/任务，SGD constant，lr 0.01，20 epochs，batch 32。

## 执行规则

- 使用 GPU 0,1,2,3 和 4-rank NCCL DDP。
- 所有训练由 `nohup` + `setsid` 串行队列执行。
- 每个运行使用独立输出目录；禁止隐式 resume。
- 每个运行记录 Git commit、配置 SHA-256、开始/结束时间和 exit status。
- 任一运行失败时立即中止后续队列，不跳过失败项。
- 监控进程启动后立即检查，之后每 30 分钟记录进程、GPU、日志错误和磁盘。

## 统计口径

完成后分别对两个数据集报告每个 seed 及 `mean +/- sample std`：

- Final Top1；
- AAA；
- Forgetting。

主要配对比较：

1. 完整方法减 SD-LoRA；
2. 完整方法减 EXP-009。

报告配对均值、95% CI 和双侧配对 t-test。n=3 统计功效有限，因此不以
`p>0.05` 证明等价；EXP-009 的等价性只能在预先给定的容差下使用 TOST 判断。

## 结果解释边界

- 该实验是 P5 开发种子趋势的确认，不用于重新选择方法或超参数。
- 无论结果是否支持原趋势，都必须完整报告 18 个运行。
- 当前论文主方法继续冻结；P7 只补证据，不引入新组件。
