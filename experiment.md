# SD-LoRA-CL 实验记录

最后更新：2026-08-04 21:34 CST

## 记录规范

每个实验记录以下信息：

- 实验 ID、目的、状态和时间；
- 代码提交、配置、数据集、seed 和训练资源；
- 产物目录与日志；
- 主要指标、异常和结论；
- 与其他实验比较时只比较相同协议和相同评价空间。

状态取值：`planned`、`running`、`completed`、`failed`、`invalid`。

## 公共协议

| 数据集 | Seed | 类/任务 | 任务数 | Batch size | GPU |
|---|---:|---:|---:|---:|---|
| ImageNet-R | 1995 | 20 | 10 | 32 | cuda6 0,1,2,3 |
| CIFAR-100 | 1993 | 10 | 10 | 32 | cuda6 0,1,2,3 |

代码提交：

- `89253b4 Add prototype-routed LoRA continual learning experiments`
- `16639bc Add class-prototype Top-2 LoRA routing`

## E000：原始 SD-LoRA 基线

状态：`completed`

目的：建立与后续路由方法公平比较的 global CIL 基线。

配置：

- `exps/sdlora_inr_seed1995_proto_baseline.json`
- `exps/sdlora_c100_seed1993_proto_baseline.json`

结果：

| 数据集 | Final Top1 | Average Accuracy | Forgetting |
|---|---:|---:|---:|
| ImageNet-R | 78.76 | 83.128 | 5.606 |
| CIFAR-100 | 86.89 | 待从日志回填 | 待从日志回填 |

ImageNet-R Top1 曲线：

```text
92.04, 88.20, 86.27, 83.67, 82.22,
81.86, 80.35, 79.65, 78.26, 78.76
```

日志（cuda6）：

- `/home/zhaoyang/SD-Lora-CL/sdlora_inr_seed1995_proto_baseline.log`
- `/home/zhaoyang/SD-Lora-CL/sdlora_c100_seed1993_proto_baseline.log`

结论：后续方法必须使用全局标签空间并超过上述结果，才能宣称优于原始 SD-LoRA。

## E001：Task-Prototype Routed SD-LoRA

状态：`completed`

目的：验证“一任务一 LoRA + 冻结 Base ViT task prototype + hard Top-1”是否可行。

配置：

- `exps/proto_routed_sdlora_inr_seed1995.json`
- `exps/proto_routed_sdlora_c100_seed1993.json`

结果：

| 数据集 | Prototype/global | Oracle/global | Task route Top1 | 原始 SD-LoRA |
|---|---:|---:|---:|---:|
| ImageNet-R | 75.54 | 78.54 | 46.43 | 78.76 |
| CIFAR-100 | 85.57 | 88.45 | 53.98 | 86.89 |

ImageNet-R：

- Average Accuracy：`81.520`
- Forgetting：`6.23`
- All-LoRA/global：`6.60`

日志（cuda6）：

- `/home/zhaoyang/SD-Lora-CL/proto_routed_sdlora_inr_seed1995.log`
- `/home/zhaoyang/SD-Lora-CL/proto_routed_sdlora_c100_seed1993.log`

结论：专家上限接近或高于基线，但单任务平均 prototype 路由精度不足；将所有独立 LoRA 相加会产生严重干扰。

## E002：Class-Prototype 离线复评

状态：`completed`

时间：2026-08-04 19:46-19:54 CST

目的：不重新训练 LoRA，直接使用 E001 的完整专家检查类别 prototype、Top-2 和任务头的潜力。

实现：

- Base ViT 提取每类训练特征和、计数和全局均值；
- 对 query 和 class prototype 做中心化余弦；
- task score 为任务内最大 class score；
- Top-2 联合分数使用 `router_temperature=0.07` 和 `classifier_temperature=1.0`。

结果：

| 模式 | ImageNet-R | CIFAR-100 |
|---|---:|---:|
| class Top1/global | 75.41 | 85.76 |
| class Top1/task-head | 65.62 | 81.22 |
| class Top2/task-head joint | 67.24 | 81.84 |
| oracle/global | 78.54 | 88.45 |
| oracle/task-head | 91.96 | 98.39 |
| base-only/global | 65.82 | 77.83 |
| random/global | 74.59 | 84.27 |
| all/global | 6.60 | 4.52 |
| task route Top1 | 67.34 | 81.96 |
| task route Top2 | 77.26 | 91.38 |

产物（cuda6）：

- `/home/zhaoyang/SD-Lora-CL/offline_class_router_inr_seed1995/`
- `/home/zhaoyang/SD-Lora-CL/offline_class_router_c100_seed1993/`

结论：类别 prototype 显著改善任务召回，但固定温度的任务头联合分数没有把路由收益转化为最终分类收益。

## E003：Class-Prototype 完整增量训练

### E003-A：ImageNet-R

状态：`completed`

时间：2026-08-04 19:58-20:32 CST，退出状态 `0`。

配置：`exps/class_proto_routed_sdlora_inr_seed1995.json`

产物：`/home/zhaoyang/SD-Lora-CL/ImageNetR_CLASS_PROTO_ROUTED_SDLORA_SEED1995/`

结果：

| 模式 | Final Top1 | Average Accuracy | Forgetting |
|---|---:|---:|---:|
| class Top1/global | 75.48 | 81.602 | 6.514 |
| class Top1/task-head | 65.62 | 73.345 | 9.586 |
| class Top2/task-head joint | 67.34 | 75.279 | 9.393 |
| oracle/global | 78.61 | 83.995 | 5.292 |
| oracle/task-head | 92.01 | 92.416 | 0.000 |
| base-only/global | 65.81 | 71.307 | 7.421 |
| random/global | 74.69 | 80.468 | 6.462 |
| all/global | 5.60 | 49.119 | 47.238 |

主返回模式 `class Top2/task-head joint` 的 Top1 曲线：

```text
94.38, 84.40, 79.30, 75.01, 73.65,
72.03, 69.95, 68.97, 67.76, 67.34
```

路由 Top1 曲线：

```text
100.00, 86.00, 80.87, 75.77, 74.01,
72.58, 70.37, 69.06, 67.72, 67.34
```

最终路由：Top1 `67.34%`，Top2 `77.26%`，平均 margin `0.187954`。

关键比较：

- 路由比 E001 提升约 `20.91` 个点。
- class Top1/global 与 E001 只相差 `-0.06`，说明专家选择对 global head 的影响很弱。
- random/global 为 `74.69`，只比正确路由低 `0.79`。
- oracle/task-head 达到 `92.01`，但错误任务头会排除真实标签。

日志：`/home/zhaoyang/SD-Lora-CL/class_proto_routed_sdlora_inr_seed1995.log`

结论：路由识别得到改善，但当前 task-head 联合公式不成立；该结果不优于原始 SD-LoRA。

### E003-B：CIFAR-100

状态：`running`

开始时间：2026-08-04 20:32 CST。

配置：`exps/class_proto_routed_sdlora_c100_seed1993.json`

截至 2026-08-04 21:34：Task 0-4 已完成，Task 5 正在训练。

主返回模式 Top1 曲线（阶段性）：

```text
99.10, 95.30, 93.13, 91.05, 88.74
```

已保存 scale：

```text
Task0 1.256801
Task1 1.195964
Task2 1.134071
Task3 1.201321
Task4 1.191248
```

日志：`/home/zhaoyang/SD-Lora-CL/class_proto_routed_sdlora_c100_seed1993.log`

注意：任务未完成，阶段曲线不能与最终 CIFAR-100 基线比较。

## E004：中心化与对角标准化消融

状态：`completed`（离线路由消融，未改变 LoRA）

时间：2026-08-04 20:55 左右 CST。

结果：

| 归一化 | ImageNet-R Top1/Top2 | CIFAR-100 Top1/Top2 |
|---|---:|---:|
| raw cosine | 67.44 / 76.98 | 82.03 / 91.38 |
| sample-mean centered | 67.34 / 77.26 | 81.96 / 91.38 |
| prototype-mean centered | 67.32 / 77.31 | 81.96 / 91.38 |
| task-mean centered | 67.32 / 77.31 | 81.96 / 91.38 |
| diag standardized | **67.52 / 77.26** | **83.16 / 92.19** |

特征统计：

| 数据集 | 维度 std P90/P10 | 类间有效秩 | 前 10 方向解释方差 |
|---|---:|---:|---:|
| ImageNet-R | 1.281 | 68.54 / 199 | 28.58% |
| CIFAR-100 | 1.445 | 25.32 / 99 | 48.93% |

结论：仅替换公共均值没有明显收益；对角标准化在 CIFAR-100 上值得继续验证。直接完整白化存在秩不足和噪声放大风险。

## E005：SD-LoRA 200 类输出聚合为 10-Task Router

状态：`completed`（ImageNet-R 近似恢复评估）

目的：判断原始 SD-LoRA 能否作为 10 个任务 LoRA 的候选路由器。

限制：原 SD-LoRA 没有保存最终 scale，因此使用默认 scale 恢复，类别 Top1 比原日志低 `0.55`。

结果：

| 指标 | ImageNet-R |
|---|---:|
| 原日志 200 类 Top1 | 78.76 |
| 恢复模型 200 类 Top1 | 78.21 |
| 按任务内最大 logit 的 10-task Top1 | 79.78 |
| 按任务 LogSumExp 的 10-task Top1 | 79.71 |

分析：

- 大多数类别错误同时是跨任务错误，降为 10-way 后只增加约 `1.57` 个点。
- 若独立任务头 oracle accuracy 为 `92.01%`，按近似独立关系估计最终约 `73.41%`。
- 要超过 SD-LoRA `78.76%`，任务路由需要高于约 `85.60%`。
- SD-LoRA 更适合提供 Top-k 候选，而不是直接 hard Top-1 切换任务头。

## 历史 K-CMS/共享 LoRA 实验

以下实验已有日志，但尚未在本文档中系统回填全部指标：

- K-CMS `k=3/k=4`、balance、hardcap、anchor、bwscale；
- `clshared6`、`clshared4`、Task0 shared down projection freeze；
- CIFAR-100 与 ImageNet-R 对应运行。

后续回填时应直接从日志解析，不根据对话记忆补写数值。相关日志位于项目根目录，例如：

- `seed_1995_k4_bwscale_inr.log`
- `seed_1995_k4_clshared6_bwscale_inr.log`
- `seed_1995_k4_clshared4_bwscale_inr.log`
- `k4_clshared6_hardcap_bwscale_c100.log`
