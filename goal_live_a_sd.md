# Live-A Aggregate-B 下一阶段研究目标

## 1. 总目标

在不使用旧样本回放、task-id 推理、router 或随任务数增长的 LoRA bank 的前提下，将
**Live-A Aggregate-B** 验证并完善为固定 `O(1)` 持久状态的 rehearsal-free CIL 方法。

最终需要同时证明：

1. 在 ImageNet-R 和 CIFAR-100 上，Final Top1 达到或优于同协议原始 SD-LoRA；
2. 相对 EXP-009，Final/AvgAcc 的平均损失不超过 `0.3`，Forgetting 恶化不超过 `0.3`；
3. 在 ImageNet-R `N=5/10/20` 三种任务长度下不出现明显性能失稳；
4. LoRA 持久参数相对原始 SD-LoRA 至少减少 `80%`，并保持与任务数无关；
5. 所有主结论均由相同 seed、类序、全局 batch、训练轮数和代码版本下的配对实验支持。

只有以上核心条件完成，才能把 Live-A 定为论文主方法。单 seed 最好结果不能单独作为完成
依据。

## 2. 当前起点

Live-A Aggregate-B 当前状态：

- 代码版本：v4 state，最新结果记录 commit `b90dbfe`；
- 单元与集成测试：`79 passed`；
- 4 卡 DDP smoke：通过；
- LoRA 参数：`368,640`，为原始 SD-LoRA 的 `10%`；
- 含 ImageNet-R prototype：`522,240`，相对原始 SD-LoRA 减少约 `85.8%`；
- 无逐任务 B 文件，artifact consistency PASS；
- bank-to-aggregate feature 误差约 `6e-6`，logit 误差约 `3e-7`。

ImageNet-R seed1995 两次完整运行：

| 运行 | Final Top1 | AvgAcc | Forgetting |
| --- | ---: | ---: | ---: |
| Live-A run1 | 79.64 | 82.10 | 6.67 |
| Live-A run2 | 79.43 | 81.99 | 7.08 |
| 两次均值 | 79.54 | 82.05 | 约 6.88 |
| EXP-009 | 79.34 | 82.47 | 7.26 |
| cumulative+gauge | 79.06 | 81.77 | 6.82 |
| 当前协议原始 SD-LoRA | 78.76 | 83.13 | 5.61 |

当前判断：Live-A 已验证历史分支到共享 A 的梯度路径，并稳定改善 Final；但 AvgAcc 比
EXP-009 低约 `0.42`，相对原始 SD-LoRA 的 Forgetting 仍较差。当前只属于“单数据集、
单 seed 机制验证成功”，不是最终方法成功。

## 3. 实验协议原则

主实验统一使用当前 released-code 协议：

- 4 卡 DDP，每卡 batch size `32`，全局 batch size `128`；
- 当前代码实际使用的 SGD、学习率 `0.01`、20 epochs；
- ImageNet-R 主 seed 集与已有配对实验一致；
- CIFAR-100 主 seed 集与已有配对实验一致；
- 所有比较方法使用相同代码提交和数据划分；
- 输出目录必须独立，禁止复用或覆盖旧 artifact。


## 4. Stage A：立即完成第二数据集验证

首先运行 Live-A Aggregate-B 的 CIFAR-100 seed1993，不增加新模块，不调整超参数。

进入多 seed 的最低门槛：

- Final Top1 `>= 88.10`；
- AvgAcc `>= 91.70`；
- Forgetting `<= 8.70`；
- consistency audit PASS；
- LoRA 参数仍为 `368,640`；
- 无逐任务 B、旧数据或 task-id 推理。

决策规则：

- 三项均通过：进入 Stage B；
- Final 只差 `<=0.3` 且其余通过：允许作为诊断进入一个额外 seed，但不得调参；
- Final 低于 `87.80` 或 AvgAcc 低于 `91.40`：停止 Live-A 主线，先分析 C100 失败原因；
- 不得因为看到 C100 结果后修改上述门槛。

## 5. Stage B：多 seed 配对验证

只有 Stage A 通过后运行多 seed。

ImageNet-R 使用与已有主方法/EXP-009完全相同的 paired seeds；CIFAR-100同理。至少完成
4 个相同 seed 的 Live-A、EXP-009 和原始 SD-LoRA结果。若原始 SD-LoRA 某些 seed 缺失，
必须补跑，不能拿单 seed 或论文均值代替。

报告：

- mean、standard deviation；
- 每个 seed 的逐对差值；
- paired t-test；
- 95% confidence interval；
- Cohen's dz；
- TOST，预注册等价区间为 `+/-0.5`；
- Final、AvgAcc、Forgetting三项全部报告。

Live-A 相对 EXP-009 的多 seed 门槛：

- Final 平均差值 `>= -0.30`；
- AvgAcc 平均差值 `>= -0.30`；
- Forgetting 平均差值 `<= +0.30`；
- Final 或 AvgAcc 至少有一项不低于 EXP-009；
- LoRA 状态至少减少 `75%`。

Live-A 相对原始 SD-LoRA 的门槛：

- 两个数据集 Final 平均值均不低于 SD-LoRA；
- AvgAcc 不能显著下降超过 `0.5`；
- Forgetting 若明显恶化，必须在论文中作为代价报告，不能只展示 Final；
- 总附加状态至少减少 `80%`。

若 ImageNet-R AvgAcc 仍稳定低于 EXP-009 `0.3` 以上，不允许仅凭 Final 提升宣称全面优越。

## 6. Stage B2：保持 Final Top1 的 AAA 修复

若 Live-A 在配对实验中继续保持 Final Top1 优于同协议 SD-LoRA，但 AvgAcc/AAA 仍低于
SD-LoRA，则优先修复分类头生命周期，不继续调整 LoRA rank、damping 或聚合机制。

当前 ImageNet-R seed1995 的阶段性证据是：Live-A 两次均值约为 Final `79.54`、AvgAcc
`82.05`，原始 SD-LoRA 为 Final `78.76`、AvgAcc `83.13`。差距主要集中在 Task 0--3，
而最终 Task 9 的 Live-A 约高 `0.78`。因此优先验证“早期 FC、后期 prototype”的双头
渐进融合，而不是改变已经带来 Final 收益的 LoRA 表征。

### 6.1 方法设计

同时维护归一化 cosine-FC 分类头和当前 prototype cosine 分类头：

```text
logits_t = (1 - lambda_t) * calibrated_fc_logits
         + lambda_t * calibrated_prototype_logits
```

- 早期任务以 FC 为主，利用其当前阶段分类能力恢复早期准确率；
- 随已学习任务数增加，单调提高 `lambda_t`，逐渐转向更抗分类偏置的 prototype；
- 最终任务必须固定 `lambda_final = 1.0`，主结果完全使用现有 prototype 路径；
- FC 和 prototype logits 分别进行温度校准，校准参数只能由当前任务训练集或预留验证集
  确定，禁止使用测试标签；
- 不使用样本级 task-id、router、旧样本回放或逐任务 LoRA bank；
- 主表同时报告纯 FC、纯 prototype 和融合结果，不能只报告每阶段三者中的最好值。

第一轮仅允许比较两个实验前预注册的单调调度，例如：

```text
schedule_A = [0.00, 0.25, 0.50, 0.75, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
schedule_B = [0.00, 0.00, 0.25, 0.50, 0.75, 1.00, 1.00, 1.00, 1.00, 1.00]
```

任务数变化时必须把调度定义为归一化任务进度的固定函数，不能为 N=5/10/20 分别查看
测试结果后人工指定切换点。

### 6.2 执行顺序

1. 先对现有 Live-A checkpoint 做 evaluation-only 诊断，在每个任务结束点同时导出 FC、
   prototype 和固定融合调度的 Top1/Top5；不重新训练，不改变 artifact；
2. 检查 FC 或融合头是否能够补回 Task 0--3 的主要损失，并计算在保持最终
   `lambda_final=1` 时的理论 AvgAcc 上限；
3. 只有诊断显示预注册融合调度能提升 AvgAcc 至少 `0.7`，才实现正式双头训练和恢复逻辑；
4. 完整重跑 ImageNet-R seed1995，通过后再运行 CIFAR-100 和 paired multi-seed；
5. 若两个预注册调度都未达到门槛，停止调度扫描，转向每类双 prototype 诊断。

### 6.3 参数和验收标准

ImageNet-R 的 FC 额外状态约为 `200 x 768 = 153,600` 参数。Live-A LoRA、prototype 和 FC
合计约 `675,840`，相对原始 SD-LoRA 的 `3,686,400` 仍减少约 `81.7%`。统计时必须同时
报告 LoRA-only、classifier/prototype 和总持久状态，不能只报告 LoRA 参数。

严格验收标准：

- Final Top1 不低于对应纯 prototype Live-A 配对结果 `0.1` 以上；
- ImageNet-R AvgAcc 达到同协议 SD-LoRA，当前 seed1995 参考值为 `83.13`；
- 若考虑运行波动，等效下限暂定为 SD-LoRA AvgAcc 减 `0.3`，当前为 `82.83`；
- Forgetting 不高于纯 prototype Live-A `0.3` 以上；
- 总附加状态相对原始 SD-LoRA 仍至少减少 `80%`；
- CIFAR-100 和多 seed 结果使用相同调度函数，不得按数据集单独测试后调参。

这一路径属于分类器校准/融合，而不是新的 LoRA 聚合贡献。论文中必须把其收益与 Live-A
主体贡献分开消融。若最终阶段不是纯 prototype，或调度由测试曲线选择，则不能声称
“在保留当前 Final Top1 的同时提升 AAA”。

### 6.4 失败后的唯一后备方向

若双头诊断的上限不足，再尝试每类保存 `K=2` 个 prototype，并用 `max` 或
`logsumexp` 聚合，以覆盖 ImageNet-R 的类内多风格分布。该方向会改变最终分类路径，必须
重新验证 Final，且不能与双头调度同时引入。第一轮禁止全协方差、合成特征回放或 SLCA
式大规模分类器对齐，以免改变 rehearsal-free 和固定小状态的主张。

## 7. Stage C：任务长度稳定性

多 seed 主结果通过后，运行 ImageNet-R `N=5/10/20`。先使用 seed1995 做结构验证，再对
有争议的任务长度补相同 paired seeds。

必须比较：

- Live-A Aggregate-B；
- cumulative+gauge；
- EXP-009；
- 原始 SD-LoRA；
- SD-LoRA 论文 Table 2，仅作为单独的 published reference 行。

最低目标：

| 任务长度 | Live-A Final相对同协议SD-LoRA | Live-A AvgAcc相对同协议SD-LoRA |
| ---: | ---: | ---: |
| N=5 | `>= 0` | `>= -0.3` |
| N=10 | `>= 0` | `>= -0.3` |
| N=20 | `>= 0` | `>= -0.3` |

额外运行 N=40 只作为资源压力测试，不与论文 Table 2 混合。若 EXP-009或SD-LoRA发生
OOM，必须记录：

- 失败任务编号；
- GPU峰值显存；
- 已累积LoRA参数；
- Live-A在相同硬件上是否完成；
- 不得把“基线OOM”写成准确率胜利。

若Live-A在N=5仍比SD-LoRA低超过 `0.5`，必须判定“跨任务长度稳定优越”失败，并分析
大任务内40类对固定rank/shared-A优化的影响。

## 8. Stage D：效率证据

现有合成 forward/backward 显存只能作为辅助结果。方法定型后必须测量真实训练流程：

- 包含真实DataLoader、optimizer、forward、backward和step；
- 相同GPU、batch、AMP设置和任务阶段；
- T5/T10/T20分别测量峰值显存；
- 测量每epoch时间和总训练时间；
- artifact大小和持久参数随任务数的曲线；
- 推理FLOPs、吞吐和峰值显存；
- 明确Live-A的优势主要在训练/恢复状态，而非推理算子。

验收：Live-A训练峰值显存与任务数基本无关；EXP-009/SD-LoRA随任务数增长；测量脚本和
原始输出必须保留。

## 9. 暂时禁止的改进

在Stage A/B结果出来前，不实施以下内容：

- K-group amplitude；
- Protected Union；
- LRPT或prototype transport；
- damping/rank无边界扫描；
- router/task-specific推理；
- replay或保存旧样本特征。

Stage B2 的 evaluation-only 双头诊断不改变训练和 artifact，可提前执行；正式双头实现仍
必须满足 Stage B2 的诊断触发条件。除这一项明确例外外，不得借分类头调度绕过 Stage A/B
的跨数据集与多 seed 验证。

原因：freeze-old-scale相对完整EXP-009的Final没有下降，尚无证据说明历史scale适配是
主要瓶颈；当前最重要的是跨数据集和多seed验证，而不是继续增加组件。

只有满足以下触发条件才能尝试K-group：多seed下freeze-old-scale相对完整EXP-009的
Final或AvgAcc稳定下降超过 `0.3`。否则不得实现。

## 10. 论文协议复现

Stage A--C完成后，再建立独立的paper-protocol配置：

- Adam；
- learning rate `0.008`；
- ImageNet-R 30 epochs；
- global batch size `128`；
- 五次运行；
- N=5/10/20。

当前训练代码存在配置写Adam但主路径硬编码SGD的情况。若修复optimizer选择：

- 必须单独commit；
- 为旧released-code协议保留显式兼容选项；
- SD-LoRA和Live-A必须同时重跑；
- 不得用修复后的Live-A对比修复前的SD-LoRA。

论文最终应分开报告“released-code paired protocol”和“paper-declared protocol”，直到确认
两者可以严格对齐。

## 11. 停止条件

出现以下任一情况，停止继续微调Live-A：

- CIFAR-100单seed明显低于Stage A最低线；
- 多seed下INR/C100 Final均没有优于Gauge或SD-LoRA；
- AvgAcc相对EXP-009稳定低超过 `0.5`；
- N=5和N=20至少两档比SD-LoRA Final低超过 `0.5`；
- 改进必须依赖O(T)状态、旧数据或task-id才能成立；
- 连续两轮新增组件没有达到预注册门槛。
- 双头融合只能依赖测试集选择切换点，或必须令最终 `lambda < 1` 才能提高 AvgAcc。

停止后应保留Live-A作为“固定状态、Final优先”的消融，不再围绕同一机制继续扫参。

## 12. 完成标准

本目标只有在以下事项全部完成后才能标记完成：

- CIFAR-100单seed验证完成；
- INR/C100多seed paired结果完成；
- 若触发 Stage B2，双头诊断、参数统计和预注册调度结果完成；
- 原始SD-LoRA同seed基线齐全；
- N=5/10/20任务长度比较完成；
- 真实训练显存/时间/存储测量完成；
- 所有artifact consistency通过；
- 失败实验和负结果已记录；
- `plan_sd.md`、`experiment_sd.md`、`experiment_note_sd.md`和论文证据同步；
- 最终代码、配置、统计脚本和文档均已提交Git；
- 工作区无未解释的实验代码修改。

## 13. 执行纪律

- 每次实验前读取 `experiment_sd.md`，不得重复已有实验；
- 每个独立实现、诊断修复和实验记录分别commit；
- 不覆盖其他Agent未提交修改；
- 队列运行时不得修改后续进程会导入的训练代码；
- 每次实验结束立即分析，再决定是否满足下一阶段触发条件；
- 禁止在看到结果后追改验收门槛；
- 不得只汇报最好一次运行。

本阶段的核心不是继续提高一个seed的最高Top1，而是验证Live-A能否在约90% LoRA压缩下，
跨数据集、跨seed和跨任务长度保持可重复的竞争性能。
