# cuda8/cuda9 联合队列与 CIFAR 结构消融

日期：2026-10-20。用户确认先做 seed1993、10次正式训练。

## 资源与任务归属

- cuda8 GPU4–7：35组从 cuda9 待运行队列划出的对比实验；CL-LoRA 8、CODA-Prompt 7、InfLoRA 3、EWC-LoRA 7、SD-LoRA 10。
- cuda9 GPU0–7：保留80组（划分时4组完成、8组运行、68组等待）。在训任务保持原进程。
- 划分依据是待运行状态、按原队列每3组分出1组；没有按精度或 seed 筛选。
- BiLoRA 的既有36组完整结果继续复用；RanPAC 不在计划内。
- cuda8 GPU0–3：以下结构消融与 Transport 消融。
- cuda7 GPU6：HOEP无NormCap T50诊断训练；GPU7：只读任务边界诊断；训练结束后两卡共同处理剩余诊断。

cuda9 的216格总表保留所有既有结果和外部任务，cuda8 每30秒发送状态及经过原验收器检查的结果。
当前对比任务主目录在两台机器上均为 `/home/hongzhijun/hongshan/cuda9-baselines-multiseed-20261020`。
cuda8 采用 `cuda8_scheduler.py`，限制 GPU4–7；cuda9 采用原调度器。
重新分配前的清单、状态、源码及冻结文件在 cuda9 的 `runtime/before_cuda8_split/` 留档。
环境版本、方法源码与数据划分由原各方法的冻结校验和数据审计验证。

## 消融矩阵

每组分别跑 T=10 和 T=50；CIFAR-100 224预处理，seed1993，rank10，每任务20 epoch，单卡有效batch128，FP32。

| 标识 | A | 历史 B/G | Transport |
|---|---|---|---|
| independent_bank | 每任务独立初始化、训练并保存 | 每任务B及其缩放系数 | 开 |
| frozen_bank | Task0训练后共享冻结 | 每任务B及其缩放系数 | 开 |
| frozen_fold | Task0训练后共享冻结 | 精确折叠G | 开 |
| hoep_fold | HOEP低能量预算5% | 精确折叠G | 开 |
| hoep_fold + transport0 | HOEP低能量预算5% | 精确折叠G | 关 |

全部关闭NormCap和Dual-B。分类头与原型计算一致。LS仅在HOEP折叠组中启用；冻结A无需坐标转换，独立A银行保留原算子。
Transport开/关均在每个任务边界重建部署模型，正式新类prototype均来自同样的当前训练集测试预处理。

### 银行与折叠的对齐语义

银行保存 `s_i B_i A_i`；冻结A银行为 `sum_i s_i B_i A`。不对历史B重新归一化。
这样与关闭NormCap的精确折叠处于同一缩放协议，避免把边界归一化误认为参数结构的收益。
这两个银行控制是匹配结构的消融实现，不能标记为官方SD-LoRA基线结果。
独立A在后续任务使用任务编号决定的新正交初始化；旧A/B/scale冻结。
独立A银行的历史forward通过拼接秩维计算，保存格式仍保留每任务独立因子，不压缩到rank10。
冻结A银行只保存一份共享A，历史B逐项贡献；fold组保持A/G固定大小。

## 显存与验收

末任务合成资源测试：49个非零历史任务，100类分类空间，完整ViT-B/16、224输入、FP32，2次有效batch128 SGD更新及全batch评估。
测试只用于显存与执行路径，不作为准确率结果。

| 结构 | microbatch | 累积次数 | 峰值allocated/GiB | 峰值reserved/GiB |
|---|---:|---:|---:|---:|
| independent_bank | 64 | 2 | 7.37 | 9.15 |
| frozen_bank | 64 | 2 | 6.90 | 8.61 |
| frozen_fold | 64 | 2 | 6.87 | 8.54 |
| hoep_fold | 64 | 2 | 7.31 | 8.99 |

所有组统一microbatch64。每个逻辑batch只有一次SGD更新；HOEP梯度投影及回缩也只执行一次。短尾batch按样本数加权。
正式训练前另跑5组两任务、每任务1 epoch的短程验证，覆盖全部结构及Transport关闭路径。
单位测试覆盖银行保存/重建等价、独立A初始化与历史保持、冻结A银行/精确折叠的前向和梯度等价，以及梯度累积和诊断算子校验。
短程测试发现旧learner在Adaptive-A关闭时仍按strategy调用回缩；已增加明确开关检查，保留失败attempt并重新验证。

服务：`codex-cuda8-hoep-structure-ablation-20261020.service`。
状态：主项目 `.runtime_structure_ablation_cifar_20261020/status.json`。
结果：同目录 `results_summary.json`；原始日志、配置、源码哈希、资源证明、退出证明与checkpoint在各attempt中保存。
只有完整任务曲线、最终评估标志、模型产物及exit0俱全才记为完成。
服务启动时采纳仍活跃的worker；服务器重启后的中断训练保留旧attempt，从Task0另起新attempt，绝不拼接不同训练轨迹。

## 本机诊断恢复

原诊断在服务器重启前只完成Task0–4，Task5中断。没有完整训练循环的断点恢复实现，因此本次从Task0重新启动；旧结果不覆盖。
本次目录：`.runtime_hoep_t50_diagnosis_gpu67_recovery_20261020/`。
当前/后续尝试由 `.runtime_hoep_diagnosis_latest.json` 指向。
诊断服务已设置开机启动；重启后的包装器保留旧尝试并创建新的独立尝试。数值或验证失败不会自动改超参数重试。
以上均为启动与协议记录，不声称这些新实验已经完成，也不预设哪个方法准确率更高。

## 启动验收（23:18 CST）

- 修复后5组短程测试均完成且退出码为0，正式队列已解锁。
- 本地相关单元/回归测试17项通过；cuda8结构及诊断核心测试8项通过。
- cuda8 GPU0–3 已分别开始 independent_bank、frozen_bank、frozen_fold、hoep_fold 的 T50 正式训练；另6组等待。
- cuda8 GPU4–7 正在运行 CL-LoRA/CUB/T10/seed2、CODA-Prompt/CIFAR/T10/seed1995、InfLoRA/CIFAR/T50/seed1995、EWC-LoRA/ImageNet-R/T10/seed1997。
- cuda9 总表已正确接收 cuda8 的4组运行及31组等待状态，216格身份无重复。
