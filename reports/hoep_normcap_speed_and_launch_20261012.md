# HOEP + NormCap、数值保持的性能优化与后续队列

更新时间：2026-10-12T22:50:17.118580+08:00。

## 实现

代码版本：`2569a223e2ea8cd734fcda26303919eff71e01b9`。在新隔离工作树实现，旧实验代码及结果保留。

HOEP 接受 `bounded_norm_calibrated_absorb` 与精确吸收两种模式，仍拒绝 `normalized_absorb` 和训练分支归一化。
沿用现有 NormCap 公式：`p=(||A||+eps)(||B||+eps)`，`c=min(1,1/p)`，`gamma=s*p*c`，`G_new=G_LS+gamma*(B/(||B||+eps))`。
NormCap 仅校准当前任务写入；历史 LS 的残差界仍独立核验。合并后的总算子不再要求与未合并模型严格相同，部署模型须与实际合并公式一致。
本轮 Prototype Transport 保持开启，能量预算5%，rank10，20 epoch/task，DDP2×64全局batch128，原优化器、学习率、增强、数据顺序不变。

## 性能优化及验证边界

- 确定性测试预处理结果缓存于CPU，LRU张量预算每运行16GiB；不缓存模型特征，也不改变新类prototype的训练数据来源。训练增强不缓存、worker数仍为0。
- 训练数据使用已锁页内存进行异步拷贝；FP32、TF32开关、优化器步数和所有梯度数学保持原样。
- 梯度投影、回缩、秩检查、row-space检查每步执行；昂贵的完整历史/当前算子误差诊断每16步采样，首步也采样。日志明确记录采样次数，采样最大值不冒充全部步的最大值；每个任务边界的LS与历史残差界仍完整执行。
- 通信等待容差600→3600秒；这只改变等待时间，不是已经修复了原NCCL超时根因。每rank每300秒记录Python栈用于定位停顿。
- 测试444项通过（最终独占调度版本）；单元测试逐步比较权重、SGD动量、logits、RNG，并强制触发NormCap校准验证写入及部署公式。
- 三数据集真实数据两任务配对：NormCap关的优化版对旧代码、NormCap开的优化版对新代码未优化版，共6对。两任务post_train/post_eval模型与RNG哈希全部相同；模型存档张量、prototype张量和准确率曲线逐一一致。关版本每任务1epoch，开版本每任务3epoch；此预检不替代完整20epoch实验。

### 测速

组件测速：跳过冗余完整算子诊断的回缩组件约快1.23倍；缓存命中后的图像预处理约快13.40倍。它们不是整段训练加速比。

| 数据集 | NormCap开、未优化两任务耗时(s) | 优化后(s) | 未优化/优化 |
|---|---:|---:|---:|
| cifar224 | 83.93 | 84.96 | 0.988x |
| imagenetr | 73.83 | 67.69 | 1.091x |
| cub | 39.69 | 38.12 | 1.041x |

该短预检包含冷启动，只有3epoch且只在首epoch做训练内评估；CIFAR未体现整体加速。完整20epoch的加速幅度尚未测完，不能据组件倍率承诺整体耗时。

六模型共享卡对的短预检完成，模型/RNG/曲线相同，峰值显存17,721MiB。但本机PyTorch `distributed_c10d.py:1244` 明确要求NCCL进程独占GPU，共享可死锁；正式队列采用三个独占卡对，最大同时3组，不启用该实验性共享方式。

## 队列顺序

第一阶段仅3组，NormCap关：CIFAR T50 seed1993、ImageNet-R T50 seed1995、CUB T5 seed1。两条T50从Task0重跑，旧失败尝试保留。

第二阶段12组：NormCap开、Transport开，CIFAR seed1993、ImageNet-R seed1995、CUB seed1；各T=5/10/20/50。第二阶段每组均依赖第一阶段三组全部完成。

保留先前9组完成结果。第一阶段完成后形成NormCap关12格，第二阶段再形成开12格；当前共排队15个新运行，不筛选seed。

后台服务：`codex-hoep-recovery-then-normcap-20261012.service`。GPU0/1、2/5、6/7，避开3/4。源代码、全部配置、旧完成日志/退出证明均冻结SHA256。

运行目录：[.runtime_hoep_normcap_speed_20261012](/home/hongzhijun/hongshan/SD-lora-cl_2/SD-Lora-CL-hoep-normcap-speed-20261012/.runtime_hoep_normcap_speed_20261012)。

[实时状态](/home/hongzhijun/hongshan/SD-lora-cl_2/SD-Lora-CL-hoep-normcap-speed-20261012/.runtime_hoep_normcap_speed_20261012/parallel/status.json)，[已完成汇总](/home/hongzhijun/hongshan/SD-lora-cl_2/SD-Lora-CL-hoep-normcap-speed-20261012/.runtime_hoep_normcap_speed_20261012/parallel/results_summary.json)，[配对数值验证](/home/hongzhijun/hongshan/SD-lora-cl_2/SD-Lora-CL-hoep-normcap-speed-20261012/.runtime_hoep_normcap_speed_20261012/preflight/equivalence_audit.json)，[实际配置](/home/hongzhijun/hongshan/SD-lora-cl_2/SD-Lora-CL-hoep-normcap-speed-20261012/.runtime_hoep_normcap_speed_20261012/manifest.json)。

## 启动核验

后台服务active/running，第一阶段3组在训，第二阶段12组等待；已保留9组旧完成结果。调度依赖检查通过，在三个补跑全部完成前不会派发NormCap开实验。
- CIFAR-100 T50，GPU[0, 1]：Task 0, Epoch 10/20 => Loss 0.064, Train_accy 97.80:  50%|█████     | 10/20 [00:45<00:44,  4.42s/it]。
- ImageNet-R T50，GPU[2, 5]：Task 0, Epoch 16/20 => Loss 0.118, Train_accy 96.11, Test_accy 92.11:  80%|████████  | 16/20 [00:49<00:12,  3.17s/it]。
- CUB-200 T5，GPU[6, 7]：Task 0, Epoch 5/20 => Loss 0.421, Train_accy 88.17:  25%|██▌       | 5/20 [00:46<02:01,  8.12s/it]。

## 当前进度

2026-10-13 00:23:07 CST。

补跑3组：CUB T5完成（Final85.73%，AAA90.108%）；CIFAR T50失败；ImageNet-R T50继续训练。NormCap开启12组尚未启动，队列处于`draining_failed`，暂停新派发。

CIFAR T50已评估23/50任务，第24个任务15/20epoch后rank0发生SIGSEGV（exitcode=-11），00:07退出。本次具体底层原因尚未定位，不作为最终准确率结果。

ImageNet-R T50在GPU[2, 5]运行；最新日志：Task 36, Epoch 20/20 => Loss 0.082, Train_accy 66.82。

## 2026-10-13 队列重启

用户要求重新拉起训练队列。已归档 CIFAR-100 T50 seed1993 的 SIGSEGV 失败尝试（Task23，Epoch15/20），从 Task0 重跑；没有修改训练代码或科学配置。只停止并重启调度器，ImageNet-R T50 seed1995 的原 worker PID1310396/start_ticks7119789 保持运行并被新调度器接管。CUB T5 完成结果与原九个完成格子保留。

当前队列恢复为 training，无失败标记：CIFAR 使用 GPU0/1，ImageNet-R 使用 GPU2/5；GPU3/4 继续排除。两项补跑完成后自动启动 NormCap ON + Transport ON 的三数据集×四任务长度（5/10/20/50），共12个单 seed 实验。保留每模型 DDP 两卡、每卡 batch64/global128 的现有协议。

新增 PYTHONFAULTHANDLER=1 用于新进程的 fatal signal 堆栈诊断，不改变数值训练逻辑。此前 SIGSEGV 根因尚未确认，重启本身不构成错误修复证据。临时 KillMode=process 已撤销，服务恢复 KillMode=control-group。
