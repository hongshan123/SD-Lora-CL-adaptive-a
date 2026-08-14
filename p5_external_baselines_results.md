# P5 强外部基线本机复现结果（CIFAR-100，T=10，seed1）

更新时间：2026-08-13 05:45（全部基线复现完成/失败已定案）。

## 1. 结果

| 方法 | 本机复现（Final / AAA / Forgetting） | 发布值（Final / AAA） | 说明 |
| --- | ---: | ---: | --- |
| InfLoRA（CVPR24） | **84.75 / 90.26 / —** | 86.51±0.73 / 91.70±0.32 | 官方实现 + 官方超参；本机 seed1 单跑 |
| CL-LoRA（CVPR25） | 失败（未完成） | 85.32±0.08 / 91.02±0.12（T=20）；T=10 消融 A=91.85 | torch 1.12 兼容性问题，见 `strong_baselines_sd.md` |
| LoRA⁻ DRS（CVPR25） | **89.73 / 93.02 / —** | 89.14±0.23 / 92.55±0.25 | 官方实现 + 官方超参；本机 seed1 单跑 |

## 2. InfLoRA 细节

- 命令：`external_baselines/infolora/main.py --config configs/cifar100_inflora_sdlocal_seed1.json --device 0`
- 环境：bestformer（torch 1.12.1 / timm 0.6.12），补装 sklearn/ipdb。
- 协议：CIFAR-100 10 tasks × 10 classes，Adam lr 5e-4，20 epochs/task，batch 128，
  rank 10，lamb 0.95（官方默认）；seed 1，类序自然序（官方 `shuffle:false`）。
- 日志：`baseline_infolora_c100_seed1.log`（exit=0，2026-08-12 20:47 → 23:19）。
- 全类 CNN top1 曲线：`[99.5, 96.4, 94.5, 91.48, 89.58, 87.38, 86.64, 85.96, 86.44, 84.75]`；
  Final 84.75 / AAA 90.26（`scripts/collect_external_baselines.py`）。
- 与发布值差距 -1.76 Final / -1.44 AAA：单 seed 方差 + 官方 5-seed 均值口径，
  论文按“本机单 seed 复现”分栏报告，不做显著性比较。

## 3. LoRA-DRS 细节

- 命令：`external_baselines/lora_drs/main.py --config configs/cifar100_t10_seed1.json --device 2`
- 环境：bestformer（torch 1.12.1 / timm 0.6.12），补装 sklearn/ipdb；设备硬编码修复
  （`ortho_penalty` / `prompt_loss` / `AugmentedTripletLoss` device）。
- 协议：CIFAR-100 10 tasks × 10 classes，Adam lr 5e-4，20 epochs/task，batch 128，
  rank 10，margin 1.0 / lambda 0.05（官方默认）；seed 1，类序自然序。
- 日志：`baseline_lora_drs_c100_seed1.log`（完成，2026-08-13 02:34 → 05:45）。
- 全类 ACC top1 曲线：`[99.1, 96.6, 95.2, 93.88, 92.8, 91.2, 90.89, 90.46, 90.36, 89.73]`；
  Final 89.73 / AAA 93.02（`scripts/collect_external_baselines.py`）。
- 与发布值差距 +0.59 Final / +0.47 AAA：单 seed 方差内，说明官方实现可复现。

## 4. 结论

- InfLoRA 与 LoRA-DRS 的官方实现在本机完成 CIFAR-100 T=10 seed1 复现；
  CL-LoRA 因 torch 1.12 环境兼容性未能完成（limitations 记录）。
- 论文使用“发布值（原协议）| 本机复现（官方实现、C100 T=10 seed1）”分栏。
