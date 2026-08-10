# P1 低成本机制诊断汇总（Dual-B 头分解 + prototype 失配/表示遗忘分解）

## 1. 范围与口径

- P1 全部为离线诊断，复用冻结 P3/开发种子 checkpoint 与日志，**没有改变任何训练轨迹、没有新增训练运行**。
- P1.1 使用确认种子（INR/C100 seed1-5）的 Dual-B 日志解析 FC/prototype/fused 三种头，逐任务 top1/top5、AAA 与相对 SD-LoRA/EXP-009 的逐阶段差值；只量化贡献，不按测试结果选择逐任务最优头。
- P1.2 使用开发种子 INR seed1995、C100 seed1993 的最终产物（sa_merged_lora.pt + sa_prototypes.pt）做四种路径分解；旧训练数据仅用于离线 oracle 诊断，不得写入主方法表。

## 2. P1.1 Dual-B 头分解（确认种子 n=5，均值）

### ImageNet-R

| 头模式 | AAA（5 seed 均值） | 相对 SD-LoRA AAA 差值 | 逐任务说明 |
| --- | ---: | ---: | --- |
| FC | 83.020 | +0.059 | 早期（T0-T3）接近 fused，中期 T4-T8 低于 fused |
| prototype | 82.646 | -0.315 | 中早期低于 fused，T5 后与 fused 重合 |
| fused（完整方法） | 83.064 | +0.103 | 与严格 n=5 的 83.06±0.52 一致 |

- fused − SD-LoRA 逐任务（均值）：T0 +0.33、T1 +0.36、T2 +0.10、T3 +0.01、T4 +0.93、T5 -0.67、T6 -0.58、T7 -0.29、T8 +0.10、T9 +0.73；AAA +0.103。
- Dual-B 的贡献：INR 上 fused AAA 比纯 prototype 高 **+0.42**（83.06 vs 82.65），主要来自 T0-T4 的 FC 头部分；T5 之后 lambda 趋近 1，fused 与 prototype 一致。

### CIFAR-100

| 头模式 | AAA（5 seed 均值） | 相对 SD-LoRA AAA 差值 | 逐任务说明 |
| --- | ---: | ---: | --- |
| FC | 90.467 | -0.980 | 中期（T4-T8）持续低于 prototype/fused |
| prototype | 91.021 | -0.426 | T2-T5 低于 SD-LoRA，T9 反超 +0.60 |
| fused（完整方法） | 91.063 | -0.385 | 与严格 n=5 的 91.06±0.64 一致 |

- fused − SD-LoRA 逐任务（均值）：T0 +0.24、T1 -0.29、T2 -0.98、T3 -0.72、T4 -0.78、T5 -0.78、T6 -0.52、T7 -0.15、T8 -0.48、T9 +0.60；AAA -0.385。
- Dual-B 的贡献：C100 上 fused AAA 比纯 prototype 仅高 **+0.04**（91.06 vs 91.02），FC 头在中期明显更差（FC 相对 SD-LoRA -0.98），说明 C100 中期缺口不能被 Dual-B 头生命周期修复，缺口在原型/表示侧。

完整逐任务表和逐 seed 明细：`p1_dual_b_head_decomposition_output.txt`。

## 3. P1.2 prototype 失配与表示遗忘分解（开发种子，仅诊断）

### INR seed1995

| 路径 | Final | AAA（prototype 口径） | Forgetting | old | new |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1. 最终模型 + 保存 prototype（真实路径） | 78.84 | 83.79 | 6.19 | 78.10 | 85.54 |
| 2. 最终模型 + 重算 prototype（oracle） | 80.16 | 84.50 | 4.43 | 79.75 | 83.87 |
| 3. 任务时模型 + 保存 prototype（训练日志） | — | 82.15（逐任务均值） | — | — | — |

- oracle 改善上限：Final **+1.32**、AAA **+0.71**、Forgetting **-1.77**、old **+1.65**、new -1.67。
- 保存 vs 重算 prototype 余弦：mean=0.9478（min 0.8686 / max 0.9998），随任务年龄从 0.9096 升到 0.9478。
- 类内紧致度（重算原型，训练特征）：mean=0.5811；最近错误类 margin（测试）：mean=0.1808。
- 任务时模型 vs 最终模型（同保存原型，逐任务均值）：任务时 82.15 < 最终 83.79（-1.64），**最终模型不是更差**。

### C100 seed1993

| 路径 | Final | AAA（prototype 口径） | Forgetting | old | new |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1. 最终模型 + 保存 prototype（真实路径） | 88.32 | 92.53 | 5.62 | 88.00 | 91.20 |
| 2. 最终模型 + 重算 prototype（oracle） | 89.43 | 93.34 | 3.97 | 89.41 | 89.60 |
| 3. 任务时模型 + 保存 prototype（训练日志） | — | 91.87（逐任务均值） | — | — | — |

- oracle 改善上限：Final **+1.11**、AAA **+0.81**、Forgetting **-1.65**、old **+1.41**、new -1.60。
- 保存 vs 重算 prototype 余弦：mean=0.9622（min 0.8874 / max 0.9999），随年龄从 0.9176 升到 0.9622。
- 类内紧致度：mean=0.6919；最近错误类 margin：mean=0.2236。
- 任务时模型 vs 最终模型（同保存原型）：任务时 91.87 < 最终 92.53（-0.66），**最终模型不是更差**。

### 路径 4（任务时模型 + 重算 prototype）

**不可重构**：Live-A 只持久化最终累计 O(1) 状态（sa_state.pt），逐任务模型快照未保存；P1 纪律禁止重训，因此无法在同一轨迹上计算该路径。本诊断以路径 1/2/3 覆盖决策所需的三个输入。

## 4. 决策判定（任务书 §7.3）

- oracle 刷新（路径 2 − 路径 1）在最终模型上：INR old **+1.65**、Forgetting **-1.77**；C100 old **+1.41**、Forgetting **-1.65**。
- 两个开发种子均满足“旧类准确率提升 ≥1.0 且 Forgetting 降低 ≥0.75” → **prototype 坐标失配是主要瓶颈之一**。
- 任务时模型并未优于最终模型（INR -1.64、C100 -0.66，均为最终模型更好），因此“任务时模型明显更优而重算帮助小”的共享 A 表示遗忘分支不成立。
- 结论：**P1 支持共享 A/历史分支漂移假设**——旧 prototype 与最终特征空间的失配可修复上限大（Final +1.1~+1.3、old +1.4~+1.7、Forgetting -1.7 左右），允许进入 P2 Historical-Branch Activation Distillation；不需要也不允许自行发明其他 transport。

## 5. 生成命令

```bash
python3 scripts/diagnose_dual_b_head.py
source /home/zhaoyang/miniconda3/etc/profile.d/conda.sh && conda activate sdlora
python scripts/diagnose_prototype_representation_drift.py \
  --config exps/inr_p1_livea_dual_b_seed1995_nccl.json \
  --artifact INR_P1_LIVEA_DUALB_SEED1995_NCCL \
  --log inr_p1_livea_dual_b_seed1995_nccl.log --device cuda:0
python scripts/diagnose_prototype_representation_drift.py \
  --config exps/c100_p2_livea_dual_b_seed1993_nccl.json \
  --artifact C100_P2_LIVEA_DUALB_SEED1993_NCCL \
  --log c100_p2_livea_dual_b_seed1993_nccl.log --device cuda:0
```
