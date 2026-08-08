# P0 确定性与训练轨迹审计修复指引

更新时间：2026-08-09

## 1. 目的

本文件只修复当前 control / Dual-B smoke 的实验污染与不可复现问题，不增加新方法，不运行完整 ImageNet-R/CIFAR-100 实验。

P0 的唯一验收目标是：

> 在相同 seed 和配置下，两个独立 control 的训练轨迹完全一致；Dual-B 的校准与评估不改变后续 LoRA 训练轨迹。

## 2. 已确认的问题

1. 多次 smoke 复用 `CF100_LIVE_A_RNG_SMOKE_CONTROL_D2`。
2. `SharedALoRA_ViT_timm` 在初始化时会读取已有 `sa_state.pt`，导致新任务从旧 Aggregate-B 状态继续训练。
3. 历史聚合范数 `G` 在重复运行中从约 `3.04` 持续增长到 `4.79`，证明当前重复实验已被旧 artifact 污染。
4. 曾同时启动两个四卡任务并写入同一个 `filepath`，存在 checkpoint 竞争覆盖。
5. 当前确定性设置使用 `warn_only=True`，日志仍报告 memory-efficient attention backward 非确定性。
6. 现有 trajectory hash 直接计算 `torch.save` 文件字节，无法定位究竟是哪个 tensor 发生变化。
7. 运行过程中修改了代码和配置，日志、artifact 与 Git commit 无法一一对应。
8. P0 的 `85 passed` 只覆盖初版 RNG helper，尚未覆盖 fresh-dir、并发锁、完整 DDP broadcast 和真实训练轨迹。

因此，所有复用 `D2` 得到的 smoke 日志和 artifact 必须标记为 `INVALID`，不得进入实验表。

## 3. 立即停止项

在修改代码前：

1. 终止所有仍在运行的 `rng_smoke` torchrun 进程。
2. 不删除旧日志；建立 `INVALID_RUNS.md`，记录日志名、失败原因和对应时间。
3. 不再写入 `CONTROL_D2`、`DUAL_B_D2`。
4. 提交或暂存当前代码修改，确保开始修复时工作树状态明确。

## 4. 代码修改要求

### 4.1 `trainer.py`：在构造模型前设置确定性

把完整 seed 和 deterministic backend 设置放在 `DataManager`、模型和 DataLoader 创建之前：

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

`CUBLAS_WORKSPACE_CONFIG=:4096:8` 必须在启动 Python 前由脚本导出。

若严格模式报出非确定算子，应替换或关闭该算子；禁止退回 `warn_only=True`。

### 4.2 `backbone/sa_lora.py`：阻止 task0 自动加载旧状态

保留同一次增量训练中 task1 以后读取前一 task 状态的行为，但增加 fresh-run 保护：

- `cur_task_index == 0` 且输出目录存在 `sa_state.pt`：直接抛出 `FileExistsError`；
- `cur_task_index > 0`：允许加载本次运行刚保存的状态；
- 后续若需要恢复训练，必须实现显式 `resume=true` 和 run manifest，不能默认恢复。

不得通过自动删除旧目录规避检查。

### 4.3 启动脚本：唯一目录与运行锁

每次运行使用独立目录，例如：

```text
CF100_P0_CONTROL_CLEAN_R1/
CF100_P0_CONTROL_CLEAN_R2/
CF100_P0_DUAL_B_CLEAN_R1/
```

启动前要求：

- 目标目录不存在；
- 使用 `flock` 或等价机制锁定 `${filepath}.lock`；
- 同一 `filepath` 已被占用时立即退出；
- 保存 Git commit、完整配置 SHA-256、启动命令和 run ID；
- 训练期间不得修改加载中的代码或配置。

### 4.4 `models/sdlora.py`：固定数据顺序

- audit 模式使用 `num_workers=0`；
- train/test/calibration loader 分别使用固定 seed 的独立 `torch.Generator`；
- DDP 的 `DistributedSampler` 每个 epoch 显式调用 `set_epoch(epoch)`；
- control 与 Dual-B 除 `filepath`、日志前缀和 dual-head 开关外，配置必须完全一致。

### 4.5 `utils/rng_utils.py`：修正 RNG 保存范围

- 保存/恢复 Python、NumPy、Torch CPU 和当前 DDP rank 所属 CUDA device 的 RNG；
- 不让一个 rank 创建或修改其他 rank GPU 的 CUDA RNG context；
- 在 calibration/eval 前后分别计算 RNG state hash，并断言完全一致。

### 4.6 `models/sa_sdlora.py`：同步状态与单遍评估

- 保持 fused/FC/prototype 三头单次 DataLoader 遍历；
- rank0 计算 prototype 和 `lambda/tau` 后，将完整 prototype tensor 与校准参数 broadcast 到所有 rank；
- 所有 rank 调用相同的 `set_prototypes` / `set_dual_head`；
- 校准前后断言 LoRA、FC、prototype、buffer 的 `max_abs_diff == 0`；
- 最终 `lambda=1` 时断言 fused logits 与 prototype logits 的最大差不超过 `1e-6`。

## 5. 使用 canonical tensor hash

新增 tensor 级 hash，不再只比较文件字节。hash 输入至少包括：

- tensor 的完整键名；
- shape 和 dtype；
- contiguous CPU tensor 原始数据；
- LoRA A/B 或 Aggregate-B G；
- scale、FC、prototype 和必要 buffer。

分别记录：

1. `post_train_hash`：当前 task 训练刚结束、任何校准/评估之前；
2. `post_eval_hash`：校准与评估之后；
3. `rng_hash_before_eval` 与 `rng_hash_after_eval`。

hash 不一致时，脚本必须输出第一个不一致的 tensor key、shape 和 `max_abs_diff`，不能只输出“hash 不同”。

## 6. 必须补充的测试

1. task0 遇到旧 `sa_state.pt` 必须失败。
2. task1 能加载同一 run 的 task0 状态。
3. 两个进程争用同一 `filepath` 时，第二个必须失败。
4. strict deterministic 配置不再出现非确定性 warning。
5. 完整 calibration/eval 前后 RNG hash 相同。
6. 完整 calibration/eval 前后模型 tensor hash 相同。
7. prototype、`lambda/tau` 在四个 rank 上一致。
8. 三头评估只遍历测试 DataLoader 一次。
9. canonical hash 对相同 tensor 状态稳定，对单元素变化敏感。

测试完成后运行全量 `pytest`，把通过数量和命令记录到 `experiment_note_sd.md`。

## 7. 正确的 smoke 顺序

必须串行运行，前一项失败时停止：

1. `CONTROL_CLEAN_R1`：4 卡、2 tasks、2 epochs。
2. `CONTROL_CLEAN_R2`：相同配置、全新目录。
3. 比较 R1/R2 每个 task 的 `post_train_hash`、指标和 RNG hash，要求完全一致。
4. `DUAL_B_CLEAN_R1`：相同训练配置、全新目录。
5. 比较 control 与 Dual-B：
   - task0 `post_train_hash` 完全一致；
   - task0 eval 后 RNG/参数 hash 不变；
   - task1 `post_train_hash` 完全一致；
   - Dual-B 校准参数四 rank 一致；
   - 最终 fused/prototype logits 一致。

只有以上全部通过，才可写入 `SMOKE PASS`。

## 8. Git 与实验记录

建议拆成三个提交：

1. fresh-run guard、唯一目录和运行锁；
2. strict determinism、DDP 状态同步和 canonical hash；
3. 单元测试、smoke 脚本和审计记录。

每次日志必须记录 commit hash。P0 完成前禁止启动完整 seed1995、强基线、消融或多任务长度实验。

## 9. P0 完成定义

以下条件必须同时满足：

- 无目录复用、无并发覆盖、无非确定性 warning；
- control R1/R2 每 task tensor hash、RNG hash 和指标完全一致；
- control 与 Dual-B 的训练轨迹完全一致；
- Dual-B 仅改变评估 logits，不改变训练状态；
- 全量测试通过；
- 代码、配置、日志、artifact 和 Git commit 可一一追溯。

未满足任一条件时，结论仍为 `P0 FAIL`，不得用 smoke 数值解释方法性能。
