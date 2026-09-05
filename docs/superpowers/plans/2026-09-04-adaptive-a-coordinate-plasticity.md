# Adaptive-A Coordinate Plasticity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, layerwise operator-distortion-aware shared-A gradient gate to the no-AOB CoordinateStable SD-LoRA baseline.

**Architecture:** Pure math helpers split each shared-A gradient into gauge and row-space-changing components, compare the first-order effect of the latter on current and historical LoRA operators, and return a layer gate. A post-backward learner hook applies the synchronized gate before the optimizer step and records task-local diagnostics.

**Tech Stack:** Python 3.10, PyTorch, timm, pytest, torch.distributed/DDP.

**Spec:** `docs/superpowers/specs/2026-09-04-adaptive-a-coordinate-plasticity-design.md`

## Global Constraints

- Start from no-AOB commit `4ca248a`.
- Preserve CoordinateStable alignment, bounded NormCap, prototype transport, Dual-B, and classifier behavior.
- Add no trainable parameters and no per-task persistent state.
- Keep Task 0 numerically identical to the trainable-A baseline.
- Run tests through `/home/hongzhijun/miniconda3/envs/sdlora/bin/python -m pytest`.

---

### Task 1: Adaptive-A Math And Gradient Application

**Files:**
- Modify: `backbone/sa_lora.py`
- Test: `tests/test_adaptive_a.py`

**Interfaces:**
- Produces: `low_rank_product_frobenius_norm(up, down, eps=1e-8) -> Tensor`
- Produces: `adaptive_a_layer_gradient(...) -> dict[str, Tensor]`
- Produces: `SharedALoRA_ViT_timm.apply_adaptive_a_gradients() -> dict | None`
- Produces: `SharedALoRA_ViT_timm.adaptive_a_diagnostics() -> dict | None`

- [ ] Write tests for exact low-rank norm, orthogonal gradient decomposition, no-history identity, gate limits, shared Q/V layer gate, and task-local diagnostics.
- [ ] Run `python -m pytest tests/test_adaptive_a.py -q` and verify failure because the Adaptive-A APIs do not exist.
- [ ] Implement the pure functions, constructor configuration, validation, gradient mutation, EMA, and diagnostics.
- [ ] Run `python -m pytest tests/test_adaptive_a.py -q` and verify all tests pass.
- [ ] Commit with `feat: add adaptive shared-A coordinate gradients`.

### Task 2: Learner Post-Backward Integration

**Files:**
- Modify: `models/sdlora.py`
- Modify: `models/sa_sdlora.py`
- Modify: `utils/inc_net.py`
- Test: `tests/test_adaptive_a.py`

**Interfaces:**
- Consumes: `SharedALoRA_ViT_timm.apply_adaptive_a_gradients()`
- Produces: `models.sdlora.Learner._after_backward() -> None`
- Produces: Adaptive-A argument forwarding and task-boundary logging.

- [ ] Write tests that require both training loops to invoke the post-backward hook and require `sa_sdlora` to validate and forward Adaptive-A configuration.
- [ ] Run the targeted tests and verify the expected hook/configuration failures.
- [ ] Add the no-op base hook, call it after both backward operations, override it in `sa_sdlora`, forward configuration, and log final task diagnostics.
- [ ] Run `python -m pytest tests/test_adaptive_a.py tests/test_coordinate_stability.py -q` and verify success.
- [ ] Commit with `feat: integrate adaptive A into SD-LoRA training`.

### Task 3: Same-Protocol Experiment Configurations

**Files:**
- Create: `exps/c100_coordinate_stable_adaptive_a_seed1993_nccl.json`
- Create: `exps/inr_coordinate_stable_adaptive_a_seed1995_nccl.json`
- Create: `exps/cub_coordinate_stable_adaptive_a_seed1_nccl.json`
- Create: `run_coordinate_stable_adaptive_a_queue.sh`
- Test: `tests/test_adaptive_a_queue.py`

**Interfaces:**
- Consumes: the five `sa_adaptive_a_*` configuration keys from the spec.
- Produces: four-GPU, per-GPU batch-32 runs matching the historical no-AOB protocol.

- [ ] Write a queue/config test that checks no AOB merge mode, world-size-derived launch, batch size 32, Adaptive-A keys, unique output directories, and `nohup`-safe execution.
- [ ] Run the test and verify failure because files do not exist.
- [ ] Add three configs by copying the corresponding no-AOB CoordinateStable/NormCap protocol and changing only prefix, filepath, and Adaptive-A keys; add the queue script.
- [ ] Run the queue/config test and JSON parsing checks.
- [ ] Commit with `exp: add adaptive A three-dataset queue`.

### Task 4: Regression Verification And Handoff

**Files:**
- Modify only if a new regression test exposes an Adaptive-A defect.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: a tested commit ready for smoke/full experiments.

- [ ] Run targeted Adaptive-A, CoordinateStable, cumulative-state, RNG, and Dual-B tests.
- [ ] Run the full suite and record the pre-existing aggregate-formula failure separately.
- [ ] Inspect `git diff --check`, `git status`, and the commit history.
- [ ] Verify no `adaptive_normcap`, `adaptive_operator_budget`, or AOB state/config appears in the implementation branch.
- [ ] Run a one-task smoke if a compatible GPU is free; otherwise provide the exact `nohup` launch command without occupying active experiments.
