# Momentum-Aware Adaptive-A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the operator-impact Adaptive-A gate control the actual SGD momentum update, expose ratio and squared-ratio gates plus a true Tangent-A endpoint, and launch matched single-seed experiments on CIFAR-100, ImageNet-R, and CUB-200.

**Architecture:** Preserve the existing Shared-A forward, task-boundary least-squares alignment, prototype classifier, and operator-preserving absorption. For the impact gate only, form the effective SGD direction `D_eff = grad + momentum * buffer`, decompose and gate `D_eff`, then write back `grad = D_eff_gated - momentum * buffer`; add `ratio` and `squared_ratio` as a small gate-formula setting. Tangent-A uses the same path with a forced zero perpendicular gate, so its optimizer step preserves the current row space even when a momentum buffer exists.

**Tech Stack:** Python 3, PyTorch, pytest, torchrun/DDP, Bash/nohup

**Spec:** `adaptive_live_frozen_a_method_design.md`

## Global Constraints

- Keep the existing forward `G A_bar x + s B A x` unchanged.
- Keep task-boundary least-squares alignment and operator-preserving absorption unchanged.
- Do not modify prototype transport, classifier, Dual-B, or historical state format.
- Add no task-growing persistent state.
- The theoretical main configuration uses `lambda_n=lambda_o=1`, `gate_floor=0`, and gate EMA momentum `0`.
- Preserve legacy `impact_ratio` defaults for old configurations unless the new option is explicitly set.
- Use per-process batch size 64 on two GPUs, preserving effective batch size 128.

---

### Task 1: Gate Mathematics and Optimizer-Direction Contract

**Files:**
- Modify: `tests/test_adaptive_a.py`
- Modify: `backbone/sa_lora.py`

**Interfaces:**
- Consumes: `adaptive_a_layer_gradient(...)`, `SharedALoRA_ViT_timm.apply_adaptive_a_gradients(...)`
- Produces: `gate_formula: str`, momentum-aware effective gradients, and `adaptive_a_strategy="tangent"`

- [ ] **Step 1: Write failing gate-formula tests**

Add literal fixtures where `I_cur=2` and `I_hist=4`; assert ratio gate `1/3`, squared-ratio gate `1/5`, and reject an unknown formula.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `pytest -q tests/test_adaptive_a.py -k 'gate_formula or squared_ratio'`

Expected: FAIL because `gate_formula` is not accepted.

- [ ] **Step 3: Write failing momentum and Tangent-A tests**

Use a row basis `[1, 0]`, raw gradient `[0, 0]`, momentum buffer `[0, 2]`, and SGD momentum `0.9`. Assert that `impact_ratio` gates the effective direction and that Tangent-A writes raw gradient `[0, -1.8]`, causing the optimizer's effective perpendicular update to be exactly zero.

- [ ] **Step 4: Run the focused tests and confirm RED**

Run: `pytest -q tests/test_adaptive_a.py -k 'momentum_aware or tangent_strategy'`

Expected: FAIL because the impact path ignores optimizer momentum and tangent is unsupported.

- [ ] **Step 5: Implement the minimal gate and momentum logic**

Extend `adaptive_a_layer_gradient` with `gate_formula`, using:

```python
if gate_formula == "ratio":
    numerator = current_impact
    denominator = current_impact + stability_weight * historical_impact
else:
    numerator = current_impact.square()
    denominator = numerator + stability_weight * historical_impact.square()
raw_gate = (numerator / (denominator + eps)).clamp(gate_floor, 1.0)
```

In `apply_adaptive_a_gradients`, derive each effective direction from its matching buffer, pass it into the gate, and copy `gated_effective - momentum * buffer` back to `.grad`. For `tangent`, force the returned perpendicular gate to zero after Task 0.

- [ ] **Step 6: Run focused tests and confirm GREEN**

Run: `pytest -q tests/test_adaptive_a.py -k 'adaptive_layer_gradient or momentum_aware or tangent_strategy'`

Expected: PASS.

### Task 2: Configuration Validation and Learner Wiring

**Files:**
- Modify: `tests/test_adaptive_a.py`
- Modify: `models/sa_sdlora.py`
- Modify: `utils/inc_net.py`
- Modify: `backbone/sa_lora.py`

**Interfaces:**
- Consumes: JSON key `sa_adaptive_a_gate_formula`
- Produces: constructor setting `adaptive_a_gate_formula` with values `ratio|squared_ratio`

- [ ] **Step 1: Write failing validation and forwarding tests**

Assert default `ratio`, explicit `squared_ratio` forwarding, invalid-value rejection, and that learner `_after_backward` passes optimizer momentum state for `impact_ratio` and `tangent`.

- [ ] **Step 2: Run focused tests and confirm RED**

Run: `pytest -q tests/test_adaptive_a.py -k 'gate_formula or optimizer_momentum_forwarding'`

Expected: FAIL because the option and forwarding do not exist.

- [ ] **Step 3: Implement validation and forwarding**

Add `sa_adaptive_a_gate_formula` to learner validation, both backbone construction paths, and constructor validation. Refactor `_after_backward` so `impact_ratio` and `tangent` pass aligned `momentum_buffers` and optimizer momentum to the backbone.

- [ ] **Step 4: Run Adaptive-A regression tests**

Run: `pytest -q tests/test_adaptive_a.py tests/test_functional_halfspace_adaptive_a.py`

Expected: PASS.

### Task 3: Reproducible Three-Dataset Experiment Launcher

**Files:**
- Create: `scripts/generate_momentum_adaptive_a_configs.py`
- Create: `run_momentum_adaptive_a_t10_3datasets_2gpu.sh`
- Create: `tests/test_momentum_adaptive_a_queue.py`

**Interfaces:**
- Consumes: existing matched source configs for CIFAR-100, ImageNet-R, and CUB-200
- Produces: three T=10 JSON configs and independent two-GPU nohup workers

- [ ] **Step 1: Write failing generator/launcher behavior tests**

Generate configs in a temporary directory and assert dataset-specific seeds, ten equal tasks, batch size 64, rank 10, prototype classifier enabled, Dual-B disabled, coordinate alignment enabled, operator-preserving absorption, ratio gate, floor 0, gate EMA 0, and no source-config mutation.

- [ ] **Step 2: Run the queue test and confirm RED**

Run: `pytest -q tests/test_momentum_adaptive_a_queue.py`

Expected: FAIL because the generator does not exist.

- [ ] **Step 3: Implement generator and launcher**

The launcher assigns `0,1` to CIFAR-100, `4,5` to ImageNet-R, and `6,7` to CUB-200. Each worker runs through `nohup bash -lc`, activates `/home/hongzhijun/miniconda3/envs/sdlora`, sets `HF_ENDPOINT=https://hf-mirror.com`, and invokes `torchrun --standalone --nproc_per_node=2`.

- [ ] **Step 4: Run generator/launcher tests and shell syntax validation**

Run: `pytest -q tests/test_momentum_adaptive_a_queue.py && bash -n run_momentum_adaptive_a_t10_3datasets_2gpu.sh`

Expected: PASS.

### Task 4: Verification, Documentation, Commit, and Launch

**Files:**
- Modify: `adaptive_live_frozen_a_method_design.md`
- Modify: `docs/superpowers/plans/2026-09-17-momentum-aware-adaptive-a.md`

**Interfaces:**
- Consumes: completed implementation and test evidence
- Produces: committed reproducible code and live experiment PIDs/logs

- [ ] **Step 1: Update the design document**

Document optimizer-direction gating, explicit Tangent-A semantics, both gate formulas, and the exact main experiment settings without changing the historical-result sections.

- [ ] **Step 2: Run complete relevant verification**

Run: `pytest -q tests/test_adaptive_a.py tests/test_functional_halfspace_adaptive_a.py tests/test_live_a_aggregate_math.py tests/test_live_a_aggregate_backbone.py tests/test_momentum_adaptive_a_queue.py`

Expected: PASS with zero failures.

- [ ] **Step 3: Review and commit only intended files**

Run: `git diff --check`, inspect `git diff`, then stage only the files named in this plan and commit with `feat: make adaptive-a momentum aware`.

- [ ] **Step 4: Launch and verify all three experiments**

Run: `nohup ./run_momentum_adaptive_a_t10_3datasets_2gpu.sh > momentum_adaptive_a_t10_queue.log 2>&1 &`

Verify with `ps`, `nvidia-smi`, and the three generated run logs. Each log must contain configuration output or Task 0 startup, and no immediate traceback.
