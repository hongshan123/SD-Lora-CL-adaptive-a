# Risk-Budgeted Adaptive-A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an endpoint-capable, signed-utility and globally risk-budgeted Adaptive-A strategy without changing the established CoordinateStable method outside shared-A gradient processing.

**Architecture:** Pure tensor helpers implement gradient decomposition, operator risk, and global three-mode selection. The Shared-A backbone owns fixed-size historical input sketches and applies optimizer-aware selected updates using cross-minibatch signed agreement.

**Tech Stack:** Python, PyTorch, timm, pytest, torch.distributed.

**Spec:** `docs/superpowers/specs/2026-09-10-risk-budgeted-adaptive-a-design.md`

## Global Constraints

- Preserve the old `impact_ratio` strategy as an ablation.
- Do not modify prototype transport, classifier, absorption, or alignment.
- Do not add per-task LoRA parameters or task-growing metadata.
- All manual edits use `apply_patch`; preserve unrelated untracked files.

### Task 1: Lock Pure Policy Semantics With Failing Tests

**Files:**
- Modify: `tests/test_adaptive_a.py`
- Modify: `backbone/sa_lora.py`

1. Add tests for exact Frozen/Tangent/Live gradients.
2. Add a negative signed-utility case that selects Frozen.
3. Add a global-budget case and verify deterministic budget compliance.
4. Add scale-invariance and diagonal-sketch operator-risk tests.
5. Run the focused tests and observe missing-symbol failures.
6. Implement the smallest pure helper functions and rerun the tests.

### Task 2: Integrate Policy And Fixed-Size Sketch

**Files:**
- Modify: `backbone/sa_lora.py`
- Modify: `tests/test_adaptive_a.py`

1. Add strategy/budget/control/sketch constructor settings.
2. Capture task-local input second moments in Live-A wrappers.
3. Load historical sketches and commit merged sketches at task save.
4. Apply globally selected modes using supplied control gradients.
5. Add tests for task-0 Live behavior, state shape, reload, and no task growth.

### Task 3: Integrate Cross-Minibatch Control Updates

**Files:**
- Modify: `models/sdlora.py`
- Modify: `models/sa_sdlora.py`
- Modify: `tests/test_adaptive_a.py`

1. Pass optimizer step size through the backward-compatible hook.
2. Cache the preceding synchronized raw shared-A update per layer.
3. Compensate SGD momentum so selected endpoints are exact updates.
4. Verify other parameter gradients and effective batch are unchanged.
5. Add config validation and constructor-forwarding tests.

### Task 4: Diagnostics, Config, And Verification

**Files:**
- Modify: `models/sa_sdlora.py`
- Add: `exps/*risk_budgeted_adaptive_a*.json` as appropriate
- Modify: `tests/test_adaptive_a.py`

1. Add mode/risk/utility logging without changing old logs.
2. Add one same-protocol single-seed config per dataset from existing no-Dual baselines.
3. Run focused tests, then the Shared-A test suite.
4. Run a tiny CPU or available-GPU smoke test.
5. Review `git diff`, commit the implementation, and record verification evidence.
