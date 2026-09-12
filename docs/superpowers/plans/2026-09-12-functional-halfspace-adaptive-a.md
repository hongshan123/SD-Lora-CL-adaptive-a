# Functional-Halfspace Adaptive-A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a teacher-consistency halfspace projection for shared-A optimizer directions while preserving the current fixed-state SD-LoRA method and all non-A training behavior.

**Architecture:** A pure tensor kernel in `backbone/sa_lora.py` performs one global constrained projection over all shared-A branches. `models/sa_sdlora.py` reuses the task-start HBD teacher to compute synchronized functional-stability gradients, projects momentum-aware SGD directions after DDP CE backward, and records task-local diagnostics.

**Tech Stack:** Python, PyTorch autograd/DDP, timm ViT, pytest.

**Spec:** `docs/superpowers/specs/2026-09-12-functional-halfspace-adaptive-a-design.md`

## Global Constraints

- Strategy name is exactly `functional_halfspace`.
- Task 0 leaves the live shared-A gradients unchanged and creates no teacher constraint.
- The stability distance is used only for `autograd.grad`; it is never added to the training loss.
- Project the actual SGD direction `grad + momentum * momentum_buffer`, then write back `d_star - momentum * momentum_buffer`.
- Use one global halfspace across all Q/V shared-A matrices while preserving each matrix's row-space-parallel component when the normal channel is usable.
- Use a full-space projection when the normal channel is unusable; use an unchanged no-op only when the complete stability gradient is degenerate.
- Epsilon/tolerance values are branch thresholds only and are never added to projection denominators.
- Manually average stability gradients across DDP ranks.
- Do not change fixed `(A, G)` persistence, alignment, absorption/NormCap, prototype transport, classifier, B gradients, or checkpoint format.
- Add no persistent or per-task model parameters.

---

### Task 1: Global Halfspace Projection Kernel

**Files:**
- Modify: `backbone/sa_lora.py`
- Create: `tests/test_functional_halfspace_adaptive_a.py`

**Interfaces:**
- Consumes: `decompose_adaptive_a_gradient(gradient, shared_a)` for exact QR-based parallel/perpendicular components.
- Produces: `project_functional_halfspace_directions(proposed_directions, stability_gradients, shared_as, conflict_tol=1e-12, normal_tol=1e-12, min_normal_fraction=1e-4) -> dict` containing `directions` plus scalar diagnostics `mode`, `pre_inner`, `post_inner`, `normal_fraction`, and `correction_ratio`.

- [ ] **Step 1: Write failing tensor-kernel tests**

Cover shape/list validation, no-conflict identity, exact global normal-space correction over multiple matrices, row-space-parallel preservation, full-space fallback when the stability gradient is row-space-parallel, and unchanged degenerate no-op when the complete stability gradient is zero.  Assert `post_inner >= -1e-6 * max(1, abs(pre_inner))` for projected float32 cases and exact tensor equality for no-op cases.

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `python -m pytest tests/test_functional_halfspace_adaptive_a.py -q`

Expected: collection/import failure because `project_functional_halfspace_directions` does not exist.

- [ ] **Step 3: Implement the projection kernel**

Validate equal non-empty list lengths and matching tensor shapes.  Compute QR-based decompositions with the existing helper, aggregate dot products and squared norms in float64 scalars, apply the normal correction when `h2 > normal_tol` and `h2 / ||g_s||^2 >= min_normal_fraction`, otherwise apply the full-space correction when `||g_s||^2 > normal_tol`, and otherwise return exact cloned proposed directions.  Never add tolerance to a denominator.

- [ ] **Step 4: Run focused and existing Adaptive-A tests**

Run: `python -m pytest tests/test_functional_halfspace_adaptive_a.py tests/test_adaptive_a.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backbone/sa_lora.py tests/test_functional_halfspace_adaptive_a.py
git commit -m "feat: add functional halfspace projection kernel"
```

### Task 2: Teacher Gradient Integration And Diagnostics

**Files:**
- Modify: `models/sa_sdlora.py`
- Modify: `backbone/sa_lora.py`
- Modify: `tests/test_functional_halfspace_adaptive_a.py`

**Interfaces:**
- Consumes: `project_functional_halfspace_directions(...)` from Task 1 and existing `build_hbd_teacher`, capture hooks, `live_a_historical_outputs`, and `hbd_historical_branch_distance`.
- Produces: config validation for `functional_halfspace`; a learner helper that computes DDP-averaged stability gradients; momentum-aware gradient replacement through `apply_adaptive_a_gradients`; task-local functional diagnostics returned by `adaptive_a_diagnostics` and logged by `_log_adaptive_a_diagnostics`.

- [ ] **Step 1: Write failing integration tests**

Test exact config defaults and invalid bounds; task-0 unchanged gradients; task `t > 0` teacher creation even when `sa_hbd_enabled=false`; absence of HBD in `_additional_training_losses`; DDP-style averaging through a mocked distributed backend; momentum-aware writeback that makes the optimizer direction equal the kernel result; no changes to B gradients; teacher cleanup; and task diagnostics for normal projection, full fallback, and degenerate no-op counts.

- [ ] **Step 2: Run integration tests and confirm they fail**

Run: `python -m pytest tests/test_functional_halfspace_adaptive_a.py -q`

Expected: failures for unsupported strategy/config and missing learner integration.

- [ ] **Step 3: Add validated configuration**

Extend `validate_adaptive_a_config`, `SharedALoRA_ViT_timm`, and the learner initialization with the exact names/defaults from the spec.  Require SGD for `functional_halfspace`.  Reject negative conflict tolerance, non-positive normal tolerance, and minimum normal fractions outside `[0, 1]`.

- [ ] **Step 4: Compute the functional stability gradient**

For tasks after task 0, create the frozen task-start teacher through the existing HBD lifecycle even when ordinary HBD is disabled.  In `_after_backward`, make one RNG-preserving student historical-branch forward and one no-grad teacher forward, compute the unweighted HBD distance, call `torch.autograd.grad` only for `backbone.w_As`, restore input-sketch capture flags, and manually average every non-None gradient across DDP ranks.  Do not add this distance to `_additional_training_losses`.

- [ ] **Step 5: Project the momentum-aware update**

Pass CE gradients, stability gradients, shared-A tensors, momentum buffers, and optimizer momentum to the backbone.  Build proposed directions as `grad + mu * buffer`, invoke the Task 1 kernel once across all branches, then copy `direction_star - mu * buffer` into `.grad`.  Leave task 0 and missing/zero stability gradients unchanged.  Never touch current-B or classifier gradients.

- [ ] **Step 6: Add task-local diagnostics and logging**

Accumulate only transient counters and means for conflicts, normal projections, full fallbacks, degenerate no-ops, pre/post inner products, normal fractions, and correction ratios.  Reset them at task boundaries, expose them through `adaptive_a_diagnostics`, and emit one `[FunctionalHalfspace-AdaptiveA]` task summary.  Do not add these fields to saved state.

- [ ] **Step 7: Run focused and full regression tests**

Run: `python -m pytest tests/test_functional_halfspace_adaptive_a.py tests/test_hbd.py tests/test_adaptive_a.py -q`

Run: `python -m pytest -q`

Expected: all tests PASS, including the original 198-test baseline plus the new tests.

- [ ] **Step 8: Commit**

```bash
git add models/sa_sdlora.py backbone/sa_lora.py tests/test_functional_halfspace_adaptive_a.py
git commit -m "feat: integrate teacher-projected adaptive a"
```

