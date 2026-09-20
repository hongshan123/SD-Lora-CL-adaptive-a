# Recoverability-Constrained Accessibility Adaptive-A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement four cumulative validation stages for exact historical recoverability, effective-gradient accessibility, fixed-anchor online alignment, and global recoverability budgeting.

**Architecture:** Put basis-invariant low-rank geometry and candidate selection in a focused pure-PyTorch module. Extend the Live-A QKV wrapper to retain only current-batch input/output-gradient factors, then let the backbone prepare accepted row-basis candidates after backward and apply/retract/realign them after the optimizer step. Existing strategies and deployment artifacts remain unchanged.

**Tech Stack:** Python, PyTorch, pytest, torch.distributed

**Spec:** `docs/superpowers/specs/2026-09-20-recoverability-accessibility-adaptive-a-design.md`

## Global Constraints

- Existing Adaptive-A strategies remain behaviorally unchanged.
- Persistent deployment state remains the existing shared `A` and aggregate `G` per branch.
- Task-start anchors are transient and never serialized.
- Effective-weight gradients are represented by a deterministic random output
  sketch of shape `sketch_rank x d`, not dense `d x d` tensors.
- Prototype transport, classifier, Dual-B, and absorption logic are unchanged.

## Review Focus

- Rank-deficient candidate bases must produce finite risk and alignment through a thresholded pseudoinverse.
- Zero historical energy must produce zero risk and permit the full candidate.
- DDP statistics must use the averaged effective-weight gradient, not an average of local squared energies.
- Repeated online realignment must always target the original task anchor.
- A rejected candidate must leave shared A and historical aggregate exactly unchanged.

---

### Task 1: Exact Recoverability Geometry

**Files:**
- Create: `backbone/recoverability.py`
- Create: `tests/test_recoverability.py`

**Interfaces:**
- Produces: `row_polar_retraction`, `recoverability_energies`, `align_to_anchor`, and `operator_weighted_recoverability`.

- [x] Write tests comparing the low-rank risk and alignment with explicit dense least squares, including gauge transforms, equal-Grassmann unequal-energy candidates, zero history, and rank deficiency.
- [x] Run `pytest tests/test_recoverability.py -q` and verify failure because the module does not exist.
- [x] Implement the low-rank formulas using FP64 internal Gram algebra and thresholded symmetric pseudoinverses.
- [x] Run the focused tests and commit the independently passing geometry component.

### Task 2: Accessibility Geometry

**Files:**
- Modify: `backbone/recoverability.py`
- Modify: `tests/test_recoverability.py`

**Interfaces:**
- Produces: `effective_gradient_cross`, `grassmann_accessibility_direction`, `accessibility_energy`, and `accessibility_candidates`.

- [x] Write failing tests comparing factorized statistics against a dense effective-weight gradient and finite-difference checking the Grassmann ascent direction.
- [x] Run the focused tests and verify the expected missing-interface failures.
- [x] Implement exact minibatch-factor computations and polar-retracted candidate generation.
- [x] Run the focused tests and commit the accessibility component.

### Task 3: Fixed-Anchor Online Controller

**Files:**
- Modify: `backbone/sa_lora.py`
- Modify: `models/sa_sdlora.py`
- Create: `tests/test_recoverability_adaptive_a.py`

**Interfaces:**
- Produces: wrapper factor capture, task-local anchors, `prepare_recoverability_step`, and `apply_recoverability_step`.

- [x] Write failing tests proving that stage 1 uses exact risk, stage 2 works with zero current B, and stage 3 repeatedly aligns to the unchanged task-start anchor.
- [x] Run the focused tests and verify the expected missing-strategy failures.
- [x] Add opt-in capture buffers, candidate preparation after backward, polar application after optimizer step, and online aggregate realignment.
- [x] Run the focused tests and commit the integrated controller.

### Task 4: Global Budget And Configuration

**Files:**
- Modify: `backbone/recoverability.py`
- Modify: `backbone/sa_lora.py`
- Modify: `models/sa_sdlora.py`
- Modify: `tests/test_recoverability.py`
- Modify: `tests/test_recoverability_adaptive_a.py`

**Interfaces:**
- Produces: `choose_global_recoverability_candidates` and validated `sa_recoverability_*` configuration.

- [x] Write failing tests for exact global energy weighting, deterministic Pareto selection, invalid configuration, task-0 full selection, and budget compliance.
- [x] Run the focused tests and verify failures for missing selector/configuration.
- [x] Implement stage-4 global allocation, diagnostics, and model wiring.
- [x] Run focused integration tests and commit the full strategy.

### Task 5: Ordered Validation Artifacts

**Files:**
- Create: `scripts/make_recoverability_validation_configs.py`
- Create: `tests/test_recoverability_validation_configs.py`
- Create: `docs/recoverability_validation.md`

**Interfaces:**
- Produces: deterministic stage 1-4 experiment configurations from an existing baseline and a CPU toy diagnostic.

- [x] Write failing tests for cumulative stage configs and the equal-chordal/different-operator-energy toy result.
- [x] Run focused tests and verify expected failures.
- [x] Implement the generator and documented validation commands without launching expensive training.
- [x] Run all recoverability tests, then the full pytest suite, and commit validation artifacts.
