# CUO Low-Rank Projection Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Add a rank-10, fixed-state CUO low-rank projection baseline using cumulative projected normal equations and the current Q/V LoRA factor budget.

**Architecture:** A pure tensor kernel maintains projected CUO sufficient statistics in a fixed row-orthonormal coordinate. A new Q/V wrapper combines immutable historical factors and temporary trainable task LoRA. A deterministic post-task pass reduces current-task statistics across DDP ranks and writes the next fixed deployment state.

**Tech Stack:** Python 3.10, PyTorch, timm ViT, torch.distributed, pytest.

**Spec:** docs/superpowers/specs/2026-09-15-cuo-lowrank-design.md

## Global Constraints

- The only new merge mode is cuo_lowrank; gauge, union_svd, and live_a_aggregate_b behavior and artifacts remain unchanged.
- Rank-10 Q/V factors remain 368,640 FP32 LoRA scalars. Report 2,400 rank-10 Gram scalars separately.
- Never persist per-task LoRA, samples, targets, or a dense 768 x 768 covariance.
- Use positive sa_cuo_lambda, FP64 arithmetic, and torch.linalg.solve. Do not form an explicit inverse.
- Reject Adaptive-A, CoordinateStable alignment/transport, HBD, and function-safe Adaptive-A options.
- Calibration uses current-task training data with evaluation transforms, no shuffle, no gradients, and DDP all-reduction.
- Every production behavior starts with a focused failing pytest. Each task receives its own commit.

---

### Task 1: Projected-CUO Math Kernel

**Files:**
- Create: backbone/cuo_lowrank.py
- Create: tests/test_cuo_lowrank.py

**Interfaces:**
- row_orthonormal_projection(weight: Tensor) -> Tensor, shape [r, d].
- solve_projected_cuo(gram, cross, damping) -> (up, diagnostics).
- advance_projected_cuo(previous_up, previous_gram, batch_z, batch_y, damping) -> (new_up, new_gram, diagnostics).
- Shapes: gram [r,r], cross/up [d,r], batch_z [n,r], batch_y [n,d].

- [ ] **Step 1: Write the failing direct-solve test**

    def test_projected_cuo_matches_direct_concatenated_ridge_solution():
        z0, y0 = torch.randn(13, 3), torch.randn(13, 7)
        z1, y1 = torch.randn(17, 3), torch.randn(17, 7)
        up0, gram0, _ = advance_projected_cuo(
            torch.zeros(7, 3), torch.zeros(3, 3), z0, y0, damping=0.2
        )
        up1, gram1, _ = advance_projected_cuo(up0, gram0, z1, y1, damping=0.2)
        z, y = torch.cat([z0, z1]), torch.cat([y0, y1])
        direct = torch.linalg.solve(
            z.T @ z + 0.2 * torch.eye(3), (y.T @ z).T
        ).T
        assert torch.allclose(up1, direct, atol=1e-6, rtol=1e-6)
        assert torch.allclose(gram1, z.T @ z, atol=1e-6, rtol=1e-6)

- [ ] **Step 2: Verify a red test**

Run: python -m pytest tests/test_cuo_lowrank.py::test_projected_cuo_matches_direct_concatenated_ridge_solution -q

Expected: import failure for backbone.cuo_lowrank.

- [ ] **Step 3: Implement the kernel**

    def solve_projected_cuo(gram, cross, damping):
        system = gram.double() + damping * identity
        up = torch.linalg.solve(system, cross.double().T).T
        return up.to(dtype=cross.dtype), diagnostics

    def advance_projected_cuo(previous_up, previous_gram, batch_z, batch_y, damping):
        previous_cross = previous_up @ (previous_gram + damping * identity)
        new_gram = previous_gram + batch_z.T @ batch_z
        new_cross = previous_cross + batch_y.T @ batch_z
        new_up, diagnostics = solve_projected_cuo(new_gram, new_cross, damping)
        return new_up, new_gram, diagnostics

Validate finite shapes, symmetric Gram matrices, positive damping, device/dtype consistency, and reduced-QR projection orthonormality.

- [ ] **Step 4: Verify green**

Run: python -m pytest tests/test_cuo_lowrank.py -q

Expected: direct solve, sequential additivity, row orthonormality, invalid damping, and ill-conditioned damping pass.

- [ ] **Step 5: Commit**

Run:
    git add backbone/cuo_lowrank.py tests/test_cuo_lowrank.py
    git commit -m "feat: add projected CUO normal equation kernel"

### Task 2: CUO Q/V Wrapper and Validation

**Files:**
- Modify: backbone/sa_lora.py
- Modify: models/sa_sdlora.py
- Modify: tests/test_cuo_lowrank.py

**Interfaces:**
- SA_MERGE_MODE_CUO_LOWRANK = "cuo_lowrank" and SA_STATE_VERSION_CUO.
- _CUOLowRankQKV with immutable projection_q/v, unified_up_q/v, temporary a_q/v and b_q/v.
- begin_cuo_calibration(), consume_cuo_statistics(), clear_cuo_calibration(), historical_output(x), and current_output(x).

- [ ] **Step 1: Write failing wrapper tests**

    def test_cuo_historical_branch_uses_fixed_projection_after_a_changes():
        wrapper = make_cuo_wrapper(dim=8, rank=3)
        inputs = torch.randn(2, 5, 8)
        before_q, _ = wrapper.historical_output(inputs)
        with torch.no_grad():
            wrapper.a_q.weight.add_(torch.randn_like(wrapper.a_q.weight))
        after_q, _ = wrapper.historical_output(inputs)
        assert torch.allclose(before_q, after_q, atol=1e-6, rtol=1e-6)

    def test_cuo_wrapper_collects_projected_target_statistics():
        wrapper = make_cuo_wrapper(dim=6, rank=2)
        wrapper.begin_cuo_calibration()
        wrapper(torch.randn(3, 4, 6))
        gram_q, cross_q, count = wrapper.consume_cuo_statistics()[0]
        assert gram_q.shape == (2, 2)
        assert cross_q.shape == (6, 2)
        assert count == 12

- [ ] **Step 2: Verify red**

Run: python -m pytest tests/test_cuo_lowrank.py -q

Expected: CUO wrapper and mode do not exist.

- [ ] **Step 3: Add the wrapper and config validation**

Historical forward is H P x. Current forward remains raw scale times B(Ax). With collection enabled, flatten current inputs and final branch residual, calculate z=xP.T, and accumulate z.T @ z plus residual.T @ z in local FP64 buffers. Normal training must perform no collection work.

Require cumulative state, trainable temporary A, rank equality, positive sa_cuo_lambda, and reject live-coordinate settings.

- [ ] **Step 4: Verify compatibility**

Run: python -m pytest tests/test_cuo_lowrank.py tests/test_coordinate_stability.py tests/test_sa_cumulative.py -q

Expected: CUO tests and all existing cumulative modes pass.

- [ ] **Step 5: Commit**

Run:
    git add backbone/sa_lora.py models/sa_sdlora.py tests/test_cuo_lowrank.py
    git commit -m "feat: add fixed-projection CUO LoRA wrapper"

### Task 3: State, Calibration, and DDP

**Files:**
- Modify: backbone/sa_lora.py
- Modify: models/sa_sdlora.py
- Modify: tests/test_cuo_lowrank.py
- Create: tests/ddp_smoke_cuo_lowrank.py

**Interfaces:**
- prepare_cuo_calibration(), finalize_cuo_calibration(), and _save_cuo_lowrank_state(filename, task_id).
- State keys: projection_down, unified_up, projected_gram, cuo_lambda, rank, task_id, merge_mode, version.
- cuo_state_scalar_counts(num_blocks, rank, dim).

- [ ] **Step 1: Write failing roundtrip and accounting tests**

    def test_cuo_state_roundtrip_preserves_deployed_logits(tmp_path):
        model = train_tiny_cuo_task_zero(tmp_path)
        inputs = torch.randn(2, 5, 8)
        before = rebuild_cuo_deployment(model, inputs)
        rebuilt = load_tiny_cuo(tmp_path, task=1)
        assert torch.allclose(rebuilt(inputs), before, atol=2e-6, rtol=2e-6)
        state = torch.load(tmp_path / SA_STATE_FILENAME, weights_only=True)
        assert {"projection_down", "unified_up", "projected_gram"} <= set(state)

    def test_rank10_vit_state_counts_factors_and_grams():
        counts = cuo_state_scalar_counts(num_blocks=12, rank=10, dim=768)
        assert counts["lora_factor_scalars"] == 368640
        assert counts["gram_scalars"] == 2400

Add a reducer test that combines two local statistic tuples and matches their concatenated direct solve exactly once.

- [ ] **Step 2: Verify red**

Run: python -m pytest tests/test_cuo_lowrank.py -q

Expected: missing state, calibration, or accounting APIs.

- [ ] **Step 3: Implement lifecycle**

At Task 0, canonicalize trained temporary A before the calibration pass. Later tasks use stored P. Run a deterministic no-gradient pass after optimization and before save. All-reduce each branch Gram/cross/count, advance H/C on rank 0, broadcast state, persist explicit CUO fields, and reject old artifact versions.

save_merged_lora writes H/P deployment factors only. Log token count, condition number, residual, factor scalars, Gram scalars, and total count.

- [ ] **Step 4: Verify CPU and DDP**

Run: python -m pytest tests/test_cuo_lowrank.py tests/test_sa_cumulative.py tests/test_coordinate_stability.py -q

Run: CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 tests/ddp_smoke_cuo_lowrank.py

Expected: tests pass; the smoke completes Task 0/1, writes once, has identical rank hashes, and creates no per-task B files.

- [ ] **Step 5: Commit**

Run:
    git add backbone/sa_lora.py models/sa_sdlora.py tests/test_cuo_lowrank.py tests/ddp_smoke_cuo_lowrank.py
    git commit -m "feat: persist and synchronize low-rank CUO state"

### Task 4: Protocol Integration and Configurations

**Files:**
- Modify: models/sa_sdlora.py
- Create: tests/test_cuo_lowrank_queue.py
- Create: exps/cuo_lowrank_r10_c100_seed1993.json
- Create: exps/cuo_lowrank_r10_inr_seed1995.json
- Create: exps/cuo_lowrank_r10_cub_seed1.json

**Interfaces:**
- sa_cuo_lambda=1e-5 and sa_cuo_calibration_batch_size defaulting to training batch size.
- Calibration occurs after final optimizer update and before state save, never during epoch evaluation.

- [ ] **Step 1: Write failing config tests**

    @pytest.mark.parametrize("key,value", [
        ("sa_adaptive_a_enabled", True),
        ("sa_coordinate_stable_transport", True),
        ("sa_hbd_enabled", True),
    ])
    def test_cuo_rejects_live_coordinate_features(key, value):
        config = base_cuo_config()
        config[key] = value
        with pytest.raises(ValueError, match="cuo_lowrank"):
            validate_cuo_lowrank_config(config)

    def test_cuo_rank_must_match_persistent_rank():
        config = base_cuo_config()
        config["sa_cumulative_rank"] = 8
        with pytest.raises(ValueError, match="rank"):
            validate_cuo_lowrank_config(config)

- [ ] **Step 2: Verify red**

Run: python -m pytest tests/test_cuo_lowrank_queue.py -q

Expected: missing validation or invalid configurations accepted.

- [ ] **Step 3: Integrate and add protocol-matched configs**

Hook calibration into SA_SDLora immediately before its task-ending save_lora_parameters call. Copy only dataset protocol values from the current no-Dual-B historical-signal configurations. Do not set Dual-B, HBD, prototype transport, or Adaptive-A controls.

- [ ] **Step 4: Verify config and Task 0 smoke**

Run: python -m pytest tests/test_cuo_lowrank_queue.py tests/test_adaptive_a_queue.py -q

Run: write a temporary copy of the CIFAR-100 config with `"max_tasks": 1`, then run `CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 main.py --config=<temporary-config>`.

Expected: invalid settings fail clearly; Task 0 saves finite CUO and merged state only.

- [ ] **Step 5: Commit**

Run:
    git add models/sa_sdlora.py tests/test_cuo_lowrank_queue.py exps/cuo_lowrank_r10_c100_seed1993.json exps/cuo_lowrank_r10_inr_seed1995.json exps/cuo_lowrank_r10_cub_seed1.json
    git commit -m "exp: add rank10 low-rank CUO configurations"

### Task 5: State Audit and Three-Dataset Queue

**Files:**
- Modify: scripts/measure_sa_artifact.py
- Create: run_cuo_lowrank_r10_3datasets_2gpu.sh
- Modify: experiment_sd.md
- Modify: experiment_note_sd.md

**Interfaces:**
- Artifact report has lora_factor_scalars, cuo_gram_scalars, and persistent_scalar_total.
- Queue uses two GPUs per run, per-GPU batch 64, effective batch 128, individual logs, timestamps, and statuses.

- [ ] **Step 1: Write failing artifact and queue tests**

    def test_measure_artifact_separates_cuo_factors_from_gram_stats(tmp_path):
        write_minimal_cuo_state(tmp_path, branches=24, dim=768, rank=10)
        summary = measure_artifact(tmp_path)
        assert summary["lora_factor_scalars"] == 368640
        assert summary["cuo_gram_scalars"] == 2400
        assert summary["persistent_scalar_total"] == 371040

Add a static script test requiring the three CUO configs, torchrun --nproc_per_node=2, and scheduler-provided CUDA_VISIBLE_DEVICES.

- [ ] **Step 2: Verify red**

Run: python -m pytest tests/test_cuo_lowrank.py tests/test_cuo_lowrank_queue.py -q

Expected: absent accounting fields or queue script.

- [ ] **Step 3: Implement audit and queue**

Extend reporting only for CUO v5. Queue C100 and ImageNet-R on disjoint GPU pairs, then CUB when a pair becomes free. Append method, budget, and run provenance to experiment records without modifying earlier results.

- [ ] **Step 4: Final verification**

Run: python -m pytest tests/test_cuo_lowrank.py tests/test_cuo_lowrank_queue.py tests/test_sa_cumulative.py tests/test_coordinate_stability.py tests/test_adaptive_a.py -q

Run: python scripts/measure_sa_artifact.py --artifact <two-task-cuo-smoke-artifact>

Expected: regression passes; report shows 368,640 factors, 2,400 Gram scalars, and no per-task B files.

- [ ] **Step 5: Commit**

Run:
    git add scripts/measure_sa_artifact.py run_cuo_lowrank_r10_3datasets_2gpu.sh experiment_sd.md experiment_note_sd.md tests/test_cuo_lowrank.py tests/test_cuo_lowrank_queue.py
    git commit -m "docs: add low-rank CUO audit and launch queue"

## Plan Self-Review

- Tasks 1-3 cover the fixed-coordinate projected CUO normal equation, exact sufficient-statistics recovery, state format, and DDP synchronization.
- Task 4 enforces compatibility constraints and creates the three protocol-aligned configurations.
- Task 5 provides honest state accounting, reproducible launch commands, and experimental provenance.
- Names and shapes are consistent: projection_down [r,d], unified_up [d,r], projected_gram [r,r], batch_z [n,r], batch_y [n,d].
