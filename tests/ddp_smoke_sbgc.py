"""Two-process smoke for SBGC calibration, synchronization and persistence."""

import copy
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.sa_lora import (
    SA_MERGED_FILENAME,
    SA_STATE_FILENAME,
    SharedALoRA_ViT_timm,
    _SensitivityBudgetedGQKV,
)
from backbone.sbgc import normalize_sensitivity, update_running_moment


class _TinyAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)


class _TinyBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = _TinyAttention(dim)


class _TinyViT(nn.Module):
    def __init__(self, dim=6):
        super().__init__()
        self.blocks = nn.ModuleList([_TinyBlock(dim)])
        self.head = nn.Identity()

    def forward(self, inputs):
        for block in self.blocks:
            inputs = block.attn.qkv(inputs)
        return self.head(inputs.mean(dim=1))


def _make_model(run_dir, task, device):
    torch.manual_seed(901)
    return SharedALoRA_ViT_timm(
        copy.deepcopy(_TinyViT()),
        r=2,
        filepath=str(run_dir),
        cur_task_index=task,
        train_a_all_tasks=False,
        cumulative_state=True,
        cumulative_merge="sensitivity_budgeted_g",
        cumulative_rank=2,
        g_risk_budget=0.05,
        g_sensitivity_metric="fisher_diag",
    ).to(device)


def _state_hash(model):
    digest = hashlib.sha256()
    for wrapper in model._sbgc_wrappers():
        assert isinstance(wrapper, _SensitivityBudgetedGQKV)
        for tensor in (
            wrapper.projection_q,
            wrapper.unified_up_q,
            wrapper.projected_covariance_q,
            wrapper.sensitivity_q,
            wrapper.projection_v,
            wrapper.unified_up_v,
            wrapper.projected_covariance_v,
            wrapper.sensitivity_v,
        ):
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        digest.update(
            repr(
                (
                    wrapper.covariance_count_q,
                    wrapper.sensitivity_count_q,
                    wrapper.covariance_count_v,
                    wrapper.sensitivity_count_v,
                )
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _assert_ranks_match(value):
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    assert len(set(gathered)) == 1, gathered


def _calibrate(model, rank, task, device):
    torch.manual_seed(1001 + task)
    with torch.no_grad():
        if task == 0:
            for module in model.w_As + model.w_Bs:
                module.weight.copy_(torch.randn_like(module.weight))
        else:
            for wrapper in model._sbgc_wrappers():
                wrapper.b_q.weight.copy_(6.0 * wrapper.unified_up_q)
                wrapper.b_v.weight.copy_(6.0 * wrapper.unified_up_v)
        model.wrapped_param[0].param.fill_(0.61)
    before_p = [
        tensor.detach().clone()
        for wrapper in model._sbgc_wrappers()
        for tensor in (wrapper.projection_q, wrapper.projection_v)
    ]
    model.prepare_sbgc_calibration()
    wrapper = model._sbgc_wrappers()[0]
    prior_covariances = (
        wrapper.projected_covariance_q.detach().double().clone(),
        wrapper.projected_covariance_v.detach().double().clone(),
    )
    prior_sensitivities = (
        wrapper.sensitivity_q.detach().double().clone(),
        wrapper.sensitivity_v.detach().double().clone(),
    )
    prior_covariance_counts = (
        wrapper.covariance_count_q,
        wrapper.covariance_count_v,
    )
    prior_sensitivity_counts = (
        wrapper.sensitivity_count_q,
        wrapper.sensitivity_count_v,
    )
    after_p = [
        tensor.detach().clone()
        for wrapper in model._sbgc_wrappers()
        for tensor in (wrapper.projection_q, wrapper.projection_v)
    ]
    if task > 0:
        assert all(torch.equal(left, right) for left, right in zip(before_p, after_p))

    local_batch = rank + 2
    inputs = torch.randn(local_batch, 3, 6, device=device)
    gathered_inputs = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_inputs, inputs.detach().cpu())
    global_inputs = torch.cat(gathered_inputs).double()
    model(inputs)
    outputs = model.sbgc_calibration_outputs()
    weights = torch.linspace(0.2, 2.0, 6, device=device)
    loss = sum(
        (output * weights).sum(dim=(1, 2)).mean() for output in outputs
    )
    gradients = torch.autograd.grad(loss, outputs)
    model.accumulate_sbgc_sensitivities(gradients, local_batch)
    stats = model.finalize_sbgc_calibration()
    expected_tokens = sum((other + 2) * 3 for other in range(dist.get_world_size()))
    expected_sensitivity = normalize_sensitivity(weights.detach().cpu().double().square())
    for branch_index, (projection, covariance, sensitivity) in enumerate(
        (
            (wrapper.projection_q, wrapper.projected_covariance_q, wrapper.sensitivity_q),
            (wrapper.projection_v, wrapper.projected_covariance_v, wrapper.sensitivity_v),
        )
    ):
        coordinates = global_inputs.reshape(-1, 6) @ projection.detach().cpu().double().t()
        current_covariance = coordinates.t() @ coordinates / expected_tokens
        expected_covariance, _ = update_running_moment(
            prior_covariances[branch_index],
            prior_covariance_counts[branch_index],
            current_covariance,
            expected_tokens,
        )
        expected_running_sensitivity, _ = update_running_moment(
            prior_sensitivities[branch_index],
            prior_sensitivity_counts[branch_index],
            expected_sensitivity,
            expected_tokens,
        )
        expected_running_sensitivity = normalize_sensitivity(
            expected_running_sensitivity
        )
        assert torch.allclose(
            covariance.detach().cpu().double(),
            expected_covariance,
            atol=2e-6,
            rtol=2e-6,
        )
        assert torch.allclose(
            sensitivity.detach().cpu().double(),
            expected_running_sensitivity,
            atol=2e-6,
            rtol=2e-6,
        )
    if rank == 0:
        assert stats["branch_count"] == 2
        assert all(
            item["covariance_count"] == expected_tokens
            for item in model._last_sbgc_branch_diagnostics
        )
        assert all(
            item["sensitivity_count"] == expected_tokens
            for item in model._last_sbgc_branch_diagnostics
        )
        if task > 0:
            assert stats["active_constraints"] == 2
            assert stats["max_achieved_risk"] <= 0.05 + 1e-6
    _assert_ranks_match(_state_hash(model))


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    path_holder = [
        tempfile.mkdtemp(prefix="sbgc-ddp-") if rank == 0 else None
    ]
    dist.broadcast_object_list(path_holder, src=0)
    run_dir = Path(path_holder[0])
    try:
        model0 = _make_model(run_dir, task=0, device=device)
        _calibrate(model0, rank, task=0, device=device)
        if rank == 0:
            model0.save_lora_parameters(str(run_dir), task_id=0)
        dist.barrier()

        model1 = _make_model(run_dir, task=1, device=device)
        _calibrate(model1, rank, task=1, device=device)
        if rank == 0:
            model1.save_lora_parameters(str(run_dir), task_id=1)
            entries = {path.name for path in run_dir.iterdir()}
            assert entries == {SA_STATE_FILENAME, SA_MERGED_FILENAME}, entries
            print(
                "SBGC_DDP_SMOKE_PASS ranks={} hash={} artifacts={}".format(
                    dist.get_world_size(), _state_hash(model1), sorted(entries)
                ),
                flush=True,
            )
        dist.barrier()
    finally:
        dist.barrier()
        if rank == 0 and run_dir.exists():
            shutil.rmtree(run_dir)
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
