"""Two-process NCCL smoke for CUO's all-rank calibration lifecycle."""

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
    _CUOLowRankQKV,
)


class _TinyAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)


class _TinyBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = _TinyAttention(dim)


class _TinyViT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.blocks = nn.ModuleList([_TinyBlock(dim)])
        self.head = nn.Identity()

    def forward(self, inputs):
        for block in self.blocks:
            inputs = block.attn.qkv(inputs)
        return self.head(inputs)


def _make_model(run_dir, task):
    torch.manual_seed(410)
    return SharedALoRA_ViT_timm(
        _TinyViT(dim=6),
        r=2,
        filepath=str(run_dir),
        cur_task_index=task,
        train_a_all_tasks=True,
        cumulative_state=True,
        cumulative_merge="cuo_lowrank",
        cumulative_rank=2,
        cuo_lambda=0.1,
    )


def _wrapper_state_hash(model):
    digest = hashlib.sha256()
    for block in model.lora_vit.blocks:
        wrapper = block.attn.qkv
        assert isinstance(wrapper, _CUOLowRankQKV)
        for tensor in (
            wrapper.projection_q,
            wrapper.unified_up_q,
            wrapper.projected_gram_q,
            wrapper.projection_v,
            wrapper.unified_up_v,
            wrapper.projected_gram_v,
        ):
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _assert_ranks_match(value):
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    assert len(set(gathered)) == 1, gathered


def _broadcast_run_directory(rank, device):
    encoded = b""
    if rank == 0:
        encoded = tempfile.mkdtemp(prefix="cuo-lowrank-ddp-").encode("utf-8")
    length = torch.tensor([len(encoded)], dtype=torch.int64, device=device)
    dist.broadcast(length, src=0)
    path_bytes = torch.zeros(512, dtype=torch.uint8, device=device)
    if rank == 0:
        path_bytes[: len(encoded)] = torch.tensor(
            list(encoded), dtype=torch.uint8, device=device
        )
    dist.broadcast(path_bytes, src=0)
    return Path(bytes(path_bytes.cpu().tolist()[: int(length.item())]).decode("utf-8"))


def _calibrate_task(model, rank, task):
    torch.manual_seed(510 + task)
    with torch.no_grad():
        for weight in model.w_As:
            weight.weight.add_(0.1 * torch.randn_like(weight.weight))
        for weight in model.w_Bs:
            weight.weight.copy_(torch.randn_like(weight.weight))
        model.wrapped_param[0].param.fill_(0.45 + 0.1 * task)
    before_projection = [
        tensor.detach().clone()
        for block in model.lora_vit.blocks
        for tensor in (block.attn.qkv.projection_q, block.attn.qkv.projection_v)
    ]
    model.prepare_cuo_calibration()
    after_projection = [
        tensor.detach().clone()
        for block in model.lora_vit.blocks
        for tensor in (block.attn.qkv.projection_q, block.attn.qkv.projection_v)
    ]
    if task == 1:
        for before, after in zip(before_projection, after_projection):
            assert torch.equal(before, after)
    local_inputs = torch.randn(rank + 2, 3, 6, device=next(model.parameters()).device)
    with torch.no_grad():
        model(local_inputs)
    stats = model.finalize_cuo_calibration()
    expected_tokens = sum((other_rank + 2) * 3 for other_rank in range(dist.get_world_size()))
    if rank == 0:
        assert stats["token_count"] == expected_tokens
        assert stats["branch_count"] == 2
        assert stats["max_condition_number"] > 0.0
        assert stats["max_residual_norm"] >= 0.0
    _assert_ranks_match(_wrapper_state_hash(model))


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device = torch.device("cuda", local_rank)
    run_dir = _broadcast_run_directory(rank, device)
    try:
        model0 = _make_model(run_dir, task=0).to(device)
        _calibrate_task(model0, rank, task=0)
        if rank == 0:
            model0.save_lora_parameters(str(run_dir), task_id=0)
        dist.barrier()

        model1 = _make_model(run_dir, task=1).to(device)
        _calibrate_task(model1, rank, task=1)
        if rank == 0:
            model1.save_lora_parameters(str(run_dir), task_id=1)
            entries = {path.name for path in run_dir.iterdir()}
            assert entries == {SA_STATE_FILENAME, SA_MERGED_FILENAME}, entries
            assert not list(run_dir.glob("sa_lora_w_b_*.pt"))
            print(
                "CUO_DDP_SMOKE_PASS ranks={} final_hash={} artifacts={}".format(
                    dist.get_world_size(), _wrapper_state_hash(model1), sorted(entries)
                ),
                flush=True,
            )
        dist.barrier()
    finally:
        if rank == 0 and run_dir.exists():
            shutil.rmtree(run_dir)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
