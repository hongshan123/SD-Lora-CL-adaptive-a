"""Two-rank NCCL smoke for HOEP-A partition, retraction, and rebuild."""

import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.sa_lora import SharedALoRA_ViT_timm


class _TinyAttention(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.qkv = nn.Linear(dimension, 3 * dimension, bias=False)


class _TinyBlock(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.attn = _TinyAttention(dimension)


class _TinyViT(nn.Module):
    def __init__(self, dimension=6):
        super().__init__()
        self.blocks = nn.ModuleList([_TinyBlock(dimension)])
        self.head = nn.Identity()

    def forward(self, inputs):
        for block in self.blocks:
            inputs = block.attn.qkv(inputs)
        return inputs


def _model(run_dir, task):
    torch.manual_seed(731)
    return SharedALoRA_ViT_timm(
        _TinyViT(),
        r=3,
        filepath=str(run_dir),
        cur_task_index=task,
        train_a_all_tasks=True,
        cumulative_state=True,
        cumulative_merge="live_a_aggregate_b",
        live_a_coordinate_align=True,
        live_a_absorb_mode="operator_preserving_absorb",
        adaptive_a_enabled=True,
        adaptive_a_strategy="operator_energy_partition",
        hoep_energy_budget=0.5,
    )


def _hash_state(model):
    digest = hashlib.sha256()
    tensors = [module.weight for module in model.w_As]
    tensors += [module.weight for module in model.w_Bs]
    tensors += [
        tensor
        for block in model.lora_vit.blocks
        for tensor in (
            block.attn.qkv.aggregate_q,
            block.attn.qkv.aggregate_v,
        )
    ]
    for tensor in tensors:
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _assert_rank_equal(value):
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    assert len(set(gathered)) == 1, gathered


def _broadcast_run_dir(rank, device):
    encoded = (
        tempfile.mkdtemp(prefix="hoep-ddp-").encode("utf-8")
        if rank == 0
        else b""
    )
    length = torch.tensor([len(encoded)], dtype=torch.int64, device=device)
    dist.broadcast(length, src=0)
    payload = torch.zeros(512, dtype=torch.uint8, device=device)
    if rank == 0:
        payload[: len(encoded)] = torch.tensor(
            list(encoded), dtype=torch.uint8, device=device
        )
    dist.broadcast(payload, src=0)
    raw = bytes(payload.cpu().tolist()[: int(length.item())])
    return Path(raw.decode("utf-8"))


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = torch.device("cuda", local_rank)
    run_dir = _broadcast_run_dir(rank, device)
    try:
        task_zero = _model(run_dir, 0)
        with torch.no_grad():
            for module in task_zero.w_As:
                module.weight.add_(0.02 * torch.randn_like(module.weight))
            for module in task_zero.w_Bs:
                module.weight.normal_()
        dist.barrier()
        if rank == 0:
            task_zero.save_lora_parameters(str(run_dir), 0)
        dist.barrier()

        model = _model(run_dir, 1).to(device)
        distributed = DDP(model, device_ids=[local_rank])
        optimizer = torch.optim.SGD(
            [parameter for parameter in distributed.parameters() if parameter.requires_grad],
            lr=0.01,
            momentum=0.9,
            weight_decay=0.0002,
        )
        for step in range(3):
            torch.manual_seed(900 + 10 * rank + step)
            optimizer.zero_grad(set_to_none=True)
            loss = distributed(torch.randn(4, 3, 6, device=device)).square().mean()
            loss.backward()
            model.apply_operator_energy_partition_gradients(optimizer)
            optimizer.step()
            diagnostics = model.apply_operator_energy_partition_step(optimizer)
            assert diagnostics["max_historical_operator_error"] < 1e-5
            assert diagnostics["max_current_operator_error"] < 1e-5
            _assert_rank_equal(_hash_state(model))

        if rank == 0:
            model.save_lora_parameters(str(run_dir), 1)
        dist.barrier()
        rebuilt = _model(run_dir, 2).to(device)
        _assert_rank_equal(_hash_state(rebuilt))
        if rank == 0:
            print(
                "HOEP_DDP_SMOKE_PASS ranks={} hash={}".format(
                    dist.get_world_size(), _hash_state(rebuilt)
                ),
                flush=True,
            )
    finally:
        dist.barrier()
        if rank == 0 and run_dir.exists():
            shutil.rmtree(run_dir)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
