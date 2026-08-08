"""Four-process gloo rank-sync check for prototypes and dual-head scalars."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from models.sa_sdlora import (
    broadcast_dual_head_values,
    broadcast_prototypes,
)


def _worker(rank, world_size, init_file):
    dist.init_process_group(
        backend="gloo",
        init_method="file://" + init_file,
        rank=rank,
        world_size=world_size,
    )
    if rank == 0:
        prototypes = {
            0: torch.tensor([1.0, 2.0, 3.0]),
            1: torch.tensor([4.0, 5.0, 6.0]),
        }
        lambda_val, tau_fc, tau_proto = 0.7, 0.3, 0.9
    else:
        prototypes = {}
        lambda_val, tau_fc, tau_proto = 0.0, 0.0, 0.0

    synced = broadcast_prototypes(prototypes, src=0)
    lambda_val, tau_fc, tau_proto = broadcast_dual_head_values(
        lambda_val, tau_fc, tau_proto, src=0
    )

    assert set(synced) == {0, 1}
    assert torch.equal(synced[0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(synced[1], torch.tensor([4.0, 5.0, 6.0]))
    assert lambda_val == 0.7
    assert tau_fc == 0.3
    assert tau_proto == 0.9
    dist.destroy_process_group()


def main():
    with tempfile.NamedTemporaryFile(delete=False) as handle:
        init_file = handle.name
    mp.spawn(_worker, args=(4, init_file), nprocs=4, join=True)


if __name__ == "__main__":
    main()
