import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.class_proto_routed_lora import ClassPrototypeRoutedLoRAViT
from tests.test_class_proto_routed_lora import TinyViT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    return parser.parse_args()


def build_model(root, task_id):
    torch.manual_seed(1234)
    return ClassPrototypeRoutedLoRAViT(
        TinyViT(), r=2, filepath=root, cur_task_index=task_id, increment=2
    )


def aggregate_current_statistics(model, task_id, rank):
    feature = model.extract_router_features(torch.full((1, 4), float(rank + 1)))[0]
    local_class = rank % 2
    sums = torch.zeros(2, 4)
    counts = torch.zeros(2)
    sums[local_class] = feature
    counts[local_class] = 1
    dist.all_reduce(sums)
    dist.all_reduce(counts)
    model.append_class_statistics(
        task_id,
        sums,
        counts,
        (task_id * 2, task_id * 2 + 2),
        save=rank == 0,
    )


def main():
    args = parse_args()
    dist.init_process_group("gloo", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 4:
        raise RuntimeError("this smoke test intentionally requires four ranks")
    os.makedirs(args.root, exist_ok=True)
    inputs = torch.randn(2, 4)

    task0 = build_model(args.root, 0)
    ddp0 = DDP(task0, find_unused_parameters=False, broadcast_buffers=False)
    ddp0(inputs).sum().backward()
    assert all(layer.weight.grad is not None for layer in ddp0.module.w_Bs)
    if rank == 0:
        ddp0.module.save_lora_parameters(args.root, 0)
    dist.barrier()
    aggregate_current_statistics(ddp0.module, 0, rank)
    dist.barrier()

    task1 = build_model(args.root, 1)
    assert task1.router_state["task_ids"] == [0]
    ddp1 = DDP(task1, find_unused_parameters=False, broadcast_buffers=False)
    ddp1(inputs).sum().backward()
    assert all(layer.weight.grad is not None for layer in ddp1.module.w_Bs)
    if rank == 0:
        ddp1.module.save_lora_parameters(args.root, 1)
    dist.barrier()
    aggregate_current_statistics(ddp1.module, 1, rank)
    dist.barrier()

    restored = ClassPrototypeRoutedLoRAViT(
        TinyViT(),
        r=2,
        filepath=args.root,
        inference_only=True,
    )
    restored.eval()
    assert restored.router_state["task_ids"] == [0, 1]
    assert not any(parameter.requires_grad for parameter in restored.parameters())
    selected, _, _ = restored.route_from_features(
        restored.extract_router_features(inputs), topk=2
    )
    gathered = [torch.empty_like(selected) for _ in range(world_size)]
    dist.all_gather(gathered, selected)
    assert all(torch.equal(gathered[0], value) for value in gathered[1:])

    if rank == 0:
        print("PASS four-rank class prototype router task0/task1 smoke")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
