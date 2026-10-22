import os
import sys
from pathlib import Path

import timm
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.proto_routed_lora import PrototypeRoutedLoRAViT


def build_model(root, task_id, device):
    torch.manual_seed(1234)
    base = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    return PrototypeRoutedLoRAViT(
        base,
        r=2,
        filepath=root,
        cur_task_index=task_id,
    ).to(device)


def main():
    dist.init_process_group("nccl", init_method="env://")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    root = os.environ["PROTO_ROUTER_SMOKE_DIR"]
    x = torch.randn(1, 3, 224, 224, device=device)

    model = build_model(root, 0, device)
    ddp = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
        broadcast_buffers=False,
    )
    ddp.train()
    ddp(x).mean().backward()
    assert all(layer.weight.grad is not None for layer in ddp.module.w_Bs)

    features = ddp.module.extract_router_features(x)
    sums = {key: value.sum(dim=0) for key, value in features.items()}
    count = torch.tensor([x.shape[0]], dtype=torch.float32, device=device)
    for value in sums.values():
        dist.all_reduce(value)
    dist.all_reduce(count)
    prototypes = {
        key: torch.nn.functional.normalize(value / count, dim=0)
        for key, value in sums.items()
    }
    if rank == 0:
        ddp.module.save_lora_parameters(root, 0)
    dist.barrier()
    ddp.module.append_task_prototypes(
        0, prototypes, (0, 10), save=rank == 0
    )
    dist.barrier()
    ddp.eval()
    with torch.no_grad():
        ddp.module.prepare_routing(
            "selected", torch.zeros(x.shape[0], dtype=torch.long, device=device)
        )
        reference_output = ddp.module(x)

    del ddp, model
    torch.cuda.empty_cache()
    model = build_model(root, 1, device)
    model.eval()
    with torch.no_grad():
        model.prepare_routing(
            "selected", torch.zeros(x.shape[0], dtype=torch.long, device=device)
        )
        restored_output = model(x)
    assert torch.allclose(reference_output, restored_output, atol=1e-5, rtol=1e-5)
    model.train()
    ddp = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
        broadcast_buffers=False,
    )
    ddp.train()
    ddp(x).mean().backward()
    assert ddp.module.router_state["task_ids"] == [0]
    assert all(layer.weight.grad is not None for layer in ddp.module.w_Bs)

    dist.barrier()
    if rank == 0:
        print("PASS four-rank DDP task0/task1 smoke")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
