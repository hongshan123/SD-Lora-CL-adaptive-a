"""Synthetic last-task memory gate; never used as an accuracy result."""
import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import timm
import torch
from torch import nn
from backbone.structure_ablation import make_structure_backbone
from utils.diagnostic_microbatch import microbatch_backward


def run(mode, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1993)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    a = [torch.linalg.qr(torch.randn(768, 10)).Q.T.contiguous() for _ in range(24)]
    if mode.endswith("bank"):
        entries = [{"down": [x.clone() for x in a] if mode == "independent_bank" else [],
                    "up": [torch.randn(768, 10)*.002 for _ in a],
                    "scale": torch.tensor([.8])} for _ in range(49)]
        state = dict(version=101, structure_mode=mode, shared_a=a, entries=entries, rank=10)
    else:
        state = dict(version=4, task_id=49, rank=10, shared_a=a,
                     aggregate_up=[torch.randn(768, 10)*.02 for _ in a],
                     merge_mode="live_a_aggregate_b", history_groups=1,
                     coordinate_aligned=mode == "hoep_fold",
                     absorb_mode="operator_preserving_absorb")
    torch.save(state, output/"sa_state.pt")
    del state
    if mode.endswith("bank"):
        del entries
    model = make_structure_backbone(
        structure_mode=mode, vit_model=timm.create_model("vit_base_patch16_224",
        pretrained=False, num_classes=0), r=10, filepath=str(output),
        cur_task_index=49, index=False, cumulative_state=True,
        cumulative_merge="live_a_aggregate_b",
        train_a_all_tasks=mode in ("hoep_fold", "independent_bank"),
        live_a_coordinate_align=mode == "hoep_fold",
        live_a_absorb_mode="operator_preserving_absorb",
        adaptive_a_enabled=mode == "hoep_fold",
        adaptive_a_strategy="operator_energy_partition", hoep_energy_budget=.05,
        hoep_diagnostics_interval=16).cuda()
    head = nn.Linear(768, 100).cuda()
    optimizer = torch.optim.SGD(list(model.parameters()) + list(head.parameters()),
                                lr=.008, momentum=.9)
    inputs = torch.randn(128, 3, 224, 224, device="cuda")
    targets = torch.arange(128, device="cuda") % 100
    results = []
    for micro in (64, 32, 16, 8):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                def loss(xx, yy):
                    logits = head(model(xx))
                    return logits, nn.functional.cross_entropy(logits, yy), {}
                microbatch_backward(loss, inputs, targets, micro)
                if mode == "hoep_fold":
                    model.apply_operator_energy_partition_gradients(optimizer)
                optimizer.step()
                if mode == "hoep_fold":
                    model.apply_operator_energy_partition_step(optimizer)
            with torch.no_grad():
                logits = head(model(inputs))
                assert logits.shape == (128, 100) and torch.isfinite(logits).all()
            torch.cuda.synchronize()
            allocated = torch.cuda.max_memory_allocated()
            reserved = torch.cuda.max_memory_reserved()
            fits = max(allocated, reserved) < 22*1024**3
            results.append(dict(microbatch=micro, allocated=allocated, reserved=reserved, fits=fits))
            if fits:
                record = dict(passed=True, resource_only=True, synthetic_history=True,
                              mode=mode, task=49, history_tasks=49, seen_classes=100,
                              global_batch=128, microbatch=micro, fp32=True,
                              optimizer_steps=2, trials=results)
                (output/"gate.json").write_text(json.dumps(record, indent=2)+"\n")
                return
        except torch.cuda.OutOfMemoryError:
            results.append(dict(microbatch=micro, fits=False, error="CUDA OOM"))
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
    raise RuntimeError("No tested microbatch fits the last-task memory gate: "+str(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.mode, args.output)
