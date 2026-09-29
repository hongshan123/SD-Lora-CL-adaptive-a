#!/usr/bin/env python3
"""Read-only real-artifact checks before launching the ImageNet-A study."""

import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.imageneta_live_frozen import inspect_job, validate_seal, write_json


def compare_policy_tensors(frozen0, frozen1, live0, live1):
    import torch

    if not (len(frozen0) == len(frozen1) == len(live0) == len(live1)):
        raise ValueError("branch count mismatch")
    if not all(torch.equal(a, b) for a, b in zip(frozen0, live0)):
        raise ValueError("Task0 A tensors differ between paired arms")
    if not all(torch.equal(a, b) for a, b in zip(frozen0, frozen1)):
        raise ValueError("Frozen A changed during Task1")
    changes = [float((a - b).abs().max()) for a, b in zip(live0, live1)]
    if max(changes) <= 0:
        raise ValueError("Live A did not change during Task1")
    return dict(passed=True, frozen_max_abs_diff=0.0, live_branch_max_abs_diff=changes)


def verify(runtime, device="cuda:0"):
    import timm
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms
    from backbone.sa_lora import SharedALoRA_ViT_timm
    from scripts.evaluate_sa_sdlora import build_merged_backbone
    from utils.canonical_hash import hash_named_tensors
    from utils.data import build_transform

    runtime = Path(runtime).resolve()
    manifest = json.loads((runtime / "manifest.json").read_text())
    validate_seal(runtime, manifest)
    jobs = manifest["smoke"]
    if not all(inspect_job(job, verify_artifacts=True)["complete"] for job in jobs):
        raise ValueError("both smoke runs must finish successfully with verified snapshots")
    snapshots = [[Path(job["result"]) / "task_snapshots" / f"task_{task:03d}" for task in range(2)] for job in jobs]
    load = lambda path: torch.load(path, map_location="cpu", weights_only=True)
    states = [[load(path / "sa_state.pt") for path in dirs] for dirs in snapshots]
    result = compare_policy_tensors(states[0][0]["shared_a"], states[0][1]["shared_a"], states[1][0]["shared_a"], states[1][1]["shared_a"])
    for field in ("shared_a", "aggregate_up"):
        if not all(torch.equal(a, b) for a, b in zip(states[0][0][field], states[1][0][field])):
            raise ValueError(f"Task0 paired state differs: {field}")
    for job in jobs:
        text = Path(job["log"]).read_text()
        if text.count("[PrototypeSync] rank consistency PASS") != 2:
            raise ValueError("DDP prototype rank consistency was not verified twice")
    hashes = [json.loads((Path(job["result"]) / "p0_hashes.json").read_text()) for job in jobs]
    if hashes[0]["0"] != hashes[1]["0"]:
        raise ValueError("Task0 tensor/RNG hashes differ between paired arms")
    metrics = [inspect_job(job) for job in jobs]
    if metrics[0]["curve"][0] != metrics[1]["curve"][0]:
        raise ValueError("Task0 paired accuracy differs")
    prototypes0 = [load(paths[0] / "sa_prototypes.pt") for paths in snapshots]
    if set(prototypes0[0]) != set(prototypes0[1]) or not all(torch.equal(prototypes0[0][key], prototypes0[1][key]) for key in prototypes0[0]):
        raise ValueError("Task0 paired prototype tensors differ")

    torch.set_num_threads(1)
    torch.manual_seed(1995)
    base = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0).eval()
    result["base_weight_sha256"] = hash_named_tensors(dict(base.state_dict()))
    audit = json.loads((runtime / "data_audit.json").read_text())
    transform = transforms.Compose(build_transform(False, None))
    files = [item["path"] for item in audit["files"] if item["split"] == "test"][:8]
    inputs = []
    for filename in files:
        with Image.open(Path(audit["root"]) / "test" / filename) as image:
            inputs.append(transform(image.convert("RGB")))
    tensor = torch.stack(inputs).to(device)
    result["rebuild"] = []
    with torch.no_grad():
        for job, paths in zip(jobs, snapshots):
            config = json.loads(Path(job["config"]).read_text())
            for task, path in enumerate(paths):
                persistent = SharedALoRA_ViT_timm(
                    vit_model=copy.deepcopy(base), r=10, increment=20,
                    index=False, filepath=str(path), cur_task_index=task + 1,
                    shared_a_orthogonal=True,
                    train_a_all_tasks=config["sa_train_a_all_tasks"],
                    cumulative_state=True, cumulative_merge="live_a_aggregate_b",
                    live_a_coordinate_align=True,
                    live_a_absorb_mode="bounded_norm_calibrated_absorb",
                ).to(device).eval()
                merged = build_merged_backbone(copy.deepcopy(base), load(path / "sa_merged_lora.pt")).to(device).eval()
                reference = persistent(tensor)
                rebuilt = merged(tensor)
                prototypes = load(path / "sa_prototypes.pt")
                weight = F.normalize(torch.stack([prototypes[key] for key in sorted(prototypes)]).to(device), dim=1)
                logits0 = F.normalize(reference, dim=1) @ weight.t()
                logits1 = F.normalize(rebuilt, dim=1) @ weight.t()
                feature_error = float((reference - rebuilt).abs().max())
                logit_error = float((logits0 - logits1).abs().max())
                if feature_error >= 1e-5 or logit_error >= 1e-5:
                    raise ValueError(f"persistent/merged rebuild mismatch: {feature_error}, {logit_error}")
                result["rebuild"].append(dict(method=job["method"], task=task, feature_max_abs_diff=feature_error, logits_max_abs_diff=logit_error, adaptation_scalars=sum(a.numel() + g.numel() for a, g in zip(states[0][task]["shared_a"], states[0][task]["aggregate_up"]))))
                del persistent, merged, reference, rebuilt
    result.update(git_commit=manifest["git_commit"], data_sha256=manifest["data_sha256"], task0_hashes=hashes[0]["0"], task0_top1=metrics[0]["curve"][0], ddp_prototype_consistency=True, evaluation_files=files)
    write_json(runtime / "smoke_verification.json", result, exclusive=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--device", default="cuda:0")
    cli = parser.parse_args()
    print(json.dumps(verify(cli.runtime, cli.device), indent=2))
