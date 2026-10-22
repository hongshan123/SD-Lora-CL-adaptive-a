#!/usr/bin/env python3
"""Offline sequential calibration of fixed-size Live-A function statistics."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import timm
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backbone.functional_statistics import QKVFunctionStatistics
from backbone.sbgc import normalize_sensitivity, update_running_moment
from scripts.evaluate_sa_sdlora import build_merged_backbone
from utils.canonical_hash import hash_named_tensors
from utils.data_manager import DataManager
from utils.rng_utils import deterministic_loader, rng_preserving, rng_state_hash
from utils.sa_task_snapshots import audit_snapshot


def merge_task_statistics(history, current):
    layers = len(current["input_moments"])
    branches = len(current["output_sensitivities"])
    if branches != 2 * layers or len(current["input_counts"]) != layers or len(
        current["output_counts"]
    ) != branches:
        raise ValueError("current task has an incomplete Q/V branch statistic")
    if any(count <= 0 for count in current["input_counts"] + current["output_counts"]):
        raise ValueError("calibration counts must be strictly positive")
    if history is not None and (
        len(history["input_moments"]) != layers
        or len(history["output_sensitivities"]) != branches
    ):
        raise ValueError("historical and current Q/V branch counts differ")
    result = {
        "input_moments": [], "output_sensitivities": [],
        "input_counts": [], "output_counts": [],
    }
    for index, (moment, count) in enumerate(zip(
        current["input_moments"], current["input_counts"]
    )):
        if not bool(torch.isfinite(moment).all()) or bool((moment < 0).any()):
            raise ValueError("input second moment must be finite and nonnegative")
        value = moment.detach().float().clamp_min(1e-8)
        if history is None:
            result["input_moments"].append(value.clone())
            result["input_counts"].append(int(count))
        else:
            merged, total = update_running_moment(
                history["input_moments"][index], history["input_counts"][index],
                value, count,
            )
            result["input_moments"].append(merged)
            result["input_counts"].append(int(total))
    for index, (sensitivity, count) in enumerate(zip(
        current["output_sensitivities"], current["output_counts"]
    )):
        value = normalize_sensitivity(sensitivity.detach().float())
        if history is None:
            result["output_sensitivities"].append(value.clone())
            result["output_counts"].append(int(count))
        else:
            merged, total = update_running_moment(
                history["output_sensitivities"][index],
                history["output_counts"][index], value, count,
            )
            result["output_sensitivities"].append(merged)
            result["output_counts"].append(int(total))
    return result


def calibrate_task(snapshot, manager, batch_size, device, num_workers):
    before_rng = rng_state_hash()
    with rng_preserving():
        statistics, samples = _calibrate_task(
            snapshot, manager, batch_size, device, num_workers
        )
    if rng_state_hash() != before_rng:
        raise RuntimeError("offline calibration changed RNG state")
    statistics["rng_hash_preserved"] = True
    return statistics, samples


def _calibrate_task(snapshot, manager, batch_size, device, num_workers):
    task = int(snapshot.name.removeprefix("task_"))
    merged = torch.load(snapshot / "sa_merged_lora.pt", map_location="cpu", weights_only=True)
    prototypes = torch.load(snapshot / "sa_prototypes.pt", map_location="cpu", weights_only=True)
    if merged["task_id"] != task or sorted(prototypes) != list(range(len(prototypes))):
        raise ValueError("task artifact and prototype classes do not match")
    known = sum(manager.get_task_size(index) for index in range(task))
    total = known + manager.get_task_size(task)
    if len(prototypes) != total:
        raise ValueError("prototype count does not match task classes")
    dataset = manager.get_dataset(np.arange(known, total), source="train", mode="test")
    loader = deterministic_loader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        seed=7919 + task,
    )
    model = build_merged_backbone(
        timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0),
        merged,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before_model = hash_named_tensors(model.state_dict())
    weights = F.normalize(
        torch.stack([prototypes[index] for index in range(total)]).to(device).float(),
        dim=-1,
    )
    wrappers = [block.attn.qkv for block in model.blocks]
    with QKVFunctionStatistics(wrappers) as collector:
        for _, images, labels in loader:
            collector.begin_batch()
            features = model(images.to(device, non_blocking=True))
            logits = F.normalize(features, dim=-1) @ weights.T
            loss = F.cross_entropy(logits, labels.to(device))
            collector.accumulate(loss, images.shape[0])
        result = collector.summary()
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("offline calibration modified parameter gradients")
    if hash_named_tensors(model.state_dict()) != before_model:
        raise RuntimeError("offline calibration modified model tensors")
    result = {
        key: ([value.cpu() for value in values] if key in (
            "input_moments", "output_sensitivities"
        ) else values)
        for key, values in result.items()
    }
    result["model_tensor_hash_preserved"] = True
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, len(dataset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--through-task", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.through_task < 0 or args.batch_size <= 0:
        raise ValueError("through-task must be nonnegative and batch-size positive")
    history = None
    reference = None
    task_sizes = []
    task_audits = []
    for task in range(args.through_task + 1):
        snapshot = args.run_dir / "task_snapshots" / f"task_{task:03d}"
        audit_snapshot(snapshot)
        config = json.loads((snapshot / "config.json").read_text())
        seed = config["seed"]
        seed = int(seed[0] if isinstance(seed, list) else seed)
        signature = (
            config["dataset"], config["shuffle"], seed,
            config["init_cls"], config["increment"],
            tuple(config.get("task_increments", ())),
        )
        if reference is None:
            reference = signature
        elif reference != signature:
            raise ValueError("task snapshot protocols differ")
        manager = DataManager(
            config["dataset"], config["shuffle"], seed,
            config["init_cls"], config["increment"], config,
        )
        current, samples = calibrate_task(
            snapshot, manager, args.batch_size, torch.device(args.device),
            args.num_workers,
        )
        history = merge_task_statistics(history, current)
        task_sizes.append(samples)
        task_audits.append({
            "task": task,
            "model_tensor_hash_preserved": current["model_tensor_hash_preserved"],
            "rng_hash_preserved": current["rng_hash_preserved"],
        })
        cv = [
            float(value.std(unbiased=False) / value.mean())
            for value in history["output_sensitivities"]
        ]
        print(
            f"task={task} samples={samples} sensitivity_cv_mean="
            f"{sum(cv) / len(cv):.6f}", flush=True,
        )
    artifact = {
        "version": 1,
        "through_task": args.through_task,
        "dataset": reference[0],
        "seed": reference[2],
        "task_sample_counts": task_sizes,
        "calibration_audits": task_audits,
        "research_artifact_only": True,
        **history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    print(f"saved={args.output} layers={len(history['input_moments'])} "
          f"branches={len(history['output_sensitivities'])}")


if __name__ == "__main__":
    main()
