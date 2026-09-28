"""Compact, task-local snapshots for read-only LoRA boundary analysis."""

import json
import hashlib
import os
import shutil
from pathlib import Path

import torch

from backbone.sa_lora import _LiveAAggregateQKV, _SensitivityBudgetedGQKV


def _cpu(tensor):
    return tensor.detach().cpu().clone()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_snapshot(directory, finalize=False):
    directory = Path(directory)
    task_id = int(directory.name.removeprefix("task_"))
    required = (
        "pre_merge.pt", "config.json", "sa_state.pt", "sa_merged_lora.pt",
        "sa_prototypes.pt", f"CLs_weight{task_id}.pt", f"CLs_bias{task_id}.pt",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete snapshot {directory}: {missing}")
    hashes = {
        path.name: _sha256(path)
        for path in directory.iterdir()
        if path.name in required or path.name == "run_manifest.json"
    }
    completion = directory / "complete.json"
    if completion.is_file():
        recorded = json.loads(completion.read_text())
        if recorded != {"task_id": task_id, "sha256": hashes}:
            raise ValueError(f"snapshot checksum mismatch: {directory}")
    elif finalize:
        temporary = directory / "complete.json.tmp"
        temporary.write_text(json.dumps({"task_id": task_id, "sha256": hashes}, indent=2) + "\n")
        os.replace(temporary, completion)
    return {"task_id": task_id, "files": len(hashes), "verified": completion.is_file()}


def capture_pre_merge(backbone, fc, old_prototypes, task_id, known_classes, total_classes):
    wrapper_type = {
        "live_a_aggregate_b": _LiveAAggregateQKV,
        "sensitivity_budgeted_g": _SensitivityBudgetedGQKV,
    }.get(backbone.cumulative_merge)
    if wrapper_type is None:
        raise ValueError("task snapshots do not support this cumulative merge")
    wrappers = [
        block.attn.qkv for block in backbone.lora_vit.blocks
        if isinstance(block.attn.qkv, wrapper_type)
    ]
    if len(wrappers) != len(backbone.lora_vit.blocks):
        raise RuntimeError("every attention block must have the configured QKV wrapper")
    anchored_history = (
        wrapper_type is _LiveAAggregateQKV and wrappers[0].history_forward == "anchored"
    )
    branches = []
    for wrapper in wrappers:
        for suffix in ("q", "v"):
            branch = {
                "down": _cpu(getattr(wrapper, "a_" + suffix).weight),
                "current_up": _cpu(getattr(wrapper, "b_" + suffix).weight),
            }
            if backbone.cumulative_merge == "sensitivity_budgeted_g":
                branch["historical_down"] = _cpu(
                    getattr(wrapper, "projection_" + suffix)
                )
                branch["historical_up"] = _cpu(
                    getattr(wrapper, "unified_up_" + suffix)
                )
                if getattr(wrapper, "train_projected", False):
                    with torch.no_grad():
                        effective_up, alpha, raw_risk = wrapper.projected_current_up(
                            suffix
                        )
                    branch["current_effective_up"] = _cpu(effective_up)
                    branch["training_projection_alpha"] = _cpu(alpha)
                    branch["raw_historical_risk"] = _cpu(raw_risk)
            elif anchored_history:
                branch["historical_down"] = _cpu(
                    getattr(wrapper, "recoverability_anchor_a_" + suffix)
                )
                branch["historical_up"] = _cpu(
                    getattr(wrapper, "recoverability_anchor_up_" + suffix)
                )
            else:
                branch["historical_up"] = _cpu(
                    getattr(wrapper, "aggregate_" + suffix)
                )
            branches.append(branch)
    snapshot = {
        "version": (
            2 if backbone.cumulative_merge == "sensitivity_budgeted_g"
            else 3 if anchored_history else 1
        ),
        "stage": "pre_merge",
        "task_id": int(task_id),
        "known_classes": int(known_classes),
        "total_classes": int(total_classes),
        "branches": branches,
        "current_scale": _cpu(backbone.wrapped_param[0].param).reshape(()),
        "normalize_current_branch": bool(
            getattr(wrappers[0], "normalize_current_branch", False)
        ),
        "fc_weight": _cpu(fc.weight),
        "fc_bias": _cpu(fc.bias),
        "old_prototypes": {
            int(key): _cpu(value) for key, value in old_prototypes.items()
        },
    }
    if snapshot["version"] in (2, 3):
        snapshot["merge_mode"] = backbone.cumulative_merge
    if anchored_history:
        snapshot["history_forward"] = "anchored"
    return snapshot


def save_pre_merge(run_dir, backbone, fc, old_prototypes, task_id,
                   known_classes, total_classes, config):
    directory = Path(run_dir) / "task_snapshots" / f"task_{task_id:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = capture_pre_merge(
        backbone, fc, old_prototypes, task_id, known_classes, total_classes
    )
    temporary = directory / "pre_merge.pt.tmp"
    torch.save(snapshot, temporary)
    os.replace(temporary, directory / "pre_merge.pt")
    config_path = directory / "config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(config, indent=2, default=str) + "\n")
    return directory


def save_post_merge(run_dir, task_id, config):
    run_dir = Path(run_dir)
    directory = run_dir / "task_snapshots" / f"task_{task_id:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    files = (
        "sa_state.pt", "sa_merged_lora.pt", "sa_prototypes.pt",
        f"CLs_weight{task_id}.pt", f"CLs_bias{task_id}.pt",
    )
    if (run_dir / "run_manifest.json").is_file():
        files += ("run_manifest.json",)
    for name in files:
        source = run_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"task snapshot requires {source}")
        temporary = directory / (name + ".tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, directory / name)
    (directory / "config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n"
    )
    audit_snapshot(directory, finalize=True)
    return directory
