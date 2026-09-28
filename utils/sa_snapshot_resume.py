"""Restore a completed deterministic task boundary into a fresh run."""

import json
import re
import shutil
from pathlib import Path

import torch
import torch.distributed as dist

from backbone.sa_lora import SA_STATE_VERSION_LIVE_A, SA_STATE_VERSION_SBGC_GLOBAL
from utils.sa_task_snapshots import audit_snapshot


_RUNTIME_KEYS = {
    "config", "prefix", "filepath", "device", "distributed", "rank",
    "local_rank", "world_size", "_p0_run_guard_lock", "nb_classes", "nb_tasks",
    "sa_resume_snapshot",
}
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _previous_metrics(config, task_count):
    previous = {"top1": [], "top5": [], "matrix": []}
    if config.get("sa_resume_snapshot"):
        parent = Path(config["sa_resume_snapshot"])
        parent_config = json.loads((parent / "config.json").read_text())
        parent_state = torch.load(parent / "sa_state.pt", map_location="cpu", weights_only=True)
        previous = _previous_metrics(parent_config, int(parent_state["task_id"]))
    prior_count = len(previous["top1"])
    remaining = task_count - prior_count
    if remaining <= 0:
        raise ValueError("resume metric history has a non-increasing task count")
    init_cls = 0 if config["init_cls"] == config["increment"] else config["init_cls"]
    log_path = Path("logs") / config["model_name"] / config["dataset"] / str(init_cls)
    log_path = log_path / str(config["increment"]) / (
        f"{config['prefix']}_{config['seed']}_{config['backbone_type']}.log"
    )
    if not log_path.is_file():
        raise FileNotFoundError(f"resume requires the source accuracy log: {log_path}")
    top1, top5, grouped = [], [], []
    for line in log_path.read_text(errors="replace").splitlines():
        if "CNN top1 curve:" in line:
            top1.append([float(value) for value in re.findall(
                rf"np\.float64\(({_NUMBER})\)", line.split("CNN top1 curve:", 1)[1]
            )])
        elif "CNN top5 curve:" in line:
            top5.append([float(value) for value in re.findall(
                rf"np\.float64\(({_NUMBER})\)", line.split("CNN top5 curve:", 1)[1]
            )])
        elif "=> CNN: {" in line:
            values = re.findall(rf"'(\d+-\d+)': np\.float64\(({_NUMBER})\)", line)
            grouped.append([float(value) for _, value in sorted(values)])
    if len(top1) < remaining or len(top5) < remaining or len(grouped) < remaining:
        raise ValueError("source log lacks complete metrics for the resume boundary")
    if len(top1[remaining - 1]) != task_count or len(top5[remaining - 1]) != task_count:
        raise ValueError("source curves do not end at the completed task")
    if any(len(row) != prior_count + index + 1 for index, row in enumerate(grouped[:remaining])):
        raise ValueError("source grouped accuracy rows are incomplete")
    return {
        "top1": top1[remaining - 1],
        "top5": top5[remaining - 1],
        "matrix": previous["matrix"] + grouped[:remaining],
    }


def restore_sbgc_snapshot(args, learner, data_manager):
    """Compatibility entry point retaining the original global-SBGC guard."""
    if not args.get("sa_resume_snapshot"):
        return 0, None
    if args.get("model_name") != "sa_sdlora" or args.get(
        "sa_cumulative_merge"
    ) != "sensitivity_budgeted_g" or args.get("sa_g_budget_scope") != "global":
        raise ValueError("snapshot resume is restricted to global SBGC")
    return _restore_completed_snapshot(args, learner, data_manager, live=False)


def restore_task_snapshot(args, learner, data_manager):
    """Dispatch the audited global-SBGC or aligned Live-A boundary restore."""
    if not args.get("sa_resume_snapshot"):
        return 0, None
    if args.get("sa_cumulative_merge") != "live_a_aggregate_b":
        return restore_sbgc_snapshot(args, learner, data_manager)
    if (
        args.get("model_name") != "sa_sdlora"
        or not args.get("sa_cumulative_state", False)
        or not args.get("sa_deterministic_training", False)
        or args.get("sa_adaptive_a_enabled", False)
        or args.get("sa_dual_head", False)
        or args.get("sa_live_a_boundary_merge", "aligned") != "aligned"
        or args.get("sa_live_a_history_forward", "shared") != "shared"
    ):
        raise ValueError("Live-A snapshot resume requires deterministic aligned state without controllers")
    return _restore_completed_snapshot(args, learner, data_manager, live=True)


def _prefix_a_trainable(config, task):
    freeze_after = config.get("sa_freeze_a_after_tasks")
    if freeze_after is not None and (
        isinstance(freeze_after, bool) or not isinstance(freeze_after, int)
        or freeze_after <= 0
    ):
        raise ValueError("sa_freeze_a_after_tasks must be a positive integer")
    return (task == 0 or config.get("sa_train_a_all_tasks", False)) and (
        freeze_after is None or task < freeze_after
    )


def _restore_completed_snapshot(args, learner, data_manager, live):
    snapshot_arg = args.get("sa_resume_snapshot")
    if args.get("sa_resume", False):
        raise ValueError("snapshot resume requires a fresh output directory")

    snapshot = Path(snapshot_arg).resolve()
    audit = audit_snapshot(snapshot)
    if not audit["verified"]:
        raise ValueError("resume requires a finalized snapshot")
    source_config = json.loads((snapshot / "config.json").read_text())
    allowed_changes = _RUNTIME_KEYS | (
        {"max_tasks", "sa_freeze_a_after_tasks", "sa_train_a_all_tasks"}
        if live else set()
    )
    protocol_keys = source_config.keys() | args.keys() if live else source_config.keys()
    for key in protocol_keys:
        if key not in allowed_changes and args.get(key) != source_config.get(key):
            raise ValueError(f"resume protocol differs at {key}")
    state = torch.load(snapshot / "sa_state.pt", map_location="cpu", weights_only=True)
    if live:
        if state.get("version") != SA_STATE_VERSION_LIVE_A or state.get("merge_mode") != "live_a_aggregate_b":
            raise ValueError("resume snapshot is not aligned Live-A v4")
    elif state.get("version") != SA_STATE_VERSION_SBGC_GLOBAL or state.get("budget_scope") != "global":
        raise ValueError("resume snapshot is not global SBGC v8")
    start_task = int(state["task_id"])
    if start_task != int(audit["task_id"]) + 1 or not (
        0 < start_task < min(data_manager.nb_tasks, int(args.get("max_tasks", data_manager.nb_tasks)))
    ):
        raise ValueError("resume task index does not match the completed snapshot")
    if live and any(
        _prefix_a_trainable(source_config, task) != _prefix_a_trainable(args, task)
        for task in range(start_task)
    ):
        raise ValueError("resume prefix A policy differs from the source trajectory")
    completed_classes = sum(data_manager.get_task_size(i) for i in range(start_task))
    weight_name = f"CLs_weight{start_task - 1}.pt"
    bias_name = f"CLs_bias{start_task - 1}.pt"
    weight = torch.load(snapshot / weight_name, map_location="cpu", weights_only=True)
    bias = torch.load(snapshot / bias_name, map_location="cpu", weights_only=True)
    prototypes = torch.load(snapshot / "sa_prototypes.pt", map_location="cpu", weights_only=True)
    if weight.shape != (completed_classes, learner.feature_dim) or bias.shape != (completed_classes,):
        raise ValueError("snapshot FC shape does not match the completed classes")
    if set(prototypes) != set(range(completed_classes)):
        raise ValueError("snapshot prototypes do not cover exactly the completed classes")
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all() or not all(
        torch.isfinite(value).all() for value in prototypes.values()
    ):
        raise ValueError("snapshot classifier or prototypes are non-finite")
    metrics = _previous_metrics(source_config, start_task)

    output_dir = Path(args["filepath"]).resolve()
    names = ("sa_state.pt", "sa_merged_lora.pt", "sa_prototypes.pt", weight_name, bias_name)
    if any((output_dir / name).exists() for name in names):
        raise FileExistsError("fresh resume output already contains checkpoint artifacts")
    if args.get("rank", 0) == 0:
        for name in names:
            shutil.copy2(snapshot / name, output_dir / name)
        if live:
            fingerprint = json.loads((snapshot / "complete.json").read_text())["sha256"]
            (output_dir / "snapshot_resume.json").write_text(json.dumps({
                "source_snapshot": str(snapshot), "sha256": fingerprint,
            }, indent=2) + "\n")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    network = learner._raw_network()
    network.backbone = learner.update_network(index=False, task_index=start_task)
    network.update_fc(completed_classes)
    with torch.no_grad():
        network.fc.weight.copy_(weight)
        network.fc.bias.copy_(bias)
    network.set_prototypes(prototypes)
    network.to(learner._device)
    learner._cur_task = start_task - 1
    learner._known_classes = completed_classes
    learner._total_classes = completed_classes
    return start_task, metrics
