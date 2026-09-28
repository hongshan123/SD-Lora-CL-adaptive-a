"""Audit a complete same-prefix continuation without changing the model."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.sa_snapshot_resume import _previous_metrics
from utils.sa_task_snapshots import audit_snapshot


def verify_fork(run_dir, expected_source, expected_policy):
    if expected_policy not in {"freeze", "live"}:
        raise ValueError("expected policy must be freeze or live")
    run_dir = Path(run_dir)
    snapshots = sorted((run_dir / "task_snapshots").glob("task_[0-9][0-9][0-9]"))
    if not snapshots:
        raise ValueError("continuation contains no snapshots")
    config = json.loads((snapshots[-1] / "config.json").read_text())
    source = Path(config["sa_resume_snapshot"]).resolve()
    if source != Path(expected_source).resolve():
        raise ValueError("continuation source differs from the requested prefix")
    source_audit = audit_snapshot(source)
    if not source_audit["verified"]:
        raise ValueError("prefix is not finalized")
    source_state = torch.load(source / "sa_state.pt", map_location="cpu", weights_only=True)
    first_task = int(source_state["task_id"])
    total_tasks = int(config["max_tasks"])
    if [int(path.name.removeprefix("task_")) for path in snapshots] != list(range(first_task, total_tasks)):
        raise ValueError("continuation is missing a completed task")
    frozen = expected_policy == "freeze"
    if not config.get("sa_train_a_all_tasks") or config.get("sa_freeze_a_after_tasks") != (
        first_task if frozen else None
    ):
        raise ValueError("continuation policy differs from the requested arm")
    lineage = json.loads((run_dir / "snapshot_resume.json").read_text())
    fingerprint = json.loads((source / "complete.json").read_text())["sha256"]
    if Path(lineage["source_snapshot"]).resolve() != source or lineage["sha256"] != fingerprint:
        raise ValueError("import fingerprint differs from the finalized prefix")
    counts = []
    for path in snapshots:
        if not audit_snapshot(path)["verified"]:
            raise ValueError("continuation snapshot is not finalized")
        state = torch.load(path / "sa_state.pt", map_location="cpu", weights_only=True)
        if len(state["shared_a"]) != 24 or len(state["aggregate_up"]) != 24:
            raise ValueError("expected 24 Q/V branches")
        tensors = state["shared_a"] + state["aggregate_up"]
        if not all(torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("non-finite adaptation state")
        counts.append(sum(tensor.numel() for tensor in tensors))
        if frozen and not all(torch.equal(old, new) for old, new in zip(
            source_state["shared_a"], state["shared_a"]
        )):
            raise ValueError("Frozen suffix changed its shared A")
    if set(counts) != {368640}:
        raise ValueError("persistent adaptation state differs from rank-10 budget")
    metrics = _previous_metrics(config, total_tasks)
    return {
        "verified": True, "source_snapshot": str(source.resolve()),
        "source_sha256": fingerprint, "policy": expected_policy,
        "suffix_tasks": [first_task, total_tasks - 1],
        "frozen_a_bit_identical": True if frozen else None,
        "persistent_adaptation_numel": counts[0],
        "top1": metrics["top1"], "top5": metrics["top5"],
        "final": metrics["top1"][-1],
        "aaa": sum(metrics["top1"]) / total_tasks,
    }


def verify_pair(freeze_run, live_run, expected_source):
    freeze = verify_fork(freeze_run, expected_source, "freeze")
    live = verify_fork(live_run, expected_source, "live")
    prefix_count = freeze["suffix_tasks"][0]
    if freeze["source_sha256"] != live["source_sha256"] or freeze["suffix_tasks"] != live["suffix_tasks"]:
        raise ValueError("paired continuations have different source fingerprints or task ranges")
    for metric in ("top1", "top5"):
        if freeze[metric][:prefix_count] != live[metric][:prefix_count]:
            raise ValueError("paired continuations have different prefix metrics")
    return {"paired_verified": True, "freeze": freeze, "live": live}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--policy", choices=("freeze", "live"), required=True)
    parser.add_argument("--paired-run", type=Path)
    options = parser.parse_args()
    if options.paired_run:
        freeze_run, live_run = (
            (options.run_dir, options.paired_run) if options.policy == "freeze"
            else (options.paired_run, options.run_dir)
        )
        result = verify_pair(freeze_run, live_run, options.source)
    else:
        result = verify_fork(options.run_dir, options.source, options.policy)
    print(json.dumps(result, indent=2))
