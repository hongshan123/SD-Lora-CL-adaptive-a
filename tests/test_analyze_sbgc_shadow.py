import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_sbgc_shadow import (
    evaluate_go_no_go,
    summarize_shadow_artifact,
)


def _write_artifact(path, distortion=0.05, gap=0.02, cv=0.5):
    tasks = []
    for task_id in range(10):
        branches = []
        for _ in range(4):
            branches.append(
                {
                    "sensitivity_cv": cv,
                    "candidate_relative_gap": gap,
                    "fisher": {
                        "constraint_active": task_id > 0,
                        "current_distortion": distortion if task_id > 0 else 0.0,
                    },
                    "uniform": {
                        "constraint_active": task_id > 0,
                        "current_distortion": 0.5 * distortion if task_id > 0 else 0.0,
                    },
                }
            )
        tasks.append(
            {
                "task_id": task_id,
                "shadow_only": True,
                "branches": branches,
            }
        )
    torch.save({"version": 1, "tasks": tasks}, path)


def test_preregistered_go_requires_every_check(tmp_path):
    summaries = []
    for name in ("c100", "inr", "cub"):
        path = tmp_path / f"{name}.pt"
        _write_artifact(path)
        summaries.append(summarize_shadow_artifact(name, path))
    result = evaluate_go_no_go(summaries, shadow_parity_verified=True)
    assert result["decision"] == "GO"
    assert all(result["checks"].values())


def test_large_current_distortion_forces_no_go(tmp_path):
    summaries = []
    for name in ("c100", "inr", "cub"):
        path = tmp_path / f"{name}.pt"
        _write_artifact(path, distortion=0.4)
        summaries.append(summarize_shadow_artifact(name, path))
    result = evaluate_go_no_go(summaries, shadow_parity_verified=True)
    assert result["decision"] == "NO_GO"
    assert not result["checks"]["median_fisher_distortion_at_most_10pct"]


def test_incomplete_artifact_is_rejected(tmp_path):
    path = tmp_path / "partial.pt"
    _write_artifact(path)
    artifact = torch.load(path, weights_only=False)
    artifact["tasks"].pop()
    torch.save(artifact, path)
    try:
        summarize_shadow_artifact("partial", path)
    except ValueError as error:
        assert "expected 9 transitions" in str(error)
    else:
        raise AssertionError("incomplete shadow artifact was accepted")
