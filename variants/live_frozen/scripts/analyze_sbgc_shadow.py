#!/usr/bin/env python3
"""Apply the preregistered SBGC P1 Go/No-Go criteria to shadow artifacts."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


def _is_finite_tree(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_is_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_is_finite_tree(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def summarize_shadow_artifact(label, path, expected_transitions=9):
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    tasks = sorted(artifact.get("tasks", []), key=lambda item: int(item["task_id"]))
    transitions = [item for item in tasks if int(item["task_id"]) > 0]
    if len(transitions) != int(expected_transitions):
        raise ValueError(
            f"{label}: expected {expected_transitions} transitions, "
            f"found {len(transitions)}"
        )
    if not all(bool(item.get("shadow_only", False)) for item in tasks):
        raise ValueError(f"{label}: artifact contains a non-shadow task")
    if not _is_finite_tree(artifact):
        raise ValueError(f"{label}: artifact contains non-finite values")

    active_transitions = 0
    different_transitions = 0
    branch_cvs = []
    fisher_distortions = []
    uniform_distortions = []
    transition_rows = []
    for transition in transitions:
        branches = transition.get("branches") or []
        if not branches:
            raise ValueError(f"{label}: task {transition['task_id']} has no branches")
        fisher_active = any(
            bool(branch["fisher"]["constraint_active"]) for branch in branches
        )
        mean_gap = float(
            np.mean([branch["candidate_relative_gap"] for branch in branches])
        )
        active_transitions += int(fisher_active)
        different_transitions += int(mean_gap > 1e-3)
        branch_cvs.extend(float(branch["sensitivity_cv"]) for branch in branches)
        fisher_distortions.extend(
            float(branch["fisher"]["current_distortion"]) for branch in branches
        )
        uniform_distortions.extend(
            float(branch["uniform"]["current_distortion"]) for branch in branches
        )
        transition_rows.append(
            {
                "task_id": int(transition["task_id"]),
                "fisher_constraint_active": fisher_active,
                "mean_candidate_relative_gap": mean_gap,
                "median_fisher_distortion": float(
                    np.median(
                        [
                            branch["fisher"]["current_distortion"]
                            for branch in branches
                        ]
                    )
                ),
            }
        )

    count = len(transitions)
    return {
        "label": label,
        "path": str(path),
        "transition_count": count,
        "active_transition_fraction": active_transitions / count,
        "different_transition_fraction": different_transitions / count,
        "high_cv_branch_fraction": float(np.mean(np.asarray(branch_cvs) > 0.1)),
        "median_fisher_distortion": float(np.median(fisher_distortions)),
        "median_uniform_distortion": float(np.median(uniform_distortions)),
        "all_finite": True,
        "transitions": transition_rows,
    }


def evaluate_go_no_go(summaries, shadow_parity_verified=False):
    if len(summaries) < 3:
        raise ValueError("SBGC P1 requires three dataset summaries")
    active_dataset_count = sum(
        item["active_transition_fraction"] >= 0.30 for item in summaries
    )
    different_dataset_count = sum(
        item["different_transition_fraction"] >= 0.30 for item in summaries
    )
    pooled_high_cv = float(
        np.mean([item["high_cv_branch_fraction"] for item in summaries])
    )
    pooled_distortion = float(
        np.median([item["median_fisher_distortion"] for item in summaries])
    )
    checks = {
        "risk_active_in_at_least_two_datasets": active_dataset_count >= 2,
        "fisher_differs_in_at_least_two_datasets": different_dataset_count >= 2,
        "high_cv_branch_fraction_at_least_30pct": pooled_high_cv >= 0.30,
        "median_fisher_distortion_at_most_10pct": pooled_distortion <= 0.10,
        "shadow_parity_verified": bool(shadow_parity_verified),
        "all_statistics_finite": all(item["all_finite"] for item in summaries),
    }
    return {
        "decision": "GO" if all(checks.values()) else "NO_GO",
        "checks": checks,
        "active_dataset_count": active_dataset_count,
        "different_dataset_count": different_dataset_count,
        "pooled_high_cv_branch_fraction": pooled_high_cv,
        "median_of_dataset_median_fisher_distortion": pooled_distortion,
        "datasets": summaries,
    }


def _parse_artifact(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("artifact must use LABEL=PATH")
    label, path = value.split("=", 1)
    return label, Path(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", nargs="+", type=_parse_artifact)
    parser.add_argument("--expected-transitions", type=int, default=9)
    parser.add_argument("--shadow-parity-verified", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    summaries = [
        summarize_shadow_artifact(label, path, args.expected_transitions)
        for label, path in args.artifact
    ]
    result = evaluate_go_no_go(summaries, args.shadow_parity_verified)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
