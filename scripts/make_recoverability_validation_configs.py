#!/usr/bin/env python3
"""Generate ordered Recoverability Adaptive-A ablation configurations."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backbone.recoverability import operator_weighted_recoverability


STAGES = (
    "exact_risk",
    "accessibility",
    "anchor_realign",
    "global_budget",
)


def build_stage_configs(
    base_config,
    *,
    budget=0.01,
    step_size=0.1,
    interval=4,
    sketch_rank=16,
    gammas=(0.0, 0.25, 0.5, 0.75, 1.0),
):
    """Return four cumulative stage configurations without changing protocol."""
    if base_config.get("optimizer", "sgd").lower() != "sgd":
        raise ValueError("recoverability validation requires optimizer=sgd")
    if budget < 0 or step_size < 0:
        raise ValueError("budget and step_size must be non-negative")
    if interval <= 0 or sketch_rank <= 0:
        raise ValueError("interval and sketch_rank must be positive")
    base_prefix = str(base_config.get("prefix", "experiment"))
    base_filepath = str(base_config.get("filepath", "./RECOVERABILITY")).rstrip("/")
    configs = {}
    for stage in STAGES:
        config = copy.deepcopy(base_config)
        config.update(
            {
                "prefix": "{}_rga_{}".format(base_prefix, stage),
                "filepath": "{}_RGA_{}/".format(base_filepath, stage.upper()),
                "sa_adaptive_a_enabled": True,
                "sa_adaptive_a_strategy": "recoverability",
                "sa_recoverability_stage": stage,
                "sa_recoverability_budget": float(budget),
                "sa_recoverability_step_size": float(step_size),
                "sa_recoverability_interval": int(interval),
                "sa_recoverability_sketch_rank": int(sketch_rank),
                "sa_recoverability_gammas": [float(value) for value in gammas],
            }
        )
        configs[stage] = config
    return configs


def operator_weighting_toy():
    """Equal chordal movement with high- versus low-history-energy rotation."""
    theta = torch.tensor(math.pi / 6, dtype=torch.float64)
    c, s = torch.cos(theta), torch.sin(theta)
    anchor_a = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    high_candidate = torch.stack(
        [torch.tensor([c, 0.0, s, 0.0]), anchor_a[1]]
    )
    low_candidate = torch.stack(
        [anchor_a[0], torch.tensor([0.0, c, 0.0, s])]
    )
    anchor_up = torch.diag(torch.tensor([10.0, 1.0], dtype=torch.float64))
    anchor_projector = anchor_a.t() @ anchor_a

    def chordal(candidate):
        return float(
            torch.linalg.vector_norm(
                candidate.t() @ candidate - anchor_projector
            )
        )

    return {
        "chordal_high": chordal(high_candidate),
        "chordal_low": chordal(low_candidate),
        "recoverability_high": float(
            operator_weighted_recoverability(
                anchor_up, anchor_a, high_candidate
            )
        ),
        "recoverability_low": float(
            operator_weighted_recoverability(
                anchor_up, anchor_a, low_candidate
            )
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--budget", type=float, default=0.01)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--interval", type=int, default=4)
    parser.add_argument("--sketch-rank", type=int, default=16)
    parser.add_argument("--toy", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    base = json.loads(args.base.read_text(encoding="utf-8"))
    configs = build_stage_configs(
        base,
        budget=args.budget,
        step_size=args.step_size,
        interval=args.interval,
        sketch_rank=args.sketch_rank,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, (stage, config) in enumerate(configs.items(), start=1):
        path = args.output_dir / "{}_rga_s{}_{}.json".format(
            args.base.stem, index, stage
        )
        path.write_text(
            json.dumps(config, indent=4, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        print(path)
    if args.toy:
        print(json.dumps(operator_weighting_toy(), sort_keys=True))


if __name__ == "__main__":
    main()
