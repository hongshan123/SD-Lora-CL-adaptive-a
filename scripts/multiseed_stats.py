#!/usr/bin/env python3
"""Seed-keyed multi-seed statistics, paired tests, CIs, effect size, and TOST.

Groups are keyed by the seed parsed from each log (``=> seed: N`` or
``seed [N]``).  Pairing joins on seed values instead of zipping lexicographic
file lists, which silently mis-pairs runs whose seeds are ordered differently
(e.g. ``1, 2, 3, 1995`` vs ``1995, 1, 2, 3``).

Example:
  python scripts/multiseed_stats.py \
    --group main_inr 'sa_cumulative_inr_seed*_gauge.log' \
    --group exp009_inr 'sa_sdlora_proto_inr_seed*.log' \
    --metric final --paired main_inr exp009_inr \
    --margin 0.5
"""

import argparse
import glob
import os
import re
import statistics
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.parse_log_metrics import parse


SEED_RE = re.compile(r"=> seed: (-?\d+)")
SEED_BRACKET_RE = re.compile(r"\bseed \[(-?\d+)\]")


def parse_seed(path):
    """Extract the integer seed from a trainer log."""
    with open(path, errors="ignore") as handle:
        text = handle.read()
    match = SEED_RE.search(text) or SEED_BRACKET_RE.search(text)
    if match is None:
        raise ValueError(
            "cannot determine seed from {}; missing '=> seed:' line".format(path)
        )
    return int(match.group(1))


def metric_value(info, metric):
    if metric == "final":
        if not info["curves"]:
            raise ValueError("no CNN top1 curve in log")
        return float(info["curves"][-1][-1])
    if metric == "avg":
        if not info["avg"]:
            raise ValueError("no Average Accuracy in log")
        return float(info["avg"][-1])
    if metric == "forgetting":
        if not info["forgetting"]:
            raise ValueError("no Forgetting (CNN) in log")
        return float(info["forgetting"][-1])
    raise ValueError("unknown metric {}".format(metric))


def collect_paths(paths, metric):
    """Return {seed: (path, value)}.  Raises on duplicate or malformed logs."""
    out = {}
    for path in paths:
        seed = parse_seed(path)
        value = metric_value(parse(path), metric)
        if seed in out:
            raise ValueError(
                "duplicate seed {} for {} and {}".format(
                    seed, out[seed][0], path
                )
            )
        out[seed] = (path, value)
    return out


def collect_patterns(patterns, metric):
    paths = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if not matched:
            raise FileNotFoundError(
                "no logs matched {}".format(pattern)
            )
        paths.extend(matched)
    return collect_paths(paths, metric)


def pair_groups(group_a, group_b):
    """Inner join two {seed: ...} dicts; error if seed sets differ."""
    seeds_a = set(group_a)
    seeds_b = set(group_b)
    missing_a = sorted(seeds_a - seeds_b)
    missing_b = sorted(seeds_b - seeds_a)
    if missing_a or missing_b:
        raise ValueError(
            "seed sets differ: only in A={} only in B={}".format(
                missing_a, missing_b
            )
        )
    if not seeds_a:
        raise ValueError("cannot pair empty groups")
    return [
        (seed, group_a[seed], group_b[seed])
        for seed in sorted(seeds_a)
    ]


def paired_stats(diffs, alpha=0.05):
    """Mean diff, sd, t/p, 95% CI, and standardized effect size (Cohen's dz)."""
    n = len(diffs)
    if n < 2:
        raise ValueError("need at least 2 pairs")
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    sem = sd / (n ** 0.5)
    t_stat = mean / sem if sem > 0 else float("nan")
    p_two = float(stats.ttest_rel(
        np.asarray(diffs), np.zeros(n)
    ).pvalue)
    ci_low, ci_high = stats.t.interval(
        1.0 - alpha, n - 1, loc=mean, scale=sem
    )
    dz = mean / sd if sd > 0 else float("nan")
    return {
        "n": n,
        "mean_diff": mean,
        "sd_diff": sd,
        "t": t_stat,
        "p": p_two,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "cohens_dz": dz,
    }


def tost(diffs, margin, alpha=0.05):
    """Two one-sided tests for equivalence within +/- margin."""
    n = len(diffs)
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    sem = sd / (n ** 0.5)
    df = n - 1
    t_upper = (mean - margin) / sem
    p_upper = float(stats.t.cdf(t_upper, df))
    t_lower = (mean + margin) / sem
    p_lower = float(stats.t.sf(t_lower, df))
    equivalent = bool(p_upper <= alpha and p_lower <= alpha)
    return {
        "margin": margin,
        "t_upper": t_upper,
        "p_upper": p_upper,
        "t_lower": t_lower,
        "p_lower": p_lower,
        "equivalent": equivalent,
    }


def print_group(name, group, metric):
    values = [v for _, v in group.values()]
    print(
        "{:<16} n={} mean={:8.2f} std={:8.2f}".format(
            name,
            len(values),
            statistics.mean(values) if values else float("nan"),
            statistics.stdev(values) if len(values) > 1 else float("nan"),
        )
    )
    for seed in sorted(group):
        path, value = group[seed]
        print("    seed={:<6} {:<70} {:8.2f}".format(seed, path, value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        nargs=2,
        action="append",
        required=True,
        metavar=("NAME", "LOG_GLOB"),
        help="group name and log glob (repeatable)",
    )
    parser.add_argument(
        "--metric",
        default="final",
        choices=["final", "avg", "forgetting"],
    )
    parser.add_argument(
        "--paired",
        nargs=2,
        metavar=("A", "B"),
        help="paired test between two group names",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=None,
        help="equivalence margin for TOST (e.g. 0.5 for strict, 1.0 for loose)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="significance level (default 0.05)",
    )
    args = parser.parse_args()

    groups = {}
    for name, pattern in args.group:
        groups[name] = collect_patterns([pattern], args.metric)
        print_group(name, groups[name], args.metric)

    if args.paired:
        a_name, b_name = args.paired
        if a_name not in groups or b_name not in groups:
            parser.error("--paired names must be defined groups")
        pairs = pair_groups(groups[a_name], groups[b_name])
        diffs = [av - bv for _, (_, av), (_, bv) in pairs]
        print()
        print("Paired {} - {} (seed-joined, n={}):".format(
            a_name, b_name, len(pairs)
        ))
        for seed, (a_path, a_val), (b_path, b_val) in pairs:
            print(
                "    seed={:<6} A={:8.2f} B={:8.2f} diff={:+8.2f}".format(
                    seed, a_val, b_val, a_val - b_val
                )
            )
        result = paired_stats(diffs, alpha=args.alpha)
        print(
            "    mean_diff={:+.3f} sd_diff={:.3f} t={:+.3f} "
            "p={:.4f} 95%CI=[{:.3f}, {:.3f}] cohen_dz={:+.3f}".format(
                result["mean_diff"],
                result["sd_diff"],
                result["t"],
                result["p"],
                result["ci_low"],
                result["ci_high"],
                result["cohens_dz"],
            )
        )
        if args.margin is not None:
            eq = tost(diffs, args.margin, alpha=args.alpha)
            print(
                "    TOST margin={} equivalent={} p_upper={:.4f} "
                "p_lower={:.4f}".format(
                    eq["margin"],
                    eq["equivalent"],
                    eq["p_upper"],
                    eq["p_lower"],
                )
            )


if __name__ == "__main__":
    main()
