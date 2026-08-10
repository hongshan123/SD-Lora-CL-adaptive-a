"""Unit tests for seed-keyed multi-seed statistics."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.multiseed_stats import (
    collect_patterns,
    collect_paths,
    pair_groups,
    parse_seed,
    tost,
)


def _write_log(tmp_path, name, seed, curves, avg, forgetting):
    path = tmp_path / name
    lines = [
        "seed [{}]".format(seed),
        "=> config: dummy.json",
        "=> seed: {}".format(seed),
        "CNN top1 curve: [{}]".format(", ".join(map(str, curves))),
        "Average Accuracy (CNN): {}".format(avg),
        "Forgetting (CNN): {}".format(forgetting),
    ]
    path.write_text("\n".join(lines))
    return str(path)


def test_parse_seed_prefers_logged_seed(tmp_path):
    path = _write_log(tmp_path, "a.log", 1995, [1.0], 2.0, 3.0)
    assert parse_seed(path) == 1995


def test_parse_seed_missing_raises(tmp_path):
    path = tmp_path / "bad.log"
    path.write_text("no seed here")
    with pytest.raises(ValueError):
        parse_seed(str(path))


def test_lexicographic_trap_is_inner_joined(tmp_path):
    """Files with seeds ordered 1995,1,2,3 vs 1,1995,2,3 must pair by seed."""
    main = [
        _write_log(tmp_path, "main_1995.log", 1995, [80.0], 1.0, 1.0),
        _write_log(tmp_path, "main_1.log", 1, [70.0], 1.0, 1.0),
        _write_log(tmp_path, "main_2.log", 2, [75.0], 1.0, 1.0),
        _write_log(tmp_path, "main_3.log", 3, [78.0], 1.0, 1.0),
    ]
    base = [
        _write_log(tmp_path, "base_1.log", 1, [71.0], 1.0, 1.0),
        _write_log(tmp_path, "base_1995.log", 1995, [79.0], 1.0, 1.0),
        _write_log(tmp_path, "base_2.log", 2, [76.0], 1.0, 1.0),
        _write_log(tmp_path, "base_3.log", 3, [77.0], 1.0, 1.0),
    ]
    group_a = collect_paths(main, "final")
    group_b = collect_paths(base, "final")
    pairs = pair_groups(group_a, group_b)
    assert [(seed, a, b) for seed, (_, a), (_, b) in pairs] == [
        (1, 70.0, 71.0),
        (2, 75.0, 76.0),
        (3, 78.0, 77.0),
        (1995, 80.0, 79.0),
    ]


def test_pair_groups_errors_on_missing_seed(tmp_path):
    a = _write_log(tmp_path, "a.log", 1, [1.0], 1.0, 1.0)
    b = _write_log(tmp_path, "b.log", 2, [1.0], 1.0, 1.0)
    with pytest.raises(ValueError):
        pair_groups(collect_paths([a], "final"), collect_paths([b], "final"))


def test_duplicate_seed_raises(tmp_path):
    a = _write_log(tmp_path, "a1.log", 1, [1.0], 1.0, 1.0)
    b = _write_log(tmp_path, "a2.log", 1, [2.0], 2.0, 2.0)
    with pytest.raises(ValueError):
        collect_paths([a, b], "final")


def test_confirmation_seed_filter_keeps_allowed_and_reports_excluded(tmp_path):
    paths = [
        _write_log(tmp_path, "a1.log", 1, [70.0], 1.0, 1.0),
        _write_log(tmp_path, "a2.log", 2, [75.0], 1.0, 1.0),
        _write_log(tmp_path, "a_dev.log", 1995, [80.0], 1.0, 1.0),
    ]
    collected, excluded = collect_patterns(
        [str(tmp_path / "a*.log")], "final", allowed_seeds=[1, 2]
    )
    assert set(collected) == {1, 2}
    assert excluded == [(1995, str(tmp_path / "a_dev.log"))]


def test_confirmation_seed_filter_errors_on_missing_seed(tmp_path):
    paths = [
        _write_log(tmp_path, "a1.log", 1, [70.0], 1.0, 1.0),
        _write_log(tmp_path, "a2.log", 2, [75.0], 1.0, 1.0),
    ]
    with pytest.raises(ValueError):
        collect_paths(paths, "final", allowed_seeds=[1, 2, 3])


def test_tost_inside_margin_is_equivalent():
    # mean diff = 0.0, sd = 1.0, n = 8 -> inside +/-0.5
    diffs = [0.5, -0.5, 0.3, -0.2, 0.1, -0.1, 0.4, -0.4]
    result = tost(diffs, margin=0.5)
    assert result["equivalent"] is True


def test_tost_outside_margin_is_not_equivalent():
    diffs = [1.0, 1.2, 0.9, 1.1, 1.3, 0.8, 1.2, 1.0]
    result = tost(diffs, margin=0.5)
    assert result["equivalent"] is False
