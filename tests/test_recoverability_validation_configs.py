import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.make_recoverability_validation_configs import (
    STAGES,
    build_stage_configs,
    operator_weighting_toy,
)


def _base_config():
    return {
        "prefix": "baseline",
        "filepath": "./BASELINE/",
        "dataset": "cifar224",
        "seed": [1993],
        "optimizer": "sgd",
        "sa_adaptive_a_enabled": True,
        "sa_adaptive_a_strategy": "impact_ratio",
        "sa_train_a_all_tasks": True,
        "sa_cumulative_state": True,
        "sa_cumulative_merge": "live_a_aggregate_b",
        "sa_live_a_coordinate_align": True,
        "sa_live_a_absorb_mode": "bounded_norm_calibrated_absorb",
        "sa_use_prototype_classifier": True,
        "sa_coordinate_stable_transport": True,
        "sa_dual_head": False,
    }


def test_stage_configs_are_cumulative_and_preserve_training_protocol():
    configs = build_stage_configs(
        _base_config(), budget=0.02, step_size=0.15, interval=8, sketch_rank=12
    )

    assert list(configs) == list(STAGES)
    for stage, config in configs.items():
        assert config["sa_adaptive_a_strategy"] == "recoverability"
        assert config["sa_recoverability_stage"] == stage
        assert config["sa_recoverability_budget"] == pytest.approx(0.02)
        assert config["sa_recoverability_step_size"] == pytest.approx(0.15)
        assert config["sa_recoverability_interval"] == 8
        assert config["sa_recoverability_sketch_rank"] == 12
        assert config["optimizer"] == "sgd"
        assert config["sa_live_a_absorb_mode"] == "bounded_norm_calibrated_absorb"
        assert config["sa_coordinate_stable_transport"] is True
        assert config["sa_dual_head"] is False
        assert stage in config["prefix"]
        assert stage.upper() in config["filepath"]


def test_operator_weighting_toy_has_equal_geometry_and_unequal_risk():
    result = operator_weighting_toy()

    assert result["chordal_high"] == pytest.approx(result["chordal_low"], rel=1e-10)
    assert result["recoverability_high"] > 50.0 * result["recoverability_low"]


def test_cli_writes_four_json_files_without_starting_training(tmp_path):
    base = tmp_path / "base.json"
    output = tmp_path / "configs"
    base.write_text(json.dumps(_base_config()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_recoverability_validation_configs.py"),
            "--base",
            str(base),
            "--output-dir",
            str(output),
            "--toy",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    written = sorted(output.glob("*.json"))
    assert len(written) == 4
    assert "recoverability_high" in completed.stdout
    assert "torchrun" not in completed.stdout
