"""Protocol tests for momentum-aware Adaptive-A experiments."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_momentum_adaptive_a_configs import generate_configs
from scripts.generate_momentum_adaptive_a_tasklen_configs import (
    TASK_COUNTS,
    generate_configs as generate_tasklen_configs,
)


@pytest.mark.parametrize(
    "dataset_key, dataset, seed, increments, gpu_ids",
    [
        ("c100", "cifar224", 1993, [10] * 10, "0,1"),
        ("inr", "imagenetr", 1995, [20] * 10, "4,5"),
        ("cub", "cub", 1, [20] * 10, "6,7"),
    ],
)
def test_generator_builds_matched_theoretical_configs(
    tmp_path, dataset_key, dataset, seed, increments, gpu_ids
):
    before = {
        path: path.read_bytes()
        for path in (ROOT / "exps").glob("p*_livea_dual_b_seed1_nccl.json")
    }

    manifest = generate_configs(ROOT, tmp_path, "unit")
    entry = next(item for item in manifest if item["dataset"] == dataset_key)
    config = json.loads(Path(entry["config"]).read_text())

    assert entry["gpu_ids"] == gpu_ids
    assert config["dataset"] == dataset
    assert config["seed"] == [seed]
    assert config["task_increments"] == increments
    assert config["init_cls"] == increments[0]
    assert config["increment"] == increments[0]
    assert config["batch_size"] == 64
    assert config["init_epoch"] == 20
    assert config["epochs"] == 20
    assert config["optimizer"] == "sgd"
    assert config["lora_rank"] == 10
    assert config["sa_use_prototype_classifier"] is True
    assert config["sa_dual_head"] is False
    assert config["sa_hbd_enabled"] is False
    assert config["sa_coordinate_stable_transport"] is False
    assert config["sa_cumulative_merge"] == "live_a_aggregate_b"
    assert config["sa_live_a_coordinate_align"] is True
    assert config["sa_live_a_absorb_mode"] == "operator_preserving_absorb"
    assert config["sa_adaptive_a_enabled"] is True
    assert config["sa_adaptive_a_strategy"] == "impact_ratio"
    assert config["sa_adaptive_a_stability_weight"] == pytest.approx(1.0)
    assert config["sa_adaptive_a_gate_formula"] == "ratio"
    assert config["sa_adaptive_a_gate_floor"] == pytest.approx(0.0)
    assert config["sa_adaptive_a_gate_momentum"] == pytest.approx(0.0)
    assert config["sa_adaptive_a_eps"] == pytest.approx(1e-8)
    assert config["sa_resume"] is False
    assert "sa_dual_head_schedule" not in config

    after = {path: path.read_bytes() for path in before}
    assert after == before


def test_generator_creates_exactly_three_unique_runs(tmp_path):
    manifest = generate_configs(ROOT, tmp_path, "unit")

    assert len(manifest) == 3
    assert len({entry["name"] for entry in manifest}) == 3
    assert len({entry["config"] for entry in manifest}) == 3
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest


def test_launcher_has_valid_shell_syntax_and_prepare_only_mode(tmp_path):
    launcher = ROOT / "run_momentum_adaptive_a_t10_3datasets_2gpu.sh"
    subprocess.run(["bash", "-n", str(launcher)], check=True)

    runtime = tmp_path / "runtime"
    result = subprocess.run(
        ["bash", str(launcher)],
        cwd=ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "PROJECT_ROOT": str(ROOT),
            "PYTHON_BIN": sys.executable,
            "RUNTIME_DIR": str(runtime),
            "RUN_TAG": "unit",
            "PREPARE_ONLY": "1",
        },
        capture_output=True,
        text=True,
        check=True,
    )

    assert "PREPARE_ONLY complete" in result.stdout
    manifest = json.loads((runtime / "manifest.json").read_text())
    assert [entry["gpu_ids"] for entry in manifest] == ["0,1", "4,5", "6,7"]


def test_tasklen_generator_builds_three_dataset_four_length_manifest(tmp_path):
    manifest = generate_tasklen_configs(ROOT, tmp_path, "tasklen")

    assert len(manifest) == 12
    assert [entry["tasks"] for entry in manifest[:4]] == list(TASK_COUNTS)
    assert [entry["dataset"] for entry in manifest[:4]] == ["c100"] * 4
    assert [entry["dataset"] for entry in manifest[4:8]] == ["inr"] * 4
    assert [entry["dataset"] for entry in manifest[8:]] == ["cub"] * 4
    assert {entry["gpu_ids"] for entry in manifest[:4]} == {"0,1"}
    assert {entry["gpu_ids"] for entry in manifest[4:8]} == {"4,5"}
    assert {entry["gpu_ids"] for entry in manifest[8:]} == {"6,7"}

    for entry in manifest:
        config = json.loads(Path(entry["config"]).read_text())
        assert config["max_tasks"] == entry["tasks"]
        assert len(config["task_increments"]) == entry["tasks"]
        assert sum(config["task_increments"]) in (100, 200)
        assert config["batch_size"] == 64
        assert config["sa_adaptive_a_gate_floor"] == pytest.approx(0.0)
        assert config["sa_adaptive_a_gate_momentum"] == pytest.approx(0.0)
