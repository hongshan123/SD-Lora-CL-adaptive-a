"""Protocol tests for the final no-Dual-B A-policy task-length queue."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_final_a_policy_tasklen_configs import (
    METHODS,
    TASK_COUNTS,
    generate_configs,
)
from scripts.wait_for_queue_success import terminal_queue_status


def test_generator_builds_complete_final_protocol_matrix(tmp_path):
    manifest = generate_configs(
        ROOT, tmp_path / "runtime", "unit", batch_size=64, world_size=2
    )

    assert len(manifest) == 3 * 4 * 3
    assert {item["dataset"] for item in manifest} == {"c100", "inr", "cub"}
    assert {item["tasks"] for item in manifest} == set(TASK_COUNTS)
    assert {item["method"] for item in manifest} == set(METHODS)
    assert len({item["name"] for item in manifest}) == len(manifest)

    for item in manifest:
        config = json.loads(Path(item["config"]).read_text())
        assert config["batch_size"] == 64
        assert config["epochs"] == 20
        assert config["init_epoch"] == 20
        assert config["optimizer"] == "sgd"
        assert config["lora_rank"] == 10
        assert config["sa_use_prototype_classifier"] is True
        assert config["sa_dual_head"] is False
        assert "sa_dual_head_schedule" not in config
        assert config["sa_coordinate_stable_transport"] is True
        assert config["sa_coordinate_transport_rank"] == 10
        assert config["sa_coordinate_transport_reg"] == pytest.approx(1e-4)
        assert config["sa_coordinate_transport_min_gain"] == pytest.approx(0.0)
        assert config["sa_live_a_coordinate_align"] is True
        assert (
            config["sa_live_a_absorb_mode"]
            == "bounded_norm_calibrated_absorb"
        )
        assert config["sa_hbd_enabled"] is False
        assert len(config["task_increments"]) == item["tasks"]


def test_generator_changes_only_the_declared_a_policy(tmp_path):
    manifest = generate_configs(ROOT, tmp_path, "unit")
    configs = {}
    for item in manifest:
        if item["dataset"] == "inr" and item["tasks"] == 10:
            configs[item["method"]] = json.loads(Path(item["config"]).read_text())

    frozen = configs["frozen_a"]
    live = configs["live_a"]
    adaptive = configs["adaptive_a"]

    assert frozen["sa_train_a_all_tasks"] is False
    assert frozen["sa_adaptive_a_enabled"] is False
    assert live["sa_train_a_all_tasks"] is True
    assert live["sa_adaptive_a_enabled"] is False
    assert adaptive["sa_train_a_all_tasks"] is True
    assert adaptive["sa_adaptive_a_enabled"] is True
    assert adaptive["sa_adaptive_a_strategy"] == "impact_ratio"
    assert adaptive["sa_adaptive_a_gate_formula"] == "ratio"
    assert adaptive["sa_adaptive_a_gate_floor"] == pytest.approx(0.0)
    assert adaptive["sa_adaptive_a_gate_momentum"] == pytest.approx(0.0)

    ignored = {
        "prefix",
        "filepath",
        "sa_train_a_all_tasks",
        "sa_adaptive_a_enabled",
        "sa_adaptive_a_strategy",
        "sa_adaptive_a_stability_weight",
        "sa_adaptive_a_gate_formula",
        "sa_adaptive_a_gate_floor",
        "sa_adaptive_a_gate_momentum",
        "sa_adaptive_a_eps",
    }
    stripped = {
        method: {key: value for key, value in config.items() if key not in ignored}
        for method, config in configs.items()
    }
    assert stripped["frozen_a"] == stripped["live_a"] == stripped["adaptive_a"]


def test_predecessor_status_requires_an_explicit_success_marker(tmp_path):
    queue_log = tmp_path / "predecessor.log"
    assert terminal_queue_status(queue_log) is None

    queue_log.write_text("QUEUE START jobs=24\n")
    assert terminal_queue_status(queue_log) is None

    queue_log.write_text("QUEUE START jobs=24\nQUEUE END status=1 jobs=24\n")
    assert terminal_queue_status(queue_log) == 1

    queue_log.write_text("QUEUE START jobs=24\nQUEUE END status=0 jobs=24\n")
    assert terminal_queue_status(queue_log) == 0


def test_launcher_has_valid_syntax_and_prepare_only_mode(tmp_path):
    launcher = ROOT / "run_final_a_policy_tasklen_3datasets_2gpu_queue.sh"
    subprocess.run(["bash", "-n", str(launcher)], check=True)

    predecessor = tmp_path / "predecessor.log"
    predecessor.write_text("QUEUE END status=0 jobs=24\n")
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
            "PREDECESSOR_QUEUE_LOG": str(predecessor),
            "PREPARE_ONLY": "1",
        },
        capture_output=True,
        text=True,
        check=True,
    )

    assert "PREPARE_ONLY complete" in result.stdout
    manifest = json.loads((runtime / "manifest.json").read_text())
    assert len(manifest) == 36
