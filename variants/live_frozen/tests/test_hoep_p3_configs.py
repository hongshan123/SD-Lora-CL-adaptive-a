"""Protocol tests for the preregistered HOEP-A P3 matrix."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_hoep_p3_configs import generate_configs


def test_hoep_p3_matrix_is_strictly_matched(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = generate_configs(root, tmp_path, "test")

    assert len(manifest) == 12
    assert {(job["dataset"], job["method"]) for job in manifest} == {
        (dataset, method)
        for dataset in ("c100", "inr", "cub")
        for method in ("frozen", "live", "ratio", "hoep")
    }
    configs = [json.loads(Path(job["config"]).read_text()) for job in manifest]
    common_keys = (
        "init_epoch",
        "epochs",
        "batch_size",
        "lora_rank",
        "sa_use_prototype_classifier",
        "sa_cumulative_merge",
        "sa_live_a_coordinate_align",
        "sa_live_a_absorb_mode",
        "sa_coordinate_stable_transport",
        "sa_dual_head",
        "sa_hbd_enabled",
    )
    for dataset in ("c100", "inr", "cub"):
        paired = [
            config
            for job, config in zip(manifest, configs)
            if job["dataset"] == dataset
        ]
        for key in common_keys:
            assert len({json.dumps(config[key], sort_keys=True) for config in paired}) == 1
    assert all(config["batch_size"] == 64 for config in configs)
    assert all(config["device"] == ["0", "1"] for config in configs)
    assert all(config["dist_backend"] == "nccl" for config in configs)
    assert all(config["init_epoch"] == config["epochs"] == 20 for config in configs)
    assert all(config["max_tasks"] == 10 for config in configs)


def test_hoep_p3_only_changes_the_shared_a_policy(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = generate_configs(root, tmp_path, "test")
    by_method = {
        job["method"]: json.loads(Path(job["config"]).read_text())
        for job in manifest
        if job["dataset"] == "c100"
    }

    policy_keys = {
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
        "sa_hoep_energy_budget",
        "sa_hoep_eigenvalue_rtol",
    }
    normalized = {
        method: {
            key: value for key, value in config.items() if key not in policy_keys
        }
        for method, config in by_method.items()
    }
    assert normalized["frozen"] == normalized["live"]
    assert normalized["frozen"] == normalized["ratio"]
    assert normalized["frozen"] == normalized["hoep"]

    assert by_method["frozen"]["sa_train_a_all_tasks"] is False
    assert by_method["frozen"]["sa_adaptive_a_enabled"] is False
    assert by_method["live"]["sa_train_a_all_tasks"] is True
    assert by_method["live"]["sa_adaptive_a_enabled"] is False
    assert by_method["ratio"]["sa_adaptive_a_strategy"] == "impact_ratio"
    assert by_method["ratio"]["sa_adaptive_a_gate_floor"] == 0.05
    assert by_method["ratio"]["sa_adaptive_a_gate_momentum"] == 0.9
    assert by_method["hoep"]["sa_adaptive_a_strategy"] == "operator_energy_partition"
    assert by_method["hoep"]["sa_hoep_energy_budget"] == 0.05
