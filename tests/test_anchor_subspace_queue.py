"""Configuration checks for the first-task-anchor screening queue."""

import json
from pathlib import Path

from scripts.generate_anchor_subspace_configs import DATASETS, METHODS, generate_configs


ROOT = Path(__file__).resolve().parents[1]


def test_anchor_subspace_screening_matrix_is_matched(tmp_path):
    manifest = generate_configs(ROOT, tmp_path, "unit", batch_size=128)
    assert len(manifest) == len(DATASETS) * len(METHODS) == 12
    assert {(item["dataset"], item["method"]) for item in manifest} == {
        (dataset, method) for dataset in DATASETS for method in METHODS
    }

    configs = {}
    for item in manifest:
        config = json.loads(Path(item["config"]).read_text())
        configs[(item["dataset"], item["method"])] = config
        assert config["batch_size"] == 128
        assert config["device"] == ["0"]
        assert config["lora_rank"] == 10
        assert config["sa_train_a_all_tasks"] is False
        assert config["sa_coordinate_stable_transport"] is False
        assert config["sa_dual_head"] is False
        assert config["sa_adaptive_a_enabled"] is False

    for dataset in DATASETS:
        sa_lora = configs[(dataset, "sa_lora_bank")]
        frozen_bank = configs[(dataset, "frozen_b_bank")]
        exact_g = configs[(dataset, "exact_g")]
        bounded_g = configs[(dataset, "bounded_g")]
        assert sa_lora["sa_cumulative_state"] is False
        assert sa_lora["sa_cumulative_merge"] == "gauge"
        assert sa_lora["sa_freeze_old_scales"] is False
        assert frozen_bank["sa_freeze_old_scales"] is True
        assert frozen_bank["sa_cumulative_merge"] == "gauge"
        assert frozen_bank["sa_normalize_current_branch"] is True
        assert exact_g["sa_cumulative_state"] is True
        assert exact_g["sa_cumulative_merge"] == "live_a_aggregate_b"
        assert exact_g["sa_normalize_current_branch"] is True
        assert exact_g["sa_live_a_absorb_mode"] == "normalized_absorb"
        assert bounded_g["sa_cumulative_state"] is True
        assert bounded_g["sa_cumulative_merge"] == "live_a_aggregate_b"
        assert bounded_g["sa_normalize_current_branch"] is False
        assert bounded_g["sa_live_a_absorb_mode"] == "bounded_norm_calibrated_absorb"
