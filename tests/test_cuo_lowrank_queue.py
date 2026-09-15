import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.sa_sdlora import validate_cuo_lowrank_config


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "c100": ROOT / "exps/cuo_lowrank_r10_c100_seed1993.json",
    "inr": ROOT / "exps/cuo_lowrank_r10_inr_seed1995.json",
    "cub": ROOT / "exps/cuo_lowrank_r10_cub_seed1.json",
}


def _base_cuo_config():
    return {
        "sa_cumulative_merge": "cuo_lowrank",
        "sa_cumulative_state": True,
        "sa_train_a_all_tasks": True,
        "lora_rank": 10,
        "sa_cumulative_rank": 10,
        "sa_cuo_lambda": 1e-5,
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("sa_adaptive_a_enabled", True),
        ("sa_coordinate_stable_transport", True),
        ("sa_hbd_enabled", True),
    ],
)
def test_cuo_rejects_live_coordinate_features(key, value):
    config = _base_cuo_config()
    config[key] = value
    with pytest.raises(ValueError, match="cuo_lowrank"):
        validate_cuo_lowrank_config(config)


def test_cuo_rank_must_match_persistent_rank():
    config = _base_cuo_config()
    config["sa_cumulative_rank"] = 8
    with pytest.raises(ValueError, match="rank"):
        validate_cuo_lowrank_config(config)


@pytest.mark.parametrize("config_path", CONFIGS.values())
def test_cuo_configs_match_the_fixed_budget_protocol(config_path):
    with config_path.open() as handle:
        config = json.load(handle)
    assert config["model_name"] == "sa_sdlora"
    assert config["lora_rank"] == config["sa_cumulative_rank"] == 10
    assert config["sa_cumulative_state"] is True
    assert config["sa_cumulative_merge"] == "cuo_lowrank"
    assert config["sa_cuo_lambda"] == 1e-5
    assert config["batch_size"] == 64
    assert "sa_cuo_calibration_batch_size" not in config
    assert config["sa_train_a_all_tasks"] is True
    assert config["sa_use_prototype_classifier"] is True
    assert config.get("sa_dual_head", False) is False
    assert config.get("sa_hbd_enabled", False) is False
    assert config.get("sa_coordinate_stable_transport", False) is False
    assert config.get("sa_adaptive_a_enabled", False) is False
