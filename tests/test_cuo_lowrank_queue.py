import json
import sys
import importlib.util
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


def _load_measure_module():
    path = ROOT / "scripts" / "measure_sa_artifact.py"
    spec = importlib.util.spec_from_file_location("measure_sa_artifact", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def test_measure_artifact_separates_cuo_factors_from_gram_stats(tmp_path):
    branch_count, dim, rank = 24, 768, 10
    torch = pytest.importorskip("torch")
    torch.save(
        {
            "version": 5,
            "task_id": 2,
            "rank": rank,
            "merge_mode": "cuo_lowrank",
            "projection_down": [torch.zeros(rank, dim) for _ in range(branch_count)],
            "unified_up": [torch.zeros(dim, rank) for _ in range(branch_count)],
            "projected_gram": [torch.zeros(rank, rank) for _ in range(branch_count)],
            "cuo_lambda": 1e-5,
        },
        tmp_path / "sa_state.pt",
    )
    summary = _load_measure_module().measure_artifact(tmp_path)
    assert summary["lora_factor_scalars"] == 368640
    assert summary["cuo_gram_scalars"] == 2400
    assert summary["persistent_scalar_total"] == 371040


def test_cuo_queue_uses_two_gpu_torchrun_and_scheduler_visible_devices():
    script = (ROOT / "run_cuo_lowrank_r10_3datasets_2gpu.sh").read_text()
    for config_path in CONFIGS.values():
        assert config_path.name in script
    assert "CUDA_VISIBLE_DEVICES" in script
    assert "--nproc_per_node=2" in script
