import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.sa_sdlora import validate_sbgc_config


ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "c100": (1993, 100, ["0", "1"], "cifar224"),
    "inr": (1995, 200, ["4", "5"], "imagenetr"),
    "cub": (1, 200, ["6", "7"], "cub"),
}


def _config(dataset):
    seed = SPECS[dataset][0]
    path = ROOT / "exps" / f"sbgc_offline_t1_{dataset}_seed{seed}_r10.json"
    return path, json.loads(path.read_text())


@pytest.mark.parametrize("dataset", SPECS)
def test_offline_t1_is_same_budget_joint_training(dataset):
    _, config = _config(dataset)
    seed, classes, devices, dataset_name = SPECS[dataset]
    assert config["dataset"] == dataset_name
    assert config["seed"] == [seed]
    assert config["device"] == devices
    assert config["init_cls"] == config["increment"] == classes
    assert config["max_tasks"] == 1
    assert config["batch_size"] == 64
    assert config["init_epoch"] == config["epochs"] == 20
    assert config["lora_rank"] == config["sa_cumulative_rank"] == 10
    assert config["sa_cumulative_merge"] == "sensitivity_budgeted_g"
    assert config["sa_g_risk_budget"] == 0.05
    assert config["sa_g_sensitivity_metric"] == "fisher_diag"
    assert config["sa_g_shadow_only"] is False
    assert config["sa_train_a_all_tasks"] is False
    assert config["sa_dual_head"] is False
    assert config["sa_hbd_enabled"] is False
    assert config["sa_adaptive_a_enabled"] is False
    assert config["sa_coordinate_stable_transport"] is False
    validate_sbgc_config(config)


def test_offline_t1_queue_uses_three_required_pairs_and_nohup():
    script = (ROOT / "run_sbgc_offline_t1_3datasets_2gpu.sh").read_text()
    for pair in ("0,1", "4,5", "6,7"):
        assert f'run_job "{pair}"' in script
    assert "--nproc_per_node=2" in script
    assert "nohup env" in script
    assert 'CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG}"' in script
    assert "sleep 1800" in script
    for dataset in SPECS:
        path, _ = _config(dataset)
        assert path.name in script
