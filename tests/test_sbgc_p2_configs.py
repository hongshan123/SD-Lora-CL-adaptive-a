import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.sa_sdlora import validate_cuo_lowrank_config, validate_sbgc_config


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "c100": {"dataset": "cifar224", "seed": 1993, "device": ["0", "1"]},
    "inr": {"dataset": "imagenetr", "seed": 1995, "device": ["4", "5"]},
    "cub": {"dataset": "cub", "seed": 1, "device": ["6", "7"]},
}


def _path(method, dataset):
    seed = DATASETS[dataset]["seed"]
    if method == "cuo":
        name = f"sbgc_p2_cuo_r10_{dataset}_seed{seed}_t10.json"
    else:
        name = f"sbgc_p2_{method}_{dataset}_seed{seed}_t10.json"
    return ROOT / "exps" / name


def _load(method, dataset):
    return json.loads(_path(method, dataset).read_text())


@pytest.mark.parametrize("dataset", DATASETS)
def test_fisher_and_uniform_configs_only_differ_in_registered_identity(dataset):
    fisher = _load("fisher", dataset)
    uniform = _load("uniform", dataset)
    assert fisher["sa_g_sensitivity_metric"] == "fisher_diag"
    assert uniform["sa_g_sensitivity_metric"] == "uniform"
    ignored = {"prefix", "filepath", "sa_g_sensitivity_metric"}
    assert {k: v for k, v in fisher.items() if k not in ignored} == {
        k: v for k, v in uniform.items() if k not in ignored
    }


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("method", ["fisher", "uniform"])
def test_sbgc_p2_configs_match_strict_protocol(dataset, method):
    config = _load(method, dataset)
    expected = DATASETS[dataset]
    assert config["dataset"] == expected["dataset"]
    assert config["seed"] == [expected["seed"]]
    assert config["device"] == expected["device"]
    assert config["max_tasks"] == 10
    assert config["batch_size"] == 64
    assert config["init_epoch"] == config["epochs"] == 20
    assert config["lora_rank"] == config["sa_cumulative_rank"] == 10
    assert config["sa_g_risk_budget"] == 0.05
    assert config["sa_g_shadow_only"] is False
    assert config["sa_train_a_all_tasks"] is False
    assert config["sa_dual_head"] is False
    assert config["sa_hbd_enabled"] is False
    assert config["sa_adaptive_a_enabled"] is False
    assert config["sa_coordinate_stable_transport"] is False
    assert config["lrpt_enabled"] is False
    validate_sbgc_config(config)


@pytest.mark.parametrize("dataset", DATASETS)
def test_cuo_p2_configs_match_dataset_training_protocol(dataset):
    config = _load("cuo", dataset)
    reference = _load("fisher", dataset)
    expected = DATASETS[dataset]
    assert config["dataset"] == expected["dataset"]
    assert config["seed"] == [expected["seed"]]
    assert config["device"] == expected["device"]
    assert config["max_tasks"] == 10
    for key in (
        "optimizer",
        "scheduler",
        "init_epoch",
        "init_lr",
        "epochs",
        "lrate",
        "batch_size",
        "weight_decay",
        "lora_rank",
        "sa_use_prototype_classifier",
    ):
        assert config[key] == reference[key]
    assert config["sa_cuo_lambda"] == 1e-5
    validate_cuo_lowrank_config(config)


def test_queue_uses_required_pairs_order_and_nohup():
    script = (ROOT / "run_sbgc_p2_strict_3datasets_2gpu.sh").read_text()
    assert 'run_job "0,1"' in script
    assert 'run_job "4,5"' in script
    assert 'run_job "6,7"' in script
    assert 'CUDA_VISIBLE_DEVICES="${pair}"' in script
    assert "--nproc_per_node=2" in script
    assert "nohup env" in script
    assert "sleep 1800" in script
    for dataset in DATASETS:
        fisher = _path("fisher", dataset).name
        uniform = _path("uniform", dataset).name
        cuo = _path("cuo", dataset).name
        assert script.index(fisher) < script.index(uniform) < script.index(cuo)
