import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_train_projected_smokes_match_three_dataset_protocol():
    expected = {
        "c100": ("cifar224", 1993, 10),
        "inr": ("imagenetr", 1995, 20),
        "cub": ("cub", 1, 20),
    }
    output_paths = set()
    for name, (dataset, seed, classes_per_task) in expected.items():
        path = ROOT / "exps" / f"sbgc_trainproj_{name}_smoke_20260925.json"
        config = json.loads(path.read_text())
        assert config["dataset"] == dataset
        assert config["seed"] == [seed]
        assert config["init_cls"] == classes_per_task
        assert config["increment"] == classes_per_task
        assert config["max_tasks"] == 2
        assert config["batch_size"] == 64
        assert config["init_epoch"] == config["epochs"] == 2
        assert config["lora_rank"] == 10
        assert config["sa_g_train_projected"] is True
        assert config["sa_g_budget_scope"] == "branch"
        assert config["sa_g_risk_budget"] == 0.05
        assert config["sa_g_sensitivity_metric"] == "fisher_diag"
        assert config["sa_coordinate_stable_transport"] is True
        assert config["sa_dual_head"] is False
        assert config["sa_hbd_enabled"] is False
        output_paths.add(config["filepath"])
    assert len(output_paths) == len(expected)
