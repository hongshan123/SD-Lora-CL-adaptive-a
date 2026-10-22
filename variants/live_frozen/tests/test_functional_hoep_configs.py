"""Protocol tests for Functional-HOEP smoke and Phase-A configs."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_functional_hoep_configs import generate_configs


def test_functional_hoep_shadow_configs_are_preregistered(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for phase, tasks, epochs in (("smoke", 2, 2), ("phase_a", 10, 20)):
        runtime = tmp_path / phase
        manifest = generate_configs(root, runtime, "test", phase)
        assert {item["dataset"] for item in manifest} == {"c100", "inr", "cub"}
        assert len(manifest) == 3
        for item in manifest:
            config = json.loads(Path(item["config"]).read_text())
            assert config["max_tasks"] == tasks
            assert config["init_epoch"] == config["epochs"] == epochs
            assert config["batch_size"] == 64
            assert config["lora_rank"] == 10
            assert config["sa_hoep_energy_metric"] == "operator"
            assert config["sa_hoep_functional_diagnostics"] is True
            assert config["sa_hoep_activation_calibration_batch_size"] == 64
            assert config["sa_coordinate_stable_transport"] is False
            assert config["sa_dual_head"] is False
            assert config["sa_hbd_enabled"] is False
            assert config["sa_live_a_absorb_mode"] == "operator_preserving_absorb"
