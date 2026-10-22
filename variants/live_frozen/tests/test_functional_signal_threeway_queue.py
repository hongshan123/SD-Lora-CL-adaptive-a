"""Executable checks for the functional-signal three-way queue."""

import json
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_functional_signal_threeway_single_seed_2gpu.sh"
SOURCES = (
    ("c100_coordinate_stable_adaptive_a_seed1993_nccl.json", 1993),
    ("inr_coordinate_stable_adaptive_a_seed1995_nccl.json", 1995),
    ("cub_coordinate_stable_adaptive_a_seed1_nccl.json", 1),
)


def test_threeway_preflight_propagates_protocol_validation_failure(tmp_path):
    """A malformed source protocol must stop before any training is launched."""
    project_root = tmp_path / "project"
    config_dir = project_root / "exps"
    config_dir.mkdir(parents=True)
    shutil.copy2(SCRIPT, project_root)
    for source_name, _ in SOURCES:
        shutil.copy2(ROOT / "exps" / source_name, config_dir)

    bad_path = config_dir / SOURCES[0][0]
    bad_config = json.loads(bad_path.read_text())
    bad_config["epochs"] = 19
    bad_path.write_text(json.dumps(bad_config))
    conda_sh = tmp_path / "conda.sh"
    conda_sh.write_text("conda() { :; }\n")

    result = subprocess.run(
        ["bash", SCRIPT.name],
        cwd=project_root,
        env={
            **os.environ,
            "CONDA_SH": str(conda_sh),
            "PREPARE_ONLY": "1",
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "PREPARE_ONLY complete" not in result.stdout
