"""Protocol checks for the four-GPU Adaptive-A experiment queue."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    ("c100", 1993, "c100_coordinate_stable_adaptive_a_seed1993_nccl"),
    ("inr", 1995, "inr_coordinate_stable_adaptive_a_seed1995_nccl"),
    ("cub", 1, "cub_coordinate_stable_adaptive_a_seed1_nccl"),
)
FORBIDDEN_KEYS = {
    "sa_adaptive_operator_budget",
    "sa_adaptive_normcap",
}
ADAPTIVE_A = {
    "sa_adaptive_a_enabled": True,
    "sa_adaptive_a_stability_weight": 1.0,
    "sa_adaptive_a_gate_floor": 0.05,
    "sa_adaptive_a_gate_momentum": 0.9,
    "sa_adaptive_a_eps": 1e-8,
}


def test_adaptive_a_configs_preserve_the_no_aob_protocol():
    prefixes = set()
    filepaths = set()

    for dataset, seed, prefix in CONFIGS:
        base = json.loads(
            (ROOT / "exps" / f"{dataset}_coordinate_stable_opabsorb_seed{seed}_nccl.json")
            .read_text()
        )
        config = json.loads((ROOT / "exps" / f"{prefix}.json").read_text())

        expected = dict(base)
        expected.update(
            {
                "prefix": prefix,
                "filepath": f"./{prefix.upper()}/",
                "sa_live_a_absorb_mode": "bounded_norm_calibrated_absorb",
                **ADAPTIVE_A,
            }
        )
        assert config == expected
        assert config["batch_size"] == 32
        assert config["lora_rank"] == 10
        assert config["sa_resume"] is False
        assert config["sa_cumulative_merge"] == "live_a_aggregate_b"
        assert config["sa_live_a_coordinate_align"] is True
        assert config["sa_coordinate_stable_transport"] is True
        assert not FORBIDDEN_KEYS.intersection(config)
        assert not any("aob" in key.lower() for key in config)
        prefixes.add(config["prefix"])
        filepaths.add(config["filepath"])

    assert len(prefixes) == len(CONFIGS)
    assert len(filepaths) == len(CONFIGS)


def test_adaptive_a_queue_is_portable_four_gpu_fail_fast_and_nohup_safe():
    script = ROOT / "run_coordinate_stable_adaptive_a_queue.sh"
    source = script.read_text()

    subprocess.run(["bash", "-n", str(script)], check=True)

    assert 'PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"' in source
    assert 'CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"' in source
    assert 'GPU_IDS="${GPU_IDS:-0,1,2,3}"' in source
    assert 'HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"' in source
    assert "NPROC=$(awk -F',' '{print NF}' <<< \"$GPU_IDS\")" in source
    assert 'if [ "$NPROC" -ne 4 ]; then' in source
    assert "CUDA_VISIBLE_DEVICES=\"$GPU_IDS\"" in source
    assert "PYTHONUNBUFFERED=1" in source
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in source
    assert "TORCH_DETERMINISTIC=1" in source
    assert "--nproc_per_node=\"$NPROC\"" in source
    assert "preflight" in source
    assert "pgrep -f" in source
    assert "-e \"$output_dir\"" in source
    assert "-e \"$log_file\"" in source
    assert "date '+%F %T'" in source
    assert "/home/zhaoyang" not in source
    assert "/cuda" not in source
    assert '"$PROJECT_ROOT/${name}.log"' in source

    for _, _, prefix in CONFIGS:
        assert f'"exps/{prefix}.json"' in source


def test_adaptive_a_queue_runs_three_four_gpu_experiments_with_complete_logs(tmp_path):
    project_root = tmp_path / "project"
    bin_dir = tmp_path / "bin"
    project_root.mkdir()
    bin_dir.mkdir()
    shutil.copy2(ROOT / "run_coordinate_stable_adaptive_a_queue.sh", project_root)

    for _, _, prefix in CONFIGS:
        config_dir = project_root / "exps"
        config_dir.mkdir(exist_ok=True)
        shutil.copy2(ROOT / "exps" / f"{prefix}.json", config_dir)

    conda_sh = tmp_path / "conda.sh"
    conda_sh.write_text("conda() { :; }\n")
    calls_file = tmp_path / "torchrun-calls.jsonl"
    (bin_dir / "pgrep").write_text("#!/usr/bin/env bash\nexit 1\n")
    (bin_dir / "git").write_text("#!/usr/bin/env bash\necho deadbeef\n")
    (bin_dir / "torchrun").write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*|CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES\" >> \"$TORCHRUN_CALLS\"\n"
        "echo \"torchrun output: $*\"\n"
    )
    for command in bin_dir.iterdir():
        command.chmod(0o755)

    result = subprocess.run(
        ["bash", "run_coordinate_stable_adaptive_a_queue.sh"],
        cwd=project_root,
        env={
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "GPU_IDS": "0,1,2,3",
            "CONDA_SH": str(conda_sh),
            "TORCHRUN_CALLS": str(calls_file),
        },
        capture_output=True,
        text=True,
        check=True,
    )

    calls = calls_file.read_text().splitlines()
    assert [call.split("--config=./")[1].split("|")[0] for call in calls] == [
        f"exps/{prefix}.json" for _, _, prefix in CONFIGS
    ]
    assert all("--nproc_per_node=4" in call for call in calls)
    assert all("CUDA_VISIBLE_DEVICES=0,1,2,3" in call for call in calls)
    assert "ADAPTIVE-A QUEUE DONE" in result.stdout

    for _, _, prefix in CONFIGS:
        log = (project_root / f"{prefix}.log").read_text()
        assert f"START {prefix} GPUs=0,1,2,3" in log
        assert "commit=deadbeef config_sha=" in log
        assert "torchrun output:" in log
        assert f"END {prefix} status=0" in log
        assert not (project_root / prefix.upper()).exists()


@pytest.mark.parametrize("gpu_ids", ("0,1,2", "0,1,2,3,4"))
def test_adaptive_a_queue_rejects_non_four_gpu_launches(gpu_ids):
    result = subprocess.run(
        ["bash", "run_coordinate_stable_adaptive_a_queue.sh"],
        cwd=ROOT,
        env={"GPU_IDS": gpu_ids},
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "requires exactly 4 GPU IDs" in result.stderr
