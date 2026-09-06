"""Protocol checks for the two-GPU Adaptive-A three-way comparison."""

import json
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_adaptive_a_three_way_2gpu_queue.sh"
DATASETS = (
    ("c100", 1993),
    ("inr", 1995),
    ("cub", 1),
)
GROUPS = (
    ("live_a", True, False),
    ("frozen_a", False, False),
    ("adaptive_a", True, True),
)


def test_three_way_queue_runs_nine_strict_two_gpu_controls(tmp_path):
    project_root = tmp_path / "project"
    bin_dir = tmp_path / "bin"
    project_root.mkdir()
    bin_dir.mkdir()
    shutil.copy2(SCRIPT, project_root)

    config_dir = project_root / "exps"
    config_dir.mkdir()
    for dataset, seed in DATASETS:
        source = ROOT / "exps" / (
            f"{dataset}_coordinate_stable_adaptive_a_seed{seed}_nccl.json"
        )
        shutil.copy2(source, config_dir)

    conda_sh = tmp_path / "conda.sh"
    conda_sh.write_text("conda() { :; }\n")
    calls_file = tmp_path / "torchrun-calls.txt"
    (bin_dir / "pgrep").write_text("#!/usr/bin/env bash\nexit 1\n")
    (bin_dir / "git").write_text("#!/usr/bin/env bash\necho deadbeef\n")
    (bin_dir / "torchrun").write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s|%s\\n' \"$*\" \"$CUDA_VISIBLE_DEVICES\" >> \"$TORCHRUN_CALLS\"\n"
    )
    for command in bin_dir.iterdir():
        command.chmod(0o755)

    result = subprocess.run(
        ["bash", SCRIPT.name],
        cwd=project_root,
        env={
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "GPU_IDS": "2,3",
            "CONDA_SH": str(conda_sh),
            "TORCHRUN_CALLS": str(calls_file),
        },
        capture_output=True,
        text=True,
        check=True,
    )

    calls = calls_file.read_text().splitlines()
    assert len(calls) == len(DATASETS) * len(GROUPS)
    assert all("--nproc_per_node=2" in call for call in calls)
    assert all(call.endswith("|2,3") for call in calls)

    runtime_dir = project_root / ".runtime_adaptive_a_three_way_2gpu"
    expected_names = []
    for dataset, seed in DATASETS:
        for group, train_a, adaptive_a in GROUPS:
            name = f"{dataset}_coordinate_stable_{group}_seed{seed}_2gpu_bs64"
            expected_names.append(name)
            config = json.loads((runtime_dir / f"{name}.json").read_text())
            assert config["batch_size"] == 64
            assert config["sa_train_a_all_tasks"] is train_a
            assert config["sa_adaptive_a_enabled"] is adaptive_a
            assert config["prefix"] == name
            assert config["filepath"] == f"./{name.upper()}/"

    called_configs = [
        Path(call.split("--config=./", 1)[1].split("|", 1)[0]).stem
        for call in calls
    ]
    assert called_configs == expected_names
    assert "ADAPTIVE-A THREE-WAY QUEUE DONE" in result.stdout


def test_three_way_queue_rejects_non_two_gpu_launch(tmp_path):
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=tmp_path,
        env={"GPU_IDS": "2,3,4"},
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "requires exactly 2 GPU IDs" in result.stderr
