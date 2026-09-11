"""Protocol checks for the two-GPU Pareto-knee calibration queue."""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_pareto_knee_queue_is_nohup_safe_and_preserves_calibration_protocol():
    script = ROOT / "run_pareto_knee_calibration_2gpu.sh"
    source = script.read_text()

    subprocess.run(["bash", "-n", str(script)], check=True)

    assert 'GPU_IDS="${GPU_IDS:-0,1}"' in source
    assert 'batch_size": 128' in source
    assert 'max_tasks": 3' in source
    assert 'sa_dual_head": False' in source
    assert 'sa_adaptive_a_strategy": "pareto_knee"' in source
    assert 'sa_adaptive_a_crossfit_interval": 4' in source
    assert 'sa_resume": False' in source
    assert "PYTHONUNBUFFERED=1" in source
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in source
    assert "TORCH_DETERMINISTIC=1" in source
    assert "--nproc_per_node=1" in source
    assert "pgrep -f" in source
    assert "refusing to overwrite" in source
    assert "date '+%F %T'" in source
    assert "status=$status" in source
    assert "c100_coordinate_stable_adaptive_a_seed1993_nccl.json" in source
    assert "cub_coordinate_stable_adaptive_a_seed1_nccl.json" in source
