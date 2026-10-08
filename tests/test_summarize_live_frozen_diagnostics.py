import subprocess
import sys
from pathlib import Path

from scripts.summarize_live_frozen_diagnostics import task_row, paired_rows


def test_summary_uses_old_class_change_and_pairs_matching_seed_task():
    metrics = {name: {"old": {"top1": score}, "global_top1": score}
               for name, score in (("S0", 90.), ("S1", 85.),
                                   ("S2", 86.), ("S3", 87.),
                                   ("S4", 87.), ("S5", 84.),
                                   ("S6", 83.), ("H_LS", 88.),
                                   ("formal_new_prototypes", 82.))}
    base = {"dataset": "imagenetr", "seed": 1995, "task": 1,
            "aggregate": {"r_rec": .1, "r_weighted": .2}, "metrics": metrics}
    live = task_row(base, "live")
    frozen = task_row({**base, "aggregate": {"r_rec": 0., "r_weighted": 0.}},
                      "frozen")
    assert live["old_drop_hls"] == 2.0
    assert live["old_drop_deploy"] == 8.0
    pair = paired_rows([live, frozen])[0]
    assert pair["dataset"] == "imagenetr"
    assert pair["seed"] == 1995
    assert pair["task"] == 1
    assert pair["delta_r_rec"] == .1


def test_summary_script_can_be_invoked_from_project_root():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/summarize_live_frozen_diagnostics.py", "--help"],
        cwd=root, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
