import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trainer import task_count_to_run


def test_task_count_to_run_defaults_to_all_tasks():
    assert task_count_to_run({}, 10) == 10


def test_task_count_to_run_limits_calibration_runs():
    assert task_count_to_run({"max_tasks": 3}, 10) == 3
    assert task_count_to_run({"max_tasks": 20}, 10) == 10


def test_task_count_to_run_rejects_non_positive_limits():
    with pytest.raises(ValueError, match="max_tasks"):
        task_count_to_run({"max_tasks": 0}, 10)
