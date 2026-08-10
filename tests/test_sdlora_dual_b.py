"""Unit tests for the P4 rank-1 SD-LoRA + Dual-B baseline learner."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.sdlora_dual_b import Learner


def _learner(schedule):
    learner = object.__new__(Learner)
    learner._dual_schedule = schedule
    return learner


def test_dual_lambda_schedule_b_matches_complete_method():
    learner = _learner("B")
    assert learner._dual_lambda(0, 10) == 0.0
    assert learner._dual_lambda(1, 10) == 0.0
    assert abs(learner._dual_lambda(2, 10) - 0.25) < 1e-9
    assert abs(learner._dual_lambda(5, 10) - 1.0) < 1e-9
    assert learner._dual_lambda(9, 10) == 1.0


def test_dual_lambda_schedule_a_matches_complete_method():
    learner = _learner("A")
    assert learner._dual_lambda(0, 10) == 0.0
    assert abs(learner._dual_lambda(2, 10) - 0.5) < 1e-9
    assert abs(learner._dual_lambda(4, 10) - 1.0) < 1e-9
    assert learner._dual_lambda(9, 10) == 1.0


def test_dual_lambda_single_task_is_one():
    learner = _learner("B")
    assert learner._dual_lambda(0, 1) == 1.0
