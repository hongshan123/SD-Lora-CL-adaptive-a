"""Tests for the dynamic Frozen/Live/Adaptive task-length queue."""

import json
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_tasklen_fla_configs import (
    DATASETS,
    METHODS,
    TASK_COUNTS,
    build_config,
    generate_configs,
)
import scripts.tasklen_fla_queue as queue_module
from scripts.tasklen_fla_queue import active_jobs, dispatch_jobs, pending_jobs, run_one
from scripts.tasklen_fla_queue import torchrun_command
from models.base import BaseLearner


def test_generator_covers_requested_lengths_and_methods(tmp_path):
    root = Path(__file__).resolve().parents[1]
    runtime = tmp_path / "runtime"
    manifest = generate_configs(root, runtime, "test", batch_size=64, world_size=2)

    assert len(manifest) == 4 * 4 * 3
    assert set(item["tasks"] for item in manifest) == {5, 10, 20, 50}
    assert set(item["method"] for item in manifest) == set(METHODS)
    assert len({item["name"] for item in manifest}) == len(manifest)


def test_generator_sets_adaptive_only_on_coordinate_aligned_adaptive_config():
    root = Path(__file__).resolve().parents[1]
    source = json.loads(
        (root / "exps" / DATASETS["c100"]["source"]).read_text()
    )
    runtime = Path("/tmp") / "tasklen-fla-test"
    _, frozen = build_config(source, "c100", 50, "frozen_a", runtime, "test")
    _, live = build_config(source, "c100", 50, "live_a", runtime, "test")
    _, adaptive = build_config(source, "c100", 50, "adaptive_a", runtime, "test")

    assert frozen["task_increments"] == [2] * 50
    assert frozen["sa_train_a_all_tasks"] is False
    assert frozen["sa_adaptive_a_enabled"] is False
    assert live["sa_train_a_all_tasks"] is True
    assert live["sa_live_a_coordinate_align"] is False
    assert adaptive["sa_train_a_all_tasks"] is True
    assert adaptive["sa_live_a_coordinate_align"] is True
    assert adaptive["sa_adaptive_a_enabled"] is True
    assert adaptive["sa_adaptive_a_strategy"] == "impact_ratio"


def test_pending_jobs_skips_only_explicit_success(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    jobs = [
        {"canonical_name": "done"},
        {"canonical_name": "failed"},
        {"canonical_name": "missing"},
    ]
    (project / "done.log").write_text("===== END done status=0 =====\n")
    (project / "failed.log").write_text("===== END failed status=1 =====\n")

    assert [job["canonical_name"] for job in pending_jobs(project, jobs)] == [
        "failed",
        "missing",
    ]


def test_active_jobs_recovers_started_run_without_end_marker(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    jobs = [
        {"name": "running", "canonical_name": "running"},
        {"name": "missing", "canonical_name": "missing"},
    ]
    (project / "running.log").write_text(
        "===== START running GPUs=4,5 canonical=running =====\n"
        "task 0\n"
    )

    recovered = active_jobs(project, jobs)

    assert [(job["name"], gpu_ids) for job, gpu_ids in recovered] == [
        ("running", "4,5")
    ]


def test_cnn_eval_clamps_topk_to_available_classes():
    class TwoClassNetwork:
        def eval(self):
            return self

        def forward(self, inputs):
            return {"logits": torch.tensor([[2.0, 1.0], [1.0, 2.0]])}

    learner = BaseLearner.__new__(BaseLearner)
    learner._network = TwoClassNetwork()
    learner._device = torch.device("cpu")
    learner.topk = 5
    loader = [(0, torch.zeros(2, 3), torch.tensor([0, 1]))]

    predictions, targets = learner._eval_cnn(loader)

    assert predictions.shape == (2, 2)
    assert targets.tolist() == [0, 1]


def test_top5_metric_is_defined_when_only_two_classes_are_seen():
    learner = BaseLearner.__new__(BaseLearner)
    learner.topk = 5
    learner._known_classes = 0
    learner.args = {"increment": 2, "task_increments": [2]}

    result = learner._evaluate(
        np.asarray([[0, 1], [1, 0]]), np.asarray([0, 1])
    )

    assert result["top1"] == 100.0
    assert result["top5"] == 100.0


def test_dispatch_reuses_slot_before_other_long_job_finishes():
    events = []
    lock = threading.Lock()

    def fake_runner(_root, job, gpu_ids, _world_size):
        with lock:
            events.append(("start", job["name"], gpu_ids, time.monotonic()))
        time.sleep(job["duration"])
        with lock:
            events.append(("end", job["name"], gpu_ids, time.monotonic()))
        return 0

    jobs = [
        {"name": "short", "duration": 0.03},
        {"name": "long", "duration": 0.12},
        {"name": "replacement", "duration": 0.01},
    ]
    assert dispatch_jobs(".", jobs, ("0,1", "4,5"), 2, runner=fake_runner) == 0

    start = {name: stamp for kind, name, _gpu, stamp in events if kind == "start"}
    end = {name: stamp for kind, name, _gpu, stamp in events if kind == "end"}
    assert start["replacement"] < end["long"]


def test_dispatch_stop_on_failure_does_not_launch_pending_jobs():
    started = []

    def fake_runner(_root, job, _gpu_ids, _world_size):
        started.append(job["name"])
        if job["name"] == "failure":
            return 1
        time.sleep(0.02)
        return 0

    jobs = [
        {"name": "failure"},
        {"name": "already_active"},
        {"name": "must_not_start"},
    ]
    status = dispatch_jobs(
        ".",
        jobs,
        ("0", "1"),
        1,
        runner=fake_runner,
        stop_on_failure=True,
    )

    assert status == 1
    assert set(started) == {"failure", "already_active"}


def test_queue_script_is_directly_executable_from_repository_root():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "tasklen_fla_queue.py"), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--gpu-pairs" in result.stdout
    assert "--resume" in result.stdout


def test_torchrun_command_uses_the_scheduler_interpreter():
    command = torchrun_command(2, "/tmp/config.json")

    assert command[:3] == [sys.executable, "-m", "torch.distributed.run"]


def test_run_one_archives_stale_result_directory(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"filepath": str(result_dir)}))
    monkeypatch.setattr(
        queue_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )

    job = {
        "name": "fresh_attempt",
        "canonical_name": "fresh_attempt",
        "config": str(config_path),
    }

    assert run_one(project, job, "0,1", 2) == 0
    assert not result_dir.exists()
    archived = list(tmp_path.glob("result.retry_*"))
    assert len(archived) == 1
    assert "archived_stale_result_dir=" in (project / "fresh_attempt.log").read_text()
