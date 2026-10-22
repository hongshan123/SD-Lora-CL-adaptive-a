"""Dependency, exit-status and safe parent-only handover regressions."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def module():
    spec = importlib.util.find_spec("scripts.hoep_cifar_parallel_queue")
    assert spec is not None, "four-pair scheduler is missing"
    from scripts import hoep_cifar_parallel_queue
    return hoep_cifar_parallel_queue


def test_ready_jobs_exclude_active_completed_and_unfinished_reference_dependencies():
    jobs = [dict(name="low94", method="hoep", seed=1994),
            dict(name="random94", method="random", seed=1994),
            dict(name="high94", method="high", seed=1994),
            dict(name="random93", method="random", seed=1993),
            dict(name="frozen95", method="frozen", seed=1995),
            dict(name="live94", method="live", seed=1994)]
    ready = module().ready_jobs(jobs, {"frozen95"}, {"live94"})
    assert [j["name"] for j in ready] == ["low94", "random93"]
    ready = module().ready_jobs(jobs, {"low94", "frozen95"}, {"live94"})
    assert {j["name"] for j in ready} == {"random93", "random94", "high94"}


def test_reference_must_have_correct_seed_and_all_nine_complete_branch_counts(tmp_path):
    reference = tmp_path / "reference.json"
    good = {"version": 1, "seed": 1994, "tasks": {str(t): [5] * 24 for t in range(1, 10)}}
    reference.write_text(json.dumps(good))
    module().validate_reference(reference, 1994)
    with pytest.raises(ValueError, match="seed"):
        module().validate_reference(reference, 1995)
    good["tasks"].pop("9")
    reference.write_text(json.dumps(good))
    with pytest.raises(ValueError, match="task"):
        module().validate_reference(reference, 1994)


@pytest.mark.parametrize("code", [0, 7])
def test_unreaped_process_exit_status_can_be_verified_before_parent_reaps_it(code):
    implementation = module()
    child = subprocess.Popen([sys.executable, "-c", f"raise SystemExit({code})"])
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            info = implementation.process_info(child.pid)
            if info and info["state"] == "Z":
                break
            time.sleep(.01)
        assert info["state"] == "Z"
        assert info["exit_code"] == code
        assert child.wait() == code
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_pausing_scheduler_parent_leaves_its_training_child_running(tmp_path):
    implementation = module()
    heartbeat = tmp_path / "heartbeat"
    child_source = "import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); " \
                   "[(p.write_text(str(i)),time.sleep(.02)) for i in range(50)]"
    parent_source = "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]); " \
                    "print(p.pid,flush=True); sys.exit(p.wait())"
    parent = subprocess.Popen([sys.executable, "-c", parent_source, child_source, str(heartbeat)],
                              stdout=subprocess.PIPE, text=True)
    child_pid = int(parent.stdout.readline())
    try:
        deadline = time.monotonic() + 3
        while not heartbeat.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        identity = implementation.process_info(parent.pid)
        implementation.pause_parent(parent.pid, identity["start_ticks"])
        before = int(heartbeat.read_text())
        time.sleep(.1)
        assert implementation.process_info(parent.pid)["state"] == "T"
        assert implementation.process_info(child_pid)["state"] != "T"
        assert int(heartbeat.read_text()) > before
    finally:
        os.kill(parent.pid, signal.SIGCONT)
        parent.wait(timeout=4)
        parent.stdout.close()


def test_success_record_requires_zero_exit_and_a_complete_log(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("CNN top1 curve: [90.,89.]\nForgetting (CNN): 1\n")
    entry = {"name": "run", "method": "live", "log": str(log), "config": str(tmp_path / "config.json")}
    proof = {"phase": "complete", "returncode": 0}
    with pytest.raises(ValueError, match="10"):
        module().validated_result(entry, proof)
    proof["returncode"] = 7
    with pytest.raises(ValueError, match="exit"):
        module().validated_result(entry, proof)


def test_recovered_worker_disappearance_is_detected(tmp_path, monkeypatch):
    implementation = module()
    path = tmp_path / "proof.json"
    path.write_text(json.dumps({"phase": "running"}))
    monkeypatch.setattr(implementation, "process_info", lambda pid: None)
    running = {"name": "recovered", "pid": 123, "start_ticks": 456}
    with pytest.raises(RuntimeError, match="without a terminal result"):
        implementation.worker_progress(running, path)


def test_completion_proof_race_is_reread_before_missing_worker_failure(tmp_path, monkeypatch):
    implementation = module()
    path = tmp_path / "proof.json"
    path.write_text(json.dumps({"phase": "running"}))

    def worker_exited(pid):
        path.write_text(json.dumps({"phase": "complete", "returncode": 0}))
        return None

    monkeypatch.setattr(implementation, "process_info", worker_exited)
    running = {"name": "recovered", "pid": 123, "start_ticks": 456}
    assert implementation.worker_progress(running, path)["returncode"] == 0
