"""Ensure the queue has exactly the authorized cells and preserves protocol."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def module():
    spec = importlib.util.find_spec("scripts.hoep_cifar_ablation")
    assert spec is not None, "CIFAR ablation manifest/queue is missing"
    from scripts import hoep_cifar_ablation
    return hoep_cifar_ablation


def test_manifest_runs_only_missing_fifteen_cells_with_shared_protocol(tmp_path):
    root = Path(__file__).resolve().parents[1]
    data = module().generate_manifest(root, tmp_path, tmp_path / "previous", "test")
    assert len(data["jobs"]) == 15
    assert len(data["reused"]) == 3
    cells = {(job["method"], job["seed"]) for job in data["jobs"] + data["reused"]}
    assert cells == {(method, seed) for method in
                     ("frozen", "live", "hoep", "canonical_live", "random", "high")
                     for seed in (1993, 1994, 1995)}
    for entry in data["jobs"]:
        config = json.loads(Path(entry["config"]).read_text())
        assert config["seed"] == [entry["seed"]]
        assert config["max_tasks"] == 10
        assert config["init_epoch"] == config["epochs"] == 20
        assert config["batch_size"] == 64 and config["device"] == ["0", "1"]
        assert config["sa_live_a_absorb_mode"] == "operator_preserving_absorb"
        assert config["sa_live_a_coordinate_align"] is True
        assert not any(config[key] for key in
                       ("sa_normalize_current_branch", "sa_coordinate_stable_transport", "sa_dual_head", "sa_hbd_enabled"))
        if entry["method"] in ("random", "high"):
            reference = config["sa_hoep_reference_counts_path"]
            assert f"seed{entry['seed']}" in reference
            if entry["seed"] != 1993:
                dependency = next(j for j in data["jobs"] if j["method"] == "hoep" and j["seed"] == entry["seed"])
                assert data["jobs"].index(dependency) < data["jobs"].index(entry)
                assert reference == str(Path(dependency["output"]) / "hoep_partition_schedule.json")


def test_incomplete_log_cannot_be_reported_as_completed(tmp_path):
    log = tmp_path / "partial.log"
    log.write_text("CNN top1 curve: [90., 89.]\n")
    with pytest.raises(ValueError, match="10"):
        module().read_metrics(log)
    log.write_text("CNN top1 curve: [90., 89., 88., 87., 86., 85., 84., 83., 82., 81.]\nForgetting (CNN): 9\n")
    result = module().read_metrics(log)
    assert result["final"] == 81.
    assert result["aaa"] == 85.5


def test_metrics_accept_actual_numpy_scalar_log_representation(tmp_path):
    log = tmp_path / "numpy.log"
    log.write_text("CNN top1 curve: [np.float64(90.0), np.float64(89.0), np.float64(88.0), "
                   "np.float64(87.0), np.float64(86.0), np.float64(85.0), np.float64(84.0), "
                   "np.float64(83.0), np.float64(82.0), np.float64(81.0)]\nForgetting (CNN): 9\n")
    result = module().read_metrics(log)
    assert result["final"] == 81.0
    assert result["aaa"] == 85.5


def test_frozen_provenance_detects_source_and_config_changes(tmp_path):
    implementation = module()
    freeze = getattr(implementation, "freeze_sources", None)
    verify = getattr(implementation, "verify_sources", None)
    assert callable(freeze) and callable(verify), "source/config freeze is missing"
    project = tmp_path / "project"
    runtime = tmp_path / "runtime"
    project.mkdir()
    runtime.mkdir()
    source = project / "main.py"
    source.write_text("print('original')\n")
    config = runtime / "config.json"
    config.write_text('{"seed": [1993]}\n')
    manifest = {"jobs": [{"config": str(config)}], "preflight": []}
    freeze(project, runtime, manifest)
    verify(project, runtime)
    source.write_text("print('changed')\n")
    with pytest.raises(RuntimeError, match="source changed"):
        verify(project, runtime)
    source.write_text("print('original')\n")
    config.write_text('{"seed": [1994]}\n')
    with pytest.raises(RuntimeError, match="config changed"):
        verify(project, runtime)


def test_resume_reuses_only_explicitly_successful_complete_runs(tmp_path):
    implementation = module()
    completed = getattr(implementation, "completed_run_metrics", None)
    assert callable(completed), "safe completed-run recovery is missing"
    log = tmp_path / "result.log"
    log.write_text("CNN top1 curve: [90., 89.]\nForgetting (CNN): 1\n")
    entry = {"name": "example", "log": str(log)}
    events = tmp_path / "events.jsonl"
    events.write_text('{"kind":"training_start","name":"example"}\n')
    assert completed(entry, events, expected_tasks=2) is None
    events.write_text('{"kind":"training_complete","name":"example"}\n')
    assert completed(entry, events, expected_tasks=2)["final"] == 89.
    log.write_text("CNN top1 curve: [90.]\nForgetting (CNN): 1\n")
    with pytest.raises(ValueError, match="2"):
        completed(entry, events, expected_tasks=2)
