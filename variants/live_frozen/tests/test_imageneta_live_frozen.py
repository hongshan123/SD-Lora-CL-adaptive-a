"""Safety and protocol tests for the paired ImageNet-A study."""

import json
import os
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest
from PIL import Image

from scripts.imageneta_live_frozen import (
    archive_attempt,
    audit_dataset,
    build_jobs,
    paired_summary,
    parse_training_log,
    prepare_data_link,
    validate_gpu_pairs,
    validate_source,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]


def make_dataset(root):
    for split, offset in (("train", 1), ("test", 20)):
        for index, label in enumerate(("n0001", "n0002")):
            directory = root / split / label
            directory.mkdir(parents=True)
            Image.new("RGB", (8, 8), (offset + index, 2, 3)).save(directory / "x.png")


def test_full_protocol_and_order(tmp_path):
    jobs = build_jobs(ROOT, tmp_path / "configs", tmp_path / "runtime", "unit")
    assert [(job["method"], job["seed"]) for job in jobs] == [
        (method, seed) for method in ("frozen", "live") for seed in (1995, 1996, 1997)
    ]
    for job in jobs:
        config = json.loads(Path(job["config"]).read_text())
        assert config["dataset"] == "imageneta"
        assert config["batch_size"] == 64
        assert config["device"] == ["0", "1"]
        assert config["max_tasks"] == 10
        assert config["init_cls"] == config["increment"] == 20
        assert config["init_epoch"] == config["epochs"] == 20
        assert config["lora_rank"] == 10
        assert config["optimizer"] == "sgd"
        assert config["scheduler"] == "constant"
        assert config["lrate"] == config["init_lr"] == 0.01
        assert config["sa_coordinate_stable_transport"] is True
        assert config["sa_live_a_coordinate_align"] is True
        assert config["sa_live_a_absorb_mode"] == "bounded_norm_calibrated_absorb"
        assert config["sa_save_task_snapshots"] is True
        assert config["sa_dual_head"] is config["sa_adaptive_a_enabled"] is False
        assert config["memory_size"] == 0
    assert len({job["log"] for job in jobs}) == 6
    assert len({job["result"] for job in jobs}) == 6


def test_only_a_policy_differs_in_each_pair(tmp_path):
    jobs = build_jobs(ROOT, tmp_path / "configs", tmp_path, "unit")
    for frozen, live in zip(jobs[:3], jobs[3:]):
        a = json.loads(Path(frozen["config"]).read_text())
        b = json.loads(Path(live["config"]).read_text())
        assert {key for key in a if a[key] != b[key]} == {
            "prefix", "filepath", "sa_train_a_all_tasks"
        }
        assert not a["sa_train_a_all_tasks"] and b["sa_train_a_all_tasks"]


def test_ten_additional_seeds_keep_the_paired_protocol(tmp_path):
    seeds = tuple(range(1, 11))
    jobs = build_jobs(ROOT, tmp_path / "configs", tmp_path / "runtime", "ten_seeds", seeds=seeds)
    assert [(job["method"], job["seed"]) for job in jobs] == [
        (method, seed) for method in ("frozen", "live") for seed in seeds
    ]
    assert len({job["log"] for job in jobs}) == 20
    assert len({job["result"] for job in jobs}) == 20
    for frozen, live in zip(jobs[:10], jobs[10:]):
        a = json.loads(Path(frozen["config"]).read_text())
        b = json.loads(Path(live["config"]).read_text())
        assert a["seed"] == b["seed"] == [frozen["seed"]]
        assert a["batch_size"] == b["batch_size"] == 64
        assert a["max_tasks"] == b["max_tasks"] == 10
        assert a["init_epoch"] == a["epochs"] == 20
        assert {key for key in a if a[key] != b[key]} == {
            "prefix", "filepath", "sa_train_a_all_tasks"
        }


def test_custom_seed_smoke_uses_first_registered_seed(tmp_path):
    jobs = build_jobs(ROOT, tmp_path / "configs", tmp_path, "ten_seeds", smoke=True, seeds=range(1, 11))
    assert len(jobs) == 2
    assert all(job["seed"] == 1 for job in jobs)


@pytest.mark.parametrize("seeds", [(), (1, 1), (-1, 2), (True, 2), (1.5, 2)])
def test_seed_registration_rejects_invalid_or_duplicate_seeds(tmp_path, seeds):
    with pytest.raises(ValueError, match="seed"):
        build_jobs(ROOT, tmp_path / "configs", tmp_path, "invalid", seeds=seeds)
    assert not (tmp_path / "configs").exists()


def test_smoke_is_separate_and_cannot_overwrite_configs(tmp_path):
    jobs = build_jobs(ROOT, tmp_path / "configs", tmp_path, "unit", smoke=True)
    assert len(jobs) == 2
    for job in jobs:
        config = json.loads(Path(job["config"]).read_text())
        assert job["seed"] == 1995
        assert config["max_tasks"] == 2
        assert config["init_epoch"] == config["epochs"] == config["tuned_epoch"] == 2
        assert "smoke" in job["name"]
    with pytest.raises(FileExistsError):
        build_jobs(ROOT, tmp_path / "configs", tmp_path, "unit", smoke=True)


def test_registered_configs_may_be_validated_but_never_replaced(tmp_path):
    jobs = build_jobs(ROOT, tmp_path / "configs", tmp_path, "unit")
    assert build_jobs(ROOT, tmp_path / "configs", tmp_path, "unit", reuse_existing=True) == jobs
    path = Path(jobs[0]["config"])
    config = json.loads(path.read_text())
    config["batch_size"] = 128
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="registered config"):
        build_jobs(ROOT, tmp_path / "configs", tmp_path, "unit", reuse_existing=True)


def test_sealed_source_rejects_main_worktree(tmp_path):
    with pytest.raises(ValueError, match="runtime/source"):
        validate_source(tmp_path, {"source": str(ROOT), "git_commit": "unused"})


def test_pretrained_hash_check_does_not_touch_rng():
    import torch
    from scripts.train_imageneta_paired import check_base_weights
    from utils.canonical_hash import hash_named_tensors

    model = torch.nn.Linear(3, 4)
    before = torch.get_rng_state().clone()
    expected = hash_named_tensors(dict(model.state_dict()))
    assert check_base_weights(model, expected) == expected
    assert torch.equal(before, torch.get_rng_state())
    with pytest.raises(ValueError, match="pretrained weights"):
        check_base_weights(model, "wrong hash")


def test_monitor_and_queue_can_publish_json_concurrently(tmp_path, monkeypatch):
    import os

    replace = os.replace
    barrier = threading.Barrier(2)

    def synchronized_replace(source, target):
        barrier.wait(timeout=5)
        return replace(source, target)

    monkeypatch.setattr(os, "replace", synchronized_replace)
    output = tmp_path / "summary.json"
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(write_json, output, {"value": value}) for value in (1, 2)]
        for future in futures:
            future.result()
    assert json.loads(output.read_text())["value"] in (1, 2)


def test_runner_creates_parent_but_not_guarded_output_directory(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts import imageneta_live_frozen as study

    runtime = tmp_path / "runtime"
    jobs = build_jobs(ROOT, tmp_path / "configs", runtime, "unit")
    source = runtime / "source"
    source.mkdir(parents=True)
    write_json(runtime / "manifest.json", {"base_weight_sha256": "unit"})
    monkeypatch.setattr(study, "validate_source", lambda *_args: None)

    def fake_training(command, **kwargs):
        output = Path(jobs[0]["result"])
        assert output.parent.is_dir(), "missing parent for run_guard lock file"
        assert not output.exists(), "only the trainer may create the guarded run directory"
        assert command[0] == "nohup"
        assert command[-2] == "scripts/train_imageneta_paired.py"
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(study.subprocess, "run", fake_training)
    assert study.run_one(source, jobs[0], "0,1", 2) == 1


def test_reseal_is_only_allowed_before_formal_training(tmp_path):
    from scripts.imageneta_live_frozen import reset_preformal_seal

    manifest = {"git_commit": "old", "base_weight_sha256": "base", "formal": [{"log": str(tmp_path / "formal.log"), "result": str(tmp_path / "formal_result")}], "smoke": []}
    write_json(tmp_path / "manifest.json", manifest)
    (tmp_path / "formal.log").write_text("started")
    with pytest.raises(ValueError, match="formal"):
        reset_preformal_seal(tmp_path, manifest)
    (tmp_path / "formal.log").unlink()
    reset = reset_preformal_seal(tmp_path, manifest)
    assert "git_commit" not in reset
    assert len(list((tmp_path / "archive").glob("seal_*/*.json"))) == 1


def test_data_audit_is_stable_and_records_content_hashes(tmp_path):
    make_dataset(tmp_path)
    a = audit_dataset(tmp_path, expected_classes=2)
    b = audit_dataset(tmp_path, expected_classes=2)
    assert a["data_sha256"] == b["data_sha256"]
    assert a["counts"] == {"train": 2, "test": 2}
    assert a["class_to_idx"] == {"n0001": 0, "n0002": 1}
    assert a["relative_path_overlap"] == 2
    assert a["content_hash_overlap"] == 0
    assert len(a["files"]) == 4


def test_data_audit_rejects_content_leak_even_if_names_differ(tmp_path):
    make_dataset(tmp_path)
    source = tmp_path / "train/n0001/x.png"
    (tmp_path / "test/n0002/leak.png").write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="content.*overlap"):
        audit_dataset(tmp_path, expected_classes=2)


def test_data_audit_rejects_bad_images_and_class_mapping(tmp_path):
    make_dataset(tmp_path)
    (tmp_path / "test/n0001/x.png").write_bytes(b"broken image")
    with pytest.raises(ValueError, match="decode"):
        audit_dataset(tmp_path, expected_classes=2)
    (tmp_path / "test/n0001/x.png").unlink()
    with pytest.raises(ValueError, match="empty"):
        audit_dataset(tmp_path, expected_classes=2)


def test_data_link_never_replaces_existing_directory(tmp_path):
    root = tmp_path / "repo"
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    link = prepare_data_link(root, dataset)
    assert link.is_symlink() and link.resolve() == dataset.resolve()
    assert prepare_data_link(root, dataset) == link
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(FileExistsError):
        prepare_data_link(root, other)


@pytest.mark.parametrize("pairs", ["2,3", "0,1;1,4", "0", "0,0", "0,8", ""])
def test_gpu_slots_reject_unsafe_values(pairs):
    with pytest.raises(ValueError):
        validate_gpu_pairs(pairs)


def test_gpu_slots_allow_three_independent_pairs():
    assert validate_gpu_pairs("0,1;4,5;6,7") == ("0,1", "4,5", "6,7")


def test_new_gpu_exclusion_policy_allows_gpu2_and_rejects_gpu3_gpu4():
    assert validate_gpu_pairs("0,1;2,5;6,7", excluded_gpus=(3, 4)) == ("0,1", "2,5", "6,7")
    for pairs in ("0,1;3,5", "0,1;4,5", "0,1;2,8", "0,1;2,2"):
        with pytest.raises(ValueError):
            validate_gpu_pairs(pairs, excluded_gpus=(3, 4))


def test_queue_enforces_manifest_gpu_exclusions_before_launch(tmp_path):
    from scripts.imageneta_live_frozen import queue

    write_json(tmp_path / "manifest.json", {"excluded_gpus": [3, 4]})
    with pytest.raises(ValueError, match="GPU"):
        queue(tmp_path, "formal", ("0,1", "4,5"))


def test_queue_defaults_to_registered_gpu_slots(tmp_path, monkeypatch):
    from scripts import imageneta_live_frozen as study

    pairs = ("0,1", "2,5", "6,7")
    write_json(tmp_path / "manifest.json", {"excluded_gpus": [3, 4], "gpu_pairs": list(pairs), "source": str(ROOT), "git_commit": "unit", "smoke": []})
    monkeypatch.setattr(study, "validate_seal", lambda *_args: None)
    monkeypatch.setattr(study, "check_idle", lambda slots: None)
    monkeypatch.setattr(study, "report", lambda *_args: None)
    observed = []
    monkeypatch.setattr(study, "dispatch_jobs", lambda _source, _jobs, slots, _world_size, **_kwargs: observed.append(slots) or 0)
    assert study.queue(tmp_path, "smoke") == 0
    assert observed == [pairs]


def test_launcher_without_explicit_slots_uses_manifest_policy(tmp_path):
    runtime = tmp_path / "runtime"
    write_json(runtime / "manifest.json", {"source": str(ROOT), "gpu_pairs": ["0,1", "2,5", "6,7"], "excluded_gpus": [3, 4]})
    binary = tmp_path / "bin/systemd-run"
    binary.parent.mkdir()
    binary.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
    binary.chmod(0o755)
    env = dict(os.environ, PATH=str(binary.parent) + os.pathsep + os.environ["PATH"])
    result = subprocess.run(["bash", str(ROOT / "run_imageneta_live_frozen_queue.sh"), str(runtime), "formal"], env=env, capture_output=True, text=True, check=True)
    assert "--gpu-pairs\n0,1;2,5;6,7\n" in result.stdout
    assert "0,1;4,5;6,7" not in result.stdout


def complete_log():
    return (
        "CNN: {'total': np.float64(80.0), 'old': 0, 'new': 80.0}\n"
        "CNN: {'total': 70.0, 'old': 65.0, 'new': 75.0}\n"
        "CNN top1 curve: [np.float64(80.0), np.float64(70.0)]\n"
        "Accuracy Matrix (CNN):\n[[80. 65.]\n [0. 75.]]\n"
        "Forgetting (CNN): 15.0\n"
        "===== END test status=0 =====\n"
    )


def test_metrics_require_complete_success_and_preserve_old_new():
    result = parse_training_log(complete_log(), tasks=2)
    assert result["complete"]
    assert result["final"] == 70
    assert result["aaa"] == 75
    assert result["forgetting"] == 15
    assert result["old_curve"] == [0, 65]
    assert result["new_curve"] == [80, 75]
    assert not parse_training_log(complete_log(), tasks=10)["complete"]
    assert not parse_training_log(complete_log().replace("status=0", "status=1"), tasks=2)["complete"]
    assert not parse_training_log(complete_log().split("===== END")[0], tasks=2)["complete"]


def test_metrics_reject_errors_even_with_success_marker():
    assert not parse_training_log("Traceback (most recent call last):\n" + complete_log(), tasks=2)["complete"]


def test_retries_preserve_old_log_and_artifacts(tmp_path):
    log = tmp_path / "failed.log"
    result = tmp_path / "results"
    result.mkdir()
    (result / "partial.pt").write_bytes(b"partial")
    log.write_text("status=1")
    archive = archive_attempt({"log": str(log), "result": str(result)}, tmp_path / "archive")
    assert (archive / "failed.log").read_text() == "status=1"
    assert (archive / "results/partial.pt").read_bytes() == b"partial"
    assert not log.exists() and not result.exists()


def test_pair_summary_uses_sample_std_and_all_registered_seeds():
    records = []
    for method, finals in (("frozen", [60, 61, 62]), ("live", [61, 63, 63])):
        for seed, final in zip((1995, 1996, 1997), finals):
            records.append(dict(method=method, seed=seed, complete=True, final=final, aaa=final + 5, forgetting=4))
    summary = paired_summary(records)
    assert summary["frozen"]["final"] == {"mean": 61, "sample_std": 1}
    assert summary["paired_live_minus_frozen"]["final"]["values"] == [1, 2, 1]
    assert paired_summary(records[:-1]) is None


def test_pair_summary_requires_all_ten_registered_pairs():
    seeds = tuple(range(1, 11))
    records = [
        dict(method=method, seed=seed, complete=True, final=60 + seed + (method == "live"), aaa=70 + seed, forgetting=4)
        for method in ("frozen", "live") for seed in seeds
    ]
    summary = paired_summary(records, seeds=seeds)
    assert summary["frozen"]["final"]["mean"] == 65.5
    assert summary["paired_live_minus_frozen"]["final"]["values"] == [1] * 10
    assert paired_summary(records[:-1], seeds=seeds) is None
    assert paired_summary(records + [dict(records[0])], seeds=seeds) is None


def test_single_pair_has_no_defined_sample_standard_deviation():
    records = [
        dict(method=method, seed=1, complete=True, final=60, aaa=70, forgetting=4)
        for method in ("frozen", "live")
    ]
    result = paired_summary(records, seeds=(1,))
    assert result["frozen"]["final"]["sample_std"] is None


def test_report_uses_manifest_seeds_instead_of_legacy_defaults(tmp_path, monkeypatch):
    from scripts import imageneta_live_frozen as study

    seeds = tuple(range(1, 11))
    records = [
        dict(method=method, seed=seed, complete=True, final=60, aaa=70, forgetting=4, curve=[], old_curve=[], new_curve=[])
        for method in ("frozen", "live") for seed in seeds
    ]
    write_json(tmp_path / "manifest.json", {"seeds": list(seeds), "formal": records})
    monkeypatch.setattr(study, "inspect_job", lambda job, verify_artifacts: job)
    result = study.report(tmp_path, "formal")
    assert result["summary"] is not None
    assert len(result["summary"]["paired_live_minus_frozen"]["final"]["values"]) == 10


def test_smoke_policy_checks_both_endpoints_and_task0():
    import torch
    from scripts.verify_imageneta_smoke import compare_policy_tensors

    initial = [torch.ones(2, 3), torch.zeros(2, 3)]
    changed = [initial[0] + 0.01, initial[1].clone()]
    assert compare_policy_tensors(initial, initial, initial, changed)["passed"]
    with pytest.raises(ValueError, match="Frozen"):
        compare_policy_tensors(initial, changed, initial, changed)
    with pytest.raises(ValueError, match="Task0"):
        compare_policy_tensors(initial, initial, changed, changed)
    with pytest.raises(ValueError, match="Live"):
        compare_policy_tensors(initial, initial, initial, initial)


def test_rebuild_gate_uses_relative_feature_error_and_preregistered_logits():
    import torch
    from scripts.verify_imageneta_smoke import rebuild_errors

    reference = torch.tensor([100.0, 2.0])
    rebuilt = reference + 2e-5
    logits0 = torch.zeros(2)
    logits1 = logits0 + 5e-7
    result = rebuild_errors(reference, rebuilt, logits0, logits1)
    assert result["feature_max_abs_diff"] > 1e-5
    assert result["feature_relative_l2_error"] < 1e-5
    with pytest.raises(ValueError, match="rebuild"):
        rebuild_errors(reference, rebuilt, logits0, logits0 + 1e-4)
    with pytest.raises(ValueError, match="rebuild"):
        rebuild_errors(reference, reference + 1, logits0, logits1)
    with pytest.raises(ValueError, match="finite"):
        rebuild_errors(reference, reference * float("nan"), logits0, logits1)


def test_launcher_syntax_and_bad_phase():
    path = ROOT / "run_imageneta_live_frozen_queue.sh"
    assert path.is_file()
    subprocess.run(["bash", "-n", str(path)], check=True)
    result = subprocess.run(["bash", str(path), "/missing/runtime", "wrong"], capture_output=True, text=True)
    assert result.returncode != 0
