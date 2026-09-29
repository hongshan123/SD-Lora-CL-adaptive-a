"""Safety and protocol tests for the paired ImageNet-A study."""

import json
from pathlib import Path
import subprocess

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


def test_launcher_syntax_and_bad_phase():
    path = ROOT / "run_imageneta_live_frozen_queue.sh"
    assert path.is_file()
    subprocess.run(["bash", "-n", str(path)], check=True)
    result = subprocess.run(["bash", str(path), "/missing/runtime", "wrong"], capture_output=True, text=True)
    assert result.returncode != 0
