"""Guard the registered Historical-only extension, without running training."""

import importlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def module():
    assert (ROOT / "scripts/historical_only_multiseed.py").is_file()
    return importlib.import_module("scripts.historical_only_multiseed")


def templates():
    return {
        ds: json.loads((ROOT / ".runtime_historical_signal_multiseed" /
                        f"{ds}_historical_signal_seed1_t10_i4_bs128_2gpu.json").read_text())
        for ds in ("c100", "inr", "cub")
    }


def test_extension_has_exactly_thirty_jobs_and_never_repeats_seeds_one_to_three(tmp_path):
    m = module()
    jobs = m.make_jobs(templates(), tmp_path, "formal")
    assert len(jobs) == 30
    assert len({job["name"] for job in jobs}) == 30
    for dataset in ("c100", "inr", "cub"):
        assert {job["seed"] for job in jobs if job["dataset"] == dataset} == set(range(4, 14))
    assert [job["dataset"] for job in jobs[:3]] == ["c100", "inr", "cub"]


def test_extension_only_changes_seed_and_experiment_identity(tmp_path):
    m = module()
    old = templates()
    jobs = m.make_jobs(old, tmp_path, "formal")
    for job in jobs:
        new = json.loads(Path(job["config"]).read_text())
        original = old[job["dataset"]]
        for key in set(original) | set(new):
            if key not in {"prefix", "filepath", "seed"}:
                assert new[key] == original[key], (job["name"], key)
        assert new["seed"] == [job["seed"]]
        assert new["batch_size"] == 64 and new["epochs"] == 20


def test_config_generation_refuses_to_overwrite(tmp_path):
    m = module()
    m.make_jobs(templates(), tmp_path, "formal")
    with pytest.raises(FileExistsError):
        m.make_jobs(templates(), tmp_path, "formal")


def test_registered_slots_exclude_three_and_four():
    m = module()
    assert m.GPU_PAIRS == ("0,1", "2,5", "6,7")
    assert not {3, 4} & {int(gpu) for pair in m.GPU_PAIRS for gpu in pair.split(",")}
    assert m.TRAIN_COMMIT == "c68e12a557bef0b197841a0a8ecb9a308c4b83cf"


def test_three_smokes_use_seed_four_not_an_existing_seed(tmp_path):
    m = module()
    jobs = m.make_jobs(templates(), tmp_path, "smoke")
    assert len(jobs) == 3
    for job in jobs:
        config = json.loads(Path(job["config"]).read_text())
        assert config["seed"] == [4]
        assert config["max_tasks"] == 2
        assert config["epochs"] == config["init_epoch"] == 1
        assert config["sa_functional_student_scope"] == "historical"


def test_forgetting_uses_numeric_class_identity_not_string_sorted_matrix():
    m = module()
    groups = [
        {"00-19": 90.0},
        {"00-19": 85.0, "20-39": 80.0},
        {"00-19": 86.0, "20-39": 75.0, "100-119": 70.0},
    ]
    assert m.corrected_forgetting(groups) == pytest.approx(4.5)


def test_partial_curve_is_not_a_complete_experiment():
    m = module()
    text = "CNN: {'old': 0, 'new': 90, '00-09': 90}\nCNN top1 curve: [90]\n"
    parsed = m.parse_log(text, 10)
    assert not parsed["complete"]
    assert parsed["curve"] == [90]


def test_legacy_dated_end_marker_can_be_verified():
    m = module()
    text = ("CNN: {'old': 0, 'new': 90, '00-09': 90}\n"
            "CNN top1 curve: [np.float64(90)]\nAccuracy Matrix (CNN):\n"
            "Forgetting (CNN): 0\n===== 2026-09-15 14:14:14 END old GPUs=0,1 status=0 =====\n")
    assert m.parse_log(text, 1)["complete"]


def test_thirteen_seed_summary_requires_every_seed_without_selection():
    m = module()
    records = [{"dataset": ds, "seed": seed, "complete": True,
                "final": float(seed), "aaa": float(seed), "forgetting": 1.0}
               for ds in ("c100", "inr", "cub") for seed in range(1, 14)]
    summary = m.summarize(records)
    assert summary["c100"]["final"]["mean"] == 7.0
    assert summary["c100"]["n"] == 13
    assert m.summarize(records[:-1])["cub"] is None
    assert m.summarize(records + [records[0]])["c100"] is None
