import pytest

from scripts.summarize_margin_study import summarize_calibration, summarize_proxy


def proxy_record(task, old_scores, kl_scores):
    return {
        "dataset": "cub", "task": task,
        "candidate_rows": [
            {"diagnostic_old_top1": accuracy, "proxy_current_old_kl": kl}
            for accuracy, kl in zip(old_scores, kl_scores)
        ],
    }


def test_bias_prescreen_is_distinct_from_joint_gate():
    record = {
        "dataset": "cub", "bias_final": 79.0, "baseline_final": 84.0,
        "bias_aaa": 87.0, "baseline_aaa": 89.0,
    }
    summary = summarize_calibration(record)
    assert summary["bias_only_prescreen"] == "fail"
    assert "not the joint" in summary["note"]


def test_proxy_requires_informative_tasks_and_agreement():
    flat = proxy_record(1, [80.0, 80.0, 80.0], [0.0, 0.1, 0.2])
    assert summarize_proxy([flat])["status"] == "insufficient_evidence"
    first = proxy_record(1, [80.0, 81.0, 82.0], [0.2, 0.1, 0.0])
    second = proxy_record(2, [70.0, 71.0, 72.0], [0.3, 0.2, 0.1])
    summary = summarize_proxy([first, second])
    assert summary["status"] == "pass"
    assert summary["agreement"] == 1.0
    assert summary["mean_inverse_kl_spearman"] == 1.0
    reversed_proxy = proxy_record(2, [70.0, 71.0, 72.0], [0.1, 0.2, 0.3])
    assert summarize_proxy([first, reversed_proxy])["status"] == "fail"


def test_proxy_rejects_duplicate_task():
    record = proxy_record(1, [80.0, 81.0], [0.2, 0.1])
    with pytest.raises(ValueError, match="duplicate"):
        summarize_proxy([record, record])
