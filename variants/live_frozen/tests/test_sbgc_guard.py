import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.sbgc_guard import choose_guard_candidate, stratified_holdout_indices


def test_stratified_holdout_is_disjoint_reproducible_and_class_balanced():
    labels = np.repeat(np.arange(3), 10)
    train, validation = stratified_holdout_indices(labels, 0.1, 1993)
    assert train == sorted(train) and validation == sorted(validation)
    assert set(train).isdisjoint(validation)
    assert sorted(train + validation) == list(range(30))
    assert [sum(labels[validation] == class_id) for class_id in range(3)] == [1] * 3
    assert (train, validation) == stratified_holdout_indices(labels, 0.1, 1993)
    assert validation != stratified_holdout_indices(labels, 0.1, 1995)[1]


def test_guard_selects_lowest_risk_feasible_candidate():
    candidates = {
        "0.05": {"loss": 1.05, "risk": 0.05},
        "0.10": {"loss": 1.008, "risk": 0.10},
        "additive": {"loss": 1.0, "risk": 0.55},
    }
    assert choose_guard_candidate(candidates, 1.0, 0.01) == ("0.10", 1.01)
    assert choose_guard_candidate(candidates, 1.0, 0.0)[0] == "additive"


def test_guard_rejects_missing_fallback_and_invalid_metrics():
    with pytest.raises(RuntimeError, match="fallback"):
        choose_guard_candidate({"0.05": {"loss": 1.0, "risk": 0.05}}, 1.0, 0)
    with pytest.raises(ValueError, match="finite"):
        choose_guard_candidate(
            {"additive": {"loss": float("nan"), "risk": 0.0}}, 1.0, 0
        )
