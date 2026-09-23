"""Deterministic holdout and candidate selection for SBGC plasticity guard."""

import math

import numpy as np


def stratified_holdout_indices(labels, fraction, seed):
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("holdout labels must be a non-empty vector")
    if not math.isfinite(float(fraction)) or not 0.0 < fraction < 0.5:
        raise ValueError("holdout fraction must be in (0, 0.5)")
    rng = np.random.default_rng(int(seed))
    training, validation = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if len(indices) < 2:
            raise ValueError("each class needs at least two training samples")
        shuffled = rng.permutation(indices)
        count = max(1, min(len(indices) - 1, round(len(indices) * fraction)))
        validation.extend(shuffled[:count].tolist())
        training.extend(shuffled[count:].tolist())
    return sorted(training), sorted(validation)


def choose_guard_candidate(candidates, additive_loss, relative_tolerance):
    """Select minimum measured risk under a current-task CE constraint."""
    if not math.isfinite(float(additive_loss)) or additive_loss <= 0:
        raise ValueError("additive loss must be positive and finite")
    if not math.isfinite(float(relative_tolerance)) or relative_tolerance < 0:
        raise ValueError("relative tolerance must be non-negative and finite")
    threshold = additive_loss * (1.0 + relative_tolerance)
    feasible = []
    for name, item in candidates.items():
        loss = float(item["loss"])
        risk = float(item["risk"])
        if not math.isfinite(loss) or not math.isfinite(risk) or risk < 0:
            raise ValueError("candidate loss/risk must be finite and risk nonnegative")
        if loss <= threshold + 1e-12:
            feasible.append((risk, name))
    if "additive" not in candidates or not any(
        name == "additive" for _, name in feasible
    ):
        raise RuntimeError("the unconstrained additive fallback must be feasible")
    _, name = min(feasible)
    return name, threshold
