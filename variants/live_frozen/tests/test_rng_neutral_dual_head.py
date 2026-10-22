import random

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from models.sa_sdlora import dual_head_logits
from utils.rng_utils import (
    deterministic_loader,
    max_abs_diff,
    rng_preserving,
    snapshot_rng_state,
)


def test_rng_preserving_restores_all_states():
    before = snapshot_rng_state()
    with rng_preserving():
        random.seed(123)
        np.random.seed(456)
        torch.manual_seed(789)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(321)
    after = snapshot_rng_state()
    for state_before, state_after in zip(before, after):
        if isinstance(state_before, list):
            assert len(state_before) == len(state_after)
            for left, right in zip(state_before, state_after):
                assert torch.equal(left, right)
        elif isinstance(state_before, tuple) and state_before and isinstance(
            state_before[0], str
        ):
            # NumPy MT19937 state: compare raw bytes/ints.
            assert np.array_equal(state_before[1], state_after[1])
            assert state_before[2:] == state_after[2:]
        elif torch.is_tensor(state_before):
            assert torch.equal(state_before, state_after)
        else:
            assert state_before == state_after


def test_deterministic_loader_same_seed_same_order():
    dataset = TensorDataset(torch.arange(32), torch.arange(32))
    loader_a = deterministic_loader(dataset, batch_size=8, seed=7)
    loader_b = deterministic_loader(dataset, batch_size=8, seed=7)
    batches_a = [batch[0].tolist() for batch in loader_a]
    batches_b = [batch[0].tolist() for batch in loader_b]
    assert batches_a == batches_b

    loader_c = deterministic_loader(
        dataset, batch_size=8, shuffle=True, seed=8
    )
    loader_d = deterministic_loader(
        dataset, batch_size=8, shuffle=True, seed=8
    )
    loader_e = deterministic_loader(
        dataset, batch_size=8, shuffle=True, seed=9
    )
    batches_c = [batch[0].tolist() for batch in loader_c]
    batches_d = [batch[0].tolist() for batch in loader_d]
    batches_e = [batch[0].tolist() for batch in loader_e]
    assert batches_c == batches_d
    assert batches_c != batches_e


class _FakeFc:
    def __call__(self, features):
        return {"logits": features}


class _FakeProtoHead:
    def __call__(self, features):
        return {"logits": 2.0 * features}


class _FakeNetwork:
    tau_fc = 2.0
    tau_proto = 4.0
    dual_lambda = 0.25

    def __init__(self):
        self.fc = _FakeFc()
        self.prototype_head = _FakeProtoHead()


def test_dual_head_logits_formula_and_single_pass():
    network = _FakeNetwork()
    features = torch.randn(3, 5)
    fc, proto, fused = dual_head_logits(network, features)
    expected_fc = features / 2.0
    expected_proto = (2.0 * features) / 4.0
    assert torch.allclose(fc, expected_fc)
    assert torch.allclose(proto, expected_proto)
    assert torch.allclose(
        fused,
        (1.0 - network.dual_lambda) * expected_fc
        + network.dual_lambda * expected_proto,
    )


def test_max_abs_diff_identical_and_different():
    tensors_a = [torch.randn(4, 4)]
    tensors_b = [tensors_a[0].clone()]
    assert max_abs_diff(tensors_a, tensors_b) == 0.0
    tensors_c = [tensors_a[0] + 1e-3]
    assert max_abs_diff(tensors_a, tensors_c) > 0.0
