"""RNG-neutral helpers for evaluation/calibration code.

The helpers snapshot and restore Python `random`, NumPy, Torch CPU and the
*current* CUDA device RNG states, and build DataLoaders with a fixed-seed
`torch.Generator` so extra eval/calibration traversals do not perturb the
training trajectory.  Each DDP rank only touches its own CUDA device's RNG
context.
"""

import hashlib
import random

import numpy as np
import torch
from torch.utils.data import DataLoader


def snapshot_rng_state():
    cuda_state = (
        torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    )
    return (
        random.getstate(),
        np.random.get_state(),
        torch.random.get_rng_state(),
        cuda_state,
    )


def restore_rng_state(state):
    py_state, np_state, torch_state, cuda_state = state
    random.setstate(py_state)
    np.random.set_state(np_state)
    torch.random.set_rng_state(torch_state)
    if torch.cuda.is_available() and cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state)


class rng_preserving:
    """Context manager that restores all RNG states on exit."""

    def __enter__(self):
        self._state = snapshot_rng_state()
        return self._state

    def __exit__(self, exc_type, exc_value, traceback):
        restore_rng_state(self._state)
        return False


def deterministic_loader(
    dataset,
    batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    seed=0,
):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )


def max_abs_diff(tensors_a, tensors_b):
    """Max absolute elementwise difference between two parameter lists."""
    if len(tensors_a) != len(tensors_b):
        raise ValueError("parameter lists have different lengths")
    max_diff = 0.0
    for tensor_a, tensor_b in zip(tensors_a, tensors_b):
        max_diff = max(
            max_diff,
            float((tensor_a.detach() - tensor_b.detach()).abs().max()),
        )
    return max_diff


def _stable_bytes(value):
    if torch.is_tensor(value):
        return value.detach().cpu().contiguous().numpy().tobytes()
    if isinstance(value, np.ndarray):
        return value.tobytes()
    if isinstance(value, tuple):
        out = b"("
        for item in value:
            out += _stable_bytes(item) + b","
        return out + b")"
    return repr(value).encode("utf-8")


def rng_state_hash():
    """Canonical hash of Python, NumPy, Torch CPU and current CUDA RNG state."""
    state = snapshot_rng_state()
    digest = hashlib.sha256()
    for index, part in enumerate(state):
        digest.update(str(index).encode("utf-8"))
        if part is None:
            digest.update(b"none")
            continue
        digest.update(_stable_bytes(part))
        digest.update(b"\0")
    return digest.hexdigest()
