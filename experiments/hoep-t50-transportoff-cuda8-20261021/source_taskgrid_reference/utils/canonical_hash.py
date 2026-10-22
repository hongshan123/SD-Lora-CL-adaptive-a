"""Canonical, tensor-level hashing for P0 trajectory audits.

The old audit hashed raw ``torch.save`` file bytes, which could not locate the
first changed tensor.  These helpers hash each tensor by key, shape, dtype and
its contiguous CPU raw data, and can report the first mismatch with a concrete
``max_abs_diff``.
"""

import hashlib

import torch


def canonical_tensor_hash(key, tensor):
    """Return a deterministic sha256 for one tensor.

    The key is part of the hash, so the same bytes under two different names
    produce different digests.  The tensor is detached, moved to CPU and made
    contiguous before its raw data is included.
    """
    if not torch.is_tensor(tensor):
        raise TypeError("canonical_tensor_hash expects a torch.Tensor")
    t = tensor.detach().cpu().contiguous()
    try:
        raw = t.numpy().tobytes()
    except (TypeError, ValueError):
        # Some dtypes (e.g. bfloat16) are not directly convertible to numpy;
        # view them as bytes so the raw bit pattern is still hashed.
        raw = t.view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(key.encode("utf-8"))
    digest.update(str(tuple(t.shape)).encode("utf-8"))
    digest.update(str(t.dtype).encode("utf-8"))
    digest.update(raw)
    return digest.hexdigest()


def model_tensor_map(model, include_eval_scalars=True):
    """Collect named parameters, buffers and P0 eval scalars into one dict.

    Eval-only scalars (dual lambda/tau) are excluded when
    ``include_eval_scalars=False`` so the training-state hash is comparable
    between control and Dual-B runs.
    """
    tensors = {}
    for name, tensor in model.named_parameters():
        tensors["param:" + name] = tensor.detach().cpu().clone()
    for name, tensor in model.named_buffers():
        tensors["buffer:" + name] = tensor.detach().cpu().clone()
    if include_eval_scalars:
        for name in ("dual_lambda", "tau_fc", "tau_proto"):
            if hasattr(model, name):
                tensors["scalar:" + name] = torch.tensor(
                    float(getattr(model, name)), dtype=torch.float64
                )
    return tensors


def hash_named_tensors(tensors):
    """Hash an ordered-by-key map of tensors into one sha256 digest."""
    digest = hashlib.sha256()
    for key in sorted(tensors):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(canonical_tensor_hash(key, tensors[key]).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def compare_named_tensors(left, right):
    """Return the first mismatch between two tensor maps.

    Returns ``None`` when every key/value matches exactly.  Otherwise returns
    ``{"key", "shape", "dtype", "max_abs_diff", "reason"}`` so logs never have
    to settle for a bare "hash mismatch".
    """
    for key in sorted(set(left) | set(right)):
        if key not in left:
            return {
                "key": key,
                "shape": None,
                "dtype": None,
                "max_abs_diff": None,
                "reason": "missing_in_left",
            }
        if key not in right:
            return {
                "key": key,
                "shape": None,
                "dtype": None,
                "max_abs_diff": None,
                "reason": "missing_in_right",
            }
        left_tensor = left[key]
        right_tensor = right[key]
        if left_tensor.shape != right_tensor.shape:
            return {
                "key": key,
                "shape": (tuple(left_tensor.shape), tuple(right_tensor.shape)),
                "dtype": (str(left_tensor.dtype), str(right_tensor.dtype)),
                "max_abs_diff": None,
                "reason": "shape_mismatch",
            }
        if left_tensor.dtype != right_tensor.dtype:
            return {
                "key": key,
                "shape": tuple(left_tensor.shape),
                "dtype": (str(left_tensor.dtype), str(right_tensor.dtype)),
                "max_abs_diff": None,
                "reason": "dtype_mismatch",
            }
        if left_tensor.numel() == 0:
            continue
        max_diff = float(
            (left_tensor.detach() - right_tensor.detach()).abs().max()
        )
        if max_diff != 0.0:
            return {
                "key": key,
                "shape": tuple(left_tensor.shape),
                "dtype": str(left_tensor.dtype),
                "max_abs_diff": max_diff,
                "reason": "value_mismatch",
            }
    return None
