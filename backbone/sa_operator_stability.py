"""Parameter-only stability terms for continuously trained Shared-A LoRA.

The historical Shared-A LoRA branch is exactly the product of a weighted sum
of normalized old B matrices and the normalized shared A matrix.  These
helpers preserve that effective linear operator without retaining old data.
"""

from collections.abc import Sequence

import torch
from torch import Tensor


def normalize_lora_weight(weight: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize a LoRA matrix using the same global norm as Shared-A LoRA."""
    return weight / torch.linalg.vector_norm(weight).clamp_min(eps)


def aggregate_normalized_up_projections(
    up_weights: Sequence[Tensor],
    scales: Sequence[Tensor],
    *,
    device=None,
    dtype=None,
    eps: float = 1e-8,
) -> Tensor:
    """Return ``sum_i scale_i * normalize(B_i)`` for one Q or V branch."""
    if len(up_weights) == 0:
        raise ValueError("at least one historical up projection is required")
    if len(up_weights) != len(scales):
        raise ValueError("up projection and scale counts must match")

    if device is None:
        device = up_weights[0].device
    if dtype is None:
        dtype = up_weights[0].dtype

    total = torch.zeros_like(up_weights[0], device=device, dtype=dtype)
    for up_weight, scale in zip(up_weights, scales):
        if scale.numel() != 1:
            raise ValueError("each historical scale must be scalar")
        normalized_up = normalize_lora_weight(
            up_weight.to(device=device, dtype=dtype), eps=eps
        )
        total = total + scale.to(device=device, dtype=dtype).reshape(()) * normalized_up
    return total


def relative_effective_operator_drift(
    current_down: Tensor,
    reference_down: Tensor,
    current_up: Tensor,
    reference_up: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Relative squared drift of the effective historical LoRA operator.

    ``current_up`` may depend on live historical scales, while the reference
    tensors are detached snapshots from the start of the current task.
    """
    current_down = normalize_lora_weight(current_down, eps=eps)
    reference_down = normalize_lora_weight(
        reference_down.to(device=current_down.device, dtype=current_down.dtype),
        eps=eps,
    )
    current_up = current_up.to(device=current_down.device, dtype=current_down.dtype)
    reference_up = reference_up.to(
        device=current_down.device, dtype=current_down.dtype
    )

    current_operator = current_up @ current_down
    reference_operator = reference_up @ reference_down
    numerator = (current_operator - reference_operator).square().sum()
    denominator = reference_operator.square().sum().clamp_min(eps)
    return numerator / denominator
