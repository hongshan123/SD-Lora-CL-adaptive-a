"""Shared-A SD-LoRA backbone.

Each task trains only the up-projection B (plus a scalar scale) against a
task-invariant down-projection A.  Because A is shared, the final model can
store a single merged B per layer:

    B* = sum_{i<t} (s_i / (||A|| * ||B_i||)) * B_i  +  s_t * B_t

which reproduces the exact forward of the per-task bank at evaluation time.
"""

import math
import copy
import os
from collections.abc import Sequence

import timm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer as timm_ViT
from torch import Tensor

from backbone.coordinate_stability import align_live_a_aggregate
from backbone.linears import SimpleLinear
from backbone.lora import ParameterWrapper
from backbone.sa_operator_stability import (
    aggregate_normalized_up_projections,
    relative_effective_operator_drift,
)


SA_STATE_VERSION = 2
SA_STATE_VERSION_LEGACY = 1
SA_STATE_VERSION_UNION = 3
SA_STATE_VERSION_LIVE_A = 4
SA_MERGE_MODE_GAUGE = "gauge"
SA_MERGE_MODE_UNION_SVD = "union_svd"
SA_MERGE_MODE_LIVE_A_AGGREGATE_B = "live_a_aggregate_b"
SA_ABSORB_MODE_NORMALIZED = "normalized_absorb"
SA_ABSORB_MODE_OPERATOR_PRESERVING = "operator_preserving_absorb"
SA_ABSORB_MODE_BOUNDED_NORM_CALIBRATED = (
    "bounded_norm_calibrated_absorb"
)
SA_STATE_FILENAME = "sa_state.pt"
SA_MERGED_FILENAME = "sa_merged_lora.pt"


def hbd_historical_branch_distance(
    student_outputs, teacher_outputs
):
    """Normalized cosine distance between historical-branch responses.

    ``student_outputs`` / ``teacher_outputs`` are lists of ``(q, v)`` tensors
    shaped ``[B, T, D]`` (one entry per LoRA block, in block order).  For every
    Q/V branch the tokens are L2-normalized and the mean cosine similarity is
    taken; the loss is ``1 - mean`` averaged over branches and blocks (the
    guide's preferred token/branch-averaged normalized cosine distance).

    Gradients flow only through ``student_outputs``; teacher tensors should be
    detached (the training hook runs the teacher under ``torch.no_grad()``).
    """
    if len(student_outputs) != len(teacher_outputs):
        raise ValueError(
            "student/teacher historical output counts differ: {} vs {}".format(
                len(student_outputs), len(teacher_outputs)
            )
        )
    branch_losses = []
    for (sq, sv), (tq, tv) in zip(student_outputs, teacher_outputs):
        for s, t in ((sq, tq), (sv, tv)):
            s_flat = s.reshape(-1, s.shape[-1])
            t_flat = t.reshape(-1, t.shape[-1])
            s_norm = F.normalize(s_flat, p=2, dim=1)
            t_norm = F.normalize(t_flat, p=2, dim=1)
            branch_losses.append(
                1.0 - (s_norm * t_norm).sum(dim=1).mean()
            )
    if not branch_losses:
        raise ValueError("HBD requires at least one historical branch output")
    return torch.stack(branch_losses).mean()


def register_live_a_historical_capture_hooks(model, capture_list):
    """Register forward hooks capturing per-block historical-branch outputs.

    Each hook appends ``(q, v)`` for the block's ``_LiveAAggregateQKV``
    wrapper using that block's actual input tokens.  Returns the hook handles;
    callers must remove them after the forward.
    """
    handles = []
    for blk in model.lora_vit.blocks:
        wrapper = blk.attn.qkv
        if not isinstance(wrapper, _LiveAAggregateQKV):
            continue

        def _make_hook(capture_list):
            def _hook(module, args, output):
                hq, hv = module.historical_output(args[0])
                capture_list.append((hq, hv))

            return _hook

        handles.append(
            wrapper.register_forward_hook(_make_hook(capture_list))
        )
    if not handles:
        raise ValueError(
            "no _LiveAAggregateQKV wrappers found; HBD requires "
            "sa_cumulative_merge=live_a_aggregate_b"
        )
    return handles


def live_a_historical_outputs(model, x, capture_list):
    """Run a frozen Live-A model forward and return per-block (q, v) outputs.

    The caller must have registered capture hooks on ``model`` and must wrap
    this call in ``torch.no_grad()`` when a teacher is used.
    """
    capture_list.clear()
    model(x)
    return list(capture_list)


def _join_path(prefix, name):
    return os.path.join(prefix, name)


def _fixed_orthogonal_down(in_dim, target_rank, seed, dtype=torch.float32):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    random_matrix = torch.randn(in_dim, target_rank, generator=generator)
    q, _ = torch.linalg.qr(random_matrix, mode="reduced")
    return q.t().contiguous().to(dtype)


def absorb_live_a_current_projection(
    shared_a,
    current_b,
    scale,
    mode=SA_ABSORB_MODE_OPERATOR_PRESERVING,
    eps=1e-8,
):
    if mode not in (
        SA_ABSORB_MODE_NORMALIZED,
        SA_ABSORB_MODE_OPERATOR_PRESERVING,
        SA_ABSORB_MODE_BOUNDED_NORM_CALIBRATED,
    ):
        raise ValueError("unsupported live-a absorption mode: {}".format(mode))
    if shared_a.ndim != 2 or current_b.ndim != 2:
        raise ValueError("shared_a and current_b must be matrices")
    if current_b.shape[1] != shared_a.shape[0]:
        raise ValueError("current_b rank must match shared_a rank")
    if scale.numel() != 1:
        raise ValueError("scale must be scalar")

    a = shared_a.to(device=current_b.device, dtype=current_b.dtype)
    s = scale.to(device=current_b.device, dtype=current_b.dtype).reshape(())
    norm_a = torch.linalg.vector_norm(a) + eps
    norm_b = torch.linalg.vector_norm(current_b) + eps
    a_hat = a / norm_a
    b_hat = current_b / norm_b
    norm_product = norm_a * norm_b
    gamma = s
    if mode == SA_ABSORB_MODE_OPERATOR_PRESERVING:
        consolidation_gain = torch.ones_like(norm_product)
        gamma = s * norm_product
    elif mode == SA_ABSORB_MODE_BOUNDED_NORM_CALIBRATED:
        # Turn the legacy task-boundary normalization into an explicit,
        # bounded consolidation rule: attenuate oversized current operators
        # but never amplify operators whose factor norms are below one.
        consolidation_gain = torch.clamp(
            torch.reciprocal(norm_product), max=1.0
        )
        gamma = s * norm_product * consolidation_gain
    else:
        consolidation_gain = torch.reciprocal(norm_product)
    absorbed_up = gamma * b_hat

    before_operator = s * (current_b @ a)
    after_operator = absorbed_up @ a_hat
    relative_error = torch.linalg.vector_norm(
        after_operator - before_operator
    ) / (torch.linalg.vector_norm(before_operator) + eps)
    return absorbed_up, {
        "norm_A": float(norm_a),
        "norm_B": float(norm_b),
        "scaling": float(s),
        "gamma": float(gamma),
        "norm_product": float(norm_product),
        "consolidation_gain": float(consolidation_gain),
        "absorption_relative_error": float(relative_error),
    }


def fold_cumulative_up_projection(
    shared_a: Tensor,
    up_weights: Sequence[Tensor],
    scales: Sequence[Tensor],
    *,
    device=None,
    dtype=None,
    eps: float = 1e-8,
) -> Tensor:
    """Fold the historical B bank into one cumulative up projection.

    For a fixed down projection ``A``, each historical task contributes
    ``scale_i * B_i(A x) / (||A|| ||B_i||)`` to the Shared-A LoRA forward.
    The exact bank sum is reproduced by a single cumulative up projection

        H = sum_i scale_i * B_i / (||A|| ||B_i||)

    followed by ``H(A x)``.  The returned tensor has the same shape as each
    ``B_i`` (``dim x rank``).
    """
    if len(up_weights) == 0:
        raise ValueError("at least one historical up projection is required")
    if len(up_weights) != len(scales):
        raise ValueError("up projection and scale counts must match")

    if device is None:
        device = up_weights[0].device
    if dtype is None:
        dtype = up_weights[0].dtype

    shared_a = shared_a.to(device=device, dtype=dtype)
    norm_a = torch.linalg.vector_norm(shared_a)
    total = torch.zeros_like(up_weights[0], device=device, dtype=dtype)
    for up_weight, scale in zip(up_weights, scales):
        if scale.numel() != 1:
            raise ValueError("each historical scale must be scalar")
        up = up_weight.to(device=device, dtype=dtype)
        denom = norm_a * torch.linalg.vector_norm(up) + eps
        total = total + scale.to(device=device, dtype=dtype).reshape(()) * up / denom
    return total


def fold_all_cumulative_up_projections(
    shared_a_list: Sequence[Tensor],
    saved_b_tasks: dict[int, Sequence[Tensor]],
    scales: dict[int, Tensor],
    *,
    device=None,
    dtype=None,
    eps: float = 1e-8,
) -> list[Tensor]:
    """Fold every Q/V branch across all historical tasks."""
    task_ids = sorted(saved_b_tasks)
    if not task_ids:
        raise ValueError("at least one historical task is required")
    branch_count = len(shared_a_list)
    if branch_count == 0:
        raise ValueError("at least one shared down projection is required")
    for task_id in task_ids:
        if len(saved_b_tasks[task_id]) != branch_count:
            raise ValueError(
                "task {} has {} branches; expected {}".format(
                    task_id, len(saved_b_tasks[task_id]), branch_count
                )
            )
        if task_id not in scales:
            raise ValueError("missing scale for task {}".format(task_id))
    cumulative = []
    for branch in range(branch_count):
        cumulative.append(
            fold_cumulative_up_projection(
                shared_a_list[branch],
                [saved_b_tasks[task_id][branch] for task_id in task_ids],
                [scales[task_id] for task_id in task_ids],
                device=device,
                dtype=dtype,
                eps=eps,
            )
        )
    return cumulative


def canonical_down_projection(shared_a: Tensor) -> tuple[Tensor, Tensor]:
    """Return the canonical down projection and the triangular factor of ``A``.

    With the thin QR decomposition ``A^T = Q R`` we have
    ``A = R^T Q^T`` and ``B A = (B R^T) Q^T``.  ``Q^T`` is the canonical
    down projection (``rank x dim``) and ``R`` is the ``rank x rank``
    triangular factor that is absorbed into the up projection.
    """
    a_t = shared_a.to(dtype=torch.float64).t().contiguous()
    q, r = torch.linalg.qr(a_t, mode="reduced")
    return q.t().to(shared_a.dtype), r.to(shared_a.dtype)


def low_rank_product_frobenius_norm(
    up: Tensor, down: Tensor, eps: float = 1e-8
) -> Tensor:
    """Return ``||up @ down||_F`` without forming the dense product."""
    if up.ndim != 2 or down.ndim != 2:
        raise ValueError("up and down must be matrices")
    if up.shape[1] != down.shape[0]:
        raise ValueError("up and down inner dimensions must match")
    if eps <= 0:
        raise ValueError("eps must be positive")
    up_gram = up.t() @ up
    down_gram = down @ down.t()
    squared_norm = torch.sum(up_gram * down_gram)
    return torch.sqrt(torch.clamp_min(squared_norm, 0.0))


ADAPTIVE_A_MODES = ("frozen", "tangent", "live")


def decompose_adaptive_a_gradient(
    gradient: Tensor, shared_a: Tensor
) -> dict[str, Tensor]:
    """Return the exact Frozen/Tangent/Live shared-A candidates."""
    if gradient.shape != shared_a.shape:
        raise ValueError("gradient and shared_a must have the same shape")
    q_t, _ = canonical_down_projection(shared_a.detach())
    q_t = q_t.to(device=gradient.device, dtype=gradient.dtype)
    tangent = (gradient @ q_t.t()) @ q_t
    perpendicular = gradient - tangent
    return {
        "frozen": torch.zeros_like(gradient),
        "tangent": tangent,
        "live": gradient,
        "perpendicular": perpendicular,
    }


def signed_gradient_utility(
    candidate: Tensor, control_gradient: Tensor, eps: float = 1e-8
) -> Tensor:
    """Cosine agreement between a candidate update and held-out gradient."""
    if candidate.shape != control_gradient.shape:
        raise ValueError("candidate and control_gradient must have the same shape")
    if eps <= 0:
        raise ValueError("eps must be positive")
    numerator = torch.sum(candidate * control_gradient)
    denominator = (
        torch.linalg.vector_norm(candidate)
        * torch.linalg.vector_norm(control_gradient)
    )
    if bool(denominator.detach() <= eps):
        return numerator.new_zeros(())
    return numerator / (denominator + eps)


def weighted_operator_risk(
    historical_up: Tensor,
    shared_a: Tensor,
    delta_a: Tensor,
    input_rms: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Exact one-step normalized historical-operator drift under a sketch."""
    if shared_a.shape != delta_a.shape:
        raise ValueError("shared_a and delta_a must have the same shape")
    if historical_up.shape[1] != shared_a.shape[0]:
        raise ValueError("historical_up and shared_a ranks must match")
    if input_rms.ndim != 1 or input_rms.shape[0] != shared_a.shape[1]:
        raise ValueError("input_rms must match the shared-A input dimension")
    if eps <= 0:
        raise ValueError("eps must be positive")
    dtype = shared_a.dtype
    device = shared_a.device
    historical_up = historical_up.to(device=device, dtype=dtype)
    input_rms = input_rms.to(device=device, dtype=dtype)
    a_hat = shared_a / (torch.linalg.vector_norm(shared_a) + eps)
    updated_a = shared_a - delta_a
    updated_a_hat = updated_a / (torch.linalg.vector_norm(updated_a) + eps)
    weighted_delta = (updated_a_hat - a_hat) * input_rms.unsqueeze(0)
    weighted_reference = a_hat * input_rms.unsqueeze(0)
    drift = low_rank_product_frobenius_norm(historical_up, weighted_delta, eps)
    reference = low_rank_product_frobenius_norm(
        historical_up, weighted_reference, eps
    )
    return drift.square() / (reference.square() + eps)


def effective_risk_budget(
    layer_candidates: list[dict[str, dict[str, float]]],
    configured_budget: float,
    risk_budget_mode: str = "absolute",
) -> float:
    """Resolve an absolute budget or a fraction of the all-live risk."""
    if configured_budget < 0:
        raise ValueError("configured_budget must be non-negative")
    if risk_budget_mode not in ("absolute", "relative"):
        raise ValueError(
            "risk_budget_mode must be absolute or relative"
        )
    if risk_budget_mode == "absolute":
        return float(configured_budget)
    live_risk = sum(
        float(layer["live"]["risk"]) for layer in layer_candidates
    )
    return float(configured_budget) * live_risk


def choose_risk_budgeted_modes(
    layer_candidates: list[dict[str, dict[str, float]]],
    risk_budget: float,
    eps: float = 1e-12,
) -> dict:
    """Solve the small multiple-choice risk budget via Pareto pruning."""
    if risk_budget < 0:
        raise ValueError("risk_budget must be non-negative")
    frontier = [(0.0, 0.0, ())]
    for layer in layer_candidates:
        missing = set(ADAPTIVE_A_MODES) - set(layer)
        if missing:
            raise ValueError("missing Adaptive-A modes: {}".format(sorted(missing)))
        expanded = []
        for total_risk, total_utility, modes in frontier:
            for mode in ADAPTIVE_A_MODES:
                risk = float(layer[mode]["risk"])
                utility = float(layer[mode]["utility"])
                if risk < 0:
                    raise ValueError("candidate risk must be non-negative")
                next_risk = total_risk + risk
                if next_risk <= risk_budget + eps:
                    expanded.append(
                        (next_risk, total_utility + utility, modes + (mode,))
                    )
        expanded.sort(key=lambda item: (item[0], -item[1], item[2]))
        frontier = []
        best_utility = float("-inf")
        for state in expanded:
            if state[1] > best_utility + eps:
                frontier.append(state)
                best_utility = state[1]
        if not frontier:
            raise RuntimeError("no Adaptive-A candidate satisfies the risk budget")
    selected = max(frontier, key=lambda item: (item[1], -item[0], item[2]))
    return {
        "modes": list(selected[2]),
        "selected_risk": selected[0],
        "selected_utility": selected[1],
    }


def adaptive_a_layer_gradient(
    *,
    gradient_q: Tensor,
    gradient_v: Tensor,
    shared_a_q: Tensor,
    shared_a_v: Tensor,
    current_up_q: Tensor,
    current_up_v: Tensor,
    historical_up_q: Tensor | None,
    historical_up_v: Tensor | None,
    scale: Tensor,
    stability_weight: float,
    gate_floor: float,
    momentum: float,
    eps: float = 1e-8,
    previous_gate: Tensor | float | None = None,
) -> dict[str, Tensor]:
    """Gate one Q/V layer's coordinate-changing shared-A gradients."""
    if not 0.0 <= gate_floor <= 1.0:
        raise ValueError("gate_floor must be in [0, 1]")
    if not 0.0 <= momentum < 1.0:
        raise ValueError("momentum must be in [0, 1)")
    if stability_weight < 0:
        raise ValueError("stability_weight must be non-negative")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if scale.numel() != 1:
        raise ValueError("scale must be scalar")
    if (historical_up_q is None) != (historical_up_v is None):
        raise ValueError("historical Q and V projections must be supplied together")

    def _decompose(gradient, shared_a):
        q_t, _ = canonical_down_projection(shared_a.detach())
        q_t = q_t.to(device=gradient.device, dtype=gradient.dtype)
        parallel = (gradient @ q_t.t()) @ q_t
        return parallel, gradient - parallel

    parallel_q, perpendicular_q = _decompose(gradient_q, shared_a_q)
    parallel_v, perpendicular_v = _decompose(gradient_v, shared_a_v)
    zero_perpendicular = bool(
        torch.linalg.vector_norm(perpendicular_q).detach() <= eps
        and torch.linalg.vector_norm(perpendicular_v).detach() <= eps
    )
    has_history = historical_up_q is not None
    one = gradient_q.new_ones(())
    if not has_history:
        return {
            "gradient_q": gradient_q,
            "gradient_v": gradient_v,
            "parallel_q_gradient": parallel_q,
            "parallel_v_gradient": parallel_v,
            "perpendicular_q_gradient": perpendicular_q,
            "perpendicular_v_gradient": perpendicular_v,
            "current_impact": gradient_q.new_zeros(()),
            "historical_impact": gradient_q.new_zeros(()),
            "raw_gate": one,
            "gate": one,
            "perpendicular_retention": one,
        }
    if zero_perpendicular:
        raw_gate = one
        if previous_gate is None:
            gate = raw_gate
        else:
            previous_gate = torch.as_tensor(
                previous_gate, device=raw_gate.device, dtype=raw_gate.dtype
            )
            gate = momentum * previous_gate + (
                1.0 - momentum
            ) * raw_gate
        return {
            "gradient_q": gradient_q,
            "gradient_v": gradient_v,
            "parallel_q_gradient": parallel_q,
            "parallel_v_gradient": parallel_v,
            "perpendicular_q_gradient": perpendicular_q,
            "perpendicular_v_gradient": perpendicular_v,
            "current_impact": gradient_q.new_zeros(()),
            "historical_impact": gradient_q.new_zeros(()),
            "raw_gate": raw_gate,
            "gate": gate,
            "perpendicular_retention": gate,
        }

    scalar_scale = scale.detach().to(
        device=gradient_q.device, dtype=gradient_q.dtype
    ).reshape(())
    current_q = scalar_scale * current_up_q.to(
        device=gradient_q.device, dtype=gradient_q.dtype
    )
    current_v = scalar_scale * current_up_v.to(
        device=gradient_v.device, dtype=gradient_v.dtype
    )
    historical_q = historical_up_q.to(
        device=gradient_q.device, dtype=gradient_q.dtype
    ) / (torch.linalg.vector_norm(shared_a_q.detach()) + eps)
    historical_v = historical_up_v.to(
        device=gradient_v.device, dtype=gradient_v.dtype
    ) / (torch.linalg.vector_norm(shared_a_v.detach()) + eps)
    current_impact = torch.sqrt(
        low_rank_product_frobenius_norm(current_q, perpendicular_q, eps).square()
        + low_rank_product_frobenius_norm(current_v, perpendicular_v, eps).square()
    )
    historical_impact = torch.sqrt(
        low_rank_product_frobenius_norm(
            historical_q, perpendicular_q, eps
        ).square()
        + low_rank_product_frobenius_norm(
            historical_v, perpendicular_v, eps
        ).square()
    )
    raw_gate = torch.clamp(
        current_impact
        / (current_impact + stability_weight * historical_impact + eps),
        min=gate_floor,
        max=1.0,
    )
    if previous_gate is None:
        gate = raw_gate
    else:
        previous_gate = torch.as_tensor(
            previous_gate, device=raw_gate.device, dtype=raw_gate.dtype
        )
        gate = momentum * previous_gate + (
            1.0 - momentum
        ) * raw_gate
    return {
        "gradient_q": parallel_q + gate * perpendicular_q,
        "gradient_v": parallel_v + gate * perpendicular_v,
        "parallel_q_gradient": parallel_q,
        "parallel_v_gradient": parallel_v,
        "perpendicular_q_gradient": perpendicular_q,
        "perpendicular_v_gradient": perpendicular_v,
        "current_impact": current_impact,
        "historical_impact": historical_impact,
        "raw_gate": raw_gate,
        "gate": gate,
        "perpendicular_retention": gate,
    }


def canonicalize_effective_up_projection(
    up_raw: Tensor, triangular_r: Tensor
) -> Tensor:
    """Convert a raw cumulative up projection into canonical coordinates.

    If ``A = R^T Q^T``, the effective operator ``H_raw @ A`` equals
    ``(H_raw @ R^T) Q^T``; the canonical up projection is ``H_raw @ R^T``.
    """
    return up_raw @ triangular_r.t()


def gauge_align_up_projection(
    cumulative_up: Tensor, q_old_t: Tensor, q_new_t: Tensor
) -> Tensor:
    """Closed-form best approximation of the old operator in the new basis.

    ``H_old Q_old^T`` is approximated by ``H_aligned Q_new^T`` with
    ``H_aligned = H_old (Q_old^T Q_new)``, the least-squares solution of
    ``min_H ||H_old Q_old^T - H Q_new^T||_F^2``.
    """
    return cumulative_up @ (q_old_t @ q_new_t.t())


def gauge_projection_residual(
    cumulative_up: Tensor, q_old_t: Tensor, q_new_t: Tensor
) -> Tensor:
    """Frobenius norm of the part of the old operator outside ``span(Q_new)``.

    This equals ``||H_old Q_old^T (I - Q_new Q_new^T)||_F``.
    """
    identity = torch.eye(
        q_new_t.shape[1], device=q_new_t.device, dtype=torch.float64
    )
    projector = q_new_t.double().t() @ q_new_t.double()
    old_operator = cumulative_up.double() @ q_old_t.double()
    residual = old_operator @ (identity - projector)
    return residual.norm().to(cumulative_up.dtype)


def union_svd_factors(
    left_factors: Sequence[Tensor],
    right_factors: Sequence[Tensor],
    rank: int,
    eps: float = 1e-12,
) -> tuple[Tensor, Tensor, Tensor, float]:
    """Fixed-rank optimal approximation of ``M = sum_i L_i R_i^T``.

    Returns ``(Q^T, H, singular_values, relative_truncation_error)`` such that
    ``H Q^T`` is the rank-``rank`` truncated SVD of ``M``.  The dense ``d x d``
    matrix is never materialized: the joint left/right factors are QR-reduced
    first and only the small ``K x K`` core is decomposed.
    """
    if not left_factors:
        raise ValueError("at least one left factor is required")
    if len(left_factors) != len(right_factors):
        raise ValueError("left and right factor counts must match")
    left = torch.cat([t.to(torch.float64) for t in left_factors], dim=1)
    right = torch.cat([t.to(torch.float64) for t in right_factors], dim=1)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("factors must be 2D matrices")
    if left.shape[0] != right.shape[0]:
        raise ValueError(
            "left and right factors must share the output dimension; "
            "got {} vs {}".format(left.shape[0], right.shape[0])
        )
    if left.shape[1] != right.shape[1]:
        raise ValueError(
            "left and right factors must share the joint rank; "
            "got {} vs {}".format(left.shape[1], right.shape[1])
        )
    k = left.shape[1]
    if rank <= 0 or rank > k:
        raise ValueError("rank must satisfy 0 < rank <= {}".format(k))

    q_l, r_l = torch.linalg.qr(left, mode="reduced")
    q_r, r_r = torch.linalg.qr(right, mode="reduced")
    core = r_l @ r_r.t()
    u_c, s, v_c = torch.linalg.svd(core, full_matrices=False)
    u_full = q_l @ u_c
    v_full = q_r @ v_c.t()

    canonical_down = v_full[:, :rank].t().contiguous().float()
    cumulative_up = (u_full[:, :rank] * s[:rank]).contiguous().float()
    total_sq = (s.square().sum()).clamp_min(eps)
    trunc_sq = (s[rank:].square().sum()).clamp_min(0.0)
    relative_error = float((trunc_sq / total_sq).sqrt())
    return canonical_down, cumulative_up, s.float(), relative_error


def migrate_sa_state_v1_to_v2(filepath: str, force: bool = False) -> dict:
    """Explicitly migrate a legacy v1 artifact to the v2 cumulative state.

    The original ``sa_state.pt`` is moved to ``sa_state.pt.v1`` and the
    per-task B files are left untouched as a backup.  The new state contains
    ``canonical_down`` / ``cumulative_up`` / ``triangular_r`` and is the exact
    canonicalization of the v1 bank at its final shared A.
    """
    state_path = _join_path(filepath, SA_STATE_FILENAME)
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if int(state.get("version", -1)) != SA_STATE_VERSION_LEGACY:
        raise ValueError(
            "artifact is not legacy v1 (version={}); nothing to migrate".format(
                state.get("version", -1)
            )
        )
    task_ids = sorted(int(task_id) for task_id in state.get("scales", {}))
    if not task_ids:
        raise ValueError("legacy artifact has no saved tasks")
    shared_a = state["shared_a"]
    b_lists = []
    for task_id in task_ids:
        path = _join_path(filepath, "sa_lora_w_b_{}.pt".format(task_id))
        if not os.path.exists(path):
            raise FileNotFoundError(
                "missing B file for task {}".format(task_id)
            )
        b_lists.append(
            torch.load(path, map_location="cpu", weights_only=True)
        )
    canonical_down = []
    cumulative_up = []
    triangular_r = []
    for branch in range(len(shared_a)):
        h_raw = fold_cumulative_up_projection(
            shared_a[branch],
            [b_list[branch] for b_list in b_lists],
            [state["scales"][task_id] for task_id in task_ids],
        )
        q_t, r = canonical_down_projection(shared_a[branch])
        canonical_down.append(q_t)
        cumulative_up.append(canonicalize_effective_up_projection(h_raw, r))
        triangular_r.append(r)
    new_state = {
        "version": SA_STATE_VERSION,
        "task_id": len(task_ids),
        "rank": int(shared_a[0].shape[0]),
        "canonical_down": canonical_down,
        "cumulative_up": cumulative_up,
        "triangular_r": triangular_r,
        "migrated_from": {
            "version": SA_STATE_VERSION_LEGACY,
            "task_ids": task_ids,
        },
    }
    backup_path = state_path + ".v1"
    if os.path.exists(backup_path):
        if not force:
            raise FileExistsError(
                "backup already exists: {}; pass --force to overwrite".format(
                    backup_path
                )
            )
        os.remove(backup_path)
    os.replace(state_path, backup_path)
    torch.save(new_state, state_path)
    return new_state


def migrate_sa_state_v2_to_v3(
    filepath: str, rank: int | None = None, force: bool = False
) -> dict:
    """Explicitly migrate a v2 gauge state to a v3 union-SVD state.

    The old ``sa_state.pt`` is backed up to ``sa_state.pt.v2``.  Each branch's
    exact historical operator ``H Q^T`` is re-factorized with a fixed-rank
    union SVD at ``rank`` (defaults to the state's stored rank).  Truncating
    below the stored rank is intentionally lossy; this is an explicit,
    non-silent migration.
    """
    state_path = _join_path(filepath, SA_STATE_FILENAME)
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if int(state.get("version", -1)) != SA_STATE_VERSION:
        raise ValueError(
            "artifact is not v2 gauge (version={}); nothing to migrate".format(
                state.get("version", -1)
            )
        )
    old_rank = int(state["rank"])
    target_rank = int(rank) if rank is not None else old_rank
    if target_rank <= 0 or target_rank > old_rank:
        raise ValueError(
            "target rank must satisfy 0 < rank <= {}".format(old_rank)
        )
    canonical_down = []
    cumulative_up = []
    triangular_r = []
    for h, q_t in zip(state["cumulative_up"], state["canonical_down"]):
        q_new, h_new, _, _ = union_svd_factors(
            [h], [q_t.t()], rank=target_rank
        )
        canonical_down.append(q_new)
        cumulative_up.append(h_new)
        triangular_r.append(torch.eye(target_rank, dtype=torch.float32))
    new_state = {
        "version": SA_STATE_VERSION_UNION,
        "task_id": int(state["task_id"]),
        "rank": target_rank,
        "merge_mode": SA_MERGE_MODE_UNION_SVD,
        "canonical_down": canonical_down,
        "cumulative_up": cumulative_up,
        "triangular_r": triangular_r,
        "migrated_from": {
            "version": SA_STATE_VERSION,
            "old_rank": old_rank,
        },
    }
    backup_path = state_path + ".v2"
    if os.path.exists(backup_path):
        if not force:
            raise FileExistsError(
                "backup already exists: {}; pass --force to overwrite".format(
                    backup_path
                )
            )
        os.remove(backup_path)
    os.replace(state_path, backup_path)
    torch.save(new_state, state_path)
    return new_state


def migrate_sa_state_v1_to_v4(
    filepath: str, force: bool = False
) -> dict:
    """Explicitly migrate a v1 EXP-009 artifact to v4 Live-A Aggregate-B.

    Computes ``G = sum_i s_i B_i / ||B_i||`` from the saved B files and the
    final shared A, then writes a v4 ``sa_state.pt``.  The original v1 state
    is backed up to ``sa_state.pt.v1`` and the per-task B files are kept as a
    backup.
    """
    state_path = _join_path(filepath, SA_STATE_FILENAME)
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if int(state.get("version", -1)) != SA_STATE_VERSION_LEGACY:
        raise ValueError(
            "artifact is not legacy v1 (version={}); nothing to migrate".format(
                state.get("version", -1)
            )
        )
    task_ids = sorted(int(task_id) for task_id in state.get("scales", {}))
    if not task_ids:
        raise ValueError("legacy artifact has no saved tasks")
    shared_a = [w.detach().cpu().float() for w in state["shared_a"]]
    branches = len(shared_a)
    aggregate_up = []
    for branch in range(branches):
        total = None
        for task_id in task_ids:
            path = _join_path(
                filepath, "sa_lora_w_b_{}.pt".format(task_id)
            )
            b_list = torch.load(path, map_location="cpu", weights_only=True)
            b = b_list[branch].float()
            s = state["scales"][task_id].reshape(())
            term = s * b / (torch.linalg.vector_norm(b) + 1e-8)
            total = term if total is None else total + term
        aggregate_up.append(total)
    new_state = {
        "version": SA_STATE_VERSION_LIVE_A,
        "task_id": len(task_ids),
        "rank": shared_a[0].shape[0],
        "merge_mode": SA_MERGE_MODE_LIVE_A_AGGREGATE_B,
        "aggregate_up": aggregate_up,
        "shared_a": shared_a,
        "history_groups": 1,
        "coordinate_aligned": False,
        "absorb_mode": SA_ABSORB_MODE_NORMALIZED,
        "migrated_from": {"version": SA_STATE_VERSION_LEGACY},
    }
    backup_path = state_path + ".v1"
    if os.path.exists(backup_path):
        if not force:
            raise FileExistsError(
                "backup already exists: {}; pass --force to overwrite".format(
                    backup_path
                )
            )
        os.remove(backup_path)
    os.replace(state_path, backup_path)
    torch.save(new_state, state_path)
    return new_state


class _SharedAQKV(nn.Module):
    """QKV wrapper: base qkv + shared-A LoRA bank with per-task B matrices."""

    def __init__(
        self,
        qkv,
        a_q,
        a_v,
        b_q,
        b_v,
        saved_b_q,
        saved_b_v,
        scaling_cur,
        scaling_prev,
        layer_index,
    ):
        super().__init__()
        self.qkv = qkv
        self.a_q = a_q
        self.a_v = a_v
        self.b_q = b_q
        self.b_v = b_v
        self.scaling_cur = scaling_cur
        self.scaling_prev = scaling_prev
        self.layer_index = layer_index
        self.dim = qkv.in_features
        self.saved_b_q = list(saved_b_q)
        self.saved_b_v = list(saved_b_v)

    def _norm_lora(self, x, a_weight, b_weight):
        denom = torch.norm(a_weight) * torch.norm(b_weight) + 1e-8
        return F.linear(F.linear(x, a_weight), b_weight) / denom

    def forward(self, x):
        new_q = 0
        new_v = 0
        a_q_w = self.a_q.weight
        a_v_w = self.a_v.weight

        for idx, (b_q_w, b_v_w) in enumerate(
            zip(self.saved_b_q, self.saved_b_v)
        ):
            b_q_w = b_q_w.to(device=x.device, dtype=x.dtype)
            b_v_w = b_v_w.to(device=x.device, dtype=x.dtype)
            scale_idx = min(idx, len(self.scaling_prev) - 1)
            new_q = new_q + self.scaling_prev[scale_idx](
                self._norm_lora(x, a_q_w, b_q_w)
            )
            new_v = new_v + self.scaling_prev[scale_idx](
                self._norm_lora(x, a_v_w, b_v_w)
            )

        new_q = new_q + self.scaling_cur[0](self.b_q(self.a_q(x)))
        new_v = new_v + self.scaling_cur[0](self.b_v(self.a_v(x)))

        qkv = self.qkv(x)
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv


class _CumulativeSharedAQKV(nn.Module):
    """QKV wrapper for the canonical cumulative Shared-A state.

    Historical tasks are represented by the exact effective operator
    ``H @ Q^T`` (fixed while the current task trains).  The current task adds
    its own raw LoRA branch ``scale * B(A x)``, matching the legacy v1
    training-time semantics; the normalization is folded in only when the
    task is saved into the cumulative state.
    """

    def __init__(
        self,
        qkv,
        a_q,
        a_v,
        b_q,
        b_v,
        h_q,
        q_q_t,
        h_v,
        q_v_t,
        scaling_cur,
        layer_index,
    ):
        super().__init__()
        self.qkv = qkv
        self.a_q = a_q
        self.a_v = a_v
        self.b_q = b_q
        self.b_v = b_v
        self.scaling_cur = scaling_cur
        self.layer_index = layer_index
        self.dim = qkv.in_features
        self.register_buffer("h_q", h_q.clone(), persistent=False)
        self.register_buffer("q_q_t", q_q_t.clone(), persistent=False)
        self.register_buffer("h_v", h_v.clone(), persistent=False)
        self.register_buffer("q_v_t", q_v_t.clone(), persistent=False)

    def forward(self, x):
        new_q = F.linear(F.linear(x, self.q_q_t), self.h_q)
        new_v = F.linear(F.linear(x, self.q_v_t), self.h_v)
        new_q = new_q + self.scaling_cur[0](self.b_q(self.a_q(x)))
        new_v = new_v + self.scaling_cur[0](self.b_v(self.a_v(x)))
        qkv = self.qkv(x)
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv


class _LiveAAggregateQKV(nn.Module):
    """QKV wrapper for Live-A Aggregate-B.

    The historical branch is ``G A x / ||A||`` with the *live* shared A (the
    same differentiable A used by the current task), matching EXP-009's bank
    gradient path when historical scales are folded and frozen.  The current
    task branch keeps the legacy raw semantics ``scale * B(Ax)`` (no
    normalization), so non-final evaluation after save still sees
    ``old G + current raw B`` exactly like the v1 bank.
    """

    def __init__(
        self,
        qkv,
        a_q,
        a_v,
        b_q,
        b_v,
        aggregate_q,
        aggregate_v,
        scaling_cur,
        layer_index,
        history_groups=1,
        historical_input_rms=None,
        historical_input_count=0.0,
        capture_input_sketch=False,
    ):
        super().__init__()
        self.qkv = qkv
        self.a_q = a_q
        self.a_v = a_v
        self.b_q = b_q
        self.b_v = b_v
        self.scaling_cur = scaling_cur
        self.layer_index = layer_index
        self.dim = qkv.in_features
        self.history_groups = int(history_groups)
        self.capture_input_sketch = bool(capture_input_sketch)
        if historical_input_rms is None:
            historical_input_rms = torch.ones(self.dim)
        self.register_buffer(
            "historical_input_rms",
            historical_input_rms.detach().clone().float(),
            persistent=False,
        )
        self.historical_input_count = float(historical_input_count)
        self.register_buffer(
            "task_input_square_sum", torch.zeros(self.dim), persistent=False
        )
        self.task_input_count = 0.0
        self.register_buffer(
            "pending_input_square_sum", torch.zeros(self.dim), persistent=False
        )
        self.pending_input_count = 0.0
        self.register_buffer(
            "aggregate_q", aggregate_q.clone(), persistent=False
        )
        self.register_buffer(
            "aggregate_v", aggregate_v.clone(), persistent=False
        )

    def _norm_live_a(self, x, a_weight, aggregate):
        denom = torch.linalg.vector_norm(a_weight) + 1e-8
        return F.linear(F.linear(x, a_weight), aggregate / denom)

    def forward(self, x):
        if self.training and self.capture_input_sketch:
            with torch.no_grad():
                flat = x.detach().float().reshape(-1, self.dim)
                self.pending_input_square_sum.add_(flat.square().sum(dim=0))
                self.pending_input_count += float(flat.shape[0])
        new_q = self._norm_live_a(x, self.a_q.weight, self.aggregate_q)
        new_v = self._norm_live_a(x, self.a_v.weight, self.aggregate_v)
        new_q = new_q + self.scaling_cur[0](self.b_q(self.a_q(x)))
        new_v = new_v + self.scaling_cur[0](self.b_v(self.a_v(x)))
        qkv = self.qkv(x)
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv

    def consume_input_sketch(self):
        """Synchronize one pending minibatch and add it to task statistics."""
        if self.pending_input_count <= 0:
            return
        square_sum = self.pending_input_square_sum
        count = square_sum.new_tensor(self.pending_input_count)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(square_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
        self.task_input_square_sum.add_(square_sum)
        self.task_input_count += float(count)
        square_sum.zero_()
        self.pending_input_count = 0.0

    def merged_input_sketch(self):
        self.consume_input_sketch()
        historical_sum = (
            self.historical_input_rms.square() * self.historical_input_count
        )
        count = self.historical_input_count + self.task_input_count
        if count <= 0:
            return torch.ones_like(self.historical_input_rms), 0.0
        rms = torch.sqrt(
            torch.clamp_min(
                (historical_sum + self.task_input_square_sum) / count, 0.0
            )
        )
        return rms, count

    def historical_output(self, x):
        return (
            self._norm_live_a(x, self.a_q.weight, self.aggregate_q),
            self._norm_live_a(x, self.a_v.weight, self.aggregate_v),
        )

    def current_output(self, x):
        return (
            self.scaling_cur[0](self.b_q(self.a_q(x))),
            self.scaling_cur[0](self.b_v(self.a_v(x))),
        )


class SharedALoRA_ViT_timm(nn.Module):
    def __init__(
        self,
        vit_model: timm_ViT,
        r: int,
        num_classes: int = 0,
        increment=10,
        filepath="./",
        lora_layer=None,
        eval=False,
        index=True,
        cur_task_index=None,
        shared_a_orthogonal=True,
        train_a_all_tasks=False,
        delete_per_task_files=False,
        cumulative_state=False,
        cumulative_gauge=True,
        cumulative_merge=SA_MERGE_MODE_GAUGE,
        cumulative_rank=None,
        freeze_old_scales=False,
        live_a_history_groups=1,
        live_a_coordinate_align=False,
        live_a_absorb_mode=SA_ABSORB_MODE_OPERATOR_PRESERVING,
        adaptive_a_enabled=False,
        adaptive_a_stability_weight=1.0,
        adaptive_a_gate_floor=0.05,
        adaptive_a_gate_momentum=0.9,
        adaptive_a_eps=1e-8,
        adaptive_a_strategy="impact_ratio",
        adaptive_a_risk_budget=0.05,
        adaptive_a_risk_budget_mode="absolute",
        resume=False,
    ):
        super().__init__()
        assert r > 0
        if cumulative_merge not in (
            SA_MERGE_MODE_GAUGE,
            SA_MERGE_MODE_UNION_SVD,
            SA_MERGE_MODE_LIVE_A_AGGREGATE_B,
        ):
            raise ValueError(
                "cumulative_merge must be gauge/union_svd/live_a_aggregate_b; "
                "got {}".format(
                    cumulative_merge
                )
            )
        if (
            cumulative_merge == SA_MERGE_MODE_LIVE_A_AGGREGATE_B
            and not cumulative_state
        ):
            raise ValueError(
                "live_a_aggregate_b requires sa_cumulative_state=True"
            )
        if int(live_a_history_groups) < 1:
            raise ValueError("sa_live_a_history_groups must be >= 1")
        self.rank = r
        self.cumulative_rank = (
            int(cumulative_rank) if cumulative_rank is not None else r
        )
        if self.cumulative_rank <= 0 or self.cumulative_rank > r:
            raise ValueError(
                "cumulative_rank must satisfy 0 < rank <= lora_rank; "
                "got {}".format(self.cumulative_rank)
            )
        self.cumulative_merge = cumulative_merge
        self.freeze_old_scales = bool(freeze_old_scales)
        self.live_a_history_groups = int(live_a_history_groups)
        self.live_a_coordinate_align = bool(live_a_coordinate_align)
        if live_a_absorb_mode not in (
            SA_ABSORB_MODE_NORMALIZED,
            SA_ABSORB_MODE_OPERATOR_PRESERVING,
            SA_ABSORB_MODE_BOUNDED_NORM_CALIBRATED,
        ):
            raise ValueError(
                "sa_live_a_absorb_mode must be normalized_absorb, "
                "operator_preserving_absorb, or "
                "bounded_norm_calibrated_absorb"
            )
        self.live_a_absorb_mode = live_a_absorb_mode
        self.adaptive_a_enabled = bool(adaptive_a_enabled)
        self.adaptive_a_stability_weight = float(adaptive_a_stability_weight)
        self.adaptive_a_gate_floor = float(adaptive_a_gate_floor)
        self.adaptive_a_gate_momentum = float(adaptive_a_gate_momentum)
        self.adaptive_a_eps = float(adaptive_a_eps)
        self.adaptive_a_strategy = str(adaptive_a_strategy)
        self.adaptive_a_risk_budget = float(adaptive_a_risk_budget)
        self.adaptive_a_risk_budget_mode = str(adaptive_a_risk_budget_mode)
        if self.adaptive_a_strategy not in ("impact_ratio", "risk_budgeted"):
            raise ValueError(
                "sa_adaptive_a_strategy must be impact_ratio or risk_budgeted"
            )
        if self.adaptive_a_risk_budget < 0:
            raise ValueError("sa_adaptive_a_risk_budget must be non-negative")
        if self.adaptive_a_risk_budget_mode not in ("absolute", "relative"):
            raise ValueError(
                "sa_adaptive_a_risk_budget_mode must be absolute or relative"
            )
        if self.adaptive_a_stability_weight < 0:
            raise ValueError("sa_adaptive_a_stability_weight must be non-negative")
        if not 0.0 <= self.adaptive_a_gate_floor <= 1.0:
            raise ValueError("sa_adaptive_a_gate_floor must be in [0, 1]")
        if not 0.0 <= self.adaptive_a_gate_momentum < 1.0:
            raise ValueError("sa_adaptive_a_gate_momentum must be in [0, 1)")
        if self.adaptive_a_eps <= 0:
            raise ValueError("sa_adaptive_a_eps must be positive")
        if self.adaptive_a_enabled and not train_a_all_tasks:
            raise ValueError(
                "sa_adaptive_a_enabled requires sa_train_a_all_tasks=True"
            )
        if self.adaptive_a_enabled and not cumulative_state:
            raise ValueError(
                "sa_adaptive_a_enabled requires sa_cumulative_state=True"
            )
        if (
            self.adaptive_a_enabled
            and cumulative_merge != SA_MERGE_MODE_LIVE_A_AGGREGATE_B
        ):
            raise ValueError(
                "sa_adaptive_a_enabled requires "
                "sa_cumulative_merge=live_a_aggregate_b"
            )
        if self.adaptive_a_enabled and not live_a_coordinate_align:
            raise ValueError(
                "sa_adaptive_a_enabled requires "
                "sa_live_a_coordinate_align=True"
            )
        if (
            self.live_a_coordinate_align
            and cumulative_merge != SA_MERGE_MODE_LIVE_A_AGGREGATE_B
        ):
            raise ValueError(
                "sa_live_a_coordinate_align requires "
                "sa_cumulative_merge=live_a_aggregate_b"
            )
        self.save_file = filepath
        if cur_task_index is not None and cur_task_index == 0:
            state_path = _join_path(filepath, SA_STATE_FILENAME)
            manifest_path = _join_path(filepath, "run_manifest.json")
            if os.path.exists(state_path) and not resume:
                raise FileExistsError(
                    "task0 fresh-run guard: {} already exists. "
                    "Refusing to silently continue from an old Aggregate-B "
                    "state; pass sa_resume=true together with a run manifest "
                    "to explicitly recover.".format(state_path)
                )
            if resume and not os.path.exists(manifest_path):
                raise RuntimeError(
                    "sa_resume=true requires a run_manifest.json in {}; "
                    "cannot infer the original run metadata".format(filepath)
                )
        self.increment = increment
        self.shared_a_orthogonal = bool(shared_a_orthogonal)
        self.train_a_all_tasks = bool(train_a_all_tasks)
        self.delete_per_task_files = bool(delete_per_task_files)
        self.cumulative_state = bool(cumulative_state)
        self.cumulative_gauge = bool(cumulative_gauge)
        self.base_vit = vit_model

        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.blocks)))

        self.w_As, self.w_Bs = [], []
        if index:
            self.task_id, self.cur_id = 0, 0
        if cur_task_index is not None:
            self.task_id = cur_task_index

        for param in vit_model.parameters():
            param.requires_grad = False

        state = self._load_state()
        state_version = int(state.get("version", -1))
        if state_version == SA_STATE_VERSION_LEGACY:
            if self.cumulative_state:
                raise ValueError(
                    "sa_cumulative_state=True (merge={}) but artifact is "
                    "legacy v1; run the explicit migrate script first".format(
                        self.cumulative_merge
                    )
                )
            self.cumulative_state = False
        elif state_version == SA_STATE_VERSION:
            if self.cumulative_merge in (
                SA_MERGE_MODE_UNION_SVD,
                SA_MERGE_MODE_LIVE_A_AGGREGATE_B,
            ):
                raise ValueError(
                    "sa_cumulative_merge={} but artifact is v2 gauge; "
                    "run scripts/migrate_sa_state_v2_to_v3.py first "
                    "(union) or scripts/migrate_sa_v1_to_live_a_aggregate.py "
                    "first (live-a)".format(
                        self.cumulative_merge
                    )
                )
            self.cumulative_state = True
        elif state_version == SA_STATE_VERSION_UNION:
            if self.cumulative_merge != SA_MERGE_MODE_UNION_SVD:
                raise ValueError(
                    "artifact is v3 union-SVD but sa_cumulative_merge={}; "
                    "use union_svd or migrate back explicitly".format(
                        self.cumulative_merge
                    )
                )
            self.cumulative_state = True
        elif state_version == SA_STATE_VERSION_LIVE_A:
            if self.cumulative_merge != SA_MERGE_MODE_LIVE_A_AGGREGATE_B:
                raise ValueError(
                    "artifact is v4 live-a-aggregate-b but "
                    "sa_cumulative_merge={}; use live_a_aggregate_b".format(
                        self.cumulative_merge
                    )
                )
            state_coordinate_aligned = bool(
                state.get("coordinate_aligned", False)
            )
            state_absorb_mode = state.get(
                "absorb_mode", SA_ABSORB_MODE_NORMALIZED
            )
            if (
                self.task_id > 0
                and state_coordinate_aligned != self.live_a_coordinate_align
            ):
                raise ValueError(
                    "live-a coordinate alignment setting differs from the "
                    "saved state (saved={}, requested={})".format(
                        state_coordinate_aligned,
                        self.live_a_coordinate_align,
                    )
                )
            if self.task_id > 0 and state_absorb_mode != self.live_a_absorb_mode:
                raise ValueError(
                    "live-a absorption mode differs from the saved state "
                    "(saved={}, requested={})".format(
                        state_absorb_mode, self.live_a_absorb_mode
                    )
                )
            self.cumulative_state = True
        elif state_version == -1:
            # Fresh run: the flag decides which version new artifacts use.
            pass
        else:
            raise ValueError(
                "unsupported shared-A state version {}".format(state_version)
            )

        shared_a = state.get("shared_a", [])
        adaptive_a_input_rms = state.get("adaptive_a_input_rms", [])
        adaptive_a_input_counts = state.get("adaptive_a_input_counts", [])
        if self.cumulative_state:
            expected_branches = 2 * len(self.lora_layer)
            if self.cumulative_merge == SA_MERGE_MODE_LIVE_A_AGGREGATE_B:
                self.aggregate_up = [
                    t.detach().cpu().float()
                    for t in state.get("aggregate_up", [])
                ]
                self.cumulative_up = []
                self.canonical_down = []
                self.triangular_r = []
                if self.task_id > 0 and len(self.aggregate_up) != expected_branches:
                    raise ValueError(
                        "live-a aggregate state must contain {} branches; "
                        "got aggregate_up={}".format(
                            expected_branches, len(self.aggregate_up)
                        )
                    )
                if self.task_id > 0 and len(self.aggregate_up) == 0:
                    raise FileNotFoundError(
                        "{} is required before training task {}".format(
                            _join_path(self.save_file, SA_STATE_FILENAME),
                            self.task_id,
                        )
                    )
                loaded_a = state.get("shared_a", [])
                if self.task_id > 0 and len(loaded_a) != expected_branches:
                    raise ValueError(
                        "live-a state shared_a must contain {} branches; "
                        "got {}".format(expected_branches, len(loaded_a))
                    )
                shared_a = [t.detach().cpu().float() for t in loaded_a]
            else:
                self.aggregate_up = []
                self.cumulative_up = [
                    t.detach().cpu().float()
                    for t in state.get("cumulative_up", [])
                ]
                self.canonical_down = [
                    t.detach().cpu().float()
                    for t in state.get("canonical_down", [])
                ]
                self.triangular_r = [
                    t.detach().cpu().float()
                    for t in state.get("triangular_r", [])
                ]
                if self.task_id > 0 and not (
                    len(self.cumulative_up)
                    == len(self.canonical_down)
                    == len(self.triangular_r)
                    == expected_branches
                ):
                    raise ValueError(
                        "cumulative Shared-A state must contain {} branches; "
                        "got cumulative_up={} canonical_down={} triangular_r={}".format(
                            expected_branches,
                            len(self.cumulative_up),
                            len(self.canonical_down),
                            len(self.triangular_r),
                        )
                    )
                if self.task_id > 0 and len(self.cumulative_up) == 0:
                    raise FileNotFoundError(
                        "{} is required before training task {}".format(
                            _join_path(self.save_file, SA_STATE_FILENAME),
                            self.task_id,
                        )
                    )
                shared_a = [
                    (r.t() @ q).float()
                    for q, r in zip(self.canonical_down, self.triangular_r)
                ]
            self.saved_b_tasks = {}
        elif self.task_id > 0 and len(shared_a) == 0:
            raise FileNotFoundError(
                "{} is required before training task {}".format(
                    _join_path(self.save_file, SA_STATE_FILENAME), self.task_id
                )
            )

        if not self.cumulative_state:
            self.saved_b_tasks = {}
            for task_id in range(self.task_id):
                path = _join_path(
                    self.save_file, "sa_lora_w_b_{}.pt".format(task_id)
                )
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        "missing saved shared-A B for task {}".format(task_id)
                    )
                self.saved_b_tasks[task_id] = torch.load(
                    path, map_location="cpu", weights_only=True
                )

        # These are task-local, non-persistent snapshots used only while the
        # next task is trained. They never enter the saved Shared-A artifact.
        self._live_a_previous_shared_a = (
            [tensor.detach().cpu().float().clone() for tensor in shared_a]
            if self.cumulative_merge == SA_MERGE_MODE_LIVE_A_AGGREGATE_B
            else []
        )
        self._last_live_a_coordinate_diagnostics = None
        self._last_saved_live_a_aggregate = None
        self._operator_reference_down = []
        self._operator_reference_up = []
        self._operator_reference_task_count = 0
        # Diagnostics computed from the *pre-save* historical state; populated
        # by _save_cumulative_state before the state is overwritten.
        self._last_cumulative_gauge_diagnostics = None
        self._last_union_svd_truncation_error = None
        self._last_live_a_save_stats = None
        # HBD teacher lifecycle (plain attributes; never enter state_dict).
        self._hbd_capture_list = None
        # These values are deliberately task-local and are not buffers or
        # modules, so deployment state remains the existing (A, G) artifact.
        self._adaptive_a_gate_ema = {}
        self._adaptive_a_observations = []
        self._adaptive_a_statistics_task_id = None
        self._adaptive_a_last_modes = {}
        self._adaptive_a_previous_gradients = None
        self._adaptive_a_last_effective_budget = None
        self._adaptive_a_last_live_risk = None

        scaling_factor = nn.Parameter(torch.Tensor([0.8]))
        self.wrapped_param = nn.ModuleList([ParameterWrapper(scaling_factor)])
        if self.cumulative_state:
            self.wrapped_param_prev = nn.ModuleList()
        else:
            saved_scales = state.get("scales", {})
            residual_scale_values = [
                saved_scales.get(task_id, torch.Tensor([0.8]))
                for task_id in range(self.task_id)
            ]
            residual_scale_values.extend(
                [
                    torch.Tensor([0.8])
                    for _ in range(
                        max(0, 20 - len(residual_scale_values))
                    )
                ]
            )
            self.wrapped_param_prev = nn.ModuleList(
                [
                    ParameterWrapper(
                        nn.Parameter(value.detach().clone().float())
                    )
                    for value in residual_scale_values
                ]
            )
            if self.freeze_old_scales:
                for idx in range(self.task_id):
                    self.wrapped_param_prev[idx].param.requires_grad_(False)

        for layer_index, blk in enumerate(vit_model.blocks):
            if layer_index not in self.lora_layer:
                continue
            qkv = blk.attn.qkv
            dim = qkv.in_features
            a_q = nn.Linear(dim, r, bias=False)
            a_v = nn.Linear(dim, r, bias=False)
            b_q = nn.Linear(r, dim, bias=False)
            b_v = nn.Linear(r, dim, bias=False)

            offset = len(self.w_As)
            self.w_As.extend([a_q, a_v])
            self.w_Bs.extend([b_q, b_v])

            if self.task_id > 0 and not self.train_a_all_tasks:
                a_q.weight.data.copy_(shared_a[offset].to(a_q.weight.dtype))
                a_v.weight.data.copy_(shared_a[offset + 1].to(a_v.weight.dtype))
                a_q.weight.requires_grad_(False)
                a_v.weight.requires_grad_(False)
            elif self.task_id > 0:
                a_q.weight.data.copy_(shared_a[offset].to(a_q.weight.dtype))
                a_v.weight.data.copy_(shared_a[offset + 1].to(a_v.weight.dtype))
            elif self.shared_a_orthogonal:
                a_q.weight.data.copy_(
                    _fixed_orthogonal_down(
                        dim, r, 1729 + offset, dtype=a_q.weight.dtype
                    )
                )
                a_v.weight.data.copy_(
                    _fixed_orthogonal_down(
                        dim, r, 1729 + offset + 1, dtype=a_v.weight.dtype
                    )
                )

            if self.cumulative_state:
                if (
                    self.cumulative_merge
                    == SA_MERGE_MODE_LIVE_A_AGGREGATE_B
                ):
                    agg_q = (
                        self.aggregate_up[offset]
                        if offset < len(self.aggregate_up)
                        else torch.zeros(dim, r)
                    )
                    agg_v = (
                        self.aggregate_up[offset + 1]
                        if offset + 1 < len(self.aggregate_up)
                        else torch.zeros(dim, r)
                    )
                    blk.attn.qkv = _LiveAAggregateQKV(
                        qkv,
                        a_q,
                        a_v,
                        b_q,
                        b_v,
                        agg_q,
                        agg_v,
                        self.wrapped_param,
                        layer_index,
                        history_groups=self.live_a_history_groups,
                        historical_input_rms=(
                            adaptive_a_input_rms[offset // 2]
                            if len(adaptive_a_input_rms) > offset // 2
                            else None
                        ),
                        historical_input_count=(
                            float(adaptive_a_input_counts[offset // 2])
                            if len(adaptive_a_input_counts) > offset // 2
                            else 0.0
                        ),
                        capture_input_sketch=(
                            self.adaptive_a_enabled
                            and self.adaptive_a_strategy == "risk_budgeted"
                        ),
                    )
                else:
                    if offset < len(self.cumulative_up):
                        h_q = self.cumulative_up[offset]
                        q_q_t = self.canonical_down[offset]
                        h_v = self.cumulative_up[offset + 1]
                        q_v_t = self.canonical_down[offset + 1]
                    else:
                        h_q = torch.zeros(dim, r)
                        h_v = torch.zeros(dim, r)
                        q_q_t, _ = canonical_down_projection(
                            a_q.weight.detach().cpu()
                        )
                        q_v_t, _ = canonical_down_projection(
                            a_v.weight.detach().cpu()
                        )
                    blk.attn.qkv = _CumulativeSharedAQKV(
                        qkv,
                        a_q,
                        a_v,
                        b_q,
                        b_v,
                        h_q,
                        q_q_t,
                        h_v,
                        q_v_t,
                        self.wrapped_param,
                        layer_index,
                    )
            else:
                saved_b_q = [
                    self.saved_b_tasks[task_id][offset]
                    for task_id in range(self.task_id)
                ]
                saved_b_v = [
                    self.saved_b_tasks[task_id][offset + 1]
                    for task_id in range(self.task_id)
                ]
                blk.attn.qkv = _SharedAQKV(
                    qkv,
                    a_q,
                    a_v,
                    b_q,
                    b_v,
                    saved_b_q,
                    saved_b_v,
                    self.wrapped_param,
                    self.wrapped_param_prev,
                    layer_index,
                )

        self.reset_parameters()
        if (
            not self.cumulative_state
            and self.task_id > 0
            and self.train_a_all_tasks
        ):
            self._capture_old_operator_reference()
        self.lora_vit = vit_model
        self.lora_vit.head = nn.Identity()
        self.out_dim = 768

    def _capture_old_operator_reference(self):
        """Snapshot the historical LoRA operator before shared A is updated."""
        self._operator_reference_task_count = self.task_id
        reference_scales = [
            self.wrapped_param_prev[task_id].param.detach().cpu().clone()
            for task_id in range(self.task_id)
        ]
        self._operator_reference_down = [
            w_a.weight.detach().cpu().clone() for w_a in self.w_As
        ]
        self._operator_reference_up = []
        for index in range(len(self.w_As)):
            historical_up = [
                self.saved_b_tasks[task_id][index]
                for task_id in range(self.task_id)
            ]
            self._operator_reference_up.append(
                aggregate_normalized_up_projections(
                    historical_up, reference_scales
                ).detach().cpu()
            )

    def old_operator_stability_loss(self):
        """Relative drift of all historical Shared-A LoRA Q/V operators.

        The live term includes the current values of historical task scales;
        therefore it protects the exact old LoRA bank used in the forward pass,
        not just the shared down projection in isolation.
        """
        if self.cumulative_state:
            # The historical operator is fixed in canonical coordinates while
            # the current task trains; the gauge residual is logged separately.
            return self.w_As[0].weight.new_zeros(())
        if self._operator_reference_task_count == 0:
            return self.w_As[0].weight.new_zeros(())
        if not self._operator_reference_down or not self._operator_reference_up:
            raise RuntimeError("missing Shared-A historical operator reference")

        current_scales = [
            self.wrapped_param_prev[task_id].param
            for task_id in range(self._operator_reference_task_count)
        ]
        losses = []
        for index, w_a in enumerate(self.w_As):
            historical_up = [
                self.saved_b_tasks[task_id][index]
                for task_id in range(self._operator_reference_task_count)
            ]
            current_up = aggregate_normalized_up_projections(
                historical_up,
                current_scales,
                device=w_a.weight.device,
                dtype=w_a.weight.dtype,
            )
            losses.append(
                relative_effective_operator_drift(
                    current_down=w_a.weight,
                    reference_down=self._operator_reference_down[index],
                    current_up=current_up,
                    reference_up=self._operator_reference_up[index],
                )
            )
        return torch.stack(losses).mean()

    def cumulative_gauge_residual(self) -> float:
        """Mean relative projection residual of the historical operator.

        Computes ``||H_old Q_old^T (I - Q_new Q_new^T)||_F /
        ||H_old Q_old^T||_F`` per branch with ``Q_new`` derived from the
        current (trained) shared A.  Zero when no historical operator exists.
        """
        if not self.cumulative_state or not self.cumulative_up:
            return 0.0
        residuals = []
        for idx, w_a in enumerate(self.w_As):
            q_t, _ = canonical_down_projection(w_a.weight.detach().cpu())
            h = self.cumulative_up[idx]
            q_old_t = self.canonical_down[idx]
            old_norm = (h.double() @ q_old_t.double()).norm()
            rel = gauge_projection_residual(h, q_old_t, q_t) / (
                old_norm + 1e-8
            )
            residuals.append(float(rel))
        return float(torch.tensor(residuals).mean())

    def _compute_gauge_diagnostics_between(
        self, cumulative_up, canonical_down, q_t_list
    ) -> dict:
        """Per-branch gauge diagnostics between an old state and new bases.

        ``cumulative_up``/``canonical_down`` describe the old historical
        operator ``H_old Q_old^T`` and ``q_t_list`` are the new canonical
        down-projections ``Q_new^T`` of the trained shared A.  Returns mean
        relative projection residual (out-of-span part), RMS basis rotation
        ``||Q_old^T Q_new - I||_F / sqrt(r)``, and relative operator
        preservation error after closed-form gauge alignment.
        """
        if len(cumulative_up) == 0:
            return {
                "residual": 0.0,
                "rotation_fro": 0.0,
                "preservation": 0.0,
                "branches": 0,
            }
        residuals = []
        rotations = []
        preservations = []
        for idx, q_t in enumerate(q_t_list):
            h = cumulative_up[idx]
            q_old_t = canonical_down[idx]
            old_operator = h.double() @ q_old_t.double()
            old_norm = old_operator.norm()
            rel_residual = gauge_projection_residual(h, q_old_t, q_t) / (
                old_norm + 1e-8
            )
            identity = torch.eye(
                q_old_t.shape[0], dtype=torch.float64
            )
            overlap = q_old_t.double() @ q_t.double().t()
            rotation = (
                overlap - identity
            ).norm() / (q_old_t.shape[0] ** 0.5)
            h_aligned = gauge_align_up_projection(h, q_old_t, q_t)
            preservation = (
                h_aligned.double() @ q_t.double() - old_operator
            ).norm() / (old_norm + 1e-8)
            residuals.append(float(rel_residual))
            rotations.append(float(rotation))
            preservations.append(float(preservation))
        return {
            "residual": float(torch.tensor(residuals).mean()),
            "rotation_fro": float(torch.tensor(rotations).mean()),
            "preservation": float(torch.tensor(preservations).mean()),
            "branches": len(residuals),
        }

    def cumulative_gauge_diagnostics(self) -> dict:
        """Current-state diagnostics (for tests and manual inspection).

        NOTE: after ``_save_cumulative_state`` the stored state has already
        been overwritten, so this method compares the new state with itself
        and returns ~0.  Training logs must use
        ``_last_cumulative_gauge_diagnostics``, which is captured before the
        state is replaced.
        """
        if not self.cumulative_state or not self.cumulative_up:
            return {
                "residual": 0.0,
                "rotation_fro": 0.0,
                "preservation": 0.0,
                "branches": 0,
            }
        q_t_list = [
            canonical_down_projection(w_a.weight.detach().cpu())[0]
            for w_a in self.w_As
        ]
        return self._compute_gauge_diagnostics_between(
            self.cumulative_up, self.canonical_down, q_t_list
        )

    def apply_adaptive_a_gradients(
        self,
        control_gradients: list[Tensor | None] | None = None,
        step_size: float = 1.0,
        momentum_buffers: list[Tensor | None] | None = None,
        momentum: float = 0.0,
    ) -> dict | None:
        """Apply the synchronized Adaptive-A gate to current shared-A grads."""
        if not self.adaptive_a_enabled:
            return None
        if self._adaptive_a_statistics_task_id != self.task_id:
            self._adaptive_a_gate_ema = {}
            self._adaptive_a_observations = []
            self._adaptive_a_statistics_task_id = self.task_id
            self._adaptive_a_last_modes = {}
            self._adaptive_a_previous_gradients = None
            self._adaptive_a_last_effective_budget = None
            self._adaptive_a_last_live_risk = None
        if self.adaptive_a_strategy == "risk_budgeted":
            if control_gradients is None:
                control_gradients = self._adaptive_a_previous_gradients
            return self._apply_risk_budgeted_adaptive_a(
                control_gradients,
                step_size=step_size,
                momentum_buffers=momentum_buffers,
                momentum=momentum,
            )
        layer_gradients = []
        for block in self.lora_vit.blocks:
            wrapper = block.attn.qkv
            if not isinstance(wrapper, _LiveAAggregateQKV):
                continue
            gradient_q = wrapper.a_q.weight.grad
            gradient_v = wrapper.a_v.weight.grad
            if gradient_q is None or gradient_v is None:
                continue
            has_history = self.task_id > 0
            result = adaptive_a_layer_gradient(
                gradient_q=gradient_q,
                gradient_v=gradient_v,
                shared_a_q=wrapper.a_q.weight,
                shared_a_v=wrapper.a_v.weight,
                current_up_q=wrapper.b_q.weight,
                current_up_v=wrapper.b_v.weight,
                historical_up_q=(wrapper.aggregate_q if has_history else None),
                historical_up_v=(wrapper.aggregate_v if has_history else None),
                scale=self.wrapped_param[0].param,
                stability_weight=self.adaptive_a_stability_weight,
                gate_floor=self.adaptive_a_gate_floor,
                momentum=self.adaptive_a_gate_momentum,
                eps=self.adaptive_a_eps,
                previous_gate=self._adaptive_a_gate_ema.get(
                    wrapper.layer_index
                ),
            )
            with torch.no_grad():
                if has_history:
                    gradient_q.copy_(result["gradient_q"])
                    gradient_v.copy_(result["gradient_v"])
            gate = float(result["gate"].detach())
            self._adaptive_a_gate_ema[wrapper.layer_index] = gate
            self._adaptive_a_observations.append(
                {
                    "layer": wrapper.layer_index,
                    "gate": gate,
                    "current_impact": float(result["current_impact"].detach()),
                    "historical_impact": float(
                        result["historical_impact"].detach()
                    ),
                    "perpendicular_retention": float(
                        result["perpendicular_retention"].detach()
                    ),
                }
            )
            layer_gradients.append(result)
        diagnostics = self.adaptive_a_diagnostics()
        if diagnostics is None:
            return None
        return {**diagnostics, "layer_gradients": layer_gradients}

    def _apply_risk_budgeted_adaptive_a(
        self, control_gradients, step_size, momentum_buffers, momentum
    ):
        """Select exact Frozen/Tangent/Live gradients under one global budget."""
        if step_size < 0:
            raise ValueError("step_size must be non-negative")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if momentum_buffers is not None and len(momentum_buffers) != len(self.w_As):
            raise ValueError("momentum_buffers must match shared-A branches")
        wrappers = [
            block.attn.qkv
            for block in self.lora_vit.blocks
            if isinstance(block.attn.qkv, _LiveAAggregateQKV)
        ]
        if control_gradients is not None and len(control_gradients) != len(self.w_As):
            raise ValueError("control_gradients must match shared-A branches")
        candidates_by_layer = []
        tensor_candidates = []
        raw_gradients = []
        has_history = self.task_id > 0
        for wrapper_index, wrapper in enumerate(wrappers):
            wrapper.consume_input_sketch()
            gradient_q = wrapper.a_q.weight.grad
            gradient_v = wrapper.a_v.weight.grad
            if gradient_q is None or gradient_v is None:
                continue
            momentum_q = (
                momentum_buffers[2 * wrapper_index]
                if momentum_buffers is not None
                else None
            )
            momentum_v = (
                momentum_buffers[2 * wrapper_index + 1]
                if momentum_buffers is not None
                else None
            )
            proposed_q = gradient_q + (
                momentum * momentum_q if momentum_q is not None else 0.0
            )
            proposed_v = gradient_v + (
                momentum * momentum_v if momentum_v is not None else 0.0
            )
            raw_gradients.extend(
                [proposed_q.detach().clone(), proposed_v.detach().clone()]
            )
            q_candidates = decompose_adaptive_a_gradient(
                proposed_q, wrapper.a_q.weight
            )
            v_candidates = decompose_adaptive_a_gradient(
                proposed_v, wrapper.a_v.weight
            )
            control_q = (
                control_gradients[2 * wrapper_index]
                if control_gradients is not None
                else None
            )
            control_v = (
                control_gradients[2 * wrapper_index + 1]
                if control_gradients is not None
                else None
            )
            layer = {}
            input_rms = wrapper.historical_input_rms.to(
                device=gradient_q.device, dtype=gradient_q.dtype
            )
            for mode in ADAPTIVE_A_MODES:
                if not has_history:
                    utility = 1.0 if mode == "live" else 0.0
                    risk = 0.0
                else:
                    utility = 0.0
                    if control_q is not None:
                        utility += float(
                            signed_gradient_utility(
                                q_candidates[mode], control_q, self.adaptive_a_eps
                            ).detach()
                        )
                    if control_v is not None:
                        utility += float(
                            signed_gradient_utility(
                                v_candidates[mode], control_v, self.adaptive_a_eps
                            ).detach()
                        )
                    risk = float(
                        (
                            weighted_operator_risk(
                                wrapper.aggregate_q,
                                wrapper.a_q.weight,
                                step_size * q_candidates[mode],
                                input_rms,
                                self.adaptive_a_eps,
                            )
                            + weighted_operator_risk(
                                wrapper.aggregate_v,
                                wrapper.a_v.weight,
                                step_size * v_candidates[mode],
                                input_rms,
                                self.adaptive_a_eps,
                            )
                        ).detach()
                    )
                layer[mode] = {"utility": utility, "risk": risk}
            candidates_by_layer.append(layer)
            tensor_candidates.append(
                (wrapper, q_candidates, v_candidates, momentum_q, momentum_v)
            )
        live_risk = sum(
            layer["live"]["risk"] for layer in candidates_by_layer
        )
        budget = (
            float("inf")
            if not has_history
            else effective_risk_budget(
                candidates_by_layer,
                self.adaptive_a_risk_budget,
                self.adaptive_a_risk_budget_mode,
            )
        )
        self._adaptive_a_last_effective_budget = budget
        self._adaptive_a_last_live_risk = live_risk
        if not has_history:
            selection = {
                "modes": ["live"] * len(candidates_by_layer),
                "selected_risk": 0.0,
                "selected_utility": float(len(candidates_by_layer)),
            }
        else:
            selection = choose_risk_budgeted_modes(candidates_by_layer, budget)
        for mode, layer, tensors in zip(
            selection["modes"], candidates_by_layer, tensor_candidates
        ):
            wrapper, q_candidates, v_candidates, momentum_q, momentum_v = tensors
            with torch.no_grad():
                q_gradient = q_candidates[mode]
                v_gradient = v_candidates[mode]
                if momentum_q is not None:
                    q_gradient = q_gradient - momentum * momentum_q
                if momentum_v is not None:
                    v_gradient = v_gradient - momentum * momentum_v
                wrapper.a_q.weight.grad.copy_(q_gradient)
                wrapper.a_v.weight.grad.copy_(v_gradient)
            self._adaptive_a_last_modes[wrapper.layer_index] = mode
            self._adaptive_a_observations.append(
                {
                    "layer": wrapper.layer_index,
                    "mode": mode,
                    "gate": {"frozen": 0.0, "tangent": 0.5, "live": 1.0}[mode],
                    "current_impact": layer[mode]["utility"],
                    "historical_impact": layer[mode]["risk"],
                    "perpendicular_retention": float(mode == "live"),
                    "selected_utility": layer[mode]["utility"],
                    "selected_risk": layer[mode]["risk"],
                }
            )
        self._adaptive_a_previous_gradients = raw_gradients
        diagnostics = self.adaptive_a_diagnostics()
        return {**diagnostics, "selection": selection} if diagnostics else None

    def adaptive_a_diagnostics(self) -> dict | None:
        """Return aggregate task-local Adaptive-A gate diagnostics."""
        if not self.adaptive_a_enabled or not self._adaptive_a_observations:
            return None
        observations = self._adaptive_a_observations
        gates = [item["gate"] for item in observations]
        per_layer = {}
        for item in observations:
            per_layer.setdefault(item["layer"], []).append(item["gate"])
        diagnostics = {
            "observations": len(observations),
            "mean_gate": sum(gates) / len(gates),
            "min_gate": min(gates),
            "max_gate": max(gates),
            "fraction_gate_below_0_1": sum(gate < 0.1 for gate in gates)
            / len(gates),
            "fraction_gate_above_0_9": sum(gate > 0.9 for gate in gates)
            / len(gates),
            "mean_current_impact": sum(
                item["current_impact"] for item in observations
            )
            / len(observations),
            "mean_historical_impact": sum(
                item["historical_impact"] for item in observations
            )
            / len(observations),
            "mean_perpendicular_retention": sum(
                item["perpendicular_retention"] for item in observations
            )
            / len(observations),
            "per_layer_mean_gate": [
                sum(per_layer[layer]) / len(per_layer[layer])
                for layer in sorted(per_layer)
            ],
        }
        if self.adaptive_a_strategy == "risk_budgeted":
            mode_counts = {
                mode: sum(item.get("mode") == mode for item in observations)
                for mode in ADAPTIVE_A_MODES
            }
            diagnostics.update(
                {
                    "strategy": "risk_budgeted",
                    "mode_counts": mode_counts,
                    "mode_fractions": {
                        mode: mode_counts[mode] / len(observations)
                        for mode in ADAPTIVE_A_MODES
                    },
                    "mean_selected_utility": sum(
                        item["selected_utility"] for item in observations
                    )
                    / len(observations),
                    "mean_selected_risk": sum(
                        item["selected_risk"] for item in observations
                    )
                    / len(observations),
                    "risk_budget_mode": self.adaptive_a_risk_budget_mode,
                    "effective_risk_budget": self._adaptive_a_last_effective_budget,
                    "live_risk_reference": self._adaptive_a_last_live_risk,
                }
            )
        return diagnostics

    def live_a_gradient_diagnostics(self, x) -> dict | None:
        """First-batch Live-A training-path diagnostics.

        Returns norms of the historical/current branch gradients w.r.t. the
        shared A and the per-branch G/A/B/scale norms, or None when the model
        is not in Live-A mode.
        """
        if self.cumulative_merge != SA_MERGE_MODE_LIVE_A_AGGREGATE_B:
            return None
        wrappers = [
            blk.attn.qkv
            for blk in self.lora_vit.blocks
            if isinstance(blk.attn.qkv, _LiveAAggregateQKV)
        ]
        if not wrappers:
            return None
        a_params = [w.weight for w in self.w_As]
        if not all(p.requires_grad for p in a_params):
            # Frozen-A ablation (P4-C): the Live-A gradient path is disabled
            # by construction; there is no A gradient to report.
            return None

        hist_loss = torch.zeros((), device=x.device)
        cur_loss = torch.zeros((), device=x.device)
        g_norm = 0.0
        a_norm = 0.0
        b_norm = 0.0
        for wrapper in wrappers:
            hq, hv = wrapper.historical_output(x)
            cq, cv = wrapper.current_output(x)
            hist_loss = hist_loss + hq.square().sum() + hv.square().sum()
            cur_loss = cur_loss + cq.square().sum() + cv.square().sum()
            g_norm = g_norm + (
                wrapper.aggregate_q.square().sum()
                + wrapper.aggregate_v.square().sum()
            ).item()
            a_norm = a_norm + (
                wrapper.a_q.weight.square().sum()
                + wrapper.a_v.weight.square().sum()
            ).item()
            b_norm = b_norm + (
                wrapper.b_q.weight.square().sum()
                + wrapper.b_v.weight.square().sum()
            ).item()
        hist_grads = torch.autograd.grad(
            hist_loss, a_params, retain_graph=True
        )
        cur_grads = torch.autograd.grad(
            cur_loss, a_params, retain_graph=True
        )
        hist_da_norm = sum(
            g.detach().square().sum().item() for g in hist_grads
        ) ** 0.5
        cur_da_norm = sum(
            g.detach().square().sum().item() for g in cur_grads
        ) ** 0.5
        return {
            "historical_dL_dA": hist_da_norm,
            "current_dL_dA": cur_da_norm,
            "ratio_hist_cur": (
                hist_da_norm / max(cur_da_norm, 1e-12)
            ),
            "mean_G_norm": (g_norm / len(wrappers)) ** 0.5,
            "mean_A_norm": (a_norm / len(wrappers)) ** 0.5,
            "mean_B_norm": (b_norm / len(wrappers)) ** 0.5,
            "scale": float(self.wrapped_param[0].param.detach().item()),
        }

    def _fold_live_a_teacher_aggregate(self):
        """Return ``G_prev`` per Q/V branch from the task-start model state.

        The in-memory Live-A semantics keep ``aggregate_up == G_{t-2}`` while
        the current raw ``B_{t-1}`` is still in the wrappers.  The teacher
        snapshot for task ``t`` needs ``G_{t-1} = G_{t-2} + s B_{t-1}/||B_{t-1}||``,
        i.e. exactly what ``_save_live_a_state`` writes to disk.
        """
        if self.cumulative_merge != SA_MERGE_MODE_LIVE_A_AGGREGATE_B:
            raise ValueError(
                "HBD teacher requires sa_cumulative_merge=live_a_aggregate_b"
            )
        folded = []
        for idx, w_b in enumerate(self.w_Bs):
            b = w_b.weight.detach().cpu().float()
            scale = (
                self.wrapped_param[0]
                .param.detach()
                .cpu()
                .float()
                .reshape(())
            )
            g_old = (
                self.aggregate_up[idx]
                if idx < len(self.aggregate_up)
                else torch.zeros_like(b)
            )
            current_up, _ = absorb_live_a_current_projection(
                self.w_As[idx].weight.detach().cpu().float(),
                b,
                scale,
                mode=self.live_a_absorb_mode,
            )
            folded.append(g_old + current_up)
        return folded

    def build_hbd_teacher(self):
        """Build the frozen task-start teacher (``A_prev``, ``G_prev``, zero B).

        The teacher is a deep copy of the current model with:
          * shared A frozen at the current (task-start) value;
          * historical aggregate folded to ``G_prev = G_{t-2} + s B_{t-1}/||B_{t-1}||``;
          * current-task B zeroed so the full forward equals ``G_prev A/||A||``.
        It is not registered anywhere, never saved, and must be released at
        the end of the task.
        """
        teacher = copy.deepcopy(self)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        folded = self._fold_live_a_teacher_aggregate()
        wrappers = [
            blk.attn.qkv
            for blk in teacher.lora_vit.blocks
            if isinstance(blk.attn.qkv, _LiveAAggregateQKV)
        ]
        if len(wrappers) * 2 != len(folded):
            raise ValueError(
                "HBD teacher branch mismatch: {} wrappers vs {} folded".format(
                    len(wrappers), len(folded)
                )
            )
        for wrapper_index, wrapper in enumerate(wrappers):
            wrapper.aggregate_q.copy_(
                folded[2 * wrapper_index].to(wrapper.aggregate_q.device)
            )
            wrapper.aggregate_v.copy_(
                folded[2 * wrapper_index + 1].to(
                    wrapper.aggregate_v.device
                )
            )
        for w_b in teacher.w_Bs:
            w_b.weight.zero_()
        return teacher

    def _load_state(self):
        path = _join_path(self.save_file, SA_STATE_FILENAME)
        if os.path.exists(path):
            state = torch.load(path, map_location="cpu", weights_only=True)
            version = int(state.get("version", -1))
            if version not in (
                SA_STATE_VERSION_LEGACY,
                SA_STATE_VERSION,
                SA_STATE_VERSION_UNION,
                SA_STATE_VERSION_LIVE_A,
            ):
                raise ValueError("unsupported shared-A state version")
            return state
        return {"version": -1, "shared_a": [], "scales": {}}

    def reset_parameters(self):
        if self.task_id == 0:
            for w_a in self.w_As:
                if not self.shared_a_orthogonal:
                    nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_Bs:
            nn.init.zeros_(w_b.weight)

    def generate_fc(self, in_dim, out_dim):
        return SimpleLinear(in_dim, out_dim)

    def save_lora_parameters(self, filename: str, task_id) -> None:
        if self.cumulative_state:
            if (
                self.cumulative_merge
                == SA_MERGE_MODE_LIVE_A_AGGREGATE_B
            ):
                self._save_live_a_state(filename, task_id)
            else:
                self._save_cumulative_state(filename, task_id)
            return
        self.task_id += 1
        if not os.path.exists(filename):
            os.makedirs(filename)
        torch.save(
            [w.weight.detach().cpu() for w in self.w_Bs],
            _join_path(filename, "sa_lora_w_b_{}.pt".format(task_id)),
        )
        scales = {}
        for idx, residual_task_id in enumerate(range(self.task_id - 1)):
            scales[residual_task_id] = self.wrapped_param_prev[
                idx
            ].param.detach().cpu()
        scales[task_id] = self.wrapped_param[0].param.detach().cpu()
        torch.save(
            {
                "version": SA_STATE_VERSION_LEGACY,
                "shared_a": [w.weight.detach().cpu() for w in self.w_As],
                "scales": scales,
            },
            _join_path(filename, SA_STATE_FILENAME),
        )
        self.save_merged_lora(filename)

    def _save_cumulative_state(self, filename: str, task_id) -> None:
        """Fold the current task into the canonical cumulative state.

        The historical operator ``H_old Q_old^T`` is aligned to the new
        canonical basis with the closed-form gauge solution, then the current
        task's normalized LoRA contribution is added.  Per-task B files are
        never written; the only artifact is the v2 state file.
        """
        if task_id != self.task_id:
            raise ValueError(
                "cumulative state save called with task_id={} but "
                "task_id is {}".format(task_id, self.task_id)
            )
        self.task_id += 1
        if not os.path.exists(filename):
            os.makedirs(filename)
        # Capture the *pre-save* gauge diagnostics against the newly trained
        # shared A before the historical state is overwritten below.
        if len(self.cumulative_up) > 0:
            q_t_list = [
                canonical_down_projection(w_a.weight.detach().cpu())[0]
                for w_a in self.w_As
            ]
            self._last_cumulative_gauge_diagnostics = (
                self._compute_gauge_diagnostics_between(
                    self.cumulative_up, self.canonical_down, q_t_list
                )
            )
        else:
            self._last_cumulative_gauge_diagnostics = {
                "residual": 0.0,
                "rotation_fro": 0.0,
                "preservation": 0.0,
                "branches": 0,
            }
        canonical_down = []
        cumulative_up = []
        triangular_r = []
        if self.cumulative_merge == SA_MERGE_MODE_UNION_SVD:
            self._last_union_svd_truncation_error = 0.0
            for idx, (w_a, w_b) in enumerate(zip(self.w_As, self.w_Bs)):
                a = w_a.weight.detach().cpu().float()
                b = w_b.weight.detach().cpu().float()
                s = (
                    self.wrapped_param[0]
                    .param.detach()
                    .cpu()
                    .float()
                    .reshape(())
                )
                norm_a = torch.linalg.vector_norm(a)
                norm_b = torch.linalg.vector_norm(b) + 1e-8
                left = []
                right = []
                if idx < len(self.cumulative_up):
                    left.append(self.cumulative_up[idx])
                    right.append(self.canonical_down[idx].t())
                left.append(s * b / (norm_a * norm_b))
                right.append(a.t())
                q_t, h_t, _, rel_err = union_svd_factors(
                    left, right, rank=self.cumulative_rank
                )
                self._last_union_svd_truncation_error = max(
                    self._last_union_svd_truncation_error, rel_err
                )
                canonical_down.append(q_t)
                cumulative_up.append(h_t)
                triangular_r.append(
                    torch.eye(self.cumulative_rank, dtype=torch.float32)
                )
        else:
            for idx, (w_a, w_b) in enumerate(zip(self.w_As, self.w_Bs)):
                a = w_a.weight.detach().cpu().float()
                b = w_b.weight.detach().cpu().float()
                s = (
                    self.wrapped_param[0]
                    .param.detach()
                    .cpu()
                    .float()
                    .reshape(())
                )
                q_t, r = canonical_down_projection(a)
                if idx < len(self.cumulative_up) and self.task_id > 1:
                    if self.cumulative_gauge:
                        h_hist = gauge_align_up_projection(
                            self.cumulative_up[idx],
                            self.canonical_down[idx],
                            q_t,
                        )
                    else:
                        # Ablation: keep the old cumulative up projection
                        # without aligning it to the new canonical basis.
                        h_hist = self.cumulative_up[idx]
                else:
                    h_hist = torch.zeros_like(b)
                norm_a = torch.linalg.vector_norm(a)
                norm_b = torch.linalg.vector_norm(b) + 1e-8
                h_cur_raw = s * b / (norm_a * norm_b)
                h_cur = canonicalize_effective_up_projection(h_cur_raw, r)
                cumulative_up.append(h_hist + h_cur)
                canonical_down.append(q_t)
                triangular_r.append(r)

        self.cumulative_up = cumulative_up
        self.canonical_down = canonical_down
        self.triangular_r = triangular_r
        state = {
            "version": (
                SA_STATE_VERSION_UNION
                if self.cumulative_merge == SA_MERGE_MODE_UNION_SVD
                else SA_STATE_VERSION
            ),
            "task_id": self.task_id,
            "rank": self.cumulative_rank,
            "merge_mode": self.cumulative_merge,
            "canonical_down": canonical_down,
            "cumulative_up": cumulative_up,
            "triangular_r": triangular_r,
        }
        torch.save(
            state,
            _join_path(filename, SA_STATE_FILENAME),
        )
        self.save_merged_lora(filename)

    def _save_live_a_state(self, filename: str, task_id) -> None:
        """Save the Live-A Aggregate-B state.

        Folds the current raw operator into the configured fixed-state
        representation and saves it with the current shared A. The in-memory
        aggregate remains unchanged until the deployment backbone is rebuilt.
        """
        if task_id != self.task_id:
            raise ValueError(
                "live-a state save called with task_id={} but "
                "task_id is {}".format(task_id, self.task_id)
            )
        self.task_id += 1
        if not os.path.exists(filename):
            os.makedirs(filename)
        aggregate_up = []
        g_sum = 0.0
        a_sum = 0.0
        b_sum = 0.0
        scale_value = None
        gamma_sum = 0.0
        norm_product_sum = 0.0
        consolidation_gain_sum = 0.0
        consolidation_gain_min = float("inf")
        consolidation_gain_max = 0.0
        absorption_error = 0.0
        coordinate_diagnostics = []
        for idx, (w_a, w_b) in enumerate(zip(self.w_As, self.w_Bs)):
            b = w_b.weight.detach().cpu().float()
            current_a = w_a.weight.detach().cpu().float()
            s = (
                self.wrapped_param[0]
                .param.detach()
                .cpu()
                .float()
                .reshape(())
            )
            g_old = (
                self.aggregate_up[idx]
                if idx < len(self.aggregate_up)
                else torch.zeros_like(b)
            )
            g_history = g_old
            if self.live_a_coordinate_align and idx < len(
                self._live_a_previous_shared_a
            ):
                g_history, diagnostics = align_live_a_aggregate(
                    g_old,
                    self._live_a_previous_shared_a[idx],
                    current_a,
                )
                coordinate_diagnostics.append(diagnostics)
            current_up, absorption = absorb_live_a_current_projection(
                current_a,
                b,
                s,
                mode=self.live_a_absorb_mode,
            )
            g_new = g_history + current_up
            aggregate_up.append(g_new)
            g_sum = g_sum + g_new.square().sum().item()
            a_sum = a_sum + w_a.weight.detach().square().sum().item()
            b_sum = b_sum + b.square().sum().item()
            scale_value = float(s)
            gamma_sum += absorption["gamma"]
            norm_product_sum += absorption["norm_product"]
            consolidation_gain_sum += absorption["consolidation_gain"]
            consolidation_gain_min = min(
                consolidation_gain_min,
                absorption["consolidation_gain"],
            )
            consolidation_gain_max = max(
                consolidation_gain_max,
                absorption["consolidation_gain"],
            )
            absorption_error = max(
                absorption_error,
                absorption["absorption_relative_error"],
            )
        n = len(aggregate_up)
        self._last_live_a_save_stats = {
            "mean_G_norm": (g_sum / n) ** 0.5,
            "mean_A_norm": (a_sum / n) ** 0.5,
            "mean_B_norm": (b_sum / n) ** 0.5,
            "scale": scale_value,
            "mean_gamma": gamma_sum / n,
            "mean_norm_product": norm_product_sum / n,
            "mean_consolidation_gain": consolidation_gain_sum / n,
            "min_consolidation_gain": consolidation_gain_min,
            "max_consolidation_gain": consolidation_gain_max,
            "absorption_relative_error": absorption_error,
            "absorb_mode": self.live_a_absorb_mode,
        }
        if coordinate_diagnostics:
            self._last_live_a_coordinate_diagnostics = {
                "branches": len(coordinate_diagnostics),
                "before_relative_error": sum(
                    item["before_relative_error"]
                    for item in coordinate_diagnostics
                )
                / len(coordinate_diagnostics),
                "after_relative_error": sum(
                    item["after_relative_error"]
                    for item in coordinate_diagnostics
                )
                / len(coordinate_diagnostics),
                "max_condition": max(
                    item["condition"] for item in coordinate_diagnostics
                ),
            }
        else:
            self._last_live_a_coordinate_diagnostics = {
                "branches": 0,
                "before_relative_error": 0.0,
                "after_relative_error": 0.0,
                "max_condition": 0.0,
            }
        self._last_saved_live_a_aggregate = [
            tensor.detach().clone() for tensor in aggregate_up
        ]
        shared_a = [
            w_a.weight.detach().cpu().float() for w_a in self.w_As
        ]
        wrappers = [
            block.attn.qkv
            for block in self.lora_vit.blocks
            if isinstance(block.attn.qkv, _LiveAAggregateQKV)
        ]
        input_sketches = [wrapper.merged_input_sketch() for wrapper in wrappers]
        state = {
                "version": SA_STATE_VERSION_LIVE_A,
                "task_id": self.task_id,
                "rank": self.rank,
                "merge_mode": SA_MERGE_MODE_LIVE_A_AGGREGATE_B,
                "aggregate_up": aggregate_up,
                "shared_a": shared_a,
                "history_groups": self.live_a_history_groups,
                "coordinate_aligned": self.live_a_coordinate_align,
                "absorb_mode": self.live_a_absorb_mode,
            }
        if self.adaptive_a_strategy == "risk_budgeted":
            state.update(
                {
                "adaptive_a_input_rms": [
                    rms.detach().cpu().float() for rms, _ in input_sketches
                ],
                "adaptive_a_input_counts": torch.tensor(
                    [count for _, count in input_sketches], dtype=torch.float64
                ),
                }
            )
        torch.save(
            state,
            _join_path(filename, SA_STATE_FILENAME),
        )
        self.save_merged_lora(filename)

    def cleanup_per_task_files(self, filename: str) -> None:
        """Delete per-task B files after the final task; merged LoRA remains."""
        if not self.delete_per_task_files:
            return
        for task_id in range(self.task_id):
            path = _join_path(filename, "sa_lora_w_b_{}.pt".format(task_id))
            if os.path.exists(path):
                os.remove(path)

    def save_merged_lora(self, filename: str) -> None:
        """Store the exact combined bank as one B per layer (storage metric)."""
        if not os.path.exists(filename):
            os.makedirs(filename)
        current_task = self.task_id - 1
        if self.cumulative_state:
            if (
                self.cumulative_merge
                == SA_MERGE_MODE_LIVE_A_AGGREGATE_B
            ):
                aggregate_total = []
                if self._last_saved_live_a_aggregate is not None:
                    aggregate_total = [
                        tensor.detach().cpu().float().clone()
                        for tensor in self._last_saved_live_a_aggregate
                    ]
                else:
                    for idx, w_b in enumerate(self.w_Bs):
                        b = w_b.weight.detach().cpu().float()
                        s = (
                            self.wrapped_param[0]
                            .param.detach()
                            .cpu()
                            .float()
                            .reshape(())
                        )
                        g_old = (
                            self.aggregate_up[idx]
                            if idx < len(self.aggregate_up)
                            else torch.zeros_like(b)
                        )
                        current_up, _ = absorb_live_a_current_projection(
                            self.w_As[idx].weight.detach().cpu().float(),
                            b,
                            s,
                            mode=self.live_a_absorb_mode,
                        )
                        g_total = g_old + current_up
                        aggregate_total.append(g_total)
                shared_a = [
                    w.weight.detach().cpu().float() for w in self.w_As
                ]
                merged_b = [
                    g / (torch.linalg.vector_norm(a) + 1e-8)
                    for g, a in zip(aggregate_total, shared_a)
                ]
                torch.save(
                    {
                        "version": SA_STATE_VERSION_LIVE_A,
                        "merge_mode": SA_MERGE_MODE_LIVE_A_AGGREGATE_B,
                        "shared_a": shared_a,
                        "merged_b": merged_b,
                        "aggregate_up": aggregate_total,
                        "task_id": current_task,
                        "absorb_mode": self.live_a_absorb_mode,
                    },
                    _join_path(filename, SA_MERGED_FILENAME),
                )
                return
            torch.save(
                {
                    "version": (
                        SA_STATE_VERSION_UNION
                        if self.cumulative_merge == SA_MERGE_MODE_UNION_SVD
                        else SA_STATE_VERSION
                    ),
                    "merge_mode": self.cumulative_merge,
                    "shared_a": [q.clone() for q in self.canonical_down],
                    "merged_b": [h.clone() for h in self.cumulative_up],
                    "canonical_down": [q.clone() for q in self.canonical_down],
                    "cumulative_up": [h.clone() for h in self.cumulative_up],
                    "triangular_r": [r.clone() for r in self.triangular_r],
                    "rank": self.cumulative_rank,
                    "task_id": current_task,
                },
                _join_path(filename, SA_MERGED_FILENAME),
            )
            return
        num_saved = len(self.saved_b_tasks)
        merged_b = []
        for idx, w_b in enumerate(self.w_Bs):
            total = torch.zeros(
                w_b.weight.shape, dtype=torch.float32
            )
            a_w = self.w_As[idx].weight.detach().cpu().float()
            norm_a = torch.norm(a_w)
            # All tasks loaded from disk (0..num_saved-1). In the post-training
            # state the current task is not yet in saved_b_tasks and its B is
            # non-zero; in a rebuilt eval backbone the current B is zero and the
            # last task is already in saved_b_tasks, so it must not be skipped.
            for task_id in range(num_saved):
                b_i = self.saved_b_tasks[task_id][idx].cpu().float()
                s_i = self.wrapped_param_prev[task_id].param.detach().cpu().float()
                total = total + s_i * b_i / (norm_a * torch.norm(b_i) + 1e-8)
            if torch.any(w_b.weight != 0):
                s_cur = self.wrapped_param[0].param.detach().cpu().float()
                total = total + s_cur * w_b.weight.detach().cpu().float()
            merged_b.append(total.cpu())
        torch.save(
            {
                "version": SA_STATE_VERSION_LEGACY,
                "shared_a": [w.weight.detach().cpu() for w in self.w_As],
                "merged_b": merged_b,
                "task_id": current_task,
            },
            _join_path(filename, SA_MERGED_FILENAME),
        )

    def forward(self, x: Tensor, loss=False, eval=False) -> Tensor:
        if loss:
            return self.lora_vit(x), torch.tensor(0.0, device=x.device)
        return self.lora_vit(x)
