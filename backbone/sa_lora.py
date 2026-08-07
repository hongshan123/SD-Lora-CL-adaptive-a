"""Shared-A SD-LoRA backbone.

Each task trains only the up-projection B (plus a scalar scale) against a
task-invariant down-projection A.  Because A is shared, the final model can
store a single merged B per layer:

    B* = sum_{i<t} (s_i / (||A|| * ||B_i||)) * B_i  +  s_t * B_t

which reproduces the exact forward of the per-task bank at evaluation time.
"""

import math
import os
from collections.abc import Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer as timm_ViT
from torch import Tensor

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
SA_STATE_FILENAME = "sa_state.pt"
SA_MERGED_FILENAME = "sa_merged_lora.pt"


def _join_path(prefix, name):
    return os.path.join(prefix, name)


def _fixed_orthogonal_down(in_dim, target_rank, seed, dtype=torch.float32):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    random_matrix = torch.randn(in_dim, target_rank, generator=generator)
    q, _ = torch.linalg.qr(random_matrix, mode="reduced")
    return q.t().contiguous().to(dtype)


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
        new_q = self._norm_live_a(x, self.a_q.weight, self.aggregate_q)
        new_v = self._norm_live_a(x, self.a_v.weight, self.aggregate_v)
        new_q = new_q + self.scaling_cur[0](self.b_q(self.a_q(x)))
        new_v = new_v + self.scaling_cur[0](self.b_v(self.a_v(x)))
        qkv = self.qkv(x)
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv


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
        self.save_file = filepath
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
            self.cumulative_state = True
        elif state_version == -1:
            # Fresh run: the flag decides which version new artifacts use.
            pass
        else:
            raise ValueError(
                "unsupported shared-A state version {}".format(state_version)
            )

        shared_a = state.get("shared_a", [])
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
        self._operator_reference_down = []
        self._operator_reference_up = []
        self._operator_reference_task_count = 0
        # Diagnostics computed from the *pre-save* historical state; populated
        # by _save_cumulative_state before the state is overwritten.
        self._last_cumulative_gauge_diagnostics = None
        self._last_union_svd_truncation_error = None

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

        Writes ``G_next = G_old + s_t B_t / ||B_t||`` plus the current shared
        A to disk, but deliberately keeps the in-memory ``aggregate_up`` as the
        *old* G so that non-final evaluation after save still uses the v1
        semantics ``old G + current raw B``.  The next task's rebuild loads
        ``G_next`` with a fresh zero B.
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
        for idx, (w_a, w_b) in enumerate(zip(self.w_As, self.w_Bs)):
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
            g_new = g_old + s * b / (
                torch.linalg.vector_norm(b) + 1e-8
            )
            aggregate_up.append(g_new)
        shared_a = [
            w_a.weight.detach().cpu().float() for w_a in self.w_As
        ]
        torch.save(
            {
                "version": SA_STATE_VERSION_LIVE_A,
                "task_id": self.task_id,
                "rank": self.rank,
                "merge_mode": SA_MERGE_MODE_LIVE_A_AGGREGATE_B,
                "aggregate_up": aggregate_up,
                "shared_a": shared_a,
                "history_groups": self.live_a_history_groups,
            },
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
                    g_total = g_old + s * b / (
                        torch.linalg.vector_norm(b) + 1e-8
                    )
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
