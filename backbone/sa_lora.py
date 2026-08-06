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
    its own normalized LoRA branch ``scale * B(A x) / (||A|| ||B||)``.
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

    def _norm_cur(self, x, a_weight, b_weight):
        denom = torch.norm(a_weight) * torch.norm(b_weight) + 1e-8
        return F.linear(F.linear(x, a_weight), b_weight) / denom

    def forward(self, x):
        new_q = F.linear(F.linear(x, self.q_q_t), self.h_q)
        new_v = F.linear(F.linear(x, self.q_v_t), self.h_v)
        new_q = new_q + self.scaling_cur[0](
            self._norm_cur(x, self.a_q.weight, self.b_q.weight)
        )
        new_v = new_v + self.scaling_cur[0](
            self._norm_cur(x, self.a_v.weight, self.b_v.weight)
        )
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
    ):
        super().__init__()
        assert r > 0
        self.rank = r
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
                    "sa_cumulative_state=True but artifact is legacy v1; "
                    "run scripts/migrate_sa_state_v1_to_v2.py first"
                )
            self.cumulative_state = False
        elif state_version == SA_STATE_VERSION:
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
            expected_branches = 2 * len(self.lora_layer)
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

    def _load_state(self):
        path = _join_path(self.save_file, SA_STATE_FILENAME)
        if os.path.exists(path):
            state = torch.load(path, map_location="cpu", weights_only=True)
            version = int(state.get("version", -1))
            if version not in (SA_STATE_VERSION_LEGACY, SA_STATE_VERSION):
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
        canonical_down = []
        cumulative_up = []
        triangular_r = []
        for idx, (w_a, w_b) in enumerate(zip(self.w_As, self.w_Bs)):
            a = w_a.weight.detach().cpu().float()
            b = w_b.weight.detach().cpu().float()
            s = self.wrapped_param[0].param.detach().cpu().float().reshape(())
            q_t, r = canonical_down_projection(a)
            if idx < len(self.cumulative_up) and self.task_id > 1:
                if self.cumulative_gauge:
                    h_hist = gauge_align_up_projection(
                        self.cumulative_up[idx],
                        self.canonical_down[idx],
                        q_t,
                    )
                else:
                    # Ablation: keep the old cumulative up projection without
                    # aligning it to the new canonical basis (cumulative only).
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
        torch.save(
            {
                "version": SA_STATE_VERSION,
                "task_id": self.task_id,
                "rank": self.rank,
                "canonical_down": canonical_down,
                "cumulative_up": cumulative_up,
                "triangular_r": triangular_r,
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
            torch.save(
                {
                    "version": SA_STATE_VERSION,
                    "shared_a": [q.clone() for q in self.canonical_down],
                    "merged_b": [h.clone() for h in self.cumulative_up],
                    "canonical_down": [q.clone() for q in self.canonical_down],
                    "cumulative_up": [h.clone() for h in self.cumulative_up],
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
