"""Shared-A SD-LoRA backbone.

Each task trains only the up-projection B (plus a scalar scale) against a
task-invariant down-projection A.  Because A is shared, the final model can
store a single merged B per layer:

    B* = sum_{i<t} (s_i / (||A|| * ||B_i||)) * B_i  +  s_t * B_t

which reproduces the exact forward of the per-task bank at evaluation time.
"""

import math
import os

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer as timm_ViT
from torch import Tensor

from backbone.linears import SimpleLinear
from backbone.lora import ParameterWrapper


SA_STATE_VERSION = 1
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
        delete_per_task_files=False,
    ):
        super().__init__()
        assert r > 0
        self.rank = r
        self.save_file = filepath
        self.increment = increment
        self.shared_a_orthogonal = bool(shared_a_orthogonal)
        self.delete_per_task_files = bool(delete_per_task_files)
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
        shared_a = state.get("shared_a", [])
        if self.task_id > 0 and len(shared_a) == 0:
            raise FileNotFoundError(
                "{} is required before training task {}".format(
                    _join_path(self.save_file, SA_STATE_FILENAME), self.task_id
                )
            )

        self.saved_b_tasks = {}
        for task_id in range(self.task_id):
            path = _join_path(
                self.save_file, "sa_lora_w_b_{}.pt".format(task_id)
            )
            if not os.path.exists(path):
                raise FileNotFoundError("missing saved shared-A B for task {}".format(task_id))
            self.saved_b_tasks[task_id] = torch.load(
                path, map_location="cpu", weights_only=True
            )

        scaling_factor = nn.Parameter(torch.Tensor([0.8]))
        self.wrapped_param = nn.ModuleList([ParameterWrapper(scaling_factor)])
        saved_scales = state.get("scales", {})
        residual_scale_values = [
            saved_scales.get(task_id, torch.Tensor([0.8]))
            for task_id in range(self.task_id)
        ]
        residual_scale_values.extend(
            [torch.Tensor([0.8]) for _ in range(max(0, 20 - len(residual_scale_values)))]
        )
        self.wrapped_param_prev = nn.ModuleList(
            [
                ParameterWrapper(nn.Parameter(value.detach().clone().float()))
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

            if self.task_id > 0:
                a_q.weight.data.copy_(shared_a[offset].to(a_q.weight.dtype))
                a_v.weight.data.copy_(shared_a[offset + 1].to(a_v.weight.dtype))
                a_q.weight.requires_grad_(False)
                a_v.weight.requires_grad_(False)
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
        self.lora_vit = vit_model
        self.lora_vit.head = nn.Identity()
        self.out_dim = 768

    def _load_state(self):
        path = _join_path(self.save_file, SA_STATE_FILENAME)
        if os.path.exists(path):
            state = torch.load(path, map_location="cpu", weights_only=True)
            if int(state.get("version", -1)) != SA_STATE_VERSION:
                raise ValueError("unsupported shared-A state version")
            return state
        return {"version": SA_STATE_VERSION, "shared_a": [], "scales": {}}

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
                "version": SA_STATE_VERSION,
                "shared_a": [w.weight.detach().cpu() for w in self.w_As],
                "scales": scales,
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
        merged_b = []
        for idx, w_b in enumerate(self.w_Bs):
            total = torch.zeros(
                w_b.weight.shape, dtype=torch.float32
            )
            a_w = self.w_As[idx].weight.detach().cpu().float()
            norm_a = torch.norm(a_w)
            for task_id in range(current_task):
                b_i = self.saved_b_tasks[task_id][idx].cpu().float()
                s_i = self.wrapped_param_prev[task_id].param.detach().float()
                total = total + s_i * b_i / (norm_a * torch.norm(b_i) + 1e-8)
            s_cur = self.wrapped_param[0].param.detach().float()
            total = total + s_cur * w_b.weight.detach().cpu().float()
            merged_b.append(total.cpu())
        torch.save(
            {
                "version": SA_STATE_VERSION,
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
