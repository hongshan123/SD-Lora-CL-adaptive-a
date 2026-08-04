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


def _join_path(prefix, name):
    return os.path.join(prefix, name)


def _linear_lora(x, a_weight, b_weight):
    return F.linear(F.linear(x, a_weight), b_weight)


def _normalized_lora(x, a_weight, b_weight, eps=1e-8):
    denom = torch.norm(a_weight) * torch.norm(b_weight) + eps
    return _linear_lora(x, a_weight, b_weight) / denom


def _svd_merge(low_a, low_b, new_a, new_b, target_rank):
    if low_a is None or low_b is None:
        low_a = torch.zeros(
            target_rank, new_a.shape[1], dtype=new_a.dtype, device=new_a.device
        )
        low_b = torch.zeros(
            new_b.shape[0], target_rank, dtype=new_b.dtype, device=new_b.device
        )

    cat_a = torch.cat([low_a, new_a], dim=0)
    cat_b = torch.cat([low_b, new_b], dim=1)

    q_a, r_a = torch.linalg.qr(cat_a.T)
    q_b, r_b = torch.linalg.qr(cat_b)
    core = r_b @ r_a.T
    u_core, s_core, vh_core = torch.linalg.svd(core, full_matrices=False)

    u_r = u_core[:, :target_rank]
    s_r = s_core[:target_rank]
    vh_r = vh_core[:target_rank, :]
    sqrt_s = torch.diag(torch.sqrt(s_r + 1e-8))

    merged_a = sqrt_s @ vh_r @ q_a.T
    merged_b = q_b @ u_r @ sqrt_s
    return merged_a.to(new_a.dtype), merged_b.to(new_b.dtype)


def _delta_to_lora(delta, target_rank):
    u, s, vh = torch.linalg.svd(delta.float(), full_matrices=False)
    rank = min(target_rank, s.shape[0])
    u_r = u[:, :rank]
    s_r = s[:rank]
    vh_r = vh[:rank, :]
    sqrt_s = torch.diag(torch.sqrt(s_r + 1e-8))

    merged_a = sqrt_s @ vh_r
    merged_b = u_r @ sqrt_s
    return merged_a.to(delta.dtype), merged_b.to(delta.dtype)


def _scalar_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone().float().view(1)
    return torch.Tensor([float(value)])


def _orthogonalize_delta(new_delta, reference_delta, strength=1.0, eps=1e-8):
    ref = reference_delta.flatten()
    new = new_delta.flatten()
    denom = torch.dot(ref, ref) + eps
    projection = torch.dot(new, ref) / denom * reference_delta
    return new_delta - strength * projection


def _decoupled_delta_merge(old_delta, new_delta, old_count, eps=1e-8):
    total = float(old_count + 1)
    old_weight = float(old_count) / total
    new_weight = 1.0 / total

    old_magnitude = old_delta.abs()
    new_magnitude = new_delta.abs()
    old_direction = old_delta / (old_magnitude + eps)
    new_direction = new_delta / (new_magnitude + eps)

    merged_magnitude = old_weight * old_magnitude + new_weight * new_magnitude
    merged_direction = old_weight * old_direction + new_weight * new_direction
    return merged_magnitude * merged_direction


def _do_merge(
    low_a,
    low_b,
    new_a,
    new_b,
    target_rank,
    mode="svd",
    old_count=1,
    orthogonal_strength=1.0,
    old_scale=1.0,
    new_scale=1.0,
    scale_merge_mode="separate",
):
    if scale_merge_mode == "separate" and (
        mode == "svd" or low_a is None or low_b is None
    ):
        return _svd_merge(low_a, low_b, new_a, new_b, target_rank)

    new_delta = new_b.float() @ new_a.float()
    if low_a is None or low_b is None:
        if scale_merge_mode == "effective_delta":
            new_delta = _scalar_tensor(new_scale).item() * new_delta
        return _delta_to_lora(new_delta, target_rank)

    old_delta = low_b.float() @ low_a.float()
    if scale_merge_mode == "effective_delta":
        old_delta = _scalar_tensor(old_scale).item() * old_delta
        new_delta = _scalar_tensor(new_scale).item() * new_delta

    if mode == "svd":
        return _delta_to_lora(old_delta + new_delta, target_rank)

    if mode in {"orthogonalize", "do"}:
        new_delta = _orthogonalize_delta(
            new_delta,
            old_delta,
            strength=orthogonal_strength,
        )

    if mode in {"decouple", "do"}:
        merged_delta = _decoupled_delta_merge(old_delta, new_delta, old_count)
    elif mode == "orthogonalize":
        merged_delta = old_delta + new_delta
    else:
        raise ValueError("Unknown cms_merge_mode: {}".format(mode))

    return _delta_to_lora(merged_delta, target_rank)


class _CMSLoRAQKVTrain(nn.Module):
    def __init__(
        self,
        qkv,
        linear_a_q,
        linear_b_q,
        linear_a_v,
        linear_b_v,
        residual_task_ids,
        saved_a,
        saved_b,
        layer_index,
        rank,
        scaling_factor,
        scaling_factor_prev,
        low_a_q=None,
        low_b_q=None,
        low_a_v=None,
        low_b_v=None,
        low_scale=None,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q.cuda()
        self.linear_b_q = linear_b_q.cuda()
        self.linear_a_v = linear_a_v.cuda()
        self.linear_b_v = linear_b_v.cuda()
        self.scaling_factor = scaling_factor.cuda()
        self.scaling_factor_prev = scaling_factor_prev.cuda()
        self.low_scale = low_scale.cuda()

        self.residual_task_ids = list(residual_task_ids)
        self.saved_a = saved_a
        self.saved_b = saved_b
        self.layer_index = layer_index
        self.rank = rank
        self.dim = qkv.in_features

        self._register_optional_buffer("low_a_q", low_a_q)
        self._register_optional_buffer("low_b_q", low_b_q)
        self._register_optional_buffer("low_a_v", low_a_v)
        self._register_optional_buffer("low_b_v", low_b_v)

    def _register_optional_buffer(self, name, value):
        if value is None:
            self.register_buffer(name, None)
        else:
            self.register_buffer(name, value.detach().clone().float())

    def _low_out(self, x, a_name, b_name):
        a_weight = getattr(self, a_name)
        b_weight = getattr(self, b_name)
        if a_weight is None or b_weight is None:
            return 0
        return self.low_scale[0](_normalized_lora(x, a_weight, b_weight))

    def forward(self, x):
        new_q = self._low_out(x, "low_a_q", "low_b_q")
        new_v = self._low_out(x, "low_a_v", "low_b_v")

        for idx, task_id in enumerate(self.residual_task_ids):
            saved_a_i = self.saved_a["saved_A_" + str(task_id)]
            saved_b_i = self.saved_b["saved_B_" + str(task_id)]
            q_pair, v_pair = list(enumerate(zip(saved_a_i, saved_b_i)))[
                self.layer_index * 2 : self.layer_index * 2 + 2
            ]
            _, (a_q, b_q) = q_pair
            _, (a_v, b_v) = v_pair
            scale_idx = min(idx, len(self.scaling_factor_prev) - 1)

            a_q_w = a_q.weight.to(x.device)
            b_q_w = b_q.weight.to(x.device)
            a_v_w = a_v.weight.to(x.device)
            b_v_w = b_v.weight.to(x.device)
            new_q = new_q + self.scaling_factor_prev[scale_idx](
                _normalized_lora(x, a_q_w, b_q_w)
            )
            new_v = new_v + self.scaling_factor_prev[scale_idx](
                _normalized_lora(x, a_v_w, b_v_w)
            )

        new_q = new_q + self.scaling_factor[0](self.linear_b_q(self.linear_a_q(x)))
        new_v = new_v + self.scaling_factor[0](self.linear_b_v(self.linear_a_v(x)))

        qkv = self.qkv(x)
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv


class CMSLoRA_ViT_timm(nn.Module):
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
        cms_recent_tasks=2,
        cms_low_rank=None,
        cms_low_scale_init=1.0,
        cms_merge_mode="svd",
        cms_orthogonal_strength=1.0,
        cms_scale_merge_mode="separate",
    ):
        super(CMSLoRA_ViT_timm, self).__init__()
        assert r > 0

        self.rank = r
        self.cms_recent_tasks = cms_recent_tasks
        self.cms_low_rank = cms_low_rank or r
        self.cms_merge_mode = cms_merge_mode
        self.cms_orthogonal_strength = cms_orthogonal_strength
        self.cms_scale_merge_mode = cms_scale_merge_mode
        self.save_file = filepath
        self.increment = increment
        self.base_vit = vit_model

        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.blocks)))

        if index:
            print("Initialize task-id and curtask id")
            self.task_id, self.cur_id = 0, 0

        if cur_task_index is not None:
            self.task_id = cur_task_index

        for param in vit_model.parameters():
            param.requires_grad = False

        state = self._load_cms_state()
        self.consolidated_until = min(state["consolidated_until"], self.task_id)
        low_a, low_b = state["low_A"], state["low_B"]
        low_scale_value = (
            state["low_scale"]
            if self.cms_scale_merge_mode == "separate"
            else torch.Tensor([1.0])
        )

        self.residual_task_ids = list(range(self.consolidated_until, self.task_id))
        task_scales = state["task_scales"]
        saved_lora_a, saved_lora_b = {}, {}
        for task_id in self.residual_task_ids:
            saved_lora_a["saved_A_" + str(task_id)] = torch.load(
                _join_path(self.save_file, "lora_w_a_" + str(task_id) + ".pt")
            )
            saved_lora_b["saved_B_" + str(task_id)] = torch.load(
                _join_path(self.save_file, "lora_w_b_" + str(task_id) + ".pt")
            )

        scaling_factor = nn.Parameter(torch.Tensor([0.8]))
        low_scale = nn.Parameter(_scalar_tensor(low_scale_value))
        self.wrapped_param = nn.ModuleList([ParameterWrapper(scaling_factor)])
        self.low_scale = nn.ModuleList([ParameterWrapper(low_scale)])
        residual_scale_values = [
            task_scales.get(task_id, torch.Tensor([0.8]))
            for task_id in self.residual_task_ids
        ]
        residual_scale_values.extend(
            [torch.Tensor([0.8]) for _ in range(max(0, 20 - len(residual_scale_values)))]
        )
        self.wrapped_param_prev = nn.ModuleList(
            [
                ParameterWrapper(nn.Parameter(_scalar_tensor(scale)))
                for scale in residual_scale_values
            ]
        )

        self.w_As, self.w_Bs = [], []
        for layer_index, blk in enumerate(vit_model.blocks):
            if layer_index not in self.lora_layer:
                continue
            qkv = blk.attn.qkv
            dim = qkv.in_features
            w_a_linear_q = nn.Linear(dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, dim, bias=False)
            w_a_linear_v = nn.Linear(dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, dim, bias=False)

            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)

            low_q_idx = layer_index * 2
            low_v_idx = layer_index * 2 + 1
            blk.attn.qkv = _CMSLoRAQKVTrain(
                qkv,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
                self.residual_task_ids,
                saved_lora_a,
                saved_lora_b,
                layer_index,
                self.rank,
                self.wrapped_param,
                self.wrapped_param_prev,
                low_a[low_q_idx] if low_a is not None else None,
                low_b[low_q_idx] if low_b is not None else None,
                low_a[low_v_idx] if low_a is not None else None,
                low_b[low_v_idx] if low_b is not None else None,
                self.low_scale,
            )

        self.reset_parameters()
        self.lora_vit = vit_model
        self.lora_vit.head = torch.nn.Identity()
        self.out_dim = 768

    def _load_cms_state(self):
        path = _join_path(self.save_file, "cms_state.pt")
        if os.path.exists(path):
            state = torch.load(path, map_location="cpu")
            return {
                "consolidated_until": int(state.get("consolidated_until", 0)),
                "low_A": state.get("low_A", None),
                "low_B": state.get("low_B", None),
                "merge_counts": state.get("merge_counts", None),
                "task_scales": state.get("task_scales", {}),
                "low_scale": state.get("low_scale", torch.Tensor([1.0])),
            }
        return {
            "consolidated_until": 0,
            "low_A": None,
            "low_B": None,
            "merge_counts": None,
            "task_scales": {},
            "low_scale": torch.Tensor([1.0]),
        }

    def reset_parameters(self):
        for w_a in self.w_As:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_Bs:
            nn.init.zeros_(w_b.weight)

    def generate_fc(self, in_dim, out_dim):
        return SimpleLinear(in_dim, out_dim)

    def _consolidate_old_tasks(self, filename, current_task_id):
        total_saved = current_task_id + 1
        cutoff = max(0, total_saved - self.cms_recent_tasks)
        state = self._load_cms_state()
        start = min(state["consolidated_until"], cutoff)
        low_a = state["low_A"]
        low_b = state["low_B"]
        merge_counts = state["merge_counts"]
        task_scales = state["task_scales"]
        old_low_scale = state["low_scale"]

        if start >= cutoff:
            return

        for task_id in range(start, cutoff):
            task_a = torch.load(
                _join_path(filename, "lora_w_a_" + str(task_id) + ".pt"),
                map_location="cpu",
            )
            task_b = torch.load(
                _join_path(filename, "lora_w_b_" + str(task_id) + ".pt"),
                map_location="cpu",
            )
            task_scale = task_scales.get(task_id, torch.Tensor([0.8]))
            if low_a is None or low_b is None:
                low_a = [None for _ in range(len(task_a))]
                low_b = [None for _ in range(len(task_b))]
            if merge_counts is None:
                merge_counts = [0 for _ in range(len(task_a))]
            for idx, (a_layer, b_layer) in enumerate(zip(task_a, task_b)):
                merged_a, merged_b = _do_merge(
                    low_a[idx],
                    low_b[idx],
                    a_layer.weight.detach().float(),
                    b_layer.weight.detach().float(),
                    self.cms_low_rank,
                    mode=self.cms_merge_mode,
                    old_count=max(merge_counts[idx], 1),
                    orthogonal_strength=self.cms_orthogonal_strength,
                    old_scale=old_low_scale,
                    new_scale=task_scale,
                    scale_merge_mode=self.cms_scale_merge_mode,
                )
                low_a[idx] = merged_a.cpu()
                low_b[idx] = merged_b.cpu()
                merge_counts[idx] += 1
            if self.cms_scale_merge_mode == "effective_delta":
                old_low_scale = torch.Tensor([1.0])

        if not os.path.exists(filename):
            os.makedirs(filename)
        saved_low_scale = (
            self.low_scale[0].param.detach().cpu()
            if self.cms_scale_merge_mode == "separate"
            else torch.Tensor([1.0])
        )
        torch.save(
            {
                "consolidated_until": cutoff,
                "low_A": low_a,
                "low_B": low_b,
                "merge_counts": merge_counts,
                "task_scales": task_scales,
                "low_scale": saved_low_scale,
                "merge_mode": self.cms_merge_mode,
                "orthogonal_strength": self.cms_orthogonal_strength,
                "scale_merge_mode": self.cms_scale_merge_mode,
            },
            _join_path(filename, "cms_state.pt"),
        )
        print(
            "[CMS-SDLoRA] Consolidated tasks {}-{} into low memory with {} merge.".format(
                start, cutoff - 1, self.cms_merge_mode
            )
        )

    def save_lora_parameters(self, filename: str, task_id) -> None:
        self.task_id += 1
        if not os.path.exists(filename):
            os.makedirs(filename)
        torch.save(self.w_As, _join_path(filename, "lora_w_a_" + str(task_id) + ".pt"))
        torch.save(self.w_Bs, _join_path(filename, "lora_w_b_" + str(task_id) + ".pt"))
        state = self._load_cms_state()
        for idx, residual_task_id in enumerate(self.residual_task_ids):
            state["task_scales"][residual_task_id] = (
                self.wrapped_param_prev[idx].param.detach().cpu()
            )
        state["task_scales"][task_id] = self.wrapped_param[0].param.detach().cpu()
        if self.cms_scale_merge_mode == "separate":
            state["low_scale"] = self.low_scale[0].param.detach().cpu()
        else:
            state["low_scale"] = torch.Tensor([1.0])
        torch.save(state, _join_path(filename, "cms_state.pt"))
        self._consolidate_old_tasks(filename, task_id)

    def forward(self, x: Tensor, loss=False, eval=False) -> Tensor:
        if loss:
            return self.lora_vit(x), torch.tensor(0.0, device=x.device)
        return self.lora_vit(x)
