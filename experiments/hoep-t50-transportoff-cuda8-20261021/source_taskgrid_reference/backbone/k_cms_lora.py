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


def _svd_merge_weighted(
    low_a, low_b, new_a, new_b, target_rank, old_weight=1.0, new_weight=1.0
):
    if low_a is None or low_b is None:
        return _svd_merge(low_a, low_b, new_a, new_b, target_rank)

    old_scale = math.sqrt(float(old_weight))
    new_scale = math.sqrt(float(new_weight))
    cat_a = torch.cat([low_a * old_scale, new_a * new_scale], dim=0)
    cat_b = torch.cat([low_b * old_scale, new_b * new_scale], dim=1)

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


def _fixed_orthogonal_down(in_dim, target_rank, seed, dtype=torch.float32):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    random_matrix = torch.randn(in_dim, target_rank, generator=generator)
    q, _ = torch.linalg.qr(random_matrix, mode="reduced")
    return q.T.contiguous().to(dtype)


def _delta_to_fixed_down_lora(delta, target_rank, seed, ref_a=None):
    if ref_a is None:
        down = _fixed_orthogonal_down(
            delta.shape[1], target_rank, seed, dtype=delta.dtype
        ).to(delta.device)
    else:
        down = ref_a.detach().clone().to(delta.device).to(delta.dtype)
    up = delta.float() @ down.float().T
    return down.to(delta.dtype), up.to(delta.dtype)


def _svd_merge_weighted_effective_delta(
    low_a,
    low_b,
    new_a,
    new_b,
    target_rank,
    old_weight=1.0,
    new_weight=1.0,
    old_scale=1.0,
    new_scale=1.0,
):
    new_delta = _scalar_tensor(new_scale).item() * (new_b.float() @ new_a.float())
    if low_a is None or low_b is None:
        return _delta_to_lora(new_delta, target_rank)

    old_delta = _scalar_tensor(old_scale).item() * (low_b.float() @ low_a.float())
    merged_delta = float(old_weight) * old_delta + float(new_weight) * new_delta
    return _delta_to_lora(merged_delta, target_rank)


def _delta_cosine(a_1, b_1, a_2, b_2, eps=1e-8):
    delta_1 = b_1 @ a_1
    delta_2 = b_2 @ a_2
    return F.cosine_similarity(delta_1.flatten(), delta_2.flatten(), dim=0, eps=eps)


def _task_cluster_similarity(task_a, task_b, cluster):
    sims = []
    for idx, (a_layer, b_layer) in enumerate(zip(task_a, task_b)):
        low_a = cluster["low_A"][idx]
        low_b = cluster["low_B"][idx]
        if low_a is None or low_b is None:
            continue
        sims.append(
            _delta_cosine(
                a_layer.weight.detach().float(),
                b_layer.weight.detach().float(),
                low_a.float(),
                low_b.float(),
            )
        )
    if len(sims) == 0:
        return float("-inf")
    return torch.stack(sims).mean().item()


def _float_vector(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone().float().view(-1)
    return torch.tensor([float(value)], dtype=torch.float32)


def _scalar_tensor(value):
    value = _float_vector(value)
    if value.numel() != 1:
        raise ValueError(
            "Expected a scalar scale value, but got {} values.".format(value.numel())
        )
    return value.view(1)


def _scale_vector(value, length):
    value = _float_vector(value)
    if value.numel() == 1:
        return value.repeat(length)
    if value.numel() < length:
        return value.mean().repeat(length)
    return value[:length]


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
        low_a_q_clusters=None,
        low_b_q_clusters=None,
        low_a_v_clusters=None,
        low_b_v_clusters=None,
        low_scale=None,
        lora_pair_offset=None,
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
        self.lora_pair_offset = (
            layer_index * 2 if lora_pair_offset is None else int(lora_pair_offset)
        )
        self.rank = rank
        self.dim = qkv.in_features

        self.num_low_clusters = max(
            len(low_a_q_clusters or []),
            len(low_a_v_clusters or []),
        )
        self._register_cluster_buffers("low_a_q", low_a_q_clusters)
        self._register_cluster_buffers("low_b_q", low_b_q_clusters)
        self._register_cluster_buffers("low_a_v", low_a_v_clusters)
        self._register_cluster_buffers("low_b_v", low_b_v_clusters)

    def _register_cluster_buffers(self, name, values):
        for idx, value in enumerate(values or []):
            self.register_buffer(
                "{}_{}".format(name, idx),
                value.detach().clone().float(),
            )

    def _low_scale_value(self, idx, x):
        scale_module = self.low_scale[min(idx, len(self.low_scale) - 1)]
        scale = scale_module.param
        if scale.numel() > 1:
            scale = scale[min(self.layer_index, scale.numel() - 1)].view(1)
        return scale.to(x.device)

    def _low_out(self, x, a_name, b_name):
        out = 0
        for idx in range(self.num_low_clusters):
            a_weight = getattr(self, "{}_{}".format(a_name, idx), None)
            b_weight = getattr(self, "{}_{}".format(b_name, idx), None)
            if a_weight is None or b_weight is None:
                continue
            out = out + self._low_scale_value(idx, x) * _normalized_lora(
                x, a_weight, b_weight
            )
        return out

    def forward(self, x):
        new_q = self._low_out(x, "low_a_q", "low_b_q")
        new_v = self._low_out(x, "low_a_v", "low_b_v")

        for idx, task_id in enumerate(self.residual_task_ids):
            saved_a_i = self.saved_a["saved_A_" + str(task_id)]
            saved_b_i = self.saved_b["saved_B_" + str(task_id)]
            a_q = saved_a_i[self.lora_pair_offset]
            b_q = saved_b_i[self.lora_pair_offset]
            a_v = saved_a_i[self.lora_pair_offset + 1]
            b_v = saved_b_i[self.lora_pair_offset + 1]
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


class KCMSLoRA_ViT_timm(nn.Module):
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
        k_cms_clusters=4,
        k_cms_similarity_threshold=None,
        k_cms_threshold_mode="none",
        k_cms_adaptive_quantile=0.5,
        k_cms_min_threshold_samples=3,
        k_cms_balance_lambda=0.0,
        k_cms_age_lambda=0.0,
        k_cms_low_sim_keep_recent=False,
        k_cms_low_sim_recent_limit=0,
        k_cms_reallocate_on_low_sim=False,
        k_cms_verbose_merge=False,
        k_cms_hard_capacity=False,
        k_cms_max_cluster_size=0,
        k_cms_anchor_protection=False,
        k_cms_anchor_tasks=None,
        k_cms_anchor_margin=0.0,
        k_cms_blockwise_cluster_scale=False,
        k_cms_fixed_orthogonal_down_layers=0,
        k_cms_shared_lora_layers=0,
        k_cms_shared_fixed_orthogonal_down=True,
        k_cms_shared_freeze_down_after_task0=False,
        k_cms_delete_consolidated_files=False,
        cms_scale_merge_mode="separate",
    ):
        super(KCMSLoRA_ViT_timm, self).__init__()
        assert r > 0

        self.rank = r
        self.cms_recent_tasks = cms_recent_tasks
        self.cms_low_rank = cms_low_rank or r
        self.k_cms_clusters = k_cms_clusters
        self.k_cms_similarity_threshold = k_cms_similarity_threshold
        self.k_cms_threshold_mode = k_cms_threshold_mode
        self.k_cms_adaptive_quantile = k_cms_adaptive_quantile
        self.k_cms_min_threshold_samples = k_cms_min_threshold_samples
        self.k_cms_balance_lambda = float(k_cms_balance_lambda or 0.0)
        self.k_cms_age_lambda = float(k_cms_age_lambda or 0.0)
        self.k_cms_low_sim_keep_recent = bool(k_cms_low_sim_keep_recent)
        self.k_cms_low_sim_recent_limit = int(k_cms_low_sim_recent_limit or 0)
        self.k_cms_reallocate_on_low_sim = bool(k_cms_reallocate_on_low_sim)
        self.k_cms_verbose_merge = bool(k_cms_verbose_merge)
        self.k_cms_hard_capacity = bool(k_cms_hard_capacity)
        self.k_cms_max_cluster_size = int(k_cms_max_cluster_size or 0)
        self.k_cms_anchor_protection = bool(k_cms_anchor_protection)
        self.k_cms_anchor_tasks = set(k_cms_anchor_tasks or [0])
        self.k_cms_anchor_margin = float(k_cms_anchor_margin or 0.0)
        self.k_cms_blockwise_cluster_scale = bool(k_cms_blockwise_cluster_scale)
        self.k_cms_fixed_orthogonal_down_layers = int(
            k_cms_fixed_orthogonal_down_layers or 0
        )
        self.k_cms_shared_lora_layers = int(k_cms_shared_lora_layers or 0)
        self.k_cms_shared_fixed_orthogonal_down = bool(
            k_cms_shared_fixed_orthogonal_down
        )
        self.k_cms_shared_freeze_down_after_task0 = bool(
            k_cms_shared_freeze_down_after_task0
        )
        self.k_cms_delete_consolidated_files = bool(
            k_cms_delete_consolidated_files
        )
        self.cms_scale_merge_mode = cms_scale_merge_mode
        self.save_file = filepath
        self.increment = increment
        self.base_vit = vit_model

        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.blocks)))
        self.k_cms_shared_layers = set(
            self.lora_layer[: min(self.k_cms_shared_lora_layers, len(self.lora_layer))]
        )

        if index:
            print("Initialize task-id and curtask id")
            self.task_id, self.cur_id = 0, 0

        if cur_task_index is not None:
            self.task_id = cur_task_index

        for param in vit_model.parameters():
            param.requires_grad = False

        state = self._load_k_cms_state()
        self.consolidated_until = min(state["consolidated_until"], self.task_id)
        clusters = state["clusters"]
        task_scales = state["task_scales"]
        deferred_tasks = [
            task_id for task_id in state.get("deferred_tasks", []) if task_id < self.task_id
        ]

        self.residual_task_ids = sorted(
            set(range(self.consolidated_until, self.task_id)).union(deferred_tasks)
        )
        saved_lora_a, saved_lora_b = {}, {}
        for task_id in self.residual_task_ids:
            saved_lora_a["saved_A_" + str(task_id)] = torch.load(
                _join_path(self.save_file, "lora_w_a_" + str(task_id) + ".pt")
            )
            saved_lora_b["saved_B_" + str(task_id)] = torch.load(
                _join_path(self.save_file, "lora_w_b_" + str(task_id) + ".pt")
            )

        initial_scale = (
            state.get("shared_scale", torch.Tensor([0.8]))
            if len(self.k_cms_shared_layers) > 0
            else torch.Tensor([0.8])
        )
        scaling_factor = nn.Parameter(_scalar_tensor(initial_scale))
        self.wrapped_param = nn.ModuleList([ParameterWrapper(scaling_factor)])
        self.low_scale = nn.ModuleList(
            [
                ParameterWrapper(
                    nn.Parameter(
                        self._cluster_scale_tensor(
                            cluster,
                            cms_low_scale_init,
                        )
                    )
                )
                for cluster in clusters
            ]
        )

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
        self.layer_to_lora_offset = {}
        for layer_index, blk in enumerate(vit_model.blocks):
            if layer_index not in self.lora_layer:
                continue
            qkv = blk.attn.qkv
            dim = qkv.in_features
            w_a_linear_q = nn.Linear(dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, dim, bias=False)
            w_a_linear_v = nn.Linear(dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, dim, bias=False)

            lora_offset = len(self.w_As)
            self.layer_to_lora_offset[layer_index] = lora_offset
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)

            low_q_idx = lora_offset
            low_v_idx = lora_offset + 1
            is_shared_layer = self._is_shared_layer(layer_index)
            blk.attn.qkv = _CMSLoRAQKVTrain(
                qkv,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
                [] if is_shared_layer else self.residual_task_ids,
                {} if is_shared_layer else saved_lora_a,
                {} if is_shared_layer else saved_lora_b,
                layer_index,
                self.rank,
                self.wrapped_param,
                self.wrapped_param_prev,
                []
                if is_shared_layer
                else [
                    cluster["low_A"][low_q_idx]
                    for cluster in clusters
                    if low_q_idx < len(cluster["low_A"])
                    and cluster["low_A"][low_q_idx] is not None
                ],
                []
                if is_shared_layer
                else [
                    cluster["low_B"][low_q_idx]
                    for cluster in clusters
                    if low_q_idx < len(cluster["low_B"])
                    and cluster["low_B"][low_q_idx] is not None
                ],
                []
                if is_shared_layer
                else [
                    cluster["low_A"][low_v_idx]
                    for cluster in clusters
                    if low_v_idx < len(cluster["low_A"])
                    and cluster["low_A"][low_v_idx] is not None
                ],
                []
                if is_shared_layer
                else [
                    cluster["low_B"][low_v_idx]
                    for cluster in clusters
                    if low_v_idx < len(cluster["low_B"])
                    and cluster["low_B"][low_v_idx] is not None
                ],
                self.low_scale,
                lora_pair_offset=lora_offset,
            )

        self.reset_parameters()
        self._initialize_shared_lora()
        self.lora_vit = vit_model
        self.lora_vit.head = torch.nn.Identity()
        self.out_dim = 768

    def _load_k_cms_state(self):
        path = _join_path(self.save_file, "k_cms_state.pt")
        if os.path.exists(path):
            state = torch.load(path, map_location="cpu")
            return {
                "consolidated_until": int(state.get("consolidated_until", 0)),
                "clusters": state.get("clusters", []),
                "task_scales": state.get("task_scales", {}),
                "similarity_history": state.get("similarity_history", []),
                "deferred_tasks": state.get("deferred_tasks", []),
                "merge_log": state.get("merge_log", []),
                "shared_scale": state.get("shared_scale", torch.Tensor([0.8])),
            }
        return {
            "consolidated_until": 0,
            "clusters": [],
            "task_scales": {},
            "similarity_history": [],
            "deferred_tasks": [],
            "merge_log": [],
            "shared_scale": torch.Tensor([0.8]),
        }

    def _num_lora_blocks(self):
        return len(self.lora_layer)

    def _is_shared_layer(self, layer_index):
        return int(layer_index) in self.k_cms_shared_layers

    def _is_shared_lora_index(self, lora_index):
        layer_index = self.lora_layer[int(lora_index) // 2]
        return self._is_shared_layer(layer_index)

    def _shared_lora_path(self, filename=None):
        root = self.save_file if filename is None else filename
        return _join_path(root, "k_cms_shared_lora.pt")

    def _load_shared_lora_state(self):
        path = self._shared_lora_path()
        if os.path.exists(path):
            return torch.load(path, map_location="cpu")
        return None

    def _should_freeze_shared_down(self):
        if self.k_cms_shared_freeze_down_after_task0:
            return int(self.task_id) > 0
        return self.k_cms_shared_fixed_orthogonal_down

    def _initialize_shared_lora(self):
        if len(self.k_cms_shared_layers) == 0:
            return

        shared_state = self._load_shared_lora_state()
        saved_a = [] if shared_state is None else shared_state.get("w_A", [])
        saved_b = [] if shared_state is None else shared_state.get("w_B", [])

        for layer_index in sorted(self.k_cms_shared_layers):
            if layer_index not in self.layer_to_lora_offset:
                continue
            for offset in (0, 1):
                lora_idx = self.layer_to_lora_offset[layer_index] + offset
                w_a = self.w_As[lora_idx]
                w_b = self.w_Bs[lora_idx]

                if lora_idx < len(saved_a) and saved_a[lora_idx].shape == w_a.weight.shape:
                    w_a.weight.data.copy_(saved_a[lora_idx].to(w_a.weight.dtype))
                elif self.k_cms_shared_fixed_orthogonal_down:
                    w_a.weight.data.copy_(
                        _fixed_orthogonal_down(
                            w_a.in_features,
                            w_a.out_features,
                            self._fixed_down_seed(lora_idx),
                            dtype=w_a.weight.dtype,
                        )
                    )

                if lora_idx < len(saved_b) and saved_b[lora_idx].shape == w_b.weight.shape:
                    w_b.weight.data.copy_(saved_b[lora_idx].to(w_b.weight.dtype))

                if self._should_freeze_shared_down():
                    w_a.weight.requires_grad_(False)

    def _save_shared_lora_parameters(self, filename):
        if len(self.k_cms_shared_layers) == 0:
            return
        if not os.path.exists(filename):
            os.makedirs(filename)
        torch.save(
            {
                "shared_layers": sorted(self.k_cms_shared_layers),
                "fixed_orthogonal_down": self.k_cms_shared_fixed_orthogonal_down,
                "freeze_down_after_task0": self.k_cms_shared_freeze_down_after_task0,
                "w_A": [w_a.weight.detach().cpu() for w_a in self.w_As],
                "w_B": [w_b.weight.detach().cpu() for w_b in self.w_Bs],
            },
            self._shared_lora_path(filename),
        )

    def _cluster_scale_tensor(self, cluster, default_scale):
        num_lora_blocks = self._num_lora_blocks()
        if self.cms_scale_merge_mode == "effective_delta":
            base = torch.ones(num_lora_blocks, dtype=torch.float32)
        else:
            base_value = cluster.get("scale", torch.tensor([default_scale]))
            base = _scale_vector(base_value, num_lora_blocks)
        if not self.k_cms_blockwise_cluster_scale:
            return _scalar_tensor(base.mean())
        return base.contiguous()

    def _task_scale_for_layer(self, scale, lora_index):
        if isinstance(scale, torch.Tensor) and scale.numel() > 1:
            layer_index = min(lora_index // 2, scale.numel() - 1)
            return scale.detach().float().view(-1)[layer_index]
        return _scalar_tensor(scale).view(1)[0]

    def _initial_cluster_scale(self, scale):
        if self.k_cms_blockwise_cluster_scale:
            return _scale_vector(scale, self._num_lora_blocks()).cpu()
        return _scalar_tensor(scale).cpu()

    def _fixed_down_seed(self, lora_index):
        return 1729 + int(lora_index)

    def _use_fixed_down_for_lora(self, lora_index):
        return (lora_index // 2) < self.k_cms_fixed_orthogonal_down_layers

    def _merge_lora_pair(
        self,
        low_a,
        low_b,
        new_a,
        new_b,
        lora_index,
        old_weight=1.0,
        new_weight=1.0,
        old_scale=1.0,
        new_scale=1.0,
    ):
        if self.cms_scale_merge_mode in ("effective_delta", "scale_weighted"):
            new_delta = self._task_scale_for_layer(new_scale, lora_index) * (
                new_b.float() @ new_a.float()
            )
            if low_a is None or low_b is None:
                merged_delta = float(new_weight) * new_delta
            else:
                old_delta = self._task_scale_for_layer(old_scale, lora_index) * (
                    low_b.float() @ low_a.float()
                )
                merged_delta = float(old_weight) * old_delta + float(new_weight) * new_delta
        else:
            new_delta = new_b.float() @ new_a.float()
            if low_a is None or low_b is None:
                merged_delta = new_delta
            else:
                old_delta = low_b.float() @ low_a.float()
                merged_delta = float(old_weight) * old_delta + float(new_weight) * new_delta

        if self._use_fixed_down_for_lora(lora_index):
            return _delta_to_fixed_down_lora(
                merged_delta,
                self.cms_low_rank,
                self._fixed_down_seed(lora_index),
                ref_a=low_a if low_a is not None else None,
            )

        if self.cms_scale_merge_mode in ("effective_delta", "scale_weighted"):
            return _delta_to_lora(merged_delta, self.cms_low_rank)
        if low_a is None or low_b is None:
            return _svd_merge(None, None, new_a, new_b, self.cms_low_rank)
        return _svd_merge_weighted(
            low_a,
            low_b,
            new_a,
            new_b,
            self.cms_low_rank,
            old_weight=old_weight,
            new_weight=new_weight,
        )

    def reset_parameters(self):
        for w_a in self.w_As:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_Bs:
            nn.init.zeros_(w_b.weight)

    def generate_fc(self, in_dim, out_dim):
        return SimpleLinear(in_dim, out_dim)

    def _create_cluster(self, task_a, task_b, task_id, scale):
        low_a, low_b = [], []
        for idx, (a_layer, b_layer) in enumerate(zip(task_a, task_b)):
            if self._is_shared_lora_index(idx):
                low_a.append(None)
                low_b.append(None)
                continue
            merged_a, merged_b = self._merge_lora_pair(
                None,
                None,
                a_layer.weight.detach().float(),
                b_layer.weight.detach().float(),
                idx,
                new_scale=scale,
            )
            low_a.append(merged_a.cpu())
            low_b.append(merged_b.cpu())
        return {
            "low_A": low_a,
            "low_B": low_b,
            "count": 1,
            "tasks": [task_id],
            "scale": (
                self._initial_cluster_scale(scale)
                if self.cms_scale_merge_mode != "effective_delta"
                else torch.Tensor([1.0])
            ),
        }

    def _merge_task_into_cluster(self, cluster, task_a, task_b, task_id, scale):
        count = int(cluster.get("count", max(1, len(cluster.get("tasks", [])))))
        old_weight = count / float(count + 1)
        new_weight = 1.0 / float(count + 1)
        for idx, (a_layer, b_layer) in enumerate(zip(task_a, task_b)):
            if self._is_shared_lora_index(idx):
                continue
            merged_a, merged_b = self._merge_lora_pair(
                cluster["low_A"][idx],
                cluster["low_B"][idx],
                a_layer.weight.detach().float(),
                b_layer.weight.detach().float(),
                idx,
                old_weight=old_weight,
                new_weight=new_weight,
                old_scale=cluster.get("scale", torch.Tensor([1.0])),
                new_scale=scale,
            )
            cluster["low_A"][idx] = merged_a.cpu()
            cluster["low_B"][idx] = merged_b.cpu()
        cluster["count"] = count + 1
        cluster.setdefault("tasks", []).append(task_id)
        if self.cms_scale_merge_mode == "effective_delta":
            cluster["scale"] = torch.Tensor([1.0])
        else:
            old_scale = self._cluster_scale_tensor(cluster, 1.0)
            new_scale = self._initial_cluster_scale(scale)
            cluster["scale"] = (
                (old_scale * count + new_scale) / float(count + 1)
            ).cpu()

    def _cluster_cluster_similarity(self, cluster_a, cluster_b):
        sims = []
        for low_a_1, low_b_1, low_a_2, low_b_2 in zip(
            cluster_a["low_A"],
            cluster_a["low_B"],
            cluster_b["low_A"],
            cluster_b["low_B"],
        ):
            if low_a_1 is None or low_b_1 is None or low_a_2 is None or low_b_2 is None:
                continue
            sims.append(
                _delta_cosine(
                    low_a_1.float(),
                    low_b_1.float(),
                    low_a_2.float(),
                    low_b_2.float(),
                )
            )
        if len(sims) == 0:
            return float("-inf")
        return torch.stack(sims).mean().item()

    def _find_most_similar_cluster_pair(self, clusters):
        if len(clusters) < 2:
            return None, None, float("-inf")
        best_i, best_j, best_similarity = 0, 1, float("-inf")
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                sim = self._cluster_cluster_similarity(clusters[i], clusters[j])
                if sim > best_similarity:
                    best_i, best_j, best_similarity = i, j, sim
        return best_i, best_j, best_similarity

    def _merge_cluster_into_cluster(self, dst_cluster, src_cluster):
        dst_count = int(
            dst_cluster.get("count", max(1, len(dst_cluster.get("tasks", []))))
        )
        src_count = int(
            src_cluster.get("count", max(1, len(src_cluster.get("tasks", []))))
        )
        total = max(dst_count + src_count, 1)
        old_weight = dst_count / float(total)
        new_weight = src_count / float(total)
        for idx, (src_a, src_b) in enumerate(
            zip(src_cluster["low_A"], src_cluster["low_B"])
        ):
            if self._is_shared_lora_index(idx) or src_a is None or src_b is None:
                continue
            merged_a, merged_b = self._merge_lora_pair(
                dst_cluster["low_A"][idx],
                dst_cluster["low_B"][idx],
                src_a.detach().float(),
                src_b.detach().float(),
                idx,
                old_weight=old_weight,
                new_weight=new_weight,
                old_scale=dst_cluster.get("scale", torch.Tensor([1.0])),
                new_scale=src_cluster.get("scale", torch.Tensor([1.0])),
            )
            dst_cluster["low_A"][idx] = merged_a.cpu()
            dst_cluster["low_B"][idx] = merged_b.cpu()
        dst_cluster["count"] = total
        dst_cluster.setdefault("tasks", []).extend(src_cluster.get("tasks", []))
        if self.cms_scale_merge_mode == "effective_delta":
            dst_cluster["scale"] = torch.Tensor([1.0])
        else:
            dst_scale = self._cluster_scale_tensor(dst_cluster, 1.0)
            src_scale = self._cluster_scale_tensor(src_cluster, 1.0)
            dst_cluster["scale"] = (
                (dst_scale * dst_count + src_scale * src_count) / float(total)
            ).cpu()

    def _cluster_sizes(self, clusters):
        return [
            int(cluster.get("count", max(1, len(cluster.get("tasks", [])))))
            for cluster in clusters
        ]

    def _cluster_ages(self, clusters, task_id):
        ages = []
        for cluster in clusters:
            tasks = cluster.get("tasks", [])
            last_task = max(tasks) if tasks else task_id
            ages.append(max(0, int(task_id) - int(last_task)))
        return ages

    def _cluster_has_anchor(self, cluster):
        return any(task_id in self.k_cms_anchor_tasks for task_id in cluster.get("tasks", []))

    def _eligible_cluster_mask(self, clusters, similarities):
        if len(clusters) == 0:
            return []

        mask = [True for _ in clusters]
        if self.k_cms_hard_capacity and self.k_cms_max_cluster_size > 0:
            capacity_mask = [
                size < self.k_cms_max_cluster_size for size in self._cluster_sizes(clusters)
            ]
            if any(capacity_mask):
                mask = [keep and cap for keep, cap in zip(mask, capacity_mask)]

        if self.k_cms_anchor_protection:
            non_anchor_sims = [
                sim
                for sim, cluster, keep in zip(similarities, clusters, mask)
                if keep and not self._cluster_has_anchor(cluster)
            ]
            if len(non_anchor_sims) > 0:
                best_non_anchor = max(non_anchor_sims)
                anchor_mask = []
                for sim, cluster, keep in zip(similarities, clusters, mask):
                    if not keep:
                        anchor_mask.append(False)
                    elif not self._cluster_has_anchor(cluster):
                        anchor_mask.append(True)
                    else:
                        anchor_mask.append(
                            float(sim) >= float(best_non_anchor) + self.k_cms_anchor_margin
                        )
                if any(anchor_mask):
                    mask = anchor_mask
        return mask

    def _score_clusters(self, similarities, clusters, task_id):
        if len(similarities) == 0:
            return []
        sizes = self._cluster_sizes(clusters)
        avg_size = max(sum(sizes) / float(len(sizes)), 1e-8)
        ages = self._cluster_ages(clusters, task_id)
        age_denom = max(max(ages), 1)
        scores = []
        for sim, size, age in zip(similarities, sizes, ages):
            score = float(sim)
            if self.k_cms_balance_lambda != 0.0:
                score -= self.k_cms_balance_lambda * (float(size) / avg_size)
            if self.k_cms_age_lambda != 0.0:
                score -= self.k_cms_age_lambda * (float(age) / float(age_denom))
            scores.append(score)
        return scores

    def _select_cluster(self, clusters, task_a, task_b, task_id=None):
        if len(clusters) == 0:
            return {
                "selected_idx": None,
                "selected_similarity": float("-inf"),
                "best_similarity": float("-inf"),
                "best_similarity_idx": None,
                "similarities": [],
                "scores": [],
                "sizes": [],
                "ages": [],
                "eligible": [],
            }
        similarities = [
            _task_cluster_similarity(task_a, task_b, cluster) for cluster in clusters
        ]
        best_similarity_idx = max(
            range(len(similarities)), key=lambda idx: similarities[idx]
        )
        task_id_for_age = 0 if task_id is None else int(task_id)
        scores = self._score_clusters(similarities, clusters, task_id_for_age)
        eligible = self._eligible_cluster_mask(clusters, similarities)
        masked_scores = [
            score if eligible[idx] else float("-inf") for idx, score in enumerate(scores)
        ]
        selected_idx = max(range(len(masked_scores)), key=lambda idx: masked_scores[idx])
        return {
            "selected_idx": selected_idx,
            "selected_similarity": similarities[selected_idx],
            "best_similarity": similarities[best_similarity_idx],
            "best_similarity_idx": best_similarity_idx,
            "similarities": similarities,
            "scores": scores,
            "masked_scores": masked_scores,
            "sizes": self._cluster_sizes(clusters),
            "ages": self._cluster_ages(clusters, task_id_for_age),
            "eligible": eligible,
        }

    def _should_create_cluster(self, clusters, best_similarity):
        if len(clusters) >= self.k_cms_clusters:
            return False
        if len(clusters) == 0:
            return True

        threshold = self._resolve_similarity_threshold()
        if threshold is None:
            return True
        return best_similarity < threshold

    def _resolve_similarity_threshold(self):
        if self.k_cms_threshold_mode == "none":
            return None
        if self.k_cms_threshold_mode == "fixed":
            return self.k_cms_similarity_threshold
        if self.k_cms_threshold_mode == "adaptive":
            if self.k_cms_similarity_threshold is not None:
                return self.k_cms_similarity_threshold
            state = self._load_k_cms_state()
            history = state.get("similarity_history", [])
            if len(history) < self.k_cms_min_threshold_samples:
                return None
            values = torch.tensor(history, dtype=torch.float32)
            quantile = min(max(float(self.k_cms_adaptive_quantile), 0.0), 1.0)
            return torch.quantile(values, quantile).item()
        raise ValueError(
            "Unknown k_cms_threshold_mode: {}".format(self.k_cms_threshold_mode)
        )

    def _consolidate_old_tasks(self, filename, current_task_id):
        total_saved = current_task_id + 1
        cutoff = max(0, total_saved - self.cms_recent_tasks)
        state = self._load_k_cms_state()
        start = min(state["consolidated_until"], cutoff)
        clusters = state["clusters"]
        task_scales = state["task_scales"]
        similarity_history = state["similarity_history"]
        deferred_tasks = set(state.get("deferred_tasks", []))
        merge_log = state.get("merge_log", [])

        if start >= cutoff:
            return

        for task_id in range(start, cutoff):
            if task_id in deferred_tasks:
                continue
            task_a = torch.load(
                _join_path(filename, "lora_w_a_" + str(task_id) + ".pt"),
                map_location="cpu",
            )
            task_b = torch.load(
                _join_path(filename, "lora_w_b_" + str(task_id) + ".pt"),
                map_location="cpu",
            )
            selection = self._select_cluster(clusters, task_a, task_b, task_id)
            selected_idx = selection["selected_idx"]
            selected_similarity = selection["selected_similarity"]
            best_similarity = selection["best_similarity"]
            threshold = self._resolve_similarity_threshold()
            task_scale = task_scales.get(task_id, torch.Tensor([0.8]))
            if self._should_create_cluster(clusters, best_similarity):
                clusters.append(
                    self._create_cluster(task_a, task_b, task_id, task_scale)
                )
                action = "created"
                cluster_idx = len(clusters) - 1
            elif (
                self.k_cms_low_sim_keep_recent
                and threshold is not None
                and best_similarity < threshold
                and len(deferred_tasks) < self.k_cms_low_sim_recent_limit
            ):
                deferred_tasks.add(task_id)
                action = "deferred"
                cluster_idx = -1
            elif (
                self.k_cms_reallocate_on_low_sim
                and threshold is not None
                and best_similarity < threshold
                and len(clusters) >= self.k_cms_clusters
                and len(clusters) >= 2
            ):
                merge_i, merge_j, cluster_sim = self._find_most_similar_cluster_pair(
                    clusters
                )
                self._merge_cluster_into_cluster(clusters[merge_i], clusters[merge_j])
                del clusters[merge_j]
                clusters.append(
                    self._create_cluster(task_a, task_b, task_id, task_scale)
                )
                action = "reallocated"
                cluster_idx = len(clusters) - 1
                selection["reallocated_pair"] = (merge_i, merge_j)
                selection["reallocated_pair_similarity"] = cluster_sim
            else:
                self._merge_task_into_cluster(
                    clusters[selected_idx], task_a, task_b, task_id, task_scale
                )
                action = "merged"
                cluster_idx = selected_idx
            print(
                "[K-CMS-SDLoRA] {} task {} in cluster {} (sim={:.4f}, threshold={}).".format(
                    action,
                    task_id,
                    cluster_idx,
                    selected_similarity,
                    "None" if threshold is None else "{:.4f}".format(threshold),
                )
            )
            if self.k_cms_verbose_merge:
                print(
                    "[K-CMS-SDLoRA] task {} diagnostics: sims={}, scores={}, sizes={}, ages={}, best_sim_cluster={}, selected_cluster={}.".format(
                        task_id,
                        ["{:.4f}".format(v) for v in selection["similarities"]],
                        ["{:.4f}".format(v) for v in selection["scores"]],
                        selection["sizes"],
                        selection["ages"],
                        selection["best_similarity_idx"],
                        cluster_idx,
                    )
                )
                if self.k_cms_hard_capacity or self.k_cms_anchor_protection:
                    print(
                        "[K-CMS-SDLoRA] task {} constraints: eligible={}, masked_scores={}.".format(
                            task_id,
                            selection.get("eligible", []),
                            [
                                "-inf" if v == float("-inf") else "{:.4f}".format(v)
                                for v in selection.get("masked_scores", [])
                            ],
                        )
                    )
                if "reallocated_pair" in selection:
                    print(
                        "[K-CMS-SDLoRA] task {} reallocation: merged_clusters={} cluster_sim={:.4f}.".format(
                            task_id,
                            selection["reallocated_pair"],
                            selection["reallocated_pair_similarity"],
                        )
                    )
            if best_similarity != float("-inf"):
                similarity_history.append(float(best_similarity))
            merge_log.append(
                {
                    "task": task_id,
                    "action": action,
                    "cluster": cluster_idx,
                    "selected_similarity": float(selected_similarity),
                    "best_similarity": float(best_similarity),
                    "threshold": None if threshold is None else float(threshold),
                    "similarities": [float(v) for v in selection["similarities"]],
                    "scores": [float(v) for v in selection["scores"]],
                    "masked_scores": [
                        float(v) if v != float("-inf") else float("-inf")
                        for v in selection.get("masked_scores", [])
                    ],
                    "sizes": [int(v) for v in selection["sizes"]],
                    "ages": [int(v) for v in selection["ages"]],
                    "eligible": [bool(v) for v in selection.get("eligible", [])],
                    "reallocated_pair": selection.get("reallocated_pair"),
                    "reallocated_pair_similarity": selection.get(
                        "reallocated_pair_similarity"
                    ),
                }
            )

        if not os.path.exists(filename):
            os.makedirs(filename)
        torch.save(
            {
                "consolidated_until": cutoff,
                "clusters": clusters,
                "task_scales": task_scales,
                "similarity_history": similarity_history,
                "deferred_tasks": sorted(deferred_tasks),
                "merge_log": merge_log,
                "threshold_mode": self.k_cms_threshold_mode,
                "similarity_threshold": self.k_cms_similarity_threshold,
                "adaptive_quantile": self.k_cms_adaptive_quantile,
                "balance_lambda": self.k_cms_balance_lambda,
                "age_lambda": self.k_cms_age_lambda,
                "low_sim_keep_recent": self.k_cms_low_sim_keep_recent,
                "low_sim_recent_limit": self.k_cms_low_sim_recent_limit,
                "reallocate_on_low_sim": self.k_cms_reallocate_on_low_sim,
                "scale_merge_mode": self.cms_scale_merge_mode,
                "hard_capacity": self.k_cms_hard_capacity,
                "max_cluster_size": self.k_cms_max_cluster_size,
                "anchor_protection": self.k_cms_anchor_protection,
                "anchor_tasks": sorted(self.k_cms_anchor_tasks),
                "anchor_margin": self.k_cms_anchor_margin,
                "blockwise_cluster_scale": self.k_cms_blockwise_cluster_scale,
                "fixed_orthogonal_down_layers": self.k_cms_fixed_orthogonal_down_layers,
                "shared_lora_layers": sorted(self.k_cms_shared_layers),
                "shared_fixed_orthogonal_down": self.k_cms_shared_fixed_orthogonal_down,
                "shared_freeze_down_after_task0": self.k_cms_shared_freeze_down_after_task0,
                "shared_scale": self.wrapped_param[0].param.detach().cpu(),
            },
            _join_path(filename, "k_cms_state.pt"),
        )
        if self.k_cms_delete_consolidated_files:
            for task_id in range(start, cutoff):
                if task_id in deferred_tasks:
                    continue
                for suffix in ("lora_w_a_", "lora_w_b_"):
                    path = _join_path(
                        filename, "{}{}.pt".format(suffix, task_id)
                    )
                    if os.path.exists(path):
                        os.remove(path)
        print(
            "[K-CMS-SDLoRA] Consolidated tasks {}-{} into {} clusters.".format(
                start, cutoff - 1, len(clusters)
            )
        )

    def save_lora_parameters(self, filename: str, task_id) -> None:
        self.task_id += 1
        if not os.path.exists(filename):
            os.makedirs(filename)
        torch.save(self.w_As, _join_path(filename, "lora_w_a_" + str(task_id) + ".pt"))
        torch.save(self.w_Bs, _join_path(filename, "lora_w_b_" + str(task_id) + ".pt"))
        self._save_shared_lora_parameters(filename)
        state = self._load_k_cms_state()
        if len(self.k_cms_shared_layers) > 0:
            state["shared_scale"] = self.wrapped_param[0].param.detach().cpu()
        for idx, residual_task_id in enumerate(self.residual_task_ids):
            if idx < len(self.wrapped_param_prev):
                state["task_scales"][residual_task_id] = (
                    self.wrapped_param_prev[idx].param.detach().cpu()
                )
        state["task_scales"][task_id] = self.wrapped_param[0].param.detach().cpu()
        for idx, cluster in enumerate(state["clusters"]):
            if self.cms_scale_merge_mode == "effective_delta":
                cluster["scale"] = torch.Tensor([1.0])
            elif idx < len(self.low_scale):
                cluster["scale"] = self.low_scale[idx].param.detach().cpu()
        torch.save(state, _join_path(filename, "k_cms_state.pt"))
        self._consolidate_old_tasks(filename, task_id)

    def forward(self, x: Tensor, loss=False, eval=False) -> Tensor:
        if loss:
            return self.lora_vit(x), torch.tensor(0.0, device=x.device)
        return self.lora_vit(x)
