import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer as TimmVisionTransformer

from backbone.linears import SimpleLinear


ROUTER_STATE_VERSION = 1
ROUTER_STATE_FILENAME = "task_prototypes.pt"


def _join_path(root, name):
    return os.path.join(root, name)


def _weight_tensor(value):
    if isinstance(value, nn.Module):
        value = value.weight
    return value.detach().cpu().float().clone()


def _load_lora_weights(path):
    values = torch.load(path, map_location="cpu", weights_only=True)
    return [_weight_tensor(value) for value in values]


def select_task_by_cosine(query, prototypes):
    if query.ndim != 2 or prototypes.ndim != 2:
        raise ValueError("query and prototypes must both be rank-2 tensors")
    if query.shape[1] != prototypes.shape[1]:
        raise ValueError("query and prototype feature dimensions do not match")
    if prototypes.shape[0] == 0:
        raise ValueError("at least one task prototype is required for routing")
    query = F.normalize(query.float(), dim=1)
    prototypes = F.normalize(prototypes.float(), dim=1)
    scores = query @ prototypes.t()
    return scores.argmax(dim=1), scores


def targets_to_task_ids(targets, class_ranges):
    task_ids = torch.full_like(targets, -1, dtype=torch.long)
    for task_id, class_range in enumerate(class_ranges):
        low, high = int(class_range[0]), int(class_range[1])
        in_task = (targets >= low) & (targets < high)
        task_ids[in_task] = task_id
    if torch.any(task_ids < 0):
        raise ValueError("some targets are outside the saved task class ranges")
    return task_ids


def mask_logits_to_tasks(logits, task_ids, class_ranges):
    if logits.shape[0] != task_ids.shape[0]:
        raise ValueError("logits and task_ids batch dimensions do not match")
    masked = torch.full_like(logits, torch.finfo(logits.dtype).min)
    for task_id in torch.unique(task_ids).tolist():
        if task_id < 0 or task_id >= len(class_ranges):
            raise ValueError("routed task id is outside the saved class ranges")
        low, high = class_ranges[task_id]
        sample_mask = task_ids == int(task_id)
        masked[sample_mask, int(low) : int(high)] = logits[
            sample_mask, int(low) : int(high)
        ]
    return masked


class _RoutingContext:
    def __init__(self, current_task_id):
        self.current_task_id = int(current_task_id)
        self.mode = "current"
        self.task_ids = None

    def prepare(self, mode, task_ids=None):
        if mode not in {"current", "selected", "all"}:
            raise ValueError("unknown routed LoRA mode: {}".format(mode))
        self.mode = mode
        self.task_ids = task_ids

    def reset(self):
        self.mode = "current"
        self.task_ids = None


class _PrototypeRoutedQKV(nn.Module):
    def __init__(
        self,
        qkv,
        current_a_q,
        current_b_q,
        current_a_v,
        current_b_v,
        historical_weights,
        historical_scales,
        current_scale,
        current_task_id,
        routing_context,
    ):
        super().__init__()
        self.qkv = qkv
        self.current_a_q = current_a_q
        self.current_b_q = current_b_q
        self.current_a_v = current_a_v
        self.current_b_v = current_b_v
        self.current_scale = current_scale
        self.current_task_id = int(current_task_id)
        self.routing_context = routing_context
        self.dim = qkv.in_features

        if len(historical_weights) != self.current_task_id:
            raise ValueError("historical LoRA count must match current task id")
        if len(historical_scales) != self.current_task_id:
            raise ValueError("historical scale count must match current task id")

        self.historical_task_ids = []
        for task_id, weights in enumerate(historical_weights):
            self.historical_task_ids.append(task_id)
            for name, value in zip(("a_q", "b_q", "a_v", "b_v"), weights):
                self.register_buffer(
                    "expert_{}_{}".format(name, task_id),
                    value.detach().clone().float(),
                )
        self.register_buffer(
            "historical_scales",
            torch.as_tensor(historical_scales, dtype=torch.float32).view(-1),
        )

    def _expert_weights(self, task_id):
        if task_id == self.current_task_id:
            return (
                self.current_a_q.weight,
                self.current_b_q.weight,
                self.current_a_v.weight,
                self.current_b_v.weight,
                self.current_scale,
            )
        if task_id < 0 or task_id >= self.current_task_id:
            raise ValueError("requested LoRA expert does not exist")
        return (
            getattr(self, "expert_a_q_{}".format(task_id)),
            getattr(self, "expert_b_q_{}".format(task_id)),
            getattr(self, "expert_a_v_{}".format(task_id)),
            getattr(self, "expert_b_v_{}".format(task_id)),
            self.historical_scales[task_id],
        )

    @staticmethod
    def _lora(x, a_weight, b_weight, scale):
        return scale.to(device=x.device, dtype=x.dtype) * F.linear(
            F.linear(x, a_weight.to(dtype=x.dtype)),
            b_weight.to(dtype=x.dtype),
        )

    def _apply_expert(self, x, task_id):
        a_q, b_q, a_v, b_v, scale = self._expert_weights(task_id)
        return (
            self._lora(x, a_q, b_q, scale),
            self._lora(x, a_v, b_v, scale),
        )

    def forward(self, x):
        new_q = torch.zeros_like(x)
        new_v = torch.zeros_like(x)
        mode = self.routing_context.mode

        if mode == "all":
            for task_id in range(self.current_task_id + 1):
                task_q, task_v = self._apply_expert(x, task_id)
                new_q = new_q + task_q
                new_v = new_v + task_v
        else:
            task_ids = self.routing_context.task_ids
            if mode == "current" or task_ids is None:
                task_ids = torch.full(
                    (x.shape[0],),
                    self.current_task_id,
                    dtype=torch.long,
                    device=x.device,
                )
            else:
                task_ids = task_ids.to(device=x.device, dtype=torch.long)
            if task_ids.shape != (x.shape[0],):
                raise ValueError("routed task ids must contain one id per sample")

            for task_id in torch.unique(task_ids).tolist():
                sample_mask = task_ids == int(task_id)
                task_q, task_v = self._apply_expert(x[sample_mask], int(task_id))
                new_q[sample_mask] = task_q
                new_v[sample_mask] = task_v

        qkv = self.qkv(x)
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv


class PrototypeRoutedLoRAViT(nn.Module):
    def __init__(
        self,
        vit_model: TimmVisionTransformer,
        r,
        filepath,
        cur_task_index=0,
        increment=10,
        router_depths=(4, 6, 12),
        router_primary="full",
        lora_layer=None,
    ):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be positive")
        self.rank = int(r)
        self.save_file = filepath
        self.increment = int(increment)
        self.task_id = int(cur_task_index)
        self.router_depths = tuple(sorted(set(int(v) for v in router_depths)))
        self.router_primary = str(router_primary)
        self.base_vit = copy.deepcopy(vit_model).eval()
        self.lora_vit = vit_model
        self.out_dim = int(vit_model.embed_dim)
        self.lora_layer = (
            list(range(len(vit_model.blocks))) if lora_layer is None else list(lora_layer)
        )
        self.routing_context = _RoutingContext(self.task_id)

        valid_depths = set(range(1, len(vit_model.blocks) + 1))
        if not set(self.router_depths).issubset(valid_depths):
            raise ValueError("prototype router depth is outside the ViT block range")
        if self.router_primary not in {"full", "block4", "block6"}:
            raise ValueError("prototype_router_primary must be full, block4, or block6")

        for parameter in self.base_vit.parameters():
            parameter.requires_grad_(False)
        for parameter in self.lora_vit.parameters():
            parameter.requires_grad_(False)

        self.router_state = self._load_router_state(required_tasks=self.task_id)
        historical_scales = self.router_state.get("lora_scales", torch.empty(0))
        historical_scales = historical_scales.float().view(-1).tolist()

        historical_a = []
        historical_b = []
        for task_id in range(self.task_id):
            a_path = _join_path(self.save_file, "lora_w_a_{}.pt".format(task_id))
            b_path = _join_path(self.save_file, "lora_w_b_{}.pt".format(task_id))
            if not os.path.exists(a_path) or not os.path.exists(b_path):
                raise FileNotFoundError(
                    "missing saved LoRA expert for task {} in {}".format(
                        task_id, self.save_file
                    )
                )
            historical_a.append(_load_lora_weights(a_path))
            historical_b.append(_load_lora_weights(b_path))

        self.current_scale = nn.Parameter(torch.tensor(0.8, dtype=torch.float32))
        self.w_As = nn.ModuleList()
        self.w_Bs = nn.ModuleList()
        for layer_index, block in enumerate(self.lora_vit.blocks):
            if layer_index not in self.lora_layer:
                continue
            qkv = block.attn.qkv
            dim = qkv.in_features
            current_a_q = nn.Linear(dim, self.rank, bias=False)
            current_b_q = nn.Linear(self.rank, dim, bias=False)
            current_a_v = nn.Linear(dim, self.rank, bias=False)
            current_b_v = nn.Linear(self.rank, dim, bias=False)
            lora_offset = len(self.w_As)
            self.w_As.extend([current_a_q, current_a_v])
            self.w_Bs.extend([current_b_q, current_b_v])

            historical_weights = []
            for task_id in range(self.task_id):
                historical_weights.append(
                    (
                        historical_a[task_id][lora_offset],
                        historical_b[task_id][lora_offset],
                        historical_a[task_id][lora_offset + 1],
                        historical_b[task_id][lora_offset + 1],
                    )
                )
            block.attn.qkv = _PrototypeRoutedQKV(
                qkv=qkv,
                current_a_q=current_a_q,
                current_b_q=current_b_q,
                current_a_v=current_a_v,
                current_b_v=current_b_v,
                historical_weights=historical_weights,
                historical_scales=historical_scales,
                current_scale=self.current_scale,
                current_task_id=self.task_id,
                routing_context=self.routing_context,
            )

        self.reset_parameters()
        self.lora_vit.head = nn.Identity()

    @property
    def state_path(self):
        return _join_path(self.save_file, ROUTER_STATE_FILENAME)

    def _empty_router_state(self):
        return {
            "version": ROUTER_STATE_VERSION,
            "task_ids": [],
            "class_ranges": [],
            "depths": list(self.router_depths),
            "prototypes": {
                "block4": torch.empty(0, self.out_dim),
                "block6": torch.empty(0, self.out_dim),
                "full": torch.empty(0, self.out_dim),
            },
            "lora_scales": torch.empty(0),
        }

    def _load_router_state(self, required_tasks):
        if not os.path.exists(self.state_path):
            if required_tasks == 0:
                return self._empty_router_state()
            raise FileNotFoundError(
                "{} is required before training task {}".format(
                    self.state_path, required_tasks
                )
            )
        state = torch.load(self.state_path, map_location="cpu", weights_only=True)
        if int(state.get("version", -1)) != ROUTER_STATE_VERSION:
            raise ValueError("unsupported task prototype state version")
        task_ids = [int(v) for v in state.get("task_ids", [])]
        if task_ids != list(range(required_tasks)):
            raise ValueError(
                "prototype task ids {} do not match expected tasks {}".format(
                    task_ids, list(range(required_tasks))
                )
            )
        class_ranges = state.get("class_ranges", [])
        if [int(v) for v in state.get("depths", [])] != list(self.router_depths):
            raise ValueError("saved prototype depths do not match the current config")
        scales = torch.as_tensor(state.get("lora_scales", []), dtype=torch.float32)
        if len(class_ranges) != required_tasks or scales.numel() != required_tasks:
            raise ValueError("prototype metadata count does not match saved task count")
        for key in ("block4", "block6", "full"):
            values = state.get("prototypes", {}).get(key)
            if values is None or tuple(values.shape) != (required_tasks, self.out_dim):
                raise ValueError("invalid {} prototype matrix".format(key))
            norms = torch.linalg.vector_norm(values.float(), dim=1)
            if required_tasks and not torch.allclose(
                norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4
            ):
                raise ValueError("saved task prototypes must be unit normalized")
        return state

    def reset_parameters(self):
        for layer in self.w_As:
            nn.init.kaiming_uniform_(layer.weight, a=5**0.5)
        for layer in self.w_Bs:
            nn.init.zeros_(layer.weight)

    def generate_fc(self, in_dim, out_dim):
        return SimpleLinear(in_dim, out_dim)

    def extract_router_features(self, x):
        self.base_vit.eval()
        intermediate_indices = [depth - 1 for depth in self.router_depths if depth < 12]
        with torch.no_grad():
            final_tokens, intermediates = self.base_vit.forward_intermediates(
                x,
                indices=intermediate_indices,
                return_prefix_tokens=True,
                norm=True,
                output_fmt="NLC",
            )
            features = {}
            for depth, (_, prefix_tokens) in zip(
                [v for v in self.router_depths if v < 12], intermediates
            ):
                features["block{}".format(depth)] = F.normalize(
                    prefix_tokens[:, 0].float(), dim=1
                )
            features["full"] = F.normalize(
                self.base_vit.forward_head(final_tokens, pre_logits=True).float(),
                dim=1,
            )
        return features

    def route_from_features(self, features, prototype_key):
        prototypes = self.router_state["prototypes"][prototype_key].to(features.device)
        if prototypes.shape[0] != self.task_id + 1:
            raise ValueError(
                "routing requires prototypes for all {} available experts".format(
                    self.task_id + 1
                )
            )
        return select_task_by_cosine(features, prototypes)

    def prepare_routing(self, mode, task_ids=None):
        self.routing_context.prepare(mode, task_ids)

    def save_lora_parameters(self, filename, task_id):
        if int(task_id) != self.task_id:
            raise ValueError("attempted to save LoRA under the wrong task id")
        os.makedirs(filename, exist_ok=True)
        torch.save(
            [layer.weight.detach().cpu() for layer in self.w_As],
            _join_path(filename, "lora_w_a_{}.pt".format(task_id)),
        )
        torch.save(
            [layer.weight.detach().cpu() for layer in self.w_Bs],
            _join_path(filename, "lora_w_b_{}.pt".format(task_id)),
        )

    def append_task_prototypes(
        self,
        task_id,
        prototypes,
        class_range,
        save=False,
    ):
        task_id = int(task_id)
        if task_id != len(self.router_state["task_ids"]):
            raise ValueError("task prototypes must be appended in task order")
        for key in ("block4", "block6", "full"):
            value = F.normalize(prototypes[key].detach().cpu().float().view(1, -1), dim=1)
            self.router_state["prototypes"][key] = torch.cat(
                [self.router_state["prototypes"][key], value], dim=0
            )
        self.router_state["task_ids"].append(task_id)
        self.router_state["class_ranges"].append(
            [int(class_range[0]), int(class_range[1])]
        )
        self.router_state["lora_scales"] = torch.cat(
            [
                self.router_state["lora_scales"].float().view(-1),
                self.current_scale.detach().cpu().float().view(1),
            ]
        )
        if save:
            os.makedirs(self.save_file, exist_ok=True)
            torch.save(self.router_state, self.state_path)

    def forward(self, x, loss=False, eval=False):
        prepared = self.routing_context.mode != "current"
        if not self.training and not prepared:
            if len(self.router_state["task_ids"]) == self.task_id + 1:
                features = self.extract_router_features(x)
                key = self.router_primary
                task_ids, _ = self.route_from_features(features[key], key)
                self.routing_context.prepare("selected", task_ids)
        try:
            output = self.lora_vit(x)
        finally:
            self.routing_context.reset()
        if loss:
            return output, torch.zeros((), device=x.device)
        return output
