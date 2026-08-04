import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer as TimmVisionTransformer

from backbone.linears import SimpleLinear
from backbone.proto_routed_lora import (
    _PrototypeRoutedQKV,
    _RoutingContext,
    _join_path,
    _load_lora_weights,
)


CLASS_ROUTER_STATE_VERSION = 2
CLASS_ROUTER_STATE_FILENAME = "class_router_state.pt"


def centered_class_task_scores(
    query,
    class_feature_sums,
    class_counts,
    global_feature_sum,
    global_feature_count,
    class_to_task,
):
    """Return centered cosine class scores and max-over-class task scores."""
    if query.ndim != 2 or class_feature_sums.ndim != 2:
        raise ValueError("query and class_feature_sums must be rank-2 tensors")
    if query.shape[1] != class_feature_sums.shape[1]:
        raise ValueError("query and class prototype dimensions do not match")

    device = query.device
    dtype = torch.float32
    sums = class_feature_sums.to(device=device, dtype=dtype)
    counts = torch.as_tensor(class_counts, device=device, dtype=dtype).view(-1)
    mapping = torch.as_tensor(class_to_task, device=device, dtype=torch.long).view(-1)
    global_count = torch.as_tensor(
        global_feature_count, device=device, dtype=dtype
    ).reshape(())
    if sums.shape[0] == 0 or counts.numel() != sums.shape[0]:
        raise ValueError("class router requires one positive count per class")
    if mapping.numel() != sums.shape[0] or torch.any(counts <= 0):
        raise ValueError("invalid class counts or class-to-task mapping")
    if global_count.item() <= 0:
        raise ValueError("global feature count must be positive")
    if mapping.min().item() < 0:
        raise ValueError("class-to-task mapping cannot contain negative ids")

    global_mean = torch.as_tensor(
        global_feature_sum, device=device, dtype=dtype
    ).view(1, -1) / global_count
    class_means = sums / counts[:, None]
    centered_query = F.normalize(query.float() - global_mean, dim=1)
    centered_classes = F.normalize(class_means - global_mean, dim=1)
    class_scores = centered_query @ centered_classes.t()

    num_tasks = int(mapping.max().item()) + 1
    task_scores = torch.full(
        (query.shape[0], num_tasks),
        torch.finfo(class_scores.dtype).min,
        device=device,
        dtype=class_scores.dtype,
    )
    for task_id in range(num_tasks):
        class_mask = mapping == task_id
        if not class_mask.any():
            raise ValueError("every saved task must contain at least one class")
        task_scores[:, task_id] = class_scores[:, class_mask].amax(dim=1)
    return task_scores, class_scores


def select_topk_tasks(task_scores, topk=1):
    if task_scores.ndim != 2 or task_scores.shape[1] == 0:
        raise ValueError("task_scores must contain at least one task")
    topk = min(max(int(topk), 1), task_scores.shape[1])
    try:
        order = torch.argsort(task_scores, dim=1, descending=True, stable=True)
    except TypeError:
        order = torch.argsort(task_scores, dim=1, descending=True)
    return order[:, :topk]


def task_ids_from_ranges(targets, class_ranges):
    task_ids = torch.full_like(targets, -1, dtype=torch.long)
    for task_id, (low, high) in enumerate(class_ranges):
        task_ids[(targets >= int(low)) & (targets < int(high))] = task_id
    if torch.any(task_ids < 0):
        raise ValueError("some labels are outside the saved task class ranges")
    return task_ids


def task_head_logits(features, weight, bias, task_ids, class_ranges):
    """Apply each sample's routed task-specific slice of the global classifier."""
    total_classes = int(weight.shape[0])
    output = torch.full(
        (features.shape[0], total_classes),
        torch.finfo(features.dtype).min,
        device=features.device,
        dtype=features.dtype,
    )
    for task_id in torch.unique(task_ids).tolist():
        low, high = class_ranges[int(task_id)]
        mask = task_ids == int(task_id)
        output[mask, int(low) : int(high)] = F.linear(
            features[mask],
            weight[int(low) : int(high)],
            None if bias is None else bias[int(low) : int(high)],
        )
    return output


def joint_topk_task_class_logits(
    task_scores,
    candidate_task_ids,
    candidate_features,
    weight,
    bias,
    class_ranges,
    router_temperature=0.07,
    classifier_temperature=1.0,
):
    """Combine log P(task) and task-local log P(class|task) for top-k experts."""
    if candidate_task_ids.ndim != 2:
        raise ValueError("candidate_task_ids must be [batch, topk]")
    if len(candidate_features) != candidate_task_ids.shape[1]:
        raise ValueError("one feature matrix is required for each top-k slot")
    if router_temperature <= 0 or classifier_temperature <= 0:
        raise ValueError("routing and classifier temperatures must be positive")

    batch_size = candidate_task_ids.shape[0]
    output = torch.full(
        (batch_size, weight.shape[0]),
        torch.finfo(weight.dtype).min,
        device=weight.device,
        dtype=weight.dtype,
    )
    task_log_prob = F.log_softmax(task_scores / router_temperature, dim=1)
    for slot, features in enumerate(candidate_features):
        selected = candidate_task_ids[:, slot]
        for task_id in torch.unique(selected).tolist():
            low, high = class_ranges[int(task_id)]
            mask = selected == int(task_id)
            local_logits = F.linear(
                features[mask],
                weight[int(low) : int(high)],
                None if bias is None else bias[int(low) : int(high)],
            )
            local_log_prob = F.log_softmax(
                local_logits / classifier_temperature, dim=1
            )
            values = task_log_prob[mask, int(task_id), None] + local_log_prob
            current = output[mask, int(low) : int(high)]
            output[mask, int(low) : int(high)] = torch.maximum(current, values)
    return output


class ClassPrototypeRoutedLoRAViT(nn.Module):
    """Independent task LoRA experts routed by centered class prototypes."""

    def __init__(
        self,
        vit_model: TimmVisionTransformer,
        r,
        filepath,
        cur_task_index=0,
        increment=10,
        lora_layer=None,
        inference_only=False,
        router_state=None,
    ):
        super().__init__()
        if int(r) <= 0:
            raise ValueError("LoRA rank must be positive")
        self.rank = int(r)
        self.save_file = filepath
        self.increment = int(increment)
        self.inference_only = bool(inference_only)
        self.base_vit = copy.deepcopy(vit_model).eval()
        self.lora_vit = vit_model
        self.out_dim = int(vit_model.embed_dim)
        self.lora_layer = (
            list(range(len(vit_model.blocks))) if lora_layer is None else list(lora_layer)
        )
        for parameter in self.base_vit.parameters():
            parameter.requires_grad_(False)
        for parameter in self.lora_vit.parameters():
            parameter.requires_grad_(False)

        if router_state is None:
            required_tasks = None if self.inference_only else int(cur_task_index)
            self.router_state = self._load_router_state(required_tasks)
        else:
            self.router_state = self._validate_router_state(router_state, None)

        if self.inference_only:
            num_tasks = len(self.router_state["task_ids"])
            if num_tasks <= 0:
                raise ValueError("inference-only routing requires saved experts")
            self.task_id = num_tasks - 1
        else:
            self.task_id = int(cur_task_index)
        self.routing_context = _RoutingContext(self.task_id)

        historical_a, historical_b = [], []
        historical_scales = self.router_state["lora_scales"].float().view(-1).tolist()
        history_count = self.task_id
        for task_id in range(history_count):
            historical_a.append(self._load_expert_part(task_id, "a"))
            historical_b.append(self._load_expert_part(task_id, "b"))

        initial_scale = 0.8
        if self.inference_only:
            initial_scale = float(self.router_state["lora_scales"][self.task_id])
            historical_scales = historical_scales[: self.task_id]
        self.current_scale = nn.Parameter(
            torch.tensor(initial_scale, dtype=torch.float32)
        )
        self.w_As = nn.ModuleList()
        self.w_Bs = nn.ModuleList()
        for layer_index, block in enumerate(self.lora_vit.blocks):
            if layer_index not in self.lora_layer:
                continue
            qkv = block.attn.qkv
            dim = qkv.in_features
            a_q = nn.Linear(dim, self.rank, bias=False)
            b_q = nn.Linear(self.rank, dim, bias=False)
            a_v = nn.Linear(dim, self.rank, bias=False)
            b_v = nn.Linear(self.rank, dim, bias=False)
            offset = len(self.w_As)
            self.w_As.extend([a_q, a_v])
            self.w_Bs.extend([b_q, b_v])
            historical_weights = []
            for task_id in range(history_count):
                historical_weights.append(
                    (
                        historical_a[task_id][offset],
                        historical_b[task_id][offset],
                        historical_a[task_id][offset + 1],
                        historical_b[task_id][offset + 1],
                    )
                )
            block.attn.qkv = _PrototypeRoutedQKV(
                qkv=qkv,
                current_a_q=a_q,
                current_b_q=b_q,
                current_a_v=a_v,
                current_b_v=b_v,
                historical_weights=historical_weights,
                historical_scales=historical_scales,
                current_scale=self.current_scale,
                current_task_id=self.task_id,
                routing_context=self.routing_context,
            )

        self.reset_parameters()
        if self.inference_only:
            self._restore_current_expert(self.task_id)
            for parameter in self.parameters():
                parameter.requires_grad_(False)
        self.lora_vit.head = nn.Identity()

    @property
    def state_path(self):
        return _join_path(self.save_file, CLASS_ROUTER_STATE_FILENAME)

    def _empty_router_state(self):
        return {
            "version": CLASS_ROUTER_STATE_VERSION,
            "task_ids": [],
            "class_ranges": [],
            "class_ids": torch.empty(0, dtype=torch.long),
            "class_to_task": torch.empty(0, dtype=torch.long),
            "class_feature_sums": torch.empty(0, self.out_dim),
            "class_counts": torch.empty(0),
            "global_feature_sum": torch.zeros(self.out_dim),
            "global_feature_count": torch.tensor(0.0),
            "lora_scales": torch.empty(0),
        }

    def _validate_router_state(self, state, required_tasks):
        if int(state.get("version", -1)) != CLASS_ROUTER_STATE_VERSION:
            raise ValueError("unsupported class router state version")
        task_ids = [int(v) for v in state.get("task_ids", [])]
        if task_ids != list(range(len(task_ids))):
            raise ValueError("class router task ids must be contiguous from zero")
        if required_tasks is not None and len(task_ids) != int(required_tasks):
            raise ValueError(
                "class router contains {} tasks, expected {}".format(
                    len(task_ids), required_tasks
                )
            )
        class_ranges = [[int(a), int(b)] for a, b in state.get("class_ranges", [])]
        if len(class_ranges) != len(task_ids):
            raise ValueError("class range count does not match task count")
        class_ids = torch.as_tensor(state.get("class_ids", []), dtype=torch.long)
        class_to_task = torch.as_tensor(
            state.get("class_to_task", []), dtype=torch.long
        )
        sums = torch.as_tensor(
            state.get("class_feature_sums", []), dtype=torch.float32
        )
        if sums.numel() == 0:
            sums = torch.empty(0, self.out_dim)
        counts = torch.as_tensor(state.get("class_counts", []), dtype=torch.float32)
        scales = torch.as_tensor(state.get("lora_scales", []), dtype=torch.float32)
        num_classes = class_ids.numel()
        if tuple(sums.shape) != (num_classes, self.out_dim):
            raise ValueError("invalid class feature sum matrix")
        if counts.numel() != num_classes or class_to_task.numel() != num_classes:
            raise ValueError("class router metadata count mismatch")
        if scales.numel() != len(task_ids):
            raise ValueError("LoRA scale count does not match task count")
        if num_classes:
            if not torch.equal(class_ids, torch.arange(num_classes)):
                raise ValueError("saved class ids must be contiguous from zero")
            if torch.any(counts <= 0):
                raise ValueError("saved class counts must be positive")
            expected_mapping = torch.empty(num_classes, dtype=torch.long)
            expected_low = 0
            for task_id, (low, high) in enumerate(class_ranges):
                if low != expected_low or high <= low or high > num_classes:
                    raise ValueError("invalid task class range")
                expected_mapping[low:high] = task_id
                expected_low = high
            if expected_low != num_classes:
                raise ValueError("task class ranges do not cover every saved class")
            if not torch.equal(class_to_task, expected_mapping):
                raise ValueError("class-to-task mapping disagrees with class ranges")
        global_sum = torch.as_tensor(
            state.get("global_feature_sum", []), dtype=torch.float32
        )
        global_count = torch.as_tensor(
            state.get("global_feature_count", 0), dtype=torch.float32
        ).reshape(())
        if tuple(global_sum.shape) != (self.out_dim,):
            raise ValueError("invalid global feature sum")
        if not torch.isclose(global_count, counts.sum(), atol=1e-3, rtol=1e-5):
            raise ValueError("global and class feature counts disagree")
        return {
            "version": CLASS_ROUTER_STATE_VERSION,
            "task_ids": task_ids,
            "class_ranges": class_ranges,
            "class_ids": class_ids,
            "class_to_task": class_to_task,
            "class_feature_sums": sums,
            "class_counts": counts,
            "global_feature_sum": global_sum,
            "global_feature_count": global_count,
            "lora_scales": scales,
        }

    def _load_router_state(self, required_tasks):
        if not os.path.exists(self.state_path):
            if required_tasks == 0:
                return self._empty_router_state()
            raise FileNotFoundError(
                "{} is required before task {}".format(self.state_path, required_tasks)
            )
        state = torch.load(self.state_path, map_location="cpu", weights_only=True)
        return self._validate_router_state(state, required_tasks)

    def _load_expert_part(self, task_id, part):
        path = _join_path(self.save_file, "lora_w_{}_{}.pt".format(part, task_id))
        if not os.path.exists(path):
            raise FileNotFoundError("missing task {} LoRA file: {}".format(task_id, path))
        return _load_lora_weights(path)

    def _restore_current_expert(self, task_id):
        saved_a = self._load_expert_part(task_id, "a")
        saved_b = self._load_expert_part(task_id, "b")
        if len(saved_a) != len(self.w_As) or len(saved_b) != len(self.w_Bs):
            raise ValueError("saved current expert has an incompatible layer count")
        with torch.no_grad():
            for layer, value in zip(self.w_As, saved_a):
                layer.weight.copy_(value)
            for layer, value in zip(self.w_Bs, saved_b):
                layer.weight.copy_(value)

    def reset_parameters(self):
        for layer in self.w_As:
            nn.init.kaiming_uniform_(layer.weight, a=5**0.5)
        for layer in self.w_Bs:
            nn.init.zeros_(layer.weight)

    def generate_fc(self, in_dim, out_dim):
        return SimpleLinear(in_dim, out_dim)

    def extract_router_features(self, x):
        self.base_vit.eval()
        with torch.no_grad():
            tokens = self.base_vit.forward_features(x)
            return self.base_vit.forward_head(tokens, pre_logits=True).float()

    def route_from_features(self, features, topk=1):
        state = self.router_state
        task_scores, class_scores = centered_class_task_scores(
            features,
            state["class_feature_sums"],
            state["class_counts"],
            state["global_feature_sum"],
            state["global_feature_count"],
            state["class_to_task"],
        )
        return select_topk_tasks(task_scores, topk), task_scores, class_scores

    def prepare_routing(self, mode, task_ids=None):
        self.routing_context.prepare(mode, task_ids)

    def save_lora_parameters(self, filename, task_id):
        if self.inference_only or int(task_id) != self.task_id:
            raise ValueError("attempted to save an unavailable current LoRA expert")
        os.makedirs(filename, exist_ok=True)
        torch.save(
            [layer.weight.detach().cpu() for layer in self.w_As],
            _join_path(filename, "lora_w_a_{}.pt".format(task_id)),
        )
        torch.save(
            [layer.weight.detach().cpu() for layer in self.w_Bs],
            _join_path(filename, "lora_w_b_{}.pt".format(task_id)),
        )

    def append_class_statistics(
        self, task_id, class_feature_sums, class_counts, class_range, save=False
    ):
        task_id = int(task_id)
        if self.inference_only or task_id != len(self.router_state["task_ids"]):
            raise ValueError("class statistics must be appended in training task order")
        low, high = int(class_range[0]), int(class_range[1])
        sums = class_feature_sums.detach().cpu().float()
        counts = class_counts.detach().cpu().float().view(-1)
        if tuple(sums.shape) != (high - low, self.out_dim):
            raise ValueError("current class feature sums have an invalid shape")
        if counts.numel() != high - low or torch.any(counts <= 0):
            raise ValueError("every current class requires a positive sample count")
        if low != self.router_state["class_ids"].numel():
            raise ValueError("new class range must immediately follow saved classes")

        state = self.router_state
        state["task_ids"].append(task_id)
        state["class_ranges"].append([low, high])
        state["class_ids"] = torch.cat([state["class_ids"], torch.arange(low, high)])
        state["class_to_task"] = torch.cat(
            [state["class_to_task"], torch.full((high - low,), task_id)]
        )
        state["class_feature_sums"] = torch.cat(
            [state["class_feature_sums"], sums], dim=0
        )
        state["class_counts"] = torch.cat([state["class_counts"], counts])
        state["global_feature_sum"] = state["global_feature_sum"] + sums.sum(dim=0)
        state["global_feature_count"] = state["global_feature_count"] + counts.sum()
        state["lora_scales"] = torch.cat(
            [state["lora_scales"], self.current_scale.detach().cpu().view(1)]
        )
        self.router_state = self._validate_router_state(state, task_id + 1)
        if save:
            os.makedirs(self.save_file, exist_ok=True)
            torch.save(self.router_state, self.state_path)

    def save_task_head(self, weight, bias, task_id, class_range):
        low, high = [int(v) for v in class_range]
        payload = {
            "version": CLASS_ROUTER_STATE_VERSION,
            "task_id": int(task_id),
            "class_range": [low, high],
            "weight": weight[low:high].detach().cpu(),
            "bias": None if bias is None else bias[low:high].detach().cpu(),
        }
        torch.save(payload, _join_path(self.save_file, "task_head_{}.pt".format(task_id)))

    def forward(self, x, loss=False, eval=False):
        prepared = self.routing_context.mode != "current"
        if not self.training and not prepared:
            if len(self.router_state["task_ids"]) == self.task_id + 1:
                selected, _, _ = self.route_from_features(
                    self.extract_router_features(x), topk=1
                )
                self.routing_context.prepare("selected", selected[:, 0])
        try:
            output = self.lora_vit(x)
        finally:
            self.routing_context.reset()
        if loss:
            return output, torch.zeros((), device=x.device)
        return output
