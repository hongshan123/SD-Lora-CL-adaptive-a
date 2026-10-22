import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.class_proto_routed_lora import (
    CLASS_ROUTER_STATE_VERSION,
    ClassPrototypeRoutedLoRAViT,
    centered_class_task_scores,
    joint_topk_task_class_logits,
    select_topk_tasks,
    task_head_logits,
)


class TinyAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=False)


class TinyBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = TinyAttention(dim)


class TinyViT(nn.Module):
    def __init__(self, dim=4, depth=2):
        super().__init__()
        self.embed_dim = dim
        self.blocks = nn.ModuleList([TinyBlock(dim) for _ in range(depth)])
        self.head = nn.Identity()

    def forward_features(self, x):
        tokens = x[:, None, :]
        for block in self.blocks:
            qkv = block.attn.qkv(tokens)
            tokens = tokens + qkv[:, :, : self.embed_dim] + qkv[:, :, -self.embed_dim :]
        return tokens

    def forward_head(self, tokens, pre_logits=False):
        return tokens[:, 0]

    def forward(self, x):
        return self.forward_head(self.forward_features(x), pre_logits=True)


def make_state(num_tasks, dim=4):
    num_classes = num_tasks * 2
    sums = torch.arange(1, num_classes * dim + 1, dtype=torch.float32).view(
        num_classes, dim
    )
    counts = torch.full((num_classes,), 2.0)
    ranges = [[task * 2, task * 2 + 2] for task in range(num_tasks)]
    return {
        "version": CLASS_ROUTER_STATE_VERSION,
        "task_ids": list(range(num_tasks)),
        "class_ranges": ranges,
        "class_ids": torch.arange(num_classes),
        "class_to_task": torch.arange(num_tasks).repeat_interleave(2),
        "class_feature_sums": sums,
        "class_counts": counts,
        "global_feature_sum": sums.sum(dim=0),
        "global_feature_count": counts.sum(),
        "lora_scales": torch.linspace(0.7, 0.8, num_tasks),
    }


def save_expert(root, task_id, depth=2, rank=2, dim=4):
    generator = torch.Generator().manual_seed(100 + task_id)
    a_values = [torch.randn(rank, dim, generator=generator) for _ in range(depth * 2)]
    b_values = [torch.randn(dim, rank, generator=generator) for _ in range(depth * 2)]
    torch.save(a_values, root / "lora_w_a_{}.pt".format(task_id))
    torch.save(b_values, root / "lora_w_b_{}.pt".format(task_id))


def test_centered_class_router_and_stable_topk():
    sums = torch.tensor([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    counts = torch.ones(4)
    query = torch.tensor([[-1.5, 0.0], [1.5, 0.0]])
    task_scores, class_scores = centered_class_task_scores(
        query,
        sums,
        counts,
        sums.sum(dim=0),
        counts.sum(),
        torch.tensor([0, 0, 1, 1]),
    )

    assert class_scores.shape == (2, 4)
    assert select_topk_tasks(task_scores, 2).tolist() == [[0, 1], [1, 0]]


def test_task_head_and_top2_joint_scores_use_only_candidate_classes():
    features = torch.tensor([[3.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 3.0]])
    weight = torch.eye(4)
    bias = torch.zeros(4)
    ranges = [[0, 2], [2, 4]]
    routed = torch.tensor([0, 1])
    local = task_head_logits(features, weight, bias, routed, ranges)
    assert local.argmax(dim=1).tolist() == [0, 3]
    assert torch.all(local[0, 2:] < -1e20)
    assert torch.all(local[1, :2] < -1e20)

    task_scores = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    candidates = torch.tensor([[0, 1], [1, 0]])
    joint = joint_topk_task_class_logits(
        task_scores,
        candidates,
        [features, features],
        weight,
        bias,
        ranges,
    )
    assert joint.argmax(dim=1).tolist() == [0, 3]
    assert torch.isfinite(joint).all()


def test_append_statistics_saves_counts_scale_and_task_head(tmp_path):
    model = ClassPrototypeRoutedLoRAViT(
        TinyViT(), r=2, filepath=str(tmp_path), cur_task_index=0, increment=2
    )
    sums = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
    counts = torch.tensor([2.0, 2.0])
    model.append_class_statistics(0, sums, counts, (0, 2), save=True)
    model.save_lora_parameters(str(tmp_path), 0)
    model.save_task_head(torch.eye(2, 4), torch.zeros(2), 0, (0, 2))

    state = torch.load(
        tmp_path / "class_router_state.pt", map_location="cpu", weights_only=True
    )
    assert state["task_ids"] == [0]
    assert state["class_counts"].tolist() == [2.0, 2.0]
    assert state["global_feature_count"].item() == 4.0
    assert state["lora_scales"].numel() == 1
    assert (tmp_path / "task_head_0.pt").exists()


def test_training_uses_only_current_expert_and_inference_restores_all(tmp_path):
    state_one = make_state(1)
    torch.save(state_one, tmp_path / "class_router_state.pt")
    save_expert(tmp_path, 0)
    training_model = ClassPrototypeRoutedLoRAViT(
        TinyViT(), r=2, filepath=str(tmp_path), cur_task_index=1, increment=2
    )
    training_model.train()
    training_model(torch.randn(3, 4)).sum().backward()
    assert all(layer.weight.grad is not None for layer in training_model.w_Bs)
    routed_modules = [
        block.attn.qkv for block in training_model.lora_vit.blocks
    ]
    assert all(not module.expert_a_q_0.requires_grad for module in routed_modules)

    state_two = make_state(2)
    torch.save(state_two, tmp_path / "class_router_state.pt")
    save_expert(tmp_path, 1)
    model_a = ClassPrototypeRoutedLoRAViT(
        TinyViT(),
        r=2,
        filepath=str(tmp_path),
        inference_only=True,
        router_state=state_two,
    )
    model_b = ClassPrototypeRoutedLoRAViT(
        TinyViT(),
        r=2,
        filepath=str(tmp_path),
        inference_only=True,
        router_state=state_two,
    )
    assert not any(parameter.requires_grad for parameter in model_a.parameters())
    inputs = torch.randn(4, 4)
    selected = torch.tensor([0, 1, 0, 1])
    model_a.prepare_routing("selected", selected)
    grouped = model_a(inputs)
    separate = []
    for index, task_id in enumerate(selected.tolist()):
        model_a.prepare_routing("selected", torch.tensor([task_id]))
        separate.append(model_a(inputs[index : index + 1]))
    assert torch.allclose(grouped, torch.cat(separate), atol=1e-6, rtol=1e-6)

    features = model_a.extract_router_features(inputs)
    route_a = model_a.route_from_features(features, topk=2)[0]
    route_b = model_b.route_from_features(model_b.extract_router_features(inputs), 2)[0]
    assert torch.equal(route_a, route_b)
