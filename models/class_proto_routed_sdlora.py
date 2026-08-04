import logging

import numpy as np
import timm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset

from backbone.class_proto_routed_lora import (
    ClassPrototypeRoutedLoRAViT,
    joint_topk_task_class_logits,
    task_head_logits,
    task_ids_from_ranges,
)
from models.sdlora import Learner as SDLoraLearner


num_workers = 8


class Learner(SDLoraLearner):
    """Class-prototype Top-2 router over independent task LoRA experts."""

    def __init__(self, args):
        super().__init__(args)
        self.router_topk = int(args.get("class_router_topk", 2))
        self.router_temperature = float(args.get("class_router_temperature", 0.07))
        self.classifier_temperature = float(
            args.get("class_router_classifier_temperature", 1.0)
        )
        if self.router_topk != 2:
            raise ValueError("class prototype validation currently requires top-k=2")
        self._routing_history = {}

    def _wrap_ddp(self):
        if self._is_distributed():
            local_rank = self.args["local_rank"]
            self._network = DDP(
                self._network,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
                broadcast_buffers=False,
            )

    def update_network(self, index=True):
        model = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=0
        )
        model = ClassPrototypeRoutedLoRAViT(
            vit_model=model.eval(),
            r=self.args.get("lora_rank", 10),
            filepath=self.args["filepath"],
            cur_task_index=self._cur_task,
            increment=self.args["increment"],
        )
        model.out_dim = 768
        return model

    def incremental_train(self, data_manager):
        super().incremental_train(data_manager)
        self._compute_and_save_class_statistics(data_manager)

    def _prototype_loader(self, data_manager):
        dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
        )
        if self._is_distributed():
            dataset = Subset(
                dataset,
                list(range(self.args["rank"], len(dataset), self.args["world_size"])),
            )
        return DataLoader(
            dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    def _compute_and_save_class_statistics(self, data_manager):
        self._unwrap_network()
        raw_network = self._network
        backbone = raw_network.backbone
        backbone.eval()
        task_size = self._total_classes - self._known_classes
        sums = torch.zeros(task_size, backbone.out_dim, device=self._device)
        counts = torch.zeros(task_size, device=self._device)

        with torch.no_grad():
            for _, inputs, targets in self._prototype_loader(data_manager):
                inputs = inputs.to(self._device, non_blocking=True)
                local_targets = targets.to(self._device) - self._known_classes
                features = backbone.extract_router_features(inputs)
                sums.index_add_(0, local_targets, features)
                counts.index_add_(
                    0, local_targets, torch.ones_like(local_targets, dtype=counts.dtype)
                )

        if self._is_distributed():
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        if torch.any(counts <= 0):
            raise RuntimeError("each current class must contribute prototype samples")

        backbone.append_class_statistics(
            task_id=self._cur_task,
            class_feature_sums=sums,
            class_counts=counts,
            class_range=(self._known_classes, self._total_classes),
            save=self._is_main_process(),
        )
        if self._is_main_process():
            backbone.save_task_head(
                raw_network.fc.weight,
                raw_network.fc.bias,
                self._cur_task,
                (self._known_classes, self._total_classes),
            )
            logging.info(
                "[ClassProtoRouter] saved task %d: classes=%d samples=%d scale=%.6f",
                self._cur_task,
                task_size,
                int(counts.sum().item()),
                float(backbone.current_scale.detach().cpu()),
            )
        self._barrier()

    @staticmethod
    def _topk(logits, k=5):
        k = min(int(k), logits.shape[1])
        return torch.topk(logits, k=k, dim=1, largest=True, sorted=True).indices

    @staticmethod
    def _per_task_accuracy(top1, targets, class_ranges):
        result = []
        for low, high in class_ranges:
            mask = (targets >= int(low)) & (targets < int(high))
            value = (top1[mask] == targets[mask]).float().mean().item() * 100
            result.append(round(float(value), 4))
        return result

    def _record_result(self, mode, result, per_task):
        history = self._routing_history.setdefault(mode, [])
        history.append({"top1": float(result["top1"]), "per_task": per_task})
        average = float(np.mean([entry["top1"] for entry in history]))
        forgetting = 0.0
        if len(history) > 1:
            current = history[-1]["per_task"]
            losses = []
            for task_id in range(len(current) - 1):
                best = max(
                    entry["per_task"][task_id]
                    for entry in history
                    if task_id < len(entry["per_task"])
                )
                losses.append(best - current[task_id])
            forgetting = float(np.mean(losses)) if losses else 0.0
        return average, forgetting

    def _forward_selected(self, raw_network, backbone, inputs, task_ids):
        backbone.prepare_routing("selected", task_ids)
        return raw_network(inputs)

    def eval_task(self):
        self._network.eval()
        raw_network = self._raw_network()
        backbone = raw_network.backbone
        state = backbone.router_state
        class_ranges = state["class_ranges"]
        num_tasks = len(class_ranges)
        if num_tasks != self._cur_task + 1:
            raise RuntimeError("class prototypes and available LoRA experts are misaligned")

        mode_names = (
            "class_top1_global",
            "class_top1_task_head",
            "class_top2_task_head_joint",
            "oracle_global",
            "oracle_task_head",
            "base_only_global",
            "random_global",
            "all_global",
        )
        predictions = {name: [] for name in mode_names}
        all_targets, top1_routes, top2_routes, margins = [], [], [], []
        random_generator = torch.Generator().manual_seed(
            int(self.args["seed"]) + 1009 * self._cur_task
        )

        with torch.no_grad():
            for _, inputs, targets in self.test_loader:
                inputs = inputs.to(self._device, non_blocking=True)
                targets_device = targets.to(self._device)
                true_tasks = task_ids_from_ranges(targets_device, class_ranges)
                router_features = backbone.extract_router_features(inputs)
                candidates, task_scores, _ = backbone.route_from_features(
                    router_features, topk=self.router_topk
                )
                top1 = candidates[:, 0]
                top1_output = self._forward_selected(
                    raw_network, backbone, inputs, top1
                )
                predictions["class_top1_global"].append(
                    self._topk(top1_output["logits"], self.topk).cpu()
                )
                top1_task_logits = task_head_logits(
                    top1_output["features"],
                    raw_network.fc.weight,
                    raw_network.fc.bias,
                    top1,
                    class_ranges,
                )
                predictions["class_top1_task_head"].append(
                    self._topk(top1_task_logits, self.topk).cpu()
                )

                candidate_features = [top1_output["features"]]
                for slot in range(1, candidates.shape[1]):
                    output = self._forward_selected(
                        raw_network, backbone, inputs, candidates[:, slot]
                    )
                    candidate_features.append(output["features"])
                joint_logits = joint_topk_task_class_logits(
                    task_scores,
                    candidates,
                    candidate_features,
                    raw_network.fc.weight,
                    raw_network.fc.bias,
                    class_ranges,
                    self.router_temperature,
                    self.classifier_temperature,
                )
                predictions["class_top2_task_head_joint"].append(
                    self._topk(joint_logits, self.topk).cpu()
                )

                oracle_output = self._forward_selected(
                    raw_network, backbone, inputs, true_tasks
                )
                predictions["oracle_global"].append(
                    self._topk(oracle_output["logits"], self.topk).cpu()
                )
                oracle_task_logits = task_head_logits(
                    oracle_output["features"],
                    raw_network.fc.weight,
                    raw_network.fc.bias,
                    true_tasks,
                    class_ranges,
                )
                predictions["oracle_task_head"].append(
                    self._topk(oracle_task_logits, self.topk).cpu()
                )

                base_logits = raw_network.fc(router_features)["logits"]
                predictions["base_only_global"].append(
                    self._topk(base_logits, self.topk).cpu()
                )

                random_tasks = torch.randint(
                    num_tasks,
                    (inputs.shape[0],),
                    generator=random_generator,
                ).to(self._device)
                random_output = self._forward_selected(
                    raw_network, backbone, inputs, random_tasks
                )
                predictions["random_global"].append(
                    self._topk(random_output["logits"], self.topk).cpu()
                )

                backbone.prepare_routing("all")
                all_output = raw_network(inputs)
                predictions["all_global"].append(
                    self._topk(all_output["logits"], self.topk).cpu()
                )

                all_targets.append(targets.cpu())
                top1_routes.append(top1.cpu())
                top2_routes.append(candidates.cpu())
                if task_scores.shape[1] > 1:
                    ordered = torch.topk(task_scores, k=2, dim=1).values
                    margins.append((ordered[:, 0] - ordered[:, 1]).cpu())
                else:
                    margins.append(torch.full((inputs.shape[0],), float("nan")))

        targets = torch.cat(all_targets)
        true_tasks = task_ids_from_ranges(targets, class_ranges)
        evaluated = {}
        for mode in mode_names:
            values = torch.cat(predictions[mode])
            result = self._evaluate(values.numpy(), targets.numpy())
            per_task = self._per_task_accuracy(values[:, 0], targets, class_ranges)
            average, forgetting = self._record_result(mode, result, per_task)
            evaluated[mode] = result
            logging.info(
                "[ClassProtoRouter][%s] Top1=%.4f Top5=%.4f AvgAcc=%.4f "
                "LastAcc=%.4f Forgetting=%.4f PerTask=%s",
                mode,
                float(result["top1"]),
                float(result["top5"]),
                average,
                float(result["top1"]),
                forgetting,
                per_task,
            )

        selected_top1 = torch.cat(top1_routes)
        selected_top2 = torch.cat(top2_routes)
        top1_accuracy = (selected_top1 == true_tasks).float().mean().item() * 100
        top2_accuracy = (
            (selected_top2 == true_tasks[:, None]).any(dim=1).float().mean().item()
            * 100
        )
        confusion = torch.bincount(
            true_tasks * num_tasks + selected_top1,
            minlength=num_tasks * num_tasks,
        ).view(num_tasks, num_tasks)
        utilization = torch.bincount(selected_top1, minlength=num_tasks)
        margin_values = torch.cat(margins)
        finite_margins = margin_values[torch.isfinite(margin_values)]
        margin_mean = float(finite_margins.mean()) if finite_margins.numel() else 0.0
        logging.info(
            "[ClassProtoRouter][routing] Top1Acc=%.4f Top2Acc=%.4f "
            "MarginMean=%.6f Utilization=%s Confusion=%s",
            top1_accuracy,
            top2_accuracy,
            margin_mean,
            utilization.tolist(),
            confusion.tolist(),
        )

        return evaluated["class_top2_task_head_joint"], None
