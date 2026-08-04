import logging

import numpy as np
import timm
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset

from backbone.proto_routed_lora import (
    PrototypeRoutedLoRAViT,
    mask_logits_to_tasks,
    targets_to_task_ids,
)
from models.sdlora import Learner as SDLoraLearner


num_workers = 8


class Learner(SDLoraLearner):
    """Task-isolated LoRA experts with non-parametric prototype routing."""

    def __init__(self, args):
        if int(args.get("prototype_router_topk", 1)) != 1:
            raise ValueError("phase-one prototype routing supports top-k=1 only")
        super().__init__(args)
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
        model = PrototypeRoutedLoRAViT(
            vit_model=model.eval(),
            r=self.args.get("lora_rank", 10),
            filepath=self.args["filepath"],
            cur_task_index=self._cur_task,
            increment=self.args["increment"],
            router_depths=self.args.get("prototype_router_depths", [4, 6, 12]),
            router_primary=self.args.get("prototype_router_primary", "full"),
        )
        model.out_dim = 768
        return model

    def incremental_train(self, data_manager):
        super().incremental_train(data_manager)
        self._compute_and_save_task_prototypes(data_manager)

    def _prototype_loader(self, data_manager):
        dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
        )
        if self._is_distributed():
            indices = list(
                range(self.args["rank"], len(dataset), self.args["world_size"])
            )
            dataset = Subset(dataset, indices)
        return DataLoader(
            dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    def _compute_and_save_task_prototypes(self, data_manager):
        self._unwrap_network()
        backbone = self._network.backbone
        backbone.eval()
        sums = {
            "block4": torch.zeros(768, device=self._device),
            "block6": torch.zeros(768, device=self._device),
            "full": torch.zeros(768, device=self._device),
        }
        count = torch.zeros(1, device=self._device)

        for _, inputs, _ in self._prototype_loader(data_manager):
            inputs = inputs.to(self._device, non_blocking=True)
            features = backbone.extract_router_features(inputs)
            for key in sums:
                sums[key] += features[key].sum(dim=0)
            count += inputs.shape[0]

        if self._is_distributed():
            for value in sums.values():
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
        if count.item() <= 0:
            raise RuntimeError("cannot build a prototype from an empty task dataset")

        prototypes = {
            key: torch.nn.functional.normalize(value / count, dim=0)
            for key, value in sums.items()
        }
        backbone.append_task_prototypes(
            task_id=self._cur_task,
            prototypes=prototypes,
            class_range=(self._known_classes, self._total_classes),
            save=self._is_main_process(),
        )
        self._barrier()
        if self._is_main_process():
            logging.info(
                "[ProtoRouter] saved task %d prototypes (samples=%d, norms=%s)",
                self._cur_task,
                int(count.item()),
                {
                    key: round(float(torch.linalg.vector_norm(value).item()), 6)
                    for key, value in prototypes.items()
                },
            )

    @staticmethod
    def _topk(logits, k=5):
        k = min(int(k), logits.shape[1])
        return torch.topk(logits, k=k, dim=1, largest=True, sorted=True).indices

    @staticmethod
    def _per_task_accuracy(top1, targets, class_ranges):
        values = []
        for low, high in class_ranges:
            mask = (targets >= int(low)) & (targets < int(high))
            if mask.any():
                values.append(
                    round(
                        float((top1[mask] == targets[mask]).float().mean().item() * 100),
                        4,
                    )
                )
            else:
                values.append(float("nan"))
        return values

    def _record_routing_result(self, mode, result, per_task):
        history = self._routing_history.setdefault(mode, [])
        history.append({"top1": float(result["top1"]), "per_task": per_task})
        average_accuracy = float(np.mean([entry["top1"] for entry in history]))
        forgetting = 0.0
        if len(history) > 1:
            losses = []
            current = history[-1]["per_task"]
            for task_id in range(len(current) - 1):
                best = max(
                    entry["per_task"][task_id]
                    for entry in history[task_id:]
                    if task_id < len(entry["per_task"])
                )
                losses.append(best - current[task_id])
            forgetting = float(np.mean(losses)) if losses else 0.0
        return average_accuracy, forgetting

    def eval_task(self):
        self._network.eval()
        raw_network = self._raw_network()
        backbone = raw_network.backbone
        class_ranges = backbone.router_state["class_ranges"]
        expected_tasks = self._cur_task + 1
        if len(class_ranges) != expected_tasks:
            raise RuntimeError(
                "routing evaluation expected {} task prototypes, found {}".format(
                    expected_tasks, len(class_ranges)
                )
            )

        mode_predictions = {
            "prototype_block4_global": [],
            "prototype_block4_mask": [],
            "prototype_block6_global": [],
            "prototype_block6_mask": [],
            "prototype_full_global": [],
            "prototype_full_mask": [],
            "oracle_global": [],
            "all_global": [],
        }
        all_targets = []
        routed_ids = {"block4": [], "block6": [], "full": []}

        with torch.no_grad():
            for _, inputs, targets in self.test_loader:
                inputs = inputs.to(self._device, non_blocking=True)
                targets_device = targets.to(self._device)
                true_task_ids = targets_to_task_ids(targets_device, class_ranges)
                router_features = backbone.extract_router_features(inputs)

                for key in ("block4", "block6", "full"):
                    selected, _ = backbone.route_from_features(router_features[key], key)
                    backbone.prepare_routing("selected", selected)
                    logits = raw_network(inputs)["logits"]
                    mode_predictions["prototype_{}_global".format(key)].append(
                        self._topk(logits, self.topk).cpu()
                    )
                    masked_logits = mask_logits_to_tasks(logits, selected, class_ranges)
                    mode_predictions["prototype_{}_mask".format(key)].append(
                        self._topk(masked_logits, self.topk).cpu()
                    )
                    routed_ids[key].append(selected.cpu())

                backbone.prepare_routing("selected", true_task_ids)
                oracle_logits = raw_network(inputs)["logits"]
                mode_predictions["oracle_global"].append(
                    self._topk(oracle_logits, self.topk).cpu()
                )

                backbone.prepare_routing("all")
                all_logits = raw_network(inputs)["logits"]
                mode_predictions["all_global"].append(
                    self._topk(all_logits, self.topk).cpu()
                )
                all_targets.append(targets.cpu())

        targets = torch.cat(all_targets)
        true_task_ids = targets_to_task_ids(targets, class_ranges)
        evaluated = {}
        for mode, values in mode_predictions.items():
            predictions = torch.cat(values)
            result = self._evaluate(predictions.numpy(), targets.numpy())
            per_task = self._per_task_accuracy(predictions[:, 0], targets, class_ranges)
            average_accuracy, forgetting = self._record_routing_result(
                mode, result, per_task
            )
            evaluated[mode] = result
            logging.info(
                "[ProtoRouter][%s] Top1=%.4f Top5=%.4f AvgAcc=%.4f Forgetting=%.4f PerTask=%s",
                mode,
                float(result["top1"]),
                float(result["top5"]),
                average_accuracy,
                forgetting,
                per_task,
            )

        for key, values in routed_ids.items():
            selected = torch.cat(values)
            routing_accuracy = float(
                (selected == true_task_ids).float().mean().item() * 100
            )
            utilization = torch.bincount(selected, minlength=expected_tasks).tolist()
            logging.info(
                "[ProtoRouter][%s] RoutingAcc=%.4f Utilization=%s",
                key,
                routing_accuracy,
                utilization,
            )
        oracle_utilization = torch.bincount(
            true_task_ids, minlength=expected_tasks
        ).tolist()
        logging.info(
            "[ProtoRouter][oracle] RoutingAcc=100.0000 Utilization=%s",
            oracle_utilization,
        )
        logging.info(
            "[ProtoRouter][all] RoutingAcc=N/A Utilization=%s ActiveExpertsPerSample=%d",
            [len(targets) for _ in range(expected_tasks)],
            expected_tasks,
        )

        return evaluated["prototype_full_global"], None
