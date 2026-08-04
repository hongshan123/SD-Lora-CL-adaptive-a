import logging

import numpy as np
import timm
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from backbone.sa_lora import SharedALoRA_ViT_timm
from models.sdlora import Learner as SDLoraLearner


num_workers = 8


class Learner(SDLoraLearner):
    """Shared-A SD-LoRA: task-invariant A, per-task B, exact final merging."""

    def update_network(self, index=True):
        model = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=0
        )
        model = SharedALoRA_ViT_timm(
            vit_model=model.eval(),
            r=self.args.get("lora_rank", 10),
            num_classes=10,
            index=index,
            increment=self.args["increment"],
            filepath=self.args["filepath"],
            cur_task_index=self._cur_task,
            shared_a_orthogonal=self.args.get("sa_shared_a_orthogonal", True),
            train_a_all_tasks=self.args.get("sa_train_a_all_tasks", False),
            delete_per_task_files=self.args.get("sa_delete_per_task_files", False),
        )
        model.out_dim = 768
        return model

    def incremental_train(self, data_manager):
        super().incremental_train(data_manager)
        if self._is_main_process():
            raw_network = self._raw_network()
            raw_network.backbone.save_merged_lora(self.args["filepath"])
            logging.info(
                "[SharedA-SDLoRA] saved merged LoRA after task %d",
                self._cur_task,
            )
            if self._cur_task == data_manager.nb_tasks - 1:
                if self.args.get("sa_final_head_align", False):
                    raw_network.weight_align(int(self.args["increment"]))
                    logging.info("[SharedA-SDLoRA] final classifier weight aligned")
                tune_epochs = int(self.args.get("sa_final_head_tune_epochs", 0))
                if tune_epochs > 0:
                    self._final_head_tune(
                        data_manager,
                        raw_network,
                        epochs=tune_epochs,
                        lr=float(self.args.get("sa_final_head_lr", 1e-3)),
                    )
                raw_network.backbone.cleanup_per_task_files(self.args["filepath"])

    def _final_head_tune(self, data_manager, raw_network, epochs, lr):
        """Freeze the backbone and fine-tune only the global classifier."""
        raw_network.eval()
        for param in raw_network.backbone.parameters():
            param.requires_grad_(False)

        dataset = data_manager.get_dataset(
            np.arange(0, data_manager.nb_classes),
            source="train",
            mode="train",
        )
        loader = DataLoader(
            dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        features, targets = [], []
        with torch.no_grad():
            for _, inputs, batch_targets in loader:
                inputs = inputs.to(self._device, non_blocking=True)
                features.append(raw_network.backbone(inputs).detach())
                targets.append(batch_targets)
        feature_matrix = torch.cat(features)
        target_vector = torch.cat(targets).to(self._device)

        raw_network.fc.train()
        optimizer = optim.SGD(
            raw_network.fc.parameters(), lr=lr, momentum=0.9
        )
        for epoch in range(epochs):
            permutation = torch.randperm(feature_matrix.shape[0], device=self._device)
            epoch_loss = 0.0
            steps = 0
            for start in range(0, feature_matrix.shape[0], self.args["batch_size"]):
                indices = permutation[start : start + self.args["batch_size"]]
                batch_x = feature_matrix[indices]
                batch_y = target_vector[indices]
                logits = raw_network.fc(batch_x)["logits"]
                loss = F.cross_entropy(logits, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                steps += 1
            logging.info(
                "[SharedA-SDLoRA] final head tune epoch %d/%d loss=%.4f",
                epoch + 1,
                epochs,
                epoch_loss / max(steps, 1),
            )
