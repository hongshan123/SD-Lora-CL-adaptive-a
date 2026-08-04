import logging

import timm
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from backbone.k_cms_lora import KCMSLoRA_ViT_timm
from models.sdlora import Learner as SDLoraLearner


num_workers = 8


class Learner(SDLoraLearner):
    """
    K-CMS-SDLoRA keeps recent task LoRAs explicit and consolidates older task
    LoRAs into a bounded set of similarity-aware low-rank memory clusters.
    """

    def _is_distributed(self):
        return self.args.get("distributed", False)

    def _is_main_process(self):
        return self.args.get("rank", 0) == 0

    def _barrier(self):
        if self._is_distributed() and dist.is_initialized():
            dist.barrier()

    def _unwrap_network(self):
        if isinstance(self._network, DDP) or isinstance(self._network, nn.DataParallel):
            self._network = self._network.module
        return self._network

    def _wrap_ddp(self):
        if self._is_distributed():
            local_rank = self.args["local_rank"]
            self._network = DDP(
                self._network,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
                broadcast_buffers=False,
            )

    def _raw_network(self):
        if isinstance(self._network, DDP) or isinstance(self._network, nn.DataParallel):
            return self._network.module
        return self._network

    def _sync_sum(self, value):
        tensor = torch.tensor(float(value), device=self._device)
        if self._is_distributed():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.item()

    def update_network(self, index=True, task_index=None):
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        rank = self.args.get("lora_rank", 10)
        cur_task_index = self._cur_task if task_index is None else task_index
        model = KCMSLoRA_ViT_timm(
            vit_model=model.eval(),
            r=rank,
            num_classes=10,
            index=index,
            increment=self.args["increment"],
            filepath=self.args["filepath"],
            cur_task_index=cur_task_index,
            cms_recent_tasks=self.args.get("cms_recent_tasks", 2),
            cms_low_rank=self.args.get("cms_low_rank", rank),
            cms_low_scale_init=self.args.get("cms_low_scale_init", 1.0),
            k_cms_clusters=self.args.get("k_cms_clusters", 4),
            k_cms_similarity_threshold=self.args.get(
                "k_cms_similarity_threshold", None
            ),
            k_cms_threshold_mode=self.args.get("k_cms_threshold_mode", "none"),
            k_cms_adaptive_quantile=self.args.get("k_cms_adaptive_quantile", 0.5),
            k_cms_min_threshold_samples=self.args.get(
                "k_cms_min_threshold_samples", 3
            ),
            k_cms_balance_lambda=self.args.get("k_cms_balance_lambda", 0.0),
            k_cms_age_lambda=self.args.get("k_cms_age_lambda", 0.0),
            k_cms_low_sim_keep_recent=self.args.get(
                "k_cms_low_sim_keep_recent", False
            ),
            k_cms_low_sim_recent_limit=self.args.get(
                "k_cms_low_sim_recent_limit", 0
            ),
            k_cms_reallocate_on_low_sim=self.args.get(
                "k_cms_reallocate_on_low_sim", False
            ),
            k_cms_verbose_merge=self.args.get("k_cms_verbose_merge", False),
            k_cms_hard_capacity=self.args.get("k_cms_hard_capacity", False),
            k_cms_max_cluster_size=self.args.get("k_cms_max_cluster_size", 0),
            k_cms_anchor_protection=self.args.get(
                "k_cms_anchor_protection", False
            ),
            k_cms_anchor_tasks=self.args.get("k_cms_anchor_tasks", [0]),
            k_cms_anchor_margin=self.args.get("k_cms_anchor_margin", 0.0),
            k_cms_blockwise_cluster_scale=self.args.get(
                "k_cms_blockwise_cluster_scale", False
            ),
            k_cms_fixed_orthogonal_down_layers=self.args.get(
                "k_cms_fixed_orthogonal_down_layers", 0
            ),
            k_cms_shared_lora_layers=self.args.get("k_cms_shared_lora_layers", 0),
            k_cms_shared_fixed_orthogonal_down=self.args.get(
                "k_cms_shared_fixed_orthogonal_down", True
            ),
            k_cms_shared_freeze_down_after_task0=self.args.get(
                "k_cms_shared_freeze_down_after_task0", False
            ),
            cms_scale_merge_mode=self.args.get("cms_scale_merge_mode", "separate"),
        )
        model.out_dim = 768
        return model

    def _reload_consolidated_backbone_for_eval(self):
        self._unwrap_network()
        self._network.backbone = self.update_network(
            index=False, task_index=self._cur_task + 1
        )
        if len(self._multiple_gpus) > 1 and not self._is_distributed():
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._network.to(self._device)

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        if self._is_main_process():
            logging.info(
                "Learning on {}-{}".format(self._known_classes, self._total_classes)
            )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        train_sampler = (
            DistributedSampler(
                train_dataset,
                num_replicas=self.args["world_size"],
                rank=self.args["rank"],
                shuffle=True,
                drop_last=False,
            )
            if self._is_distributed()
            else None
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        if len(self._multiple_gpus) > 1 and not self._is_distributed():
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)

        if (
            len(self._multiple_gpus) > 1
            and not self._is_distributed()
            and isinstance(self._network, nn.DataParallel)
        ):
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._unwrap_network()
        if self._cur_task == 0:
            self._network.backbone = self.update_network(index=True, task_index=0)
        self._network.to(self._device)
        if self._cur_task == 0:
            if len(self._multiple_gpus) > 1 and not self._is_distributed():
                self._network = nn.DataParallel(self._network, self._multiple_gpus)
            self._wrap_ddp()
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.args["init_lr"],
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=self.args["init_milestones"],
                gamma=self.args["init_lr_decay"],
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            self._unwrap_network()
            self._network.backbone = self.update_network(index=False)
            if len(self._multiple_gpus) > 1 and not self._is_distributed():
                self._network = nn.DataParallel(self._network, self._multiple_gpus)
            self._network.to(self._device)
            self._wrap_ddp()

            optimizer = optim.SGD(
                self._network.parameters(),
                lr=self.args["lrate"],
                momentum=0.9,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=self.args["milestones"],
                gamma=self.args["lrate_decay"],
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

        save_lora_name = self.args["filepath"]
        raw_network = self._raw_network()
        if self._is_main_process():
            raw_network.backbone.save_lora_parameters(save_lora_name, self._cur_task)
            raw_network.save_fc(save_lora_name, self._cur_task)

        self._barrier()
        self._unwrap_network()

        if self._cur_task > 0:
            self._reload_consolidated_backbone_for_eval()
        self._barrier()

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(
            range(self.args["init_epoch"]),
            disable=not self._is_main_process(),
        )
        for _, epoch in enumerate(prog_bar):
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            num_batches = 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                num_batches += 1

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).sum()
                total += len(targets)

            scheduler.step()
            loss_sum = self._sync_sum(losses)
            batch_sum = self._sync_sum(num_batches)
            correct_sum = self._sync_sum(correct.item())
            total_sum = self._sync_sum(total)
            train_acc = round(correct_sum * 100 / max(total_sum, 1), 2)

            test_acc = None
            if epoch % 5 == 0:
                self._barrier()
                if self._is_main_process():
                    test_acc = self._compute_accuracy(self._raw_network(), test_loader)
                self._barrier()

            if test_acc is not None:
                info = (
                    "Task {}, Epoch {}/{} => Loss {:.3f}, "
                    "Train_accy {:.2f}, Test_accy {:.2f}"
                ).format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    loss_sum / max(batch_sum, 1),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    loss_sum / max(batch_sum, 1),
                    train_acc,
                )
            if self._is_main_process():
                prog_bar.set_description(info)
        if self._is_main_process():
            logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["epochs"]), disable=not self._is_main_process())
        for _, epoch in enumerate(prog_bar):
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            num_batches = 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits, _ = self._network(inputs, ortho_loss=True)
                logits = logits["logits"]

                fake_targets = targets - self._known_classes
                loss = F.cross_entropy(logits[:, self._known_classes :], fake_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                num_batches += 1

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).sum()
                total += len(targets)

            scheduler.step()
            loss_sum = self._sync_sum(losses)
            batch_sum = self._sync_sum(num_batches)
            correct_sum = self._sync_sum(correct.item())
            total_sum = self._sync_sum(total)
            train_acc = round(correct_sum * 100 / max(total_sum, 1), 2)

            test_acc = None
            if epoch % 5 == 0:
                self._barrier()
                if self._is_main_process():
                    test_acc = self._compute_accuracy(self._raw_network(), test_loader)
                self._barrier()

            if test_acc is not None:
                info = (
                    "Task {}, Epoch {}/{} => Loss {:.3f}, "
                    "Train_accy {:.2f}, Test_accy {:.2f}"
                ).format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    loss_sum / max(batch_sum, 1),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    loss_sum / max(batch_sum, 1),
                    train_acc,
                )
            if self._is_main_process():
                prog_bar.set_description(info)
        if self._is_main_process():
            logging.info(info)
