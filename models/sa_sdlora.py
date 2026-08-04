import logging

import timm

from backbone.sa_lora import SharedALoRA_ViT_timm
from models.sdlora import Learner as SDLoraLearner


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
                raw_network.backbone.cleanup_per_task_files(self.args["filepath"])
