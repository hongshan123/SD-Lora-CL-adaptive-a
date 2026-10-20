"""Matched task-bank controls with exact, unnormalized historical operators.

These controls keep each task's B and scale. The independent-A control also
keeps each task's A. No bank is truncated or normalized at a task boundary.
"""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from backbone.sa_lora import SharedALoRA_ViT_timm, _fixed_orthogonal_down


class BankQKV(nn.Module):
    def __init__(self, original, downs, ups, shared):
        super().__init__()
        for name in ("qkv", "a_q", "a_v", "b_q", "b_v", "scaling_cur"):
            setattr(self, name, getattr(original, name))
        self.dim = original.dim
        self.shared = shared
        for suffix, down, up in zip(("q", "v"), downs, ups):
            self.register_buffer("history_down_" + suffix, down, persistent=False)
            self.register_buffer("history_up_" + suffix, up, persistent=False)

    def historical(self, x, suffix):
        down = getattr(self, "history_down_" + suffix)
        up = getattr(self, "history_up_" + suffix)
        if up.numel() == 0:
            return torch.zeros_like(x)
        if not self.shared:
            # Concatenated rank axes evaluate the separate frozen operators.
            return F.linear(F.linear(x, down), up)
        hidden = F.linear(x, down)
        total = torch.zeros_like(x)
        for task_up in up.unbind(0):
            total = total + F.linear(hidden, task_up)
        return total

    def forward(self, x):
        q = self.historical(x, "q") + self.scaling_cur[0](self.b_q(self.a_q(x)))
        v = self.historical(x, "v") + self.scaling_cur[0](self.b_v(self.a_v(x)))
        original = self.qkv(x)
        return torch.cat((original[..., :self.dim] + q,
                          original[..., self.dim:2*self.dim],
                          original[..., 2*self.dim:] + v), dim=-1)


class StructureBankLoRA(SharedALoRA_ViT_timm):
    def __init__(self, *, structure_mode, **kwargs):
        if structure_mode not in ("independent_bank", "frozen_bank"):
            raise ValueError(structure_mode)
        if kwargs.get("adaptive_a_enabled") or kwargs.get("live_a_coordinate_align"):
            raise ValueError("Task banks require adaptive-A and historical LS disabled")
        if kwargs.get("live_a_absorb_mode") != "operator_preserving_absorb":
            raise ValueError("Task banks require exact operator-preserving semantics")
        self.structure_mode = structure_mode
        super().__init__(**kwargs)
        shared = structure_mode == "frozen_bank"
        training = kwargs.get("initialize_training_partition", True)
        if not shared and self.task_id > 0 and training:
            for i, module in enumerate(self.w_As):
                module.weight.data.copy_(_fixed_orthogonal_down(
                    module.weight.shape[1], self.rank, 1729 + i + 100000*self.task_id,
                    dtype=module.weight.dtype))
                module.weight.requires_grad_(True)
        record = self._bank_record
        entries = record["entries"] if record else []
        if len(entries) != self.task_id:
            raise ValueError("Bank length does not match task index")
        for layer, block in enumerate(self.lora_vit.blocks):
            if layer not in self.lora_layer:
                continue
            offset = self.lora_layer.index(layer)*2
            downs, ups = [], []
            for i in (offset, offset+1):
                if shared:
                    downs.append((record["shared_a"][i] if entries else
                                  self.w_As[i].weight.detach().cpu()).clone())
                    ups.append(torch.stack([e["scale"]*e["up"][i] for e in entries])
                               if entries else torch.empty(0, self.w_Bs[i].weight.shape[0], self.rank))
                else:
                    downs.append(torch.cat([e["down"][i] for e in entries], dim=0)
                                 if entries else torch.empty(0, self.w_As[i].weight.shape[1]))
                    ups.append(torch.cat([e["scale"]*e["up"][i] for e in entries], dim=1)
                               if entries else torch.empty(self.w_Bs[i].weight.shape[0], 0))
            block.attn.qkv = BankQKV(block.attn.qkv, downs, ups, shared)

    def _load_state(self):
        path = Path(self.save_file)/"sa_state.pt"
        self._bank_record = None
        if not path.exists():
            return {"version": -1, "shared_a": [], "scales": {}}
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("structure_mode") != self.structure_mode or state.get("version") != 101:
            raise ValueError("Incompatible structure-bank state")
        self._bank_record = state
        a = state["shared_a"]
        # Compatibility metadata initializes current modules only. The actual
        # historical operators are loaded into BankQKV from the separate bank.
        return {"version": 4, "task_id": len(state["entries"]), "rank": self.rank,
                "merge_mode": "live_a_aggregate_b", "shared_a": a,
                "aggregate_up": [torch.zeros(x.shape[1], x.shape[0]) for x in a],
                "coordinate_aligned": False, "history_groups": 1,
                "absorb_mode": "operator_preserving_absorb"}

    def save_lora_parameters(self, filename, task_id):
        if task_id != self.task_id:
            raise ValueError("Saving a bank at the wrong boundary")
        old = self._bank_record
        entries = list(old["entries"]) if old else []
        a = [m.weight.detach().cpu().clone() for m in self.w_As]
        entries.append({"up": [m.weight.detach().cpu().clone() for m in self.w_Bs],
                        "scale": self.wrapped_param[0].param.detach().cpu().clone(),
                        "down": a if self.structure_mode == "independent_bank" else []})
        self._bank_record = {"version": 101, "structure_mode": self.structure_mode,
                             "task_id": task_id, "rank": self.rank, "shared_a": a,
                             "entries": entries, "absorb_mode": "operator_preserving_absorb"}
        Path(filename).mkdir(parents=True, exist_ok=True)
        torch.save(self._bank_record, Path(filename)/"sa_state.pt")
        self.task_id += 1
        self.save_merged_lora(filename)

    def save_merged_lora(self, filename):
        # Export preserves the bank: this is deliberately not a rank-r merge.
        torch.save(self._bank_record, Path(filename)/"sa_merged_lora.pt")


def make_structure_backbone(*, structure_mode="default", **kwargs):
    if structure_mode in ("independent_bank", "frozen_bank"):
        return StructureBankLoRA(structure_mode=structure_mode, **kwargs)
    if structure_mode not in ("default", "frozen_fold", "hoep_fold"):
        raise ValueError("Unknown structure mode: " + structure_mode)
    return SharedALoRA_ViT_timm(**kwargs)
