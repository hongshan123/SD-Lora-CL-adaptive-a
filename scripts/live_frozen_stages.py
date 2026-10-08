"""Pure, read-only factor reconstructions for Live-A boundary diagnostics."""

from __future__ import annotations

import torch

from backbone.coordinate_stability import align_live_a_aggregate
from backbone.sa_lora import absorb_live_a_current_projection


def effective_operator(state, index):
    return state["merged_b"][index] @ state["shared_a"][index]


def recoverability_diagnostics(old_g, old_a, new_a, eps=1e-8):
    old_a = old_a.detach().double()
    new_a = new_a.detach().double()
    old_g = old_g.detach().double()
    historical = old_g @ (old_a / (old_a.norm() + eps))
    projector = torch.linalg.pinv(new_a) @ new_a
    lost = historical @ (torch.eye(new_a.shape[1], dtype=new_a.dtype) - projector)
    total_energy = historical.square().sum()
    lost_energy = lost.square().sum()
    return {
        "r_rec": float(lost_energy / total_energy.clamp_min(eps)),
        "historical_energy": float(total_energy),
        "lost_energy": float(lost_energy),
    }


def weighted_response_energy(old_g, old_a, new_a, inputs, eps=1e-8):
    old_a = old_a.detach().double()
    new_a = new_a.detach().double()
    historical = old_g.detach().double() @ (old_a / (old_a.norm() + eps))
    projector = torch.linalg.pinv(new_a) @ new_a
    flat = inputs.detach().double().reshape(-1, new_a.shape[1])
    old_response = flat @ historical.T
    lost_response = (flat - flat @ projector) @ historical.T
    return float(lost_response.square().sum()), float(old_response.square().sum())


def build_stage_factors(previous, pre):
    if pre.get("normalize_current_branch", False):
        raise ValueError("normalized current branches require a separate reconstruction")
    if pre.get("history_forward", "shared") != "shared":
        raise ValueError("this protocol requires shared historical forward")
    branches = pre["branches"]
    if len(branches) != len(previous["shared_a"]) or len(branches) != len(previous["aggregate_up"]):
        raise ValueError("previous and current Q/V branch counts differ")
    states = {name: {"task_id": pre["task_id"], "shared_a": [], "merged_b": []}
              for name in ("S0", "S1", "S2", "S3", "S4", "S5", "H_LS")}
    diagnostics = []
    for index, branch in enumerate(branches):
        old_a = previous["shared_a"][index].float()
        old_g = previous["aggregate_up"][index].float()
        a = branch["down"].float()
        b = branch["current_up"].float()
        scale = pre["current_scale"].float()
        if "historical_up" in branch and not torch.allclose(branch["historical_up"].float(), old_g, atol=1e-6):
            raise ValueError(f"historical G mismatch at branch {index}")
        aligned_g, ls = align_live_a_aggregate(old_g, old_a, a)
        exact_up, _ = absorb_live_a_current_projection(
            a, b, scale, mode="operator_preserving_absorb"
        )
        capped_up, normcap = absorb_live_a_current_projection(
            a, b, scale, mode="bounded_norm_calibrated_absorb"
        )
        old_norm = old_a.norm() + 1e-8
        new_norm = a.norm() + 1e-8
        factors = {
            "S0": (old_a, old_g / old_norm),
            "S1": (a, old_g / new_norm),
            "S2": (a, old_g / new_norm + scale * b),
            "S3": (a, aligned_g / new_norm + scale * b),
            "S4": (a, (aligned_g + exact_up) / new_norm),
            "S5": (a, (aligned_g + capped_up) / new_norm),
            "H_LS": (a, aligned_g / new_norm),
        }
        for name, (down, up) in factors.items():
            states[name]["shared_a"].append(down)
            states[name]["merged_b"].append(up)
        diagnostics.append({
            **recoverability_diagnostics(old_g, old_a, a),
            "ls_relative_error": ls["after_relative_error"],
            "normcap_gain": normcap["consolidation_gain"],
            "s3_s4_operator_max_abs": float((effective_operator(states["S3"], index)
                                               - effective_operator(states["S4"], index)).abs().max()),
            "s4_s5_operator_fro": float(torch.linalg.vector_norm(
                effective_operator(states["S4"], index)
                - effective_operator(states["S5"], index))),
        })
    return states, diagnostics
