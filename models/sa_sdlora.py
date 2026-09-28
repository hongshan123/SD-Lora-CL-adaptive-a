import logging
import copy
import json
import math
import os
import time
from contextlib import nullcontext
from numbers import Integral

import numpy as np
import timm
import torch
import torch.distributed as dist
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from backbone.lrpt import (
    apply_transport,
    apply_transport_adaptive,
    bias_relative_error,
    fit_affine_low_rank_transport,
    fit_affine_map,
    fit_affine_rank_residuals,
    fit_layerwise_weights,
    fit_low_rank_transport,
    fit_rank_residuals,
    low_rank_factors,
    project_map_to_basis,
    transport_prediction_error,
)
from backbone.coordinate_stability import (
    apply_residual_orthogonal_transport,
    fit_residual_orthogonal_transport,
)
from backbone.sa_lora import (
    SharedALoRA_ViT_timm,
    cuo_state_scalar_counts,
    hbd_historical_branch_distance,
    live_a_historical_outputs,
    register_live_a_historical_capture_hooks,
)
from backbone.sbgc import (
    historical_response_risk,
    sbgc_state_scalar_counts,
    solve_sensitivity_budgeted_g,
    weighted_response_energy,
)
from backbone.sbgc_guard import (
    choose_guard_candidate,
    stratified_holdout_indices,
)
from backbone.linears import PrototypeCosineHead, MultiPrototypeCosineHead
from utils.canonical_hash import (
    compare_named_tensors,
    hash_named_tensors,
    model_tensor_map,
)
from utils.inc_net import SimpleCosineIncrementalNet, SharedAPrototypeNet
from utils.rng_utils import (
    deterministic_loader,
    rng_preserving,
    rng_state_hash,
)
from utils.sa_task_snapshots import save_post_merge, save_pre_merge
from models.sdlora import Learner as SDLoraLearner


num_workers = 8
PROTOTYPES_FILENAME = "sa_prototypes.pt"
P0_HASHES_FILENAME = "p0_hashes.json"
COORDINATE_DIAGNOSTICS_FILENAME = "sa_coordinate_diagnostics.json"
HOEP_FUNCTIONAL_DIAGNOSTICS_FILENAME = "hoep_functional_diagnostics.pt"
SBGC_DIAGNOSTICS_FILENAME = "sbgc_diagnostics.pt"
SBGC_ATTRIBUTION_FILENAME = "sbgc_boundary_attribution.json"
SBGC_GUARD_FILENAME = "sbgc_guard_selection.json"


def summarize_boundary_logits(logits, targets, known_classes):
    """Summarize a fixed classifier's old/new class behavior."""
    logits = logits.detach().float().cpu()
    targets = targets.detach().long().cpu()
    if logits.ndim != 2 or targets.ndim != 1 or len(logits) != len(targets):
        raise ValueError("boundary logits and targets have incompatible shapes")
    if len(targets) == 0 or targets.min() < 0 or targets.max() >= logits.shape[1]:
        raise ValueError("boundary targets are empty or outside the classifier")
    correct_logits = logits.gather(1, targets[:, None]).squeeze(1)
    competing = logits.clone()
    competing.scatter_(1, targets[:, None], -torch.inf)
    margins = correct_logits - competing.max(dim=1).values
    correct = logits.argmax(dim=1).eq(targets)
    losses = F.cross_entropy(logits, targets, reduction="none")
    result = {}
    for name, mask in (
        ("all", torch.ones_like(targets, dtype=torch.bool)),
        ("old", targets < known_classes),
        ("new", targets >= known_classes),
    ):
        count = int(mask.sum())
        result[name] = {
            "count": count,
            "top1": 100.0 * float(correct[mask].float().mean()) if count else None,
            "mean_margin": float(margins[mask].mean()) if count else None,
            "mean_ce": float(losses[mask].mean()) if count else None,
        }
    return result


def validate_cuo_lowrank_config(args):
    """Validate the fixed-coordinate CUO mode before model construction."""
    if args.get("sa_cumulative_merge", "gauge") != "cuo_lowrank":
        return None
    if not args.get("sa_cumulative_state", False):
        raise ValueError("cuo_lowrank requires sa_cumulative_state=True")
    if not args.get("sa_train_a_all_tasks", False):
        raise ValueError("cuo_lowrank requires sa_train_a_all_tasks=True")
    lora_rank = int(args.get("lora_rank", 10))
    cumulative_rank = args.get("sa_cumulative_rank", lora_rank)
    if cumulative_rank is None:
        cumulative_rank = lora_rank
    if int(cumulative_rank) != lora_rank:
        raise ValueError(
            "cuo_lowrank requires sa_cumulative_rank to equal lora_rank"
        )
    try:
        cuo_lambda = float(args.get("sa_cuo_lambda", 0.1))
    except (TypeError, ValueError) as error:
        raise ValueError("sa_cuo_lambda must be strictly positive and finite") from error
    if not math.isfinite(cuo_lambda) or cuo_lambda <= 0:
        raise ValueError("sa_cuo_lambda must be strictly positive and finite")
    if (
        args.get("sa_coordinate_stable_transport", False)
        or args.get("sa_live_a_coordinate_align", False)
    ):
        raise ValueError(
            "cuo_lowrank is incompatible with CoordinateStable/transport settings"
        )
    if args.get("sa_hbd_enabled", False):
        raise ValueError("cuo_lowrank is incompatible with HBD")
    if args.get("sa_adaptive_a_enabled", False):
        raise ValueError("cuo_lowrank is incompatible with Adaptive-A")
    if args.get("sa_adaptive_a_strategy", "impact_ratio") in (
        "functional_halfspace",
        "function_safe_pareto",
    ):
        raise ValueError("cuo_lowrank is incompatible with function-safe settings")
    return cuo_lambda


def validate_sbgc_config(args):
    """Validate sensitivity-budgeted G consolidation in isolation."""
    if args.get("sa_cumulative_merge", "gauge") != "sensitivity_budgeted_g":
        return None
    settings = {
        "g_risk_budget": float(args.get("sa_g_risk_budget", 0.05)),
        "g_budget_scope": str(args.get("sa_g_budget_scope", "branch")),
        "g_sensitivity_metric": str(
            args.get("sa_g_sensitivity_metric", "fisher_diag")
        ),
        "g_sensitivity_floor": float(
            args.get("sa_g_sensitivity_floor", 1e-4)
        ),
        "g_solver_ridge": float(args.get("sa_g_solver_ridge", 1e-6)),
        "g_bisection_steps": args.get("sa_g_bisection_steps", 40),
        "g_shadow_only": bool(args.get("sa_g_shadow_only", False)),
        "g_train_projected": bool(args.get("sa_g_train_projected", False)),
        "g_plasticity_guard": bool(args.get("sa_g_plasticity_guard", False)),
        "g_holdout_fraction": float(args.get(
            "sa_g_holdout_fraction",
            0.1 if args.get("sa_g_plasticity_guard", False) else 0.0,
        )),
        "g_guard_ce_tolerance": float(args.get("sa_g_guard_ce_tolerance", 0.01)),
    }
    if not args.get("sa_cumulative_state", False):
        raise ValueError(
            "sensitivity_budgeted_g requires sa_cumulative_state=True"
        )
    if args.get("sa_train_a_all_tasks", False):
        raise ValueError(
            "sensitivity_budgeted_g requires sa_train_a_all_tasks=False"
        )
    lora_rank = int(args.get("lora_rank", 10))
    cumulative_rank = args.get("sa_cumulative_rank", lora_rank)
    if cumulative_rank is None:
        cumulative_rank = lora_rank
    if int(cumulative_rank) != lora_rank:
        raise ValueError(
            "sensitivity_budgeted_g requires sa_cumulative_rank=lora_rank"
        )
    if not math.isfinite(settings["g_risk_budget"]) or not (
        0.0 <= settings["g_risk_budget"] <= 1.0
    ):
        raise ValueError("sa_g_risk_budget must be finite and in [0, 1]")
    if settings["g_budget_scope"] not in ("branch", "global"):
        raise ValueError("sa_g_budget_scope must be branch or global")
    if settings["g_budget_scope"] == "global" and settings["g_plasticity_guard"]:
        raise ValueError("global SBGC budget is incompatible with plasticity guard")
    if settings["g_budget_scope"] == "global" and settings["g_shadow_only"]:
        raise ValueError("global SBGC budget requires deployed consolidation")
    if settings["g_train_projected"] and (
        settings["g_budget_scope"] != "branch"
        or settings["g_sensitivity_metric"] != "fisher_diag"
        or settings["g_shadow_only"]
        or settings["g_plasticity_guard"]
    ):
        raise ValueError(
            "sa_g_train_projected requires branch Fisher deployment "
            "without shadow or guard"
        )
    if settings["g_budget_scope"] == "global" and args.get(
        "sa_g_boundary_attribution", False
    ):
        raise ValueError("global SBGC budget is incompatible with boundary attribution")
    if settings["g_sensitivity_metric"] not in ("uniform", "fisher_diag"):
        raise ValueError(
            "sa_g_sensitivity_metric must be uniform or fisher_diag"
        )
    if not math.isfinite(settings["g_sensitivity_floor"]) or not (
        0.0 < settings["g_sensitivity_floor"] <= 1.0
    ):
        raise ValueError(
            "sa_g_sensitivity_floor must be finite and in (0, 1]"
        )
    if not math.isfinite(settings["g_solver_ridge"]) or (
        settings["g_solver_ridge"] <= 0
    ):
        raise ValueError("sa_g_solver_ridge must be positive and finite")
    steps = settings["g_bisection_steps"]
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps <= 0:
        raise ValueError("sa_g_bisection_steps must be a positive integer")
    settings["g_bisection_steps"] = int(steps)
    incompatible_flags = {
        "sa_live_a_coordinate_align": "coordinate alignment",
        "lrpt_enabled": "prototype transport",
        "sa_hbd_enabled": "HBD",
        "sa_adaptive_a_enabled": "Adaptive-A",
        "sa_dual_head": "Dual-B",
        "sa_normalize_current_branch": "normalized current branch",
    }
    enabled = [
        label for key, label in incompatible_flags.items() if args.get(key, False)
    ]
    if args.get(
        "sa_live_a_absorb_mode", "operator_preserving_absorb"
    ) == "bounded_norm_calibrated_absorb":
        enabled.append("NormCap")
    if enabled:
        raise ValueError(
            "sensitivity_budgeted_g must be isolated from: {}".format(
                ", ".join(enabled)
            )
        )
    return settings


def validate_coordinate_transport_config(
    args, use_prototypes, lrpt_enabled, transport_rank
):
    """Validate transport requirements without coupling it to alignment."""
    if not use_prototypes:
        raise ValueError(
            "sa_coordinate_stable_transport requires "
            "sa_use_prototype_classifier=true"
        )
    if args.get("sa_cumulative_merge") not in (
        "live_a_aggregate_b", "sensitivity_budgeted_g"
    ):
        raise ValueError(
            "sa_coordinate_stable_transport requires "
            "sa_cumulative_merge=live_a_aggregate_b or sensitivity_budgeted_g"
        )
    if lrpt_enabled:
        raise ValueError(
            "sa_coordinate_stable_transport and lrpt_enabled are "
            "mutually exclusive"
        )
    if transport_rank <= 0:
        raise ValueError("sa_coordinate_transport_rank must be positive")


def validate_adaptive_a_config(args):
    """Validate Adaptive-A settings and return backbone constructor kwargs."""
    settings = {
        "adaptive_a_enabled": bool(args.get("sa_adaptive_a_enabled", False)),
        "adaptive_a_stability_weight": float(
            args.get("sa_adaptive_a_stability_weight", 1.0)
        ),
        "adaptive_a_gate_floor": float(args.get("sa_adaptive_a_gate_floor", 0.05)),
        "adaptive_a_gate_momentum": float(
            args.get("sa_adaptive_a_gate_momentum", 0.9)
        ),
        "adaptive_a_gate_formula": str(
            args.get("sa_adaptive_a_gate_formula", "ratio")
        ),
        "adaptive_a_eps": float(args.get("sa_adaptive_a_eps", 1e-8)),
        "adaptive_a_strategy": str(
            args.get("sa_adaptive_a_strategy", "impact_ratio")
        ),
        "adaptive_a_risk_budget": float(
            args.get("sa_adaptive_a_risk_budget", 0.05)
        ),
        "adaptive_a_risk_budget_mode": str(
            args.get("sa_adaptive_a_risk_budget_mode", "absolute")
        ),
        "hoep_energy_budget": float(args.get("sa_hoep_energy_budget", 0.05)),
        "hoep_eigenvalue_rtol": float(
            args.get("sa_hoep_eigenvalue_rtol", 1e-6)
        ),
        "hoep_energy_metric": str(
            args.get("sa_hoep_energy_metric", "operator")
        ),
        "hoep_functional_diagnostics": bool(
            args.get("sa_hoep_functional_diagnostics", False)
        ),
    }
    functional_conflict_tol = float(args.get("sa_functional_conflict_tol", 1e-12))
    functional_normal_tol = float(args.get("sa_functional_normal_tol", 1e-12))
    functional_min_normal_fraction = float(
        args.get("sa_functional_min_normal_fraction", 1e-4)
    )
    functional_conflict_cosine = float(
        args.get("sa_functional_conflict_cosine", 0.05)
    )
    functional_temperature = float(
        args.get("sa_functional_temperature", 2.0)
    )
    if settings["adaptive_a_stability_weight"] < 0:
        raise ValueError("sa_adaptive_a_stability_weight must be non-negative")
    if not 0.0 <= settings["adaptive_a_gate_floor"] <= 1.0:
        raise ValueError("sa_adaptive_a_gate_floor must be in [0, 1]")
    if not 0.0 <= settings["adaptive_a_gate_momentum"] < 1.0:
        raise ValueError("sa_adaptive_a_gate_momentum must be in [0, 1)")
    if settings["adaptive_a_eps"] <= 0:
        raise ValueError("sa_adaptive_a_eps must be positive")
    if settings["adaptive_a_strategy"] not in (
        "impact_ratio",
        "tangent",
        "risk_budgeted",
        "pareto_knee",
        "functional_halfspace",
        "function_safe_pareto",
        "recoverability",
        "operator_energy_partition",
    ):
        raise ValueError(
            "sa_adaptive_a_strategy must be impact_ratio, tangent, "
            "risk_budgeted, pareto_knee, functional_halfspace, "
            "function_safe_pareto, recoverability, or "
            "operator_energy_partition"
        )
    if not 0.0 <= settings["hoep_energy_budget"] <= 1.0:
        raise ValueError("sa_hoep_energy_budget must be in [0, 1]")
    if (
        not math.isfinite(settings["hoep_eigenvalue_rtol"])
        or settings["hoep_eigenvalue_rtol"] < 0
    ):
        raise ValueError(
            "sa_hoep_eigenvalue_rtol must be finite and non-negative"
        )
    if settings["hoep_energy_metric"] not in (
        "operator",
        "functional_diag",
    ):
        raise ValueError(
            "sa_hoep_energy_metric must be operator or functional_diag"
        )
    if (
        settings["hoep_energy_metric"] != "operator"
        or settings["hoep_functional_diagnostics"]
    ) and settings["adaptive_a_strategy"] != "operator_energy_partition":
        raise ValueError(
            "functional HOEP settings require "
            "sa_adaptive_a_strategy=operator_energy_partition"
        )
    if settings["adaptive_a_strategy"] == "recoverability":
        recoverability = {
            "recoverability_stage": str(
                args.get("sa_recoverability_stage", "global_budget")
            ),
            "recoverability_budget": float(
                args.get("sa_recoverability_budget", 0.01)
            ),
            "recoverability_step_size": float(
                args.get("sa_recoverability_step_size", 0.1)
            ),
            "recoverability_interval": args.get(
                "sa_recoverability_interval", 4
            ),
            "recoverability_sketch_rank": args.get(
                "sa_recoverability_sketch_rank", 16
            ),
            "recoverability_gammas": tuple(
                float(value)
                for value in args.get(
                    "sa_recoverability_gammas",
                    [0.0, 0.25, 0.5, 0.75, 1.0],
                )
            ),
        }
        if recoverability["recoverability_stage"] not in (
            "exact_risk",
            "accessibility",
            "anchor_realign",
            "global_budget",
        ):
            raise ValueError(
                "sa_recoverability_stage must be exact_risk, accessibility, "
                "anchor_realign, or global_budget"
            )
        if recoverability["recoverability_budget"] < 0:
            raise ValueError("sa_recoverability_budget must be non-negative")
        if recoverability["recoverability_step_size"] < 0:
            raise ValueError("sa_recoverability_step_size must be non-negative")
        interval = recoverability["recoverability_interval"]
        if (
            isinstance(interval, bool)
            or not isinstance(interval, Integral)
            or interval <= 0
        ):
            raise ValueError("sa_recoverability_interval must be a positive integer")
        recoverability["recoverability_interval"] = int(interval)
        sketch_rank = recoverability["recoverability_sketch_rank"]
        if (
            isinstance(sketch_rank, bool)
            or not isinstance(sketch_rank, Integral)
            or sketch_rank <= 0
        ):
            raise ValueError("sa_recoverability_sketch_rank must be a positive integer")
        recoverability["recoverability_sketch_rank"] = int(sketch_rank)
        gammas = recoverability["recoverability_gammas"]
        if (
            not gammas
            or gammas != tuple(sorted(set(gammas)))
            or gammas[0] != 0.0
            or gammas[-1] != 1.0
            or any(not 0.0 <= value <= 1.0 for value in gammas)
        ):
            raise ValueError(
                "sa_recoverability_gammas must be sorted unique values in [0, 1] "
                "including 0 and 1"
            )
        settings.update(recoverability)
    if settings["adaptive_a_gate_formula"] not in ("ratio", "squared_ratio"):
        raise ValueError(
            "sa_adaptive_a_gate_formula must be ratio or squared_ratio"
        )
    if (
        settings["adaptive_a_strategy"]
        in ("functional_halfspace", "function_safe_pareto")
        and args.get("sa_hbd_enabled", False)
    ):
        raise ValueError(
            "functional_halfspace/function_safe_pareto cannot be combined "
            "with sa_hbd_enabled=true"
        )
    if settings["adaptive_a_risk_budget"] < 0:
        raise ValueError("sa_adaptive_a_risk_budget must be non-negative")
    if settings["adaptive_a_risk_budget_mode"] not in ("absolute", "relative"):
        raise ValueError(
            "sa_adaptive_a_risk_budget_mode must be absolute or relative"
        )
    if functional_conflict_tol < 0:
        raise ValueError("sa_functional_conflict_tol must be non-negative")
    if functional_normal_tol <= 0:
        raise ValueError("sa_functional_normal_tol must be positive")
    if not 0.0 <= functional_min_normal_fraction <= 1.0:
        raise ValueError("sa_functional_min_normal_fraction must be in [0, 1]")
    if not 0.0 <= functional_conflict_cosine <= 1.0:
        raise ValueError("sa_functional_conflict_cosine must be in [0, 1]")
    if functional_temperature <= 0.0:
        raise ValueError("sa_functional_temperature must be positive")
    if settings["adaptive_a_strategy"] == "function_safe_pareto":
        if args.get("sa_dual_head", False):
            raise ValueError(
                "function_safe_pareto cannot be combined with sa_dual_head=true"
            )
        absorb_mode = args.get(
            "sa_live_a_absorb_mode", "operator_preserving_absorb"
        )
        if absorb_mode != "operator_preserving_absorb" and not (
            args.get("sa_use_prototype_classifier", False)
            and args.get("sa_coordinate_stable_transport", False)
        ):
            raise ValueError(
                "function_safe_pareto with non-operator-preserving absorption "
                "requires sa_use_prototype_classifier=true and "
                "sa_coordinate_stable_transport=true"
            )
    if settings["adaptive_a_strategy"] in (
        "functional_halfspace",
        "function_safe_pareto",
    ):
        settings.update(
            {
                "functional_conflict_tol": functional_conflict_tol,
                "functional_normal_tol": functional_normal_tol,
                "functional_min_normal_fraction": functional_min_normal_fraction,
                "functional_conflict_cosine": functional_conflict_cosine,
                "functional_temperature": functional_temperature,
            }
        )
    if not settings["adaptive_a_enabled"]:
        return settings
    if not args.get("sa_train_a_all_tasks", False):
        raise ValueError(
            "sa_adaptive_a_enabled requires sa_train_a_all_tasks=True"
        )
    if not args.get("sa_cumulative_state", False):
        raise ValueError(
            "sa_adaptive_a_enabled requires sa_cumulative_state=True"
        )
    if args.get("sa_cumulative_merge", "gauge") != "live_a_aggregate_b":
        raise ValueError(
            "sa_adaptive_a_enabled requires "
            "sa_cumulative_merge=live_a_aggregate_b"
        )
    if not args.get("sa_live_a_coordinate_align", False):
        raise ValueError(
            "sa_adaptive_a_enabled requires "
            "sa_live_a_coordinate_align=True"
        )
    if (
        settings["adaptive_a_strategy"]
        in (
            "risk_budgeted",
            "pareto_knee",
            "functional_halfspace",
            "function_safe_pareto",
            "recoverability",
            "operator_energy_partition",
        )
        and args.get("optimizer", "sgd").lower() != "sgd"
    ):
        raise ValueError("functional and discrete Adaptive-A strategies require SGD")
    if settings["adaptive_a_strategy"] == "operator_energy_partition":
        if args.get(
            "sa_live_a_absorb_mode", "operator_preserving_absorb"
        ) != "operator_preserving_absorb":
            raise ValueError(
                "operator_energy_partition requires "
                "sa_live_a_absorb_mode=operator_preserving_absorb"
            )
        if args.get("sa_normalize_current_branch", False):
            raise ValueError(
                "operator_energy_partition requires "
                "sa_normalize_current_branch=false"
            )
    return settings


def validate_pareto_crossfit_interval(args):
    """Return the positive minibatch cadence for Pareto cross-fit refreshes."""
    interval = args.get("sa_adaptive_a_crossfit_interval", 4)
    if isinstance(interval, bool) or not isinstance(interval, Integral):
        raise ValueError("sa_adaptive_a_crossfit_interval must be a positive integer")
    if interval <= 0:
        raise ValueError("sa_adaptive_a_crossfit_interval must be a positive integer")
    return int(interval)


def validate_functional_diagnostics_interval(args):
    """Return the optional step cadence for function-safety diagnostics."""
    interval = args.get("sa_functional_diagnostics_interval", 0)
    if isinstance(interval, bool) or not isinstance(interval, Integral):
        raise ValueError(
            "sa_functional_diagnostics_interval must be a non-negative integer"
        )
    if interval < 0:
        raise ValueError(
            "sa_functional_diagnostics_interval must be a non-negative integer"
        )
    return int(interval)


def validate_functional_student_scope(args):
    """Return which student LoRA branches define old-logit stability."""
    scope = args.get("sa_functional_student_scope", "full")
    if scope not in ("full", "historical"):
        raise ValueError(
            "sa_functional_student_scope must be full or historical"
        )
    if (
        scope == "historical"
        and args.get("sa_adaptive_a_strategy") != "function_safe_pareto"
    ):
        raise ValueError(
            "sa_functional_student_scope=historical requires "
            "sa_adaptive_a_strategy=function_safe_pareto"
        )
    return scope


def validate_functional_stability_signal(args):
    """Validate the Function-Safe stability signal and fixed blend weight."""
    signal = args.get("sa_functional_stability_signal", "historical")
    choices = ("historical", "fixed_hybrid", "reliability_hybrid")
    if signal not in choices:
        raise ValueError(
            "sa_functional_stability_signal must be historical, "
            "fixed_hybrid, or reliability_hybrid"
        )
    weight = float(args.get("sa_functional_hybrid_weight", 0.5))
    if not 0.0 <= weight <= 1.0:
        raise ValueError("sa_functional_hybrid_weight must be in [0, 1]")
    if signal != "historical":
        if args.get("sa_adaptive_a_strategy") != "function_safe_pareto":
            raise ValueError(
                "hybrid functional stability requires function_safe_pareto"
            )
        if args.get("sa_functional_student_scope", "full") != "historical":
            raise ValueError(
                "hybrid functional stability requires historical student scope"
            )
    return signal, weight


def functional_teacher_reliability(
    teacher_logits, augmented_teacher_logits, temperature=2.0, eps=1e-8
):
    """Measure old-head confidence and horizontal-flip consistency."""
    if teacher_logits.shape != augmented_teacher_logits.shape:
        raise ValueError("teacher reliability logits must have matching shapes")
    if teacher_logits.ndim != 2 or teacher_logits.shape[1] < 2:
        raise ValueError("teacher reliability requires [batch, classes>=2] logits")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    with torch.no_grad():
        def _standardize(logits):
            centered = logits - logits.mean(dim=1, keepdim=True)
            scale = centered.square().mean(dim=1, keepdim=True).sqrt()
            return centered / scale.clamp_min(eps)

        probabilities = F.softmax(
            _standardize(teacher_logits) / temperature, dim=1
        )
        augmented_probabilities = F.softmax(
            _standardize(augmented_teacher_logits) / temperature, dim=1
        )
        entropy = -(
            probabilities * probabilities.clamp_min(eps).log()
        ).sum(dim=1)
        normalized_entropy = entropy / math.log(probabilities.shape[1])
        confidence = (1.0 - normalized_entropy).clamp(0.0, 1.0)
        midpoint = 0.5 * (probabilities + augmented_probabilities)
        js_divergence = 0.5 * (
            (
                probabilities
                * (
                    probabilities.clamp_min(eps).log()
                    - midpoint.clamp_min(eps).log()
                )
            ).sum(dim=1)
            + (
                augmented_probabilities
                * (
                    augmented_probabilities.clamp_min(eps).log()
                    - midpoint.clamp_min(eps).log()
                )
            ).sum(dim=1)
        )
        consistency = (1.0 - js_divergence / math.log(2.0)).clamp(0.0, 1.0)
        top1_agreement = (
            probabilities.argmax(dim=1)
            == augmented_probabilities.argmax(dim=1)
        ).to(consistency)
        consistency = consistency * top1_agreement
        weight = (confidence * consistency).mean()
    return {
        "weight": weight.detach(),
        "normalized_entropy": normalized_entropy.mean().detach(),
        "augmentation_consistency": consistency.mean().detach(),
    }


def _gradient_list_norm(gradients, eps=0.0):
    terms = [gradient.float().square().sum() for gradient in gradients if gradient is not None]
    if not terms:
        return torch.tensor(float(eps))
    return torch.stack(terms).sum().sqrt().clamp_min(eps)


def blend_functional_stability_gradients(
    historical_gradients, hbd_gradients, historical_weight, eps=1e-12
):
    """Blend unit-norm signal gradients so loss scale cannot select the signal."""
    if len(historical_gradients) != len(hbd_gradients):
        raise ValueError("functional stability gradient lists must match")
    if not 0.0 <= float(historical_weight) <= 1.0:
        raise ValueError("historical_weight must be in [0, 1]")
    hist_norm = _gradient_list_norm(historical_gradients, eps=eps)
    hbd_norm = _gradient_list_norm(hbd_gradients, eps=eps)
    weight = float(historical_weight)
    blended = []
    for historical, hbd in zip(historical_gradients, hbd_gradients):
        if historical is None and hbd is None:
            blended.append(None)
            continue
        if historical is None or hbd is None:
            raise ValueError("functional stability gradient sparsity must match")
        blended.append(
            weight * historical / hist_norm.to(historical)
            + (1.0 - weight) * hbd / hbd_norm.to(hbd)
        )
    return blended


def should_capture_functional_diagnostic(task_id, step, interval):
    """Sample diagnostics by optimizer step, independently of cross-fit refreshes."""
    return task_id > 0 and interval > 0 and step % interval == 0


def old_logits_kl(student_logits, teacher_logits, temperature=2.0):
    """Temperature-scaled KL from a frozen old-class teacher."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher old logits must have the same shape")
    if student_logits.ndim != 2 or student_logits.shape[1] == 0:
        raise ValueError("old logits must be a non-empty [batch, classes] tensor")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    teacher_probabilities = F.softmax(
        teacher_logits.detach() / temperature, dim=1
    )
    student_log_probabilities = F.log_softmax(
        student_logits / temperature, dim=1
    )
    return (
        F.kl_div(
            student_log_probabilities,
            teacher_probabilities,
            reduction="batchmean",
        )
        * temperature
        * temperature
    )


def should_refresh_pareto_crossfit(task_id, step, interval, has_cache):
    """Refresh on the first historical step and then at the configured cadence."""
    if task_id <= 0:
        return False
    if interval <= 0:
        raise ValueError("sa_adaptive_a_crossfit_interval must be positive")
    return not has_cache or step % interval == 0


def crossfit_classification_gradients(
    network, parameters, inputs, targets, known_classes
):
    """Compute CE gradients on disjoint alternating folds of one batch."""
    if inputs.shape[0] < 2:
        raise ValueError("crossfit classification requires at least two samples")
    output = network(inputs, ortho_loss=True)
    if isinstance(output, tuple):
        output = output[0]
    logits = output["logits"][:, known_classes:]
    local_targets = targets - known_classes
    train_loss = F.cross_entropy(logits[0::2], local_targets[0::2])
    control_loss = F.cross_entropy(logits[1::2], local_targets[1::2])
    train_gradients = torch.autograd.grad(
        train_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    control_gradients = torch.autograd.grad(
        control_loss,
        parameters,
        allow_unused=True,
    )
    return (
        [None if gradient is None else gradient.detach() for gradient in train_gradients],
        [
            None if gradient is None else gradient.detach()
            for gradient in control_gradients
        ],
    )


def collective_device():
    """Return the tensor device required by the current process group backend.

    NCCL collectives require CUDA tensors; Gloo accepts CPU tensors.
    """
    if dist.is_initialized():
        backend = dist.get_backend()
        if backend == dist.Backend.NCCL:
            return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def dual_head_logits(network, features):
    """Return calibrated fc/prototype/fused logits for one feature batch."""
    fc_logits = network.fc(features)["logits"] / network.tau_fc
    proto_logits = (
        network.prototype_head(features)["logits"] / network.tau_proto
    )
    fused_logits = (
        (1.0 - network.dual_lambda) * fc_logits
        + network.dual_lambda * proto_logits
    )
    return fc_logits, proto_logits, fused_logits


def serialize_prototypes(prototypes):
    """Flatten a prototype dict into a broadcastable tensor + metadata."""
    if not prototypes:
        return [0], torch.zeros(1, 1, 1), torch.zeros(1, dtype=torch.long)
    keys = sorted(int(key) for key in prototypes)
    first = prototypes[keys[0]]
    if isinstance(first, (list, tuple)):
        max_k = max(len(prototypes[key]) for key in keys)
        dim = first[0].shape[0]
        stacked = torch.full(
            (len(keys), max_k, dim),
            float("nan"),
            dtype=first[0].dtype,
        )
        counts = torch.zeros(len(keys), dtype=torch.long)
        for index, key in enumerate(keys):
            vectors = prototypes[key]
            counts[index] = len(vectors)
            for vector_index, vector in enumerate(vectors):
                stacked[index, vector_index] = vector
    else:
        stacked = torch.stack([prototypes[key] for key in keys]).unsqueeze(1)
        counts = torch.ones(len(keys), dtype=torch.long)
    return keys, stacked, counts


def deserialize_prototypes(keys, stacked, counts):
    """Rebuild a prototype dict from broadcasted metadata."""
    prototypes = {}
    for index, key in enumerate(keys):
        key = int(key)
        num = int(counts[index].item()) if counts.numel() else 1
        if num > 1:
            prototypes[key] = [
                stacked[index, vector_index].detach().clone()
                for vector_index in range(num)
            ]
        else:
            prototypes[key] = stacked[index, 0].detach().clone()
    return prototypes


def broadcast_prototypes(prototypes, src=0):
    """Broadcast prototype tensors from ``src`` to every DDP rank."""
    if not (dist.is_initialized() and dist.get_world_size() > 1):
        return prototypes
    keys, stacked, counts = serialize_prototypes(prototypes)
    rank = dist.get_rank()
    device = collective_device()
    size_tensor = torch.zeros(4, dtype=torch.long, device=device)
    if rank == src:
        size_tensor = torch.tensor(
            [len(keys), stacked.shape[0], stacked.shape[1], stacked.shape[2]],
            dtype=torch.long,
            device=device,
        )
    dist.broadcast(size_tensor, src=src)
    flat = torch.zeros(
        int(size_tensor[1] * size_tensor[2] * size_tensor[3]),
        dtype=torch.float32,
        device=device,
    )
    if rank == src:
        flat = stacked.to(device=device, dtype=torch.float32).reshape(-1)
    dist.broadcast(flat, src=src)
    stacked = flat.cpu().reshape(
        int(size_tensor[1]), int(size_tensor[2]), int(size_tensor[3])
    )
    keys_tensor = torch.zeros(int(size_tensor[0]), dtype=torch.long, device=device)
    if rank == src:
        keys_tensor = torch.tensor(keys, dtype=torch.long, device=device)
    dist.broadcast(keys_tensor, src=src)
    broadcast_counts = torch.zeros(
        int(size_tensor[0]), dtype=torch.long, device=device
    )
    if rank == src:
        broadcast_counts = counts.to(device=device)
    dist.broadcast(broadcast_counts, src=src)
    return deserialize_prototypes(
        keys_tensor.tolist(), stacked, broadcast_counts.cpu()
    )


def broadcast_dual_head_values(lambda_val, tau_fc, tau_proto, src=0):
    """Broadcast calibration scalars from ``src`` to every DDP rank."""
    values = torch.tensor(
        [lambda_val, tau_fc, tau_proto],
        dtype=torch.float64,
        device=collective_device(),
    )
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.broadcast(values, src=src)
    return float(values[0]), float(values[1]), float(values[2])


def all_ranks_equal(value):
    """True when ``value`` is bit-identical on every DDP rank."""
    if not (dist.is_initialized() and dist.get_world_size() > 1):
        return True
    device = collective_device()
    flat = value.detach().to(device=device, dtype=torch.float64).reshape(-1)
    gathered = [
        torch.zeros_like(flat) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(gathered, flat)
    return all(torch.equal(item, flat) for item in gathered)


def evaluate_dual_head_once(network, loader, device=None, topk=5):
    """One DataLoader traversal computing fc/proto/fused logits and top-k preds.

    Returns ``(preds, y_true, max_fused_proto_diff)``.  When ``lambda==1`` the
    fused logits must equal prototype logits up to 1e-6; a violation raises
    ``RuntimeError`` with the concrete diff.
    """
    network.eval()
    preds = {"fc": [], "proto": [], "fused": []}
    y_true = []
    max_fused_proto_diff = 0.0
    with torch.no_grad():
        for _, inputs, targets in loader:
            if device is not None:
                inputs = inputs.to(device, non_blocking=True)
            features = network.backbone(inputs)
            fc_logits, proto_logits, fused_logits = dual_head_logits(
                network, features
            )
            if float(network.dual_lambda) >= 1.0:
                fused_proto_diff = (
                    (fused_logits - proto_logits).abs().max().item()
                )
                max_fused_proto_diff = max(
                    max_fused_proto_diff, fused_proto_diff
                )
                if fused_proto_diff > 1e-6:
                    raise RuntimeError(
                        "lambda=1 fused/prototype logits differ by "
                        "{:.3e} (max allowed 1e-6)".format(
                            fused_proto_diff
                        )
                    )
            for mode, logits in (
                ("fc", fc_logits),
                ("proto", proto_logits),
                ("fused", fused_logits),
            ):
                topk_indices = torch.topk(
                    logits,
                    k=topk,
                    dim=1,
                    largest=True,
                    sorted=True,
                ).indices.cpu().numpy()
                preds[mode].append(topk_indices)
            y_true.append(targets.cpu().numpy())
    return preds, y_true, max_fused_proto_diff


class SharedACosineNet(SimpleCosineIncrementalNet):
    """SimpleCosineIncrementalNet with SD-LoRA-style fc artifact saving."""

    def save_fc(self, filename, task_id):
        torch.save(
            self.fc.weight.detach(), filename + "CLs_weight" + str(task_id) + ".pt"
        )
        torch.save(
            torch.zeros(self.fc.out_features),
            filename + "CLs_bias" + str(task_id) + ".pt",
        )


class Learner(SDLoraLearner):
    """Shared-A SD-LoRA: task-invariant A, per-task B, exact final merging."""

    def __init__(self, args):
        super().__init__(args)
        self._sa_cuo_lambda = validate_cuo_lowrank_config(args)
        self._sa_sbgc_settings = validate_sbgc_config(args)
        self._sa_g_boundary_attribution = bool(
            args.get("sa_g_boundary_attribution", False)
        )
        if self._sa_g_boundary_attribution and self._sa_sbgc_settings is None:
            raise ValueError(
                "sa_g_boundary_attribution requires sensitivity_budgeted_g"
            )
        self._sbgc_boundary_candidates = None
        self._sbgc_premerge_logits = None
        self._sa_g_plasticity_guard = bool(args.get("sa_g_plasticity_guard", False))
        self._sa_g_holdout_fraction = float(
            args.get("sa_g_holdout_fraction", 0.1 if self._sa_g_plasticity_guard else 0.0)
        )
        self._sa_g_guard_ce_tolerance = float(
            args.get("sa_g_guard_ce_tolerance", 0.01)
        )
        if not math.isfinite(self._sa_g_holdout_fraction) or not (
            0.0 <= self._sa_g_holdout_fraction < 0.5
        ):
            raise ValueError("sa_g_holdout_fraction must be in [0, 0.5)")
        if self._sa_g_holdout_fraction > 0 and self._sa_sbgc_settings is None:
            raise ValueError("sa_g_holdout_fraction requires sensitivity_budgeted_g")
        if self._sa_g_plasticity_guard:
            if self._sa_sbgc_settings is None:
                raise ValueError("sa_g_plasticity_guard requires sensitivity_budgeted_g")
            if self._sa_sbgc_settings["g_shadow_only"]:
                raise ValueError("sa_g_plasticity_guard requires deployed SBGC")
            if self._sa_sbgc_settings["g_sensitivity_metric"] != "fisher_diag":
                raise ValueError("sa_g_plasticity_guard requires fisher_diag")
            if not args.get("sa_use_prototype_classifier", False):
                raise ValueError("sa_g_plasticity_guard requires prototype classifier")
            if self._sa_g_holdout_fraction <= 0:
                raise ValueError("sa_g_plasticity_guard requires a holdout")
            if not math.isfinite(self._sa_g_guard_ce_tolerance) or (
                self._sa_g_guard_ce_tolerance < 0
            ):
                raise ValueError("sa_g_guard_ce_tolerance must be nonnegative")
            if abs(self._sa_sbgc_settings["g_risk_budget"] - 0.05) > 1e-12:
                raise ValueError("sa_g_plasticity_guard requires initial 0.05 budget")
        self._sa_g_train_indices = None
        self._sa_g_validation_indices = None
        self._sa_g_task_labels = None
        sbgc_calibration_batch_size = args.get(
            "sa_g_calibration_batch_size", 16
        )
        if (
            isinstance(sbgc_calibration_batch_size, bool)
            or not isinstance(sbgc_calibration_batch_size, Integral)
            or sbgc_calibration_batch_size <= 0
        ):
            raise ValueError(
                "sa_g_calibration_batch_size must be a positive integer"
            )
        self._sa_g_calibration_batch_size = int(sbgc_calibration_batch_size)
        adaptive_a_settings = validate_adaptive_a_config(args)
        self._sa_adaptive_a_enabled = adaptive_a_settings["adaptive_a_enabled"]
        self._sa_adaptive_a_strategy = adaptive_a_settings["adaptive_a_strategy"]
        self._sa_hoep_energy_metric = adaptive_a_settings[
            "hoep_energy_metric"
        ]
        self._sa_hoep_functional_diagnostics = adaptive_a_settings[
            "hoep_functional_diagnostics"
        ]
        calibration_batch_size = args.get(
            "sa_hoep_activation_calibration_batch_size",
            args["batch_size"],
        )
        if (
            isinstance(calibration_batch_size, bool)
            or not isinstance(calibration_batch_size, Integral)
            or calibration_batch_size <= 0
        ):
            raise ValueError(
                "sa_hoep_activation_calibration_batch_size must be a "
                "positive integer"
            )
        self._sa_hoep_activation_calibration_batch_size = int(
            calibration_batch_size
        )
        self._sa_functional_halfspace_enabled = (
            self._sa_adaptive_a_enabled
            and self._sa_adaptive_a_strategy == "functional_halfspace"
        )
        self._sa_function_safe_pareto_enabled = (
            self._sa_adaptive_a_enabled
            and self._sa_adaptive_a_strategy == "function_safe_pareto"
        )
        self._sa_functional_temperature = adaptive_a_settings.get(
            "functional_temperature", 2.0
        )
        self._sa_functional_student_scope = validate_functional_student_scope(args)
        (
            self._sa_functional_stability_signal,
            self._sa_functional_hybrid_weight,
        ) = validate_functional_stability_signal(args)
        self._sa_adaptive_a_crossfit_interval = validate_pareto_crossfit_interval(
            args
        )
        self._sa_functional_diagnostics_interval = (
            validate_functional_diagnostics_interval(args)
        )
        if (
            self._sa_functional_diagnostics_interval > 0
            and not self._sa_function_safe_pareto_enabled
        ):
            raise ValueError(
                "sa_functional_diagnostics_interval requires "
                "sa_adaptive_a_strategy=function_safe_pareto"
            )
        self._sa_functional_diagnostic_pending = None
        self._sa_functional_diagnostic_observations = []
        self._sa_functional_signal_observations = []
        self._sa_pareto_schedule_task_id = None
        self._sa_pareto_step = 0
        self._sa_pareto_cached_modes = None
        self._sa_pareto_cached_utilities = None
        self._sa_pareto_cached_stability_gradients = None
        self._sa_pareto_crossfit_refreshes = 0
        self._sa_pareto_cached_reuses = 0
        self._sa_pareto_task0_skips = 0
        use_cosine = args.get("sa_use_cosine_head", False)
        use_prototypes = args.get("sa_use_prototype_classifier", False)
        if use_cosine and use_prototypes:
            raise ValueError(
                "sa_use_cosine_head and sa_use_prototype_classifier "
                "are mutually exclusive"
            )
        if use_cosine:
            self._network = SharedACosineNet(args, True)
        elif use_prototypes:
            self._network = SharedAPrototypeNet(args, True)
        self._dual_head = bool(args.get("sa_dual_head", False))
        self._dual_schedule = args.get("sa_dual_head_schedule", "A")
        if self._dual_head and not use_prototypes:
            raise ValueError(
                "sa_dual_head requires sa_use_prototype_classifier=True"
            )
        if self._dual_schedule not in ("A", "B"):
            raise ValueError("sa_dual_head_schedule must be A or B")
        if self._dual_head and int(args.get("sa_k_prototypes", 1)) > 1:
            raise ValueError(
                "sa_dual_head and sa_k_prototypes>1 are mutually exclusive"
            )
        self._dual_num_tasks = 0
        self._lrpt_enabled = bool(args.get("lrpt_enabled", False))
        if self._lrpt_enabled and not use_prototypes:
            raise ValueError(
                "lrpt_enabled requires sa_use_prototype_classifier=True"
            )
        self._lrpt_rank = int(args.get("lrpt_rank", args.get("lora_rank", 10)))
        self._lrpt_reg = float(args.get("lrpt_reg", 1e-2))
        self._lrpt_bias = bool(args.get("lrpt_bias", False))
        self._lrpt_damping = float(args.get("lrpt_damping", 1.0))
        self._lrpt_dual = bool(args.get("lrpt_dual", False))
        self._lrpt_class_weight = float(args.get("lrpt_class_weight", 0.3))
        self._lrpt_basis = args.get("lrpt_basis", "generic")
        if self._lrpt_basis not in ("generic", "delta_a", "delta_a_b", "layerwise"):
            raise ValueError(
                "lrpt_basis must be generic/delta_a/delta_a_b/layerwise"
            )
        self._lrpt_basis_rank = int(
            args.get("lrpt_basis_rank", self._lrpt_rank)
        )
        self._lrpt_basis_max_samples = int(
            args.get("lrpt_basis_max_samples", 4096)
        )
        self._lrpt_basis_cap = int(args.get("lrpt_basis_cap", 128))
        self._lrpt_adaptive = bool(args.get("lrpt_adaptive", False))
        self._lrpt_fit_target = args.get("lrpt_fit_target", "sample")
        if self._lrpt_fit_target not in ("sample", "classmean"):
            raise ValueError(
                "lrpt_fit_target must be 'sample' or 'classmean'"
            )
        self._lrpt_diagnostics = bool(args.get("lrpt_diagnostics", True))
        self._lrpt_pre_features = None
        self._lrpt_pre_targets = None
        self._lrpt_a_old = None
        self._lrpt_res_pre = None
        self._coordinate_stable_transport = bool(
            args.get("sa_coordinate_stable_transport", False)
        )
        self._coordinate_transport_rank = int(
            args.get("sa_coordinate_transport_rank", args.get("lora_rank", 10))
        )
        self._coordinate_transport_reg = float(
            args.get("sa_coordinate_transport_reg", 1e-4)
        )
        self._coordinate_transport_min_gain = float(
            args.get("sa_coordinate_transport_min_gain", 0.0)
        )
        self._coordinate_pre_features = None
        self._coordinate_pre_targets = None
        if self._coordinate_stable_transport:
            validate_coordinate_transport_config(
                args,
                use_prototypes=use_prototypes,
                lrpt_enabled=self._lrpt_enabled,
                transport_rank=self._coordinate_transport_rank,
            )
        self._sa_operator_stability_lambda = float(
            args.get("sa_operator_stability_lambda", 0.0)
        )
        self._sa_prototype_consistency_weight = float(
            args.get("sa_prototype_consistency_weight", 0.0)
        )
        self._hbd_enabled = bool(args.get("sa_hbd_enabled", False))
        self._hbd_lambda = float(args.get("sa_hbd_lambda", 0.1))
        if self._hbd_enabled:
            if self._hbd_lambda <= 0:
                raise ValueError(
                    "sa_hbd_lambda must be > 0 when sa_hbd_enabled=true"
                )
            if args.get("sa_cumulative_merge", "gauge") != "live_a_aggregate_b":
                raise ValueError(
                    "sa_hbd_enabled requires "
                    "sa_cumulative_merge=live_a_aggregate_b"
                )
        self._hbd_teacher = None
        self._functional_old_head = None
        self._hbd_teacher_captures = []
        self._hbd_teacher_handles = []
        self._hbd_first_batch = True
        self._hbd_ratio_logged = False
        self._proto_ema = {}
        self._proto_ema_task = None
        if self._sa_operator_stability_lambda < 0:
            raise ValueError("sa_operator_stability_lambda must be non-negative")
        if (
            self._sa_operator_stability_lambda > 0
            and not args.get("sa_train_a_all_tasks", False)
        ):
            raise ValueError(
                "sa_operator_stability_lambda requires sa_train_a_all_tasks=True"
            )
        if (
            self._sa_operator_stability_lambda > 0
            and args.get("sa_cumulative_state", False)
        ):
            raise ValueError(
                "operator-stability loss is incompatible with cumulative "
                "Shared-A state (route closed; use gauge residual diagnostics)"
            )

    def update_network(self, index=True, task_index=None):
        model = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=0
        )
        cur_task_index = self._cur_task if task_index is None else task_index
        adaptive_a_settings = validate_adaptive_a_config(self.args)
        sbgc_settings = validate_sbgc_config(self.args) or {}
        model = SharedALoRA_ViT_timm(
            vit_model=model.eval(),
            r=self.args.get("lora_rank", 10),
            num_classes=10,
            index=index,
            increment=self.args["increment"],
            filepath=self.args["filepath"],
            cur_task_index=cur_task_index,
            shared_a_orthogonal=self.args.get("sa_shared_a_orthogonal", True),
            train_a_all_tasks=self.args.get("sa_train_a_all_tasks", False),
            delete_per_task_files=self.args.get("sa_delete_per_task_files", False),
            cumulative_state=self.args.get("sa_cumulative_state", False),
            cumulative_gauge=self.args.get("sa_cumulative_gauge", True),
            cumulative_merge=self.args.get("sa_cumulative_merge", "gauge"),
            cumulative_rank=self.args.get("sa_cumulative_rank", None),
            cuo_lambda=self.args.get("sa_cuo_lambda", 0.1),
            **sbgc_settings,
            freeze_old_scales=self.args.get("sa_freeze_old_scales", False),
            normalize_current_branch=self.args.get(
                "sa_normalize_current_branch", False
            ),
            live_a_history_groups=self.args.get("sa_live_a_history_groups", 1),
            live_a_coordinate_align=self.args.get(
                "sa_live_a_coordinate_align", False
            ),
            live_a_absorb_mode=self.args.get(
                "sa_live_a_absorb_mode", "operator_preserving_absorb"
            ),
            live_a_boundary_merge=self.args.get(
                "sa_live_a_boundary_merge", "aligned"
            ),
            live_a_history_forward=self.args.get(
                "sa_live_a_history_forward", "shared"
            ),
            **adaptive_a_settings,
            resume=self.args.get("sa_resume", False),
        )
        model.out_dim = 768
        return model

    def _after_backward(self, inputs=None, targets=None, optimizer=None):
        """Apply Adaptive-A only after DDP has synchronized shared-A grads."""
        if not self._sa_adaptive_a_enabled:
            return None
        raw_network = self._raw_network()
        backbone = raw_network.backbone
        strategy = getattr(self, "_sa_adaptive_a_strategy", "impact_ratio")
        if strategy == "recoverability":
            step_size = (
                float(optimizer.param_groups[0]["lr"])
                if optimizer is not None
                else 1.0
            )
            return backbone.prepare_recoverability_step(step_size=step_size)
        if strategy == "operator_energy_partition":
            return backbone.apply_operator_energy_partition_gradients(optimizer)
        if strategy == "functional_halfspace":
            momentum = (
                float(optimizer.param_groups[0].get("momentum", 0.0))
                if optimizer is not None
                else 0.0
            )
            momentum_buffers = (
                [
                    optimizer.state.get(module.weight, {}).get(
                        "momentum_buffer"
                    )
                    for module in backbone.w_As
                ]
                if optimizer is not None
                else None
            )
            stability_gradients = (
                self._functional_stability_gradients(inputs)
                if self._cur_task > 0 and inputs is not None
                else None
            )
            backbone.apply_adaptive_a_gradients(
                stability_gradients=stability_gradients,
                momentum_buffers=momentum_buffers,
                momentum=momentum,
            )
            return None
        if strategy in (
            "risk_budgeted",
            "pareto_knee",
            "function_safe_pareto",
        ):
            step_size = (
                float(optimizer.param_groups[0]["lr"])
                if optimizer is not None
                else 1.0
            )
            momentum = (
                float(optimizer.param_groups[0].get("momentum", 0.0))
                if optimizer is not None
                else 0.0
            )
            momentum_buffers = None
            if optimizer is not None:
                momentum_buffers = [
                    optimizer.state.get(module.weight, {}).get("momentum_buffer")
                    for module in backbone.w_As
                ]
            crossfit_gradients = None
            cached_modes = None
            cached_utilities = None
            stability_gradients = None
            refresh_crossfit = False
            if strategy in ("pareto_knee", "function_safe_pareto"):
                self._reset_pareto_schedule_if_needed()
                refresh_crossfit = should_refresh_pareto_crossfit(
                    self._cur_task,
                    self._sa_pareto_step,
                    getattr(self, "_sa_adaptive_a_crossfit_interval", 4),
                    self._sa_pareto_cached_modes is not None,
                )
                if refresh_crossfit:
                    if inputs is None or targets is None:
                        raise ValueError(
                            "pareto_knee crossfit refresh requires batch inputs "
                            "and targets"
                        )
                    crossfit_gradients = self._crossfit_adaptive_a_gradients(
                        inputs, targets
                    )
                    if strategy == "function_safe_pareto":
                        stability_gradients = (
                            self._functional_old_logit_gradients(inputs)
                        )
                    self._sa_pareto_crossfit_refreshes += 1
                elif self._cur_task > 0:
                    cached_modes = self._sa_pareto_cached_modes
                    cached_utilities = self._sa_pareto_cached_utilities
                    if strategy == "function_safe_pareto":
                        stability_gradients = (
                            self._sa_pareto_cached_stability_gradients
                        )
                        if stability_gradients is None:
                            raise RuntimeError(
                                "function-safe Pareto stability cache missing"
                            )
                    self._sa_pareto_cached_reuses += 1
                else:
                    self._sa_pareto_task0_skips += 1
            result = backbone.apply_adaptive_a_gradients(
                step_size=step_size,
                momentum_buffers=momentum_buffers,
                momentum=momentum,
                crossfit_gradients=crossfit_gradients,
                cached_modes=cached_modes,
                cached_utilities=cached_utilities,
                stability_gradients=stability_gradients,
            )
            if strategy in ("pareto_knee", "function_safe_pareto"):
                if refresh_crossfit and result is not None:
                    selection = result["selection"]
                    self._sa_pareto_cached_modes = list(selection["modes"])
                    self._sa_pareto_cached_utilities = list(
                        selection["selected_utilities"]
                    )
                    if strategy == "function_safe_pareto":
                        self._sa_pareto_cached_stability_gradients = [
                            None if gradient is None else gradient.detach().clone()
                            for gradient in stability_gradients
                        ]
                if strategy == "function_safe_pareto":
                    self._prepare_functional_step_diagnostic(
                        inputs, stability_gradients
                    )
                self._sa_pareto_step += 1
        else:
            momentum = (
                float(optimizer.param_groups[0].get("momentum", 0.0))
                if optimizer is not None
                else 0.0
            )
            momentum_buffers = (
                [
                    optimizer.state.get(module.weight, {}).get(
                        "momentum_buffer"
                    )
                    for module in backbone.w_As
                ]
                if optimizer is not None
                else None
            )
            backbone.apply_adaptive_a_gradients(
                momentum_buffers=momentum_buffers,
                momentum=momentum,
            )
        return None

    @staticmethod
    def _snapshot_functional_lora_state(backbone):
        parameter_names = {
            id(parameter): name for name, parameter in backbone.named_parameters()
        }
        scale = backbone.wrapped_param[0].param
        return {
            "a": {
                parameter_names[id(module.weight)]: module.weight.detach().clone()
                for module in backbone.w_As
            },
            "b": {
                parameter_names[id(module.weight)]: module.weight.detach().clone()
                for module in backbone.w_Bs
            },
            "scale_name": parameter_names[id(scale)],
            "scale": scale.detach().clone(),
        }

    @staticmethod
    def _restore_functional_lora_state(backbone, state):
        named_parameters = dict(backbone.named_parameters())
        values = {**state["a"], **state["b"]}
        values[state["scale_name"]] = state["scale"]
        with torch.no_grad():
            for name, value in values.items():
                named_parameters[name].copy_(value)

    @staticmethod
    def _relative_state_update(before, after, key):
        numerator = torch.sqrt(
            sum(
                (new.float() - old.float()).square().sum()
                for old, new in zip(
                    before[key].values(), after[key].values()
                )
            )
        )
        denominator = torch.sqrt(
            sum(old.float().square().sum() for old in before[key].values())
        )
        return float(numerator / (denominator + 1e-12))

    def _prepare_functional_step_diagnostic(self, inputs, gradients):
        interval = getattr(self, "_sa_functional_diagnostics_interval", 0)
        if not should_capture_functional_diagnostic(
            self._cur_task, self._sa_pareto_step, interval
        ):
            return
        backbone = self._raw_network().backbone
        heldout_inputs = inputs[1::2]
        before = self._snapshot_functional_lora_state(backbone)
        with rng_preserving(), torch.no_grad():
            teacher_features = self._hbd_teacher(heldout_inputs)
            teacher_output = self._functional_old_head(teacher_features)
            teacher_logits = (
                teacher_output["logits"]
                if isinstance(teacher_output, dict)
                else teacher_output
            )
        gradient_norm = torch.sqrt(
            sum(
                gradient.detach().float().square().sum()
                for gradient in gradients
                if gradient is not None
            )
        )
        self._sa_functional_diagnostic_pending = {
            "inputs": heldout_inputs.detach(),
            "teacher_logits": teacher_logits.detach(),
            "stability_gradient_norm": float(gradient_norm),
            "state": before,
        }

    def _functional_logits_for_diagnostic(self, backbone, inputs, state):
        self._restore_functional_lora_state(backbone, state)
        with self._functional_student_context(backbone):
            features = backbone(inputs)
        output = self._functional_old_head(features)
        return output["logits"] if isinstance(output, dict) else output

    def _after_optimizer_step(self, inputs=None, targets=None, optimizer=None):
        backbone = self._raw_network().backbone
        if getattr(self, "_sa_adaptive_a_strategy", None) == "recoverability":
            backbone.apply_recoverability_step(optimizer)
        elif (
            getattr(self, "_sa_adaptive_a_strategy", None)
            == "operator_energy_partition"
        ):
            backbone.apply_operator_energy_partition_step(optimizer)
        pending = getattr(self, "_sa_functional_diagnostic_pending", None)
        if pending is None:
            return None
        self._sa_functional_diagnostic_pending = None
        before = pending["state"]
        after = self._snapshot_functional_lora_state(backbone)
        diagnostic_backbone = copy.deepcopy(backbone).eval()

        def evaluate(state):
            logits = self._functional_logits_for_diagnostic(
                diagnostic_backbone, pending["inputs"], state
            )
            return float(
                old_logits_kl(
                    logits,
                    pending["teacher_logits"],
                    temperature=self._sa_functional_temperature,
                )
            )

        with rng_preserving(), torch.no_grad():
            pre_kl = evaluate(before)
            full_kl = evaluate(after)
            a_only = {
                "a": after["a"],
                "b": before["b"],
                "scale_name": before["scale_name"],
                "scale": before["scale"],
            }
            b_scale_only = {
                "a": before["a"],
                "b": after["b"],
                "scale_name": after["scale_name"],
                "scale": after["scale"],
            }
            a_only_kl = evaluate(a_only)
            b_scale_only_kl = evaluate(b_scale_only)

        teacher_probabilities = F.softmax(
            pending["teacher_logits"].float()
            / self._sa_functional_temperature,
            dim=1,
        )
        entropy = -(
            teacher_probabilities
            * torch.log(torch.clamp_min(teacher_probabilities, 1e-12))
        ).sum(dim=1).mean()
        class_count = teacher_probabilities.shape[1]
        normalized_entropy = (
            float(entropy / math.log(class_count)) if class_count > 1 else 0.0
        )
        top_values = torch.topk(
            teacher_probabilities,
            k=min(2, class_count),
            dim=1,
        ).values
        margin = (
            float((top_values[:, 0] - top_values[:, 1]).mean())
            if class_count > 1
            else 1.0
        )
        full_delta = full_kl - pre_kl
        a_delta = a_only_kl - pre_kl
        b_scale_delta = b_scale_only_kl - pre_kl
        values = {
            "teacher_normalized_entropy": normalized_entropy,
            "teacher_max_probability": float(
                teacher_probabilities.max(dim=1).values.mean()
            ),
            "teacher_probability_margin": margin,
            "pre_kl": pre_kl,
            "full_delta_kl": full_delta,
            "a_only_delta_kl": a_delta,
            "b_scale_only_delta_kl": b_scale_delta,
            "interaction_delta_kl": full_delta - a_delta - b_scale_delta,
            "stability_gradient_norm": pending["stability_gradient_norm"],
            "relative_a_update": self._relative_state_update(before, after, "a"),
            "relative_b_update": self._relative_state_update(before, after, "b"),
            "relative_scale_update": float(
                torch.linalg.vector_norm(
                    after["scale"].float() - before["scale"].float()
                )
                / (torch.linalg.vector_norm(before["scale"].float()) + 1e-12)
            ),
            "nonincreasing_full_kl": float(full_delta <= 0.0),
        }
        keys = list(values)
        packed = torch.tensor(
            [values[key] for key in keys], device=after["scale"].device
        )
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            packed.div_(dist.get_world_size())
        observation = {
            key: float(value) for key, value in zip(keys, packed.cpu().tolist())
        }
        modes = list(getattr(backbone, "_adaptive_a_last_modes", {}).values())
        for mode in ("frozen", "tangent", "live"):
            observation["selected_{}_fraction".format(mode)] = (
                modes.count(mode) / len(modes) if modes else 0.0
            )
        self._sa_functional_diagnostic_observations.append(observation)
        return None

    def _functional_stability_gradients(self, inputs):
        """Return DDP-averaged task-start-teacher gradients for shared A only."""
        if self._cur_task == 0:
            return None
        if self._hbd_teacher is None:
            raise RuntimeError(
                "functional_halfspace teacher missing for task {}; snapshot "
                "must be created before training".format(self._cur_task)
            )
        raw_network = self._raw_network()
        backbone = raw_network.backbone
        parameters = [module.weight for module in backbone.w_As]
        wrappers = [
            block.attn.qkv
            for block in getattr(backbone.lora_vit, "blocks", [])
            if hasattr(block.attn.qkv, "capture_input_sketch")
        ] if hasattr(backbone, "lora_vit") else []
        capture_states = [wrapper.capture_input_sketch for wrapper in wrappers]
        for wrapper in wrappers:
            wrapper.capture_input_sketch = False
        student_captures = []
        handles = register_live_a_historical_capture_hooks(
            backbone, student_captures
        )
        try:
            with rng_preserving(), torch.enable_grad():
                backbone(inputs)
                with torch.no_grad():
                    teacher_outputs = live_a_historical_outputs(
                        self._hbd_teacher,
                        inputs,
                        self._hbd_teacher_captures,
                    )
                distance = hbd_historical_branch_distance(
                    student_captures, teacher_outputs
                )
                gradients = list(
                    torch.autograd.grad(
                        distance, parameters, allow_unused=True
                    )
                )
        finally:
            for handle in handles:
                handle.remove()
            for wrapper, capture_state in zip(wrappers, capture_states):
                wrapper.capture_input_sketch = capture_state
        if dist.is_initialized() and dist.get_world_size() > 1:
            for index, gradient in enumerate(gradients):
                if gradient is not None:
                    gradient = gradient.contiguous()
                    dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
                    gradient.div_(dist.get_world_size())
                    gradients[index] = gradient
        return gradients

    def _functional_old_logit_gradients(self, inputs):
        """Return historical-logit or hybrid stability gradients for shared A."""
        if self._cur_task == 0:
            return None
        if self._hbd_teacher is None or self._functional_old_head is None:
            raise RuntimeError(
                "function-safe Pareto teacher backbone/head missing for task "
                "{}".format(self._cur_task)
            )
        if inputs.shape[0] < 2:
            raise ValueError(
                "function-safe Pareto stability requires at least two samples"
            )
        heldout_inputs = inputs[1::2]
        raw_network = self._raw_network()
        backbone = raw_network.backbone
        parameters = [module.weight for module in backbone.w_As]
        wrappers = (
            [
                block.attn.qkv
                for block in getattr(backbone.lora_vit, "blocks", [])
                if hasattr(block.attn.qkv, "capture_input_sketch")
            ]
            if hasattr(backbone, "lora_vit")
            else []
        )
        capture_states = [wrapper.capture_input_sketch for wrapper in wrappers]
        for wrapper in wrappers:
            wrapper.capture_input_sketch = False
        signal = getattr(
            self, "_sa_functional_stability_signal", "historical"
        )
        student_captures = []
        student_handles = []
        if signal != "historical":
            student_handles = register_live_a_historical_capture_hooks(
                backbone, student_captures
            )
            if not self._hbd_teacher_handles:
                raise RuntimeError(
                    "hybrid functional stability requires teacher capture hooks"
                )
        try:
            with (
                rng_preserving(),
                torch.enable_grad(),
                self._functional_student_context(backbone),
            ):
                student_features = backbone(heldout_inputs)
                student_output = self._functional_old_head(student_features)
                student_logits = (
                    student_output["logits"]
                    if isinstance(student_output, dict)
                    else student_output
                )
                with torch.no_grad():
                    if signal != "historical":
                        self._hbd_teacher_captures.clear()
                    teacher_features = self._hbd_teacher(heldout_inputs)
                    teacher_historical_outputs = (
                        list(self._hbd_teacher_captures)
                        if signal != "historical"
                        else None
                    )
                    teacher_output = self._functional_old_head(teacher_features)
                    teacher_logits = (
                        teacher_output["logits"]
                        if isinstance(teacher_output, dict)
                        else teacher_output
                    )
                historical_loss = old_logits_kl(
                    student_logits,
                    teacher_logits,
                    temperature=self._sa_functional_temperature,
                )
                historical_gradients = list(
                    torch.autograd.grad(
                        historical_loss,
                        parameters,
                        retain_graph=signal != "historical",
                        allow_unused=True,
                    )
                )
                if signal == "historical":
                    gradients = historical_gradients
                else:
                    hbd_loss = hbd_historical_branch_distance(
                        student_captures, teacher_historical_outputs
                    )
                    hbd_gradients = list(
                        torch.autograd.grad(
                            hbd_loss, parameters, allow_unused=True
                        )
                    )
                    if signal == "fixed_hybrid":
                        reliability = {
                            "weight": teacher_logits.new_tensor(
                                self._sa_functional_hybrid_weight
                            ),
                            "normalized_entropy": teacher_logits.new_tensor(
                                float("nan")
                            ),
                            "augmentation_consistency": teacher_logits.new_tensor(
                                float("nan")
                            ),
                        }
                    elif signal == "reliability_hybrid":
                        if heldout_inputs.ndim < 3:
                            raise ValueError(
                                "reliability hybrid requires image-like inputs"
                            )
                        with torch.no_grad():
                            augmented_features = self._hbd_teacher(
                                torch.flip(heldout_inputs, dims=(-1,))
                            )
                            augmented_output = self._functional_old_head(
                                augmented_features
                            )
                            augmented_logits = (
                                augmented_output["logits"]
                                if isinstance(augmented_output, dict)
                                else augmented_output
                            )
                        reliability = functional_teacher_reliability(
                            teacher_logits,
                            augmented_logits,
                            temperature=self._sa_functional_temperature,
                        )
                    else:
                        raise RuntimeError(
                            "unknown functional stability signal: {}".format(
                                signal
                            )
                        )
                    gradients = blend_functional_stability_gradients(
                        historical_gradients,
                        hbd_gradients,
                        historical_weight=float(reliability["weight"]),
                    )
                    self._sa_functional_signal_observations.append(
                        {
                            "weight": float(reliability["weight"]),
                            "normalized_entropy": float(
                                reliability["normalized_entropy"]
                            ),
                            "augmentation_consistency": float(
                                reliability["augmentation_consistency"]
                            ),
                            "historical_gradient_norm": float(
                                _gradient_list_norm(historical_gradients)
                            ),
                            "hbd_gradient_norm": float(
                                _gradient_list_norm(hbd_gradients)
                            ),
                        }
                    )
        finally:
            for handle in student_handles:
                handle.remove()
            for wrapper, capture_state in zip(wrappers, capture_states):
                wrapper.capture_input_sketch = capture_state
        gradients = [
            None if gradient is None else gradient.detach()
            for gradient in gradients
        ]
        if dist.is_initialized() and dist.get_world_size() > 1:
            for index, gradient in enumerate(gradients):
                if gradient is not None:
                    gradient = gradient.contiguous()
                    dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
                    gradient.div_(dist.get_world_size())
                    gradients[index] = gradient
        return gradients

    def _functional_student_context(self, backbone):
        scope = getattr(self, "_sa_functional_student_scope", "full")
        if scope == "full":
            return nullcontext()
        if scope == "historical":
            if not hasattr(backbone, "historical_only_forward"):
                raise RuntimeError(
                    "historical functional student scope requires a backbone "
                    "with historical_only_forward()"
                )
            return backbone.historical_only_forward()
        raise RuntimeError("unknown functional student scope: {}".format(scope))

    def _reset_pareto_schedule_if_needed(self):
        """Keep the transient Pareto decision cache strictly task-local."""
        if getattr(self, "_sa_pareto_schedule_task_id", None) == self._cur_task:
            return
        self._sa_pareto_schedule_task_id = self._cur_task
        self._sa_pareto_step = 0
        self._sa_pareto_cached_modes = None
        self._sa_pareto_cached_utilities = None
        self._sa_pareto_cached_stability_gradients = None
        self._sa_pareto_crossfit_refreshes = 0
        self._sa_pareto_cached_reuses = 0
        self._sa_pareto_task0_skips = 0
        self._sa_functional_diagnostic_pending = None
        self._sa_functional_diagnostic_observations = []
        self._sa_functional_signal_observations = []

    def _crossfit_adaptive_a_gradients(self, inputs, targets):
        """Compute synchronized two-fold CE gradients for Pareto utility."""
        raw_network = self._raw_network()
        backbone = raw_network.backbone
        parameters = [module.weight for module in backbone.w_As]
        wrappers = [
            block.attn.qkv
            for block in getattr(backbone.lora_vit, "blocks", [])
            if hasattr(block.attn.qkv, "capture_input_sketch")
        ] if hasattr(backbone, "lora_vit") else []
        capture_states = [wrapper.capture_input_sketch for wrapper in wrappers]
        for wrapper in wrappers:
            wrapper.capture_input_sketch = False
        sync_context = (
            self._network.no_sync()
            if hasattr(self._network, "no_sync")
            else nullcontext()
        )
        try:
            with sync_context, rng_preserving():
                folds = crossfit_classification_gradients(
                    self._network,
                    parameters,
                    inputs,
                    targets,
                    known_classes=self._known_classes,
                )
        finally:
            for wrapper, capture_state in zip(wrappers, capture_states):
                wrapper.capture_input_sketch = capture_state
        if dist.is_initialized() and dist.get_world_size() > 1:
            for fold in folds:
                for gradient in fold:
                    if gradient is not None:
                        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
                        gradient.div_(dist.get_world_size())
        return folds

    def _log_adaptive_a_diagnostics(self):
        """Log task-local Adaptive-A aggregates without serializing them."""
        diagnostics = self._raw_network().backbone.adaptive_a_diagnostics()
        if diagnostics is None:
            return
        if diagnostics.get("strategy") == "operator_energy_partition":
            logging.info(
                "[HOEP-A] task %d: budget=%.6f selected=%d/%d "
                "energy_ratio=%.6e steps=%d init_equiv=%.6e "
                "orth_err=%.6e rowspace_err=%.6e hist_equiv=%.6e "
                "current_equiv=%.6e",
                self._cur_task,
                diagnostics["energy_budget"],
                diagnostics["selected_directions"],
                diagnostics["total_directions"],
                diagnostics["selected_energy_ratio"],
                diagnostics["steps"],
                diagnostics["max_initial_operator_error"],
                diagnostics["max_orthogonality_error"],
                diagnostics["max_rowspace_reconstruction_error"],
                diagnostics["max_historical_operator_error"],
                diagnostics["max_current_operator_error"],
            )
            if "functional_jaccard" in diagnostics:
                logging.info(
                    "[Functional-HOEP] task %d: metric=%s jaccard=%.6f "
                    "spearman=%.6f operator_mask_functional_ratio=%.6f "
                    "functional_selected=%d/%d",
                    self._cur_task,
                    diagnostics["energy_metric"],
                    diagnostics["functional_jaccard"],
                    diagnostics["functional_spearman"],
                    diagnostics["operator_mask_functional_energy_ratio"],
                    diagnostics["functional_selected_directions"],
                    diagnostics["total_directions"],
                )
                self._save_hoep_functional_diagnostic()
            return
        if diagnostics.get("strategy") == "recoverability":
            logging.info(
                "[Recoverability-AdaptiveA] task %d: stage=%s observations=%d "
                "budget=%.6e step_size=%.6e interval=%d sketch_rank=%d "
                "mean_gamma=%.6f min_gamma=%.6f max_gamma=%.6f "
                "zero_fraction=%.6f full_fraction=%.6f mean_risk=%.6e "
                "max_risk=%.6e mean_utility=%.6e",
                self._cur_task,
                diagnostics["stage"],
                diagnostics["observations"],
                diagnostics["budget"],
                diagnostics["step_size"],
                diagnostics["interval"],
                diagnostics["sketch_rank"],
                diagnostics["mean_gamma"],
                diagnostics["min_gamma"],
                diagnostics["max_gamma"],
                diagnostics["fraction_zero_gamma"],
                diagnostics["fraction_full_gamma"],
                diagnostics["mean_risk"],
                diagnostics["max_risk"],
                diagnostics["mean_utility"],
            )
            return
        if diagnostics.get("strategy") == "functional_halfspace":
            logging.info(
                "[FunctionalHalfspace-AdaptiveA] task %d: observations=%d "
                "conflicts=%d normal_projections=%d full_fallbacks=%d "
                "degenerate_noops=%d mean_pre_inner=%.6e "
                "mean_post_inner=%.6e mean_normal_fraction=%.6e "
                "mean_correction_ratio=%.6e",
                self._cur_task,
                diagnostics["observations"],
                diagnostics["conflicts"],
                diagnostics["normal_projections"],
                diagnostics["full_fallbacks"],
                diagnostics["degenerate_noops"],
                diagnostics["mean_pre_inner"],
                diagnostics["mean_post_inner"],
                diagnostics["mean_normal_fraction"],
                diagnostics["mean_correction_ratio"],
            )
            return
        logging.info(
            "[AdaptiveA-SDLoRA] task %d: observations=%d mean_gate=%.6f "
            "min_gate=%.6f max_gate=%.6f fraction_gate_below_0_1=%.6f "
            "fraction_gate_above_0_9=%.6f mean_current_impact=%.6e "
            "mean_historical_impact=%.6e mean_perpendicular_retention=%.6f "
            "per_layer_mean_gate=%s",
            self._cur_task,
            diagnostics["observations"],
            diagnostics["mean_gate"],
            diagnostics["min_gate"],
            diagnostics["max_gate"],
            diagnostics["fraction_gate_below_0_1"],
            diagnostics["fraction_gate_above_0_9"],
            diagnostics["mean_current_impact"],
            diagnostics["mean_historical_impact"],
            diagnostics["mean_perpendicular_retention"],
            diagnostics["per_layer_mean_gate"],
        )
        if diagnostics.get("strategy") == "risk_budgeted":
            logging.info(
                "[RiskBudgeted-AdaptiveA] task %d: mode_counts=%s "
                "mode_fractions=%s mean_signed_utility=%.6e "
                "mean_operator_risk=%.6e configured_budget=%.6e "
                "effective_budget=%.6e budget_mode=%s live_risk=%.6e",
                self._cur_task,
                diagnostics["mode_counts"],
                diagnostics["mode_fractions"],
                diagnostics["mean_selected_utility"],
                diagnostics["mean_selected_risk"],
                float(self.args.get("sa_adaptive_a_risk_budget", 0.05)),
                diagnostics["effective_risk_budget"],
                diagnostics["risk_budget_mode"],
                diagnostics["live_risk_reference"],
            )
        elif diagnostics.get("strategy") in (
            "pareto_knee",
            "function_safe_pareto",
        ):
            prefix = (
                "FunctionSafePareto"
                if diagnostics["strategy"] == "function_safe_pareto"
                else "ParetoKnee"
            )
            logging.info(
                "[%s-AdaptiveA] task %d: mode_counts=%s "
                "mode_fractions=%s mean_crossfit_utility=%.6e "
                "mean_operator_risk=%.6e implied_risk_ratio_mean=%.6f "
                "implied_risk_ratio_range=[%.6f,%.6f] "
                "mean_pareto_points=%.2f live_risk=%.6e "
                "crossfit_interval=%d refreshes=%d cached_reuses=%d "
                "task0_skips=%d",
                prefix,
                self._cur_task,
                diagnostics["mode_counts"],
                diagnostics["mode_fractions"],
                diagnostics["mean_selected_utility"],
                diagnostics["mean_selected_risk"],
                diagnostics["mean_implied_risk_ratio"],
                diagnostics["min_implied_risk_ratio"],
                diagnostics["max_implied_risk_ratio"],
                diagnostics["mean_pareto_points"],
                diagnostics["live_risk_reference"],
                getattr(self, "_sa_adaptive_a_crossfit_interval", 4),
                getattr(self, "_sa_pareto_crossfit_refreshes", 0),
                getattr(self, "_sa_pareto_cached_reuses", 0),
                getattr(self, "_sa_pareto_task0_skips", 0),
            )
            if diagnostics["strategy"] == "function_safe_pareto":
                safety = diagnostics["functional_safety"]
                logging.info(
                    "[FunctionSafePareto] task %d: blocks=%d conflicts=%d "
                    "normal_projections=%d full_fallbacks=%d "
                    "mean_pre_cosine=%.6f mean_correction_ratio=%.6f",
                    self._cur_task,
                    safety["observations"],
                    safety["conflicts"],
                    safety["normal_projections"],
                    safety["full_fallbacks"],
                    safety["mean_pre_cosine"],
                    safety["mean_correction_ratio"],
                )
                self._log_functional_step_diagnostics()
                self._log_functional_signal_diagnostics()

    def _log_functional_signal_diagnostics(self):
        observations = getattr(
            self, "_sa_functional_signal_observations", []
        )
        if not observations:
            return
        mean = lambda key: sum(item[key] for item in observations) / len(
            observations
        )
        logging.info(
            "[FunctionalSignal] task %d: signal=%s observations=%d "
            "historical_weight=%.6f teacher_entropy=%.6f "
            "augmentation_consistency=%.6f historical_grad_norm=%.6e "
            "hbd_grad_norm=%.6e",
            self._cur_task,
            self._sa_functional_stability_signal,
            len(observations),
            mean("weight"),
            mean("normalized_entropy"),
            mean("augmentation_consistency"),
            mean("historical_gradient_norm"),
            mean("hbd_gradient_norm"),
        )

    def _log_functional_step_diagnostics(self):
        observations = getattr(
            self, "_sa_functional_diagnostic_observations", []
        )
        if not observations:
            return
        mean = lambda key: sum(item[key] for item in observations) / len(observations)
        logging.info(
            "[FunctionSafeDiagnostic] task %d: observations=%d "
            "teacher_entropy=%.6f teacher_max_prob=%.6f teacher_margin=%.6f "
            "pre_kl=%.6e full_delta_kl=%.6e safe_step_fraction=%.6f "
            "a_only_delta_kl=%.6e b_scale_only_delta_kl=%.6e "
            "interaction_delta_kl=%.6e stability_grad_norm=%.6e "
            "relative_a_update=%.6e relative_b_update=%.6e "
            "relative_scale_update=%.6e selected_modes=[%.4f,%.4f,%.4f]",
            self._cur_task,
            len(observations),
            mean("teacher_normalized_entropy"),
            mean("teacher_max_probability"),
            mean("teacher_probability_margin"),
            mean("pre_kl"),
            mean("full_delta_kl"),
            mean("nonincreasing_full_kl"),
            mean("a_only_delta_kl"),
            mean("b_scale_only_delta_kl"),
            mean("interaction_delta_kl"),
            mean("stability_gradient_norm"),
            mean("relative_a_update"),
            mean("relative_b_update"),
            mean("relative_scale_update"),
            mean("selected_frozen_fraction"),
            mean("selected_tangent_fraction"),
            mean("selected_live_fraction"),
        )

    def _save_hoep_functional_diagnostic(self):
        record = self._raw_network().backbone.hoep_functional_diagnostic_record()
        if record is None:
            return
        path = os.path.join(
            self.args["filepath"], HOEP_FUNCTIONAL_DIAGNOSTICS_FILENAME
        )
        artifact = {"version": 1, "tasks": []}
        if os.path.exists(path):
            artifact = torch.load(path, map_location="cpu", weights_only=False)
        tasks = [
            item
            for item in artifact.get("tasks", [])
            if int(item["task_id"]) != int(record["task_id"])
        ]
        tasks.append(record)
        tasks.sort(key=lambda item: int(item["task_id"]))
        artifact["tasks"] = tasks
        torch.save(artifact, path)

    @staticmethod
    def _sbgc_up_tensors(backbone):
        return [
            tensor
            for wrapper in backbone._sbgc_wrappers()
            for tensor in (wrapper.unified_up_q, wrapper.unified_up_v)
        ]

    @staticmethod
    def _set_sbgc_up_tensors(backbone, values):
        tensors = Learner._sbgc_up_tensors(backbone)
        if len(tensors) != len(values):
            raise ValueError("SBGC attribution candidate branch count mismatch")
        with torch.no_grad():
            for destination, source in zip(tensors, values):
                if destination.shape != source.shape:
                    raise ValueError("SBGC attribution candidate shape mismatch")
                destination.copy_(source.to(destination))

    def _collect_sbgc_boundary_logits(self, raw_network, include_proto=True):
        loader = deterministic_loader(
            self._eval_test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=0,
            seed=0,
        )
        outputs = {"fc": [], "proto": []}
        targets = []
        with torch.no_grad():
            for _, inputs, batch_targets in loader:
                features = raw_network.backbone(
                    inputs.to(self._device, non_blocking=True)
                )
                outputs["fc"].append(raw_network.fc(features)["logits"].cpu())
                if include_proto and raw_network.prototype_head is not None:
                    outputs["proto"].append(
                        raw_network.prototype_head(features)["logits"].cpu()
                    )
                targets.append(batch_targets.cpu())
        return {
            name: torch.cat(parts) for name, parts in outputs.items() if parts
        }, torch.cat(targets)

    def _sbgc_candidate_prototype_head(self, data_manager, raw_network):
        path = os.path.join(self.args["filepath"], PROTOTYPES_FILENAME)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        prototypes = {
            int(class_id): vectors
            for class_id, vectors in saved.items()
            if int(class_id) < self._known_classes
        }
        if len(prototypes) != self._known_classes:
            raise RuntimeError("SBGC attribution is missing old class prototypes")
        dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
        )
        dataset = self._sbgc_split_dataset(dataset, "train")
        per_class = {
            class_id: []
            for class_id in range(self._known_classes, self._total_classes)
        }
        loader = deterministic_loader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            seed=0,
        )
        with torch.no_grad():
            for _, inputs, targets in loader:
                features = raw_network.backbone(
                    inputs.to(self._device, non_blocking=True)
                )
                if not self.args.get("sa_raw_prototypes", False):
                    features = F.normalize(features, p=2, dim=1)
                for feature, target in zip(features.cpu(), targets):
                    per_class[int(target)].append(feature)
        k = int(self.args.get("sa_k_prototypes", 1))
        for class_id, features in per_class.items():
            if not features:
                raise RuntimeError("SBGC attribution class has no calibration data")
            if k <= 1:
                prototypes[class_id] = F.normalize(
                    torch.stack(features).mean(dim=0), p=2, dim=0
                )
            else:
                prototypes[class_id] = self._cluster_prototypes(features, k)
        if len(prototypes) != self._total_classes:
            raise RuntimeError("SBGC attribution prototype count mismatch")
        if k <= 1:
            return PrototypeCosineHead(prototypes).to(self._device)
        return MultiPrototypeCosineHead(
            prototypes,
            aggregate=self.args.get("sa_k_prototype_aggregate", "max"),
        ).to(self._device)

    def _prepare_task_train_dataset(self, data_manager, dataset):
        self._sa_g_train_indices = None
        self._sa_g_validation_indices = None
        self._sa_g_task_labels = None
        if self._cur_task == 0 or self._sa_g_holdout_fraction == 0:
            return dataset
        if not hasattr(dataset, "labels"):
            raise RuntimeError("SBGC holdout requires dataset.labels")
        labels = np.asarray(dataset.labels)
        train, validation = stratified_holdout_indices(
            labels,
            self._sa_g_holdout_fraction,
            int(self.args["seed"]) + 1000 * self._cur_task,
        )
        self._sa_g_train_indices = train
        self._sa_g_validation_indices = validation
        self._sa_g_task_labels = labels.copy()
        logging.info(
            "[SBGC Holdout] task %d train=%d validation=%d fraction=%.3f",
            self._cur_task,
            len(train),
            len(validation),
            self._sa_g_holdout_fraction,
        )
        return Subset(dataset, train)

    def _sbgc_split_dataset(self, dataset, split):
        if getattr(self, "_sa_g_train_indices", None) is None:
            if split == "validation":
                raise RuntimeError("SBGC validation split is unavailable")
            return dataset
        if not hasattr(dataset, "labels") or not np.array_equal(
            np.asarray(dataset.labels), self._sa_g_task_labels
        ):
            raise RuntimeError("SBGC train/test-preprocessing sample order differs")
        indices = (
            self._sa_g_train_indices
            if split == "train"
            else self._sa_g_validation_indices
        )
        if split not in ("train", "validation"):
            raise ValueError("SBGC split must be train or validation")
        return Subset(dataset, indices)

    def _sbgc_guard_validation_ce(self, raw_network, dataset, head):
        total_loss = 0.0
        total_count = 0
        loader = deterministic_loader(
            dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=0,
            seed=0,
        )
        with torch.no_grad():
            for _, inputs, targets in loader:
                features = raw_network.backbone(inputs.to(self._device))
                logits = head(features)["logits"]
                loss = F.cross_entropy(
                    logits,
                    targets.to(self._device),
                    reduction="sum",
                )
                total_loss += float(loss)
                total_count += len(targets)
        if total_count == 0 or not math.isfinite(total_loss):
            raise RuntimeError("SBGC guard validation CE is empty or nonfinite")
        return total_loss / total_count

    def _sbgc_guard_candidates(self, backbone):
        problems = backbone._sbgc_guard_problems
        if not problems or len(problems) != len(self._sbgc_up_tensors(backbone)):
            raise RuntimeError("SBGC guard is missing branch problems")
        candidates = {}
        for budget in (0.05, 0.10, 0.20, 0.40, 0.80, None):
            name = "additive" if budget is None else f"{budget:.2f}"
            values, diagnostics = [], []
            numerator = 0.0
            denominator = 0.0
            for problem, destination in zip(
                problems, self._sbgc_up_tensors(backbone)
            ):
                if budget is None:
                    candidate = problem["target"]
                    result = {
                        "constraint_active": False,
                        "current_distortion": 0.0,
                        "eta": 0.0,
                    }
                else:
                    candidate, result = solve_sensitivity_budgeted_g(
                        problem["historical"],
                        problem["target"],
                        problem["historical_covariance"],
                        problem["current_covariance"],
                        problem["historical_sensitivity"],
                        problem["current_sensitivity"],
                        risk_budget=budget,
                        ridge=backbone.g_solver_ridge,
                        bisection_steps=backbone.g_bisection_steps,
                    )
                deployed = candidate.to(destination).detach().clone()
                risk = historical_response_risk(
                    deployed.to(torch.float64),
                    problem["historical"],
                    problem["historical_covariance"],
                    problem["historical_sensitivity"],
                )
                if budget is not None and float(risk) > budget + 1e-6:
                    raise RuntimeError("SBGC guard candidate violates its budget")
                diff = deployed.to(torch.float64) - problem["historical"]
                numerator += float(weighted_response_energy(
                    diff,
                    problem["historical_covariance"],
                    problem["historical_sensitivity"],
                ))
                denominator += float(weighted_response_energy(
                    problem["historical"],
                    problem["historical_covariance"],
                    problem["historical_sensitivity"],
                ))
                result = dict(result)
                result["achieved_risk"] = float(risk)
                result["target_risk"] = float(historical_response_risk(
                    problem["target"],
                    problem["historical"],
                    problem["historical_covariance"],
                    problem["historical_sensitivity"],
                ))
                diagnostics.append(result)
                values.append(deployed)
            candidates[name] = {
                "values": values,
                "diagnostics": diagnostics,
                "risk": numerator / (denominator + 1e-12),
                "max_branch_risk": max(
                    item["achieved_risk"] for item in diagnostics
                ),
                "budget": budget,
            }
        return candidates

    def _apply_sbgc_plasticity_guard(self, raw_network, data_manager):
        backbone = raw_network.backbone
        selected_name = None
        record = None
        if self._is_main_process():
            original = [tensor.detach().clone() for tensor in self._sbgc_up_tensors(backbone)]
            was_training = raw_network.training
            rng_before = rng_state_hash()
            try:
                raw_network.eval()
                with rng_preserving():
                    candidates = self._sbgc_guard_candidates(backbone)
                    validation = data_manager.get_dataset(
                        np.arange(self._known_classes, self._total_classes),
                        source="train",
                        mode="test",
                    )
                    validation = self._sbgc_split_dataset(
                        validation, "validation"
                    )
                    for name, candidate in candidates.items():
                        self._set_sbgc_up_tensors(backbone, candidate["values"])
                        head = self._sbgc_candidate_prototype_head(
                            data_manager, raw_network
                        )
                        candidate["loss"] = self._sbgc_guard_validation_ce(
                            raw_network, validation, head
                        )
            finally:
                self._set_sbgc_up_tensors(backbone, original)
                raw_network.train(was_training)
            if rng_state_hash() != rng_before:
                raise RuntimeError("SBGC plasticity guard changed RNG state")
            selected_name, threshold = choose_guard_candidate(
                candidates,
                candidates["additive"]["loss"],
                self._sa_g_guard_ce_tolerance,
            )
            selected = candidates[selected_name]
            self._set_sbgc_up_tensors(backbone, selected["values"])
            record = {
                "task_id": self._cur_task,
                "selected": selected_name,
                "additive_loss": candidates["additive"]["loss"],
                "loss_threshold": threshold,
                "validation_count": len(validation),
                "candidates": {
                    name: {
                        "loss": value["loss"],
                        "risk": value["risk"],
                        "max_branch_risk": value["max_branch_risk"],
                        "budget": value["budget"],
                    }
                    for name, value in candidates.items()
                },
            }
            path = os.path.join(self.args["filepath"], SBGC_GUARD_FILENAME)
            artifact = {"version": 1, "tasks": []}
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as handle:
                    artifact = json.load(handle)
            artifact["tasks"] = [
                item for item in artifact["tasks"]
                if item["task_id"] != self._cur_task
            ] + [record]
            artifact["tasks"].sort(key=lambda item: item["task_id"])
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(artifact, handle, indent=2, sort_keys=True)
                handle.write("\n")
            branches = backbone._last_sbgc_branch_diagnostics
            for branch, selected_diag in zip(
                branches, selected["diagnostics"]
            ):
                branch["pre_guard_selected"] = branch["selected"]
                branch["selected"] = selected_diag
                branch["deployed_historical_risk"] = selected_diag[
                    "achieved_risk"
                ]
            stats = backbone._last_sbgc_calibration_stats
            stats["pre_guard_max_risk"] = stats["max_achieved_risk"]
            stats["max_achieved_risk"] = selected["max_branch_risk"]
            stats["guard_selected_aggregate_risk"] = selected["risk"]
            stats["guard_selected_budget"] = selected["budget"]
            stats["active_constraints"] = sum(
                item["constraint_active"] for item in selected["diagnostics"]
            )
            stats["mean_current_distortion"] = sum(
                item["current_distortion"] for item in selected["diagnostics"]
            ) / len(selected["diagnostics"])
            logging.info(
                "[SBGC Guard] task %d selected=%s validation_ce=%.6f "
                "additive_ce=%.6f threshold=%.6f aggregate_risk=%.6f "
                "max_branch_risk=%.6f",
                self._cur_task,
                selected_name,
                selected["loss"],
                record["additive_loss"],
                threshold,
                selected["risk"],
                selected["max_branch_risk"],
            )
            backbone._sbgc_guard_selection = {
                "name": selected_name,
                "budget": selected["budget"],
                "aggregate_risk": selected["risk"],
                "max_branch_risk": selected["max_branch_risk"],
            }
        if dist.is_available() and dist.is_initialized():
            for tensor in self._sbgc_up_tensors(backbone):
                dist.broadcast(tensor, src=0)
                if not all_ranks_equal(tensor):
                    raise RuntimeError("SBGC guard G differs across DDP ranks")
        backbone._sbgc_guard_problems = None
        return record

    def _run_sbgc_boundary_attribution(self, data_manager=None):
        if not getattr(self, "_sa_g_boundary_attribution", False) or self._cur_task == 0:
            return
        if not self._is_main_process():
            return
        candidates = self._sbgc_boundary_candidates
        if candidates is None or self._sbgc_premerge_logits is None:
            raise RuntimeError("SBGC boundary attribution is missing a candidate")
        if data_manager is None:
            raise RuntimeError("SBGC boundary attribution requires the data manager")
        raw_network = self._raw_network()
        backbone = raw_network.backbone
        deployed = [tensor.detach().clone() for tensor in self._sbgc_up_tensors(backbone)]
        deployed_head = raw_network.prototype_head
        was_training = raw_network.training
        rng_before = rng_state_hash()
        modes = {}
        additive_logits = None
        try:
            raw_network.eval()
            with rng_preserving():
                for mode in ("additive", "uniform", "fisher"):
                    self._set_sbgc_up_tensors(backbone, candidates[mode])
                    raw_network.prototype_head = deployed_head
                    logits, targets = self._collect_sbgc_boundary_logits(raw_network)
                    if additive_logits is None:
                        additive_logits = logits["fc"]
                        reference_targets = targets
                    elif not torch.equal(reference_targets, targets):
                        raise RuntimeError("SBGC attribution target order changed")
                    modes[mode] = {
                        name: summarize_boundary_logits(
                            value, targets, self._known_classes
                        )
                        for name, value in logits.items()
                    }
                    if deployed_head is not None:
                        raw_network.prototype_head = (
                            self._sbgc_candidate_prototype_head(
                                data_manager, raw_network
                            )
                        )
                        recalibrated, recalibrated_targets = (
                            self._collect_sbgc_boundary_logits(raw_network)
                        )
                        if not torch.equal(reference_targets, recalibrated_targets):
                            raise RuntimeError(
                                "SBGC recalibrated target order changed"
                            )
                        modes[mode]["proto_recalibrated"] = (
                            summarize_boundary_logits(
                                recalibrated["proto"],
                                recalibrated_targets,
                                self._known_classes,
                            )
                        )
        finally:
            self._set_sbgc_up_tensors(backbone, deployed)
            raw_network.prototype_head = deployed_head
            raw_network.train(was_training)
        if rng_state_hash() != rng_before:
            raise RuntimeError("SBGC boundary attribution changed RNG state")
        if any(
            not torch.equal(destination, saved)
            for destination, saved in zip(self._sbgc_up_tensors(backbone), deployed)
        ):
            raise RuntimeError("SBGC boundary attribution changed deployed G")
        premerge_logits, premerge_targets = self._sbgc_premerge_logits
        if not torch.equal(premerge_targets, reference_targets):
            raise RuntimeError("SBGC premerge target order changed")
        difference = additive_logits - premerge_logits
        record = {
            "task_id": self._cur_task,
            "known_classes": self._known_classes,
            "total_classes": self._total_classes,
            "premerge_fc": summarize_boundary_logits(
                premerge_logits, premerge_targets, self._known_classes
            ),
            "premerge_to_additive_max_abs": float(difference.abs().max()),
            "premerge_to_additive_relative_l2": float(
                torch.linalg.vector_norm(difference)
                / torch.linalg.vector_norm(premerge_logits).clamp_min(1e-12)
            ),
            "candidates": modes,
        }
        path = os.path.join(self.args["filepath"], SBGC_ATTRIBUTION_FILENAME)
        artifact = {
            "version": 1,
            "evaluation_only": True,
            "head_policy": "fixed_deployed_and_candidate_current_only_recalibrated",
            "tasks": [],
        }
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                artifact = json.load(handle)
        artifact["tasks"] = [
            item for item in artifact["tasks"]
            if item["task_id"] != self._cur_task
        ] + [record]
        artifact["tasks"].sort(key=lambda item: item["task_id"])
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2, sort_keys=True)
            handle.write("\n")
        logging.info(
            "[SBGC Attribution] task %d premerge/additive max_abs=%.3e "
            "FC old/new additive=%.2f/%.2f uniform=%.2f/%.2f "
            "fisher=%.2f/%.2f",
            self._cur_task,
            record["premerge_to_additive_max_abs"],
            modes["additive"]["fc"]["old"]["top1"],
            modes["additive"]["fc"]["new"]["top1"],
            modes["uniform"]["fc"]["old"]["top1"],
            modes["uniform"]["fc"]["new"]["top1"],
            modes["fisher"]["fc"]["old"]["top1"],
            modes["fisher"]["fc"]["new"]["top1"],
        )
        self._sbgc_boundary_candidates = None
        self._sbgc_premerge_logits = None

    def _before_task_save(self, raw_network, train_loader):
        """Run deterministic task-boundary calibrations before persistence."""
        backbone = raw_network.backbone
        if self.args.get("sa_save_task_snapshots", False) and self._is_main_process():
            prototype_path = os.path.join(self.args["filepath"], PROTOTYPES_FILENAME)
            old_prototypes = (
                torch.load(prototype_path, map_location="cpu")
                if self._known_classes > 0 else {}
            )
            if self._known_classes > 0 and len(old_prototypes) != self._known_classes:
                raise RuntimeError("pre-merge snapshot has incomplete old prototypes")
            directory = save_pre_merge(
                self.args["filepath"], backbone, raw_network.fc,
                old_prototypes, self._cur_task, self._known_classes,
                self._total_classes, self.args,
            )
            logging.info("[TaskSnapshot] pre-merge task %d: %s", self._cur_task, directory)
        run_cuo = backbone.cumulative_merge == "cuo_lowrank"
        run_sbgc = backbone.cumulative_merge == "sensitivity_budgeted_g"
        run_hoep = (
            backbone.adaptive_a_strategy == "operator_energy_partition"
            and (
                backbone.hoep_energy_metric == "functional_diag"
                or backbone.hoep_functional_diagnostics
            )
        )
        if not (run_cuo or run_sbgc or run_hoep):
            return None
        data_manager = getattr(self, "_cuo_calibration_data_manager", None)
        if data_manager is None:
            raise RuntimeError(
                "task-boundary calibration requires the current data manager"
            )
        cur_classes = np.arange(self._known_classes, self._total_classes)
        dataset = data_manager.get_dataset(
            cur_classes, source="train", mode="test"
        )
        dataset = self._sbgc_split_dataset(dataset, "train")
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            dataset = Subset(dataset, range(rank, len(dataset), world_size))
        if run_sbgc:
            batch_size = self._sa_g_calibration_batch_size
        elif run_cuo:
            batch_size = int(
                self.args.get(
                    "sa_cuo_calibration_batch_size", self.args["batch_size"]
                )
            )
        else:
            batch_size = self._sa_hoep_activation_calibration_batch_size
        if batch_size <= 0:
            raise ValueError("task-boundary calibration batch size must be positive")

        def deployment_tensor_map(protected_only=False):
            tensors = {
                key: value
                for key, value in model_tensor_map(
                    raw_network, include_eval_scalars=False
                ).items()
                if not key.endswith("task_input_square_sum")
                and not key.endswith("pending_input_square_sum")
            }
            if not protected_only:
                return tensors
            mutable_fragments = (
                ".attn.qkv.a_q.weight",
                ".attn.qkv.a_v.weight",
                ".attn.qkv.b_q.weight",
                ".attn.qkv.b_v.weight",
                ".attn.qkv.projection_q",
                ".attn.qkv.projection_v",
                ".attn.qkv.unified_up_q",
                ".attn.qkv.unified_up_v",
                ".attn.qkv.projected_covariance_q",
                ".attn.qkv.projected_covariance_v",
                ".attn.qkv.sensitivity_q",
                ".attn.qkv.sensitivity_v",
                ".attn.qkv._sbgc_",
            )
            return {
                key: value
                for key, value in tensors.items()
                if not any(fragment in key for fragment in mutable_fragments)
            }

        def parameter_gradient_map():
            return {
                name: (
                    None
                    if parameter.grad is None
                    else parameter.grad.detach().cpu().clone()
                )
                for name, parameter in raw_network.named_parameters()
            }

        def assert_gradients_unchanged(before, after):
            if set(before) != set(after):
                raise RuntimeError("SBGC calibration changed parameter topology")
            for name in before:
                left, right = before[name], after[name]
                if left is None or right is None:
                    if left is not None or right is not None:
                        raise RuntimeError(
                            "SBGC calibration changed .grad presence for {}".format(
                                name
                            )
                        )
                elif not torch.equal(left, right):
                    raise RuntimeError(
                        "SBGC calibration changed existing .grad for {}".format(
                            name
                        )
                    )

        before_map = deployment_tensor_map(protected_only=run_sbgc)
        before_gradients = parameter_gradient_map() if run_sbgc else None
        rng_before = rng_state_hash()
        was_training = raw_network.training
        sbgc_boundary_started = None
        sbgc_calibration_started = None
        sbgc_calibration_seconds = 0.0
        sbgc_memory_baseline = 0
        if run_sbgc:
            if self._device.type == "cuda":
                torch.cuda.synchronize(self._device)
                sbgc_memory_baseline = torch.cuda.memory_allocated(self._device)
                torch.cuda.reset_peak_memory_stats(self._device)
            sbgc_boundary_started = time.perf_counter()
        raw_network.eval()
        try:
            if run_sbgc:
                backbone._sbgc_capture_boundary_candidates = (
                    getattr(self, "_sa_g_boundary_attribution", False)
                    and self._cur_task > 0
                )
                backbone._sbgc_guard_capture = (
                    getattr(self, "_sa_g_plasticity_guard", False)
                    and self._cur_task > 0
                )
                if (
                    getattr(self, "_sa_g_boundary_attribution", False)
                    and self._cur_task > 0
                    and self._is_main_process()
                ):
                    with rng_preserving():
                        pre_logits, pre_targets = self._collect_sbgc_boundary_logits(
                            raw_network, include_proto=False
                        )
                    self._sbgc_premerge_logits = (
                        pre_logits["fc"], pre_targets
                    )
            if run_sbgc:
                backbone.prepare_sbgc_calibration()
            if run_cuo:
                backbone.prepare_cuo_calibration()
            if run_hoep:
                backbone.prepare_hoep_activation_calibration()
            if run_sbgc:
                if self._device.type == "cuda":
                    torch.cuda.synchronize(self._device)
                sbgc_calibration_started = time.perf_counter()
            with rng_preserving():
                loader = deterministic_loader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=self._loader_workers(),
                    pin_memory=self._device.type == "cuda",
                    seed=0,
                )
                if run_sbgc:
                    for _, inputs, targets in loader:
                        inputs = inputs.to(self._device, non_blocking=True)
                        targets = targets.to(self._device, non_blocking=True)
                        features = backbone(inputs)
                        logits = raw_network.fc(features)["logits"]
                        if self._cur_task > 0:
                            logits = logits[
                                :, self._known_classes : self._total_classes
                            ]
                            targets = targets - self._known_classes
                        loss = F.cross_entropy(logits, targets)
                        outputs = backbone.sbgc_calibration_outputs()
                        gradients = torch.autograd.grad(
                            loss,
                            outputs,
                            retain_graph=False,
                            create_graph=False,
                            allow_unused=False,
                        )
                        backbone.accumulate_sbgc_sensitivities(
                            gradients, inputs.shape[0]
                        )
                else:
                    with torch.no_grad():
                        for _, inputs, _ in loader:
                            raw_network(
                                inputs.to(self._device, non_blocking=True)
                            )
            if run_sbgc:
                if self._device.type == "cuda":
                    torch.cuda.synchronize(self._device)
                sbgc_calibration_seconds = (
                    time.perf_counter() - sbgc_calibration_started
                )
                stats = backbone.finalize_sbgc_calibration()
                guard_record = None
                if (
                    getattr(self, "_sa_g_plasticity_guard", False)
                    and self._cur_task > 0
                ):
                    guard_record = self._apply_sbgc_plasticity_guard(
                        raw_network, data_manager
                    )
                if self._is_main_process() and getattr(
                    self, "_sa_g_boundary_attribution", False
                ):
                    self._sbgc_boundary_candidates = (
                        backbone._sbgc_boundary_candidates
                    )
                if self._device.type == "cuda":
                    torch.cuda.synchronize(self._device)
                    peak_allocated = torch.cuda.max_memory_allocated(self._device)
                else:
                    peak_allocated = 0
                runtime = torch.tensor(
                    [
                        sbgc_calibration_seconds,
                        time.perf_counter() - sbgc_boundary_started,
                        peak_allocated / (1024.0**2),
                        max(peak_allocated - sbgc_memory_baseline, 0) / (1024.0**2),
                    ],
                    dtype=torch.float64,
                    device=self._device,
                )
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(runtime, op=dist.ReduceOp.MAX)
                if self._is_main_process():
                    stats.update(
                        {
                            "calibration_seconds": float(runtime[0]),
                            "boundary_seconds": float(runtime[1]),
                            "peak_cuda_allocated_mib": float(runtime[2]),
                            "additional_peak_cuda_allocated_mib": float(runtime[3]),
                        }
                    )
                    path = os.path.join(
                        self.args["filepath"], SBGC_DIAGNOSTICS_FILENAME
                    )
                    artifact = {"version": 1, "tasks": []}
                    if os.path.exists(path):
                        artifact = torch.load(
                            path, map_location="cpu", weights_only=False
                        )
                    record = {
                        "task_id": self._cur_task,
                        "metric": backbone.g_sensitivity_metric,
                        "shadow_only": backbone.g_shadow_only,
                        "risk_budget": backbone.g_risk_budget,
                        "stats": stats,
                        "branches": backbone._last_sbgc_branch_diagnostics,
                        "plasticity_guard": guard_record,
                    }
                    tasks = [
                        item
                        for item in artifact.get("tasks", [])
                        if int(item["task_id"]) != self._cur_task
                    ]
                    tasks.append(record)
                    tasks.sort(key=lambda item: int(item["task_id"]))
                    artifact["tasks"] = tasks
                    torch.save(artifact, path)
            if run_cuo:
                backbone.finalize_cuo_calibration()
            if run_hoep:
                summaries = backbone.finalize_hoep_activation_calibration()
                logging.info(
                    "[Functional-HOEP calibration] task %d blocks=%d "
                    "tokens=%.0f finite=%s",
                    self._cur_task,
                    len(summaries),
                    min(item["token_count"] for item in summaries),
                    all(item["finite"] for item in summaries),
                )
        finally:
            raw_network.train(was_training)
        if run_hoep or run_sbgc:
            mismatch = compare_named_tensors(
                before_map,
                deployment_tensor_map(protected_only=run_sbgc),
            )
            if mismatch is not None:
                raise RuntimeError(
                    "task-boundary calibration mutated protected model tensor "
                    "{} (max_abs_diff={})".format(
                        mismatch["key"], mismatch["max_abs_diff"]
                    )
                )
        if run_sbgc:
            assert_gradients_unchanged(
                before_gradients,
                parameter_gradient_map(),
            )
        rng_after = rng_state_hash()
        if rng_before != rng_after:
            raise RuntimeError(
                "task-boundary calibration perturbed RNG state: {} -> {}".format(
                    rng_before, rng_after
                )
            )
        logging.info(
            "[TaskBoundaryCalibration] task %d tensor_hash=%s rng_hash=%s PASS",
            self._cur_task,
            hash_named_tensors(before_map),
            rng_before,
        )
        return None

    def incremental_train(self, data_manager):
        self._dual_num_tasks = data_manager.nb_tasks
        if (
            self._hbd_enabled
            or getattr(self, "_sa_functional_halfspace_enabled", False)
            or getattr(self, "_sa_function_safe_pareto_enabled", False)
        ) and self._cur_task >= 0:
            backbone = self._raw_network().backbone
            self._hbd_teacher = backbone.build_hbd_teacher()
            self._hbd_teacher_captures = []
            if (
                self._hbd_enabled
                or getattr(self, "_sa_functional_halfspace_enabled", False)
                or (
                    getattr(self, "_sa_function_safe_pareto_enabled", False)
                    and getattr(
                        self,
                        "_sa_functional_stability_signal",
                        "historical",
                    )
                    != "historical"
                )
            ):
                self._hbd_teacher_handles = (
                    register_live_a_historical_capture_hooks(
                        self._hbd_teacher, self._hbd_teacher_captures
                    )
                )
            if getattr(self, "_sa_function_safe_pareto_enabled", False):
                raw_network = self._raw_network()
                old_head = getattr(raw_network, "prototype_head", None)
                if old_head is None:
                    old_head = getattr(raw_network, "fc", None)
                if old_head is None:
                    raise RuntimeError(
                        "function-safe Pareto requires an old classifier head"
                    )
                self._functional_old_head = copy.deepcopy(old_head).eval()
                for parameter in self._functional_old_head.parameters():
                    parameter.requires_grad_(False)
            self._hbd_first_batch = True
            self._hbd_ratio_logged = False
        if self._dual_head:
            raw_network = self._raw_network()
            raw_network.set_head_mode("fc")
            # Calibration/eval scalars are eval-only; do not let them persist
            # into the next task's training-state hash.
            raw_network.dual_head = False
            raw_network.dual_lambda = 0.0
            raw_network.tau_fc = 1.0
            raw_network.tau_proto = 1.0
        # Capture the current task's features in the *previous* model state
        # before Shared-A is updated. Only current-task data is touched.
        if (
            self._lrpt_enabled
            and self._is_main_process()
            and self._cur_task >= 0
        ):
            self._lrpt_pre_features, self._lrpt_pre_targets = (
                self._extract_current_task_features(
                    data_manager,
                    task_index=self._cur_task + 1,
                    return_targets=True,
                )
            )
            raw_network = self._raw_network()
            self._lrpt_a_old = [
                w.weight.detach().cpu().clone()
                for w in raw_network.backbone.w_As
            ]
            logging.info(
                "[SharedA-SDLoRA] LRPT pre-update features captured for task %d (%d samples)",
                self._cur_task + 1,
                self._lrpt_pre_features.shape[0],
            )
        if (
            self._coordinate_stable_transport
            and self._is_main_process()
            and self._cur_task >= 0
        ):
            (
                self._coordinate_pre_features,
                self._coordinate_pre_targets,
            ) = self._extract_current_task_features(
                data_manager,
                task_index=self._cur_task + 1,
                return_targets=True,
            )
            logging.info(
                "[CoordinateStable] pre-update features captured for task "
                "%d (%d samples)",
                self._cur_task + 1,
                self._coordinate_pre_features.shape[0],
            )
        self._cuo_calibration_data_manager = data_manager
        try:
            super().incremental_train(data_manager)
        finally:
            self._cuo_calibration_data_manager = None
        if self._is_main_process():
            self._log_adaptive_a_diagnostics()
        if self._hbd_teacher is not None:
            for handle in self._hbd_teacher_handles:
                handle.remove()
            self._hbd_teacher_handles = []
            self._hbd_teacher_captures = []
            self._hbd_teacher = None
        self._functional_old_head = None
        if self._is_main_process():
            self._log_post_train_hash(self._cur_task)
            backbone = self._raw_network().backbone
            if (
                backbone.cumulative_state
                and backbone.cumulative_merge == "cuo_lowrank"
            ):
                stats = backbone._last_cuo_calibration_stats
                if stats is not None:
                    counts = cuo_state_scalar_counts(
                        num_blocks=len(backbone.lora_layer),
                        rank=backbone.rank,
                        dim=backbone.w_As[0].weight.shape[1],
                    )
                    logging.info(
                        "[CUO-LowRank] task %d tokens=%d branches=%d "
                        "max_condition=%.6e max_residual=%.6e "
                        "lora_factor_scalars=%d gram_scalars=%d "
                        "persistent_scalar_total=%d",
                        self._cur_task,
                        stats["token_count"],
                        stats["branch_count"],
                        stats["max_condition_number"],
                        stats["max_residual_norm"],
                        counts["lora_factor_scalars"],
                        counts["gram_scalars"],
                        counts["persistent_scalar_total"],
                    )
            if (
                backbone.cumulative_state
                and backbone.cumulative_merge == "sensitivity_budgeted_g"
            ):
                stats = backbone._last_sbgc_calibration_stats
                if stats is not None:
                    counts = sbgc_state_scalar_counts(
                        num_blocks=len(backbone.lora_layer),
                        rank=backbone.rank,
                        dim=backbone.w_As[0].weight.shape[1],
                    )
                    logging.info(
                        "[SBGC] task %d branches=%d active=%d "
                        "max_risk=%.6e mean_distortion=%.6e "
                        "mean_sensitivity_cv=%.6e mean_candidate_gap=%.6e "
                        "task0_operator_error=%.6e calibration_seconds=%.3f "
                        "solver_seconds=%.3f boundary_seconds=%.3f "
                        "peak_cuda_mib=%.1f additional_peak_cuda_mib=%.1f "
                        "lora_scalars=%d "
                        "covariance_scalars=%d sensitivity_scalars=%d "
                        "count_scalars=%d persistent_scalar_total=%d",
                        self._cur_task,
                        stats["branch_count"],
                        stats["active_constraints"],
                        stats["max_achieved_risk"],
                        stats["mean_current_distortion"],
                        stats["mean_sensitivity_cv"],
                        stats["mean_candidate_gap"],
                        stats["task0_operator_error"],
                        stats["calibration_seconds"],
                        stats["solver_seconds"],
                        stats["boundary_seconds"],
                        stats["peak_cuda_allocated_mib"],
                        stats["additional_peak_cuda_allocated_mib"],
                        counts["lora_factor_scalars"],
                        counts["covariance_scalars"],
                        counts["sensitivity_scalars"],
                        counts["count_scalars"],
                        counts["persistent_scalar_total"],
                    )
                    if backbone.g_train_projected:
                        logging.info(
                            "[SBGC TrainProjected] task %d mean_alpha=%.6f "
                            "min_alpha=%.6f active_fraction=%.6f "
                            "max_absorption_error=%.6e",
                            self._cur_task,
                            stats["mean_training_alpha"],
                            stats["min_training_alpha"],
                            stats["mean_active_projection_fraction"],
                            stats["max_absorption_relative_error"],
                        )
                    global_result = backbone._last_sbgc_global_diagnostics
                    if global_result is not None:
                        logging.info(
                            "[SBGC Global] task %d aggregate_risk=%.6e "
                            "max_branch_risk=%.6e eta=%.6e "
                            "current_distortion=%.6e",
                            self._cur_task,
                            global_result["deployed_aggregate_risk"],
                            stats["max_achieved_risk"],
                            global_result["eta"],
                            global_result["current_distortion"],
                        )
            if (
                backbone.cumulative_state
                and backbone.cumulative_merge == "live_a_aggregate_b"
            ):
                stats = getattr(backbone, "_last_live_a_save_stats", None)
                if stats is not None:
                    logging.info(
                        "[LiveA-SDLoRA] absorption task %d: mode=%s "
                        "norm_A=%.4e norm_B=%.4e scaling=%.4f gamma=%.4e "
                        "norm_product=%.4e consolidation_gain=%.4f "
                        "gain_range=[%.4f,%.4f] "
                        "absorption_relative_error=%.6e mean_G=%.4e",
                        self._cur_task,
                        stats["absorb_mode"],
                        stats["mean_A_norm"],
                        stats["mean_B_norm"],
                        stats["scale"],
                        stats["mean_gamma"],
                        stats["mean_norm_product"],
                        stats["mean_consolidation_gain"],
                        stats["min_consolidation_gain"],
                        stats["max_consolidation_gain"],
                        stats["absorption_relative_error"],
                        stats["mean_G_norm"],
                    )
        if self._is_main_process():
            if self._cur_task > 0:
                with torch.no_grad():
                    backbone = self._raw_network().backbone
                    if backbone.cumulative_state:
                        if (
                            backbone.cumulative_merge
                            == "live_a_aggregate_b"
                        ):
                            coordinate_stats = getattr(
                                backbone,
                                "_last_live_a_coordinate_diagnostics",
                                None,
                            )
                            if backbone.live_a_boundary_merge == "joint_svd":
                                joint_stats = getattr(
                                    backbone,
                                    "_last_live_a_joint_svd_diagnostics",
                                    None,
                                )
                                if joint_stats is not None:
                                    logging.info(
                                        "[JointAB] boundary task %d: branches=%d "
                                        "mean_relative_truncation=%.6e "
                                        "max_relative_truncation=%.6e",
                                        self._cur_task,
                                        joint_stats["branches"],
                                        joint_stats["mean_relative_error"],
                                        joint_stats["max_relative_error"],
                                    )
                            elif coordinate_stats is not None:
                                logging.info(
                                    "[CoordinateStable] operator alignment "
                                    "task %d: branches=%d before=%.6e "
                                    "after=%.6e max_condition=%.3f",
                                    self._cur_task,
                                    coordinate_stats["branches"],
                                    coordinate_stats[
                                        "before_relative_error"
                                    ],
                                    coordinate_stats[
                                        "after_relative_error"
                                    ],
                                    coordinate_stats["max_condition"],
                                )
                                if (
                                    coordinate_stats.get(
                                        "predicted_energy_budget"
                                    )
                                    is not None
                                ):
                                    logging.info(
                                        "[HOEP-A] boundary task %d: "
                                        "global_squared_residual=%.6e "
                                        "predicted_budget=%.6e within_bound=%s",
                                        self._cur_task,
                                        coordinate_stats[
                                            "global_squared_residual_ratio"
                                        ],
                                        coordinate_stats[
                                            "predicted_energy_budget"
                                        ],
                                        coordinate_stats[
                                            "global_squared_residual_ratio"
                                        ]
                                        <= coordinate_stats[
                                            "predicted_energy_budget"
                                        ]
                                        + 1e-6,
                                    )
                        elif backbone.cumulative_merge == "union_svd":
                            trunc_error = getattr(
                                backbone,
                                "_last_union_svd_truncation_error",
                                None,
                            )
                            logging.info(
                                "[SharedA-SDLoRA] union-svd task %d: "
                                "max_relative_truncation_error=%.6e",
                                self._cur_task,
                                trunc_error
                                if trunc_error is not None
                                else float("nan"),
                            )
                        else:
                            gauge_diag = getattr(
                                backbone,
                                "_last_cumulative_gauge_diagnostics",
                                None,
                            )
                            if gauge_diag is None:
                                logging.warning(
                                    "[SharedA-SDLoRA] pre-save cumulative "
                                    "gauge diagnostics missing; logging zeros"
                                )
                                gauge_diag = {
                                    "residual": 0.0,
                                    "rotation_fro": 0.0,
                                    "preservation": 0.0,
                                    "branches": 0,
                                }
                            logging.info(
                                "[SharedA-SDLoRA] cumulative gauge task %d: "
                                "relative_projection_residual=%.6e "
                                "basis_rotation_fro=%.6e "
                                "operator_preservation=%.6e",
                                self._cur_task,
                                gauge_diag["residual"],
                                gauge_diag["rotation_fro"],
                                gauge_diag["preservation"],
                            )
                    else:
                        operator_drift = float(
                            backbone.old_operator_stability_loss().item()
                        )
                        logging.info(
                            "[SharedA-SDLoRA] operator-stability task %d: "
                            "relative_effective_drift=%.6f lambda=%.4g",
                            self._cur_task,
                            operator_drift,
                            self._sa_operator_stability_lambda,
                        )
            if self._coordinate_stable_transport:
                self._rebuild_eval_backbone()
                logging.info(
                    "[CoordinateStable] rebuilt task %d deployment state "
                    "before prototype transport",
                    self._cur_task,
                )
            elif self._cur_task == data_manager.nb_tasks - 1:
                self._rebuild_eval_backbone()
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
        if self.args.get("sa_use_prototype_classifier", False):
            if self._is_main_process():
                if (
                    self._coordinate_stable_transport
                    and self._coordinate_pre_features is not None
                ):
                    self._apply_coordinate_transport_to_old_prototypes(
                        data_manager
                    )
                elif (
                    self._lrpt_enabled
                    and self._lrpt_pre_features is not None
                ):
                    self._apply_lrpt_to_old_prototypes(data_manager)
                prototypes = self._compute_prototypes(
                    data_manager, raw_network
                )
            else:
                prototypes = {}
            prototypes = broadcast_prototypes(prototypes, src=0)
            raw_network = self._raw_network()
            raw_network.set_prototypes(prototypes)
            prototype_head = raw_network.prototype_head
            if prototype_head is not None and hasattr(
                prototype_head, "weight"
            ):
                if not all_ranks_equal(prototype_head.weight):
                    raise RuntimeError(
                        "prototype head differs across ranks after broadcast"
                    )
            if self._is_main_process():
                logging.info(
                    "[SharedA-SDLoRA] prototype classifier active for %d classes",
                    len(prototypes),
                )
                logging.info("[PrototypeSync] rank consistency PASS")
                raw_network.backbone.cleanup_per_task_files(
                    self.args["filepath"]
                )
        if self._dual_head:
            self._prepare_dual_head(data_manager, self._raw_network())
        self._run_sbgc_boundary_attribution(data_manager)
        if self.args.get("sa_save_task_snapshots", False):
            if self._is_main_process():
                directory = save_post_merge(
                    self.args["filepath"], self._cur_task, self.args
                )
                logging.info("[TaskSnapshot] post-merge task %d: %s", self._cur_task, directory)
            self._barrier()

    def _p0_hash_store(self):
        """JSON file holding per-task tensor/RNG hashes for cross-run audits."""
        return os.path.join(self.args["filepath"], P0_HASHES_FILENAME)

    def _store_p0_hash(self, task_id, kind, model_hash, rng_hash):
        path = self._p0_hash_store()
        data = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        data.setdefault(str(task_id), {})[kind + "_model_hash"] = model_hash
        data[str(task_id)][kind + "_rng_hash"] = rng_hash
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _log_post_train_hash(self, task_id):
        """Canonical tensor/RNG hash immediately after training, before eval."""
        raw_network = self._raw_network()
        tensors = model_tensor_map(raw_network, include_eval_scalars=False)
        model_hash = hash_named_tensors(tensors)
        rng_hash = rng_state_hash()
        self._store_p0_hash(task_id, "post_train", model_hash, rng_hash)
        logging.info(
            "[PostTrainHash] task %d hash=%s rng=%s",
            task_id,
            model_hash,
            rng_hash,
        )

    def _log_post_eval_hash(self, task_id):
        """Canonical tensor/RNG hash after calibration and evaluation."""
        raw_network = self._raw_network()
        tensors = model_tensor_map(raw_network, include_eval_scalars=False)
        model_hash = hash_named_tensors(tensors)
        rng_hash = rng_state_hash()
        self._store_p0_hash(task_id, "post_eval", model_hash, rng_hash)
        logging.info(
            "[PostEvalHash] task %d hash=%s rng=%s",
            task_id,
            model_hash,
            rng_hash,
        )

    def _assert_eval_tensor_invariance(self, before_map, task_id):
        after_map = model_tensor_map(self._raw_network())
        mismatch = compare_named_tensors(before_map, after_map)
        if mismatch is not None:
            raise RuntimeError(
                "eval mutated model tensors: first mismatch {} "
                "(max_abs_diff={})".format(
                    mismatch["key"], mismatch["max_abs_diff"]
                )
            )
        logging.info(
            "[EvalTensorHash] task %d before=%s after=%s PASS",
            task_id,
            hash_named_tensors(before_map),
            hash_named_tensors(after_map),
        )

    def eval_task(self):
        if not self._is_main_process():
            return None, None
        if not self._dual_head:
            before_map = model_tensor_map(self._raw_network())
            rng_before = rng_state_hash()
            result = super().eval_task()
            rng_after = rng_state_hash()
            if rng_before != rng_after:
                raise RuntimeError(
                    "control eval perturbed RNG state: {} -> {}".format(
                        rng_before, rng_after
                    )
                )
            logging.info(
                "[RNGHash] task %d eval before=%s after=%s PASS",
                self._cur_task,
                rng_before,
                rng_after,
            )
            self._assert_eval_tensor_invariance(before_map, self._cur_task)
            self._log_post_eval_hash(self._cur_task)
            return result

        before_map = model_tensor_map(self._raw_network())
        rng_before = rng_state_hash()
        loader = deterministic_loader(
            self._eval_test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=self._loader_workers(),
            seed=0,
        )
        preds, y_true, fused_proto_diff = evaluate_dual_head_once(
            self._raw_network(),
            loader,
            device=self._device,
            topk=self.topk,
        )
        if float(self._raw_network().dual_lambda) >= 1.0:
            logging.info(
                "[DualHead] task %d fused_proto_max_diff=%.3e PASS",
                self._cur_task,
                fused_proto_diff,
            )
        rng_after = rng_state_hash()
        if rng_before != rng_after:
            raise RuntimeError(
                "dual eval perturbed RNG state: {} -> {}".format(
                    rng_before, rng_after
                )
            )
        logging.info(
            "[RNGHash] task %d eval before=%s after=%s PASS",
            self._cur_task,
            rng_before,
            rng_after,
        )
        self._assert_eval_tensor_invariance(before_map, self._cur_task)
        if fused_proto_diff > 0.0:
            logging.info(
                "[DualHead] task %d fused_proto_max_diff=%.3e PASS",
                self._cur_task,
                fused_proto_diff,
            )
        accuracies = {}
        for mode in preds:
            accuracies[mode] = self._evaluate(
                np.concatenate(preds[mode]), np.concatenate(y_true)
            )
            logging.info(
                "[DualHead] task %d mode=%s top1=%.2f top5=%.2f",
                self._cur_task,
                mode,
                accuracies[mode]["top1"],
                accuracies[mode]["top5"],
            )
        self._log_post_eval_hash(self._cur_task)
        return accuracies["fused"], None

    def _dual_lambda(self, task_id, num_tasks):
        if num_tasks <= 1:
            return 1.0
        progress = task_id / (num_tasks - 1)
        width = 4.0 / 9.0
        start = 0.0 if self._dual_schedule == "A" else 1.0 / 9.0
        return min(1.0, max(0.0, (progress - start) / width))

    def _loader_workers(self):
        if self.args.get("sa_deterministic_training", False):
            return 0
        return num_workers

    @staticmethod
    def _fit_dual_temperature(logits, targets, lo=0.05, hi=5.0, steps=60):
        best_tau, best_loss = 1.0, float("inf")
        for tau in torch.linspace(lo, hi, steps).tolist():
            loss = F.cross_entropy(logits / tau, targets).item()
            if loss < best_loss:
                best_loss, best_tau = loss, tau
        return best_tau

    def _prepare_dual_head(self, data_manager, raw_network):
        lambda_val, tau_fc, tau_proto = 0.0, 1.0, 1.0
        if self._is_main_process():
            raw_network.eval()
            before_map = model_tensor_map(raw_network)
            rng_before = rng_state_hash()
            with rng_preserving(), torch.no_grad():
                if (
                    getattr(self, "_last_proto_task", None) == self._cur_task
                    and getattr(self, "_last_proto_features", None) is not None
                ):
                    features = self._last_proto_features
                    targets = self._last_proto_targets
                else:
                    features, targets = self._extract_current_task_features(
                        data_manager, return_targets=True
                    )
                features = features.to(self._device)
                targets = targets.to(self._device)
                fc_logits = raw_network.fc(features)["logits"]
                proto_logits = raw_network.prototype_head(features)["logits"]
            tau_fc = self._fit_dual_temperature(fc_logits, targets)
            tau_proto = self._fit_dual_temperature(proto_logits, targets)
            lambda_val = self._dual_lambda(
                self._cur_task, self._dual_num_tasks
            )
            mismatch = compare_named_tensors(
                before_map, model_tensor_map(raw_network)
            )
            if mismatch is not None:
                raise RuntimeError(
                    "Dual-head calibration mutated model tensors: first "
                    "mismatch {} (max_abs_diff={})".format(
                        mismatch["key"], mismatch["max_abs_diff"]
                    )
                )
            rng_after = rng_state_hash()
            if rng_before != rng_after:
                raise RuntimeError(
                    "Dual-head calibration perturbed RNG state: {} -> {}".format(
                        rng_before, rng_after
                    )
                )
            logging.info(
                "[RNGHash] task %d calibration before=%s after=%s PASS",
                self._cur_task,
                rng_before,
                rng_after,
            )
            logging.info(
                "[DualHead] calibration parameter invariance PASS "
                "(max_abs_diff=0.000e+00)"
            )
        lambda_val, tau_fc, tau_proto = broadcast_dual_head_values(
            lambda_val, tau_fc, tau_proto, src=0
        )
        raw_network.set_dual_head(lambda_val, tau_fc, tau_proto)
        if not all_ranks_equal(
            torch.tensor(
                [lambda_val, tau_fc, tau_proto], dtype=torch.float64
            )
        ):
            raise RuntimeError(
                "dual-head calibration scalars differ across ranks"
            )
        print(
            "[DualHead] rank {} task {} lambda={:.6f} tau_fc={:.6f} "
            "tau_proto={:.6f}".format(
                self.args.get("rank", 0),
                self._cur_task,
                lambda_val,
                tau_fc,
                tau_proto,
            ),
            flush=True,
        )
        if self._is_main_process():
            state = {
                "schedule": self._dual_schedule,
                "task_id": self._cur_task,
                "num_tasks": self._dual_num_tasks,
                "lambda": lambda_val,
                "tau_fc": tau_fc,
                "tau_proto": tau_proto,
            }
            torch.save(
                state,
                os.path.join(self.args["filepath"], "sa_dual_head.pt"),
            )
            logging.info(
                "[DualHead] task %d schedule=%s lambda=%.3f tau_fc=%.3f "
                "tau_proto=%.3f",
                self._cur_task,
                self._dual_schedule,
                lambda_val,
                tau_fc,
                tau_proto,
            )

    def _additional_training_losses(
        self, inputs=None, targets=None, features=None
    ):
        losses = {}
        if self._hbd_enabled and self._cur_task > 0:
            hbd_loss = self._hbd_training_loss(
                inputs, targets, features
            )
            if hbd_loss is not None:
                losses["hbd"] = hbd_loss
        if self._sa_operator_stability_lambda > 0 and self._cur_task > 0:
            operator_loss = (
                self._raw_network().backbone.old_operator_stability_loss()
            )
            losses["operator_stability"] = (
                self._sa_operator_stability_lambda * operator_loss
            )
        w = self._sa_prototype_consistency_weight
        if w <= 0 or features is None or targets is None:
            return losses
        if self._proto_ema_task != self._cur_task:
            self._proto_ema = {}
            self._proto_ema_task = self._cur_task
        feats = F.normalize(features.float(), p=2, dim=1)
        momentum = 0.1
        with torch.no_grad():
            for f, t in zip(feats, targets):
                c = int(t.item())
                if c not in self._proto_ema:
                    self._proto_ema[c] = f.clone()
                else:
                    self._proto_ema[c] = F.normalize(
                        (1.0 - momentum) * self._proto_ema[c]
                        + momentum * f,
                        p=2,
                        dim=0,
                    )
        proto_matrix = torch.stack(
            [self._proto_ema[int(t.item())] for t in targets]
        ).to(feats.device)
        cosine = (feats * proto_matrix).sum(dim=1)
        losses["prototype_consistency"] = w * (1.0 - cosine).mean()
        return losses

    def _hbd_training_loss(
        self, inputs=None, targets=None, features=None
    ):
        """Historical-Branch Activation Distillation loss (task t > 0).

        Runs one extra autograd forward over the live backbone to collect the
        per-block historical-branch responses (``G_prev * normalize(A_live) * x``)
        and compares them with the frozen task-start teacher responses
        (``G_prev * normalize(A_prev) * x_teacher``).  Only the historical
        branch is constrained; fresh ``B_t`` receives no HBD gradient.
        """
        if not self._hbd_enabled or self._cur_task == 0:
            return None
        if self._hbd_teacher is None:
            raise RuntimeError(
                "HBD teacher missing for task {}; snapshot must be created "
                "before training".format(self._cur_task)
            )
        raw_network = self._raw_network()
        backbone = raw_network.backbone
        student_captures = []
        handles = register_live_a_historical_capture_hooks(
            backbone, student_captures
        )
        try:
            with torch.enable_grad():
                backbone(inputs)
        finally:
            for handle in handles:
                handle.remove()
        with torch.no_grad():
            teacher_outputs = live_a_historical_outputs(
                self._hbd_teacher, inputs, self._hbd_teacher_captures
            )
        distance = hbd_historical_branch_distance(
            student_captures, teacher_outputs
        )
        hbd_loss = self._hbd_lambda * distance
        if self._hbd_first_batch:
            self._hbd_first_batch = False
            self._log_hbd_first_batch(
                raw_network, inputs, targets, features, label="first"
            )
        if (
            not self._hbd_ratio_logged
            and float(distance.detach()) > 1e-8
        ):
            self._hbd_ratio_logged = True
            self._log_hbd_first_batch(
                raw_network,
                inputs,
                targets,
                features,
                label="first_drift",
            )
        return hbd_loss

    def _log_hbd_first_batch(
        self, raw_network, inputs, targets, features, label="first"
    ):
        """Record CE/HBD values and the CE-vs-HBD gradient norm ratio on A.

        The ratio is computed on a deep-copied network so the diagnostic
        backwards never trigger DDP reduction hooks on the live parameters.
        RNG is restored afterwards so the diagnostic does not perturb the
        training trajectory.
        """
        if features is None:
            return
        with rng_preserving():
            # Deep-copy only the modules that contribute to the CE/HBD graphs
            # (the full network holds unpicklable runtime references).
            clone = copy.deepcopy(raw_network.backbone)
            clone_fc = copy.deepcopy(raw_network.fc)
            clone.train()
            clone_feats = clone(inputs)
            clone_logits = clone_fc(clone_feats)["logits"]
            ce_loss = F.cross_entropy(
                clone_logits[:, self._known_classes :],
                targets - self._known_classes,
            )
            clone_captures = []
            clone_handles = register_live_a_historical_capture_hooks(
                clone, clone_captures
            )
            try:
                with torch.enable_grad():
                    clone(inputs)
            finally:
                for handle in clone_handles:
                    handle.remove()
            with torch.no_grad():
                teacher_outputs = live_a_historical_outputs(
                    self._hbd_teacher, inputs, self._hbd_teacher_captures
                )
            clone_distance = hbd_historical_branch_distance(
                clone_captures, teacher_outputs
            )
            clone_hbd_loss = self._hbd_lambda * clone_distance
            a_params = [w.weight for w in clone.w_As]
            ce_norm, hbd_norm = 0.0, 0.0
            for parameter in a_params:
                parameter.grad = None
            ce_loss.backward(retain_graph=True)
            ce_norm = sum(
                p.grad.detach().square().sum().item() for p in a_params
            ) ** 0.5
            for parameter in a_params:
                parameter.grad = None
            clone_hbd_loss.backward(retain_graph=True)
            hbd_norm = sum(
                p.grad.detach().square().sum().item() for p in a_params
            ) ** 0.5
            for parameter in a_params:
                parameter.grad = None
        if self._is_main_process():
            logging.info(
                "[HBD] task %d %s batch: CE=%.4f HBD=%.4f "
                "dL/dA_ce=%.4e dL/dA_hbd=%.4e ratio_hbd_ce=%.4f",
                self._cur_task,
                label,
                ce_loss.detach().item(),
                clone_hbd_loss.detach().item(),
                ce_norm,
                hbd_norm,
                hbd_norm / max(ce_norm, 1e-12),
            )

    def _extract_current_task_features(
        self, data_manager, task_index=None, normalize=False, return_targets=False
    ):
        """Extract features of the current task's training data (test transform,
        no augmentation) with the current model state. By default returns raw
        features; pass normalize=True for the L2-normalized classifier space."""
        if task_index is None:
            task_index = self._cur_task
        cur_classes = np.arange(
            self._known_classes,
            self._known_classes + data_manager.get_task_size(task_index),
        )
        dataset = data_manager.get_dataset(
            cur_classes, source="train", mode="test"
        )
        raw_network = self._raw_network()
        raw_network.eval()
        features = []
        targets = []
        with rng_preserving(), torch.no_grad():
            loader = deterministic_loader(
                dataset,
                batch_size=64,
                shuffle=False,
                num_workers=self._loader_workers(),
                seed=0,
            )
            for _, inputs, batch_targets in loader:
                inputs = inputs.to(self._device, non_blocking=True)
                feats = raw_network.backbone(inputs)
                if normalize:
                    feats = F.normalize(feats, p=2, dim=1)
                features.append(feats.detach().cpu())
                targets.append(batch_targets)
        features = torch.cat(features, dim=0)
        if return_targets:
            return features, torch.cat(targets, dim=0)
        return features

    def _collect_anchor_images(self, data_manager, task_index=None, per_class=2):
        """Collect a few real current-task images per class as anchors for the
        LoRA parameter JVP (J_A(x) ΔA + J_B(x) ΔB)."""
        if task_index is None:
            task_index = self._cur_task
        cur_classes = np.arange(
            self._known_classes,
            self._known_classes + data_manager.get_task_size(task_index),
        )
        dataset = data_manager.get_dataset(
            cur_classes, source="train", mode="test"
        )
        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        selected = {int(c): [] for c in cur_classes.tolist()}
        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs = inputs.to(self._device, non_blocking=True)
                for img, target in zip(inputs, targets):
                    c = int(target.item())
                    if len(selected[c]) < per_class:
                        selected[c].append(img)
                if all(len(v) >= per_class for v in selected.values()):
                    break
        anchors = torch.stack(
            [img for values in selected.values() for img in values]
        )
        return anchors

    def _lora_jvp_basis(self, data_manager, mode="delta_a_b"):
        """Compute the final-feature drift responses induced by the actual LoRA
        parameter changes (ΔA and/or new B_t) via JVP on current-task anchors,
        then take the top-r right singular vectors as the transport basis."""
        from torch.func import functional_call
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual

        anchors = self._collect_anchor_images(data_manager, per_class=2)
        module = self._raw_network().backbone
        module.eval()
        a_weights = [w.weight for w in module.w_As]
        b_weights = [w.weight for w in module.w_Bs]

        # Match LoRA parameter names by object identity.
        name_to_weight = {}
        for name, param in module.named_parameters():
            for weight in a_weights + b_weights:
                if param is weight and name not in name_to_weight:
                    name_to_weight[name] = weight
                    break
        order = []
        for weight in a_weights + b_weights:
            for name, mapped in name_to_weight.items():
                if mapped is weight and name not in order:
                    order.append(name)
                    break
        lora_names = set(order)
        base_params = {
            name: param.detach()
            for name, param in module.named_parameters()
            if name not in lora_names
        }

        a_new = [w.detach().clone().to(self._device) for w in a_weights]
        b_cur = [w.detach().clone().to(self._device) for w in b_weights]
        a_old = [t.to(self._device) for t in self._lrpt_a_old]
        delta_a = [a_new[i] - a_old[i] for i in range(len(a_new))]
        lora_cur = [t.clone() for t in a_new + b_cur]

        def compute_jvp(v_vec, anchor_batch):
            with sdpa_kernel(SDPBackend.MATH):
                with dual_level():
                    dual_params = dict(base_params)
                    for name, primal, tangent in zip(
                        order, lora_cur, v_vec
                    ):
                        dual_params[name] = make_dual(primal, tangent)
                    out = functional_call(
                        module, dual_params, anchor_batch
                    )
                    return unpack_dual(out).tangent.detach()

        if mode == "layerwise":
            layer_bases = []
            num_a = len(a_new)
            for layer_id in range(12):
                v = [torch.zeros_like(t) for t in lora_cur]
                for offset in (2 * layer_id, 2 * layer_id + 1):
                    v[offset] = delta_a[offset]
                    v[num_a + offset] = b_cur[offset]
                jvp_out = torch.cat(
                    [compute_jvp(v, chunk) for chunk in anchors.split(8)],
                    dim=0,
                )
                _, _, vh = torch.linalg.svd(
                    jvp_out.float(), full_matrices=False
                )
                layer_basis = (
                    vh[: self._lrpt_basis_rank].t().contiguous().cpu()
                )
                layer_bases.append(layer_basis)
            return layer_bases

        if mode == "delta_a":
            v = delta_a + [torch.zeros_like(t) for t in b_cur]
        elif mode == "delta_a_b":
            v = delta_a + b_cur
        else:
            raise ValueError("unknown LoRA basis mode {}".format(mode))
        jvp_out = torch.cat(
            [compute_jvp(v, chunk) for chunk in anchors.split(8)],
            dim=0,
        )
        _, _, vh = torch.linalg.svd(jvp_out.float(), full_matrices=False)
        basis = vh[: self._lrpt_basis_rank].t().contiguous().cpu()
        return basis

    def _fit_lora_aware_transport(self, data_manager, fit_old, fit_new):
        """Fit a transport constrained to the final-feature drift subspace
        induced by the real LoRA parameter changes (JVP-based)."""
        try:
            if self._lrpt_a_old is None:
                raise RuntimeError("missing pre-update shared-A capture")
            w_emp, bias = fit_affine_map(
                fit_old,
                fit_new,
                reg=self._lrpt_reg,
            )

            if self._lrpt_basis == "layerwise":
                layer_bases = self._lora_jvp_basis(
                    data_manager, mode="layerwise"
                )
                layer_maps = [
                    project_map_to_basis(w_emp, layer_basis)
                    for layer_basis in layer_bases
                ]
                alpha = fit_layerwise_weights(
                    fit_old,
                    fit_new,
                    layer_maps,
                    reg=self._lrpt_reg,
                )
                w = torch.zeros_like(w_emp)
                for a_l, w_l in zip(alpha.tolist(), layer_maps):
                    w = w + a_l * w_l
                logging.info(
                    "[SharedA-SDLoRA] LRPT layerwise alpha=%s",
                    [round(float(a), 4) for a in alpha.tolist()],
                )
            else:
                basis = self._lora_jvp_basis(
                    data_manager, mode=self._lrpt_basis
                )
                w = project_map_to_basis(w_emp, basis)
                logging.info(
                    "[SharedA-SDLoRA] LRPT basis=%s dims=%d",
                    self._lrpt_basis,
                    basis.shape[1],
                )

            u, v = low_rank_factors(w, rank=self._lrpt_rank)
            self._lrpt_a_old = None
            return u, v, bias
        except Exception:
            logging.exception("[SharedA-SDLoRA] LRPT LoRA-aware transport failed")
            raise

    def _apply_coordinate_transport_to_old_prototypes(self, data_manager):
        """Move historical prototypes with a gated orthogonal residual map.

        The previous and freshly rebuilt deployment states are evaluated on
        identical current-task samples. Coordinate alignment is an independent
        option so this transport can be measured as a standalone ablation. The
        fitted map is applied once to old prototypes and then discarded; only
        scalar diagnostics are persisted.
        """
        old_features = self._coordinate_pre_features
        old_targets = self._coordinate_pre_targets
        new_features, new_targets = self._extract_current_task_features(
            data_manager, return_targets=True
        )
        self._coordinate_pre_features = None
        self._coordinate_pre_targets = None
        if not torch.equal(old_targets, new_targets):
            raise RuntimeError(
                "coordinate transport paired-feature targets differ"
            )

        transport = fit_residual_orthogonal_transport(
            old_features,
            new_features,
            rank=self._coordinate_transport_rank,
            identity_reg=self._coordinate_transport_reg,
            min_validation_gain=self._coordinate_transport_min_gain,
        )
        path = os.path.join(self.args["filepath"], PROTOTYPES_FILENAME)
        if not os.path.exists(path):
            raise FileNotFoundError(
                "coordinate transport requires historical prototypes at {}".format(
                    path
                )
            )
        old_prototypes = torch.load(
            path, map_location="cpu", weights_only=True
        )
        updated = apply_residual_orthogonal_transport(
            old_prototypes, transport
        )
        torch.save(updated, path)

        scalar_diagnostics = {
            key: value
            for key, value in transport.items()
            if key not in ("basis", "rotation")
        }
        scalar_diagnostics["task_id"] = self._cur_task
        scalar_diagnostics["num_prototypes"] = len(updated)
        diagnostics_path = os.path.join(
            self.args["filepath"], COORDINATE_DIAGNOSTICS_FILENAME
        )
        diagnostics = []
        if os.path.exists(diagnostics_path):
            with open(diagnostics_path, "r", encoding="utf-8") as handle:
                diagnostics = json.load(handle)
        diagnostics.append(scalar_diagnostics)
        with open(diagnostics_path, "w", encoding="utf-8") as handle:
            json.dump(diagnostics, handle, indent=2, sort_keys=True)
            handle.write("\n")
        logging.info(
            "[CoordinateStable] prototype transport task %d: enabled=%s "
            "rank=%d explained=%.4f val_gain=%.4f old_classes=%d",
            self._cur_task,
            transport["enabled"],
            transport["rank"],
            transport["explained_drift"],
            transport["validation_gain"],
            len(updated),
        )

    def _apply_lrpt_to_old_prototypes(self, data_manager):
        """Fit the low-rank transport from the paired current-task features and
        recursively move all previously stored prototypes into the updated
        feature space. The transport itself is discarded after application."""
        z_old_raw = self._lrpt_pre_features
        z_new_raw, z_new_targets = self._extract_current_task_features(
            data_manager, return_targets=True
        )
        z_old_targets = self._lrpt_pre_targets
        self._lrpt_pre_features = None
        self._lrpt_pre_targets = None
        if self.args.get("sa_raw_prototypes", False):
            z_old = z_old_raw.float()
            z_new = z_new_raw.float()
        else:
            z_old = F.normalize(z_old_raw, p=2, dim=1)
            z_new = F.normalize(z_new_raw, p=2, dim=1)
        if self._lrpt_fit_target == "classmean":
            fit_old = self._class_mean_prototypes(z_old, z_old_targets)
            fit_new = self._class_mean_prototypes(z_new, z_new_targets)
            logging.info(
                "[SharedA-SDLoRA] LRPT class-mean fit: %d current classes",
                fit_old.shape[0],
            )
        else:
            fit_old, fit_new = z_old, z_new

        if self._lrpt_basis != "generic":
            u, v, bias = self._fit_lora_aware_transport(
                data_manager, fit_old, fit_new
            )
            rel_err, _ = transport_prediction_error(
                fit_old, fit_new, u, v, bias=bias
            )
        elif self._lrpt_bias:
            if self._lrpt_dual:
                u_s, v_s, b_s = fit_affine_low_rank_transport(
                    z_old,
                    z_new,
                    rank=self._lrpt_rank,
                    reg=self._lrpt_reg,
                )
                u_c, v_c, b_c = fit_affine_low_rank_transport(
                    fit_old,
                    fit_new,
                    rank=self._lrpt_rank,
                    reg=self._lrpt_reg,
                )
                w = max(self._lrpt_class_weight, 0.0)
                u = torch.cat([u_s, math.sqrt(w) * u_c], dim=1)
                v = torch.cat([v_s, math.sqrt(w) * v_c], dim=1)
                bias = b_s + w * b_c
                rel_err, _ = transport_prediction_error(
                    fit_old, fit_new, u, v, bias=bias
                )
                logging.info(
                    "[SharedA-SDLoRA] LRPT dual-space: sample_rank=%d class_rank=%d class_weight=%.2f",
                    u_s.shape[1],
                    u_c.shape[1],
                    w,
                )
            else:
                u, v, bias = fit_affine_low_rank_transport(
                    fit_old,
                    fit_new,
                    rank=self._lrpt_rank,
                    reg=self._lrpt_reg,
                )
                rel_err, _ = transport_prediction_error(
                    fit_old, fit_new, u, v, bias=bias
                )
        else:
            if self._lrpt_dual:
                raise ValueError("lrpt_dual requires lrpt_bias=True")
            u, v, bias = (
                *fit_low_rank_transport(
                    fit_old,
                    fit_new,
                    rank=self._lrpt_rank,
                    reg=self._lrpt_reg,
                ),
                None,
            )
            rel_err, _ = transport_prediction_error(fit_old, fit_new, u, v)
        if self._lrpt_diagnostics:
            diag_norm = fit_rank_residuals(
                z_old, z_new, (10, 16, 32, 768), reg=self._lrpt_reg
            )
            diag_raw = fit_rank_residuals(
                z_old_raw.to(torch.float32),
                z_new_raw.to(torch.float32),
                (10, 16, 32, 768),
                reg=self._lrpt_reg,
            )
            diag_affine_norm = fit_affine_rank_residuals(
                z_old, z_new, (10, 16, 32, 768), reg=self._lrpt_reg
            )
            diag_affine_raw = fit_affine_rank_residuals(
                z_old_raw.to(torch.float32),
                z_new_raw.to(torch.float32),
                (10, 16, 32, 768),
                reg=self._lrpt_reg,
            )
            classmean_affine = {}
            if (
                z_old_targets is not None
                and z_new_targets is not None
                and torch.equal(z_old_targets, z_new_targets)
            ):
                old_means = self._class_mean_prototypes(
                    z_old, z_old_targets
                )
                new_means = self._class_mean_prototypes(
                    z_new, z_new_targets
                )
                classmean_affine = fit_affine_rank_residuals(
                    old_means,
                    new_means,
                    (10, 16, 32, 768),
                    reg=self._lrpt_reg,
                )
            bias_rel = bias_relative_error(z_old, z_new)
            logging.info(
                "[SharedA-SDLoRA] LRPT-DIAG task %d norm=%s raw=%s affine_norm=%s affine_raw=%s classmean_affine=%s bias_rel=%.4f",
                self._cur_task,
                {k: round(v, 4) for k, v in diag_norm.items()},
                {k: round(v, 4) for k, v in diag_raw.items()},
                {k: round(v, 4) for k, v in diag_affine_norm.items()},
                {k: round(v, 4) for k, v in diag_affine_raw.items()},
                {k: round(v, 4) for k, v in classmean_affine.items()},
                bias_rel,
            )
        logging.info(
            "[SharedA-SDLoRA] LRPT task %d: rank=%d reg=%.2e relative_drift_error=%.4f",
            self._cur_task,
            self._lrpt_rank,
            self._lrpt_reg,
            rel_err,
        )

        path = os.path.join(self.args["filepath"], PROTOTYPES_FILENAME)
        if not os.path.exists(path):
            logging.warning(
                "[SharedA-SDLoRA] LRPT skipped: no stored prototypes at %s",
                path,
            )
            return
        old = torch.load(path, map_location="cpu", weights_only=True)
        if self._lrpt_adaptive and bias is not None:
            updated = apply_transport_adaptive(old, u, v, bias=bias)
        else:
            updated = apply_transport(
                old, u, v, bias=bias, damping=self._lrpt_damping
            )
        torch.save(updated, path)
        if bias is not None:
            logging.info(
                "[SharedA-SDLoRA] LRPT bias norm=%.4f",
                float(torch.linalg.norm(bias)),
            )
        logging.info(
            "[SharedA-SDLoRA] LRPT moved %d old prototypes to updated feature space",
            len(updated),
        )

    def _class_mean_prototypes(self, features, targets):
        """Per-class mean of L2-normalized features, then L2-normalized, which
        matches the stored prototype representation used by PrototypeCosineHead."""
        class_ids = torch.unique(targets)
        means = []
        for class_id in class_ids:
            mask = targets == class_id
            mean = features[mask].mean(dim=0)
            means.append(F.normalize(mean, p=2, dim=0))
        return torch.stack(means)

    def _compute_prototypes(self, data_manager, raw_network):
        """Compute L2-normalized per-class mean prototypes for the classes of
        the current task, using only that task's training data at training time.
        Old prototypes are loaded from disk and merged (no old data re-use)."""
        cur_classes = np.arange(self._known_classes, self._total_classes)
        dataset = data_manager.get_dataset(
            cur_classes, source="train", mode="test"
        )
        dataset = self._sbgc_split_dataset(dataset, "train")
        raw_network.eval()
        per_class = {c: [] for c in cur_classes.tolist()}
        all_features = []
        all_targets = []
        with rng_preserving(), torch.no_grad():
            loader = deterministic_loader(
                dataset,
                batch_size=64,
                shuffle=False,
                num_workers=self._loader_workers(),
                seed=0,
            )
            for _, inputs, targets in loader:
                inputs = inputs.to(self._device, non_blocking=True)
                feats = raw_network.backbone(inputs)
                if not self.args.get("sa_raw_prototypes", False):
                    feats = F.normalize(feats, p=2, dim=1)
                all_features.append(feats.cpu())
                all_targets.append(targets)
                for f, target in zip(feats.cpu(), targets):
                    per_class[int(target.item())].append(f)
        self._last_proto_features = torch.cat(all_features)
        self._last_proto_targets = torch.cat(all_targets)
        self._last_proto_task = self._cur_task
        k_prototypes = int(self.args.get("sa_k_prototypes", 1))
        if k_prototypes <= 1:
            prototypes = {
                c: F.normalize(torch.stack(per_class[c]).mean(dim=0), p=2, dim=0)
                for c in cur_classes.tolist()
            }
        else:
            prototypes = {
                c: self._cluster_prototypes(per_class[c], k_prototypes)
                for c in cur_classes.tolist()
            }
        path = os.path.join(self.args["filepath"], PROTOTYPES_FILENAME)
        if os.path.exists(path):
            old = torch.load(path, map_location="cpu", weights_only=True)
            for class_id, vector in old.items():
                prototypes.setdefault(int(class_id), vector)
        torch.save(prototypes, path)
        return prototypes

    @staticmethod
    def _cluster_prototypes(features, k, iterations=20):
        feats = torch.stack(features)
        if len(feats) <= k:
            mean = F.normalize(feats.mean(dim=0), p=2, dim=0)
            return [mean.clone() for _ in range(k)]
        centers = [feats[0].clone()]
        for _ in range(1, k):
            dists = (feats - centers[-1]).pow(2).sum(dim=1)
            centers.append(feats[dists.argmax()].clone())
        centers = torch.stack(centers)
        for _ in range(iterations):
            dists = torch.stack(
                [(feats - center).pow(2).sum(dim=1) for center in centers]
            )
            assignments = dists.argmin(dim=0)
            new_centers = []
            for j in range(k):
                mask = assignments == j
                if mask.sum() > 0:
                    new_centers.append(feats[mask].mean(dim=0))
                else:
                    new_centers.append(centers[j].clone())
            centers = torch.stack(new_centers)
        return [
            F.normalize(center, p=2, dim=0).detach().clone()
            for center in centers
        ]

    def _rebuild_eval_backbone(self):
        """Rebuild the backbone from saved artifacts so each task counts once."""
        self._unwrap_network()
        self._network.backbone = self.update_network(
            index=False, task_index=self._cur_task + 1
        )
        self._network.to(self._device)

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
            batch_size=min(int(self.args["batch_size"]), 16),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        features, targets = [], []
        torch.cuda.empty_cache()
        with torch.no_grad():
            for _, inputs, batch_targets in loader:
                inputs = inputs.to(self._device, non_blocking=True)
                features.append(raw_network.backbone(inputs).detach().cpu())
                targets.append(batch_targets)
        feature_matrix = torch.cat(features)
        target_vector = torch.cat(targets)
        del features, targets
        torch.cuda.empty_cache()

        raw_network.fc.train()
        optimizer = optim.SGD(
            raw_network.fc.parameters(), lr=lr, momentum=0.9
        )
        for epoch in range(epochs):
            permutation = torch.randperm(feature_matrix.shape[0])
            epoch_loss = 0.0
            steps = 0
            for start in range(0, feature_matrix.shape[0], self.args["batch_size"]):
                indices = permutation[start : start + self.args["batch_size"]]
                batch_x = feature_matrix[indices].to(self._device)
                batch_y = target_vector[indices].to(self._device)
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
