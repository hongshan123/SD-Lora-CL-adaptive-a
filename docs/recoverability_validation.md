# Recoverability Adaptive-A Validation

## Stages

The `recoverability` strategy exposes four cumulative validation stages:

1. `exact_risk`: use the synchronized shared-A loss gradient direction and replace the old impact magnitude with exact fixed-anchor recoverability.
2. `accessibility`: use the sketched effective-weight gradient and its exact Grassmann accessibility ascent direction.
3. `anchor_realign`: additionally re-express the same immutable task-start historical operator after each accepted basis update.
4. `global_budget`: allocate one network-wide historical operator-energy budget across all Q/V branches.

Only stage 4 is the complete proposed method. Earlier stages are mechanism ablations, not alternative final methods.

## Configuration

Generate protocol-matched JSON files from an existing experiment:

```bash
cd /home/hongzhijun/hongshan/SD-lora-cl_2/SD-Lora-CL-adaptive-a
python scripts/make_recoverability_validation_configs.py \
  --base exps/c100_coordinate_stable_adaptive_a_seed1993_nccl.json \
  --output-dir exps/recoverability_c100_seed1993 \
  --budget 0.01 \
  --step-size 0.1 \
  --interval 4 \
  --sketch-rank 16 \
  --toy
```

The generator changes only the Recoverability Adaptive-A fields, output prefix, and artifact directory. Dataset order, epochs, optimizer, learning rate, batch size, classifier, prototype transport, absorption mode, and all other protocol fields are inherited unchanged.

## Primary Diagnostics

Each task logs:

- validation stage and number of controller refreshes;
- mean/min/max selected `gamma`;
- zero/full-gamma fractions;
- exact selected recoverability risk and configured global budget;
- sketched accessibility utility;
- effective-gradient sketch rank and refresh interval.

The task-start `(A-, G-)` anchor is a non-persistent buffer. Saved `sa_state.pt` artifacts continue to contain only deployment state and existing class-dependent components.

## Interpretation

The CPU toy uses two candidates with identical projector distance. One rotates a high historical-energy direction and the other rotates a low-energy direction. Their chordal distances are equal, while operator-weighted recoverability strongly prefers the low-energy rotation. This is the required distinction from an unweighted Grassmann smoothness penalty.

The default output sketch has rank 16. It is deterministic across ranks and preserves effective-gradient energies in expectation. Set `sa_recoverability_sketch_rank` to the projection output dimension for an exact-gradient overhead ablation.
