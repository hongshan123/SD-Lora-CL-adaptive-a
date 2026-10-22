# HOEP A-step ablations

The source snapshot isolates the step size of shared A within the existing
HOEP plastic directions. After Task0, multiply the projected A gradient by
10 or 100 while retaining the original B, head and scale optimizer settings.
Task0 is unchanged. Coupled weight-decay cancellation is preserved explicitly.
Under the existing SGD and momentum, this changes the A tangent-step scale.
It adds no deployment parameters.

The input configurations use CUB seed1 and ImageNet-R seed1995, T50, rank10,
20 epochs per task, single-GPU batch128, HOEP5%, historical LS on, NormCap off,
and Coordinate Transport on. Main-suite Frozen-A and HOEP controls are reused.
Other prescribed confirmation seeds remain in the parent goal protocol.

Seven CPU mathematical tests cover SGD/momentum equivalence, decay
cancellation, protected rows, identity behavior, zero and invalid inputs.
Six three-task model checks must precede formal runs: multiplier1, 10 and100
on both datasets. Their Task0 uses20 epochs and later tasks use2. Multiplier1
must match the original HOEP model and RNG at all three boundaries; all arms
must share Task0. Each nonunit multiplier needs valid checks on both datasets
before its two formal jobs can start. Numerical invalidity is distinct from
poor accuracy; all valid complete runs are retained.

`run_queue.py` consumes the machine-local, frozen `manifest.json`, acquires
shared GPU leases, waits for free memory, and refuses existing output paths.
It does not preempt other jobs. `run_job.py` validates full curves, snapshots
and actual gain diagnostics before writing metrics. Training source and input
configurations must remain frozen once the queue is prepared.

The canonical record is the parent suite's `RESEARCH_LOG.md`, which includes
this directory's generated component journal. Runtime records and model data
are separate from source publication.

Prepare the adjacent paired suite first. From the repository root, create
machine-local inputs in a fresh copy; preparation does not start training:

```bash
python scripts/prepare_hoep_queue.py \
  --suite experiments/hoep-a-step-t50-cuda8-20261023 \
  --primary-root experiments/hoep-t50-improvement-cuda8-20261023 \
  --data-root /path/to/data --gpus 0 1
cd experiments/hoep-a-step-t50-cuda8-20261023
python run_queue.py
```

The published queue plan contains planned inputs only. It omits machine-local
paths, execution signatures, output hashes and run state. Preparation defaults
to the ordinary GPU lease pool; the optional one-shot priority mechanism in the
controller requires an explicit `priority_wait_gpu` in a local manifest.

Different A/B rates have prior motivation in
[LoRA+](https://proceedings.mlr.press/v235/hayou24a.html). Its ADAM/width analysis
does not establish an optimal A multiplier for this HOEP/SGD continual-learning
setting. These are independently motivated step-size diagnostics, with no
verified performance claim before complete paired experiments.
