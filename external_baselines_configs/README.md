# External-baseline local-reproduction configs (P5)

These configs are copied into the official clones under `external_baselines/`
by `run_external_baselines_c100.sh` (and future INR launchers).  The clones
themselves are gitignored; the configs here are the reproducible record.

Layout:

- `infolora/cifar100_inflora_sdlocal_seed1.json` — InfLoRA CIFAR-100 T=10 seed 1 (GPU 0)
- `infolora/mimg10_inflora_sdlocal_seed1.json` — InfLoRA ImageNet-R T=10 seed 1 (GPU 0)
- `cllora/cifar_t10_seed1.json` — CL-LoRA CIFAR-100 T=10 seed 1 (GPU 1)
- `cllora/inr_t10_seed1.json` — CL-LoRA ImageNet-R T=10 seed 1 (GPU 1)
- `lora_drs/cifar100_t10_seed1.json` — LoRA-Sub-DRS CIFAR-100 T=10 seed 1 (GPU 2)
- `lora_drs/imagenetr_t10_seed1.json` — LoRA-Sub-DRS ImageNet-R T=10 seed 1 (GPU 2)

Protocol notes: task splits match this project (CIFAR-100: 10 tasks x 10
classes; ImageNet-R: 10 tasks x 20 classes); all other hyper-parameters are
the official defaults of each implementation.  Results are reported in
separate columns from published values (see `strong_baselines_sd.md`).
