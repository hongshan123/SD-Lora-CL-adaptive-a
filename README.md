# HOEP-A / Adaptive-A Continual Learning

Research implementations of shared-A LoRA, fixed-size aggregate state, HOEP
energy-based direction selection, historical least-squares alignment and
prototype transport. This project builds on SD-LoRA and includes experimental
variants and diagnostic tools.

The repository contains source code, tests, input configurations and usage
documentation. Logs, weights, snapshots, metrics, result tables, execution
manifests and experiment reports are generated locally and ignored by Git.

## Layout

- `backbone/`, `models/`, `utils/`: current HOEP implementation and existing
  structure, transport, consolidation and gradient-accumulation controls.
- `exps/`: training configuration templates.
- `scripts/`, `tests/`: launchers, diagnostics, analysis code and tests.
- `experiments/hoep-lora-adaptation-t50-cuda8-20261022/`: isolated historical-input
  total-update regularization and all-seen-class prototype training, with a
  paired queue and mathematical tests.
- `experiments/hoep-t50-improvement-cuda8-20261023/`: CUB/ImageNet-R paired
  Frozen-A, HOEP and optional training-objective implementations.
- `experiments/hoep-bmetric-t50-cuda8-20261023/`: optional input-metric
  preconditioning of current-task B gradients, with current/history variants.
- Other `experiments/` directories: snapshot, prototype replay, transport and
  resume utilities. Runtime inputs and outputs are not distributed.
- `variants/live_frozen/`: the separate Live-A/Frozen-A implementation family,
  including source, configurations and tests.
- `external_baselines_configs/`: input configurations for external baselines.

## Environment and data

Use Python 3.10, install a PyTorch build compatible with the local CUDA driver,
then install the remaining requirements:

```bash
pip install -r requirements.no_torch.txt
```

Prepare datasets locally:

```text
data/cifar-100-python/
data/imagenet-r/train/
data/imagenet-r/test/
data/cub/train/
data/cub/test/
```

Pretrained ViT weights are loaded through timm/Hugging Face. Offline execution
requires the corresponding weights to be cached beforehand.

## Training

Copy a configuration from `exps/`, set the dataset, GPU devices and a fresh
`filepath`, then run:

```bash
python main.py --config=/path/to/config.json
```

Older configuration templates contain server-specific paths; update them
before use. Record effective batch size and single-GPU/distributed/accumulation
settings when comparing runs.

For the current paired LoRA adaptation queue, see its
[setup instructions](experiments/hoep-t50-improvement-cuda8-20261023/README.md)
and the [B-gradient extension](experiments/hoep-bmetric-t50-cuda8-20261023/README.md).
See also [the method overview](docs/hoep_method.md). Experimental extensions are
included for reproducibility; inclusion does not imply verified improvement.

## Upstream attribution

This research repository is derived from SD-LoRA; it is not the authors'
official SD-LoRA release. Retain the original license and cite the upstream work:

```bibtex
@inproceedings{wu2025sdlora,
  title={{SD}-Lo{RA}: Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning},
  author={Yichen Wu and Hongming Piao and Long-Kai Huang and Renzhen Wang and Wanhua Li and Hanspeter Pfister and Deyu Meng and Kede Ma and Ying Wei},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=5U1rlpX68A}
}
```

Related projects include [LoRA-ViT](https://github.com/JamesQFreeman/LoRA-ViT)
and [PILOT](https://github.com/sun-hailong/LAMDA-PILOT).
