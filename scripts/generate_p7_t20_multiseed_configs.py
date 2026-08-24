#!/usr/bin/env python3
"""Generate the preregistered P7 T=20 confirmation configs.

The generated matrix is:
  datasets: CIFAR-100, ImageNet-R
  methods:  frozen full method, SD-LoRA, EXP-009
  seeds:    1, 2, 3

Only run identity fields (prefix, seed, filepath) differ from the completed
P5 development-seed T=20 configs.
"""

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPS = ROOT / "exps"

SOURCES = {
    ("c100", "full"): "p5_full_t20_c100_nccl.json",
    ("c100", "sdlora"): "p5_sdlora_t20_c100_nccl.json",
    ("c100", "exp009"): "p5_exp009_t20_c100_nccl.json",
    ("inr", "full"): "p5_full_t20_inr_nccl.json",
    ("inr", "sdlora"): "p5_sdlora_t20_inr_nccl.json",
    ("inr", "exp009"): "p5_exp009_t20_inr_nccl.json",
}

DATASET_TAG = {"c100": "C100", "inr": "INR"}
METHOD_TAG = {"full": "FULL", "sdlora": "SDLORA", "exp009": "EXP009"}


def main():
    generated = []
    for seed in (1, 2, 3):
        for dataset in ("c100", "inr"):
            for method in ("full", "sdlora", "exp009"):
                source_path = EXPS / SOURCES[(dataset, method)]
                with source_path.open(encoding="utf-8") as handle:
                    config = copy.deepcopy(json.load(handle))

                name = f"p7_t20_{dataset}_{method}_seed{seed}_nccl"
                outdir = (
                    f"./{DATASET_TAG[dataset]}_P7_T20_"
                    f"{METHOD_TAG[method]}_SEED{seed}_NCCL/"
                )
                config["prefix"] = name
                config["seed"] = [seed]
                config["filepath"] = outdir
                config["sa_resume"] = False

                destination = EXPS / f"{name}.json"
                with destination.open("w", encoding="utf-8") as handle:
                    json.dump(config, handle, indent=4)
                    handle.write("\n")
                generated.append(destination.relative_to(ROOT))

    print(f"generated {len(generated)} configs")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
