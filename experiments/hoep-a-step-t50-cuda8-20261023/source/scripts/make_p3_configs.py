"""Generate P3 multi-seed configs from the P1/P2 NCCL templates."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPS = ROOT / "exps"
SEEDS = [1, 2, 3, 4, 5]

TEMPLATES = {
    "inr": {
        "livea_dual_b": "inr_p1_livea_dual_b_seed1995_nccl.json",
        "sdlora": "inr_p1_sdlora_seed1995_nccl.json",
        "exp009": "inr_p1_exp009_seed1995_nccl.json",
    },
    "c100": {
        "livea_dual_b": "c100_p2_livea_dual_b_seed1993_nccl.json",
        "sdlora": "c100_p2_sdlora_seed1993_nccl.json",
        "exp009": "c100_p2_exp009_seed1993_nccl.json",
    },
}

DIR_PREFIX = {
    ("inr", "livea_dual_b"): "INR_P3_LIVEA_DUALB_SEED{}_NCCL",
    ("inr", "sdlora"): "INR_P3_SDLORA_SEED{}_NCCL",
    ("inr", "exp009"): "INR_P3_EXP009_SEED{}_NCCL",
    ("c100", "livea_dual_b"): "C100_P3_LIVEA_DUALB_SEED{}_NCCL",
    ("c100", "sdlora"): "C100_P3_SDLORA_SEED{}_NCCL",
    ("c100", "exp009"): "C100_P3_EXP009_SEED{}_NCCL",
}


def main():
    written = []
    for dataset, methods in TEMPLATES.items():
        for method, template_name in methods.items():
            template = json.loads((EXPS / template_name).read_text(encoding="utf-8"))
            for seed in SEEDS:
                config = json.loads(json.dumps(template))
                config["seed"] = [seed]
                config["prefix"] = "p3_{}_{}_seed{}_nccl".format(
                    dataset, method, seed
                )
                config["filepath"] = "./{}/".format(
                    DIR_PREFIX[(dataset, method)].format(seed)
                )
                out = EXPS / "p3_{}_{}_seed{}_nccl.json".format(
                    dataset, method, seed
                )
                out.write_text(
                    json.dumps(config, indent=4, sort_keys=False) + "\n",
                    encoding="utf-8",
                )
                written.append(out.name)
    print("generated {} configs".format(len(written)))


if __name__ == "__main__":
    main()
