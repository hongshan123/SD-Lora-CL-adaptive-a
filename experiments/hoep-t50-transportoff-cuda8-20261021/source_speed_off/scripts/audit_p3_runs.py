#!/usr/bin/env python3
"""Audit the 30 P3 runs frozen at commit 15ab791 (train commit c256cc4).

For every P3 run directory this checks:
  * run_manifest.json exists and records the exact train commit c256cc4;
  * manifest.config_sha256 equals the SHA-256 of the referenced config file;
  * the config's seed list contains exactly one seed and matches the seed
    parsed from the trainer log;
  * the log has a full 10-task top-1 curve, average accuracy and forgetting;
  * the expected persistent artifacts exist;
  * the queue log shows ``END <NAME> status=0`` for the run.

Usage:
  python scripts/audit_p3_runs.py
"""

import glob
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.parse_log_metrics import parse


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_COMMIT = "c256cc45b45f9983c84afd7063ebeab891e0b68f"
RUN_DIR_GLOBS = [
    "INR_P3_LIVEA_DUALB_SEED*_NCCL",
    "INR_P3_SDLORA_SEED*_NCCL",
    "INR_P3_EXP009_SEED*_NCCL",
    "C100_P3_LIVEA_DUALB_SEED*_NCCL",
    "C100_P3_SDLORA_SEED*_NCCL",
    "C100_P3_EXP009_SEED*_NCCL",
]
EXPECTED_ARTIFACTS = {
    "dual": [
        "sa_state.pt",
        "sa_merged_lora.pt",
        "sa_prototypes.pt",
        "sa_dual_head.pt",
    ],
    "exp009": ["sa_state.pt", "sa_merged_lora.pt", "sa_prototypes.pt"],
    "sdlora": ["lora_w_a_0.pt", "lora_w_b_0.pt", "lora_w_b_9.pt"],
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_seed_from_log(path):
    text = open(path, errors="ignore").read()
    match = re.search(r"=> seed: (-?\d+)", text) or re.search(
        r"\bseed \[(-?\d+)\]", text
    )
    if match is None:
        return None
    return int(match.group(1))


def queue_status(queue_log):
    """Return {run_name: exit_status_int} parsed from a p3_queue.log."""
    out = {}
    if not os.path.exists(queue_log):
        return out
    text = open(queue_log, errors="ignore").read()
    for match in re.finditer(
        r"===== .* END (\S+) status=(\d+) =====", text
    ):
        out[match.group(1)] = int(match.group(2))
    return out


def run_kind(name):
    if "LIVEA_DUALB" in name:
        return "dual"
    if "SDLORA" in name:
        return "sdlora"
    if "EXP009" in name:
        return "exp009"
    raise ValueError("unknown run kind: {}".format(name))


def main():
    queue_log = os.path.join(ROOT, "p3_queue.log")
    statuses = queue_status(queue_log)
    problems = []
    audited = []

    run_dirs = sorted(
        path
        for pattern in RUN_DIR_GLOBS
        for path in glob.glob(os.path.join(ROOT, pattern))
    )
    if len(run_dirs) != 30:
        problems.append("expected 30 P3 run dirs, found {}".format(len(run_dirs)))

    for run_dir in run_dirs:
        name = os.path.basename(run_dir)
        manifest_path = os.path.join(run_dir, "run_manifest.json")
        if not os.path.exists(manifest_path):
            problems.append("{}: missing run_manifest.json".format(name))
            continue
        manifest = json.load(open(manifest_path))

        commit = manifest.get("git_commit")
        if commit != TRAIN_COMMIT:
            problems.append(
                "{}: git_commit {} != {}".format(name, commit, TRAIN_COMMIT)
            )

        command = manifest.get("command", [])
        config_rel = None
        for token in command:
            if token.startswith("--config="):
                config_rel = token.split("=", 1)[1]
        if config_rel is None:
            problems.append("{}: no --config in manifest command".format(name))
            config_ok = None
        else:
            config_path = os.path.join(ROOT, config_rel)
            if not os.path.exists(config_path):
                problems.append("{}: config missing {}".format(name, config_rel))
                config_ok = False
            else:
                config_sha = sha256_file(config_path)
                config_ok = config_sha == manifest.get("config_sha256")
                if not config_ok:
                    problems.append(
                        "{}: config_sha256 {} != manifest {}".format(
                            name, config_sha, manifest.get("config_sha256")
                        )
                    )

        seed_cfg = None
        if config_ok:
            cfg = json.load(open(config_path))
            seed_cfg = cfg.get("seed")
            if not isinstance(seed_cfg, list) or len(seed_cfg) != 1:
                problems.append(
                    "{}: config seed is not a single-element list".format(name)
                )

        log_path = os.path.join(ROOT, name.lower() + ".log")
        if not os.path.exists(log_path):
            log_path = os.path.join(
                ROOT,
                "p3_{}_{}_seed{}_nccl.log".format(
                    "inr" if name.startswith("INR") else "c100",
                    "livea_dual_b"
                    if "LIVEA_DUALB" in name
                    else "sdlora"
                    if "SDLORA" in name
                    else "exp009",
                    re.search(r"SEED(\d+)", name).group(1),
                ),
            )
        if not os.path.exists(log_path):
            problems.append("{}: log missing".format(name))
            continue

        seed_log = parse_seed_from_log(log_path)
        expected_seed = seed_cfg[0] if isinstance(seed_cfg, list) and seed_cfg else None
        if expected_seed is not None and seed_log != expected_seed:
            problems.append(
                "{}: log seed {} != config seed {}".format(
                    name, seed_log, expected_seed
                )
            )

        info = parse(log_path)
        curves = [c for c in info["curves"] if len(c) == 10]
        if not curves:
            problems.append(
                "{}: no full 10-task top1 curve in log".format(name)
            )
            full_curve = None
        else:
            full_curve = curves[-1]
            if not all(0.0 <= value <= 100.0 for value in full_curve):
                problems.append(
                    "{}: final curve contains out-of-range values".format(name)
                )
        if not info["avg"]:
            problems.append("{}: no Average Accuracy in log".format(name))
        if not info["forgetting"]:
            problems.append("{}: no Forgetting in log".format(name))

        missing_artifacts = [
            artifact
            for artifact in EXPECTED_ARTIFACTS[run_kind(name)]
            if not os.path.exists(os.path.join(run_dir, artifact))
        ]
        if missing_artifacts:
            problems.append(
                "{}: missing artifacts {}".format(name, missing_artifacts)
            )

        queue_code = statuses.get(
            name.replace("_P3_", "_").replace("_NCCL", "")
        )
        if queue_code is None:
            problems.append("{}: no END line in queue log".format(name))
        elif queue_code != 0:
            problems.append("{}: queue status={}".format(name, queue_code))

        audited.append(
            {
                "name": name,
                "seed": seed_log,
                "final": full_curve[-1] if full_curve is not None else None,
                "avg": info["avg"][-1] if info["avg"] else None,
                "forgetting": info["forgetting"][-1] if info["forgetting"] else None,
                "config_sha_ok": config_ok,
                "queue_status": queue_code,
            }
        )

    print("Audited {} P3 runs".format(len(audited)))
    for row in audited:
        print(
            "  {:<32} seed={:<6} final={:>7.2f} avg={:>7.2f} "
            "forgetting={:>6.2f} config_sha_ok={} queue_status={}".format(
                row["name"],
                row["seed"],
                row["final"],
                row["avg"],
                row["forgetting"],
                row["config_sha_ok"],
                row["queue_status"],
            )
        )
    if problems:
        print("\nPROBLEMS ({}):".format(len(problems)))
        for problem in problems:
            print("  - " + problem)
        return 1
    print("\nP3 AUDIT PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
