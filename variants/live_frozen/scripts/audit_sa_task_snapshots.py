#!/usr/bin/env python3
"""Verify or finalize immutable per-task LoRA snapshots."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.sa_task_snapshots import audit_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    directories = sorted((args.run_dir / "task_snapshots").glob("task_[0-9][0-9][0-9]"))
    if not directories:
        raise FileNotFoundError(f"no task snapshots in {args.run_dir}")
    print(json.dumps([audit_snapshot(directory, finalize=args.finalize) for directory in directories], indent=2))


if __name__ == "__main__":
    main()
