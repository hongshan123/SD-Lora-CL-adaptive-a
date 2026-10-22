#!/usr/bin/env python3
"""Explicitly migrate a legacy v1 Shared-A artifact to the v2 cumulative state.

The v1 ``sa_state.pt`` is backed up to ``sa_state.pt.v1``; per-task B files are
left untouched.  The new state contains canonical down projections, cumulative
up projections, and triangular factors, so training can continue with a single
artifact instead of an O(T) B bank.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.sa_lora import migrate_sa_state_v1_to_v2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", required=True, help="artifact directory")
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing sa_state.pt.v1 backup",
    )
    args = parser.parse_args()
    new_state = migrate_sa_state_v1_to_v2(args.filepath, force=args.force)
    print(
        "migrated to v2: task_id={} branches={}".format(
            new_state["task_id"], len(new_state["canonical_down"])
        )
    )


if __name__ == "__main__":
    main()
