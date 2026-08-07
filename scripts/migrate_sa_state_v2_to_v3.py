#!/usr/bin/env python3
"""Explicitly migrate a v2 gauge cumulative state to v3 union-SVD state.

Usage:
  python scripts/migrate_sa_state_v2_to_v3.py --artifact DIR [--rank R] [--force]

The original v2 ``sa_state.pt`` is preserved as ``sa_state.pt.v2``.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.sa_lora import migrate_sa_state_v2_to_v3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, help="output directory")
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="fixed union-SVD rank (default: state's stored rank)",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not os.path.isdir(args.artifact):
        parser.error("artifact directory does not exist: {}".format(args.artifact))
    state = migrate_sa_state_v2_to_v3(
        args.artifact, rank=args.rank, force=args.force
    )
    print(
        "migrated {} -> v3 rank={} branches={} task_id={}".format(
            args.artifact,
            state["rank"],
            len(state["canonical_down"]),
            state["task_id"],
        )
    )


if __name__ == "__main__":
    main()
