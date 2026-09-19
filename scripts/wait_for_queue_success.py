#!/usr/bin/env python3
"""Wait until a predecessor experiment queue records an explicit success."""

import argparse
from pathlib import Path
import re
import time


QUEUE_END_RE = re.compile(r"^QUEUE END status=(?P<status>-?\d+)(?:\s|$)", re.MULTILINE)


def terminal_queue_status(log_path):
    path = Path(log_path)
    if not path.is_file():
        return None
    matches = list(QUEUE_END_RE.finditer(path.read_text(errors="replace")))
    return int(matches[-1].group("status")) if matches else None


def wait_for_success(log_path, poll_seconds=60):
    while True:
        status = terminal_queue_status(log_path)
        if status is not None:
            return status
        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("queue_log")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be positive")
    status = wait_for_success(args.queue_log, args.poll_seconds)
    if status != 0:
        raise SystemExit(
            "predecessor queue ended with status {}; refusing to launch".format(status)
        )


if __name__ == "__main__":
    main()
