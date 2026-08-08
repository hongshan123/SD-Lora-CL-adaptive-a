"""Fresh-run directory guard and run manifest for the P0 audit.

Every P0 run must target a directory that does not already exist, must hold an
exclusive ``flock`` on ``<filepath>.lock`` for its lifetime, and must record
the Git commit, config digest, command and run ID so artifacts are traceable.
"""

import fcntl
import hashlib
import json
import os
import subprocess
import time
import uuid


RUN_MANIFEST_FILENAME = "run_manifest.json"


def _git_commit(project_root):
    try:
        proc = subprocess.run(
            ["git", "-C", project_root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def config_sha256(config_bytes):
    return hashlib.sha256(config_bytes).hexdigest()


def acquire_run_guard(
    filepath,
    config_bytes=None,
    command=None,
    run_id=None,
    resume=False,
    project_root=None,
):
    """Acquire an exclusive lock and create a fresh run directory.

    Returns the open lock file object.  The caller must keep a reference to it
    for the whole run (the process-level flock is released when the file is
    closed or the process exits).

    Raises ``RuntimeError`` when another process already holds the lock and
    ``FileExistsError`` when the target directory already exists without an
    explicit ``resume=True`` manifest.
    """
    target = os.path.abspath(os.path.expanduser(filepath))
    if target.endswith(os.sep):
        target = target.rstrip(os.sep)
    lock_path = target + ".lock"
    if project_root is None:
        project_root = os.getcwd()

    lock_file = open(lock_path, "a+")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        lock_file.close()
        raise RuntimeError(
            "filepath is locked by another run: {}".format(lock_path)
        ) from exc

    manifest_path = os.path.join(target, RUN_MANIFEST_FILENAME)
    if os.path.exists(target):
        if not resume or not os.path.exists(manifest_path):
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
            raise FileExistsError(
                "fresh-run directory already exists: {}; "
                "refusing to reuse artifacts (use explicit resume=true "
                "and a run_manifest.json to recover)".format(target)
            )
        return lock_file

    os.makedirs(target)
    if resume:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()
        raise FileExistsError(
            "resume requested but no existing run directory found at {}"
            .format(target)
        )

    run_id = run_id or uuid.uuid4().hex
    manifest = {
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": _git_commit(project_root),
        "config_sha256": (
            config_sha256(config_bytes) if config_bytes is not None else None
        ),
        "command": command if command is not None else [],
        "resume": False,
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return lock_file
