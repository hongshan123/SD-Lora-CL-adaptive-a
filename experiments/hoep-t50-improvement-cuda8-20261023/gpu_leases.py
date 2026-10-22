"""Advisory GPU leases shared by independent training queues.

Workers inherit the locked descriptors via subprocess.Popen(pass_fds=...).
Closing a coordinator's descriptors must NOT call LOCK_UN: the worker shares
those open file descriptions and keeps the lease until its own process exits.
Lock files stay in place so every queue locks the same inode.
"""
import fcntl
import os
from pathlib import Path

LOCK_DIR = Path('/home/hongzhijun/.cache/codex-training-gpu-locks')


class GpuLease:
    def __init__(self, gpus, fds):
        self.gpus = tuple(gpus)
        self.fds = tuple(fds)

    @classmethod
    def acquire(cls, gpus, lock_dir=LOCK_DIR):
        """Reserve all requested cards without waiting, or reserve none."""
        gpus = sorted(set(gpus))
        if not gpus or any(not isinstance(gpu, int) or gpu < 0 for gpu in gpus):
            raise ValueError('Expected one or more nonnegative GPU indices')
        directory = Path(lock_dir)
        directory.mkdir(parents=True, exist_ok=True)
        fds = []
        try:
            for gpu in gpus:
                fd = os.open(directory / f'gpu{gpu}.lock', os.O_RDWR | os.O_CREAT, 0o600)
                fds.append(fd)
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return cls(gpus, fds)
        except BlockingIOError:
            for fd in fds:
                os.close(fd)
            return None
        except BaseException:
            for fd in fds:
                os.close(fd)
            raise

    def close(self):
        fds, self.fds = self.fds, ()
        for fd in fds:
            os.close(fd)
