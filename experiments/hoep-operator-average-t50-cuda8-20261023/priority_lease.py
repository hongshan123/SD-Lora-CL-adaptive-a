"""One queued lease request for the next release, without touching its owner."""
import queue
import threading

from gpu_leases import GpuLease, LOCK_DIR


class OneShotPriority:
    def __init__(self, gpu, lock_dir=LOCK_DIR):
        self.gpu = gpu
        self.used = False
        self.ready = queue.Queue(maxsize=1)
        def acquire():
            try:
                self.ready.put(GpuLease.acquire([gpu], lock_dir=lock_dir, blocking=True))
            except BaseException as error:
                self.ready.put(error)
        self.thread = threading.Thread(target=acquire, daemon=True,
                                       name=f'wait-gpu-{gpu}')
        self.thread.start()

    def try_take(self, gpu):
        if gpu != self.gpu or self.used:
            return False, None
        try:
            result = self.ready.get_nowait()
        except queue.Empty:
            return True, None
        self.used = True
        if isinstance(result, BaseException):
            raise result
        return True, result

    def status(self):
        return dict(gpu=self.gpu, consumed=self.used,
                    state='consumed' if self.used else 'awaiting_release' if self.ready.empty() else 'granted')

    def close_unused(self):
        if self.used:
            return
        try:
            result = self.ready.get_nowait()
        except queue.Empty:
            return
        if isinstance(result, GpuLease):
            result.close()
