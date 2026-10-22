from pathlib import Path
import select
import subprocess
import sys
import tempfile
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gpu_leases import GpuLease
from priority_lease import OneShotPriority


class QueuedLeaseTest(unittest.TestCase):
    def test_waiting_process_cannot_overlap_owner(self):
        with tempfile.TemporaryDirectory() as folder:
            holder=GpuLease.acquire([3],lock_dir=folder)
            code='''import sys
sys.path.insert(0,sys.argv[1])
from gpu_leases import GpuLease
lease=GpuLease.acquire([3],lock_dir=sys.argv[2],blocking=True)
print("acquired",flush=True)
sys.stdin.read(1)
lease.close()
'''
            child=subprocess.Popen([sys.executable,'-u','-c',code,str(ROOT),folder],
                                   stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
            try:
                self.assertFalse(select.select([child.stdout],[],[],.2)[0])
                holder.close()
                self.assertTrue(select.select([child.stdout],[],[],5)[0])
                self.assertEqual(child.stdout.readline().strip(),'acquired')
                self.assertIsNone(GpuLease.acquire([3],lock_dir=folder))
                child.stdin.write('x');child.stdin.flush()
                self.assertEqual(child.wait(timeout=5),0)
                again=GpuLease.acquire([3],lock_dir=folder)
                self.assertIsNotNone(again);again.close()
            finally:
                holder.close()
                if child.poll() is None:child.kill();child.wait()
                child.stdin.close();child.stdout.close()

    def test_reservation_is_taken_once(self):
        with tempfile.TemporaryDirectory() as folder:
            holder=GpuLease.acquire([3],lock_dir=folder)
            request=OneShotPriority(3,lock_dir=folder)
            self.assertEqual(request.try_take(1),(False,None))
            self.assertEqual(request.try_take(3),(True,None))
            holder.close();lease=None;deadline=time.monotonic()+5
            while lease is None and time.monotonic()<deadline:
                _,lease=request.try_take(3)
                if lease is None:time.sleep(.01)
            self.assertIsNotNone(lease)
            self.assertEqual(request.try_take(3),(False,None))
            lease.close()

    def test_multi_gpu_blocking_reservation_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                GpuLease.acquire([0,3],lock_dir=folder,blocking=True)


if __name__=='__main__':unittest.main()
