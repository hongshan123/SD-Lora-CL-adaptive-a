"""Resume missing cache and all rollouts after an audited numerical argmax tie."""
import concurrent.futures
import json
import subprocess
import time
import traceback
import run_queue as q


def main():
    lease = None
    try:
        q.verify_source()
        q.publish(state='running',phase='finishing_cache',error=None)
        lease=q.GpuLease.acquire([1])
        assert lease is not None
        used=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used',
                  '--format=csv,noheader,nounits'],text=True))
        assert used<300,used
        q.run(['--extract','49'],'recovered_cache_049',1,lease)
        lease.close();lease=None
        deadline=time.time()+1200
        while len(list((q.ROOT/'cache').glob('task_*.json')))<50:
            assert time.time()<deadline,'Remaining cache worker did not complete; inspect supplement logs'
            q.publish()
            time.sleep(10)
        q.write(q.ROOT/'cache_manifest.json',{p.name:json.loads(p.read_text())
                  for p in sorted((q.ROOT/'cache').glob('task_*.json'))})
        q.publish(phase='prototype_rollouts',gpu_workers={})
        q.evaluation('current')
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(q.evaluation,v) for v in ['none','half','reliable']]
            for future in concurrent.futures.as_completed(futures):future.result()
        q.verify_source()
        q.write(q.ROOT/'results.json',{v:json.loads((q.ROOT/'experiments'/v/'metrics.json').read_text())
                                      for v in ['none','current','half','reliable']})
        q.publish(state='complete',phase='complete')
        q.write(q.ROOT/'exit.json',dict(returncode=0,finished=time.time()))
    except BaseException:
        q.publish(state='failed',error=traceback.format_exc())
        q.write(q.ROOT/'exit.json',dict(returncode=1,finished=time.time()))
        raise
    finally:
        if lease:lease.close()


if __name__=='__main__':main()
