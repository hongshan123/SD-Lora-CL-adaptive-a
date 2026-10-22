"""Resume fixed-reference extraction after resolving the CIFAR data root."""
import json
import subprocess
import time
import traceback
from pathlib import Path

from run_queue import ROOT, GpuLease, verify, run, write, torch_load


def main():
    lease=None
    try:
        verify()
        dataset=ROOT/'data/cifar-100-python'
        assert dataset.is_symlink() and (dataset/'train').is_file()
        oracle=json.loads((ROOT/'experiments/oracle_margin/metrics.json').read_text())
        assert oracle['complete'] and oracle['completed_tasks']==50
        lease=GpuLease.acquire([1]);assert lease is not None
        used=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used',
                     '--format=csv,noheader,nounits'],text=True))
        assert used<300,used
        write(ROOT/'status.json',dict(state='running',phase='anchor_extract',gpu=1,
             resumed=time.time(),oracle_complete=True))
        run('anchor_extract',1,lease)
        write(ROOT/'status.json',dict(state='running',phase='anchor_eval',gpu=1,
             oracle_complete=True))
        run('anchor_eval',1,lease)
        verify()
        prev=Path(json.loads((ROOT/'manifest.json').read_text())['previous_runtime'])
        names=['none','current','half','reliable','oracle_cosine','oracle_margin',
               'anchor_none','anchor_full','anchor_half']
        results={}
        for name in names:
            parent=prev if name in names[:4] else ROOT
            item=json.loads((parent/'experiments'/name/'metrics.json').read_text())
            assert item['complete'] and item['completed_tasks']==50,name
            results[name]=item
        write(ROOT/'results.json',results)
        maps=[torch_load(ROOT/'anchor_maps'/f'task_{t:03d}.pt') for t in range(1,50)]
        write(ROOT/'anchor_fit_summary.json',dict(enabled=sum(int(m['enabled']) for m in maps),
             total=49,mean_validation_gain=sum(m['validation_gain'] for m in maps)/49,
             mean_train_before=sum(m['train_before'] for m in maps)/49,
             mean_train_after=sum(m['train_after'] for m in maps)/49))
        write(ROOT/'status.json',dict(state='complete',finished=time.time()))
        write(ROOT/'exit.json',dict(returncode=0,finished=time.time()))
    except BaseException:
        write(ROOT/'status.json',dict(state='failed',error=traceback.format_exc(),updated=time.time()))
        write(ROOT/'exit.json',dict(returncode=1,finished=time.time()))
        raise
    finally:
        if lease:lease.close()


if __name__=='__main__':main()
