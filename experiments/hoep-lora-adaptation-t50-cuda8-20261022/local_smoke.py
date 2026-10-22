"""Local integration check on an idle GPU, without touching cuda8 jobs."""
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent


def main():
    from gpu_leases import GpuLease
    lease=GpuLease.acquire([6])
    assert lease is not None, 'local GPU6 is leased'
    os.environ.update(CUDA_VISIBLE_DEVICES='6',CUDA_DEVICE_ORDER='PCI_BUS_ID',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',
                      OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    import run_job
    import torch
    os.chdir(ROOT/'source')
    outputs={}
    for arm, enabled, hw, pw in [('original',False,0.,0.),('baseline',True,0.,0.),('combined',True,1.,.1)]:
        name='local_'+arm+'_smoke'
        config=json.loads((ROOT/'configs/local_original_smoke.json').read_text())
        config.update(prefix=name,sa_p0_run_id=name,filepath=str(ROOT/'outputs'/name)+'/',
                      sa_adaptation_experiment=enabled,sa_total_history_weight=hw,sa_allseen_proto_weight=pw)
        path=ROOT/'configs'/f'{name}.json'
        path.write_text(json.dumps(config,indent=2)+'\n')
        sys.argv=[str(ROOT/'run_job.py'),str(path)]
        print('LOCAL_SMOKE_START',arm,time.time(),flush=True)
        run_job.main()
        outputs[arm]=Path(config['filepath'])
        gc.collect()
        torch.cuda.empty_cache()
    hashes={arm:json.loads((out/'p0_hashes.json').read_text()) for arm,out in outputs.items()}
    keys=['post_train_model_hash','post_train_rng_hash','post_eval_model_hash','post_eval_rng_hash']
    for task in ('0','1'):
        for key in keys:
            assert hashes['original'][task][key]==hashes['baseline'][task][key],(task,key,'zero-weight equivalence')
    for key in keys:
        assert hashes['original']['0'][key]==hashes['combined']['0'][key],(key,'Task0 equivalence')
    assert hashes['original']['1']['post_train_model_hash'] != hashes['combined']['1']['post_train_model_hash']
    proof={'passed':True,'zero_weight_tasks_bitwise_equal':[0,1],
           'combined_task0_model_and_rng_bitwise_equal':True,'combined_task1_changed':True,
           'metrics':{arm:json.loads((out/'metrics.json').read_text()) for arm,out in outputs.items()},
           'finished':time.time()}
    (ROOT/'proofs/local_integration.json').write_text(json.dumps(proof,indent=2)+'\n')
    print('LOCAL_INTEGRATION_PASS',flush=True)
    lease.close()


if __name__=='__main__':main()
