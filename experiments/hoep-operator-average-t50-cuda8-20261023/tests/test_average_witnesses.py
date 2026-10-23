import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from average_checks import validate_average_run


def write(path, value):path.write_text(json.dumps(value) + '\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class WitnessGateTests(unittest.TestCase):
    def test_actual_average_verified_and_rehashed_forged_witness_rejected(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'branch_average';folder.mkdir()
            (folder/'task_001').mkdir()
            write(folder/'task_000.json',dict(task=0,enabled=False,task0_unchanged=True,
                                             average_count=0,optimizer_steps=0))
            generator=torch.Generator().manual_seed(514)
            a=torch.eye(768)[:10].unsqueeze(0).repeat(24,1,1)
            b1=torch.randn(24,768,10,generator=generator)*.01
            b2=b1+torch.randn(24,768,10,generator=generator)*.001
            s=torch.tensor([.8]);history=torch.zeros_like(b1)
            witnesses=[]
            for step,b in enumerate([b1,b2],1):
                path=folder/'task_001'/f'step_{step:06d}.pt'
                torch.save(dict(task_id=1,step=step,down=a,current_up=b,current_scale=s),path)
                witnesses.append(dict(step=step,file=str(path.relative_to(root)),sha256=sha(path)))
            mean=(s.double()*b1.double()@a.double()+s.double()*b2.double()@a.double())/2
            coefficient=((s.double()*b1.double()+s.double()*b2.double())/2).float()
            archive=folder/'task_001_pre_average.pt'
            torch.save(dict(task_id=1,down=a,current_up=b2,current_scale=s,
                            historical_up=history,mean_current_operators=mean),archive)
            snapshot=root/'task_snapshots/task_001';snapshot.mkdir(parents=True)
            pre=snapshot/'pre_merge.pt'
            torch.save(dict(branches=[dict(down=a[i],current_up=coefficient[i],
                            historical_up=history[i]) for i in range(24)],
                            current_scale=torch.tensor(1.)),pre)
            write(snapshot/'complete.json',dict(sha256={'pre_merge.pt':sha(pre)}))
            record=dict(task=1,enabled=True,fraction=.25,current_training_images=120,
                        train_loader_drop_last=False,train_loader_batches=1,optimizer_steps=2,
                        expected_optimizer_steps=2,window_start_step=1,average_count=2,
                        witnesses=witnesses,pre_average_archive=str(archive),pre_average_sha256=sha(archive),
                        written_scale=1.,current_write_relative_error=1e-7,a_unchanged=True,
                        historical_g_unchanged=True,rng_unchanged=True,
                        projection_residual_squared_relative=0.,relative_current_operator_change=.1)
            write(folder/'task_001.json',record)
            config=dict(sa_branch_average_fraction=.25,batch_size=128,epochs=2)
            self.assertEqual(len(validate_average_run(config,root,2)),2)
            path=root/witnesses[0]['file']
            forged=torch.load(path,weights_only=True)
            forged['current_up'][0,0,0]+=1.
            torch.save(forged,path)
            record['witnesses'][0]['sha256']=sha(path)
            write(folder/'task_001.json',record)
            with self.assertRaises(AssertionError):validate_average_run(config,root,2)


if __name__=='__main__':unittest.main()
