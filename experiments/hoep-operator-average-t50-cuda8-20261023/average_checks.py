"""Independently recompute each recorded average from actual factor witnesses."""
import hashlib
import json
import math
from pathlib import Path
import torch


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_load(path, digest):
    assert sha(path) == digest, path
    return torch.load(path, map_location='cpu', weights_only=True)


def validate_average_run(config, output, tasks):
    output=Path(output);records=[]
    for task in range(tasks):
        record=read(output/'branch_average'/f'task_{task:03d}.json')
        assert record['task']==task
        if task==0:
            assert record['enabled'] is False and record['task0_unchanged'] is True
            assert record['average_count']==record['optimizer_steps']==0
            records.append(record);continue
        assert record['enabled'] is True
        assert record['fraction']==config['sa_branch_average_fraction']
        batches=(record['current_training_images']//config['batch_size']
                 if record['train_loader_drop_last'] else
                 math.ceil(record['current_training_images']/config['batch_size']))
        assert batches==record['train_loader_batches']
        steps=batches*config['epochs']
        count=min(steps,max(2,math.ceil(steps*record['fraction'])))
        start=steps-count+1
        assert record['optimizer_steps']==record['expected_optimizer_steps']==steps
        assert record['window_start_step']==start and record['average_count']==count and count>=2
        witnesses=record['witnesses']
        assert [w['step'] for w in witnesses]==list(range(start,steps+1))
        archive=checked_load(Path(record['pre_average_archive']),record['pre_average_sha256'])
        assert archive['task_id']==task
        total=None;scale_norm=0.;last=None
        for witness in witnesses:
            path=output/witness['file'];assert path.resolve().is_relative_to(output.resolve())
            state=checked_load(path,witness['sha256'])
            assert state['task_id']==task and state['step']==witness['step']
            a,b,s=state['down'].double(),state['current_up'].double(),state['current_scale'].double()
            assert a.shape==b.transpose(1,2).shape==(24,10,768) and s.numel()==1
            assert all(bool(torch.isfinite(x).all()) for x in (a,b,s))
            value=(b*s.reshape(()))@a
            scale_norm+=float(value.norm())
            total=value.clone() if total is None else total+value
            last=state
        mean=total/count
        saved=archive['mean_current_operators'].double()
        error=float((mean-saved).norm()/max(scale_norm/count,1e-30))
        assert error<1e-9,('mean does not match actual witnesses',task,error)
        for key in ('down','current_up','current_scale'):
            assert torch.equal(archive[key],last[key]),('raw final state mismatch',task,key)
        directory=output/'task_snapshots'/f'task_{task:03d}'
        marker=read(directory/'complete.json')
        pre=checked_load(directory/'pre_merge.pt',marker['sha256']['pre_merge.pt'])
        down=torch.stack([b['down'] for b in pre['branches']])
        up=torch.stack([b['current_up'] for b in pre['branches']])
        history=torch.stack([b['historical_up'] for b in pre['branches']])
        assert torch.equal(down,archive['down']) and torch.equal(history,archive['historical_up'])
        assert float(pre['current_scale'])==1. and record['written_scale']==1.
        a=down.double();gram=a@a.transpose(1,2)
        coefficient=torch.linalg.solve(gram,(mean@a.transpose(1,2)).transpose(1,2)).transpose(1,2)
        projected=coefficient@a
        actual=up.double()*float(pre['current_scale'])@a
        relative=float((actual-projected).norm()/projected.norm().clamp_min(1e-30))
        assert relative<5e-6,('actual write mismatch',task,relative)
        assert record['current_write_relative_error']<5e-6
        assert record['a_unchanged'] and record['historical_g_unchanged'] and record['rng_unchanged']
        squared=float((mean-projected).square().sum()/mean.square().sum().clamp_min(1e-30))
        assert abs(squared-record['projection_residual_squared_relative'])<1e-9
        assert math.isfinite(record['relative_current_operator_change'])
        records.append(record)
    assert any(r.get('relative_current_operator_change',0)>1e-8 for r in records[1:]), 'averaging never changed an operator'
    return records
