"""Current-B input-metric conditioning; fixed inference state, no old images."""
import json
from pathlib import Path
import torch

from models.hoep_adaptation import Learner as Parent
from backbone.b_input_metric import condition_b_gradient


class Learner(Parent):
    def __init__(self,args):
        self.b_metric_kind=args.get('sa_b_metric_kind','off')
        if self.b_metric_kind not in ('current','history'):raise ValueError('unknown B metric')
        if args.get('sa_total_history_weight',0) or args.get('sa_allseen_proto_weight',0):
            raise ValueError('B metric experiments isolate the optimizer from extra losses')
        self.b_metric_damping=float(args.get('sa_b_metric_damping',0.1))
        self.b_metric_history_mix=float(args.get('sa_b_metric_history_mix',1.0))
        super().__init__(args)
        self._metric_handles=[]
        self._metric_wrappers=[]
        self._metric_cov={}
        self._metric_task=None
        self._metric_stats={}

    def incremental_train(self,data_manager):
        for handle in self._metric_handles:handle.remove()
        self._metric_handles=[];self._metric_cov={};self._metric_wrappers=[];self._metric_task=None
        self._metric_stats=dict(applied_branches=0,warmup_branches=0,skipped_branches=0,
                                cosine_sum=0.,max_relative_norm_error=0.)
        try:super().incremental_train(data_manager)
        finally:
            for handle in self._metric_handles:handle.remove()
            self._metric_handles=[];self._metric_cov={};self._metric_wrappers=[]
        if self.b_metric_kind=='history':
            self._collect_current_moments(data_manager)
            self._save_adaptation_state()
            directory=Path(self.args['filepath'])/'b_metric';directory.mkdir(exist_ok=True)
            torch.save(dict(task=self._cur_task,tokens=self._history_count,images=self._history_images,
                            moments=[x.detach().cpu() for x in self._history_moments],
                            source='own task training images at its deployed boundary; no old images revisited'),
                       directory/f'task_{self._cur_task:03d}_historical_moments.pt')

    def _install_metric_hooks(self):
        self._metric_task=self._cur_task
        self._metric_wrappers=self._raw_network().backbone._recoverability_wrappers()
        for i,wrapper in enumerate(self._metric_wrappers):
            for branch in ('q','v'):
                def capture(module,inputs,output,key=(i,branch)):
                    if not module.training:return
                    with torch.no_grad():
                        z=output.detach().float().reshape(-1,output.shape[-1])
                        if not bool(torch.isfinite(z).all()):raise FloatingPointError('nonfinite LoRA input')
                        self._metric_cov[key]=(z.T@z)/len(z)
                self._metric_handles.append(getattr(wrapper,'a_'+branch).register_forward_hook(capture))

    def _additional_training_losses(self,inputs=None,targets=None,features=None):
        result=super()._additional_training_losses(inputs,targets,features)
        if self._cur_task>0 and self._metric_task!=self._cur_task:
            # A first unconditioned batch installs hooks on the new task model.
            self._install_metric_hooks()
        return result

    def _after_backward(self,inputs=None,targets=None,optimizer=None):
        super()._after_backward(inputs,targets,optimizer)
        if self._cur_task==0:return
        with torch.no_grad():
            for i,wrapper in enumerate(self._metric_wrappers):
                for branch in ('q','v'):
                    b=getattr(wrapper,'b_'+branch).weight
                    cov=self._metric_cov.pop((i,branch),None)
                    if cov is None:
                        self._metric_stats['warmup_branches']+=1;continue
                    if b.grad is None:raise RuntimeError('current B has no gradient')
                    old_cov=None
                    if self.b_metric_kind=='history':
                        assert len(self._history_moments)==len(self._metric_wrappers)
                        a=getattr(wrapper,'a_'+branch).weight.detach().double()
                        old_cov=(a@self._history_moments[i])@a.T
                    new,record=condition_b_gradient(b.grad,cov,old_cov,damping=self.b_metric_damping,
                                                    history_mix=self.b_metric_history_mix)
                    if record['skipped']:
                        self._metric_stats['skipped_branches']+=1;continue
                    b.grad.copy_(new)
                    self._metric_stats['applied_branches']+=1
                    self._metric_stats['cosine_sum']+=record['cosine']
                    self._metric_stats['max_relative_norm_error']=max(
                        self._metric_stats['max_relative_norm_error'],record['relative_norm_error'])

    def _before_task_save(self,raw_network,train_loader):
        super()._before_task_save(raw_network,train_loader)
        directory=Path(self.args['filepath'])/'b_metric';directory.mkdir(exist_ok=True)
        record=dict(self._metric_stats,task=self._cur_task,kind=self.b_metric_kind,
                    damping=self.b_metric_damping,history_mix=self.b_metric_history_mix,
                    historical_images=self._history_images,
                    historical_statistics_bytes=sum(x.numel()*x.element_size() for x in self._history_moments))
        if record['applied_branches']:
            record['mean_gradient_cosine']=record['cosine_sum']/record['applied_branches']
        (directory/f'task_{self._cur_task:03d}.json').write_text(json.dumps(record,indent=2)+'\n')
