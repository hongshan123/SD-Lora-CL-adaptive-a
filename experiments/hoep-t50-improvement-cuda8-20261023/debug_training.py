"""Locate first nonfinite gradient; diagnostic output is not a method result."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import torch
from models.hoep_adaptation import Learner

original=Learner._after_backward


def checked(self,*args,**kwargs):
    for name,parameter in self._raw_network().named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError('FIRST_NONFINITE_GRADIENT_BEFORE_HOEP '+name)
    result=original(self,*args,**kwargs)
    for name,parameter in self._raw_network().named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError('FIRST_NONFINITE_GRADIENT_AFTER_HOEP '+name)
    return result


Learner._after_backward=checked
from trainer import train

config=json.loads(Path(sys.argv[1]).read_text());config['config']=sys.argv[1]
with torch.autograd.detect_anomaly(check_nan=True):train(config)
