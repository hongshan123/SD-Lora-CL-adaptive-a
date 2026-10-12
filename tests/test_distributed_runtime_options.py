import pytest
import time
from datetime import timedelta


def test_distributed_timeout_only_changes_wait_tolerance():
    from trainer import distributed_timeout
    assert distributed_timeout({})==timedelta(seconds=600)
    assert distributed_timeout({'dist_timeout_seconds':3600})==timedelta(seconds=3600)


@pytest.mark.parametrize('seconds',[0,-1,True,1.5])
def test_invalid_distributed_timeout_is_rejected(seconds):
    from trainer import distributed_timeout
    with pytest.raises(ValueError):distributed_timeout({'dist_timeout_seconds':seconds})


def test_stack_diagnostics_capture_rank_without_touching_model_rng(tmp_path):
    from trainer import start_stack_diagnostics, stop_stack_diagnostics
    from utils.rng_utils import rng_state_hash
    before=rng_state_hash()
    stream=start_stack_diagnostics({'filepath':str(tmp_path),'rank':1,'sa_stack_trace_seconds':1})
    try:time.sleep(1.1)
    finally:stop_stack_diagnostics(stream)
    assert before==rng_state_hash()
    assert 'test_distributed_runtime_options.py' in (tmp_path/'rank1_stacks.log').read_text()
