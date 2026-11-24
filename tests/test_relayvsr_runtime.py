"""Bounded dispatch, failure/interrupt behavior and checkpoint shape admission."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import time
import pytest
import torch
from raylight.relayvsr.planning import plan_relay_windows
from raylight.relayvsr.runtime import dispatch_window_jobs
from raylight.relayvsr.worker import validate_relay_inputs


def test_out_of_order_workers_preserve_pixels_and_frame_coverage():
    seen=[]
    with ThreadPoolExecutor(max_workers=2) as executor:
        def execute(worker,job):
            time.sleep(.03 if job.index==0 else .003)
            seen.append((worker,job.index))
            return torch.arange(job.pixel_start,job.pixel_stop).float()[:,None,None,None].expand(-1,2,2,3)
        result=dispatch_window_jobs(plan_relay_windows(61),2,
            lambda w,j:executor.submit(execute,w,j),
            lambda refs:list(wait(refs,timeout=.1,return_when=FIRST_COMPLETED).done),
            lambda ref:ref.result(),torch.empty(61,2,2,3))
    assert torch.equal(result[:,0,0,0],torch.arange(61).float())
    assert {w for w,_ in seen}=={0,1} and seen[0][1]!=0
    assert sorted(i for _,i in seen)==[0,1,2,3]


def test_worker_failure_never_returns_partial_video():
    def receive(ref):
        raise RuntimeError("controlled failure")
    with pytest.raises(RuntimeError,match="controlled failure"):
        dispatch_window_jobs(plan_relay_windows(31),2,lambda w,j:j.index,
            lambda refs:[refs[0]],receive,torch.empty(31,2,2,3))


def test_interrupt_checked_while_waiting_for_worker():
    def interrupted():
        raise InterruptedError("interrupted")
    with pytest.raises(InterruptedError):
        dispatch_window_jobs(plan_relay_windows(31),2,lambda w,j:j.index,
            lambda refs:[],lambda ref:None,torch.empty(31,2,2,3),interrupt_check=interrupted)


@pytest.mark.parametrize("mutation",["count","size","batch","rgba","detail"])
def test_bad_inputs_fail_before_loading_gpu(mutation):
    z=torch.zeros(1,48,3,8,8)
    images=torch.zeros(17,64,64,3)
    detail=1.
    if mutation=="count": z=z[:,:,:2]
    elif mutation=="size": z=z[:,:,:,:7]
    elif mutation=="batch": z=z.expand(2,-1,-1,-1,-1)
    elif mutation=="rgba": images=torch.zeros(17,64,64,4)
    else: detail=float("nan")
    with pytest.raises(ValueError):
        validate_relay_inputs(z,images,2,detail)


def test_short_tail_keeps_shared_context():
    jobs=validate_relay_inputs(torch.zeros(1,48,3,8,8),torch.zeros(17,64,64,3),2,.5)
    assert len(jobs)==2
    assert (jobs[-1].pixel_start,jobs[-1].pixel_stop,jobs[-1].discard_prefix)==(15,17,1)
