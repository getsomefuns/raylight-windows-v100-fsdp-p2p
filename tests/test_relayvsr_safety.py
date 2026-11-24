"""Review regressions: no foreign Ray work, deterministic cleanup and bounded lifetimes."""
from concurrent.futures import Future
import gc
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch

from raylight.relayvsr.planning import plan_relay_windows


def test_completed_result_reference_is_gone_before_polling_replacements():
    from raylight.relayvsr.runtime import dispatch_window_jobs
    refs=[]
    polls=[]
    def submit(worker,job):
        future=Future()
        future.set_result(torch.arange(job.pixel_start,job.pixel_stop).float()[:,None,None,None].expand(-1,2,2,3))
        refs.append(weakref.ref(future))
        return future
    def poll(pending):
        if polls:
            assert refs[0]() is None,"Merged output remains pinned during replacement work"
        polls.append(1)
        return [pending[0]]
    result=dispatch_window_jobs(plan_relay_windows(61),2,submit,poll,lambda ref:ref.result(),torch.empty(61,2,2,3))
    assert torch.equal(result[:,0,0,0],torch.arange(61).float())


def test_unregistered_vae_model_cycle_is_broken_without_cyclic_gc(monkeypatch):
    import comfy.model_management as mm
    from comfy.model_patcher import ModelPatcher
    from raylight.relayvsr.worker import RelayFlashDecoder
    model=torch.nn.Linear(3,3)
    patcher=ModelPatcher(model,load_device=torch.device("cpu"),offload_device=torch.device("cpu"))
    model.current_patcher=patcher
    decoder=RelayFlashDecoder("unused")
    decoder.vae=SimpleNamespace(patcher=patcher,first_stage_model=model)
    model_ref,patcher_ref=weakref.ref(model),weakref.ref(patcher)
    monkeypatch.setattr(mm,"current_loaded_models",[])
    monkeypatch.setattr(mm,"soft_empty_cache",lambda:None)
    enabled=gc.isenabled()
    gc.disable()
    try:
        del model,patcher
        decoder.close()
        assert model_ref() is None and patcher_ref() is None,"Owned GPU model needs cyclic GC to release"
    finally:
        if enabled:
            gc.enable()
        gc.collect()


def test_concurrent_node_rejected_before_comfy_unload(monkeypatch):
    import comfy.model_management as mm
    import folder_paths
    from raylight.relayvsr import local
    from raylight.relayvsr.nodes import RayRelayVSRDecode
    changed=[]
    monkeypatch.setattr(folder_paths,"get_full_path_or_raise",lambda *a:"unused")
    monkeypatch.setattr(local,"_checkpoint_scale",lambda *a:2)
    monkeypatch.setattr(mm,"unload_all_models",lambda:changed.append("unload"))
    monkeypatch.setattr(mm,"soft_empty_cache",lambda:changed.append("trim"))
    local._LOCK.acquire()
    try:
        with pytest.raises(RuntimeError,match="active|busy"):
            RayRelayVSRDecode().decode({"samples":torch.zeros(1,48,1,8,8)},torch.zeros(1,64,64,3),"unused",1.,2,False)
    finally:
        local._LOCK.release()
    assert changed==[],"Rejected operation altered global Comfy GPU state"


@pytest.mark.parametrize("free_gpus",[1,2])
def test_shared_ray_cluster_rejected_even_when_warm_pool_exists(monkeypatch,free_gpus):
    import ray
    from raylight.relayvsr import local
    path=str(Path("unused").resolve())
    fake=SimpleNamespace(key=(path,1),decoders=[SimpleNamespace(trim=lambda:None)])
    monkeypatch.setattr(local,"_POOL",fake)
    monkeypatch.setattr(local,"_checkpoint_scale",lambda *a:2)
    monkeypatch.setattr(local,"dispatch_window_jobs",lambda *a:a[5])
    monkeypatch.setattr(ray,"is_initialized",lambda:True)
    monkeypatch.setattr(ray,"available_resources",lambda:{"GPU":free_gpus})
    with pytest.raises(RuntimeError,match="Ray|cluster"):
        local.run_local_flash("unused",torch.zeros(1,48,1,8,8),torch.zeros(1,64,64,3),1.,keep_workers=True)


def test_failed_idle_trim_invalidates_pool_and_records_failure(monkeypatch,tmp_path):
    import ray
    from raylight.relayvsr import local
    path=str(Path("unused").resolve())
    closed=[]
    def fail():
        raise RuntimeError("controlled trim failure")
    pool=SimpleNamespace(key=(path,1),decoders=[SimpleNamespace(trim=fail)],close=lambda:closed.append(1))
    monkeypatch.setattr(local,"_POOL",pool)
    monkeypatch.setattr(local,"_checkpoint_scale",lambda *a:2)
    monkeypatch.setattr(local,"dispatch_window_jobs",lambda *a:a[5].zero_())
    monkeypatch.setattr(ray,"is_initialized",lambda:False)
    monkeypatch.setenv("RAYLIGHT_RELAYVSR_TRACE_DIR",str(tmp_path))
    with pytest.raises(RuntimeError,match="controlled trim failure"):
        local.run_local_flash("unused",torch.zeros(1,48,1,8,8),torch.zeros(1,64,64,3),1.,keep_workers=True)
    assert local._POOL is None and closed==[1]
    trace=json.loads(next(tmp_path.glob("*.json")).read_text())
    assert trace["completed"] is False
    created=[]
    def fresh(path,count):
        created.append(count)
        return SimpleNamespace(key=(path,count),decoders=[],close=lambda:closed.append(2))
    monkeypatch.setattr(local,"_LocalPool",fresh)
    output=local.run_local_flash("unused",torch.zeros(1,48,1,8,8),torch.zeros(1,64,64,3),1.)
    assert output.shape==(1,128,128,3) and created==[1] and closed==[1,2]
    assert local._POOL is None


def test_direct_decode_honors_real_wrapper_output_processing():
    native=pytest.importorskip("comfy.ldm.wan.model_vsr")
    from raylight.relayvsr.local import decode_flash_model
    old=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.inference_mode():
            model=native.RelayFlash(2,dim=128,depth=1,num_heads=2,kv_groups=1,hidden_dim=64,head_dim=64).float().eval()
            for p in model.parameters():
                p.zero_()
            images=torch.zeros(1,64,64,3)
            z=torch.zeros(1,48,1,8,8)
            expected=model.decode(z,lq_images=images).add_(.125).movedim(1,-1).reshape(1,128,128,3)
            actual=decode_flash_model(model,z,images,1.,torch.device("cpu"),process_output=lambda t:t.add_(.125))
            assert torch.equal(actual,expected)
    finally:
        torch.set_num_threads(old)
