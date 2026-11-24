"""Official Flash direct math and explicit local-GPU ownership, without core edits."""
import importlib

import pytest
import torch


def local_module():
    try:
        return importlib.import_module("raylight.relayvsr.local")
    except ModuleNotFoundError:
        pytest.fail("Shared-process Flash executor not implemented")


@pytest.mark.parametrize("scale", [2, 4])
def test_direct_flash_pixels_equal_native_model_decode(scale):
    native=pytest.importorskip("comfy.ldm.wan.model_vsr")
    module=local_module()
    old_threads=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.inference_mode(),torch.random.fork_rng(devices=[]):
            torch.manual_seed(681)
            model=native.RelayFlash(scale,dim=128,depth=1,num_heads=2,kv_groups=1,hidden_dim=64,head_dim=64).float().eval()
            for name,p in model.named_parameters():
                if "norm" in name and name.endswith("weight"):
                    p.fill_(1)
                else:
                    p.normal_(0,.01)
            side=64 if scale==2 else 32
            images=torch.rand(17,side,side,3)
            latents=torch.randn(1,48,3,8,8)*.01
            expected=model.decode(latents,lq_images=images,detail=.7).movedim(1,-1).reshape(17,side*scale,side*scale,3)
            actual=module.decode_flash_model(model,latents,images,.7,torch.device("cpu"))
            assert torch.equal(actual,expected)
    finally:
        torch.set_num_threads(old_threads)


def test_local_mode_rejects_borrowing_ray_actors_before_loading():
    module=local_module()
    with pytest.raises(ValueError,match="ray_actors|borrow"):
        module.run_local_flash("unused",torch.zeros(1,48,1,8,8),torch.zeros(1,64,64,3),1.,
                               borrowed_workers=[object()])


def test_node_does_not_expose_unshipped_ray_prototype(monkeypatch):
    from raylight.relayvsr.nodes import RayRelayVSRDecode
    schema=RayRelayVSRDecode.INPUT_TYPES()
    required=schema["required"]
    assert "execution_mode" not in required
    assert "ray_actors" not in schema.get("optional",{})


@pytest.mark.parametrize("payload",[{}, {"workers": []}])
def test_connected_ray_payload_rejected_before_model_unload(monkeypatch,payload):
    import comfy.model_management as mm
    from raylight.relayvsr.nodes import RayRelayVSRDecode
    def fail():
        pytest.fail("Invalid Ray payload changed Comfy state")
    monkeypatch.setattr(mm,"unload_all_models",fail)
    with pytest.raises(ValueError,match="unshipped|ray_actors"):
        RayRelayVSRDecode().decode({},torch.zeros(1,64,64,3),"unused",1.,2,False,ray_actors=payload)


def test_node_local_route_uses_direct_flash_executor(monkeypatch):
    import comfy.model_management as mm
    import folder_paths
    from raylight.relayvsr.nodes import RayRelayVSRDecode
    module=local_module()
    result=torch.ones(1,128,128,3)
    def execute(path,weights,images,detail,**kwargs):
        assert kwargs["worker_count"]==2 and kwargs["keep_workers"] is False
        return result
    monkeypatch.setattr(module,"run_local_flash",execute)
    monkeypatch.setattr(folder_paths,"get_full_path_or_raise",lambda *a:"flash.safetensors")
    monkeypatch.setattr(mm,"unload_all_models",lambda:None)
    monkeypatch.setattr(mm,"soft_empty_cache",lambda:None)
    output=RayRelayVSRDecode().decode({"samples":torch.zeros(1,48,1,8,8)},torch.zeros(1,64,64,3),
        "flash.safetensors",1.,2,False,execution_mode="local_threads")
    assert output[0] is result
