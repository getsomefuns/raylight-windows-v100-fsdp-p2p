"""Explicit node contract and isolation from generic Wan USP injections."""
import importlib
from types import SimpleNamespace

import pytest
import torch


def node_module():
    try:
        return importlib.import_module("raylight.relayvsr.nodes")
    except ModuleNotFoundError:
        pytest.fail("Explicit RelayVSR Flash node is missing")


def test_node_import_and_schema_do_not_require_native_relay_support(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "comfy.ldm.wan.model_vsr", None)
    node = node_module().RayRelayVSRDecode
    schema = node.INPUT_TYPES()
    assert schema["required"]["samples"][0] == "LATENT"
    assert schema["required"]["images"][0] == "IMAGE"
    assert schema["required"]["worker_count"][0] == [1, 2]
    assert schema["required"]["keep_workers"][1]["default"] is False
    assert "ray_actors" not in schema.get("optional", {})
    assert node.RETURN_TYPES == ("IMAGE",)


def test_node_returns_image_from_local_executor(monkeypatch):
    import comfy.model_management as mm
    import folder_paths
    runtime = importlib.import_module("raylight.relayvsr.local")
    module = node_module()
    images = torch.zeros(1, 64, 64, 3)
    result = torch.ones(1, 128, 128, 3)
    weights = torch.zeros(1, 48, 1, 8, 8)
    received = {}
    def run(path, samples, frames, detail, **kwargs):
        received.update(kwargs)
        assert str(path).endswith("flash.safetensors")
        assert samples is weights and frames is images
        return result
    monkeypatch.setattr(runtime, "run_local_flash", run)
    monkeypatch.setattr(folder_paths, "get_full_path_or_raise", lambda *a: "flash.safetensors")
    monkeypatch.setattr(mm, "unload_all_models", lambda: None)
    monkeypatch.setattr(mm, "soft_empty_cache", lambda: None)
    output = module.RayRelayVSRDecode().decode({"samples": weights}, images,
        "flash.safetensors", 1., 1, False)
    assert output[0] is result
    assert callable(received["prepare"])
    assert received["worker_count"] == 1


def test_relayvsr_is_rejected_before_generic_wan_handler(monkeypatch):
    from comfy import model_base
    from raylight.distributed_modules.usp import USPInjectRegistry
    class Relay(model_base.WAN21):
        pass
    monkeypatch.setattr(model_base, "WAN22_RelayVSR", Relay, raising=False)
    called = []
    monkeypatch.setattr(USPInjectRegistry, "_REGISTRY", {model_base.WAN21: lambda *a: called.append(1)})
    model = object.__new__(Relay)
    with pytest.raises(ValueError, match="RelayVSR.*Flash"):
        USPInjectRegistry.inject(SimpleNamespace(model=model), None, 0, False, False)
    assert not called


@pytest.mark.parametrize("relay_class_present", [True, False])
def test_other_models_keep_their_registered_handler(monkeypatch, relay_class_present):
    from comfy import model_base
    from raylight.distributed_modules.usp import USPInjectRegistry
    if not relay_class_present:
        monkeypatch.delattr(model_base, "WAN22_RelayVSR", raising=False)
    else:
        monkeypatch.setattr(model_base, "WAN22_RelayVSR", type("Relay", (), {}), raising=False)
    class ExistingModel:
        pass
    expected = object()
    monkeypatch.setattr(USPInjectRegistry, "_REGISTRY", {ExistingModel: lambda *a: expected})
    assert USPInjectRegistry.inject(SimpleNamespace(model=ExistingModel()), None, 0, False, False) is expected
