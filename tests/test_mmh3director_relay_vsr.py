"""Director VSR must preserve native routing and all user sampling settings."""
from contextlib import contextmanager
import importlib
from types import SimpleNamespace

import pytest
import torch

from tests.test_mmh3director_bridge import _proxy


def module():
    return importlib.import_module("raylight.integrations.mmh3director.relay_vsr")


def test_native_director_vsr_passthrough_does_not_acquire_gpu_lease():
    bridge = module()
    frames, pack = object(), {"enabled": True}
    wrapped = bridge.make_relay_vsr_wrapper(lambda f, p: (f, p))
    assert wrapped(frames, pack) == (frames, pack)


def test_execute_scope_activates_only_ray_director_with_vsr():
    bridge = module()
    proxy, workers = _proxy()
    def execute(*, model, vsr=None):
        return bridge.current_director_vsr_workers()
    wrapped = bridge.make_execute_wrapper(execute)
    assert wrapped(model=object(), vsr={"enabled": True}) is None
    assert wrapped(model=proxy) is None
    assert wrapped(model=proxy, vsr={"enabled": True}) == tuple(workers)
    assert bridge.current_director_vsr_workers() is None


@pytest.fixture
def pipeline(monkeypatch):
    bridge = module()
    events = []
    workers = (object(), object())
    @contextmanager
    def authorize(actual, *, cleanup):
        assert tuple(actual) == workers
        events.append("lease")
        try:
            yield
        finally:
            cleanup()
            events.append("return")
    monkeypatch.setattr(bridge, "authorize_flash", authorize)
    monkeypatch.setattr(bridge, "extract_vae_source_path", lambda vae: "flash.safetensors")
    monkeypatch.setattr(bridge, "_checkpoint_scale", lambda path: 2)
    monkeypatch.setattr(bridge, "_release_main_gpu_models", lambda: events.append("unload"))
    monkeypatch.setattr(bridge, "release_local_pool", lambda: events.append("close"))
    class RelayVSRModel:
        latent_stride = 8
    model = SimpleNamespace(model=SimpleNamespace(diffusion_model=RelayVSRModel()))
    pack = {"enabled": True, "model": model, "vae": object(), "seed": 73,
            "steps": 2, "cfg": .75, "sampler": "heun", "scheduler": "normal",
            "denoise": .9, "detail": .65}
    return bridge, workers, pack, events


def test_scoped_pipeline_forwards_settings_and_preserves_frame_order(pipeline, monkeypatch):
    bridge, workers, pack, events = pipeline
    frames = torch.arange(3).float()[:, None, None, None].expand(3, 64, 64, 3).clone()
    sampled = {"samples": torch.ones(1, 48, 2, 8, 8)}
    def sample(images, settings):
        assert settings is pack
        assert (settings["seed"], settings["steps"], settings["cfg"], settings["sampler"],
                settings["scheduler"], settings["denoise"]) == (73, 2, .75, "heun", "normal", .9)
        events.append("sample")
        return sampled
    def decode(path, latent, images, detail, **kwargs):
        assert path == "flash.safetensors" and latent is sampled["samples"]
        assert detail == .65 and kwargs["worker_count"] == 2 and kwargs["keep_workers"] is False
        events.append("flash")
        return images.repeat_interleave(2, 1).repeat_interleave(2, 2)
    monkeypatch.setattr(bridge, "_sample_relay_vsr", sample)
    monkeypatch.setattr(bridge, "run_local_flash", decode)
    output = bridge.run_director_relay_vsr(frames, pack, workers)
    assert output.shape == (3, 128, 128, 3)
    assert output[:, 0, 0, 0].tolist() == [0., 1., 2.]
    assert events.index("lease") < events.index("sample") < events.index("flash") < events.index("close") < events.index("return")


def test_bad_model_or_scale_is_rejected_before_lease_and_unload(pipeline):
    bridge, workers, pack, events = pipeline
    pack["model"].model.diffusion_model.latent_stride = 4
    with pytest.raises(ValueError, match="scale|factor|stride"):
        bridge.run_director_relay_vsr(torch.zeros(1, 64, 64, 3), pack, workers)
    assert events == []


def test_sampling_failure_cleans_before_returning_to_h3(pipeline, monkeypatch):
    bridge, workers, pack, events = pipeline
    def fail(*args):
        raise RuntimeError("sampling failed")
    monkeypatch.setattr(bridge, "_sample_relay_vsr", fail)
    with pytest.raises(RuntimeError, match="sampling failed"):
        bridge.run_director_relay_vsr(torch.zeros(1, 64, 64, 3), pack, workers)
    assert events[-3:] == ["close", "unload", "return"]


def test_wrong_decode_frame_count_does_not_escape_as_video(pipeline, monkeypatch):
    bridge, workers, pack, events = pipeline
    monkeypatch.setattr(bridge, "_sample_relay_vsr", lambda *a: {"samples": torch.zeros(1)})
    monkeypatch.setattr(bridge, "run_local_flash", lambda *a, **k: torch.zeros(1, 128, 128, 3))
    with pytest.raises(RuntimeError, match="frame"):
        bridge.run_director_relay_vsr(torch.zeros(2, 64, 64, 3), pack, workers)
    assert events[-1] == "return"


def test_director_flash_receives_real_interrupt_check_and_cleans_before_lease_return(pipeline, monkeypatch):
    import comfy.model_management as mm
    bridge,workers,pack,events=pipeline
    class Cancelled(Exception):
        pass
    def interrupt():
        raise Cancelled('user stopped Flash')
    monkeypatch.setattr(mm,'throw_exception_if_processing_interrupted',interrupt)
    monkeypatch.setattr(bridge,'_sample_relay_vsr',lambda *_args:{'samples':torch.zeros(1)})
    def decode(*_args,**kwargs):
        checker=kwargs.get('interrupt_check')
        assert callable(checker), 'Director must forward cancellation into Flash dispatch'
        checker()
    monkeypatch.setattr(bridge,'run_local_flash',decode)
    with pytest.raises(Cancelled,match='user stopped Flash'):
        bridge.run_director_relay_vsr(torch.zeros(2,64,64,3),pack,workers)
    assert events[-3:]==['close','unload','return']


def test_native_sampler_receives_all_configured_parameters(pipeline, monkeypatch):
    bridge, _, pack, _ = pipeline
    import comfy_extras.nodes_wan as native
    import nodes
    positive, latent = [[torch.ones(1), {}]], {"samples": torch.zeros(1)}
    monkeypatch.setattr(native, "RelayVSRConditioning", SimpleNamespace(execute=lambda m, i: (positive, latent)), raising=False)
    def sample(model, seed, steps, cfg, sampler, scheduler, pos, neg, samples, *, denoise):
        assert model is pack["model"]
        assert (seed, steps, cfg, sampler, scheduler, denoise) == (73, 2, .75, "heun", "normal", .9)
        assert pos is positive and neg is positive and samples is latent
        return (latent,)
    monkeypatch.setattr(nodes, "common_ksampler", sample)
    assert bridge._sample_relay_vsr(torch.zeros(1, 64, 64, 3), pack) is latent


def test_only_vsr_sample_and_flash_use_relaxed_allocator(pipeline,monkeypatch):
    bridge,workers,pack,events=pipeline
    active=[]
    @contextmanager
    def scope():
        active.append(True)
        try:yield
        finally:active.pop();events.append('allocator restored')
    monkeypatch.setattr(bridge,'native_vsr_allocator',scope,raising=False)
    def sample(*_):
        assert active, 'Native VSR sampling retained the H3 split limit'
        return {'samples':torch.zeros(1)}
    def decode(*_,**__):
        assert active, 'Flash decoding retained the H3 split limit'
        return torch.zeros(1,128,128,3)
    monkeypatch.setattr(bridge,'_sample_relay_vsr',sample)
    monkeypatch.setattr(bridge,'run_local_flash',decode)
    bridge.run_director_relay_vsr(torch.zeros(1,64,64,3),pack,workers)
    assert not active
    assert events.index('allocator restored')<events.index('return')


def test_zero_vsr_knobs_do_not_collide_with_one_in_ray_final_fingerprints():
    bridge = module()
    # The current upstream uses `value or 1`, so its legal zero settings collide.
    original = lambda plan: {'vsr': True, 'vsr_cfg': 1., 'vsr_denoise': 1., 'vsr_detail': 1.}
    wrapped = bridge.make_fingerprint_wrapper(original)
    plan = SimpleNamespace(vsr={'enabled': True, 'cfg': 0., 'denoise': 0., 'detail': 0.})
    assert wrapped(plan) == original(plan)  # Native Director remains unchanged.
    marker = bridge._ACTIVE.set((object(), object()))
    try:
        result=wrapped(plan)
        assert {key:result[key] for key in ('vsr','vsr_cfg','vsr_denoise','vsr_detail')} == {'vsr': True, 'vsr_cfg': 0., 'vsr_denoise': 0., 'vsr_detail': 0.}
    finally:
        bridge._ACTIVE.reset(marker)
