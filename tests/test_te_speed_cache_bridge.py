"""Real TE 3.3 cache math, without replacing H3's mask/USP forward."""
import importlib
from types import SimpleNamespace

import pytest
import torch

from .path_helpers import comfy_root
from .test_mmh3director_bridge import _proxy
from .test_minimax_adaln_curves import make_model, parallel  # noqa: F401


CONFIG = dict(processing_control_value=.08, processing_percent_1=.1,
              processing_percent_2=.9, mcs=2, device='cpu', mode='4-step LoRA')


def bridge():
    return importlib.import_module('raylight.integrations.mmh3director.te_speed_cache')


@pytest.fixture
def api():
    path=comfy_root()/'custom_nodes/TE-Speed-MiniMaxH3'
    if not (path/'nodes.pyd').is_file():
        pytest.skip('Optional Windows TE 3.3 binary is not installed')
    from raylight.integrations.mmh3director.external_acceleration import _load_package
    package=_load_package('_raylight_te_cache_contract',path)
    return importlib.import_module(package.__name__+'.nodes')


def test_ray_proxy_keeps_plain_config_and_does_not_call_native_forward_guard():
    module=bridge()
    proxy,_=_proxy()
    def original(self, model, processing_control_value, processing_percent_1, processing_percent_2, mcs, device, mode):
        raise RuntimeError('native forward guard')
    wrapped=module.make_proxy_te_patch(original)
    result=wrapped(object(),proxy,**CONFIG)[0]
    assert result is not proxy
    assert result.model_options['transformer_options']['raylight_external_te_speed']==CONFIG
    assert proxy.model_options['transformer_options']=={}
    with pytest.raises(RuntimeError,match='native forward guard'):
        wrapped(object(),object(),**CONFIG)


def test_runtime_does_not_patch_any_native_forward(api):
    module=bridge()
    from comfy.ldm.minimax.model import MiniMaxH3Model
    before=(MiniMaxH3Model.forward,MiniMaxH3Model._forward)
    runtime=module.create_runtime(CONFIG,50,api=api)
    assert (MiniMaxH3Model.forward,MiniMaxH3Model._forward)==before
    runtime.finish()


def test_rank_layouts_are_local_even_when_audio_is_entirely_on_other_rank():
    module=bridge()
    layout=SimpleNamespace(segments=[(0,2,'text'),(2,4,'audio'),(4,8,'video')])
    assert module.local_cache_layout(layout,0,4).segments==[(0,2,'text'),(2,4,'audio')]
    assert module.local_cache_layout(layout,1,4).segments==[(0,4,'video')]


def test_real_cache_skips_middle_only_after_consensus_and_preserves_rejected_input(api):
    module=bridge()
    runtime=module.create_runtime(CONFIG,50,api=api)
    layout=SimpleNamespace(segments=[(0,2,'text'),(2,4,'audio'),(4,8,'video')])
    calls=[]
    def run(hidden,indices):
        for index in indices:
            calls.append(index)
            hidden=hidden+(index+1)/100.
        return hidden
    def once(sigma,agree):
        opts={'sample_sigmas':torch.tensor([1.,.75,.5,.25,0.]),
              'sigmas':torch.tensor([sigma]),'cond_or_uncond':[0]}
        return module.run_cached_blocks(runtime,torch.ones(8,4),layout,opts,run,agree,
                                        rank=0,local_rows=8,block_count=50)
    first=once(1.,lambda value:value)
    assert len(calls)==50
    # One rank is ready to reuse; the other rejects. No cached residual may
    # touch this input before agreement, or the full run would add it twice.
    second=once(.75,lambda value:False)
    assert len(calls)==100
    third=once(.5,lambda value:value)
    assert len(calls)==112  # two exact prefix/suffix spans, 38 middle blocks skipped
    for result in (first,second,third):
        torch.testing.assert_close(result,torch.full((8,4),13.75),rtol=1e-5,atol=1e-5)
    assert runtime._raylight_stats==dict(full_forwards=2,cached_forwards=1,
                                        consensus_forced_full=1,skipped_blocks=38)
    runtime.finish()
    assert not runtime.states


def test_default_worker_replay_constructs_controller_not_foreign_model_forward(api, monkeypatch):
    module=bridge()
    from raylight.integrations.mmh3director import external_acceleration as external
    proxy,_=_proxy()
    proxy.model_options['transformer_options']['raylight_external_te_speed']=dict(CONFIG)
    def forbidden():
        raise AssertionError('foreign whole-model forward must not be installed')
    monkeypatch.setattr(external,'_te_speed_factory',forbidden)
    # Real package API; only discovery is supplied because tests may use an
    # optional-node root different from Comfy's process-global folder registry.
    create=module.create_runtime
    monkeypatch.setattr(module,'create_runtime',lambda config,count:create(config,count,api=api))
    result=external.apply_external_acceleration(proxy)
    runtime=result.model_options['transformer_options']['te_speed_minimax_h3_runtime']
    assert isinstance(runtime,module.OwnedTECache)
    assert module.finish_owned_runtime(result)['cached_forwards']==0
    assert 'te_speed_minimax_h3_runtime' not in result.model_options['transformer_options']


def test_real_usp_forward_uses_cache_controller_and_preserves_dense_math(api,parallel):
    import copy
    module=bridge()
    model=make_model()
    reference=copy.deepcopy(model)
    runtime=module.create_runtime(CONFIG,len(model.blocks),api=api)
    options={'te_speed_minimax_h3_runtime':runtime,'sample_sigmas':torch.tensor([1.,.5,0.]),'sigmas':torch.tensor([1.]),'cond_or_uncond':[0]}
    inputs=([torch.ones(1,24,2,4,4),torch.ones(1,32,2,4)],torch.tensor([650.]),torch.ones(1,3,96))
    with torch.inference_mode():
        wanted=reference._forward(*inputs)
        actual=parallel.usp_dit_forward(model,*inputs,transformer_options=options)
    assert runtime._raylight_stats['full_forwards']==1
    for observed,expected in zip(actual,wanted):
        torch.testing.assert_close(observed,expected,rtol=1e-5,atol=1e-5)
    runtime.finish()


def test_model_option_clone_starts_fresh_cache_without_copying_live_residuals(api):
    import copy
    module=bridge()
    runtime=module.create_runtime(CONFIG,50,api=api)
    state=runtime.state_for({'sample_sigmas':torch.tensor([1.,.75,.5,.25,0.]),'sigmas':torch.tensor([1.]),'cond_or_uncond':[0]})
    state.set_cache_ranges(SimpleNamespace(segments=[(0,8,'video')]))
    assert not state.prepare_middle(torch.ones(8,4),{'sample_sigmas':torch.tensor([1.,.75,.5,.25,0.]),'sigmas':torch.tensor([1.])})
    state.store_middle(torch.ones(8,4)*2)
    cloned=copy.deepcopy(runtime)
    assert cloned is not runtime and not cloned.states
    assert runtime.states
    cloned.finish()
    runtime.finish()


def test_selected_block_ranges_keep_memory_trace_points(parallel, monkeypatch):
    traces=[]
    monkeypatch.setattr(parallel, '_trace_cuda_memory', traces.append)
    def block(hidden, *_args, **_kwargs):
        return hidden + 1
    model=SimpleNamespace(blocks=[block, block, block])
    result=parallel._run_block_range(model, torch.ones(2,4), [0,2], None, None, None, {}, None)
    assert torch.equal(result, torch.full((2,4),3.))
    assert traces == ['block_0_end', 'block_2_end']


def test_te_consensus_requires_every_rank_and_preserves_no_transport_single_rank(parallel, monkeypatch):
    assert parallel._te_agreement(True, 'cpu')
    monkeypatch.setattr(parallel, 'get_sequence_parallel_world_size', lambda:2)
    monkeypatch.setattr(parallel, 'get_sp_group', lambda:SimpleNamespace(all_gather=lambda vote,dim:torch.cat((vote,torch.zeros_like(vote)),dim)))
    assert not parallel._te_agreement(True,'cpu')
    monkeypatch.setattr(parallel, 'get_sp_group', lambda:SimpleNamespace(all_gather=lambda vote,dim:torch.cat((vote,torch.ones_like(vote)),dim)))
    assert parallel._te_agreement(True,'cpu')
    assert not parallel._te_agreement(False,'cpu')


def test_installer_is_idempotent_and_finishing_always_drops_failed_cache(monkeypatch):
    import nodes
    module=bridge()
    class TE:
        def patch(self,model,**kwargs):
            return model
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS,'TESpeedMiniMaxH3',TE)
    assert module.install_te_proxy_bridge()
    wrapped=TE.patch
    assert not module.install_te_proxy_bridge() and TE.patch is wrapped
    class Broken:
        states={1:object()}
        def finish(self):
            raise RuntimeError('failed cache cleanup')
    runtime=module.OwnedTECache(Broken())
    model=SimpleNamespace(model_options={'transformer_options':{module.RUNTIME_KEY:runtime}})
    with pytest.raises(RuntimeError,match='failed cache cleanup'):
        module.finish_owned_runtime(model)
    assert not runtime.states
    assert module.RUNTIME_KEY not in model.model_options['transformer_options']
