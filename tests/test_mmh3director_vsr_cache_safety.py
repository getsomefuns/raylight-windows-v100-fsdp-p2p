"""Do not splice stale VSR pixels into a selectively regenerated export."""
import importlib
import json
from types import SimpleNamespace

import pytest
import torch

from .test_director_vsr_cache_contract import native_cache  # noqa: F401


def bridge():
    return importlib.import_module('raylight.integrations.mmh3director.relay_vsr')


def test_same_factor_different_checkpoint_or_options_changes_source_identity(tmp_path):
    module=bridge()
    identity=getattr(module,'vsr_source_fingerprint',None)
    assert callable(identity),'Missing VSR checkpoint/patch source identity'
    paths=[]
    for name in ('unet-a','unet-b','vae'):
        path=tmp_path/name
        path.write_bytes(name.encode())
        paths.append(path)
    def model(path):
        return SimpleNamespace(cached_patcher_init=(None,(str(path),{})),patches={},object_patches={},model_options={'transformer_options':{}})
    first=model(paths[0])
    pack={'model':first,'vae':SimpleNamespace(patcher=model(paths[2]))}
    base=identity(pack)
    assert identity(pack)==base
    pack['model']=model(paths[1])
    assert identity(pack)!=base
    pack['model']=first
    first.model_options['transformer_options']['strength']=.5
    assert identity(pack)!=base
    first.model_options['transformer_options'].clear()
    paths[2].write_bytes(b'replaced decoder weights')
    assert identity(pack)!=base


def test_vsr_stale_fill_rejects_changed_parameters_before_frame_decode(native_cache,tmp_path,monkeypatch):
    module=bridge()
    maker=getattr(module,'make_cache_loader_wrapper',None)
    assert callable(maker),'Missing scoped final-cache stale-fill validation'
    cache,segment,plan=native_cache
    stored={'vsr':True,'vsr_detail':1.,'raylight_vsr_sources':{'model':'a','vae':'v'}}
    expected={**stored,'vsr_detail':.5}
    (tmp_path/'seg_0000.meta.json').write_text(json.dumps(stored))
    monkeypatch.setattr(cache,'_cache_dir_or_none',lambda _:tmp_path)
    monkeypatch.setattr(cache,'segment_cache_fingerprint',lambda *_:expected)
    calls=[]
    def original(*_args,**_kwargs):
        calls.append('decode')
        return torch.ones(2,2,2,3)
    wrapped=maker(original,cache)
    marker=module._ACTIVE.set((object(),object()))
    try:
        with pytest.raises(RuntimeError,match='VSR.*[Ss]egment|segment.*VSR'):
            wrapped('case',segment,plan,allow_stale=True)
        assert not calls
        # Exact-miss path can still return None and regenerate the selected
        # segment; only stale final reuse must be refused.
        wrapped('case',segment,plan,allow_stale=False)
        assert calls==['decode']
        calls.clear()
        expected['vsr_detail']=1.
        assert wrapped('case',segment,plan,allow_stale=True).shape==(2,2,2,3)
        assert calls==['decode']
    finally:module._ACTIVE.reset(marker)
    expected['vsr_detail']=.5
    assert wrapped('case',segment,plan,allow_stale=True).shape==(2,2,2,3)  # unrelated native call unchanged


def test_final_source_identity_does_not_modify_first_pass(native_cache,tmp_path,monkeypatch):
    module=bridge()
    cache,segment,plan=native_cache
    pack_module=importlib.import_module(cache.__package__+'.relay_vsr.pack')
    original=pack_module.relay_vsr_fingerprint
    wrapped=module.make_fingerprint_wrapper(original)
    monkeypatch.setattr(pack_module,'relay_vsr_fingerprint',wrapped)
    plan.vsr={'enabled':True,'model':object(),'vae':object(),'factor_tag':'2:8','steps':1,'cfg':1.,'seed':42,'detail':1.,'denoise':1.,'sampler':'euler','scheduler':'simple'}
    first=cache.first_pass_cache_fingerprint(segment,plan)
    marker=module._ACTIVE.set((object(),object()))
    try:
        final=cache.segment_cache_fingerprint(segment,plan)
        assert 'raylight_vsr_sources' in final
        assert cache.first_pass_cache_fingerprint(segment,plan)==first
    finally:module._ACTIVE.reset(marker)
