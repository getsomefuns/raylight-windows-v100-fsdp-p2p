"""Characterize the installed Director's real first/final fingerprint contract.

Imports only pure Director submodules; never starts Director or reads a cache.
"""
import importlib
import sys
from types import ModuleType

import pytest
import torch

from .path_helpers import comfy_root


@pytest.fixture
def native_cache():
    root = comfy_root() / 'custom_nodes/ComfyUI_MiniMaxH3_Director'
    if not (root / 'director/relay_vsr/pack.py').is_file():
        pytest.skip('This optional Director version has no RelayVSR pack')
    name = '_raylight_test_director_cache_contract'
    namespace = ModuleType(name)
    namespace.__path__ = [str(root)]
    sys.modules[name] = namespace
    try:
        plans = importlib.import_module(name + '.director.plan')
        cache = importlib.import_module(name + '.director.segment_cache')
        segment = plans.SegmentPlan(0, 0, 124, 'Robots walking', 'r2v', 'r2v', False)
        plan = plans.DirectorPlan(24., 124, 864, 480, 864, 'fixed', 864, 480,
                                  'r2v', 'r2v', '', [], [segment], torch.empty(0), 'segment', {})
        yield cache, segment, plan
    finally:
        for key in tuple(sys.modules):
            if key == name or key.startswith(name + '.'):
                del sys.modules[key]


def test_vsr_config_changes_only_final_fingerprint_and_never_decodes(native_cache, monkeypatch):
    cache, segment, plan = native_cache
    def forbidden(*args, **kwargs):
        raise AssertionError('Fingerprint inspection must not decode media or load tensors')
    monkeypatch.setattr(cache, 'decode_frames_ffv1', forbidden)
    monkeypatch.setattr(torch, 'load', forbidden)
    first = cache.first_pass_cache_fingerprint(segment, plan)
    final = cache.segment_cache_fingerprint(segment, plan)
    plan.vsr = {'enabled': True, 'model': object(), 'vae': object(), 'factor_tag': '2:8',
                'steps': 1, 'cfg': 1., 'seed': 42, 'detail': 1., 'denoise': 1., 'sampler': 'euler', 'scheduler': 'simple'}
    assert cache.first_pass_cache_fingerprint(segment, plan) == first
    with_vsr = cache.segment_cache_fingerprint(segment, plan)
    assert with_vsr != final and with_vsr['vsr_factor'] == '2:8'
    for key, value in [('detail', .6), ('seed', 43), ('steps', 2), ('factor_tag', '4:4')]:
        original = plan.vsr[key]
        plan.vsr[key] = value
        assert cache.segment_cache_fingerprint(segment, plan) != with_vsr
        assert cache.first_pass_cache_fingerprint(segment, plan) == first
        plan.vsr[key] = original
    plan.vsr = None
    assert cache.segment_cache_fingerprint(segment, plan) == final
