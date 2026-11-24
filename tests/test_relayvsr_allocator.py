"""VSR may relax split policy, never permanently change H3's allocator."""
import copy
import importlib
import re
import threading

import pytest
import torch


def allocator_module():
    try:
        return importlib.import_module('raylight.relayvsr.allocator')
    except ModuleNotFoundError:
        pytest.fail('Scoped VSR allocator policy is missing')


class AllocatorState:
    def __init__(self, config):
        self.config=config
        self.calls=[]
        self.fail_restore=False
        self.original=config

    def snapshot(self):
        limited=re.search(r'max_split_size_mb\s*:\s*(\d+)',self.config)
        return {'allocator_settings':{
            'PYTORCH_CUDA_ALLOC_CONF':self.config,
            'max_split_size':int(limited[1])*2**20 if limited else -1,
            'garbage_collection_threshold':.8,
            'roundup_power2_divisions':{'256':1,'512':2},
        }}

    def set(self, config):
        if self.fail_restore and config==self.original:
            raise RuntimeError('controlled restore failure')
        self.calls.append(config)
        self.config=config


@pytest.fixture
def state(monkeypatch):
    value=AllocatorState('max_split_size_mb:128,garbage_collection_threshold:0.8,roundup_power2_divisions:[256:1,512:2]')
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:True)
    monkeypatch.setattr(torch.cuda.memory,'get_allocator_backend',lambda:'native')
    monkeypatch.setattr(torch.cuda.memory,'_snapshot',value.snapshot)
    monkeypatch.setattr(torch.cuda.memory,'_set_allocator_settings',value.set)
    from raylight.relayvsr import admission
    monkeypatch.setattr(admission,'_QUARANTINED',[])
    return value


def test_scope_relaxes_only_max_split_then_restores_live_settings(state,monkeypatch):
    module=allocator_module()
    before=copy.deepcopy(state.snapshot())
    monkeypatch.setenv('PYTORCH_CUDA_ALLOC_CONF','max_split_size_mb:512')
    with module.native_vsr_allocator():
        assert state.snapshot()['allocator_settings']['max_split_size']==-1
        assert state.config=='garbage_collection_threshold:0.8,roundup_power2_divisions:[256:1,512:2]'
    assert state.snapshot()==before


def test_body_failure_still_restores_h3_policy(state):
    module=allocator_module()
    with pytest.raises(ValueError,match='sample failed'):
        with module.native_vsr_allocator():
            raise ValueError('sample failed')
    assert state.config==state.original


def test_nested_director_and_decoder_scopes_do_not_restore_too_early(state):
    module=allocator_module()
    with module.native_vsr_allocator():
        with module.native_vsr_allocator():
            assert 'max_split_size_mb' not in state.config
        assert 'max_split_size_mb' not in state.config
    assert state.config==state.original
    assert len(state.calls)==2


def test_already_unlimited_allocator_is_not_changed(state):
    module=allocator_module()
    state.config='garbage_collection_threshold:0.8'
    with module.native_vsr_allocator():
        assert state.config=='garbage_collection_threshold:0.8'
    assert not state.calls


def test_no_cuda_context_is_created_for_cpu_only_calls(state,monkeypatch):
    module=allocator_module()
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:False)
    monkeypatch.setattr(torch.cuda.memory,'_snapshot',lambda:pytest.fail('CPU call initialized CUDA'))
    with module.native_vsr_allocator():pass
    assert not state.calls


def test_unknown_snapshot_contract_refuses_before_modifying_allocator(state,monkeypatch):
    module=allocator_module()
    monkeypatch.setattr(torch.cuda.memory,'_snapshot',lambda:{'allocator_settings':{'max_split_size':128*2**20}})
    with pytest.raises(RuntimeError,match='allocator|contract|settings'):
        with module.native_vsr_allocator():pytest.fail('Unverified restoration admitted GPU work')
    assert not state.calls


def test_failed_restore_quarantines_further_gpu_reuse(state):
    module=allocator_module()
    with pytest.raises(RuntimeError,match='restor|restart'):
        with module.native_vsr_allocator():
            state.fail_restore=True
    from raylight.relayvsr.admission import assert_runtime_healthy
    with pytest.raises(RuntimeError,match='restart|quarantin'):
        assert_runtime_healthy()


def test_non_native_allocator_is_left_untouched(state,monkeypatch):
    module=allocator_module()
    monkeypatch.setattr(torch.cuda.memory,'get_allocator_backend',lambda:'cudaMallocAsync')
    with module.native_vsr_allocator():pass
    assert not state.calls


def test_restore_quarantine_cleans_but_does_not_return_gpu_leases(state):
    from tests.test_relayvsr_admission import Worker,authorization
    from raylight.text_encoder_parallel.lifecycle import LEASED_STATE
    module=allocator_module()
    events=[]
    workers=[Worker(0,events),Worker(1,events)]
    with pytest.raises(RuntimeError,match='restor|restart'):
        with authorization(workers,lambda:events.append('cleanup')):
            with module.native_vsr_allocator():
                state.fail_restore=True
    assert 'cleanup' in events
    assert not any(event.startswith('release') for event in events)
    assert all(worker.lifecycle.snapshot()['state']==LEASED_STATE for worker in workers)


def test_waiting_scope_rechecks_health_after_another_scope_fails(state):
    from raylight.relayvsr.admission import assert_runtime_healthy
    module=allocator_module()
    admitted=threading.Event()
    bodies=[]
    errors=[]
    def waiter():
        assert_runtime_healthy()  # Passed before the earlier operation fails.
        admitted.set()
        try:
            with module.native_vsr_allocator():bodies.append('unsafe GPU work')
        except RuntimeError as error:errors.append(str(error))
    thread=threading.Thread(target=waiter)
    with pytest.raises(RuntimeError,match='restor|restart'):
        with module.native_vsr_allocator():
            thread.start()
            assert admitted.wait(3)
            state.fail_restore=True
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert not bodies
    assert errors and ('restart' in errors[0] or 'quarantin' in errors[0])
