"""Acceptance runs must not mistake cached outputs for actual GPU execution."""
import copy
import importlib.util
from pathlib import Path

import pytest


def campaign():
    path = Path(__file__).resolve().parents[1] / 'scripts/minimax-h3/run_migration_case.py'
    spec = importlib.util.spec_from_file_location('migration_case_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_case_validation_does_not_change_seed_or_source_graph():
    module = campaign()
    prompt = {'5': {'class_type': 'MiniMaxH3Director', 'inputs': {'seed': 42}},
              '7': {'class_type': 'SaveVideo', 'inputs': {'filename_prefix': 'video/production_migration/check'}}}
    before = copy.deepcopy(prompt)
    assert len(module.validate_prompt(prompt, expected_seed=42)) == 64
    assert prompt == before
    with pytest.raises(ValueError, match='seed'):
        module.validate_prompt(prompt, expected_seed=43)
    prompt['7']['inputs']['filename_prefix'] = '../user_output'
    with pytest.raises(ValueError, match='output'):
        module.validate_prompt(prompt, expected_seed=42)


def test_history_gate_rejects_cached_sampler_or_missing_real_operation():
    module = campaign()
    history = {'status': {'status_str': 'success'}}
    with pytest.raises(RuntimeError, match='operation'):
        module.verify_gpu_work(history, '', minimum_operations=1)
    with pytest.raises(RuntimeError, match='failed'):
        module.verify_gpu_work({'status': {'status_str': 'error'}}, '', minimum_operations=0)
    log = '[Raylight] MMH3Director remote sampling completed operation=abc ranks=2\n'
    module.verify_gpu_work(history, log, minimum_operations=1)
    with pytest.raises(RuntimeError, match='operation'):
        module.verify_gpu_work(history, log + log, minimum_operations=2)


def test_resources_report_physical_peak_separately_from_private_commit():
    module = campaign()
    rows = [{'rss': 10, 'private_commit': 30, 'system_ram_used': 40, 'system_commit': 70, 'pagefile_used_estimate': 5},
            {'rss': 20, 'private_commit': 25, 'system_ram_used': 35, 'system_commit': 80, 'pagefile_used_estimate': 7}]
    assert module.resource_peaks(rows) == {'rss': 20, 'private_commit': 30, 'system_ram_used': 40,
                                         'system_commit': 80, 'pagefile_used_estimate': 7}


def test_director_combined_node_is_not_reported_as_preprocessing():
    module = campaign()
    prompt = {'5': {'class_type': 'MiniMaxH3Director'}, '7': {'class_type': 'SaveVideo'}}
    events = [{'prompt_id': 'p', 'node': '5', 'time_ns': 0},
              {'prompt_id': 'p', 'node': '7', 'time_ns': 10_000_000_000},
              {'prompt_id': 'p', 'node': None, 'time_ns': 12_000_000_000}]
    assert module.summarize_nodes(events, prompt, 'p') == {
        '5': {'class_type': 'MiniMaxH3Director', 'seconds': 10.},
        '7': {'class_type': 'SaveVideo', 'seconds': 2.}}
    history = {'outputs': {'8': {'text': ['Segment 1 timing: cond=2.5s sample=12.0s decode=3.5s cache=skipped cleanup=skipped\nSegment 1/1: RelayVSR 1728x960 (5.0s)']}},
               'status': {'messages': [['execution_start', {'timestamp': 1000}],
                                       ['execution_success', {'timestamp': 13000}]]}}
    assert module.director_stages(history) == [{'segment': 1, 'conditioning_s': 2.5, 'sampling_s': 12., 'decode_s': 3.5, 'vsr_s': 5.}]
    assert module.execution_seconds(history) == 12.


def test_standalone_vsr_validation_keeps_sampler_seed_and_rejects_retained_pool():
    module = campaign()
    prompt = {'6': {'class_type': 'KSampler', 'inputs': {'seed': 42}},
              '7': {'class_type': 'RayRelayVSRDecode', 'inputs': {'worker_count': 2, 'keep_workers': False}},
              '9': {'class_type': 'SaveVideo', 'inputs': {'filename_prefix': 'video/production_migration/vsr'}}}
    assert len(module.validate_prompt(prompt, expected_seed=42)) == 64
    prompt['7']['inputs']['keep_workers'] = True
    with pytest.raises(ValueError, match='retain|pool'):
        module.validate_prompt(prompt, expected_seed=42)


def test_standalone_cached_sampler_is_not_a_complete_execution():
    module = campaign()
    prompt = {'6': {'class_type': 'KSampler'}, '7': {'class_type': 'RayRelayVSRDecode'}}
    events = [{'prompt_id': 'p', 'node': '7', 'time_ns': 0}]
    with pytest.raises(RuntimeError, match='cached|execut'):
        module.require_executed_nodes(events, prompt, 'p')
    events.insert(0, {'prompt_id': 'p', 'node': '6', 'time_ns': 0})
    module.require_executed_nodes(events, prompt, 'p')


def test_non_director_gate_requires_a_real_two_gpu_xfuser_route():
    module = campaign()
    prompt = {'7': {'class_type': 'XFuserSamplerCustomAdvanced', 'inputs': {'noise_seed': 42}},
              '13': {'class_type': 'RayInitializer', 'inputs': {'GPU': 2, 'FSDP': True}},
              '9': {'class_type': 'SaveVideo', 'inputs': {'filename_prefix': 'video/production_migration/non-director'}}}
    assert len(module.validate_prompt(prompt, expected_seed=42)) == 64
    prompt['13']['inputs']['GPU'] = 1
    with pytest.raises(ValueError, match='two|2'):
        module.validate_prompt(prompt, expected_seed=42)


def test_preview_probe_counts_message_types_without_storing_image_payload():
    module = campaign()
    assert module.message_kind(b'\x00\x00\x00\x01private image bytes') == 'binary_1'
    assert module.message_kind('{"type":"progress","data":{"value":2}}') == 'progress'
    assert module.message_kind('not JSON') == 'invalid_json'


def test_telemetry_gate_requires_both_gpus_host_health_and_execution_coverage():
    from datetime import datetime
    module=campaign()
    verify=getattr(module,'verify_telemetry',None)
    assert callable(verify),'Missing independent acceptance telemetry gate'
    rows=[{'time_ns':(1700000000+n)*10**9,'rss':1,'private_commit':2,'system_ram_used':3,'system_commit':4,'pagefile_used_estimate':0} for n in (9,10,11,12,13)]
    gpu=[datetime.fromtimestamp(1700000000+n).strftime('%Y/%m/%d %H:%M:%S.000')+f', {rank}, 500, 10, 50' for n in (9,10,11,12,13) for rank in (0,1)]
    verify(rows,gpu,[],execution_start_ns=1700000010*10**9,execution_end_ns=1700000012*10**9,gpu_exit_code=0)
    for host,device,errors,exit_code in [(rows,[],[],0),([],gpu,[],0),(rows,gpu,['monitor failed'],0),(rows,gpu,[],1),(rows,[r for r in gpu if ', 1,' not in r],[],0),(rows[:1],gpu,[],0)]:
        with pytest.raises(RuntimeError,match='telemetry|monitor|coverage|GPU|host'):
            verify(host,device,errors,execution_start_ns=1700000010*10**9,execution_end_ns=1700000012*10**9,gpu_exit_code=exit_code)
