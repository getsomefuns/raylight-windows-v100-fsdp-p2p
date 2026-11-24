#!/usr/bin/env python3
"""Submit an unchanged API workflow to an isolated server and retain evidence.

This is orchestration, not a direct model benchmark. Select input files and
models in the workflow before running. Existing output directories are refused.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import re
from pathlib import Path
import subprocess
import threading
import time
import uuid

import psutil
import websocket


def validate_prompt(prompt, *, expected_seed):
    directors = [n for n in prompt.values() if n.get('class_type') == 'MiniMaxH3Director']
    if directors:
        seeds = [n['inputs'].get('seed') for n in directors]
    else:
        seeds = [n['inputs'].get('seed', n['inputs'].get('noise_seed')) for n in prompt.values()
                 if n.get('class_type') in ('KSampler', 'RandomNoise', 'XFuserKSamplerAdvanced', 'XFuserSamplerCustomAdvanced')]
        decoders = [n['inputs'] for n in prompt.values() if n.get('class_type') == 'RayRelayVSRDecode']
        if any(n.get('worker_count') != 2 or n.get('keep_workers') is not False for n in decoders):
            raise ValueError('Standalone gate requires two workers and no retained decoder pool')
        if not decoders:
            initializers = [n['inputs'] for n in prompt.values() if n.get('class_type') == 'RayInitializer']
            xfuser = any(n.get('class_type') in ('XFuserKSamplerAdvanced', 'XFuserSamplerCustomAdvanced') for n in prompt.values())
            if not xfuser or len(initializers) != 1 or initializers[0].get('GPU') != 2 or not initializers[0].get('FSDP'):
                raise ValueError('Non-Director acceptance requires the supported two-GPU FSDP XFuser route')
    if len(seeds) != 1 or seeds[0] != expected_seed:
        raise ValueError('Expected exactly one primary sampler with the declared fixed seed')
    saves = [n for n in prompt.values() if n.get('class_type') == 'SaveVideo']
    if not saves or any(not n['inputs'].get('filename_prefix', '').startswith('video/production_migration/')
                        or '..' in n['inputs']['filename_prefix'].replace('\\', '/').split('/') for n in saves):
        raise ValueError('Case output must stay under video/production_migration/')
    return hashlib.sha256(json.dumps(prompt, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def verify_gpu_work(history, log, *, minimum_operations, mode='director'):
    if history.get('status', {}).get('status_str') != 'success':
        raise RuntimeError('Workflow failed; inspect history.json')
    if mode == 'director':
        count = len(set(re.findall(r'MMH3Director remote sampling completed operation=([a-fA-F0-9]+) ranks=2', log)))
    elif mode == 'relayvsr':
        count = len(re.findall(r'RelayVSR local Flash completed windows=\d+ workers=2 seconds=', log))
    else:
        # For non-Director XFuser, successful terminal history plus executed
        # sampler-node evidence below is the completion witness.
        return
    if count < minimum_operations:
        raise RuntimeError(f'Only {count} actual GPU operations; expected at least {minimum_operations}')


def require_executed_nodes(events, prompt, prompt_id):
    required = {key for key, node in prompt.items() if node.get('class_type') in
                ('KSampler', 'RayRelayVSRDecode', 'XFuserSamplerCustomAdvanced', 'XFuserKSamplerAdvanced')}
    actual = {str(event.get('node')) for event in events if event.get('prompt_id') == prompt_id}
    missing = required - actual
    if missing:
        raise RuntimeError(f'Sampler/decoder execution missing (cached?): {sorted(missing)}')


def resource_peaks(rows):
    keys = ('rss', 'private_commit', 'system_ram_used', 'system_commit', 'pagefile_used_estimate')
    return {key: max((row[key] for row in rows), default=None) for key in keys}


def summarize_nodes(events, prompt, prompt_id):
    selected = [event for event in events if event.get('prompt_id') == prompt_id and 'node' in event]
    result = {}
    for current, following in zip(selected, selected[1:]):
        key = current['node']
        if key is None:
            continue
        key = str(key)
        row = result.setdefault(key, {'class_type': prompt[key]['class_type'], 'seconds': 0.})
        row['seconds'] += max(0, following['time_ns'] - current['time_ns']) / 1e9
    # A Director node includes conditioning, sampling, decoding, caches and VSR.
    # Do NOT label this combined interval as preprocessing or VAE time.
    return result


def director_stages(history):
    texts = []
    for output in history.get('outputs', {}).values():
        text = output.get('text', [])
        texts.extend(text if isinstance(text, list) else [text])
    text = '\n'.join(str(value) for value in texts)
    rows = {}
    for match in re.finditer(r'Segment (\d+) timing: cond=([\d.]+)s sample=([\d.]+)s decode=([\d.]+)s', text):
        segment = int(match[1])
        rows[segment] = dict(segment=segment, conditioning_s=float(match[2]),
                             sampling_s=float(match[3]), decode_s=float(match[4]))
    for match in re.finditer(r'Segment (\d+)(?:/\d+)?: RelayVSR \d+x\d+ \(([\d.]+)s\)', text):
        segment = int(match[1])
        rows.setdefault(segment, {'segment': segment})['vsr_s'] = float(match[2])
    return [rows[key] for key in sorted(rows)]


def execution_seconds(history):
    stamps = {kind: data['timestamp'] for kind, data in history.get('status', {}).get('messages', [])
              if kind in ('execution_start', 'execution_success') and 'timestamp' in data}
    if len(stamps) != 2:
        return None
    return (stamps['execution_success'] - stamps['execution_start']) / 1000.


def message_kind(message):
    if isinstance(message, bytes):
        return f'binary_{int.from_bytes(message[:4], "big")}' if len(message) >= 4 else 'binary_short'
    try:
        value = json.loads(message)
    except (ValueError, TypeError):
        return 'invalid_json'
    return str(value.get('type', 'json_other')) if isinstance(value, dict) else 'json_other'


class PerformanceInfo(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong)] + [(name, ctypes.c_size_t) for name in
        ('CommitTotal', 'CommitLimit', 'CommitPeak', 'PhysicalTotal', 'PhysicalAvailable',
         'SystemCache', 'KernelTotal', 'KernelPaged', 'KernelNonpaged', 'PageSize')] + [
        ('HandleCount', ctypes.c_ulong), ('ProcessCount', ctypes.c_ulong), ('ThreadCount', ctypes.c_ulong)]


def sample_resources(server_pid, stop, path):
    with path.open('w', encoding='utf-8') as output:
        while not stop.is_set():
            try:
                parent = psutil.Process(server_pid)
                members = [parent] + parent.children(recursive=True)
                snapshots = []
                for member in members:
                    try:
                        snapshots.append((member.pid, member.memory_info()))
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                vm = psutil.virtual_memory()
                perf = PerformanceInfo()
                perf.cb = ctypes.sizeof(perf)
                if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(perf), perf.cb):
                    raise ctypes.WinError()
                row = {'time_ns': time.time_ns(), 'pids': [pid for pid, _ in snapshots],
                       'rss': sum(info.rss for _, info in snapshots),
                       'private_commit': sum(getattr(info, 'private', 0) for _, info in snapshots),
                       'system_ram_used': vm.total - vm.available,
                       'system_commit': perf.CommitTotal * perf.PageSize,
                       'pagefile_used_estimate': psutil.swap_memory().used}
                output.write(json.dumps(row) + '\n')
                output.flush()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            stop.wait(.5)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prompt', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--server-pid', type=int, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8199)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--minimum-operations', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=7200)
    args = parser.parse_args(argv)
    prompt = json.loads(args.prompt.read_text(encoding='utf-8-sig'))
    digest = validate_prompt(prompt, expected_seed=args.seed)
    types = {node['class_type'] for node in prompt.values()}
    mode = 'director' if 'MiniMaxH3Director' in types else ('relayvsr' if 'RayRelayVSRDecode' in types else 'xfuser')
    if args.minimum_operations < 0:
        raise ValueError('minimum-operations cannot be negative')
    spec = importlib.util.spec_from_file_location('migration_http_helpers', Path(__file__).with_name('benchmark_cold_warm.py'))
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    url = f'http://127.0.0.1:{args.port}'
    queue = base.request_json(url + '/queue')
    if queue.get('queue_running') or queue.get('queue_pending'):
        raise RuntimeError('Server queue is not idle')
    args.output.mkdir(parents=True, exist_ok=False)
    def dump(name, value):
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    dump('prompt.json', prompt)
    start_offset = args.log.stat().st_size
    stop = threading.Event()
    monitor = threading.Thread(target=sample_resources, args=(args.server_pid, stop, args.output / 'resources.jsonl'))
    client = uuid.uuid4().hex
    connection = websocket.create_connection(f'ws://127.0.0.1:{args.port}/ws?clientId={client}', suppress_origin=True, timeout=1)
    counts = {}
    original_recv = connection.recv
    def counted_recv(*positional, **keywords):
        message = original_recv(*positional, **keywords)
        kind = message_kind(message)
        counts[kind] = counts.get(kind, 0) + 1
        return message
    connection.recv = counted_recv
    events, errors = [], []
    receiver = threading.Thread(target=base.receive_node_events, args=(connection, stop, events, errors))
    started = time.perf_counter()
    monitor.start()
    receiver.start()
    try:
        with (args.output / 'gpu.csv').open('w', encoding='utf-8') as gpu:
            smi = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw',
                                    '--format=csv,noheader,nounits', '-lms', '500'], stdout=gpu,
                                   stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                response = base.request_json(url + '/prompt', {'prompt': prompt, 'client_id': client}, timeout=120)
                dump('submission.json', response)
                prompt_id = response['prompt_id']
                print(json.dumps({'submitted': prompt_id, 'case': args.output.name}), flush=True)
                deadline = time.monotonic() + args.timeout
                while time.monotonic() < deadline:
                    history = base.request_json(url + '/history/' + prompt_id, timeout=60)
                    if prompt_id in history:
                        record = history[prompt_id]
                        dump('history.json', record)
                        observed_seconds = time.perf_counter() - started
                        # Allow Ray's asynchronous final log forwarding to settle.
                        stop.wait(2)
                        with args.log.open('rb') as handle:
                            handle.seek(start_offset)
                            log = handle.read().decode('utf-8', errors='replace')
                        (args.output / 'server.log').write_text(log, encoding='utf-8')
                        dump('result.json', {'observed_seconds': observed_seconds, 'execution_seconds': execution_seconds(record),
                            'prompt_id': prompt_id, 'director_stages': director_stages(record),
                            'prompt_sha256': digest, 'node_timings': summarize_nodes(events, prompt, prompt_id),
                            'status': record.get('status'), 'minimum_gpu_operations': args.minimum_operations})
                        verify_gpu_work(record, log, minimum_operations=args.minimum_operations, mode=mode)
                        require_executed_nodes(events, prompt, prompt_id)
                        return 0
                    stop.wait(2)
                raise TimeoutError('Case timeout; server is NOT automatically interrupted')
            finally:
                smi.terminate()
                smi.wait(timeout=15)
    finally:
        stop.set()
        connection.close()
        receiver.join(timeout=15)
        monitor.join(timeout=15)
        dump('events.json', {'events': events, 'errors': errors})
        dump('socket-message-counts.json', counts)
        path = args.output / 'resources.jsonl'
        if path.exists():
            dump('resource-peaks.json', resource_peaks([json.loads(line) for line in path.read_text().splitlines()]))


if __name__ == '__main__':
    raise SystemExit(main())
