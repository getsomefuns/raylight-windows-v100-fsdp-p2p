"""TE's real residual controller inside Raylight's mask-aware USP block loop.

Never installs the third party's whole-model forward replacement. Its cache
proposal is deferred until both ranks agree; a rejected proposal must not have
already modified this rank's input tensor.
"""
from __future__ import annotations

import functools
import importlib
import inspect
import math
from types import SimpleNamespace

from .proxy import RayMMH3DirectorModelProxy


CONFIG_KEY = 'raylight_external_te_speed'
RUNTIME_KEY = 'te_speed_minimax_h3_runtime'
_MARKER = '_raylight_te_proxy_patch'


def validate_config(config):
    result = dict(config)
    for key in ('processing_control_value', 'processing_percent_1', 'processing_percent_2'):
        value = float(result[key])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'Invalid Raylight TE cache {key}')
        result[key] = value
    if isinstance(result['mcs'], bool) or not isinstance(result['mcs'], int) or not 1 <= result['mcs'] <= 10:
        raise ValueError('Invalid Raylight TE cache mcs')
    if result['device'] not in ('auto', 'cpu', 'gpu'):
        raise ValueError('Invalid Raylight TE cache device')
    if result['mode'] not in ('standard', '4-step LoRA', '8-step LoRA'):
        raise ValueError('Invalid Raylight TE cache mode')
    return result


def make_proxy_te_patch(original):
    signature = inspect.signature(original)
    @functools.wraps(original)
    def patch(*args, **kwargs):
        values = signature.bind(*args, **kwargs)
        values.apply_defaults()
        model = values.arguments.get('model')
        if not isinstance(model, RayMMH3DirectorModelProxy):
            return original(*args, **kwargs)
        config = validate_config({key: values.arguments[key] for key in
            ('processing_control_value', 'processing_percent_1', 'processing_percent_2', 'mcs', 'device', 'mode')})
        result = model.clone()
        options = result.model_options.setdefault('transformer_options', {})
        options.pop(RUNTIME_KEY, None)
        options[CONFIG_KEY] = config
        # Do not retain a previous native TE outer-sample forward substitution.
        result.wrappers.get('outer_sample', {}).pop('te_speed_minimax_h3', None)
        return (result,)
    setattr(patch, _MARKER, True)
    return patch


def install_te_proxy_bridge():
    import nodes
    cls = nodes.NODE_CLASS_MAPPINGS.get('TESpeedMiniMaxH3')
    if cls is None or getattr(cls.patch, _MARKER, False):
        return False
    cls.patch = make_proxy_te_patch(cls.patch)
    return True


class OwnedTECache:
    def __init__(self, native):
        self.native = native
        self._raylight_stats = dict(full_forwards=0, cached_forwards=0,
                                    consensus_forced_full=0, skipped_blocks=0)

    def __getattr__(self, name):
        native = vars(self).get('native')
        if native is None:
            raise AttributeError(name)
        return getattr(native, name)

    def __deepcopy__(self, memo):
        native = self.native
        fresh = type(native)(native.threshold, native.start_percent, native.end_percent,
            native.max_consecutive_skips, native.cache_device, native.block_count,
            turbo_variant=native.turbo_variant)
        result = type(self)(fresh)
        memo[id(self)] = result
        return result

    def finish(self):
        try:
            self.native.finish()
        finally:
            self.native.states.clear()


def create_runtime(config, block_count, *, api=None):
    config = validate_config(config)
    if api is None:
        from .external_acceleration import _find_node_package, _load_package
        package = _load_package('_raylight_external_te_speed', _find_node_package('TE-Speed-MiniMaxH3'))
        api = importlib.import_module(package.__name__ + '.nodes')
    expected = ('self', 'hidden', 'transformer_options', 'force_full')
    if tuple(inspect.signature(api._MiniMaxH3Cache.prepare_middle).parameters) != expected:
        raise RuntimeError('Unsupported TE cache-controller contract (validated with TE 3.3)')
    variant = {'standard': None, '4-step LoRA': 'four', '8-step LoRA': 'eight'}[config['mode']]
    native = api._MiniMaxH3CacheGroup(config['processing_control_value'], config['processing_percent_1'],
        config['processing_percent_2'], config['mcs'], config['device'], int(block_count), turbo_variant=variant)
    return OwnedTECache(native)


def local_cache_layout(layout, rank, local_rows):
    start, stop = rank * local_rows, (rank + 1) * local_rows
    segments = [(max(a, start) - start, min(b, stop) - start, kind)
                for a, b, kind in layout.segments if max(a, start) < min(b, stop)]
    return SimpleNamespace(segments=segments, seq_len=local_rows)


def run_cached_blocks(runtime, hidden, layout, options, run_range, agree, *, rank, local_rows,
                      block_count, force_full=False):
    state = runtime.state_for(options)
    state.set_cache_ranges(local_cache_layout(layout, rank, local_rows))
    start, stop = int(state.skip_start), int(state.skip_end)
    if not 0 <= start <= stop <= block_count:
        raise RuntimeError('Invalid TE cache block range')
    hidden = run_range(hidden, range(start))
    deferred = []
    original = state._apply_cached
    previous = vars(state).get('_apply_cached')
    def capture(*args, **kwargs):
        deferred.append((args, kwargs))
    state._apply_cached = capture
    try:
        proposed = bool(state.prepare_middle(hidden, options, force_full=force_full))
    finally:
        if previous is None:
            del state._apply_cached
        else:
            state._apply_cached = previous
    if proposed != bool(deferred) or len(deferred) > 1:
        raise RuntimeError('Unsupported TE deferred residual-application contract')
    approved = bool(agree(proposed))
    if approved and not proposed:
        raise RuntimeError('TE consensus cannot approve a missing local residual')
    if approved:
        args, kwargs = deferred.pop()
        original(*args, **kwargs)
        del args, kwargs
        runtime._raylight_stats['cached_forwards'] += 1
        runtime._raylight_stats['skipped_blocks'] += stop - start
    else:
        deferred.clear()
        if proposed:
            # prepare_middle's deferred proposal left hidden unchanged. A full
            # retry at the same sigma resets its full-run base without cloning h.
            if state.prepare_middle(hidden, options, force_full=True):
                raise RuntimeError('TE refused a required full middle run')
            runtime._raylight_stats['consensus_forced_full'] += 1
        hidden = run_range(hidden, range(start, stop))
        state.store_middle(hidden)
        runtime._raylight_stats['full_forwards'] += 1
    return run_range(hidden, range(stop, block_count))


def finish_owned_runtime(model):
    options = getattr(model, 'model_options', {}).get('transformer_options', {})
    runtime = options.get(RUNTIME_KEY)
    if not isinstance(runtime, OwnedTECache):
        return None
    stats = dict(runtime._raylight_stats)
    try:
        runtime.finish()
    finally:
        options.pop(RUNTIME_KEY, None)
    return stats
