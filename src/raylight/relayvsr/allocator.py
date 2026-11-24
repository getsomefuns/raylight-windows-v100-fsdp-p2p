"""Temporary native-allocator policy for VSR; H3 and worker settings survive."""
from contextlib import contextmanager
import threading

import torch


_SETTINGS_LOCK=threading.RLock()


def _settings():
    try:
        value=torch.cuda.memory._snapshot()['allocator_settings']
        if not isinstance(value['PYTORCH_CUDA_ALLOC_CONF'],str):
            raise TypeError('Missing live allocator config')
        if not isinstance(value['max_split_size'],int):
            raise TypeError('Missing live split limit')
        if not callable(torch.cuda.memory._set_allocator_settings):
            raise TypeError('Missing allocator setter')
        return value
    except (AttributeError,KeyError,TypeError) as error:
        raise RuntimeError('Unvalidated native CUDA allocator settings contract; cannot safely restore H3 policy') from error


def _without_split_limit(config):
    # Bracket-valued options contain commas; splitting the raw string on ','
    # would corrupt roundup_power2_divisions and change unrelated settings.
    parts=[]
    start=depth=0
    for index,char in enumerate(config):
        if char=='[':depth+=1
        elif char==']':depth-=1
        elif char==',' and depth==0:
            parts.append(config[start:index]);start=index+1
        if depth<0:
            raise RuntimeError('Invalid live CUDA allocator configuration')
    if depth:
        raise RuntimeError('Invalid live CUDA allocator configuration')
    parts.append(config[start:])
    filtered=[part for part in parts if part.partition(':')[0].strip()!='max_split_size_mb']
    if len(filtered)==len(parts):
        raise RuntimeError('Cannot identify the live allocator split-limit setting')
    return ','.join(filtered)


@contextmanager
def native_vsr_allocator():
    """Relax only max_split during serialized VSR work and prove restoration.

    No environment mutation, precision changes, stats reset or CUDA-context
    creation. Native default and non-native allocators are left untouched.
    The private Torch contract is deliberately checked on the pinned runtime.
    """
    with _SETTINGS_LOCK:
        from .admission import assert_runtime_healthy
        assert_runtime_healthy()
        if not torch.cuda.is_initialized() or torch.cuda.memory.get_allocator_backend()!='native':
            yield
            return
        before=_settings()
        if before['max_split_size']<=0:
            yield
            return
        original=before['PYTORCH_CUDA_ALLOC_CONF']
        relaxed=_without_split_limit(original)
        expected=dict(before,PYTORCH_CUDA_ALLOC_CONF=relaxed,max_split_size=-1)
        try:
            torch.cuda.memory._set_allocator_settings(relaxed)
            if _settings()!=expected:
                raise RuntimeError('VSR allocator transition changed unrelated settings')
            yield
        finally:
            try:
                torch.cuda.memory._set_allocator_settings(original)
                if _settings()!=before:
                    raise RuntimeError('Restored CUDA allocator settings do not match H3 policy')
            except BaseException as error:
                from .admission import quarantine_runtime
                quarantine_runtime(error)
                raise RuntimeError('VSR CUDA allocator restoration failed; restart ComfyUI before GPU reuse') from error
