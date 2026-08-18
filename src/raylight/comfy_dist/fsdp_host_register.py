from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Iterable, Iterator

import torch


def _cuda_succeeded(result: Any) -> bool:
    value = getattr(result, "value", result)
    try:
        return int(value) == 0
    except (TypeError, ValueError):
        return False


@dataclass
class HostRegistration:
    """Own CUDA registrations without owning or copying their CPU storage."""

    runtime: Any
    pointers: list[int] = field(default_factory=list)
    storages: list[Any] = field(default_factory=list)
    registered_bytes: int = 0
    registered_storages: int = 0
    skipped_capacity_bytes: int = 0
    failed_storages: int = 0
    unregister_failures: int = 0
    _closed: bool = False

    def close(self) -> bool:
        if self._closed:
            return True
        remaining: list[tuple[int, Any]] = []
        for pointer, storage in reversed(list(zip(self.pointers, self.storages))):
            try:
                result = self.runtime.cudaHostUnregister(pointer)
            except Exception:
                self.unregister_failures += 1
                remaining.append((pointer, storage))
                continue
            if not _cuda_succeeded(result):
                self.unregister_failures += 1
                remaining.append((pointer, storage))
        remaining.reverse()
        self.pointers = [pointer for pointer, _storage in remaining]
        self.storages = [storage for _pointer, storage in remaining]
        self._closed = not remaining
        return self._closed


def _storage_candidate(tensor: torch.Tensor) -> tuple[int, int] | None:
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
        return None
    if tensor.device.type == "meta" or tensor.is_pinned():
        return None
    try:
        storage = tensor.untyped_storage()
        pointer = int(storage.data_ptr())
        size_bytes = int(storage.nbytes())
    except (AttributeError, RuntimeError):
        return None
    if pointer == 0 or size_bytes <= 0:
        return None
    return pointer, size_bytes


def register_cpu_storages(
    tensors: Iterable[torch.Tensor],
    *,
    capacity_bytes: int,
    runtime: Any | None = None,
) -> HostRegistration:
    """Page-lock the largest CPU storages in place without duplicating them."""
    if capacity_bytes < 0:
        raise ValueError("capacity_bytes must be non-negative")
    if runtime is None:
        runtime = torch.cuda.cudart()

    candidates: dict[int, tuple[int, Any]] = {}
    for tensor in tensors:
        candidate = _storage_candidate(tensor)
        if candidate is None:
            continue
        pointer, size_bytes = candidate
        storage = tensor.untyped_storage()
        previous = candidates.get(pointer)
        if previous is None or size_bytes > previous[0]:
            candidates[pointer] = (size_bytes, storage)

    registration = HostRegistration(runtime=runtime)
    for pointer, (size_bytes, storage) in sorted(
        candidates.items(),
        key=lambda item: item[1][0],
        reverse=True,
    ):
        if registration.registered_bytes + size_bytes > capacity_bytes:
            registration.skipped_capacity_bytes += size_bytes
            continue
        try:
            result = runtime.cudaHostRegister(pointer, size_bytes, 0)
        except Exception:
            registration.failed_storages += 1
            continue
        if not _cuda_succeeded(result):
            registration.failed_storages += 1
            continue
        registration.pointers.append(pointer)
        registration.storages.append(storage)
        registration.registered_bytes += size_bytes
        registration.registered_storages += 1
    return registration


def _iter_quantized_backing_tensors(value) -> Iterator[torch.Tensor]:
    qdata = getattr(value, "_qdata", None)
    if isinstance(qdata, torch.Tensor):
        yield qdata
        params = getattr(value, "_params", None)
        if params is not None:
            if is_dataclass(params):
                param_values = (getattr(params, item.name) for item in fields(params))
            else:
                try:
                    param_values = vars(params).values()
                except TypeError:
                    param_values = ()
            for param_value in param_values:
                if isinstance(param_value, torch.Tensor):
                    yield param_value
        return
    if isinstance(value, torch.Tensor):
        yield value


def collect_fsdp_shard_cpu_tensors(
    model,
    *,
    fsdp_module_type,
) -> Iterator[torch.Tensor]:
    """Yield stable CPU shard backing tensors without triggering all-gather staging."""
    seen_groups: set[int] = set()
    for module in model.modules():
        if not isinstance(module, fsdp_module_type):
            continue
        state = module._get_fsdp_state()
        group = getattr(state, "_fsdp_param_group", None)
        if group is None or id(group) in seen_groups:
            continue
        seen_groups.add(id(group))
        for fsdp_param in group.fsdp_params:
            shard_data = getattr(fsdp_param, "_sharded_param_data", None)
            for tensor in _iter_quantized_backing_tensors(shard_data):
                if tensor.device.type == "cpu":
                    yield tensor


__all__ = [
    "HostRegistration",
    "collect_fsdp_shard_cpu_tensors",
    "register_cpu_storages",
]
