import types
from dataclasses import dataclass

import torch
import pytest

from raylight.comfy_dist.fsdp_host_register import (
    collect_fsdp_shard_cpu_tensors,
    register_cpu_storages,
)


class _FakeCudaRuntime:
    def __init__(self, *, fail_pointers=()):
        self.fail_pointers = set(fail_pointers)
        self.registered = []
        self.unregistered = []

    def cudaHostRegister(self, pointer, size_bytes, flags):
        self.registered.append((pointer, size_bytes, flags))
        return 1 if pointer in self.fail_pointers else 0

    def cudaHostUnregister(self, pointer):
        self.unregistered.append(pointer)
        return 0


def test_register_cpu_storages_deduplicates_shared_storage_and_closes_once():
    base = torch.empty(1024, dtype=torch.uint8)
    runtime = _FakeCudaRuntime()

    registration = register_cpu_storages(
        [base[:512], base[512:]],
        capacity_bytes=base.untyped_storage().nbytes(),
        runtime=runtime,
    )

    storage_pointer = base.untyped_storage().data_ptr()
    assert runtime.registered == [(storage_pointer, 1024, 0)]
    assert registration.registered_bytes == 1024
    assert registration.registered_storages == 1
    assert len(registration.storages) == 1
    assert registration.storages[0].data_ptr() == storage_pointer

    registration.close()
    registration.close()
    assert runtime.unregistered == [storage_pointer]


def test_register_cpu_storages_prefers_largest_and_respects_capacity():
    small = torch.empty(128, dtype=torch.uint8)
    large = torch.empty(512, dtype=torch.uint8)
    medium = torch.empty(256, dtype=torch.uint8)
    runtime = _FakeCudaRuntime()

    registration = register_cpu_storages(
        [small, large, medium],
        capacity_bytes=640,
        runtime=runtime,
    )

    registered_sizes = [size for _pointer, size, _flags in runtime.registered]
    assert registered_sizes == [512, 128]
    assert registration.registered_bytes == 640
    assert registration.skipped_capacity_bytes == 256


def test_register_cpu_storages_keeps_successes_when_one_registration_fails():
    first = torch.empty(512, dtype=torch.uint8)
    second = torch.empty(256, dtype=torch.uint8)
    failed_pointer = first.untyped_storage().data_ptr()
    runtime = _FakeCudaRuntime(fail_pointers={failed_pointer})

    registration = register_cpu_storages(
        [first, second],
        capacity_bytes=1024,
        runtime=runtime,
    )

    assert registration.failed_storages == 1
    assert registration.registered_storages == 1
    registration.close()
    assert runtime.unregistered == [second.untyped_storage().data_ptr()]


@pytest.mark.parametrize("first_failure", [1, RuntimeError("unregister failed")])
def test_close_retains_storage_until_cuda_unregister_succeeds(first_failure):
    tensor = torch.empty(256, dtype=torch.uint8)

    class RetryRuntime(_FakeCudaRuntime):
        def __init__(self):
            super().__init__()
            self.outcomes = [first_failure, 0]

        def cudaHostUnregister(self, pointer):
            self.unregistered.append(pointer)
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    runtime = RetryRuntime()
    registration = register_cpu_storages(
        [tensor],
        capacity_bytes=tensor.untyped_storage().nbytes(),
        runtime=runtime,
    )
    pointer = tensor.untyped_storage().data_ptr()

    registration.close()
    assert registration.pointers == [pointer]
    assert len(registration.storages) == 1
    assert registration.storages[0].data_ptr() == pointer

    registration.close()
    assert registration.pointers == []
    assert registration.storages == []
    assert runtime.unregistered == [pointer, pointer]


def test_collect_fsdp_shards_never_accesses_dynamic_all_gather_inputs_property():
    first = torch.empty(8)
    second = torch.empty(4)

    class FakeParam:
        def __init__(self, shard):
            self._sharded_param_data = shard

        @property
        def all_gather_inputs(self):
            raise AssertionError("dynamic all_gather_inputs must not be accessed")

    group = types.SimpleNamespace(
        fsdp_params=[
            FakeParam(first),
            FakeParam(second),
        ]
    )

    class FakeFSDPModule:
        def __init__(self, fsdp_group):
            self.state = types.SimpleNamespace(_fsdp_param_group=fsdp_group)

        def _get_fsdp_state(self):
            return self.state

    modules = [FakeFSDPModule(group), FakeFSDPModule(group), object()]
    model = types.SimpleNamespace(modules=lambda: iter(modules))

    inputs = list(
        collect_fsdp_shard_cpu_tensors(
            model,
            fsdp_module_type=FakeFSDPModule,
        )
    )

    assert inputs == [first, second]


def test_collect_fsdp_shards_extracts_quantized_qdata_and_scale_backing():
    qdata = torch.empty(16, dtype=torch.uint8)
    scale = torch.ones(1)
    @dataclass(slots=True)
    class Params:
        scale: torch.Tensor

    quantized = types.SimpleNamespace(
        _qdata=qdata,
        _params=Params(scale=scale),
    )
    group = types.SimpleNamespace(
        fsdp_params=[types.SimpleNamespace(_sharded_param_data=quantized)]
    )

    class FakeFSDPModule:
        def _get_fsdp_state(self):
            return types.SimpleNamespace(_fsdp_param_group=group)

    model = types.SimpleNamespace(modules=lambda: iter([FakeFSDPModule()]))

    inputs = list(
        collect_fsdp_shard_cpu_tensors(
            model,
            fsdp_module_type=FakeFSDPModule,
        )
    )

    assert inputs == [qdata, scale]
