"""Scoped admission for the Director driver's existing two local GPU workers."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import logging
import os
from types import SimpleNamespace
import uuid

from raylight.text_encoder_parallel.lifecycle import DualGpuLeaseCoordinator, LEASED_STATE


_ACTIVE = ContextVar("raylight_relayvsr_authorization", default=None)
_QUARANTINED = []
_LOG = logging.getLogger(__name__)


def identity_string(value):
    convert = getattr(value, "hex", None)
    return convert() if callable(convert) else str(value)


def visible_gpu_uuids():
    import torch
    if torch.cuda.device_count() != 2:
        raise RuntimeError("Director RelayVSR requires exactly two visible local GPUs")
    result = tuple(str(torch.cuda.get_device_properties(i).uuid) for i in (0, 1))
    if not all(result) or len(set(result)) != 2:
        raise RuntimeError("Cannot establish distinct physical GPU identities")
    return result


def _runtime_identity(ray_module):
    if not ray_module.is_initialized():
        raise RuntimeError("The owning Ray driver is no longer connected")
    context = ray_module.get_runtime_context()
    return os.getpid(), identity_string(context.get_job_id()), identity_string(context.get_node_id())


def _worker_identities(ray_module, workers):
    return ray_module.get([worker.get_relay_vsr_identity.remote() for worker in workers], timeout=15)


@dataclass
class FlashAuthorization:
    ray: object
    workers: tuple
    identities: list
    runtime: tuple
    operation: str
    owner: str
    active: bool = True

    def validate(self):
        if not self.active or _runtime_identity(self.ray) != self.runtime:
            raise RuntimeError("RelayVSR authorization belongs to another driver/session")
        if _worker_identities(self.ray, self.workers) != self.identities:
            raise RuntimeError("RelayVSR worker generation/device identity changed")
        states = self.ray.get([worker.get_external_gpu_lease.remote() for worker in self.workers], timeout=15)
        if len(states) != 2:
            raise RuntimeError("Incomplete RelayVSR GPU lease state")
        for rank, state in enumerate(states):
            lease = state.get("lease") or {}
            if (state.get("rank") != rank or state.get("state") != LEASED_STATE
                    or lease.get("operation_id") != self.operation or lease.get("owner") != self.owner):
                raise RuntimeError("RelayVSR no longer owns both GPU leases")


def current_flash_authorization():
    token = _ACTIVE.get()
    if token is None or not token.active:
        return None
    token.validate()
    return token


@contextmanager
def authorize_flash(workers, *, cleanup, ray_module=None, device_uuids=None):
    if ray_module is None:
        import ray as ray_module
    if _ACTIVE.get() is not None:
        raise RuntimeError("A Director RelayVSR authorization is already active")
    workers = tuple(workers)
    if len(workers) != 2:
        raise RuntimeError("Director RelayVSR needs both owning H3 workers")
    runtime = _runtime_identity(ray_module)
    expected = visible_gpu_uuids() if device_uuids is None else tuple(device_uuids)
    identities = _worker_identities(ray_module, workers)
    if len(identities) != 2 or len(set(expected)) != 2:
        raise RuntimeError("Incomplete RelayVSR worker/device identity")
    for rank, (worker, row) in enumerate(zip(workers, identities)):
        if (row.get("rank") != rank or row.get("job_id") != runtime[1]
                or row.get("node_id") != runtime[2] or not row.get("generation")
                or row.get("actor_id") != identity_string(worker._actor_id)):
            raise RuntimeError("Foreign or stale RelayVSR worker identity")
    if {row.get("gpu_uuid") for row in identities} != set(expected):
        raise RuntimeError("RelayVSR worker physical GPUs do not match this driver")
    # Only the strict VSR-specific cleanup entry differs from the mature TP2
    # all-or-nothing coordinator; the original worker/actor remains alive.
    endpoints = [SimpleNamespace(enter_external_gpu_lease=w.enter_relay_vsr_gpu_lease,
                                exit_external_gpu_lease=w.exit_external_gpu_lease) for w in workers]
    coordinator = DualGpuLeaseCoordinator(endpoints, ray_module=ray_module)
    operation, owner = uuid.uuid4().hex, "raylight_relayvsr"
    coordinator.acquire(operation, owner)
    token = FlashAuthorization(ray_module, workers, identities, runtime, operation, owner)
    marker = _ACTIVE.set(token)
    try:
        token.validate()
        yield token
    finally:
        try:
            # Never return the GPUs to H3 while Flash may still own weights or
            # in-flight work. A failed cleanup is deliberately quarantined.
            cleanup()
            coordinator.release(operation, owner)
        except BaseException:
            _QUARANTINED.append((token, coordinator))
            _LOG.exception("RelayVSR GPU cleanup/release failed; workers remain blocked. Restart this runtime before reuse.")
            raise
        finally:
            token.active = False
            _ACTIVE.reset(marker)
