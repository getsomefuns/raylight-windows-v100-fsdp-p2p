"""All-or-nothing worker GPU leases for the main-process TP2 text encoder."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import threading
from typing import Any, Callable, Mapping, Sequence


LEASED_STATE = "leased_for_text_encoder"
IDLE_STATE = "idle"
ACTIVITY_STATES = frozenset(("loading_diffusion", "loading_lora", "sampling"))


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class ExternalGpuLease:
    operation_id: str
    owner: str
    free_bytes: int
    worker_state: str
    rank: int

    def __post_init__(self) -> None:
        _identifier(self.operation_id, "operation_id")
        _identifier(self.owner, "owner")
        if isinstance(self.free_bytes, bool) or not isinstance(self.free_bytes, int) or self.free_bytes < 0:
            raise ValueError("free_bytes must be a non-negative integer")
        if self.worker_state != LEASED_STATE:
            raise ValueError(f"worker_state must be {LEASED_STATE!r}")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 0:
            raise ValueError("rank must be a non-negative integer")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExternalGpuLease":
        if not isinstance(value, Mapping):
            raise ValueError("worker lease response must be a mapping")
        return cls(
            operation_id=value.get("operation_id"),
            owner=value.get("owner"),
            free_bytes=value.get("free_bytes"),
            worker_state=value.get("worker_state"),
            rank=value.get("rank"),
        )

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


class WorkerGpuLifecycle:
    """Synchronous worker-side state machine independent of Ray internals."""

    def __init__(self, *, rank: int) -> None:
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ValueError("rank must be a non-negative integer")
        self.rank = rank
        self._lock = threading.RLock()
        self._state = IDLE_STATE
        self._lease: ExternalGpuLease | None = None
        self._activity_thread: int | None = None
        self._activity_stack: list[str] = []
        self._last_release: tuple[str, str] | None = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "rank": self.rank,
                "state": self._state,
                "lease": None if self._lease is None else self._lease.to_mapping(),
            }

    @contextmanager
    def activity(self, state: str):
        if state not in ACTIVITY_STATES:
            raise ValueError(f"unsupported worker activity: {state!r}")
        thread_id = threading.get_ident()
        with self._lock:
            if self._lease is not None:
                raise RuntimeError(
                    f"worker rank {self.rank} GPU is leased by {self._lease.owner} "
                    f"({self._lease.operation_id}); cannot enter {state}"
                )
            if self._state != IDLE_STATE and self._activity_thread != thread_id:
                raise RuntimeError(f"worker rank {self.rank} is already {self._state}")
            self._activity_stack.append(self._state)
            self._activity_thread = thread_id
            self._state = state
        try:
            yield
        finally:
            with self._lock:
                previous = self._activity_stack.pop() if self._activity_stack else IDLE_STATE
                self._state = previous
                if not self._activity_stack:
                    self._activity_thread = None

    def acquire_external_lease(
        self,
        operation_id: str,
        owner: str,
        *,
        cleanup: Callable[[], Any],
        free_bytes: Callable[[], int],
    ) -> ExternalGpuLease:
        operation_id = _identifier(operation_id, "operation_id")
        owner = _identifier(owner, "owner")
        with self._lock:
            if self._state != IDLE_STATE or self._lease is not None:
                if self._lease is not None:
                    detail = f"leased by {self._lease.owner} ({self._lease.operation_id})"
                else:
                    detail = self._state
                raise RuntimeError(f"worker rank {self.rank} is {detail}; external GPU lease refused")
            pending = ExternalGpuLease(operation_id, owner, 0, LEASED_STATE, self.rank)
            self._lease = pending
            self._state = LEASED_STATE
            self._last_release = None
        try:
            cleanup()
            available = free_bytes()
            if isinstance(available, bool) or not isinstance(available, int) or available < 0:
                raise ValueError("free_bytes callback must return a non-negative integer")
            lease = replace(pending, free_bytes=available)
            with self._lock:
                self._lease = lease
            return lease
        except BaseException:
            with self._lock:
                self._lease = None
                self._state = IDLE_STATE
            raise

    def release_external_lease(self, operation_id: str, owner: str) -> dict[str, Any]:
        operation_id = _identifier(operation_id, "operation_id")
        owner = _identifier(owner, "owner")
        identity = operation_id, owner
        with self._lock:
            if self._lease is None:
                if self._last_release == identity:
                    return {"rank": self.rank, "state": IDLE_STATE, "released": False}
                raise RuntimeError(
                    f"worker rank {self.rank} has no lease owned by {owner} ({operation_id})"
                )
            if (self._lease.operation_id, self._lease.owner) != identity:
                raise RuntimeError(
                    f"{owner} ({operation_id}) does not own worker rank {self.rank} lease; "
                    f"current owner is {self._lease.owner} ({self._lease.operation_id})"
                )
            self._lease = None
            self._state = IDLE_STATE
            self._last_release = identity
            return {"rank": self.rank, "state": IDLE_STATE, "released": True}


class DualGpuLeaseCoordinator:
    """Acquire or roll back the same external lease on both Ray workers."""

    def __init__(self, workers: Sequence[Any], *, ray_module=None) -> None:
        if len(workers) != 2:
            raise ValueError("MiniMax H3 TP2 requires exactly two Ray workers")
        if ray_module is None:
            import ray as ray_module

        self.workers = tuple(workers)
        self.ray = ray_module
        self._active: tuple[str, str] | None = None

    def _release_remote(self, operation_id: str, owner: str):
        refs = [
            worker.exit_external_gpu_lease.remote(operation_id, owner)
            for worker in self.workers
        ]
        return self.ray.get(refs)

    def acquire(self, operation_id: str, owner: str) -> tuple[ExternalGpuLease, ExternalGpuLease]:
        operation_id = _identifier(operation_id, "operation_id")
        owner = _identifier(owner, "owner")
        if self._active is not None:
            raise RuntimeError(f"coordinator already owns active lease {self._active}")
        refs = [
            worker.enter_external_gpu_lease.remote(operation_id, owner)
            for worker in self.workers
        ]
        try:
            responses = self.ray.get(refs)
            if not isinstance(responses, (list, tuple)) or len(responses) != 2:
                raise RuntimeError("expected one external GPU lease response from each worker")
            leases = tuple(ExternalGpuLease.from_mapping(response) for response in responses)
            if tuple(lease.rank for lease in leases) != (0, 1):
                raise RuntimeError(f"worker lease ranks must be (0, 1), got {tuple(lease.rank for lease in leases)}")
            if any(
                lease.operation_id != operation_id or lease.owner != owner for lease in leases
            ):
                raise RuntimeError("worker lease operation_id/owner mismatch")
        except BaseException:
            try:
                self._release_remote(operation_id, owner)
            except BaseException:
                pass
            raise
        self._active = operation_id, owner
        return leases

    def release(self, operation_id: str, owner: str):
        operation_id = _identifier(operation_id, "operation_id")
        owner = _identifier(owner, "owner")
        if self._active is not None and self._active != (operation_id, owner):
            raise RuntimeError(
                f"cannot release {operation_id, owner}; active coordinator lease is {self._active}"
            )
        responses = self._release_remote(operation_id, owner)
        self._active = None
        return responses

    @contextmanager
    def lease(self, operation_id: str, owner: str):
        leases = self.acquire(operation_id, owner)
        try:
            yield leases
        finally:
            self.release(operation_id, owner)


__all__ = [
    "ACTIVITY_STATES",
    "DualGpuLeaseCoordinator",
    "ExternalGpuLease",
    "IDLE_STATE",
    "LEASED_STATE",
    "WorkerGpuLifecycle",
]
