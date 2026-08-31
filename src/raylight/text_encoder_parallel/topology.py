"""Strict two-V100 topology validation for MiniMax H3 TP2."""

from __future__ import annotations

from dataclasses import dataclass
import subprocess
from typing import Callable, Mapping, Sequence


class PeerTopologyError(RuntimeError):
    """The host cannot safely run the strict two-GPU TP2 path."""


@dataclass(frozen=True)
class GpuDescriptor:
    index: int
    name: str
    driver_model: str


def validate_gpu_descriptors(
    descriptors: Sequence[GpuDescriptor],
    peer_access: Mapping[tuple[int, int], bool],
) -> tuple[int, int]:
    if len(descriptors) != 2:
        raise PeerTopologyError(f"TP2 requires exactly two GPUs, found {len(descriptors)}")
    ordered = tuple(sorted(descriptors, key=lambda descriptor: descriptor.index))
    if tuple(descriptor.index for descriptor in ordered) != (0, 1):
        raise PeerTopologyError("TP2 GPU indices must be exactly 0 and 1")
    for descriptor in ordered:
        if "V100" not in descriptor.name.upper():
            raise PeerTopologyError(
                f"GPU {descriptor.index} must be a Tesla V100, found {descriptor.name}"
            )
        if descriptor.driver_model.upper() != "TCC":
            raise PeerTopologyError(
                f"GPU {descriptor.index} must use TCC, found {descriptor.driver_model}"
            )
    if not peer_access.get((0, 1), False) or not peer_access.get((1, 0), False):
        raise PeerTopologyError("TP2 requires bidirectional CUDA peer access between GPU 0 and 1")
    return (0, 1)


def query_nvidia_smi_descriptors() -> list[GpuDescriptor]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_model.current",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if completed.returncode:
        raise PeerTopologyError(
            f"nvidia-smi topology query failed with exit code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    descriptors = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise PeerTopologyError(f"unexpected nvidia-smi topology row: {line!r}")
        descriptors.append(GpuDescriptor(index=int(fields[0]), name=fields[1], driver_model=fields[2]))
    return descriptors


def validate_windows_v100_topology(
    *,
    torch_module=None,
    descriptor_query: Callable[[], Sequence[GpuDescriptor]] = query_nvidia_smi_descriptors,
    devices: tuple[int, int] = (0, 1),
) -> tuple[int, int]:
    if devices != (0, 1):
        raise PeerTopologyError(f"TP2 devices must be (0, 1), got {devices}")
    if torch_module is None:
        import torch as torch_module

    if not torch_module.cuda.is_available():
        raise PeerTopologyError("CUDA is not available")
    if torch_module.cuda.device_count() != 2:
        raise PeerTopologyError(
            f"TP2 requires exactly two CUDA devices, found {torch_module.cuda.device_count()}"
        )
    peer_access = {
        (0, 1): bool(torch_module.cuda.can_device_access_peer(0, 1)),
        (1, 0): bool(torch_module.cuda.can_device_access_peer(1, 0)),
    }
    return validate_gpu_descriptors(descriptor_query(), peer_access)
