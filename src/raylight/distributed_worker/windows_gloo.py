"""Windows Gloo setup for Raylight's single-machine CUDA workers."""

from __future__ import annotations

import os
import socket
import sys
from datetime import timedelta

import torch.distributed as dist


BACKEND_NAME = "raylight_windows_gloo"


def is_windows() -> bool:
    return sys.platform == "win32"


def _candidate_ipv4_addresses() -> list[str]:
    configured = os.environ.get("RAYLIGHT_GLOO_HOST")
    if configured:
        return [configured]

    addresses: list[str] = []
    for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
        address = info[4][0]
        if address not in addresses and not address.startswith("127."):
            addresses.append(address)
    return addresses


def select_gloo_host() -> str:
    """Pick a real adapter IPv4, avoiding Windows loopback/TUN ambiguity."""
    addresses = _candidate_ipv4_addresses()
    if not addresses:
        return "127.0.0.1"

    # Prefer ordinary private LAN ranges. Hyper-V/WSL/TUN adapters on the
    # target machine occupy 172.x, while the physical adapter is 192.168.x.
    for prefix in ("192.168.", "10.", "172."):
        for address in addresses:
            if address.startswith(prefix):
                return address
    return addresses[0]


def register_windows_gloo_backend(host: str | None = None) -> tuple[str, str]:
    """Register a ProcessGroupGloo backend with an explicit Windows IPv4."""
    if not is_windows():
        return "nccl", ""

    os.environ["USE_LIBUV"] = "0"
    selected_host = host or select_gloo_host()

    if BACKEND_NAME not in dist.Backend.backend_list:

        def create_backend(store, rank, world_size, timeout):
            options = dist.ProcessGroupGloo._Options()
            options._timeout = timeout
            options._devices = [
                dist.ProcessGroupGloo.create_device(hostname=selected_host)
            ]
            return dist.ProcessGroupGloo(store, rank, world_size, options)

        dist.Backend.register_backend(
            BACKEND_NAME,
            create_backend,
            devices=["cpu", "cuda"],
        )

    return BACKEND_NAME, selected_host


def init_windows_gloo_process_group(
    rank: int,
    world_size: int,
    master_addr: str,
    port: int,
    timeout: timedelta = timedelta(minutes=1),
) -> str:
    backend, gloo_host = register_windows_gloo_backend()
    store = dist.TCPStore(
        master_addr,
        port,
        world_size,
        rank == 0,
        timeout=timeout,
        wait_for_workers=True,
        use_libuv=False,
    )
    dist.init_process_group(
        backend,
        store=store,
        rank=rank,
        world_size=world_size,
        timeout=timeout,
    )
    return gloo_host
