"""Windows Gloo setup for Raylight's single-machine CUDA workers."""

from __future__ import annotations

import os
import socket
import sys
from datetime import timedelta
from functools import wraps

import torch.distributed as dist


BACKEND_NAME = "raylight_windows_gloo"


def install_windows_gloo_new_group_router(dist_module=dist):
    """Route xFuser's hard-coded ``gloo`` CPU groups to our Windows backend.

    PyTorch 2.10+ restored the libuv Gloo transport on Windows, but a plain
    ``new_group(..., backend="gloo")`` still creates a second implicit Gloo
    device.  xFuser does exactly that for every coordinator's CPU group.  The
    Raylight backend already owns the explicit, validated Windows device, so
    reuse it for those groups as well.
    """
    current = dist_module.new_group
    original = getattr(current, "_raylight_original_new_group", None)
    if original is not None:
        return original
    original = current

    @wraps(original)
    def routed_new_group(*args, **kwargs):
        args = list(args)
        if len(args) >= 3 and str(args[2]).lower() == "gloo":
            args[2] = BACKEND_NAME
        if str(kwargs.get("backend", "")).lower() == "gloo":
            kwargs = dict(kwargs)
            kwargs["backend"] = BACKEND_NAME
        return original(*args, **kwargs)

    routed_new_group._raylight_original_new_group = original
    dist_module.new_group = routed_new_group
    return original


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
