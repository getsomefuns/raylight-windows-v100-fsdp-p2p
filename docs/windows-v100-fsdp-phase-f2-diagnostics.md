# Windows V100 FSDP Phase F2 Diagnostics

Date: 2026-08-16

## Scope

This report identifies the source of the recurring c10d warning addressed to
`<hostname>:<dynamic-port>`. The investigation separates Ray, Gloo,
DeviceMesh, FSDP2 and the CUDA P2P data plane so that a successful fallback is
not mistaken for a failed distributed session.

## Environment snapshot

- Windows host name and FQDN: `<hostname>`.
- PyTorch: 2.7.0+cu126, Windows build without libuv TCPStore support.
- Rendezvous address requested by Raylight: `127.0.0.1`.
- Explicit Gloo device address: `<physical-adapter-ip>`.
- Ray node, GCS, raylet and object-store addresses: loopback.
- Ray actors do not receive `MASTER_ADDR`, `MASTER_PORT`, `RANK`,
  `WORLD_SIZE`, `LOCAL_RANK`, `GLOO_SOCKET_IFNAME` or
  `RAY_NODE_IP_ADDRESS` from the environment.
- `<hostname>` resolves to multiple IPv6 addresses and three IPv4 adapter
  addresses (`<virtual-adapter-ip-a>`, `<physical-adapter-ip>`, `<virtual-adapter-ip-b>`).

## Isolation results

| Stage | CUDA | Ray | Gloo | DeviceMesh/FSDP | Result |
|---|---:|---:|---:|---:|---|
| TCPStore, one process and world size 1 | No | No | No | No | Warning reproduced; store remained usable |
| TCPStore, two ordinary Python processes | No | No | No | No | Warning reproduced on both processes; keys synchronized |
| TCPStore in two Ray actors | No | Yes | No | No | Warning reproduced at the exact store port |
| Gloo and DeviceMesh | No payload | Yes | Yes | DeviceMesh only | Warning reproduced; initialization and barrier succeeded |
| Minimal two-V100 FSDP2 forward | Yes | Yes | Control plane | Yes | Warning reproduced; sharding and forward were correct |

Changing the requested store address from `127.0.0.1` to `<physical-adapter-ip>` did
not remove the warning. `localhost` added an extra failed candidate. This rules
out Ray environment injection and a simple TCPStore/Gloo interface mismatch.

## Confirmed root cause

With `TORCH_CPP_LOG_LEVEL=INFO` and `TORCH_DISTRIBUTED_DEBUG=DETAIL`, the
single-process TCPStore probe showed the complete sequence:

1. The legacy server socket successfully listened on an IPv6 dual-stack
   address.
2. PyTorch's client path, whose `SocketOptions` defaults to
   `prefer_ipv6=true`, first tried an IPv6 candidate for the supplied
   `127.0.0.1` address.
3. Winsock rejected that candidate with error 10049 (address invalid in its
   context).
4. PyTorch immediately tried the IPv4 candidate and connected successfully.
5. TCPStore validation, synchronization and destruction completed normally.

`<hostname>` is produced by c10d's socket formatter through reverse name
lookup. It is a display name for the attempted socket address, not a different
master address selected by Raylight.

The warning therefore belongs to the address-candidate loop in PyTorch 2.7's
legacy Windows TCPStore. It occurs before Gloo, DeviceMesh, FSDP or the custom
CUDA P2P route exists. It does not indicate a CUDA payload fallback and it is
not evidence that NVLink/P2P failed.

Relevant upstream implementation:

- PyTorch 2.7 `TCPStore.cpp` constructs a client after starting the server:
  <https://github.com/pytorch/pytorch/blob/v2.7.0/torch/csrc/distributed/c10d/TCPStore.cpp>
- PyTorch 2.7 `socket.h` defaults `SocketOptions::prefer_ipv6_` to `true`:
  <https://github.com/pytorch/pytorch/blob/v2.7.0/torch/csrc/distributed/c10d/socket.h>
- PyTorch 2.7 `socket.cpp` tries IPv6 first and then IPv4:
  <https://github.com/pytorch/pytorch/blob/v2.7.0/torch/csrc/distributed/c10d/socket.cpp>

## Decision and monitoring

No production address change is justified: loopback, the physical IPv4 and
`localhost` all completed, and changing the address does not change the
IPv6-first implementation. Raylight also must not globally suppress c10d
warnings because doing so could hide a real bind, timeout or rendezvous
failure.

`tests/windows_tcpstore_probe.py` is the narrow health monitor. Acceptance
requires both ordinary Python processes to construct the store, exchange both
ready keys and exit successfully. The known 10049 candidate warning is only
