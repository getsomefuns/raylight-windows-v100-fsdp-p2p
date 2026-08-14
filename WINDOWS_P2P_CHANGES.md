# Windows V100 P2P fork changes

This repository is derived from `komikndr/raylight` at commit
`9a7c33d52b3d35e29f75ecff3c227de987f0d4cf` (Raylight 1.9.0).

The fork adds an experimental native-Windows transport for the exact
two-rank, synchronous, equal-split CUDA `torch.distributed.all_to_all_single`
calls used by the tested Ulysses configuration.

## Changed upstream files

- `src/raylight/distributed_worker/ray_worker.py`: initializes a Windows Gloo
  process group, builds and validates CUDA IPC/P2P endpoints, routes supported
  all-to-all calls, and supports optional communication tracing.
- `src/raylight/distributed_worker/parallel_group_manager.py`: explicitly
  supplies the registered Windows Gloo backend to xFuser groups.
- `src/raylight/nodes.py`: forwards Windows transport environment variables
  into Ray actors and reuses healthy workers for unchanged configurations.
- `README.md`: documents the fork and links to its Windows guide.

## New implementation files

- `windows_gloo.py`: TCPStore/Gloo bootstrap with deterministic Windows IPv4
  adapter selection and `use_libuv=False`.
- `windows_p2p.py`: persistent CUDA IPC buffers, interprocess CUDA events, a
  Windows named shared-memory control plane, strict timeouts, and the
  two-rank all-to-all router.
- `a2a_trace.py`: opt-in JSONL tracing for real all-to-all shapes and timings.
- Unit tests and standalone Windows/Ray/CUDA probes under `tests/`.

## Explicit non-goals

This fork does not implement a PyTorch ProcessGroup, NCCL, FSDP, multi-node
communication, more than two ranks, asynchronous collectives, or arbitrary
split sizes. Unsupported operations remain on the Gloo fallback path.

The original Apache-2.0 license and upstream attribution are retained.
