# Windows V100 FSDP Phase F0/F1 Progress

Date: 2026-08-16
Branch: `windows-v100-fsdp-p2p`
Status: experimental prototype; not ready for LTX production workflows

## Objective

Determine the exact CUDA collective required by Raylight FSDP inference on
PyTorch 2.7, then prove that the existing Windows TCC CUDA IPC data plane can
carry it without sending model weights through Gloo or host RAM.

## Confirmed platform gate

- 2? Tesla V100-SXM2-16GB, driver 577.00.
- Both GPUs are in TCC mode.
- CUDA peer access succeeds in both directions.
- NVIDIA sample result: about 123.7 GB/s unidirectional and 245 GB/s aggregate
  bidirectional P2P bandwidth.
- The two-Ray-actor project probe sustains about 59.3 GiB/s of useful remote
  payload per direction.
- WDDM remains unsupported because CUDA peer access and cross-process CUDA IPC
  fail on this hardware.

## F0 call-path finding

PyTorch 2.7 FSDP2 inference calls:

```text
FSDP forward
  -> foreach_all_gather
  -> torch.distributed.all_gather_into_tensor(async_op=True)
```

The backward-only path calls `reduce_scatter_tensor`; it is not required by
the current inference-only scope.

The FSDP process group is still useful for rank, world-size, DeviceMesh,
barriers and small control messages. A complete third-party ProcessGroup or
NCCL reimplementation is not required for the first inference prototype.
Raylight can keep its Windows Gloo control plane and route only CUDA
`all_gather_into_tensor` through the custom CUDA IPC data plane.

## Implemented prototype

- Two-rank all-gather byte layout and validation.
- CUDA-only routing for `all_gather_into_tensor`.
- Explicit refusal to fall back CUDA weight tensors to Gloo.
- CUDA Event-backed `torch.distributed.Work` compatibility for
  `async_op=True`.
- Paired installation and restoration of the all-to-all and all-gather
  routers.
- RayWorker now permits the fixed two-rank FSDP inference topology.
- Existing Ulysses all-to-all behavior remains available.

## Verification results

### Raw CUDA all-gather

Both Ray actors produced the ordered result `(rank0 shard, rank1 shard)`.

| Path | Rank 0 | Rank 1 |
|---|---:|---:|
| Existing all-to-all | 59.19 GiB/s | 59.18 GiB/s |
| New synchronous all-gather | 59.32 GiB/s | 59.32 GiB/s |
| Async Work type and wait | pass | pass |

### Minimal FSDP2 forward

A small CUDA model was sharded over a two-rank Gloo DeviceMesh while parameter
all-gather used the custom P2P route.

| Check | Rank 0 | Rank 1 |
|---|---:|---:|
| Model is FSDPModule | yes | yes |
| Full parameter elements | 192 | 192 |
| Local parameter elements | 96 | 96 |
| Maximum error vs unsharded baseline | 0.0 | 0.0 |

This proves functional weight sharding and CUDA P2P parameter reconstruction.
It does not yet prove that the LTX model fits or performs well.

## Current limits and open risks

1. A single local shard is currently limited by the configured P2P staging
   capacity (128 MiB in the normal launcher). Large FSDP parameter groups need
   chunked all-gather before LTX testing.
2. Only the two-rank, single-machine, inference path is supported.
3. Training `reduce_scatter`, multi-node operation and arbitrary world sizes
   are out of scope.
4. The minimal FSDP probe emitted a non-fatal c10d socket warning mentioning
   `<hostname>:7274`. Results were correct, but the source of that probe must
   be identified before large-model runs.
5. Quantized layouts need separate compatibility tests. GGUF FSDP remains
   unsupported.
6. Full ComfyUI/Raylight model loading, unloading, error propagation and memory
   release have not yet passed FSDP mode acceptance.

## Next gate

Before loading LTX weights:

1. Trace the largest FSDP all-gather shard sizes for LTXAV.
2. Add chunked all-gather for shards larger than the staging buffer.
3. Test FP16, BF16 and byte-packed mixed-dtype payloads.
4. Run a synthetic 256-512 MiB parameter-group FSDP forward and verify peak
   VRAM, system commit and repeated-run cleanup.
5. Eliminate or fully explain the `<hostname>:7274` c10d warning.
6. Only then enable an experimental ComfyUI FSDP workflow, beginning with the
   smallest 5-second LTX configuration.

## Workspace cleanup performed

The following reproducible development artifacts were removed after their
important conclusions had been preserved in repository documentation:

- top-level `benchmarks` and `benchmark_results`;
- `diagnostic_backups`;
- old top-level `logs`;
- abandoned `wsl-install` helpers;
- `comfyui-codex-run2/3/4` stdout and stderr logs.

Models, the Python environment, workflows, input assets, generated output and
the active ComfyUI user log were not removed.
