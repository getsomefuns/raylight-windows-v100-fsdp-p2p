# 1.11.0 / O9: Adaptive Windows CUDA P2P chunking

Date: 2026-08-31

## User-visible changes

- The default persistent CUDA staging budget is 128 MiB per rank instead of 256 MiB.
- Valid two-rank, equal-split, contiguous-CUDA `all_to_all_single` and FSDP `all_gather_into_tensor` collectives larger than that budget are automatically split into CUDA P2P/NVLink chunks inside one logical call.
- Normal startup no longer requires sizing a 256/512 MiB buffer from video length, resolution, or model choice.
- `-P2PCapacityMiB 256/512` remains an advanced performance-comparison override, not a payload-capacity repair.
- `-ValidateOnly` reports the staging budget and `Larger collectives are auto-chunked: enabled`.
- Detailed rank/P2P diagnostics remain off by default. The LTX benchmark sets `RAY_DEDUP_LOGS=0` so Ray cannot collapse identical rank diagnostics.

## Shipped implementation

- Exact all-to-all byte-range planning covers both ranks, local/remote halves, boundaries, and tail chunks.
- One persistent buffer is reused only after peer-consumed synchronization for each chunk.
- Full logical geometry is exchanged before payload transfer. Size, operation, copy, and timeout failures poison the endpoint; no partial success or silent CPU/Gloo payload fallback is allowed.
- Profiling separates logical `calls` from physical `chunks`.
- Worker health crosses the default 128 MiB boundary before model loading.
- The environment-root launcher is a thin forwarder, leaving one authoritative default in the repository launcher.
- The LTX benchmark resolves both the canonical sibling-ComfyUI worktree layout and the deployed `custom_nodes` layout.

## Communication and VRAM results

| Gate | Result |
|---|---|
| Historical failure | 239,826,944 remote bytes/rank complete in two chunks through a 128 MiB endpoint |
| 64 MiB live coverage | 64/128/256/512/768 MiB across FP32, FP16, BF16 bytes, and uint8; zero error |
| Release stress | 512 MiB/rank, eight chunks/call, 100 iterations; no drift, timeout, or orphan worker |
| 256 MiB payload | 61.696 GiB/s median, 98.91% of the matched-large-buffer baseline |
| 512 MiB payload | 61.774 GiB/s median, 98.50% of the matched-large-buffer baseline |
| Persistent VRAM | 128 MiB/rank less than the 256 MiB choice and 384 MiB/rank less than 512 MiB |

Double buffering/pipeline depth 2 was not shipped. The single-buffer path already clears the performance gate; a second buffer would double persistent staging VRAM and materially complicate event/control sequencing.

## Real-workflow acceptance

Every run used the 128 MiB default and the strict ten-second coordination timeout without reducing geometry or disabling Refine.

| Workflow | Cold | Warm | Result |
|---|---:|---:|---|
| MiniMax H3 I2V safe-FP16 Turbo8 | 520.55 s | 422.54 s | 1120×768/124 frames; dual-GPU PASS |
| MiniMax H3 REF2VA safe-FP16 Turbo4 | 362.21 s | 256.92 s | 1120×768/124 frames; dual-GPU PASS |
| Five-segment MMH3Director + Refine | 1445.67 s | 1362.48 s | ten samples; 1280×736/365 frames/24 FPS |
| LTX 2.3 five-second FSDP | 517.13 s | 386.36 s | 1280×704/121 frames/25 FPS; 14.9/45.9 and 14.9/47.8 s/it |

All four Director outputs and both LTX outputs decode fully, contain only unique decoded frames, have zero detected black intervals, and report zero audio NaN/Inf. Final benchmark exit returns both GPUs to 10 MiB and leaves no target worker. Physical RAM, commit, and pagefile monitoring shows no payload-sized host staging growth.

## Upgrade and compatibility

Existing workflows, `RayInitializer`, O6 safe-FP16, the MMH3Director adapter, FSDP CPU offload, RayKill, and the 128/256/512 parameters remain compatible. O9 does not add NCCL, GGUF FSDP, multi-node, more-than-two-GPU, Text Encoder/VAE sharding, or training support.

After upgrading, remove routine `-P2PCapacityMiB 256/512` arguments and start with the default. If 128 MiB still fails, inspect TCC/P2P, the tensor contract, rank geometry, endpoint poison state, and the overall VRAM peak rather than treating a larger staging buffer as the repair.

See the [O9 acceptance record](../testing/windows-p2p/ADAPTIVE_CHUNKING_2026-08.zh-CN.md) for detailed evidence.
