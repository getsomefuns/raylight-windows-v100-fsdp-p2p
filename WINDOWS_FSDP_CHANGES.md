# Windows Dual-V100 FSDP Changes

This document summarizes the changes in this repository relative to upstream [Raylight](https://github.com/komikndr/raylight) 1.9.0 at commit `9a7c33d52b3d35e29f75ecff3c227de987f0d4cf`.

The implementation is an experimental, inference-only compatibility layer for native Windows, exactly two Tesla V100 GPUs in TCC mode, and CUDA P2P/NVLink. It is not a general Windows NCCL replacement.

## FSDP2 data path

- Adds a CUDA IPC/P2P implementation for the `all_gather_into_tensor` operation used by PyTorch FSDP2 inference.
- Retains the previous CUDA P2P `all_to_all_single` route used by Ulysses.
- Chunks FSDP shards that exceed the persistent 128 MiB per-rank staging buffer.
- Uses cross-process CUDA events, operation identifiers, strict timeouts, and endpoint poisoning after data-path failures.
- Refuses to silently move a matched CUDA FSDP weight collective through Gloo or host memory.
- Keeps Gloo/TCPStore for rendezvous, process groups, barriers, and unmatched control operations.

## Model sharding and loading

- Enables the strict two-rank FSDP topology for models loaded through `RayUNETLoader`.
- Uses nested/submodule FSDP2 wrapping so weight-bearing children reshard after forward on PyTorch 2.7.
- Preserves FP8 quantization metadata and global gathered shapes when loading mmap-backed quantized weights.
- Streams LoRA sidecar data without treating LoRA as persistent sharded base weights.
- Explicitly rejects GGUF with the current FSDP path.

## V100 numerical and memory compatibility

- Provides bounded BF16-to-FP32 dense linear conversion for V100 compute paths.
- Uses memory-bounded attention routing where V100 lacks a BF16 xFormers kernel.
- Applies chunked RMSNorm into a separate output tensor so residual inputs are not overwritten.
- Applies shape-preserving activations in place, while shape-changing activations such as SwiGLU receive correctly shaped chunked output storage.
- Retains the required `--disable-cuda-malloc` Windows V100 launch mode.

## Diagnostics and release gates

- Adds P2P bandwidth and correctness preflight checks with a default 50 GiB/s minimum.
- Adds collective profiling, FSDP wrapper/state diagnostics, rank output fingerprints, memory monitoring, and standalone hardware probes.
- Adds checked-in 5-second ComfyUI workflow and API payloads for single, Ray single, Ulysses, and FSDP comparisons.
- Separates the historical P2P/Ulysses results from the current FSDP acceptance record.
- The public launcher defaults to a 256 MiB per-GPU P2P staging buffer, displays and validates the 128/256/512 MiB choices, and keeps Rank/P2P diagnostics disabled unless `-EnableDiagnostics` is supplied.
- The low-level worker fallback remains 128 MiB when Raylight is launched without the repository script. The legacy `-P2PCapacityBytes` launcher parameter remains available for automation compatibility.

## Validated scope

The validated result is the LTX 2.3 22B FP8-scaled Diffusion Model sharded across two V100-SXM2-16GB GPUs. Both ranks participate during diffusion sampling and the final 121-frame video with the original BF16 distilled LoRA passed visual and numerical checks.

Text encoders, video/audio VAEs, spatial/temporal upscalers, and ordinary ComfyUI nodes are not automatically FSDP-sharded. Training, backward collectives, arbitrary GPU counts, multi-node operation, WDDM, and GGUF FSDP remain outside the supported release scope.

The FSDP branch currently solves the per-GPU model-capacity problem. It has not yet met the separate performance target of being at least 20% faster than the fair single-GPU baseline.

For detailed evidence and failed-test history, see [docs/WINDOWS_V100_FSDP_TESTING.md](docs/WINDOWS_V100_FSDP_TESTING.md).
