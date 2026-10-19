# Raylight 1.18.1: Windows Multi-Segment Text Encoder Non-mmap Fix

## Failure

A 73-segment MiniMax H3 Director job completed segment 42 sampling, dual-GPU VideoVAE decode, and AudioVAE, then terminated the entire Python process with Windows `0xC0000005 access violation` while loading the Qwen3-VL-32B NVFP4 Text Encoder for segment 43. The top frames were `torch.storage.__getitem__` and `safetensors.safe_open().get_tensor()`. This was not a VideoVAE failure, an FSDP-worker failure, or a CUDA OOM.

## Root cause and fix

The TP2 Text Encoder created multiple `safetensors.safe_open()` mappings of the same 15.7 GB checkpoint for every segment. Windows charges large copy-on-write file mappings against the system commit limit. After prolonged repeated load/unload cycles, the Rust/PyTorch storage hand-off can terminate with an access violation instead of first raising a catchable Python allocation error.

Version 1.18.1 adds a bounded reader inside Raylight. On Windows it reads each tensor by its safetensors header offset into an owned `bytearray` through `readinto`, then uses the existing GPU copy and sharding path. Only the current tensor's CPU buffer is in flight. Embedding, vision-tower, and all 50 language-backbone layers use this path. Linux retains the official mmap reader. ComfyUI, MiniMaxH3Director, the checkpoint, and workflows are unchanged.

## Validation

- The regression test first failed because no non-mmap reader existed. It now proves that returned tensor storage remains valid after close and that Windows can immediately rename the checkpoint.
- The non-mmap reader and TP2 CLIP proxy suite passes 17 tests; the full repository reports `799 passed, 240 warnings, 35 subtests passed`. Warnings are existing PyTorch deprecations and Windows pytest temporary-directory cleanup notices, with no test failures.
- The deployed reader opens, reads, and closes a real 20 MiB `q_proj` NVFP4 tensor 64 consecutive times. Process RSS grows by only 3.58 MiB and VMS by 2.66 MiB, with no per-iteration handle accumulation.
- Real 15.7 GB checkpoint, dual-V100 text probe: 29 tokens, 19.61 seconds, finite output, 7347.60/7347.60 MiB peak, 16.25/16.25 MiB after close.
- Real REF2VA vision probe: 77 tokens, 20.89 seconds, finite output, 7370.09/7361.80 MiB peak, 25.05/16.92 MiB after close.

The patch changes Windows weight I/O and commit behavior only. Model values, TP2 partitioning, attention, diffusion sampling, and VideoVAE output are unchanged. A full 73-segment replay is intentionally not part of the short automated gate; the mapping-lifecycle regression and both real model paths cover the failing mechanism, while the next production continuation remains the long-duration stability confirmation.
