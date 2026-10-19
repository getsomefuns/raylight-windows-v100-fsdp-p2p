# Raylight 1.18.2: Windows FSDP quantized CPU-offload native-crash fix

## Failure

Before the first dual-GPU SelfLift sample, both Ray workers terminated with a native Windows `0xC0000005` access violation while registering FSDP weights. The native stack ended in `comfy_kitchen.tensor.base._handle_to()`, called by `load_from_full_model_state_dict()` while moving a quantized shard to CPU. The worker-exit text also contained non-UTF-8 localized C10d output, which triggered a Ray raylet serialization defect and surfaced in ComfyUI as `ActorUnavailableError: Connection reset`.

This is separate from the Text Encoder safetensors mmap-lifetime failure fixed in 1.18.1 and is not a CUDA OOM.

## Cause and fix

With FSDP CPU offload enabled, the old loader still materialized every quantized local shard on CUDA and immediately copied it back through `QuantizedTensor.cpu()`. A real MiniMax H3 model registers 684 FSDP wrappers, and the repeated large round trips entered comfy-kitchen's Tensor-subclass `to_copy` path and crashed natively on the target Windows machine.

Version 1.18.2 now:

1. slices and constructs each rank-local quantized payload directly on CPU when CPU offload is enabled;
2. still converts the outer DTensor wrapper to CPU, as required by PyTorch FSDP2's public-device validation;
3. therefore performs only a safe CPU-to-CPU local-payload operation instead of copying large CUDA qdata back to host;
4. leaves non-offloaded CUDA loading, values, shard boundaries, LoRA, sampling, and P2P collectives unchanged.

## Validation

- The new regression first failed against the old loader and passes after the fix.
- Focused suite: `7 passed, 2 subtests passed`.
- Full suite: `800 passed, 35 subtests passed`.
- Dual-V100 quantized FSDP CPU-offload forward probe: both ranks match the reference element-for-element with `0.0` maximum error, and local qdata resides on CPU.
- Real checkpoint preflight: `minimax_h3_ref2va_pruned_fp8_scaled.safetensors`, `fp16_h3_safe`, 0.65 Turbo4 LoRA, FSDP CPU offload, and Ulysses 2. Both ranks register 684 wrappers, 526 DTensors, and 200 FP8 layouts. Local payload is 10,477,671,768 bytes per rank, median P2P bandwidth is about 60.10 GiB/s, and teardown completes normally.

The entire 73-segment production job was not repeated as an automated gate. Normal continuation will provide the long-duration stability check; the real preflight fully covers the model-load and FSDP-registration stage where this crash occurred.
