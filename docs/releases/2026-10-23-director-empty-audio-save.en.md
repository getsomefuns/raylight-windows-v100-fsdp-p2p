# Raylight 1.18.4: Director muted-video save compatibility

## Cause

Director mute mode returns an AUDIO dictionary with `waveform.shape=[1,1,0]`. Stock `CreateVideo → SaveVideo` creates an audio stream because that dictionary is present, then submits zero samples to the PyAV 18.1.0 resampler. This produces `av.error.MemoryError: [Errno 12] Cannot allocate memory` even in a tiny encoder-only reproduction without models, LoRA, or CUDA.

In the reported run, the second segment's LoRA sampling and dual-GPU VAE decode completed. Both Director segment MP4s and the last clip written by the external saver decode fully; the external save fails while encoding empty audio after writing video.

## Behavior

- Only Ray MODEL Director outputs receive normalization: zero-length audio becomes CPU silence matching the corresponding image clip, preserving sample rate, channel count, and the AUDIO dictionary contract.
- Nonempty audio, including silence, passes through by identity without copies, waveform edits, or resampling.
- Segment lists retain their alignment. First-pass confirmation uses `images_pre_refine` when the main image output is blocked. Shared outputs use the longest available clip; the stock saver trims audio to its selected video.
- Successful dual-VAE execution, preload fallback, and unsupported parallel-decode proxy paths are covered. Non-Ray executions retain their original behavior.
- No ComfyUI or Director source changes, additional nodes, switches, or dependencies are required. Keep the existing `Director AUDIO → CreateVideo audio` connection.

Fifteen seconds of mono FP32 silence at 32 kHz costs approximately 1.83 MiB of host RAM and no additional VRAM.

## Validation

- Before the fix, regression tests report `4 failed, 2 passed`; the real stock `VideoFromComponents.save_to()` reproduces the same PyAV MemoryError stack.
- After the fix, nine focused tests pass: silence, normal stereo audio, mixed segment lists, successful dual-VAE execution, preload fallback, first-pass confirmation, non-Ray passthrough, and real MP4 encoding/decoding.
- Encoder tests recover all six video frames, zero-amplitude silent audio, and nonzero normal audio. Normal input audio objects and tensors remain unchanged.
- Full regression: `817 passed, 35 subtests passed`; 19 upstream deprecation and Windows temporary-directory cleanup warnings, with no test failures.

Validation targets the failed output/encoding stage without repeating expensive diffusion sampling. Director's segment export locations and its policy of returning only the last retained clip to downstream savers are unchanged.
