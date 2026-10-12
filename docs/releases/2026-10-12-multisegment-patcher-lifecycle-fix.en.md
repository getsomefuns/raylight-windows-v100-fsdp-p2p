# Raylight 1.16.1: Multi-segment FSDP patcher lifecycle fix

## Symptom

During a multi-segment MiniMaxH3Director run, early segments could finish and save successfully while a later segment failed in the SelfLift high-resolution stage:

```text
AttributeError: 'NoneType' object has no attribute 'prepare_state'
```

The observed failure happened on the sixth selected segment. Low-resolution sampling had completed, and the exception occurred before the high-resolution ComfyUI sampler started. It was not an OOM and occurred before FaceRefine execution.

## Root cause

FSDP `ModelPatcher` clones share the same underlying model. A new segment registers its patcher in `model.current_patcher`, but an older clone could later run `cleanup()` or be finalized by Python. The previous code cleared the shared reference unconditionally.

An obsolete clone could therefore remove a newer, still-active patcher. The next sampler access to `current_patcher.prepare_state` then saw `None`.

## Fix

Both explicit cleanup and finalization now verify ownership. A clone clears the reference only when `model.current_patcher is self`. Stale clones can no longer destroy another segment's active state.

The patch does not change:

- ComfyUI or MiniMaxH3Director source;
- model weights, dtype, Safe-FP16, or quantization;
- FSDP sharding, CUDA P2P/NVLink transport, or staging capacity;
- sampling math, noise, seed, SOL, TE-Speed, SelfLift, or Refine parameters.

## Verification

- Four focused regressions cover stale-clone `cleanup()` and `__del__()`, while also proving that the true current owner can still release its own reference.
- RED: both tests reproduced the active reference being cleared before the fix.
- GREEN: both tests pass after the fix.
- `tests/test_fsdp_mixed_precision.py`: `22 passed`, plus `2 subtests passed`.
- Focused Director worker/lifecycle regression: `44 passed`, plus `2 subtests passed`.
- Full automated project suite: `757 passed`, plus `35 subtests passed`.
- Cache-free Ruff check for the modified Python files: passed.

The release commit and deployment source marker identify the dual-V100 deployed build.

## FaceRefine behavior

FaceRefine did not cause this crash. Director's `confirm_first_pass=true` intentionally completes and caches the first pass, then waits for a confirmed follow-up execution before Refine and FaceRefine run. Disable that option for a single uninterrupted execution, or keep it enabled and confirm before rerunning as designed by Director.

## Version and compatibility

- Raylight version: `1.16.1`
- This is a patch on top of the `1.16.0` O14 SelfLift integration; no O14 capability is reverted.
- Existing workflows require no rewiring because the fix is internal to the FSDP patcher lifecycle.
