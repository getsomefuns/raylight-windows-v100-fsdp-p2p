# Windows V100 FSDP Phase F3 Plan: 5-Second LTX Integration

> Date: 2026-08-16<br>
> Branch: `windows-v100-fsdp-p2p`<br>
> Prerequisite: Phase F2 accepted<br>
> Scope: native Windows, 2 x Tesla V100-SXM2-16GB, TCC, CUDA P2P/NVLink

## Objective

Integrate the Phase F2 fixed-buffer CUDA P2P ProcessGroup into a real 5-second LTX-2.3 Raylight workflow and prove that FSDP reduces persistent model-weight VRAM per rank without introducing CPU payload fallback, pagefile pressure, numerical corruption, rank drift, or retained full parameters after forward.

This phase is an FSDP correctness and memory-capacity gate. It is not yet the final speed-optimization phase.

## Accepted F2 Baseline

- NVIDIA P2P probe: peer access enabled in both directions; approximately 245 GB/s bidirectional.
- Raylight CUDA P2P collectives: approximately 59-62 GiB/s per rank.
- Fixed 128 MiB staging buffer passes 64, 128, 256, and 384 MiB shard transfers.
- FP32, FP16, BF16, and byte-tail transfers pass in synchronous and asynchronous modes.
- FSDP2 synthetic model: 512 MiB full parameters, 256 MiB local shard per rank, exact output, full parameters released after forward.
- Five fresh Ray/Python sessions pass without orphan workers or pagefile use.
- TCPStore hostname warning is classified as a harmless IPv6-first connection attempt followed by a successful IPv4 loopback connection; it is not a payload fallback.

## Fixed F3 Configuration

- GPU workers: 2.
- Driver model: TCC on both V100s.
- Raylight mode: pure FSDP.
- `ulysses_degree=0`, `ring_degree=0`, `cfg_degree=0`.
- `dp_degree=1`: one model replica sharded across both workers.
- `FSDP=true`; `FSDP_CPU_OFFLOAD=false`.
- `use_mmap=true`; `skip_comm_test=true`.
- Attention backend: PyTorch efficient attention; no FlashAttention installation on V100.
- CUDA P2P staging capacity: 18 MiB per rank for the real LTX workflow; the
  standalone correctness probe continues to validate 128 MiB staging.
- P2P launch gate: five 100 GiB scored trials, 50 GiB/s per-rank median;
  20 GiB clock warm-up trials are recorded but not scored.
- Video VAE tiled decode: 384/64 spatial and 64/8 temporal.
- ComfyUI: `--disable-cuda-malloc`; keep the established Windows/Ray environment variables.
- Sampler: preserve the existing Raylight custom-sampling chain. `XFuserSamplerCustomAdvanced` does not require xFuser topology; it submits the same custom-sampling job to every FSDP rank. Data Parallel KSampler remains the choice for regular KSampler workflows.
- Model: the existing LTX-2.3 FP8-scaled checkpoint. Do not force FP16 inference; inspect the actual runtime dtype and quantized layout before sampling.
- No GGUF and no CPU-offload path in this phase.

## Safety and Preservation Rules

- Preserve the current Ulysses/P2P performance workflow unchanged as the comparison baseline.
- Create a separate FSDP workflow copy; do not overwrite the user's working workflow.
- Do not start the 5-second sampler until topology and load-only gates pass.
- Stop immediately on black/NaN output, rank desynchronization, retained full weights, CPU payload fallback, pagefile growth attributable to the run, or physical VRAM exhaustion.
- Do not proceed to a 10-second workflow in F3.

## Task 1: Harden and Test Bottom-Up FSDP Topology

Files:

- `ComfyUI/custom_nodes/raylight/src/raylight/comfy_dist/fsdp_utils.py`
- `ComfyUI/custom_nodes/raylight/tests/test_ltx_fsdp_topology.py`

Work:

1. Add a synthetic LTX-like transformer with multiple weight-bearing blocks.
2. Prove that `collect_bottom_up_shard_order()` places non-root weight-bearing units before the root.
3. Add a validation helper that rejects a root-only FSDP topology for inference.
4. Call the validation before `fully_shard()` mutates the model.
5. Preserve ignored quantization scales and excluded-module behavior.

Acceptance:

- The new test first fails without the guard and passes after the minimal implementation.
- Root is the final wrapper, and at least one non-root parameter-bearing wrapper exists.
- A model that would be wrapped only at root fails before model state loading.
- Existing FSDP and Windows P2P tests remain green.

## Task 2: Add Load-Time FSDP Diagnostics

Files:

- `ComfyUI/custom_nodes/raylight/src/raylight/comfy_dist/fsdp_utils.py`
- `ComfyUI/custom_nodes/raylight/src/raylight/comfy_dist/model_patcher.py`
- focused tests under `ComfyUI/custom_nodes/raylight/tests/`

Work:

1. Report per rank: wrapper count, non-root wrapper count, logical full parameter bytes, local parameter bytes, quantized layout/dtype summary, and ignored parameter bytes.
2. Add an inference-time check that the model does not present a root-only topology.
3. Keep diagnostics compact and free of machine-specific identifiers.

Acceptance:

- Both ranks report the same logical full size and compatible wrapper topology.
- Local sharded weight size is materially below full size for shardable weights.
- The checkpoint's actual FP8/quantized handling is visible before sampling.

## Task 3: Create a Separate 5-Second FSDP Workflow

Inputs:

- Baseline workflow: `ComfyUI/user/default/LTX2_3_i2v_Raylight.json`
- Test image: `ComfyUI/input/LTX2_3_i2v_Raylight.jpg`

Output:

- `ComfyUI/custom_nodes/raylight/example_workflows/LTX2_3_i2v_Raylight_Windows_FSDP_5s.json`

Work:

1. Copy the known-good 5-second LTX workflow.
2. Apply the fixed F3 RayInitializer settings.
3. Preserve the existing two-stage `RayCFGGuider` and `XFuserSamplerCustomAdvanced` chain; only the RayInitializer parallel mode changes.
4. Keep the existing Windows P2P/Ulysses workflow untouched.

Acceptance:

- Workflow JSON loads without missing-node or broken-link errors.
- Static workflow validation confirms pure FSDP settings and no CPU offload.
- The test image and original prompt remain reproducible without embedding absolute machine paths.

## Task 4: Run a Load-Only Two-Rank Preflight

Work:

1. Start ComfyUI through `<environment-root>/start_raylight_p2p.ps1` with `--disable-cuda-malloc`.
2. Initialize the two Ray workers and load/patch the LTX model without starting denoising.
3. Capture compact snapshots before load, after shard load, and after cleanup:
   - physical RAM and committed memory;
   - pagefile use;
   - per-GPU allocated/reserved/physical VRAM;
   - FSDP wrapper and shard metrics;
   - CUDA P2P operation counters.
4. Verify no Gloo/CPU payload collective is used for model-weight all-gather.

Acceptance:

- Both ranks finish model loading and FSDP patching.
- No rank retains a complete persistent copy of shardable model weights.
- No pagefile growth attributable to model payload movement.
- CUDA P2P collectives are exercised; control-plane Gloo remains metadata-only.
- No orphan Ray/Python worker remains after cleanup.

## Task 5: Execute 5-Second Cold and Warm Runs

Prerequisite: Task 4 passes all gates.

Work:

1. Run one cold 5-second generation from a clean worker/model state.
2. Run one warm 5-second generation without reloading the model.
3. Record stage timing and iterations per second for model load, conditioning, denoising, VAE decode, audio decode, and output save.
4. Monitor rank progress timestamps, system commit/pagefile, and both GPUs throughout.

Acceptance:

- Both ranks enter and finish every collective phase without timeout or growing drift.
- Both GPUs perform the denoising work.
- The run does not exceed physical VRAM and does not require CPU payload fallback.
- Cold and warm timings are preserved in the maintainable test-results document.

## Task 6: Output and Regression Gate

Work:

1. Validate video frames for finite values, non-black content, expected dimensions/frame count, and playable output.
2. Validate audio for finite values and playable output.
3. Re-run the unit suite, P2P probe, and minimal/large FSDP probes.
4. Compare FSDP memory and runtime against the preserved single-card and Ulysses baselines.

F3 GO criteria:

- Valid non-black 5-second video and valid audio.
- No NaN or infinity in sampler, video, or audio paths.
- Persistent per-rank shardable model-weight memory is materially below a full model copy.
- 128 MiB fixed staging remains active.
- No CPU payload fallback, no pagefile dependency, no rank timeout, and no orphan process.
- All focused and regression tests pass.

F3 NO-GO criteria:

- Root-only FSDP topology or retained full parameters after forward.
- FP8 checkpoint cannot execute numerically correctly on V100 under the supported runtime path.
- Any model-weight collective silently falls back through host memory.
- Repeatable rank divergence, black output, NaN/Inf, OOM, or pagefile-driven stalls.

## Deliverables

- This F3 plan and an execution-status section maintained in place.
- Topology guard and focused tests.
- Compact load-time FSDP diagnostics.
- Separate 5-second Windows FSDP workflow.
- A concise F3 result section added to the maintainable Raylight test-results document.
- No commit or GitHub push until explicitly requested by the user.

## Execution Status

- [x] Phase F2 accepted.
- [x] Task 1: topology guard and tests.
  - Synthetic LTX-like blocks are wrapped bottom-up with the root last.
  - Root-only inference topology is rejected before `fully_shard()` can mutate the model.
- [x] Task 2: load-time diagnostics.
  - Per-rank wrapper, DTensor, logical/local/unsharded payload, dtype, layout, and PyTorch VRAM fields are emitted without copying payloads.
- [x] Task 3: 5-second FSDP workflow copy.
  - Added `LTX2_3_i2v_Raylight_Windows_FSDP_5s.json` with one replica sharded over two workers.
  - Preserved the original image, prompts, model, LoRA, 121 frames, 25 fps, and two-stage custom sampler chain.
  - Preserved the Ulysses baseline unchanged.
- [x] Task 4: load-only preflight.
  - Both ranks loaded 2,999 FSDP wrappers (2,998 non-root) and 4,444 parameters with no root-only topology.
  - Plain BF16 checkpoint tensors retain BF16 storage through both ComfyUI loading stages; quantized tensors retain their descriptors and scales.
  - Per-rank persistent model payload fell to about 11,203 MiB; the large BF16 gather fell from 576 MiB FP32 to 288 MiB BF16.
  - Pagefile usage did not grow during the accepted run; mmap file reads are not classified as pagefile payload fallback.
- [x] Task 5: cold/warm generation.
  - Same-process run 1: 530.912 s, success.
  - Same-process run 2: 380.017 s, success.
  - End-to-end warm improvement: 28.42%.
  - Warm peak physical/committed memory: 48,983/68,105.7 MiB.
  - Warm peak page reads: 10.9/s; pagefile usage stayed at 426 MiB.
  - Both GPUs reached full utilization without rank timeout or drift.
- [x] Task 6: output and regression gate.
  - Both outputs are H.264 1280x704 at 25 fps with 121 video frames, AAC audio, and 4.84 s duration.
  - No black segment of 0.1 s or longer; no NaN/Inf report.
  - 104 unit tests passed.
  - Release P2P probe passed all-to-all/all-gather, 64-384 MiB payloads, FP32/FP16/BF16/uint8 tails, and sync/async modes.
  - 512 MiB FSDP probe produced exact finite output, 256 MiB local shards, and returned allocation to about 136 MiB after teardown.

## Final F3 Decision

**GO. Phase F3 is accepted on the validated native-Windows dual-V100 TCC environment.**

The FSDP mode produces valid video and audio, uses CUDA P2P/NVLink for the payload path, keeps Gloo metadata-only, avoids attributable pagefile growth, and exceeds the 20% same-process cold-to-warm improvement target. The accepted workflow uses an 18 MiB fixed P2P buffer and a 384-pixel VAE tile because the original 128 MiB/512-pixel combination did not fit repeatably beside retained FSDP shards on 16 GiB V100s.
