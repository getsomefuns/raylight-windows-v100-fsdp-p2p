# MiniMax H3 Safe FP16 on Raylight FSDP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, numerically safe MiniMax H3 FP16 compute path for native-Windows dual-V100 Raylight FSDP while preserving FP8 checkpoint storage, CUDA P2P and the accepted FP32-compute workflows.

**Architecture:** Port the MIT external patch into a focused Raylight module and install it idempotently inside every Ray actor before MiniMax H3 model construction and FSDP wrapping. Keep the condition projection, residual stream and output heads in FP32, cast only attention/MLP branch inputs to FP16, and retain FP8 FSDP storage with chunked V100 dequantization. Expose the feature as a dedicated RayUNETLoader mode so generic FP16 and all accepted baseline workflows remain unchanged.

**Tech Stack:** Windows 23H2, two Tesla V100-SXM2-16GB in TCC mode, Python 3.10.11, PyTorch 2.7.0+cu126, ComfyUI 0.31.0, Ray 2.57.0, Raylight FSDP2, yunchang/USP, comfy-kitchen FP8, Windows CUDA P2P.

**Spec:** `docs/superpowers/plans/2026-08-17-minimax-h3-memory-performance.md#phase-o6-safe-fp16-compute-for-minimax-h3-on-raylight-fsdp`

## Global constraints

- Work only in `E:\ComfyUI-py310`; do not change NVIDIA driver 577.00 or any environment outside that tree.
- Preserve `--disable-cuda-malloc --reserve-vram 2`, two FSDP workers, CPU offload, Ulysses 2, ring 1 and Windows CUDA P2P.
- Do not globally add FP16 to ComfyUI `MiniMaxH3.supported_inference_dtypes` and do not overwrite the accepted O5 workflows.
- Keep FP8 model artifacts, BF16 Turbo LoRAs, model weights, raw logs and generated media out of Git.
- Attribute the upstream MIT project and retain its license notice when adapting code.
- Treat every speed figure below as a hypothesis until reproduced on the reference dual-V100 machine.
- Synchronize the accepted O1-O5 implementation and documentation to GitHub before creating O6 code.

## Evidence baseline

### Verified locally

- External source identity: `Amduraznak/minimax-h3-fp16-fix`, local/origin `b09897c`, MIT license, tested by its author on ComfyUI 0.30 and one V100 32GB.
- The external module imports in the current ComfyUI 0.31 process; `MiniMaxH3Model`, `DiTBlock` and `MLP` names/signatures still match.
- O5 I2V/REF2VA use `minimax_h3_*_pruned_fp8_scaled.safetensors` with `weight_dtype=default`.
- Accepted logs show `[Raylight][comfy_kitchen][fp8] fallback dequantize dtype=torch.float32`; current tests intentionally describe V100 FP32 compute.
- Ray actors receive Raylight through `py_modules` and ComfyUI through `PYTHONPATH`, but they do not scan arbitrary `ComfyUI/custom_nodes`; the upstream drop-in install cannot reach the workers.
- MiniMax USP replaces each attention forward and the model `_forward`, but calls `block(...)`, `block.attn.out_proj(...)` and the existing final layer. The safe-FP16 math can coexist if installed before model construction.

### Predictions requiring proof

- FP8 storage and P2P parameter transport can stay unchanged because the V100 fallback chooses its dequantized matrix dtype from the branch input tensor.
- FP32 residuals prevent the 50-block growth from overflowing, while FP16 branch inputs allow V100 Tensor Cores to execute the dominant QKV/out-projection and MLP matrix multiplications.
- LoRA correctness should be preserved because Raylight's `LoRAAdapter.h()` casts sidecar tensors to `x.dtype`; repeated conversion overhead and memory residency still require measurement.
- Sampling should improve materially, but CPU offload, FSDP unshard/reshard, preprocessing and VAE stages cap end-to-end gains.

## Local performance baseline and release gate

No external timing number is an O6 acceptance threshold. Before Task 1 starts, create a new matched local baseline from the exact current implementation:

| Variable | I2V baseline | REF2VA baseline |
|---|---|---|
| Workflow | Turbo8 I2V | Turbo4 REF2VA |
| Compute policy | existing `default` FP32 compute | existing `default` FP32 compute |
| Resolution | 1120x768 | 1120x768 |
| Frames / FPS | 124 / 24 | 124 / 24 |
| Startup state | clean cold start | clean cold start |
| Input and prompt | existing O5 input/prompt | existing O5 reference inputs/prompt |

For each run, retain concise values in Git and keep the raw log/telemetry/media local:

- complete video wall time;
- ComfyUI preprocessing time;
- Ray worker creation and checkpoint/model-load time;
- every sampler's first-step, stable-step and aggregate `s/it` plus sampler total;
- VAE decode time and media-save/container-write time;
- GPU0/GPU1 utilization, power and VRAM peaks;
- host physical/committed memory and pagefile peaks;
- P2P health/traffic and both-rank completion.

Let `B_i` be the stable sampling `s/it` measured for workflow `i` above and `F_i` the matched O6 safe-FP16 result. The initial O6 functional/performance acceptance gate is:

```text
B_i / F_i >= 4.0
```

for both workflows. Reaching `11x` remains the O6 optimization target after the initial gate passes; it is not a blocker for the first correct safe-FP16 release candidate. The actual numeric limits are written into the validation report only after the baseline completes. Model load, preprocessing, VAE decode and media save must each be no slower than the matched baseline, and end-to-end wall time must improve. Numerical, media, rank, memory and P2P gates remain mandatory regardless of speed.

## Planned file structure

- Create `src/raylight/comfy_dist/minimax_h3_fp16.py`: idempotent patch installer, mode detection, FP32-island/FP16-branch forward functions and diagnostics.
- Modify `src/raylight/nodes.py`: add the explicit `fp16_h3_safe` RayUNETLoader choice and transmit a serializable model-option flag.
- Modify `src/raylight/distributed_worker/ray_worker.py`: consume the flag, install the patch before model construction in every rank and reject unsupported combinations clearly.
- Test `tests/test_minimax_h3_fp16_patch.py`: numerical-range, dtype-island, idempotence, worker-install and ComfyUI-signature contracts.
- Extend `tests/test_minimax_h3_fsdp_quant.py`: FP8 storage plus FP16 V100 fallback and FSDP wrapper contracts.
- Extend `tests/test_minimax_h3_turbo.py`: BF16 Turbo LoRA sidecar follows FP16 branch inputs without changing the accepted FP32 test.
- Modify `scripts/minimax-h3/build_workflows.py`: generate dedicated experimental safe-FP16 variants without changing baseline artifacts.
- Create `example_workflows/Minimax_H3_I2V_Windows_V100_FSDP_Turbo8_FP16_Experimental.json`.
- Create `example_workflows/Minimax_H3_REF2VA_Windows_V100_FSDP_Turbo4_FP16_Experimental.json`.
- Create `docs/testing/minimax-h3/SAFE_FP16_FSDP_2026-08.md`: maintained validation report with concise results and local evidence links.
- Create `docs/third-party/minimax-h3-fp16-fix.md`: upstream URL, pinned commit, MIT attribution, adapted concepts and local deviations.

### Task 0: Synchronize O1-O5 and lock the matched local baseline

- [x] Reconcile `README.md`, `README_EN.md`, implementation plans and MiniMax validation summaries with the code and workflows already accepted through O5.
- [x] Run the complete test suite, commit the reconciled repository state and push the public `main` branch before O6 implementation.
- [x] Generate isolated benchmark copies of Turbo8 I2V and Turbo4 REF2VA at 1120x768, 124 frames and 24 FPS without changing the accepted O5 GUI workflows.
- [x] Start each workflow from an idle GPU/Ray/ComfyUI state and run one complete cold job.
- [x] Record total, preprocessing, worker/model load, each sampler's first/stable/aggregate `s/it`, VAE decode, media save, VRAM/utilization/power, host RAM/commit/pagefile and P2P/rank evidence.
- [x] Validate dimensions, frame count, video/audio streams, non-black temporal variation and finite outputs.
- [x] Write the concise baseline table and the calculated per-workflow initial `B_i / 4` gates plus `B_i / 11` optimization targets to `docs/testing/minimax-h3/SAFE_FP16_FSDP_2026-08.md`; keep raw evidence local.
- [x] Do not begin Task 1 until both baselines are complete and their data is internally consistent.

### Task 1: Lock numerical and activation contracts with failing tests

**Interfaces:**

```python
def install_minimax_h3_safe_fp16_patch() -> bool: ...
def safe_fp16_requested(model_options: dict) -> bool: ...
```

- [x] Add tests proving installation is idempotent and inactive for FP32/BF16 models.
- [x] Add tiny-layer tests proving the residual remains FP32, attention/MLP inputs are FP16, condition projection receives FP32, and scaled linear outputs remain finite above 65,504-equivalent unscaled magnitude.
- [x] Add a signature guard for the current ComfyUI `DiTBlock.forward`; fail with an actionable compatibility error after an upstream refactor.
- [x] Run `E:\ComfyUI-py310\Python310\python.exe -m pytest -q tests/test_minimax_h3_fp16_patch.py` and record the expected RED failures before implementation.
- [x] Preserve the expected RED output as local evidence, implement the smallest passing change, and commit only after the focused tests are GREEN.

### Task 2: Port the safe-FP16 math and install it in both Ray workers

**Interfaces:**

```python
SAFE_FP16_OPTION = "minimax_h3_safe_fp16"
SAFE_FP16_LOADER_VALUE = "fp16_h3_safe"
```

- [x] Port the external condition-projection, residual, `out_proj` and MLP protections into `minimax_h3_fp16.py`, retaining attribution and power-of-two scale constants 64/256.
- [x] Mark patched classes/instances so worker reuse or checkpoint switching cannot wrap a forward method twice.
- [x] Map `RayUNETLoader.weight_dtype=fp16_h3_safe` to `dtype=torch.float16` plus `minimax_h3_safe_fp16=True`; keep generic `fp16` behavior unchanged.
- [x] In `RayWorker.load_unet`, consume the private flag and install the patch before calling `fsdp_load_diffusion_model`; reject non-V100/no-CUDA use unless an explicit test override is present.
- [x] Print one diagnostic line per rank containing model dtype, manual-cast dtype, safe-FP16 status and compute capability.
- [x] Run the Task 1 tests plus `tests/test_ray_worker_lifecycle.py` and `tests/test_minimax_h3_workflows.py`; require GREEN.
- [x] Commit the reviewed worker-safe patch and loader mode.

### Task 3: Prove FP8 FSDP and Turbo LoRA follow FP16 branch inputs

- [x] Add a V100 FP8 fallback test whose quantized storage remains FP8 while `fp8_linear_fallback_chunked` receives FP16 input and returns finite FP16 output.
- [x] Keep the existing FP32-compute test unchanged and add a separate safe-FP16 policy test; do not replace one with the other.
- [x] Add a LoRA sidecar test using BF16 up/down tensors and FP16 branch input; assert both sidecar matrix multiplications execute in FP16 and the result is finite.
- [x] Add an FSDP policy test proving safe FP16 does not accidentally enable the BF16-only `MixedPrecisionPolicy` and does not densify FP8 storage before all-gather.
- [x] Run `tests/test_minimax_h3_fsdp_quant.py`, `tests/test_minimax_h3_turbo.py`, `tests/test_fp8_fsdp_gather_shape.py` and `tests/test_fsdp_lora_streaming.py`; require GREEN.
- [x] Commit the reviewed FP8/FSDP/LoRA compatibility contracts.

### Task 4: Generate isolated experimental GUI workflows

- [x] Extend the workflow generator with a `compute_dtype` argument accepting `default` and `fp16_h3_safe`; reject safe FP16 outside full MiniMax H3 profiles.
- [x] Generate the two `_FP16_Experimental.json` files with the same prompts, seeds, inputs, dimensions, frames, Turbo LoRAs and 8/4 steps as O5; only the RayUNETLoader mode may differ.
- [x] Add byte-for-byte regeneration tests and SHA-256 guards proving all upstream, 20-step and O5 Turbo workflows remain unchanged.
- [x] Start ComfyUI for `/object_info`, convert both experimental GUI workflows to API prompts, verify 21 executable nodes, and return both GPUs to idle after the reduced smokes.
- [x] Commit the reviewed generator and experimental workflows.

### Task 5: Run staged CUDA validation

- [ ] Re-run the two-rank model-load probe after the reviewed LoRA projection fix and assert 684 FSDP wrappers per rank, FP8 checkpoint storage, CUDA P2P collectives and no host-staged tensor transport.
- [ ] Re-run the one-step reduced I2V smoke and record per-rank dtype/storage plus max-absolute and finite diagnostics.
- [ ] Re-run the matching reduced REF2VA smoke with the Turbo LoRA and the same diagnostics.
- [ ] Reject immediately on NaN/Inf, black output, rank mismatch, LoRA dtype mismatch, collective fallback or an FP32 attention/MLP branch.
- [ ] If smoke passes, run one cold full I2V Turbo 8 and one cold full REF2VA Turbo 4 with the accepted O5 inputs and settings.
- [ ] Record preprocessing, worker/model load, main sampling, sampler total, decode/write, VRAM, GPU utilization, physical/committed memory, pagefile and P2P traffic.

### Task 6: Compare, review and release locally

- [ ] Compare safe FP16 against the Task 0 matched FP32-compute baselines using end-to-end and stage times separately; publish measured values without substituting external or projected figures.
- [ ] Validate video dimensions/frame counts, unique-frame hashes, black detection, finite non-silent audio and same-seed visual behavior.
- [ ] Require `baseline stable s/it / safe-FP16 stable s/it >= 4.0` for both workflows as the initial gate; separately report progress toward the `11x` optimization target.
- [ ] Require model load, preprocessing, VAE decode and media-save time to be no slower than their matched baselines, and require improved end-to-end wall time.
- [ ] Run `py_compile`, the complete pytest suite, workflow hash guards and a code review with zero unresolved High/Medium findings.
- [ ] Update the O6 status, both READMEs and `docs/testing/minimax-h3/README.md`, then commit the complete consistent state.
- [ ] Keep safe FP16 opt-in until every correctness and performance gate passes.

## Current gate

Tasks 0-4 are implemented and independently reviewed with zero unresolved High/Medium/Low findings. Task 5 starts from a clean deployment commit and re-runs both reduced CUDA smokes before the full-resolution performance/quality runs.
