# Raylight 1.10.0 / O8: MMH3Director MODEL Proxy Compatibility Design

[English](2026-08-22-mmh3director-model-adapter-design.md) | [简体中文](2026-08-22-mmh3director-model-adapter-design.zh-CN.md)

## 1. Goal

Allow MiniMax H3 Director to delegate its internal `comfy.sample.sample()` call to the existing native-Windows dual-V100 Raylight FSDP + CUDA P2P workers without modifying `ComfyUI_MiniMaxH3_Director`, breaking accepted LTX/MiniMax workflows, or creating another repository or long-lived branch.

Director remains the sole generation control surface. Seed, steps, CFG, sampler, scheduler, video/audio shifts, task, prompt groups, frames, FPS, resolution, megapixels, continuity and export settings continue to come from Director. The new Raylight node only adapts `RAY_ACTORS` to the `MODEL` input expected by Director.

## 2. Release identity

- Target version: `1.10.0`.
- Phase: `O8`.
- Repository and branch remain `raylight-windows-v100-fsdp-p2p` / `windows-v100-fsdp-p2p`.
- This is an opt-in, backward-compatible integration, not a Director rewrite or a replacement for existing XFuser/FSDP sampling paths.
- Tag the accepted `1.9.0` state as `v1.9.0-pre-mmh3director` before implementation. A tag is a rollback coordinate, not a branch.
- The initial compatibility coordinate is `AIMixer/ComfyUI_MiniMaxH3_Director` `1.0.0`, commit `d655912262bcd70ad43c3a431879be9539d8e054`. Runtime support still uses capability checks because archive installations may have no Git metadata.

## 3. Node naming and contracts

### 3.1 New node

- Internal node ID: `RayMMH3DirectorModelAdapter`.
- Display name: `Ray MODEL Adapter (MMH3Director)`.
- Input: `ray_actors: RAY_ACTORS`.
- Output: `model: MODEL`.
- Category: `Raylight/integrations/MMH3Director`.

The requested spelling `MMH3Dirctor` omits the `e`. The public node name and documentation use the correct upstream project spelling `MMH3Director` to avoid a permanent API typo.

### 3.2 Frozen existing nodes

IDs, display names, input/output types, output order, defaults and workflow links remain unchanged for `RayInitializer`, `RayInitializerAdvanced`, `RayUNETLoader`, existing Ray samplers, `RayKill`, diagnostics and O6 nodes.

In particular, do not append a `MODEL` output to `RayUNETLoader`. One additional adapter node is the deliberate cost of keeping the accepted loader schema isolated.

### 3.3 Initializer naming rule

This design requires no Director-specific change to `RayInitializer`; validation and proxy construction belong to the adapter.

If implementation tests prove a Director-only initialization parameter is unavoidable, add a new internal ID `RayInitializerMMH3Director` displayed as `Ray Initializer (MMH3Director)`. Never add Director-only parameters or a Director suffix to the generic initializer.

## 4. Execution architecture

The accepted path remains:

```text
RayInitializer -> RayUNETLoader -> existing Ray guiders/schedulers/XFuser sampler -> existing worker methods
```

The new path is:

```text
RayInitializer
  -> RayUNETLoader
  -> Ray MODEL Adapter (MMH3Director)
  -> MiniMaxH3Director.model
  -> Director prepares conditioning/latent/noise/options
  -> comfy.sample.sample(RayMMH3DirectorModelProxy, ...)
  -> type-gated Raylight bridge
  -> exactly one distributed worker sample per Director pass
  -> sampled latent returns to Director
  -> Director continues VAE/audio/concatenation/save
```

The extension is split into a generic Ray MODEL proxy protocol, a type-gated `comfy.sample` bridge and the MMH3Director-specific request adapter. Future monolithic nodes receive additional adapters instead of model-name branches inside FSDP, P2P or sampler cores.

## 5. Proxy responsibilities

The proxy owns no diffusion weights. It stores the Ray actor handle, session/model identity, MiniMax sampling settings, latent-format metadata, model options, Sigma Shift object patch, lifecycle policy and diagnostics.

It implements the minimum interface actually exercised before sampling:

```python
class RayMMH3DirectorModelProxy:
    def clone(self) -> "RayMMH3DirectorModelProxy": ...
    def get_model_object(self, name: str): ...
    def add_object_patch(self, name: str, obj) -> None: ...
```

It also exposes `model.model_config`, `model.latent_format`, `model_options` and `load_device`. Workers export only compact metadata dictionaries; state dicts and model weights must never be copied to the main process.

## 6. Dispatch and failure policy

An idempotent wrapper preserves the original `comfy.sample.sample` and dispatches only exact Raylight proxy instances. All ordinary MODEL calls pass through with the original positional and keyword arguments.

Proxy incompatibility is fail-closed: emit an actionable error and never silently run the diffusion model locally. Raylight must still import and run when Director is absent, unsupported, installed but unused, or disabled by workflow topology. Optional Director discovery is lazy.

When Git identity is unavailable, validate the `MiniMaxH3Director` MODEL input, `sample_single_stage` signature and `comfy.sample.sample` call contract. Any mismatch fails closed.

The bridge consumes Director-prepared noise, latent, conditioning, steps, CFG, sampler, scheduler, denoise, mask, seed and callback. Rank results pass finite-value and rank-consistency checks before rank 0 latent is returned.

## 7. Parameter ownership

Director exclusively owns generation parameters and timeline semantics. Raylight owns only GPU count, FSDP/offload, Ulysses/ring/CFG topology, P2P, attention, mmap, diagnostics and worker lifecycle. O8 must not introduce duplicate seed, steps, FPS, resolution or prompt nodes.

## 8. Segments, refine, preview and lifecycle

- Workers are reused across segments by default.
- Director main-process cleanup and Ray worker cleanup remain separate.
- Existing `Reuse Workers` behavior remains the default; optional final destruction occurs only after the whole Director workflow.
- Same-model refine uses the same proxy and remains distributed. A separately wired ordinary refine MODEL is Director's native local path and must not be described as FSDP.
- Final media correctness is mandatory. If live TAE preview cannot safely cross Ray in the first release, report phase progress explicitly instead of pretending to provide step previews.
- A later progress relay polls worker state; callback closures are never serialized into Ray workers.

## 9. Compatibility and release gates

Regression coverage includes accepted LTX 2.3, MiniMax H3 I2V Turbo8, REF2VA Turbo4, O6 safe FP16 and FP32 controls, cold/warm reuse, checkpoint switching, P2P/FSDP preflight, cleanup, root import without Director, and Director installed but unused. Existing node schemas and workflow JSON are snapshot-tested; no migration is allowed.

New acceptance covers Director t2v/i2v/fl2v/r2v/v2v/rv2v modes, selected multi-segment execution, same-model refine, one remote invocation per pass, zero local diffusion calls for proxy passes, both-GPU compute, finite/rank-consistent latents, valid video/audio, exact requested dimensions/frame count/FPS/duration and verified reuse/destruction.

Old workflows run three matched warm trials and use the median. More than 5% unexplained sampling regression or unexplained VRAM/RAM/commit/pagefile growth blocks release. Director performance is compared to an existing Raylight workflow with the same model, resolution, frames, steps, prompt and inputs; the adapter must not cause a second model load or a second sample.

## 10. Documentation and release

After implementation and machine acceptance, update both READMEs, bilingual release notes, bilingual maintained testing summaries, example workflow and `pyproject.toml` version `1.10.0`.

The READMEs must retain the earlier branch's principal capabilities and boundaries: Windows dual-V100 TCC, sharded FSDP weights, CUDA P2P/NVLink, Gloo control plane, O6 safe FP16, original LTX/MiniMax workflows, RAM/pagefile behavior, launch settings and known limitations. The Director section is additive.

## 11. Non-goals

- Do not modify or fork Director.
- Do not sample once in Raylight and once in Director.
- Do not put full weights in the proxy.
- Do not claim that CLIP or Video/Audio VAE are FSDP-sharded.
- Do not equate Windows Gloo with NCCL.
- Do not implement O7 LTX safe FP16 as part of O8.
- Do not create a new repository or long-lived branch.
