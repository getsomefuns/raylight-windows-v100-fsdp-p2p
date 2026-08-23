# Raylight 1.10.0 / O8 MMH3Director MODEL Adapter Implementation Plan

[简体中文](2026-08-22-mmh3director-model-adapter.zh-CN.md) | [English](2026-08-22-mmh3director-model-adapter.md)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add an opt-in `Ray MODEL Adapter (MMH3Director)` that lets the unmodified MiniMax H3 Director use the accepted Windows dual-V100 Raylight FSDP + CUDA P2P workers while preserving every existing node and workflow contract.

**Architecture:** Keep the accepted `RAY_ACTORS` path untouched and introduce a lightweight `MODEL` proxy, an exact-type-gated `comfy.sample.sample` bridge and an additive worker entry point for Director-prepared requests. The proxy carries metadata and actor handles, never weights; all ordinary MODEL calls pass through to the original sampler, and incompatible proxy calls fail closed instead of silently using local single-GPU sampling.

**Tech Stack:** Native Windows 23H2, two Tesla V100-SXM2-16GB in TCC mode, NVIDIA 577.00, Python 3.10.11, PyTorch 2.7.0+cu126, ComfyUI 0.31.0, Ray 2.57.0, Raylight FSDP2, xFuser/yunchang, custom CUDA P2P/NVLink data plane, Gloo control plane.

**Spec:** `docs/superpowers/specs/2026-08-22-mmh3director-model-adapter-design.md`

## Global Constraints

- Work only under `E:\ComfyUI-py310`; do not change NVIDIA driver 577.00 or any environment outside that tree.
- Use the existing repository and branch `raylight-windows-v100-fsdp-p2p` / `windows-v100-fsdp-p2p`; do not create another repository or long-lived branch.
- Target version is `1.10.0`, phase `O8`; tag the accepted start as `v1.9.0-pre-mmh3director` before code changes.
- Pin integration acceptance to Director `1.0.0` at `d655912262bcd70ad43c3a431879be9539d8e054`; archive installs without Git metadata must pass equivalent runtime signature checks.
- Do not modify or fork `ComfyUI_MiniMaxH3_Director`.
- Display the new node as `Ray MODEL Adapter (MMH3Director)` and use internal ID `RayMMH3DirectorModelAdapter`.
- Do not modify the public schema of `RayInitializer`, `RayInitializerAdvanced`, `RayUNETLoader`, existing samplers or existing workflows.
- Do not add a Director suffix to a generic node. Add `Ray Initializer (MMH3Director)` only if a failing integration test proves a Director-only initializer is required.
- Preserve the accepted Windows CUDA P2P, FSDP, O6 safe-FP16, LTX and MiniMax H3 paths.
- Never silently fall back from a Ray proxy request to main-process diffusion sampling.
- Keep Director optional: Raylight must import and run when Director is absent.
- Do not claim CLIP, Video VAE or Audio VAE are FSDP-sharded; they remain main-process stages.
- Use tests first, focused commits and matched real-machine regression evidence.

## Planned File Structure

- Create `src/raylight/integrations/__init__.py`: integration namespace only.
- Create `src/raylight/integrations/mmh3director/__init__.py`: exports node mappings and idempotent bridge installation.
- Create `src/raylight/integrations/mmh3director/metadata.py`: compact worker metadata schema and rank validation.
- Create `src/raylight/integrations/mmh3director/proxy.py`: lightweight `MODEL` facade and clone/object-patch behavior.
- Create `src/raylight/integrations/mmh3director/bridge.py`: exact-type sampler dispatch, request construction, result validation and final-only callback policy.
- Create `src/raylight/integrations/mmh3director/nodes.py`: `RayMMH3DirectorModelAdapter` ComfyUI node.
- Modify `src/raylight/distributed_worker/ray_worker.py`: additive metadata export, Director request sampler and progress/status fields.
- Modify root `__init__.py`: optional integration registration without importing Director.
- Create `tests/test_mmh3director_proxy.py`.
- Create `tests/test_mmh3director_bridge.py`.
- Create `tests/test_mmh3director_nodes.py`.
- Create `tests/test_mmh3director_worker.py`.
- Create `tests/test_mmh3director_optional_dependency.py`.
- Create `tests/test_mmh3director_regression_contracts.py`.
- Create `tests/test_mmh3director_workflow.py`.
- Create `tests/test_mmh3director_lifecycle.py`.
- Create `example_workflows/MMH3Director_Windows_V100_FSDP.json` and deploy its tested copy to `E:\ComfyUI-py310\ComfyUI\user\default\workflows\MMH3Director.json`.
- Create bilingual O8 release notes and maintained validation reports.
- Modify `README.md`, `README_EN.md` and `pyproject.toml` only after implementation and real-machine acceptance.

---

### Task 0: Freeze the accepted 1.9.0 state

**Files:**
- Verify: `README.md`
- Verify: `README_EN.md`
- Verify: `docs/testing/minimax-h3/SAFE_FP16_FSDP_2026-08.md`
- Verify: accepted workflow JSON files under `example_workflows/`

**Interfaces:**
- Consumes: current accepted branch HEAD and documented O6/LTX baselines.
- Produces: rollback tag `v1.9.0-pre-mmh3director` and a clean baseline manifest.

- [x] Run `git status --short` and stop if an unrelated user change overlaps files in this plan.
- [x] Run the current unit suite with `E:\ComfyUI-py310\Python310\python.exe -m pytest -q tests` and retain the concise pass/fail summary.
- [x] Run the P2P/FSDP verification scripts already documented in README and retain their summary without committing raw machine logs.
- [x] Record current node schemas and hashes of accepted workflow JSON files in the O8 validation report.
- [x] Tag the accepted commit using `git tag -a v1.9.0-pre-mmh3director -m "Accepted state before MMH3Director adapter"`.
- [x] Commit only the baseline manifest/report if it changed; do not modify runtime code in this task.

### Task 1: Lock backward compatibility with contract tests

**Files:**
- Create: `tests/test_mmh3director_regression_contracts.py`

**Interfaces:**
- Consumes: current node classes and accepted workflow files.
- Produces: frozen schema assertions used as a release gate.

- [x] Write a test that asserts exact `RETURN_TYPES`, `RETURN_NAMES`, `FUNCTION` and display mapping for `RayInitializer`, `RayInitializerAdvanced`, `RayUNETLoader`, `RayKill` and accepted samplers.
- [x] Write a test that loads every accepted LTX/MiniMax workflow JSON and asserts all stored node type IDs and link output indexes still resolve.
- [x] Write a test that asserts `RayUNETLoader.RETURN_TYPES == ("RAY_ACTORS",)` so a future implementation cannot append a proxy output.
- [x] Run `E:\ComfyUI-py310\Python310\python.exe -m pytest -q tests/test_mmh3director_regression_contracts.py` and require GREEN against the unchanged 1.9.0 code.
- [x] Commit the green compatibility fence. The new-node RED/GREEN cycle belongs to Task 3 so the branch never carries a knowingly failing commit.

### Task 2: Implement compact metadata and the weightless MODEL proxy

**Files:**
- Create: `src/raylight/integrations/__init__.py`
- Create: `src/raylight/integrations/mmh3director/metadata.py`
- Create: `src/raylight/integrations/mmh3director/proxy.py`
- Create: `tests/test_mmh3director_proxy.py`

**Interfaces:**
- Consumes: compact dictionaries returned by `RayWorker.get_mmh3director_proxy_metadata()` in Task 5.
- Produces:

```python
@dataclass(frozen=True)
class MMH3DirectorModelMetadata:
    model_family: str
    sampling_settings: dict[str, object]
    latent_channels: int
    latent_dimensions: int
    spacial_downscale_ratio: int
    temporal_downscale_ratio: int
    noise_scale: float | None

def validate_rank_metadata(rows: list[dict]) -> MMH3DirectorModelMetadata: ...

class RayMMH3DirectorModelProxy:
    def clone(self) -> "RayMMH3DirectorModelProxy": ...
    def get_model_object(self, name: str): ...
    def add_object_patch(self, name: str, obj) -> None: ...
```

- [x] Write tests proving rank metadata disagreement fails with rank/field detail and matching metadata becomes an immutable dataclass.
- [x] Write tests proving the proxy owns no `torch.nn.Parameter`, state dict or diffusion model and holds only actor handles plus compact metadata.
- [x] Write tests for independent clone semantics: cloned `model_options` and object patches must not mutate the parent proxy.
- [x] Write tests for `get_model_object("model_sampling")`, `get_model_object("latent_format")`, unknown object failure, and `add_object_patch`.
- [x] Write a real MiniMax Sigma Shift compatibility test using `MiniMaxH3SigmaShift.execute(proxy, 12.0, 3.0)`; assert the clone captures both shifts and returns a usable model sampling object.
- [x] Run the focused tests and confirm RED before creating implementation.
- [x] Implement metadata facades with `types.SimpleNamespace` only for fields used by ComfyUI; do not instantiate a diffusion model.
- [x] Run focused tests, require GREEN and commit.

### Task 3: Add the opt-in adapter node and optional registration

**Files:**
- Create: `src/raylight/integrations/mmh3director/nodes.py`
- Create: `src/raylight/integrations/mmh3director/__init__.py`
- Modify: `__init__.py`
- Create: `tests/test_mmh3director_nodes.py`
- Create: `tests/test_mmh3director_optional_dependency.py`

**Interfaces:**
- Consumes: `ray_actors: dict` with `workers` and Task 2 proxy/metadata validation.
- Produces:

```python
class RayMMH3DirectorModelAdapter:
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "adapt_model"

    def adapt_model(self, ray_actors: dict):
        return (RayMMH3DirectorModelProxy(...),)
```

- [x] Write schema tests for one `RAY_ACTORS` input, one `MODEL` output, category and exact public display name.
- [x] Write tests proving no initializer or loader schema changes when the integration is registered.
- [x] Write optional-dependency tests that block all Director imports and still import the Raylight root package successfully.
- [x] Write a fake-actor test in which the adapter requests metadata from all ranks and rejects a non-MiniMax model or unloaded actor.
- [x] Run focused tests to verify RED.
- [x] Implement node registration without importing any Director module; ComfyUI itself supplies the common MODEL/Sigma Shift interfaces.
- [x] Keep generic `RayInitializer` unchanged. If no failing test proves otherwise, do not add `Ray Initializer (MMH3Director)`.
- [x] Run focused tests plus `tests/test_root_package_import.py`; require GREEN and commit.

### Task 4: Install an idempotent, exact-type sampler bridge

**Files:**
- Create: `src/raylight/integrations/mmh3director/bridge.py`
- Create: `tests/test_mmh3director_bridge.py`

**Interfaces:**
- Consumes: `RayMMH3DirectorModelProxy` and the full positional/keyword signature of current `comfy.sample.sample`.
- Produces:

```python
def install_mmh3director_sample_bridge() -> bool: ...
def uninstall_mmh3director_sample_bridge_for_test() -> None: ...
def dispatch_mmh3director_sample(model, noise, steps, cfg, sampler_name,
                                 scheduler, positive, negative, latent_image,
                                 **kwargs) -> torch.Tensor: ...
```

- [x] Write a pass-through test using an ordinary sentinel MODEL; assert the original function is called once with object-identical args/kwargs.
- [x] Write idempotence tests proving two installs still create one wrapper and test-only uninstall restores the exact original function.
- [x] Write proxy tests proving the original local sampler is never called, including error cases.
- [x] Write request-shape tests covering prepared noise, denoise, noise_mask, seed, steps, CFG, sampler, scheduler, conditioning and latent.
- [x] Write fail-closed tests for unsupported model family, stale/dead actors, unsupported Director signature and missing rank results.
- [x] Assert the accepted Director `sample_single_stage` signature from commit `d655912262bcd70ad43c3a431879be9539d8e054`; simulate one removed and one renamed parameter and require an actionable compatibility error.
- [x] Write result tests for rank mismatch, NaN/Inf and output dtype/device normalization.
- [x] Run focused tests RED, implement the smallest wrapper and run GREEN.
- [x] Register the bridge once from the integration module and commit.

### Task 5: Add an isolated Director worker entry point

**Files:**
- Modify: `src/raylight/distributed_worker/ray_worker.py`
- Create: `tests/test_mmh3director_worker.py`

**Interfaces:**
- Consumes the serializable request emitted by Task 4.
- Produces:

```python
def get_mmh3director_proxy_metadata(self) -> dict: ...
def sample_mmh3director_request(self, request: dict) -> dict: ...
def get_mmh3director_sample_status(self, operation_id: str) -> dict: ...
```

- [x] Write worker tests proving metadata contains only model family, sampling settings, latent format and noise scale and contains no tensor storage/state dict.
- [x] Write a fake-model test proving the request applies `shift_video`/`shift_audio` on the worker model before sampling.
- [x] Write a test proving provided Director noise is used exactly and not regenerated from a different seed.
- [x] Write tests for noise mask, denoise, callback-disabled operation, result finite summary and operation ID.
- [x] Write lifecycle tests proving status is cleared after success/failure and existing `common_ksampler` signatures/behavior remain unchanged.
- [x] Run focused tests RED.
- [x] Implement the new method as an additive path; share private helpers only when existing sampler tests prove behavior equivalence.
- [x] Do not call the main-process original sampler and do not serialize Director callback closures to workers.
- [x] Run `tests/test_mmh3director_worker.py`, existing sampler tests, `tests/test_ray_worker_lifecycle.py` and `tests/test_ray_worker_shutdown.py`; require GREEN and commit.

### Task 6: Complete multi-segment, refine, progress and cleanup behavior

**Files:**
- Modify: `src/raylight/integrations/mmh3director/bridge.py`
- Modify: `src/raylight/integrations/mmh3director/proxy.py`
- Modify: `tests/test_mmh3director_bridge.py`
- Create: `tests/test_mmh3director_lifecycle.py`

**Interfaces:**
- Consumes: operation status from Task 5 and existing Raylight cleanup functions.
- Produces: one-operation-per-pass accounting and explicit reuse/finalize behavior.

- [x] Test a two-segment sequence: two bridge calls, two remote operations, one model load and the same worker PIDs.
- [x] Test same-model refine: first pass plus each refine pass is remote exactly once.
- [x] Test that an ordinary explicitly wired custom refine MODEL passes through to Director's native sampler and is reported as not Raylight FSDP.
- [x] Test `clear_vram_after_sampling=False/True` without changing current defaults.
- [x] Test final-only callback behavior: at completion invoke the provided callback at most once with the returned finite latent; never pretend to stream missing intermediate x0 values.
- [x] Test failure cleanup so a failed segment cannot leave an operation marked active or poison the next hot run.
- [x] Implement polling diagnostics with bounded intervals and no debug-log flood; normal mode prints one start and one completion line per pass.
- [x] Reuse existing Worker cleanup and `RayKill` behavior; add a finalizer only if the workflow test proves ordering cannot be expressed with existing nodes.
- [x] Run lifecycle, session-cache and shutdown suites; require GREEN and commit.

### Task 7: Build the Director workflow and API/system tests

**Files:**
- Create: `example_workflows/MMH3Director_Windows_V100_FSDP.json`
- Create: `tests/test_mmh3director_workflow.py`
- Deploy after review: `E:\ComfyUI-py310\ComfyUI\user\default\workflows\MMH3Director.json`

**Interfaces:**
- Consumes: existing Ray initializer/loader, new adapter, unmodified `MiniMaxH3Director`, existing VAE/audio/save nodes.
- Produces: minimal user-loadable workflow with Director as the only generation parameter panel.

- [x] Create a workflow containing existing Ray initializer and UNET loader, exactly one `Ray MODEL Adapter (MMH3Director)`, the upstream Director node, CLIP/Video VAE/Audio VAE loaders and output/save nodes.
- [x] Do not add duplicate seed, steps, sampler, scheduler, shift, frames, FPS, size, megapixels or prompt widgets outside Director.
- [x] Add workflow JSON tests for exact node types, adapter-to-Director MODEL link, absence of XFuser sampler nodes and retained Director parameters.
- [x] Convert the workflow to API format and assert one executable model path and valid dependencies.
- [x] Start ComfyUI and verify the workflow loads without missing nodes or type mismatch.
- [x] Run a reduced single-segment smoke test before full media tests; verify both workers sample and local diffusion count is zero.
- [x] Copy the reviewed JSON to the requested user workflow path and commit only the repository example.

### Task 8: Run real-machine functional and regression acceptance

**Files:**
- Create: `docs/testing/minimax-h3/MMH3DIRECTOR_ADAPTER_2026-08.md`
- Create: `docs/testing/minimax-h3/MMH3DIRECTOR_ADAPTER_2026-08.zh-CN.md`
- Add concise data rows: `docs/windows-v100-fsdp-test-results-2026-08.csv`

**Interfaces:**
- Consumes: Task 7 workflow and accepted matched baselines.
- Produces: release decision with concise evidence links.

- [x] Validate Director task modes t2v, i2v, fl2v, r2v, v2v and rv2v using reduced samples first; mark unsupported upstream modes accurately rather than fabricating success.
- [x] Do not run the upstream single-GPU 15-second segment in Direction 1: the 12-second result already reached 329.46 s/it and is operationally impractical despite avoiding OOM. Both 5-second, 24 FPS, 1280×704 boundaries for REF2VA Turbo4 and FL2VA Turbo8 are complete.
- [x] Directions 2 and 3 completed the 864×480, 24 FPS, 3-second×5 REF2VA Turbo4 and FL2VA Turbo8 runs. The two planned 5-second×5 runs were canceled by the user after the 3-second×5 runs already took 54:48 and 01:12:19 and established the required conclusions; canceled is recorded explicitly rather than passed. The upstream baseline test phase ended on 2026-08-23.
- [x] Validate the selected multi-segment matrix and same-model refine.
- [x] For each pass record model load, preprocessing, first/stable/aggregate s/it, sampling total, VAE decode, audio decode, concatenate/save and end-to-end wall time.
- [x] Record GPU0/GPU1 utilization, power and VRAM peaks, host RAM, commit/pagefile peaks, P2P status, rank completion delta and worker PIDs.
- [x] Validate finite latents, dimensions, frame count, FPS, duration, video temporal variation, audio duration and absence of black/noise/corrupt frames.
- [x] Re-run accepted LTX 2.3, MiniMax I2V Turbo8, REF2VA Turbo4, O6 safe-FP16 and FP32 controls.
- [x] Run three matched warm trials per old workflow, compare medians, and block release for unexplained sampling regression above 5% or unexplained memory growth.
- [x] Verify proxy passes have one remote operation per Director pass, zero local diffusion operations and no second model load.
- [x] Verify hot reuse and complete final cleanup separately.
- [x] Keep raw logs/telemetry/media local; commit concise tables, conclusions and paths only.

### Task 9: Update version, READMEs and release records

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `README_EN.md`
- Create: `docs/releases/2026-08-22-o8-mmh3director-adapter.zh-CN.md`
- Create: `docs/releases/2026-08-22-o8-mmh3director-adapter.en.md`

**Interfaces:**
- Consumes: accepted Task 8 facts only.
- Produces: synchronized public documentation for `1.10.0`.

- [x] Set `project.version = "1.10.0"` only after Task 8 passes.
- [x] Add a bilingual capability matrix distinguishing existing explicit Ray samplers from the new Director MODEL adapter path.
- [x] Add a user guide from startup through loading `MMH3Director.json`, configuring Director, choosing reuse/cleanup and verifying both GPUs.
- [x] Document that Director controls generation parameters while Raylight controls distributed settings.
- [x] Document main-process CLIP/VAE/audio boundaries, multi-segment RAM/pagefile behavior, same-model refine support, custom refine limitations and preview behavior.
- [x] Preserve all earlier README material about TCC, FSDP, CUDA P2P/NVLink, Gloo, O6 safe FP16, LTX/MiniMax workflows, launch profiles and known limitations.
- [x] Add bilingual release notes with implemented changes, compatibility guarantees, measured results, limitations and rollback tag.
- [x] Run a link/path/name parity check between Chinese and English docs.
- [x] Commit the documentation/version change separately.

### Task 10: Final verification and publish gate

**Files:**
- Review: all files changed since `v1.9.0-pre-mmh3director`.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: reviewed local release candidate ready for the existing GitHub repository.

- [x] Run the complete pytest suite and retain the final concise result.
- [x] Run Ruff/pre-commit as configured by the repository and fix only O8-related findings.
- [x] Run ComfyUI import/startup once with Director installed and once with Director discovery blocked.
- [x] Load and run one accepted old workflow and the new Director workflow in the same ComfyUI session.
- [x] Inspect `git diff --check`, `git status --short`, repository size and ignored artifacts; remove no user file without explicit authorization.
- [x] Review every changed runtime file for accidental generic-node schema changes, silent fallback, model-weight serialization or global side effects.
- [x] Do not push until code review, tests, real-machine acceptance, bilingual docs and README parity are all complete.

## Acceptance Summary

O8 is complete only when the new adapter is opt-in, Director is unmodified, old workflow JSON loads unchanged, ordinary MODEL sampling passes through exactly, proxy sampling is distributed exactly once per pass, both V100s compute through the accepted FSDP/P2P path, media is correct, cleanup/reuse is deterministic, old median performance does not regress beyond the gate, and both READMEs accurately describe old and new capabilities.
