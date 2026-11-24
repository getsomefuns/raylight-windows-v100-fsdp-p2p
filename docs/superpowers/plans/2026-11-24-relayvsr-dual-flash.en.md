# RelayVSR Dual-V100 Flash Acceleration Implementation Plan

Final status2026-11-24: exact windows,explicit node,shared-process dual-GPU Flash and isolated
acceptance are complete. Actor/borrowing items below are retained implementation history and
were withdrawn after review,not delivered features. The spec's final revision and
[acceptance report](../../reports/2026-11-24-relayvsr-dual-flash.en.md) define the shipped scope.
No production deployment or remote push;14 pre-existing newer-core H3 failures still block
treating this as acceptance of a complete production-core upgrade.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add measurable dual-V100 RelayVSR Flash acceleration through this Raylight fork only.
**Architecture:** Keep native keyframe sampling; distribute exact independent Flash windows.
Use an explicit node and owned lightweight actors; borrow existing workers only when explicitly provided.
**Tech Stack:** Python3.12, Torch2.12/cu126, Ray, Windows/TCC, native ComfyUI RelayVSR.
**Spec:** ../specs/2026-11-24-relayvsr-dual-flash-design.md

## Global Constraints
- Only the canonical Raylight worktree; no second development tree.
- Deploy exclusively to .local/environments/relayvsr-v100/ComfyUI/custom_nodes/raylight.
- Never modify production ComfyUI/Python/Director/driver, ComfyUI core, or comfy-kitchen.
- FP32 Flash is the controlled baseline; preserve input, seed42, steps, scale, frame count and CRF16.
- Do not push GitHub or distribute models/media/credentials. Preserve existing H3/Director/LTX behavior.

## Review Focus
1. Endpoint/history errors: real small Flash model, 1/16/17/31-frame comparisons.
2. Out-of-order/duplicate/missing frames: strict merge coverage and shape tests.
3. x2/x4 metadata or latent mismatch: fail visibly rather than rescale or silently fall back.
4. Ray ownership, occupied GPUs, cancellation and failure: never steal or kill unrelated workers.
5. Older ComfyUI without RelayVSR: safe import and unchanged Wan/H3/LTX dispatch.

## Task 1: Exact windows and numerical contract
**Files:** src/raylight/relayvsr/{__init__,planning}.py; tests/test_relayvsr_windows.py.
**Interfaces:** plan_relay_windows(frame_count:int,interval:int=15)->tuple[RelayWindow,...];
merge_window_result(output,written,job,frames)->None.
RelayWindow fields: index,pixel_start,pixel_stop,latent_start,latent_stop,discard_prefix.
- [ ] Write independent literal boundary expectations for1/16/17/31/240 frames; 240 has16 windows and exact output coverage.
- [ ] Observe RED before implementation.
- [ ] Implement planning/CPU merging; reject empty, duplicate, malformed or out-of-range results.
- [ ] Real reduced official RelayFlash CPU-model x2/x4 tests: full decode equals windowed decode/merge.
- [ ] Run focused and bare pytest; preserve logs and commit locally.

## Task 2: Decoder and resource ownership
**Files:** src/raylight/relayvsr/{worker,runtime}.py; optional methods in distributed_worker/ray_worker.py;
tests/test_relayvsr_runtime.py.
**Interfaces:** RelayFlashDecoder(vae_path,dtype='fp32').decode_window(latents,images,detail);
run_parallel_flash(vae_path,samples,images,detail,worker_count=2,keep_workers=False,
borrowed_workers=(),interrupt_check=None)->Tensor.
- [ ] RED: scale/latent mismatch, ordering, failure without partial output, cancellation and ownership cleanup.
- [ ] Implement batch1 validation, original VAE calls and CPU results.
- [ ] One GPU per dedicated Ray actor; one in-flight window per actor; no full-output result accumulation.
- [ ] Own cluster: num_cpus=2; store=256MiB-rounded max(1GiB,2*(largest window input+output)+256MiB),
  capped at min(8GiB,25% available RAM). Reject insufficient shared-store capacity before submitting.
  Test a480p/x4 single result larger than1GiB. Use isolated temp/runtime paths.
- [ ] Occupied external GPUs fail promptly; borrowed workers use a separate VSR cache and retain other models.
- [ ] Default releases owned actors/cluster; keep_workers is explicit; test all exit paths.
- [ ] GREEN, complete pytest, local commit. No upstream edits.

## Task 3: Explicit workflow node and dispatch guard
**Files:** relayvsr/nodes.py, root__init__.py, distributed_modules/usp.py, tests/test_relayvsr_nodes.py.
**Interfaces:** RayRelayVSRDecode(samples,images,vae_name,detail,worker_count,keep_workers,ray_actors=None)->IMAGE.
- [ ] RED tests node contracts, older-core import safety and RelayVSR-only USP rejection.
- [ ] Replace native VAELoader+VSRDecode with one explicit node; retain native sampling and a one-worker control mode.
- [ ] Reject generic Wan USP for RelayVSR clearly; do not pretend a specialized USP forward is implemented.
- [ ] Existing node IDs and non-Relay routes unchanged; full pytest and local commit.

## Task 4: Isolated hardware acceptance and documentation
**Files:** .local/artifacts/relayvsr-flash-dev-20261124/;
example_workflows/RelayVSR_DualV100_Raylight.json;
docs/reports/2026-11-24-relayvsr-dual-flash.{zh-CN,en}.md; README/CHANGELOG.
- [ ] Commit tested source, then deploy only to isolated core with deployment verification; preserve previous isolated deployment.
- [ ] Real480p/31-frame raw comparison: max_abs<=1e-5, RMSE<=1e-6.
- [ ] Actual ComfyUI480p/240-frame/x2 graph with --cache-none and unchanged seed/inputs;
  fresh sampler/decoder/save events and window logs required per run.
  Test default-release cold pool and explicit keep_workers cold/warm pool runs.
- [ ] Rerun a matched cache-none single-GPU control; compare against the faster historical/matched baseline.
- [ ] FP32 Flash node including cold pool startup<=88.848s; server execution<=146.786s.
- [ ] Record whole-card GPU0/1 peaks/utilization/power every0.5s plus process/system RAM/commit/pagefile.
- [ ] Exact240 frames/24fps/960x1728, full decode, baseline SSIM>=0.9999 and PSNR>=50dB, seam checks.
- [ ] Short x4 and incomplete-tail regression; mark unexecuted long scenarios explicitly.
- [ ] Bare pytest, code review, old-node import regression; name every failure.
- [ ] Bilingual results and docs, local commits only. No production deployment or GitHub push.

Execute inline as requested. Record decisions/deviations and actual evidence in this plan's ledger.
A functional-only result is not performance acceptance; never relax thresholds after observing results.

## Evidence-driven Task4 revision

Ray cold Flash108.537s/workflow149.240s missed both gates. Add an explicit
execution_mode=local_threads|ray_actors, defaulting to a shared-process two-GPU decoder pool.
Keep native FP32 Flash/window math; avoid process startup and tensor serialization for the
default local path. Use explicit devices and invoke only the official Flash model in the
parallel loop, not concurrent global ComfyUI model scheduling. Ray remains an explicit option.
Write real reduced Flash equivalence and node-routing RED tests first; repeat raw31-frame and
full240-frame cold/warm tests. A retained pool must drop unused CUDA allocator workspace after
decoding so the next native sampler does not OOM. Retain all failed data; thresholds unchanged.

Final review scope ruling: ship only the measured shared-process local decoder. The Ray actor
prototype missed the cold gate and has shared-cluster ownership/locality boundaries beyond this
iteration. Retain its commits and measurements, remove the public actor/borrowing route and
never create or shut down a Ray cluster in the delivered decoder. Reject an already-connected
Ray session on every call, including warm reuse. Use one operation guard before model unload;
detach/cleanup owned model cycles, release completed Futures immediately, invalidate failed warm
pools and honor the actual VAE output-processing callback. Verify all with RED->GREEN regressions.

