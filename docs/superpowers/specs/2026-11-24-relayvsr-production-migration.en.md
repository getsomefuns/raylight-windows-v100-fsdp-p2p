# RelayVSR production migration assessment / proposed design

Date: 2026-11-24. Status: proposal, not approved for execution; no production upgrade or deployment performed.

## Goal and constraints

Use the isolated dual-V100 RelayVSR implementation in production Director while preserving existing workflows, audio, segment caches, selective reruns and subsequent H3 sampling. Source changes belong only to this Raylight repository. Do not edit ComfyUI core source, replace Director with a fork, change the GPU driver or touch the external ComfyUI bundle. Updating to a pinned official core is a separate decision requiring approval; this assessment is not that approval.

## Verified baseline

- Production core: `62b3c94bd45154f6486c7abf1b9efcacee96ea69`; deployed Raylight: `c403c91fad6daf5b80a7ac6954da95c23209ca6c`.
- Tested isolated core: `2fcb6cef9ab1b6dc556b7cff1765a0724ad5e884`; source Raylight: `720a258a4351381e095c7b86c871b368a1576942`.
- Python 3.12.10 / Torch 2.12.0+cu126 are shared through a system-site-packages venv. Isolated kitchen/aimdo/frontend are 0.2.42/0.6.0/1.59.11 versus recorded production 0.2.30/0.4.13/1.48.7. Re-inventory actual packages before execution.
- Production Director: `13eba63`, official `aff2579` plus preserved local overlays. Production core and Director tracked files were clean during read-only inspection.
- [Official native RelayVSR introduction](https://github.com/Comfy-Org/ComfyUI/commit/2f8943324e1eae89a5ee6c7db93602299246b5f1) changes ten files, not only node registration.
- [Existing acceptance report](../../reports/2026-11-24-relayvsr-dual-flash.en.md) and [exact test failures/data](../../reports/data/2026-11-24-relayvsr-validation.json).

## Blocking integration differences

1. The production core has no native RelayVSR model detection, loading or Flash support. Copying the Raylight decoder node alone is insufficient. RelayVSR sampling remains native single-GPU; only Flash windows are dual-GPU accelerated.
2. The isolated newer core has 14 known failing tests: 13 reach the old PackedLayout(frame_count=...) constructor and one forward-contract test exposes missing video/audio denoise-mask parameters. Their semantic effects, including per-row timestep modulation and continuity/refine, must be adapted in Raylight, not hidden by changed tests or skipped cases. New-core 865 passed/14 failed/1 skipped versus legacy 869 passed/11 skipped, both with 35 subtests.
3. The existing local Flash executor rejects any initialized Ray session. Director retains H3 workers, and its native VSR stage still calls VSRDecode, not our dual decoder. Isolated standalone success does not verify integrated Director acceleration.

## Approaches

**A — recommended:** pin the already tested official core, update its required companion packages in a production-candidate isolation, add backward/forward H3 compatibility and a scoped Director VSR bridge in Raylight. One UI, native loaders, in-process frames, no permanent external service. Requires official-core upgrade approval and full old-workflow regression. Keep DynamicVRAM disabled on V100. Earlier core candidates may be considered only with recorded compatibility evidence, not silently substituted.

**B — fallback:** freeze the production core and invoke VSR through a separate isolated backend. Reduces core-upgrade risk but introduces process/cancellation lifecycle, frame transfer, temporary storage, cache and GPU arbitration concerns. Lossy MP4 handoff cannot claim the existing exact-pixel equivalence. This alternative needs a separate concrete design if A is rejected or fails.

Vendoring the entire upstream model/detection/VAE stack into Raylight is not recommended: it creates a second core implementation to maintain.

## Proposed integration

- No VSR connection means no VSR weights, leases, sampler changes, fingerprint changes or output-path changes.
- Preserve the official Director VSR pack parameters. Scope the bridge to Director calls backed by our Ray MODEL proxy; leave native single-GPU workflows untouched. Suggested module: integrations/mmh3director/relay_vsr.py, not a global VAE.decode replacement.
- H3 sample/decode/refine/face-refine → acquire both existing workers' GPU leases → offload inactive H3/TE/VAE GPU residency → native RelayVSR sampling → existing exact dual-GPU FP32 Flash windows → release VSR GPU weights/scratch → return leases → export/next segment.
- Keep Ray sessions, H3 CPU patchers and segment caches. Do not use ray.shutdown or actor destruction as cleanup. Account for H3 reload on the next segment in end-to-end measurements.
- TP2 lease coordination is reusable groundwork, not ready-made VSR admission. Validate driver/session/generation, physical GPU mapping, exclusive ownership, worker state and actual released memory. Existing clear_sampling_vram can swallow cleanup errors; its True result alone is insufficient evidence.
- Default keep_workers=false. Preserve refusal for unknown/foreign Ray sessions. On cancellation/OOM/worker loss, finish owned cleanup before returning leases; do not resume the next segment after unresolved cleanup failure. Do not reintroduce the withdrawn actor-Flash prototype.

## Required acceptance gates (all pending)

1. Full suites on old and pinned new cores, including all 14 failures fixed without skip/xfail; numerical native-reference checks for masks, modulation, layouts, curves and continuity.
2. Real ComfyUI workflow graphs: non-Director H3, Director FL2VA/Ref2VA 4/8-step, TP2, regular/segment LoRA, SelfLift, Refine, FaceRefine and dual H3 VAE. Cover previews, silent/normal audio, cache hits and selective reruns. Representative combinations suffice; do not claim an exhaustive matrix.
3. No-VSR fingerprints unchanged. VSR setting changes invalidate final output while retaining usable first-pass caches. Status checks must not decode FFV1 or falsely claim a match.
4. At least three integrated segments plus selective reruns and H3→VSR→H3 switching: no deadlock, stale output, persistent GPU pools or accumulating idle resources.
5. Standalone 480p/10s/x2 and 1MP/5s/x2/x4 regressions; do not mark untested full 480p/10s/x4 as accepted. All raw Flash output frames match the same-latent single-GPU reference; complete outputs decode with original frame count/order/audio timing and rate.
6. Paired cold/warm stage timing and 500ms whole-card GPU/host/process resource telemetry, retaining the distinction between observed physical peaks and Torch active memory. Report TE, H3 sample/decode, VSR sample/Flash, saves and total time separately.
7. Existing standalone reference: default-release Flash 71.363s, total 110.428s, GPU0/1 observed peaks 11249/5991MiB. Retest three matched runs: median standalone runtime must not regress over 10%. Measure Director offload/reload and retained contexts separately; standalone budgets do not prove integrated budgets. Investigate old-workflow regressions over 5% rather than declaring no impact.
8. Before production changes, preserve core changes/untracked files, recoverable Python/dependency state, Director bundle/patches, Raylight deployment manifest and startup configuration. Rehearse rollback in isolation; do not touch user media/caches.
9. Deploy the exact accepted commit only after queue shutdown and backup; verify hashes and dependencies, smoke-test old Director and VSR in production, update bilingual README/release/testing records and reproducible workflows, then sync GitHub under authorization. Never publish models, user media, logs or backups.

This task only produced an assessment. Confirm approach A's official-core-upgrade boundary before a task-level implementation plan and isolated development. Keep the existing standalone VSR environment for manual checks/reference; it is not another Raylight source worktree.
