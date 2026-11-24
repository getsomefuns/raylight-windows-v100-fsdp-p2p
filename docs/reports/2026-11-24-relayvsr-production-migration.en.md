# Official-core migration / Director + dual-V100 RelayVSR regression

Status: **acceptance in progress; not a production-release claim**. Date: 2026-11-24.
Chinese: [中文](2026-11-24-relayvsr-production-migration.zh-CN.md).

## Purpose and controls

Upgrade the official core without losing existing Raylight Director behavior, then enable native RelayVSR with exact dual-GPU Flash decoding. No hand-edited core or Director source; retain the existing Director overlays. Formal deployment, restore rehearsal and GitHub synchronization are separate final gates.

Two TCC V100-SXM2-16GB GPUs, driver577.00, Python3.12.10, Torch2.12.0+cu126, xFormers0.0.35, Ray2.57.0. Old core62b3c94 / deployed Raylightc403c91; pinned candidate core2fcb6ce. Candidate companion packages: kitchen0.2.42, aimdo0.6.0, frontend1.59.11, templates0.11.118, docs0.5.17. Director officialaff2579 plus preserved local overlays13eba63. DynamicVRAM and cudaMallocAsync remain disabled.

Actual ComfyUI workflow graphs perform every sample; this is not a direct Python-model timing substitute. Seed42. Paired Director controls use the same public source image, prompt and API hashes. Forced warm replays execute real sampling while preserving workers, not output-cache-only hits. Media are checked with ffprobe and full ffmpeg decoding; first/middle/end contacts supplement, not replace, numerical gates.

## Paired Director control

REF2VA Turbo4,864×480,124frames,24fps,5.1667s; TP2 and safe FP16, no TE/Sol. Seconds below are core prompt time; conditioning includes native preprocessing and model preparation. Sampling is the Director stage including preparation, **not** isolated per-step kernel time.

| Runtime | Total s | Conditioning s | Sampling stage s | H3 decode s | SaveVideo s |
|---|---:|---:|---:|---:|---:|
| Old cold |319.887|40.4|202.4|16.2|5.910|
| Candidate cold |310.911|41.4|192.6|15.8|6.205|
| Old warm |241.586|37.3|177.6|16.5|5.667|
| Candidate warm |234.263|38.8|168.5|15.9|5.977|

Cold/warm totals are2.8%/3.0% shorter in this one pair, not a stable/general speedup claim. Old/new H3 pixels are **not identical**: observed SSIM0.784984,PSNR21.998968; subjects and composition remain recognizable without black/noise output. This does not establish broad aesthetic equivalence. Exact raw equality below applies to VSR decoding from the same sampled latent, not to old/new H3 generation.

| Runtime | GPU0/1 MiB | Process-tree RSS GiB | Private commit GiB | System RAM GiB | System commit GiB |
|---|---:|---:|---:|---:|---:|
| Old cold |9223/9201|52.06|92.46|64.86|114.79|
| Candidate cold |9223/9201|57.28|96.19|70.11|118.53|
| Old warm |9311/9289|43.96|59.94|56.26|82.24|
| Candidate warm |9301/9279|43.92|59.77|56.27|82.10|

The candidate cold run used more host memory; warm values are close. Do not claim a memory reduction. Process-tree sums can double-count shared pages. System commit is reserved backing, not physical RAM or measured pagefile traffic.

## Representative existing-feature workflows

All rows below target864×480,56frames,24fps,2.3333s unless stated. These are representative combinations, not every possible parameter permutation.

| Case | Total s | Conditioning / sampling / H3 decode s | Evidence |
|---|---:|---|---|
| REF2VA4, TE+Sol, SelfLift, segment LoRA |247.265|41.0 /142.9 /6.7|Two real low/high sample operations; first-pass cache saved. `confirm_first_pass` intentionally blocks final export.|
| Same graph queued again, Refine+FaceRefine |225.316|38.8 /54.2 /8.9|First pass reused;2-step refinement plus113.5s FaceRefine,56/56 faces,512×512,2steps. Final video saved.|
| REF2VA8, TE+Sol |278.432|39.2 /189.4 /9.2|Both ranks7full+1cached forward and38skipped blocks; rank0 one consensus veto.|
| Original non-Director XFuser REF2VA4 |168.258|Native CLIP29.484; XFuser node73.374; VAE9.564|Original native CLIP topology retained; real sample and video saved.|
| FL2VA4, silent output |287.841|56.8 /164.6 /10.3|Playable video with clip-aligned silent AAC track.|
| FL2VA8, TE+Sol |305.609|54.4 /199.2 /11.2|Playable normal-audio output.|

Actual live-preview traffic was observed: first-pass feature run15binary-preview packets+2Director preview events; Refine/FaceRefine run10+1. Missing optional taeh3 uses the same Latent2RGB fallback in both environments. TE cache hits prove routing, not stable acceleration or lossless output.

Before final safety review, three VSR-connected segments declared56+56+56frames with22frame continuity and keep-tail enabled rendered the native56+68+68=192frames,8s24fps. VSR preserved these counts and audio. Selecting only segment2 started one two-rank sample, reused the other cached segments, and completed324.530s. Idle board memory1438/1394MiB was retained contexts/staging, not a kept Flash pool. A shorter three-segment follow-up on the reviewed cache/lease fixes remains a separate gate.

## Standalone VSR and exactness

480×864,240frames,24fps,10s,x2→960×1728. Every row executes native KSampler and dual Flash, with default pool release.

| Pure dual workflow | Total s | Flash s |
|---|---:|---:|
| Repeat1 |116.007|71.469|
| Repeat2 |114.807|71.311|
| Repeat3 |113.150|71.426|
| Median |114.807|71.426|
| Historical dual baseline |110.428|71.363|

Median total+3.97%,Flash+0.088%: within the10% non-regression budget. No new acceleration over the already optimized dual baseline is claimed.

- 480p/x2: all240raw frames bitwise equal to official single-GPU Flash from the **same full latent**; max_abs0,RMSE0.
- 768×1376,120frames,24fps,5s,x2→1536×2752: all120frames bitwise equal; max_abs0,RMSE0. Dual Flash116.020s versus historical115.793s.
- Same1MP5s/x4→3072×5504: full reference verification in progress. Mixed-history dual Flash771.779s versus historical fresh534.571s; higher GPU1 residency observed. **Not accepted as non-regressing.** Fresh-process allocator/history controls are pending; normal clocks and a single owned GPU process exclude obvious foreign load, not every environmental cause.

Reference-check durations145.170/274.481s include loading, native decode and CPU comparison. Raw-verification case total/host peaks include retaining extra reference frames; do not present them as pure production throughput or memory cost.

## Failed trials and fixes, retained rather than hidden

-New-core baseline had14compatibility failures: PackedLayout, denoise masks and per-token modulation. Raylight now follows native semantics while retaining old-core behavior and safe FP16. No failures were converted into skips.
-First real H3 workflow exposed native `ModelPatcher.clone(fast_disk=...)`; both custom patchers now accept the contract. Rejected legacy initialization no longer causes destructor warnings.
-Compiled TE3.3 rejected the new whole-model forward signature. Do not disable that guard and run stale math. Ray MODEL uses the native residual controller inside Raylight's mask-aware USP loop, with both-rank consensus before cached updates. Unrelated native TE behavior is unchanged.
-A non-Director trial failed validation because the test fixture inserted TP2 while its initializer depended on conditioning. Restoring the original native CLIP topology resolved it; no runtime bug or successful sample is claimed for the rejected fixture.
-An early frontend export raced the initial default-workflow load. That export was rejected; the published example was re-exported only after verifying the Director graph.
-Independent review found7important safety gaps: quarantine bypass, physical-GPU/free-space validation, incompatible stale VSR finals, checkpoint/options identity, cancellation, unreadable backup subtrees and monitor false acceptance. Each received failing tests then fixes. Cleanup uncertainty quarantines GPU work until restart; first-pass cache is not deleted.
-Final suites: new951passed/35subtests; legacy937passed/14genuinely unavailable native-interface skips/35subtests; frontend5passed. Early wrong interpreter setup and missing Node VM flag failures were retained and corrected, not suppressed.

## Measurement and release gates

500ms host+whole-board samples, including both GPU health/coverage, not occasional visual observation or Torch active memory alone. RSS/private/system RAM/system commit/pagefile estimate remain separate. GPU numbers are observed MiB peaks, not guaranteed sub-500ms maxima. A workflow can succeed yet fail telemetry acceptance. Older14traces passed retrospective coverage checks; their unrecorded monitor exit status is explicitly unknown.

Still required: x4 root-cause controls and full raw match; final reviewed three-segment/selective/stale-cache tests; verified backup+actual restored workflow; formal migration+smoke; final bilingual usage/version documents and GitHub release. No production acceptance is implied by this draft.
