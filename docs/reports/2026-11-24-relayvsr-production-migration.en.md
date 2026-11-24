# Official-core migration / Director + dual-V100 RelayVSR regression

Status: **isolated acceptance passed; production migration/smoke pending**. Date: 2026-11-24.
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
- Same1MP5s/x4→3072×5504: all120raw frames also matched bitwise,max_abs0,RMSE0 before the allocator fix. Mixed-history dual Flash771.779s versus historical fresh534.571s was **not accepted as non-regressing**. Its full diagnostic took2456.399s, including1497.611s reference/check. Post-fix full gates remain pending.

Reference-check durations145.170/274.481s include loading, native decode and CPU comparison. Raw-verification case total/host peaks include retaining extra reference frames; do not present them as pure production throughput or memory cost.

### Allocator regression: measured A/B/A, not a weight-size guess

Fresh processes, identical1MP31-frame24fps/x4 workflow, seed/models/code/backend flags unchanged. The recovery-file copy overlapped these **diagnostic** controls; they are not the final uncontended throughput gate.

| Native split policy | Total s | Flash s | Observed GPU0/1 MiB |
|---|---:|---:|---:|
|128MiB limit, first |250.033|193.936|16223/16235|
|Default, unlimited split |173.921|135.619|15085/7725|
|128MiB limit, reverse |249.251|194.224|16237/16235|

GPU1 Flash peak allocated memory stays about5.666/5.675GiB, but reserved peak grows from7.145GiB to15.457GiB with the limit. This is allocator retention/split policy, not doubled model weights or the separate128MiB CUDA P2P staging budget. Native API probing proved complete restoration of the original live configuration, including garbage-collection and bracketed rounding options.

The Raylight fix temporarily removes only `max_split_size_mb` within VSR and verifies restoration. H3 worker policy, environment variables, precision and P2P capacity are unchanged. Director scopes native VSR sampling+Flash; standalone scopes **Flash only**, since its native KSampler already ran. Thus173.921s is not a promised fixed standalone total. Restore uncertainty quarantines the process, retains GPU leases, and refuses lock waiters until restart. New final suites963passed/legacy949passed14legitimate skips,35subtests both; focused44passed. Renewed GPU/cache/raw/performance gates still required.

### Reviewed-code integrated cache/recovery follow-up

Source `5eae991` completed three Director→VSR segments: native minimum length expands the declared3×24frames to39+51+51=141frames with22frame continuity and keep-tail,1728×960,24fps,5.875s. Total924.840s; conditioning46.0/46.7/47.5s, H3 sampling162.2/182.4/184.5s, H3 decode6.9/10.6/10.5s, VSR43.5/35.6/37.1s. Recovery disk copying overlapped this **functional** case; it is not a matched throughput comparison.

Changing detail1→0.85 was rejected before remote sampling and left every cache-file hash unchanged. Restoring the same settings and selecting only segment2 completed320.786s with one actual two-rank sample; conditioning45.4/H3 sample186.5/decode11.1/VSR36.5s. Both141frame outputs fully decode with audio. Idle boards1238/1194MiB; Flash pools released, H3 CPU caches retained.

The recoverable snapshot contains72,115verified runtime files, plus a separate297file user-state backup. A newly restored old Python/core/all-node runtime ran a real124frame Director generation and proved all Ray worker executables came from the restored Python. Its371.664s is a **functional recovery result**, not a matched performance control. Backup integrity is rechecked against the unchanged live production files before activation. No model, input, output or segment-cache directory is replaced.

The public UI example is also checked through the actual1.59.11 frontend, not only JSON topology: fixed42,H3four steps,56frames864×480/24fps,VSRfixed42/one step. Formal smoke uses that frontend-converted graph. Incorrect seed0/25step/randomize widget defaults in the initial export were caught and corrected before release.

### Final allocator-fix matrix

| Pure480p/10s/x2 repeat | Total s | Reported Flash s | Full decoder-node s |
|---|---:|---:|---:|
|1|112.913|71.371|71.515|
|2|112.084|71.357|71.488|
|3|111.362|71.278|71.399|
|Median|112.084|71.357|71.488|

Relative to the historical110.428/71.363s control, total median is+1.50%; same reported-Flash timing is essentially unchanged(-0.008%). Both meet the unchanged10% gate. Decoder-node duration additionally includes its boundary overhead, so it is listed separately. These three pure timings have no overlapping recovery copy or reference decoder.
The renewed480p/x2 and1MP/x2 raw checks passed all240/120frames bitwise,max_abs0,RMSE0,finiteFP32;1MP/x2 dual Flash115.864s. The reference independently selects native default allocator policy and restores it; it does not call Raylight's allocator helper.
The post-fix1MP/x4 full120frame check also passes bitwise,max_abs0,RMSE0,finiteFP32. Flash538.205s versus historical534.571s(+0.68%); independent reference/check1105.705s. Its1836.749s diagnostic includes both decoders,CPU comparison and save,not a pure dual-production timing. Read-only backup verification overlapped this diagnostic. The native standalone KSampler remains outside the scope and took128.513s; do not claim the Flash fix accelerates that earlier node.

Resource scope example: final480p repeat2 whole-workflow GPU0/1 peaks16209/5989MiB versus Flash-only6015/5989MiB; RSS7.79/private10.84/systemRAM25.91/systemcommit32.37GiB. The unwrapped earlier native standalone KSampler, not Flash, dominates GPU0. Final1MP/x2 before reference peaks16116/6336MiB (Flash11800/6336),RSS9.75/private14.49/systemRAM27.87/systemcommit36.03GiB. The reviewed three-segment integrated case observes9353/9321MiB and56.52/100.68/71.91/124.82GiB respectively, with recovery-copy overlap. These are different workloads,not memory-speedup pairs. Full utilization,power,paging estimates and phase-separated peaks belong in the compact data attachment.

## Failed trials and fixes, retained rather than hidden

- New-core baseline had14compatibility failures: PackedLayout, denoise masks and per-token modulation. Raylight now follows native semantics while retaining old-core behavior and safe FP16. No failures were converted into skips.
- First real H3 workflow exposed native `ModelPatcher.clone(fast_disk=...)`; both custom patchers now accept the contract. Rejected legacy initialization no longer causes destructor warnings.
- Compiled TE3.3 rejected the new whole-model forward signature. Do not disable that guard and run stale math. Ray MODEL uses the native residual controller inside Raylight's mask-aware USP loop, with both-rank consensus before cached updates. Unrelated native TE behavior is unchanged.
- A non-Director trial failed validation because the test fixture inserted TP2 while its initializer depended on conditioning. Restoring the original native CLIP topology resolved it; no runtime bug or successful sample is claimed for the rejected fixture.
- An early frontend export raced the initial default-workflow load. That export was rejected; the published example was re-exported only after verifying the Director graph.
- Independent review found7important safety gaps: quarantine bypass, physical-GPU/free-space validation, incompatible stale VSR finals, checkpoint/options identity, cancellation, unreadable backup subtrees and monitor false acceptance. Each received failing tests then fixes. Cleanup uncertainty quarantines GPU work until restart; first-pass cache is not deleted.
- Pre-allocator review suites: new951passed/35subtests; legacy937passed/14genuinely unavailable native-interface skips/35subtests; frontend5passed. Early wrong interpreter setup and missing Node VM flag failures were retained and corrected, not suppressed.

Additional retained trials: a private GPU probe initially imported before Raylight discovery and produced no allocator-detail log; explicit deployed-src resolution corrected the probe only. The whole workflow timings were retained. The new allocator review found2important restoration/quarantine races, reproduced by failing tests and fixed before deployment. A full-suite invocation overlapped native reference decoding and9GPU attention tests failed allocating2MiB with only3.25MiB free; that run is not accepted. Serial repetition after shutdown replaces it,without skips. The private final coordinator initially read the recorder's intermediate pending state,and then encountered normal venv-supervisor auto-exit after stopping its child; both private orchestration races were corrected without changing product code. All failed logs remain private evidence,not green runs.

## Measurement and release gates

500ms host+whole-board samples, including both GPU health/coverage, not occasional visual observation or Torch active memory alone. RSS/private/system RAM/system commit/pagefile estimate remain separate. GPU numbers are observed MiB peaks, not guaranteed sub-500ms maxima. A workflow can succeed yet fail telemetry acceptance. Older14traces passed retrospective coverage checks; their unrecorded monitor exit status is explicitly unknown.

Final isolated gates passed:964new-core tests,950legacy tests/14genuinely unavailable native-interface skips,35subtests both,and5frontend tests. All six renewed VSR videos fully decode with their original240/120frames at24fps; raw480p/x2,1MP/x2/x4 are bitwise equal. The declared public UI parameters were verified through the actual frontend. Backup/live hashes match72,115runtime files and297user-state files. [Compact data attachment](data/2026-11-24-relayvsr-production-migration.json) contains timings,geometry,utilization,power and separate memory/commit/phase peaks without raw logs or user media.

Still required: formal migration+both workflow smokes,final version documents and GitHub release. Production acceptance is not yet claimed.
