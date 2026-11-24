# 1.19.0 — official-core compatibility and Director dual-V100 RelayVSR

Status: isolated acceptance passed; production activation/smoke and GitHub release pending. This candidate record is not yet a completed production deployment.

## Shipped scope

- Pin official ComfyUI `2fcb6ce` (0.46.0), frontend1.59.11,kitchen0.2.42,aimdo0.6.0,templates0.11.118,docs0.5.17. Keep Python3.12.10,Torch2.12.0+cu126,driver577.00 and existing Director `aff2579`+local overlays unchanged. No hand-edited ComfyUI or Director source.
- Support old/new H3 PackedLayout, audio/video denoise masks, per-token modulation and `fast_disk` patcher cloning. Preserve safe FP16 and existing H3 FSDP/P2P behavior.
- A native RelayVSR pack connected to Ray-backed Director uses native single-GPU keyframe sampling and exact FP32 dual-V100 Flash windows. VSR itself is not FSDP-sharded; it is not a32GB pooled device or Windows NCCL.
- Lease both actual physical GPUs from existing H3 workers, strictly offload inactive residency, clean Flash, verify allocator restoration, then return leases. Keep H3 CPU patchers/actors and Director caches; release Flash GPU pools by default.
- Temporarily remove only native `max_split_size_mb` within VSR, restoring the complete live policy afterward. Director scopes sampling+Flash; standalone scopes Flash only. H3 policies,precision,environment and P2P128MiB budgets stay unchanged. Uncertain cleanup/restoration quarantines subsequent GPU work until restart, including waiting threads or replaced actors.
- TE3.3 uses its native residual controller inside mask-aware USP with unanimous rank decisions instead of applying its incompatible old whole-model forward. Native unrelated TE and workflows without TE remain unchanged.
- VSR final-cache identity includes model/VAE source stamps and recognizable patch settings. Reject incompatible stale finals during selective export without deleting usable first-pass AV caches. UI honestly labels unverified VSR final matches because the upstream cache-status API omits the full wired pack.
- Add cancellation/ownership guards, strict telemetry acceptance and verified new-directory-only runtime snapshot/recovery tooling. No second source worktree or permanent service.

## Evidence and limits

Detailed cases,failed trials,stage times,resource measurements and numerical scope: [English report](../reports/2026-11-24-relayvsr-production-migration.en.md) / [中文](../reports/2026-11-24-relayvsr-production-migration.zh-CN.md).

- Existing Director Ref2VA/FL2VA4/8,original non-Director XFuser,TP2/native CLIP,segment LoRA,SelfLift,cached Refine,FaceRefine56/56frames,dual H3 VAE,silent/normal audio and actual live-preview packets have real isolated workflow records.
- Matched864×480/124f/24fps5.1667s controls: old/new cold319.887/310.911s; warm241.586/234.263s. One pair is not a stable general speedup. Old/new H3 pixels differ (SSIM0.784984); no broad lossless/aesthetic-equivalence claim. Cold host memory increased; warm values are close.
- Post-fix480×864/240f/24fps10s/x2 pure total112.913/112.084/111.362s; logged Flash71.371/71.357/71.278s. Median total+1.50%,Flash essentially unchanged versus the already optimized dual baseline, inside the original10% gate.
- Reviewed-code three-segment VSR/continuity,stale-final rejection before sampling and same-settings selective rerun pass. Native minimum/keep-tail produces141frames24fps5.875s; the cold functional run overlapped recovery copying and is not a matched speed comparison.
- Complete240frame480p/x2 and120frame1MP/x2/x4 raw outputs are finiteFP32 and bitwise identical to independent official same-latent Flash. Post-fix1MP/x4 Flash538.205s versus historical534.571s; complete media checks pass. Serial new/old suites964/950passed,14legacy-native absences,35subtests both; frontend5passed.
-72,115file recoverable runtime snapshot plus297file user-state snapshot. A newly restored old Python/core/all-node runtime and its actual Ray workers generated a valid124frame video. This is a real recovery rehearsal,not only a package list or untested copy.

Native-reference diagnostic totals and memory include additional CPU/reference images,not pure production throughput. Whole-board NVML peaks are distinct from Torch allocated/reserved peaks; system commit is not physical RAM or measured paging traffic.

## Usage and migration

Use the unchanged environment-root `start_raylight_p2p.ps1`,then load [Director + VSR UI example](../../example_workflows/Minimax_H3_Director_DualV100_RelayVSR.json). The tested candidate uses4GiB reserve; the launcher's default3GiB is not changed. Match x2/x4UNet and FlashVAE; example settings are fixed42,H3Turbo4/4steps,864×480/56f24fps,VSR1step/euler/simple/CFG1/denoise1/detail1. Models belong in diffusion_models/vae. See the bilingual README for full wiring and standalone usage.

The migration upgrades official core and only exact offline companion wheels; it must not replace Torch,drivers,other ComfyUI environments,Director overlays,user media or segment caches. Verify the immutable Raylight deployment and run both an old Director workflow and the actual UI-derived VSR graph before release. Strong browser refresh is required after frontend upgrade.

Recovery backups and the ready old-runtime rehearsal are retained below `.local/backups/production-relayvsr-20261124` and `.local/environments/production-relayvsr-recovery-rehearsal`. Stop the owned server/workers before rollback; restore the recorded old core62b3c94,Python/dependencies and deployed nodec403c91 from these copies. Preserve failed state first. Do not blindly replace user workflows/media/caches; restore a changed database/configuration only if required. Keep the original standalone VSR isolation for manual reference.

## Explicit exclusions

Validated hardware scope is two16GBV100s,TCC,NVLink,Windows. DynamicVRAM,PDD/multi-head Flash,arbitrary third-party CUDA concurrency,arbitrary GPU counts,full480p/10s/x4 and arbitrary long/high-resolution workloads are not accepted here. Pika remains generation-unvalidated. LTX compatibility tests remain, but this migration does not rerun a full LTX GPU campaign. Future core/Director/TE versions require fresh compatibility checks.
