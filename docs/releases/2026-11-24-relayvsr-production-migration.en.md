# RelayVSR production migration — release draft, acceptance in progress

This is a local execution draft, not a completed release. Production has not been upgraded and GitHub has not been synchronized. Passing isolated cases below do not imply that every migration gate has passed.

Detailed stage/resource tables, negative trials and measurement boundaries: [test report](../reports/2026-11-24-relayvsr-production-migration.en.md). The companion requirements file is the pinned candidate profile, not evidence of production acceptance.

## Changes

- Pin official ComfyUI `2fcb6ce`. Adapt PackedLayout, video/audio masks, per-token timestep modulation and native ModelPatcher `fast_disk` cloning in Raylight while retaining old-core compatibility. No hand-edited core source.
- With a native RelayVSR pack connected to a Ray-backed Director, distribute exact FP32 Flash windows across both V100s. Keyframe sampling remains native single-GPU; the VSR model itself is not FSDP-sharded.
- Lease the actual physical GPUs from existing H3 workers, strictly offload inactive H3 residency, clean Flash before returning the leases. Release the Flash pool by default without destroying H3 CPU patchers or Director caches.
- TE 3.3's old whole-model replacement rejects the new core. Ray MODEL now uses its native residual controller inside Raylight's mask-aware USP loop, with unanimous rank decisions and safe FP16 retained. Unconnected TE/Sol remain disabled.
- Identify final VSR caches by source checkpoint stamps and recognizable patch options. Opaque mutations cannot claim cross-execution reuse. Reject incompatible stale final frames during selective merged export without deleting first-pass AV caches. Unrelated native Director calls remain native.
- Fix integrated Flash cancellation, quarantine bypass through replacement workers and Ray-disconnect fallback to standalone admission. Unproven cleanup requires a ComfyUI restart, not just actor recycling.
- Separate workflow success from acceptance: missing/unhealthy host or either GPU monitor, or incomplete time coverage, fails acceptance. Recovery copies reject unreadable subtrees, corrupt copies, linked manifests and existing restore destinations.

## Evidence so far

- New core: `951 passed`; legacy: `937 passed, 14 skipped` for genuinely unavailable native interfaces; five frontend tests pass. The original 14 compatibility failures were fixed, not skipped.
- Isolated cases exercise Director Ref2VA/FL2VA 4/8-step, the original non-Director XFuser route, TP2/native CLIP, segment LoRA, SelfLift, cached refinement, actual FaceRefine on 56/56 frames, dual H3 VAE, silent/normal audio and real live-preview packets.
- Matched image/seed, 864×480,124 frames,24fps: old/new cold `319.887 / 310.911s`; warm `241.586 / 234.263s`. One pair is not a general stable speedup claim.
- Pure 480×864,240-frame,24fps,x2 workflows: `116.007 / 114.807 / 113.150s`; Flash `71.469 / 71.311 / 71.426s`. Median changes from the existing dual baseline are about `+4.0% / +0.09%`, inside the 10% non-regression gate.
- All 240 raw 480p/x2 frames and all 120 raw 1MP/x2 frames match the same-latent official single-GPU Flash output bitwise: `max_abs=0, RMSE=0`.

## Remaining release gates

1. All-frame 1MP/5s/x4 reference comparison and complete media integrity checks.
2. Three-segment H3→VSR→H3, same-settings selective rerun and changed-settings stale-final rejection on the reviewed fixes.
3. Verified backups, an actual restored Python/core/nodes workflow rehearsal, then production migration and smoke tests.
4. Final bilingual README, data attachments, version tag and GitHub release.

## Boundaries

Windows TCC, two 16GB V100s, Python3.12.10 and Torch2.12.0+cu126 only in this acceptance. No driver replacement or DynamicVRAM; no promise of automatic compatibility with future TE/Director/core versions. Existing Director overlays are retained. Old/new H3 controls are not pixel-identical (observed SSIM0.784984); neither functional success nor TE skip counts establish lossless quality. Same-latent VSR raw-pixel equality is a separate gate.
