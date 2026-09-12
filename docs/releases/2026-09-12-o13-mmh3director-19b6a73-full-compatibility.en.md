# Raylight 1.15.0: Full MiniMaxH3Director 19b6a73 Compatibility

## Goal

This release moves the validated MiniMaxH3Director coordinate from `b8f721c` to `19b6a73` and closes four proxy gaps: Director-native Refine, continuity remasking, Refine spatial tiling, and step-wise live previews. Existing explicit Ray sampler workflows and ordinary ComfyUI MODEL objects remain unchanged.

## Shipped changes

- Extend the `sample_single_stage()` contract for `after_shift`, `enable_tiling`, `tile_count`, `tile_overlap`, and the new `shift_cache` argument.
- Serialize only the recognized `install_continue_prefix_remask` capability and reconstruct it on both real worker models. Unknown callbacks fail explicitly.
- Keep Director-native Refine on both workers. Spatial tiling installs Director's native wrapper on each real worker sampler and always restores it after errors.
- Send previews only from rank 0: at most 16 video-latent frames through a size-one non-blocking Ray queue into Director's original callback. Full queues drop stale intermediate events instead of blocking sampling; disabled preview creates no queue.
- Accept Director `19b6a73`'s `shift_cache`. It owns local MODEL clones, while real Raylight weights are remote, so the cache is intentionally not serialized across processes.
- Add regression coverage against the actually installed Director signature and report the validated commit in compatibility errors.

## Upstream Director coverage

The `b8f721c..19b6a73` range contains more than Refine and previews: mixed per-segment task modes, segment-export memory release, reference-audio slot fixes, custom-FPS fixes, guide-and-redraw/keep-full continuity, latent temporal tiling, reference-image long-edge presets, and the latest SigmaShift/dead-slot VRAM leak fix. Raylight does not copy or patch those features. Director retains planning, conditioning, decode, assembly, and export; Raylight takes over only when the proxy MODEL enters diffusion sampling.

## Performance and boundaries

- Spatial tiling is a transient-VRAM feature, not a speed feature. Untiled execution is normally faster when memory permits.
- Live preview adds a small rank-0-to-CPU latent copy plus main-process TAE/WebP work. Disabling preview removes that transport.
- Director's segment-cleanup improvement remains a main-process feature; Raylight still owns remote model, remask hook, worker reuse, and worker cleanup.
- A different checkpoint, different weights, or third-party sampling that bypasses `sample_single_stage()` and the legacy `comfy.sample.sample()` seam remains outside this guarantee.

## Verification

Automated coverage includes bridge routing, worker remask, spatial tiling, failure restoration, preview transport, the installed Director signature, and legacy routing. Dual-V100 evidence is maintained in the [O13 test record](../testing/minimax-h3/O13_MMH3DIRECTOR_19B6A73_2026-09.zh-CN.md).

