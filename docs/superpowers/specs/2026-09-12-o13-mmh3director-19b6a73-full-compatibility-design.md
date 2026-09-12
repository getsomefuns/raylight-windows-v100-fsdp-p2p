# O13 MiniMaxH3 Director 19b6a73 Full Compatibility Design

## Goal

Upgrade MiniMaxH3 Director from `b8f721c` to `19b6a73` while preserving Raylight's Windows dual-V100 FSDP/P2P path, SOL Attention, TE acceleration, warm-worker reuse, and legacy workflows. Cover Director first-pass sampling, Director-native Refine, continuity remasking, Refine spatial tiling, step-wise animated WebP previews, and the new `shift_cache` call contract.

## Boundaries

- Modify only `E:\ComfyUI-py310\raylight-windows-v100-fsdp-p2p`; never hand-edit the deployed copy.
- Fast-forward the independent Director repository without carrying a patch there.
- Do not modify ComfyUI core, SOL Attention, TE-Speed, or other custom nodes.
- Keep the `d655912` and `b8f721c` Director contracts accepted; validate `19b6a73` as the current target.
- Spatial tiling is a VRAM feature and is not promised to outperform the untiled path.

## Architecture

The bridge accepts `after_shift`, `enable_tiling`, `tile_count`, `tile_overlap`, and `shift_cache`. It serializes only a recognized `install_continue_prefix_remask` capability instead of a Python closure. Each worker reconstructs that hook on the real MODEL with the Director latent metadata and identical sigma schedule. Director's `shift_cache` owns main-process MODEL clones; Raylight's real weights live in remote workers, so the bridge accepts but does not serialize that local cache and preserves Raylight's worker lifecycle.

Spatial tiling is installed on the real sampler inside each worker. Both GPUs cooperate on every tile through the existing FSDP/P2P/USP path. The existing `comfy.sample.sample` path remains unchanged when tiling is disabled.

For live previews, rank 0 extracts at most 16 video-latent frames per eligible step, copies them to CPU, and publishes them to a size-one non-blocking Ray queue. The driver drains events while waiting for worker completion and invokes Director's original preview callback. Rank 1 emits nothing; disabled preview creates no transport.

## Acceptance

- Strictly validate the new Director contract without changing normal-model routing.
- Run first pass and Director-native Refine on both workers; fail closed on unknown `after_shift` callbacks.
- Apply continuity remasking on both actual worker models.
- Complete a four-step two-tile Refine sample with finite output and no visible tile seam.
- Deliver live step events before completion and render the looping WebP; disabled preview has zero transport.
- Preserve O11/O12 workflow and node contracts and keep one reusable worker pair.
- Deploy a byte-matching copy from the committed canonical worktree and pin Director at `19b6a73`.
