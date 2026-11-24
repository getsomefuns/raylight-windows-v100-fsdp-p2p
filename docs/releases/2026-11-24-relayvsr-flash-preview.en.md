# Unreleased: isolated dual-V100 RelayVSR Flash preview

Historical isolated-preview record. Its production/core limitations describe that earlier experiment; see the [subsequent migration record](2026-11-24-relayvsr-production-migration.en.md) for current status.

Isolation only. Production ComfyUI/Python/Director/deployed nodes remain unchanged. No GitHub push
or production version bump.

- New RelayVSR Flash Decode (Ray) node keeps native single-GPU keyframe sampling and distributes
  exact independent FP32 Flash windows across two GPUs.
- Original history,overlap warm-up and frame order remain unchanged. Merge writes contiguous CPU images.
- Default execution shares one process; no actor/cluster startup. Owned decoder weights are released by default.
- Explicit weight retention always trims idle workspace. Failure/interrupt cleanup never returns partial output.
- Regression coverage includes admission lock,foreign Ray-session rejection,deterministic patcher cleanup,
  Future lifetime,failed-pool recreation and the actual VAE output-processing callback.
- A specific RelayVSR/Wan USP rejection does not alter ordinary Wan/H3/LTX handlers or older node IDs.
- A silent benchmark API graph is included without source media or model weights.

The Ray actor prototype missed the cold gate and requires additional shared-cluster ownership
engineering; it and actor borrowing are not shipped. Failures,fixes,time and resource comparisons
remain in the [measured report](../reports/2026-11-24-relayvsr-dual-flash.en.md).

Production lacks native RelayVSR. Updating Raylight alone cannot add core operators. Fourteen
pre-existing H3 failures in the newer isolated core remain documented; this is not acceptance of
a complete production-core upgrade.
