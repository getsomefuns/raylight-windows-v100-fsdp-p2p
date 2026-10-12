# O15 MiniMaxH3Director 8151b29 Adaptation Plan

**Goal:** Upgrade the machine from upstream Director `576424f` to `8151b29`, retain all existing dual-V100 Raylight behavior, and make upstream per-segment LoRAs execute inside both FSDP workers.

**Current delta:** The machine is 25 commits behind. Upstream adds Refine canvas/keyframe realignment, FL2V selection and continuity fixes, lossless FFV1 segment cache, selected-segment export fixes, mixed-mode/shared-parameter UI changes, per-segment LoRAs, cache namespace customization, decoded-audio sample-rate preservation, and prefetch-queue cleanup.

**Compatibility assessment:** Most changes are internal to Director and require only a clean upstream fast-forward plus regression testing. The `sample_single_stage` keyword contract remains unchanged, but the newly expanded `refine_sampling` alias must remain bridged. Per-segment LoRA is the one incompatible feature: upstream expects a real ComfyUI `ModelPatcher`, while Raylight exposes a weightless main-process MODEL proxy.

**Implementation:**

1. Pin Raylight compatibility to upstream `8151b291fcacee5ddb70c6493eaed4f831bc6867` and test the `refine_sampling` imported alias.
2. Add a Ray-proxy-only wrapper for Director `apply_segment_loras`. It resolves and validates up to eight active LoRA paths but does not load weights in the main process.
3. Carry immutable `{path, strength_model}` descriptors through every first-pass, SelfLift, and Refine request.
4. Apply the stack to a temporary worker-side clone using Raylight's existing quantized/FSDP LoRA loader, then discard that clone so segment A cannot contaminate segment B or the next workflow.
5. Keep a two-entry worker CPU state-dict cache keyed by path/size/mtime and clear it on model reset, global-LoRA changes, and worker shutdown.
6. Preserve a verified `576424f` Git bundle, fast-forward the separate official Director checkout to `8151b29`, and keep it clean.
7. Run focused compatibility tests, the full suite, the dual-V100 P2P gate, the matched O14 workflow, and a real two-segment LoRA isolation workflow.
8. Release as Raylight `1.17.0` with bilingual release and test records, deploy from the committed source tree, and synchronize GitHub without editing ComfyUI or Director source.

**Acceptance:** Existing SelfLift/Refine/FaceRefine/preview/TE/SOL flows remain valid; per-segment LoRA runs on both ranks without accumulation; custom cache namespaces isolate two workflows; media and audio pass quality probes; source, deployment, and GitHub SHAs match.

The executable step-by-step TDD checklist is the Chinese companion document: `2026-10-12-mmh3director-8151b29-adaptation.zh-CN.md`.
