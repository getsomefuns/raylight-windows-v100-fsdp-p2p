# Director aff2579 update: compatibility audit, not full feature acceptance

Independent review found two outstanding overlay issues: cache-status inspection
does not reconstruct FaceRefine/RelayVSR fingerprints (UI match state may be
wrong, while executor reuse still uses the real plan), and metadata reference
paths need explicit input-directory containment. Neither is covered by the
seven cache tests or fixed in this audit. Exported patches are recovery snapshots,
not a complete feature release, and must be regenerated after those fixes.

Portable patches were replayed on a clean `aff2579` archive; all ten changed
runtime files match the installed overlay by hash. Windows tar initially failed
on an upstream Unicode example filename; a complete standard-library extraction
and replay succeeded without changing the example.

Updated upstream Director from `4c6245e` to `aff2579`, keeping PR231/cache-fingerprint fixes (`131b714`) and the five previously modified files as a preserved local commit (`0b0daef`: export-tail continuity and face tracking). A complete source copy, original dirty patch and Git bundles are retained in ignored `.local/backups/director/20261124-aff2579/`. Portable overlays and maintenance instructions are in [patches/director](../../patches/director/README.md).

Raylight runtime remains 1.18.5. Results: full suite 830 passed / 35 subtests passed (six upstream warnings); PR231 regressions 7/7; Refine request dedup/abort/latest-result/cleanup checks passed; upstream audio tests 2/2; Refine JavaScript syntax passed. No full GPU generation, new video, end-to-end speed measurement or quality claim. Detailed evidence stays in `.local/artifacts/director-aff2579-20261124/`.

RelayVSR is a separate postprocessing model after Refine/FaceRefine, using an ordinary RelayVSR MODEL plus a matching Flash VAE. Do not connect the H3 Ray MODEL adapter as its model. This update does not provide RelayVSR FSDP. Installed ComfyUI `62b3c94b` lacks `RelayVSRConditioning`, `VSRDecode` and the model backend; weights are absent too. An authorized core upgrade and real V100 validation are required before claiming VSR works. No core files were changed.

The fetched AIMixer main commits contain no Pika-specific implementation. [Pika H3 Flex](https://huggingface.co/Pika-Labs/Pika-H3-Flex) requires the correct task-specific adapter and exact sampling contract, not simply six Euler steps. No Pika weights were found locally. A third-party Director Pro branch must not silently replace the official node.

Maintain the local Director branch with backed-up fetch/rebase, never Manager force-update/reinstall. Re-run regressions after each upstream change; portable patches do not guarantee automatic future conflict resolution. Existing workflows/caches were not altered. See the [implementation plan](../superpowers/plans/2026-11-24-director-pika-relayvsr.en.md) for outstanding acceptance gates.
