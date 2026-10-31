# Raylight 1.18.5: MiniMax H3 compressed time-curve dtype compatibility

## Fix

Curve checkpoints can retain a BF16/FP16 `adaln_t_table` after loading, while USP interpolation coordinates and coefficients are FP32. `torch.lerp` then raises a coefficient (`weight`) dtype mismatch.

Only Raylight's `use_adaln_curves` branch changes: obtain a temporary FP32 table using the existing `cast_to` helper. Coordinates, interpolation and modulation input remain FP32. The checkpoint buffer, quantized weights, safe-FP16 policy, attention, sharding and caches are unchanged. The legacy `time_embedder` branch is untouched. No ComfyUI, Director, model-file or other-node changes are required.

## Auxiliary-tensor audit

The failing `10Eros_Max_h3_TURBO-hybrid_beta3_int8_convrot_skip_edges.safetensors` contains a BF16 table of shape `[1025,8]`, `adaln_basis [8,2688]`, `adaln_mean [2688]`, and `silu_t_emb_grid [1025,2688]`. First-block and final AdaLN weights both consume 8 coordinates. Reconstructing `table @ basis + mean` gives a relative L2 difference of approximately 0.6414% against the original grid. This is a pre-existing checkpoint relationship, not an error introduced by the fix or a video-quality measurement.

The three auxiliary tensors are not parameters of the current runtime architecture; expanding back to 2688 coordinates would not fit its 8-input projections. Existing unexpected-key warnings remain visible. No unknown tensors are silently discarded by new code. Auxiliary storage totals about 5.30 MiB; the temporary FP32 table is only 32,800 bytes, not a model-wide FP32 conversion.

## Verification and limits

- Windows, Python 3.12.10, PyTorch 2.12.0+cu126, two V100-SXM2-16GB GPUs.
- Red baseline: 6 failed / 4 passed, reproducing the BF16/FP16 mismatch.
- New tests: 13 passed, covering curve dtypes/endpoints/fractional coordinates, unchanged buffers, legacy embeddings, ignored auxiliaries and a small forward on each V100. The fixture has no Transformer blocks and is not a full distributed-generation test.
- Targeted USP/safe-FP16/FSDP-quant regressions: 57 passed, 2 subtests passed.
- Full suite: 830 passed, 35 subtests passed; six upstream deprecation/prototype warnings. The initial sandbox run hit temporary-directory permissions; an isolated-directory rerun passed without changing application behavior.
- Real checkpoint table and first-block modulation weights tested on each V100: finite outputs, maximum absolute error 0 against CPU FP32 reference.
- No full user-video rerun, video-quality claim, end-to-end speed claim or workflow-wide memory claim. Detailed evidence stays in ignored `.local/artifacts/adaln-curves-20261031/`, outside deployment and publication.

## Deployment

No workflow or launch-argument changes. Fully restart ComfyUI when safe so backend modules and Ray workers use the deployed code; browser refresh alone is insufficient. Director segment caches need not be deleted. This change does not automatically restart ComfyUI or push GitHub.
