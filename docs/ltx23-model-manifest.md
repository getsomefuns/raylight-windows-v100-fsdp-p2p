# LTX 2.3 validation workflow manifest

The example workflow records model-relative paths only. Model weights and the
development input image are intentionally not redistributed by this repository.
Place equivalent files in the ComfyUI model directories or update the loader
nodes in the workflow.

| Loader | Relative value recorded in the workflow |
|---|---|
| DualCLIPLoader | `gemma/gemma_3_12B_it_fp8_e4m3fn.safetensors` |
| DualCLIPLoader | `LTX/ltx-2.3_text_projection_bf16.safetensors` |
| VAELoader | `LTX/LTX23_audio_vae_bf16.safetensors` |
| VAELoader | `LTX/LTX23_video_vae_bf16.safetensors` |
| RayUNETLoader | `LTX2.3/ltx-2.3-22b-distilled_transformer_only_fp8_scaled.safetensors` |
| RayLoraLoader | `LTX2.3/ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors` |
| LatentUpscaleModelLoader | `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` |
| LoadImage | `LTX2_3_i2v_Raylight.jpg` (not included; select your own input) |

Besides this Raylight fork and the pinned ComfyUI commit, the recorded workflow
uses the following custom-node revisions:

| Custom node | Repository | Validated commit |
|---|---|---|
| ComfyUI-Easy-Use | `https://github.com/yolain/ComfyUI-Easy-Use.git` | `595e0738a9e3f8d0d9c4d875461b2d2c9e7559c7` |
| ComfyUI_LayerStyle | `https://github.com/chflame163/ComfyUI_LayerStyle.git` | `64f976fec8492ea4930c0e30c32369573189b23d` |

The repository does not claim compatibility with arbitrary versions of the
developer machine's other custom nodes.

Model licenses and download terms are controlled by their respective authors.
Users are responsible for obtaining the files from authorized sources.
