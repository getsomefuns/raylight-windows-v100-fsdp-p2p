# MiniMax H3 Director Upstream Baseline (2026-08-22)

## Purpose

Establish the unmodified MiniMax H3 Director performance, memory, and media-quality boundary on one Windows Tesla V100 16GB before implementing `Ray MODEL Adapter (MMH3Director)`. Every generation was queued from a ComfyUI workflow in the UI; no Python submission script was used.

## Fixed conditions

- Windows, 2× Tesla V100-SXM2-16GB in TCC; strict single-GPU runs use `CUDA_VISIBLE_DEVICES=0`.
- ComfyUI `v0.31.0-15-g62b3c94b`, Python 3.10.11, PyTorch 2.7.0+cu126, xFormers 0.0.30.
- Launch flags: `--disable-cuda-malloc --lowvram`. The first normal-VRAM attempt OOMed in text encoding and is not a viable baseline.
- 16:9, 864×480 (~0.4MP), 24 FPS, 124 frames for the 5-second setting, seed fixed at 42.
- FL2VA: FP8-scaled model + Turbo8 LoRA, 8 steps.
- REF2VA: FP8-scaled model + Turbo4 LoRA, 4 steps, upstream green-robot reference image.
- Raw logs, telemetry, media, and contact sheets remain local under `<repo-root>/.local/artifacts/logs/minimax-h3/director-baseline` and ComfyUI output.

## Completed results

| Workflow | Output specification (resolution / actual duration / frames @ frame rate) | Status | End to end | Sampling | Mean step | GPU0 peak | GPU1 | RAM peak | Commit peak | Pagefile delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FL2VA Turbo8, normal VRAM | Target 864×480 / 5.167 s / 124 frames @ 24 FPS; OOM, no output | Text-encoder OOM | No sampling | — | — | ~16064 MiB | base context | ~56053 MiB | ~85720 MiB | none |
| FL2VA Turbo8, low VRAM | 864×480 / 5.167 s / 124 frames @ 24 FPS | Pass | 15:45 | 11:11 / 8 | 83.92 s/it | 15534 MiB | 347 MiB base context | ~64065 MiB | ~74405 MiB | none |
| REF2VA Turbo4, low VRAM, strict single GPU | 864×480 / 5.167 s / 124 frames @ 24 FPS | Pass | 11:56 | 06:02 / 4 | 90.73 s/it | 16120 MiB | 10 MiB, no process | ~66318 MiB | ~75387 MiB | none |
| REF2VA Turbo4, 2 seconds, warm | 864×480 / 2.333 s / 56 frames @ 24 FPS | Pass | 06:49.71 | 02:16 / 4 | 34.16 s/it | 15640 MiB | 10 MiB, no process | ~65286 MiB | ~74426 MiB | none |
| REF2VA Turbo4, 10 seconds, warm | 864×480 / 10.125 s / 243 frames @ 24 FPS | Pass | 21:55 | 16:03 / 4 | 240.80 s/it | 15283 MiB | 10 MiB, no process | ~71986 MiB | ~81905 MiB | ~+658 MiB |
| REF2VA Turbo4, 12 seconds, cold | 864×480 / 12.250 s / 294 frames @ 24 FPS | Pass | 28:44 | 21:57 / 4 | 329.46 s/it | 13088 MiB | 10 MiB, no process | ~68609 MiB | ~79354 MiB | none |
| REF2VA Turbo4, 720p-class 5 seconds, cold | 1280×704 / 5.167 s / 124 frames @ 24 FPS | Pass | 27:05 | 19:41 / 4 | 295.41 s/it | 12504 MiB | 10 MiB, no process | ~69386 MiB | ~79295 MiB | none |
| FL2VA Turbo8, 720p-class 5 seconds, cold | 1280×704 / 5.167 s / 124 frames @ 24 FPS | Pass (subject-count deviation) | 41:21 | 36:14 / 8 | 271.81 s/it | 15160 MiB | 10 MiB, no process | ~68577 MiB | ~79545 MiB | ~+670 MiB |
| REF2VA Turbo4, 480p, 3 seconds×5, cold | 864×480 / 14.584 s / 350 frames @ 24 FPS | Pass | 54:48 | 29:27 / 20 | 88.35 s/it | 16230 MiB | 10 MiB, no process | ~73752 MiB | unavailable (invalid counter) | 1409–1464 MiB, no growth |
| FL2VA Turbo8, 480p, 3 seconds×5, cold | 864×480 / 14.584 s / 350 frames @ 24 FPS | Pass (prompt-adherence deviation) | 01:12:19 | 53:30 / 40 | 80.29 s/it | 16117 MiB | 10 MiB, no process | ~73055 MiB | ~84548 MiB (OS virtual used) | 1685 MiB peak, ~+290 MiB net |

### Stage observations

- FL2VA cold loading took about 3:38; stable steps were 83.76–84.97 seconds; Video/Audio VAE plus save took about 56 seconds.
- REF2VA cold loading and reference preprocessing took about 5:05; stable steps were 90.60–91.73 seconds; Video/Audio VAE plus save took about 49 seconds.
- REF conditioning added about 586 MiB sampling VRAM and made sampling about 8.1% slower than FL2VA.
- The 2-second warm run used 56 frames and held 33.97–34.20 s/it; fixed model switching, VAE, and saving overhead was about 4:34.
- The 10-second warm run used 243 frames; its four steps took 239.52, 240.13, 240.77, and 241.09 seconds. The larger latent forced the diffusion model to `0 MiB loaded / 21592.91 MiB offloaded`, making sampling CPU-offload-bound and adding about 658 MiB of pagefile use.
- The 12-second cold run used 294 frames; four steps took 21:57 total, approximately 328.4–330.6 seconds each, without progressive slowdown. The diffusion model remained at `0 MiB loaded / 21592.91 MiB offloaded`; loading, conditioning, VAE, and save together accounted for about 6:47 outside sampling.
- The 720p-class REF2VA run used the H3-aligned 1280×704 canvas and 124 frames; four steps took 19:41 total, approximately 294.0–296.0 seconds each. The diffusion model was again fully offloaded; versus 90.73 s/it at 864×480, sampling was about 3.26× slower despite only 2.17× more pixels.
- The 720p-class FL2VA run used 1280×704 and 124 frames; eight steps took 36:14 with approximately 270.7–274.3 seconds per step. Sampling VRAM peaked at 15160 MiB, very close to the single-card capacity; versus 83.92 s/it at 480p, sampling was about 3.24× slower.
- REF2VA 3-second×5 segment sampling averaged 47.15, 98.71, 98.86, 98.83, and 98.92 s/it. Adding 22 latent video frames and 40 audio steps of continuity context made segment 2 about 2.09× slower than segment 1, while segments 2–5 did not progressively degrade. GPU0 averaged about 57.65% utilization and GPU1 remained at 0%.
- FL2VA 3-second×5 segment sampling averaged 40.54, 90.14, 90.26, 90.27, and 90.23 s/it. Continuous segments were about 2.22× slower than segment 1 but segments 2–5 remained stable. GPU0 averaged about 76.05% utilization and GPU1 remained at 0%.
- Both five-segment workflows requested 365 logical frames. Director phase alignment trimmed 5 frames at each of three internal transitions, producing 350-frame exports. Neither run OOMed, but their 54:48 and 01:12:19 wall times expose the substantial single-GPU cost of upstream continuity.

## Media acceptance

- Both outputs are 864×480, 24 FPS, 124 frames, 5.167 seconds, H.264 video with 32kHz stereo AAC audio.
- FL2VA preserved robots, green armor, the rainy neon street, and continuous forward motion without black, noisy, or frozen frames.
- REF2VA preserved the reference green robots, violet portal, and weapon appearance with continuous camera motion and no black, noisy, or frozen frames.
- Both audio tracks are non-silent: FL2VA mean/peak approximately -29.3/-10.9 dB; REF2VA approximately -19.4/-2.2 dB.
- The 10-second REF2VA output is 864×480, 24 FPS, 243 frames, and 10.125 seconds. Five-point frame inspection showed continuous subjects and scenery without black frames, noise, or structural collapse, and the file contains 32kHz stereo AAC audio.
- The 12-second REF2VA output is 864×480, 24 FPS, 294 frames, and 12.25 seconds. Six-point frame inspection was continuous without black frames, noise, or structural collapse; 32kHz stereo AAC measured about -14.0/-1.2 dB mean/peak.
- The 720p-class REF2VA output is 1280×704, 24 FPS, 124 frames, and 5.167 seconds. Five-point inspection preserved the robots, portal and city motion without black frames, noise or structural collapse; AAC measured about -16.4/-2.3 dB mean/peak.
- The 720p-class FL2VA output is 1280×704, 24 FPS, 124 frames, and 5.167 seconds. Five-point inspection showed continuous robots and city motion without black frames, noise or structural collapse; AAC measured about -18.2/-4.8 dB mean/peak. However, the prompt requested three robots while the output consistently showed two, so media integrity passed but subject-count adherence did not fully pass.
- The REF2VA 3-second×5 output is 864×480, 24 FPS, 350 frames, and 14.584 seconds. Frame inspection around the actual seams at frames 73, 141, 209, and 277 preserved subjects, camera, background, and motion direction without black frames, noise, or structural collapse; AAC measured about -14.9/-0.6 dB mean/peak.
- The FL2VA 3-second×5 output is 864×480, 24 FPS, 350 frames, and 14.584 seconds. All four seams were visually continuous without black frames, noise, or structural collapse; AAC measured about -14.0/-4.0 dB mean/peak. As a t2v path, robot count and motion drifted slightly, so prompt/subject consistency was weaker than reference-conditioned REF2VA.

## Local evidence index

| Run | Workflow | Output video | Telemetry | Seam sheet | Primary log |
|---|---|---|---|---|---|
| REF2VA Turbo4, 3 seconds×5 | `ComfyUI/user/default/workflows/MMH3DirctorTest/MMH3Director_REF2VA_Turbo4_480p_3s_x5_Continuity.json` | `ComfyUI/output/video/MMH3DirctorTest/multisegment/REF2VA_Turbo4_480p_3s_x5_00001_.mp4` | `logs/minimax-h3/director-baseline/REF2VA_Turbo4_480p_3s_x5_telemetry.csv` | `logs/minimax-h3/director-baseline/REF2VA_Turbo4_480p_3s_x5_seams.jpg` | `logs/minimax-h3/director-baseline/comfyui-single-gpu-ref4-3sx5.stderr.log` |
| FL2VA Turbo8, 3 seconds×5 | `ComfyUI/user/default/workflows/MMH3DirctorTest/MMH3Director_FL2VA_Turbo8_480p_3s_x5_Continuity.json` | `ComfyUI/output/video/MMH3DirctorTest/multisegment/FL2VA_Turbo8_480p_3s_x5_00001_.mp4` | `logs/minimax-h3/director-baseline/FL2VA_Turbo8_480p_3s_x5_telemetry.csv` | `logs/minimax-h3/director-baseline/FL2VA_Turbo8_480p_3s_x5_seams.jpg` | `logs/minimax-h3/director-baseline/comfyui-single-gpu-fl8-3sx5.stderr.log` |

The two 5-second×5 workflow JSON files remain available for an optional future rerun, but they have no corresponding result, media, or telemetry and must not be cited as validated data.

## Test-phase conclusion

- The 2-, 10-, and 12-second 0.4MP/24 FPS runs passed. The 12-second run already reached 329.46 s/it, so the user stopped the 15-second test as operationally meaningless even if it might avoid OOM.
- Direction 1 completed both 5-second, 24 FPS, 720p-class REF2VA Turbo4 and FL2VA Turbo8 runs at the H3-aligned 1280×704 canvas.
- Directions 2 and 3 completed 864×480, 3-second×5 pressure and seam runs for both REF2VA Turbo4 and FL2VA Turbo8. Both finished with continuous seams, while continuous segments cost about 2.09–2.22× the first-segment step time.
- The two planned 5-second×5 runs were not executed. Based on the 54:48 and 01:12:19 results already obtained from 3-second×5, the user determined that the required conclusions were established and further runtime was operationally unjustified. Their status is canceled, not passed.
- Compare the later Raylight adapter against these upstream results using the same seed, models, and Director parameters.
