# Raylight 1.16.0: MiniMaxH3Director 576424f and SelfLift

## Upgrade scope

This release moves the validated MiniMaxH3Director coordinate from `19b6a73` to upstream `576424f`. Neither Director nor ComfyUI source is modified. Raylight changes only its MODEL proxy, remote sampling protocol, tests, and example workflows.

The 13 upstream commits include SelfLift, Semantic Bridge, FaceRefine, pre-Refine VRAM cleanup, external-group cache fixes, true reference-video duration, and independent Refine seed/aspect fixes. Raylight now bridges `on_step_state`, `zero_noise`, and `noise_override`, so the SelfLift low-resolution prefix, high-resolution tail, and subsequent Refine all execute on the same dual-V100 FSDP worker pair.

## Critical fix: SelfLift audio scale

The first O14 adapter produced correct video but broadband-noise audio. SOL, workflow controls, safe FP16, Audio VAE, and muxing were ruled out. The weightless Ray MODEL facade did not mirror `comfy.model_base.MiniMaxH3.process_latent_in/process_latent_out`.

MiniMax H3 carries the second AV stream (audio latent) on the video flow schedule using `shift_video / shift_audio`, which is `12 / 3 = 4` by default. The real model multiplies audio by four on entry and divides it by four on exit. SelfLift performs its transition against the main-process proxy; missing this transform sent the audio state into the high-resolution stage at the wrong scale, while leaving video visually valid.

Version 1.16.0 mirrors the real AV semantics in the facade. Video stays unchanged, audio is multiplied/divided by `audio_scale`, and the round trip is exact. No model weights, collective calls, or staging allocation are added.

## Hardware results

The fixed case is REF2VA, 864x480, 124 frames at 24 fps, Turbo4, SOL, SelfLift 0.5, Director Refine 2, and dual-V100 FSDP/P2P.

| Metric | Before fix | After fix |
|---|---:|---:|
| End to end | 277.71 s | 288.55 s |
| Total remote sampling | 133.13 s | 143.83 s |
| GPU peaks | 9223 / 9201 MiB | 9351 / 9329 MiB |
| Physical RAM peak | 59883.91 MiB | 59355.73 MiB |
| Commit peak | 104328.43 MiB | 105183.29 MiB |
| Audio DC offset (L/R) | +0.016429 / +0.020130 | -0.000056 / -0.000043 |
| Audio zero-crossing (L/R) | 0.122354 / 0.132005 | 0.047671 / 0.063260 |

The fixed output is 864x480, 24 fps, 124 video frames, 5.1667 seconds of video, and 5.1670 seconds of audio. All 124 frame hashes are unique and audio contains zero NaN/Inf samples. The timing difference is between independent cold runs; the fix itself adds no collective or model forward.

On the strict non-SelfLift upgrade baseline, moving Director from `52f8fb7` to `576424f` changes end-to-end time from 314.05 to 288.51 seconds and sampling from 165.75 to 150.11 seconds on the same O13 workflow. This matched result is not attributed to one upstream commit.

## Attribution controls

- Disabling SOL preserves the defect, ruling SOL out.
- Removing Refine reduces but does not remove the abnormal DC offset; Refine amplifies an already damaged first pass.
- Safe FP16 and default precision both reproduce the defect. Default precision increases sampling from 77.70 to 156.66 seconds, ruling out the safe-FP16 patch.
- SelfLift scale 1.0 falls back to one Ray sampling stage and restores normal audio statistics, isolating the low-to-high state transition.
- An unmodified upstream single-GPU SelfLift control has clean audio but takes about 648 seconds end to end; it is attribution evidence, not the recommended path.

## Boundaries

- SelfLift, Director Refine, SOL, live previews, continuity remask, and legacy workflows continue through the same explicit Ray MODEL proxy. Ordinary ComfyUI MODEL objects remain unchanged.
- Semantic Bridge and FaceRefine have contract coverage. Optional local weights were unavailable, so this release does not claim their hardware quality/performance acceptance.
- O14 workflows are under `example_workflows/o14/`. Files containing `Control`, `NoRefine`, or `DefaultDType` are attribution tools, not daily recommendations.
- Director remains a clean upstream checkout at `576424f`. Local rollback archives stay under `.local/backups/director/` and are not committed.

