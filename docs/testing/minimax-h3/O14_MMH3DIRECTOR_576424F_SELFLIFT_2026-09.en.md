# O14 MiniMaxH3Director 576424f / SelfLift Acceptance

O14 upgrades an unmodified MiniMaxH3Director from `52f8fb7` to `576424f` and extends only Raylight's proxy protocol. The fixed reference case uses dual V100 FSDP/P2P, REF2VA Turbo4, SOL, SelfLift 0.5, Director Refine 2, 864x480, 124 frames, and 24 fps.

The matched pre/post non-SelfLift baseline changes from 314.05 to 288.51 seconds end to end and from 165.75 to 150.11 seconds of remote sampling, with the same 9223/9201 MiB GPU peaks.

The initial SelfLift adapter produced valid video and broadband-noise audio. No-SOL, no-Refine, default-dtype, safe-FP16, scale-1, and unmodified single-GPU controls isolated the defect to the Ray proxy's low-to-high transition. Its weightless facade omitted the real MiniMaxH3 AV audio scaling (`shift_video / shift_audio = 4`). The proxy now mirrors `process_latent_in/process_latent_out`; video is unchanged and audio scaling round-trips exactly.

After the fix, the full three-stage run completes in 288.55 seconds, with 143.83 seconds of remote sampling, 9351/9329 MiB GPU peaks, 59355.73 MiB physical-RAM peak, and 105183.29 MiB commit peak. Audio DC offset is `-0.000056/-0.000043`, zero-crossing is `0.047671/0.063260`, and NaN/Inf counts are zero. The output has 124/124 unique frame hashes at 864x480 and 24 fps, with matched 5.1667/5.1670-second video/audio durations.

SelfLift, Refine, live preview, SOL, and existing Director paths pass. Semantic Bridge and FaceRefine have contract coverage only because their optional local weights were unavailable. Raw telemetry stays under local `.local/artifacts/o14/`; media and input assets are not committed.

The final suite reports `753 passed, 217 warnings, 35 subtests passed`. Targeted Ruff for all O14-touched Python files and `git diff --check` pass. Repository-wide Ruff still reports 29 pre-existing findings in untouched WAN/LTX/SeedVR and related files; O14 deliberately does not mix unrelated cleanup into this compatibility release.
