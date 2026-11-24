# RelayVSR dual-V100 Flash: isolated implementation and measurements

Date:2026-11-24. This isolated RelayVSR acceptance passes; this is not a production-core upgrade or release.

## Design and scope

Only this Raylight changes. `RelayVSR Flash Decode (Ray)` replaces VAELoader + VSRDecode;
native single-GPU conditioning and keyframe sampling remain unchanged. Independent official
Flash windows run on two V100s. Each window computes its shared endpoint to seed history;
only duplicate output frames are discarded. FP32, weights, RoPE, attention and detail stay unchanged.
The delivered path uses two GPU workers in one process, with no Ray cluster or tensor serialization.
CPU frames share local memory; actual Flash computation runs on both GPUs. The same Python PID on both cards is expected.
Independent windows do not require a contrived NVLink collective. Earlier FSDP probes did not
split Flash work and did not accelerate this bottleneck.

No production deployment or GitHub push. The isolated core is ComfyUI 0.46.0,
`2fcb6cef9ab1b6dc556b7cff1765a0724ad5e884`, Python3.12.10, Torch2.12.0+cu126,
xFormers0.0.35; 2xV100-SXM2-16GB/TCC/driver577.00.
The production core lacks native RelayVSR; adding this node alone does not add that capability.

## Controls and gates

480x864 input,240frames,24fps,10s; x2 output960x1728. Identical source,seed42,1step,CFG1,
euler/simple,detail1,FP16 keyframe sampling,FP32 Flash,H264 CRF16. New runs use --cache-none;
sampling,decoding and saving must actually rerun. Historical single: sample14.314s,
Flash133.272s,save27.177s,total183.482s. Use the faster of historical/matched native controls.
Flash-node lifecycle <=88.848s; workflow <=146.786s. Raw max_abs<=1e-5/RMSE<=1e-6;
encoded SSIM>=0.9999/PSNR>=50dB. These measure consistency, not ground-truth restoration quality.

## Development failures retained

- Local directory upload is invalid in actor-level runtime_env. Fixed by local shared import paths.
- A contiguous frame view still references the whole video's storage. A full run exceeded the
  plasma budget and was interrupted. Fixed with owned per-window clones; storage-size and
  serialized-size regression tests prevent recurrence.
- Available GPU resources lag actor destruction and miss CPU-only owners. Cleanup checks actor
  ownership and never shuts down a foreign cluster. Unknown ownership fails closed.

The corrected Ray prototype measured Flash108.537s/workflow149.241s and missed the cold gate.
A retained actor allocator also caused the next native sampling run to OOM. Prototype commits,
source and measurements remain in history/local artifacts, but this backend is not publicly shipped.

Default keep_workers=false detaches/cleans up owned VAE patchers and breaks model cycles. True
retains only useful decoder weights and trims unused CUDA workspace after each call. False is
recommended; extra processes are not required for the speedup. ComfyUI IMAGE caches and CUDA
contexts are outside this pool's ownership and are not promised to disappear at workflow end.
Every call rejects an existing connected Ray session, including warm reuse. It never creates
or shuts down a cluster. After an FSDP workflow, end your owned session with the existing RayKill
"Kill Entire Cluster" option or restart ComfyUI before this standalone RelayVSR workflow.
Director-tail integration and borrowing existing actors are not delivered in this iteration.

## Review and cleanup regressions

An independent Python review produced seven reproduced failing cases; fixes pass50 focused tests.
Shared-cluster driver/session/locality/store risks were removed with the unshipped actor backend.
One operation lock now covers admission,model unload,dispatch and cleanup; concurrent rejection
cannot alter Comfy state. Explicit detach/cleanup releases owned model cycles without cyclic GC.
Completed Future/result references are dropped before waiting for replacements. Failed warm
cleanup invalidates the pool, records failure and allows clean recreation while preserving the
original exception. Direct Flash honors the real VAE process_output callback, including nonidentity
regression coverage. The author reran verification; no second independent re-review is claimed.

## Measurements and evidence

Measured final code:1de3345. Timings are real uncached ComfyUI graphs, not direct diagnostic calls.
Cold process/pool does not mean the Windows filesystem cache was flushed.

### 480x864 / 10s / 240frames / 24fps / x2

| Configuration | Input+conditioning | UNet loader node | Sampling node | Flash lifecycle | Save | Server total |
|---|---:|---:|---:|---:|---:|---:|
| Historical native | Attachment | Attachment | 14.314 | 133.272 | 27.177 | 183.482 |
| Matched native | Attachment | Attachment | 14.579 | 134.421 | 26.826 | 184.477 |
| Dual default release,cold | 7.764 | 0.443 | 13.874 | 71.363 | 16.969 | 110.428 |
| Dual retained pool,first | 7.691 | 0.360 | 13.594 | 71.129 | 16.772 | 109.556 |
| Dual retained pool,warm | 7.733 | 0.338 | 13.718 | 70.853 | 16.605 | 109.256 |

Seconds. UNet GPU transfer remains inside the sampling-node measurement; the loader-node time
is not the complete model-loading time. VAE loading,pool setup,merge and cleanup are all inside
Flash. A sampling-node duration is not the progress bar's isolated s/it.
Against the faster historical baseline,default-cold Flash is1.868x (46.5% less time); total
time drops39.8% (1.662x). Both hard gates pass. Retention is not needed for the speedup.
Saving also takes about10s less; merge produces contiguous CPU images, but no separate layout-only
A/B was performed, so the entire saving difference is not attributed to one implementation detail.

| Configuration | GPU0/1 peak MiB | System RAM GiB | Process-tree RSS GiB | Private commit GiB | System commit GiB |
|---|---:|---:|---:|---:|---:|
| Matched native | 11249 / 316 | 22.678 | 7.213 | 9.314 | 27.818 |
| Dual default cold | 11249 / 5991 | 22.814 | 7.481 | 10.783 | 28.340 |
| Dual retained first | 11297 / 5989 | 23.325 | 7.927 | 10.818 | 29.022 |
| Dual retained warm | 11411 / 6015 | 24.083 | 8.473 | 10.930 | 28.847 |

Whole-board observed peaks at0.5s sampling,not Torch-active values or a guarantee of catching
microsecond transients. Native sampling still sets GPU0's workflow peak; no decrease there is
claimed. Default Flash mean GPU utilization is97.05%/94.91%; each card completes8 nonempty windows.
Pagefile-use estimates are0 in all3 runs. The primary <=12GiB/card and <=40GiB system-RAM budgets pass.
Power/utilization/allocator details remain attached; NVML transient power is not sustained power.

Final31-frame real raw comparison is bitwise identical,max_abs0,RMSE0. All3 complete240-frame
videos are960x1728/24fps/10s,fully FFmpeg-decodable,with identical decoded frame hashes,SSIM1,
PSNR infinity. Every frame,including window seams and the last frame,was checked.
The480p17-frame x4 tail case produces1920x3456/24fps,Flash44.991s,total60.191s and decodes fully.
It is a tail/capacity check,not full10s x4 performance acceptance.

Full1MP input is768x1376,120frames,5s,24fps:

| Factor | Output | Sampling | Flash | Save | Server total | Native Flash / total |
|---|---|---:|---:|---:|---:|---:|
| x2 | 1536x2752 | 18.541 | 115.793 | 16.134 | 156.380 | 219.556 / 273.646 |
| x4 | 3072x5504 | 75.210 | 534.571 | 51.201 | 666.938 | 1021.479 / 1197.313 |

Both fully decode all120frames and match the corresponding native videos frame-for-frame,
SSIM1/PSNR infinity. x2 observed GPU peaks12706/6336MiB,system RAM24.849GiB,RSS9.608GiB,
private commit14.302GiB. x4 peaks15514/7725MiB,system RAM44.933GiB,RSS29.584GiB,private
commit41.186GiB. Pagefile estimates0. The primary12GiB/40GiB budgets are not claimed for
every resolution:1MP/x4 exceeds them but fits the actual16GiB cards/128GiB host. It still
takes roughly11m07s; x4 means16x the input pixels,not fast production. Full480p10s x4 was
not rerun and is not marked accepted; do not extrapolate arbitrary size/duration.

To reproduce,use the isolated core/venv and the API example graph. Select your own video,match
x2/x4 UNet and Flash VAE,wire the same low-quality images and sampled latents,use2 workers and
keep_workers=false. Do not add RayInitializer/FSDP/USP or hide the second CUDA device. Use
--cache-none for performance comparisons; a cached second click is not warm GPU execution.
Two small console progress bars are independent windows,not two full-video decodes. Retention
is for same-VSR reuse; end its cache or restart before switching to an FSDP workflow. False is recommended.
All test videos stay under the local artifact video directory,not production output.

From the isolated ComfyUI directory,using its Python and model-path configuration:

```powershell
python main.py --listen 127.0.0.1 --port 8198 --disable-cuda-malloc --disable-dynamic-vram --fp16-unet --fp32-vae --cache-none
```

Ignored local artifacts live in .local/artifacts/relayvsr-flash-dev-20261124/:
graphs,history,node timings,0.5s whole-device monitoring,process-tree RSS/private commit,
system RAM/commit/pagefile estimate,window assignments and videos. Sum-of-RSS can double-count
shared pages; actual system RAM is recorded separately. Peaks exclude post-run FFmpeg QA.
The URI failure and interrupted spilling experiment are not counted as successful speedups.

## Regression scope

Production-core baseline:830passed. Final review fixes:869passed,11skipped because old core
has no native RelayVSR. The isolated new core already had14 failures before this work:
13 MiniMax AdaLN PackedLayout(frame_count) API mismatches and1 USP signature mismatch involving
new denoise masks. They are documented, not hidden by unrelated changes. New-core H3/Director
compatibility is not an accepted deliverable of this experiment.

Final new-core suite:865passed,14failed,1skipped,35subtests. Exact failure IDs match the pre-change
baseline. The first final invocation omitted RAYLIGHT_PYTHON,causing5 launcher cases and3subcases
to fail Python discovery; explicit harness configuration removes those failures without launcher
changes. Both logs are retained. The [maintainable validation data/failure inventory](data/2026-11-24-relayvsr-validation.json)
contains no media,absolute paths or runtime logs. All original830 tests remain;50 were added.
The11 legacy-core skips are numerical tests requiring native RelayVSR.
