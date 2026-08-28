# Raylight Windows V100 FSDP Phase F2 Implementation Plan

## Execution Status (2026-08-16)

- Task 1 complete: the `WorkStudio:<port>` warning was isolated to PyTorch 2.7 legacy Windows TCPStore's IPv6-first candidate attempt and documented.
- Task 2 complete: deterministic `chunk_ranges()` passed RED/GREEN unit coverage.
- Task 3 complete: fixed 128 MiB staging handles 256 MiB sync/async shards; 64/128/256/384 MiB bandwidth matrix passed.
- Task 4 complete: FP32, FP16, BF16-storage and uint8 tail cases passed 20 sync and 20 async iterations per rank.
- Task 5 complete: five fresh 512 MiB-full/256 MiB-local FSDP2 sessions passed with post-forward resharding and resource release.
- Task 6 complete: 22 unit tests, NVIDIA P2P, project P2P, minimal FSDP2 and final resource checks passed.
- Decision: **GO to Phase F3's gated 5-second LTX integration.**

Detailed evidence:

- `ComfyUI/custom_nodes/raylight/docs/windows-v100-fsdp-phase-f2-diagnostics.md`
- `ComfyUI/custom_nodes/raylight/docs/windows-v100-fsdp-phase-f2-results.md`


> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the two-rank Windows CUDA P2P FSDP inference path handle parameter shards larger than the 128 MiB staging buffer, explain all c10d address warnings, and pass a 256 MiB-local-shard FSDP stress test before loading LTX weights.

**Architecture:** Keep one fixed 128 MiB CUDA IPC staging buffer per rank and exchange a large local shard through deterministic chunks. Gloo remains the ProcessGroup/control plane; CUDA `all_gather_into_tensor` remains exclusively on the P2P data plane. Each chunk gets its own monotonic operation id and CUDA ready/consumed event ordering, while one external FSDP call returns a Work backed by the final chunk event.

**Tech Stack:** Windows 11, Python 3.10.11, PyTorch 2.7.0+cu126, Ray 2.57.0, Gloo control plane, CUDA IPC/P2P, 2? Tesla V100-SXM2-16GB in TCC.

**Spec:** [2026-08-15-windows-v100-fsdp-next-stage-plan.md](2026-08-15-windows-v100-fsdp-next-stage-plan.md)

## Global Constraints

- Only modify files under the isolated `<environment-root>`.
- Exactly two V100 ranks, one machine, native Windows, TCC mode.
- Do not change NVIDIA driver 577.00.
- Do not silently move CUDA model-weight collectives to Gloo, CPU RAM or the page file.
- Keep the existing Ulysses all-to-all path working.
- Inference only; do not implement training reduce-scatter.
- GGUF FSDP remains unsupported.
- Every production behavior change follows RED ? GREEN ? full regression.
- Do not start the LTX workflow until every F2 gate below passes.

---

### Task 1: Identify the `WorkStudio:<dynamic-port>` c10d warning source

**Files:**
- Modify: `ComfyUI/custom_nodes/raylight/tests/windows_p2p_fsdp_probe.py`
- Create: `ComfyUI/custom_nodes/raylight/docs/windows-v100-fsdp-phase-f2-diagnostics.md`

**Interfaces:**
- Consumes: the existing `FSDPProbeActor` and Windows Gloo initialization.
- Produces: a phase-labelled diagnostic report containing actor environment, hostname, Ray GCS address, selected Gloo host, process-group state and the first phase that emits the warning.

- [ ] Add monotonic timestamped markers before and after actor creation, Gloo init, DeviceMesh init, P2P route installation, `fully_shard`, forward, barrier and destroy.
- [ ] Capture actor-side `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`, hostname, Ray node address and GCS address without modifying them.
- [ ] Run four isolated stages: Gloo-only, DeviceMesh-only, fully-shard without forward, and full forward/teardown.
- [ ] Confirm whether the warning occurs on every run and identify the earliest stage that produces it.
- [ ] State one root-cause hypothesis supported by the captured address/port evidence.
- [ ] Only after the hypothesis is confirmed, add a failing regression probe for the originating configuration.
- [ ] Apply one minimal fix at the source and rerun all four stages.
- [ ] Record the root cause and before/after evidence in the diagnostics document.

**Gate:** No unexplained c10d connection warning may remain. If it is proven harmless and owned by Ray/PyTorch rather than this project, document the exact origin and add a stable filter/monitor only if it cannot hide real rendezvous failures.

---

### Task 2: Define deterministic chunk planning

**Files:**
- Modify: `ComfyUI/custom_nodes/raylight/src/raylight/distributed_worker/windows_p2p.py`
- Modify: `ComfyUI/custom_nodes/raylight/tests/test_windows_p2p.py`

**Interfaces:**
- Produces: `chunk_ranges(total_bytes: int, capacity_bytes: int) -> tuple[tuple[int, int], ...]`.
- Each range is half-open `[start, end)`, covers every byte exactly once, never exceeds capacity, and preserves byte order.

- [ ] Write a failing table-driven test with literal expected ranges for 0, 1, exactly-capacity, capacity+1 and 2.5-capacity payloads.
- [ ] Run the test and confirm failure because `chunk_ranges` does not exist.
- [ ] Implement the smallest range planner that passes the literals.
- [ ] Write failing tests for zero/negative capacity and negative total bytes.
- [ ] Add explicit validation and rerun the focused tests.
- [ ] Run the complete Windows P2P and RayWorker unit suite.

**Gate:** Range planning is deterministic and independently testable without CUDA.

---

### Task 3: Implement chunked two-rank CUDA all-gather

**Files:**
- Modify: `ComfyUI/custom_nodes/raylight/src/raylight/distributed_worker/windows_p2p.py`
- Modify: `ComfyUI/custom_nodes/raylight/tests/test_windows_p2p.py`
- Modify: `ComfyUI/custom_nodes/raylight/tests/windows_p2p_ray_probe.py`

**Interfaces:**
- Consumes: `chunk_ranges`, `WindowsSpinControl`, IPC send buffer and CUDA ready/consumed events.
- Produces: the existing `CudaP2PAllToAll.all_gather_into_tensor(output, input_tensor, async_op=False)` API with no shard-size limit other than output allocation and available GPU memory.

- [ ] Replace the old ?shard exceeds capacity? expectation with a failing behavior test proving a larger shard is split instead of rejected.
- [ ] Add a focused test proving chunk destination offsets place rank 0 then rank 1 without overlap.
- [ ] Refactor one chunk exchange into a private method only after the tests fail for the missing behavior.
- [ ] Assign one monotonic operation id per chunk and retain poison-on-mismatch semantics.
- [ ] Return `CudaEventWork` backed by the final chunk completion event for `async_op=True`.
- [ ] Extend the live Ray probe to transfer a shard larger than 128 MiB and verify first, boundary and last values on both ranks.
- [ ] Record effective remote payload bandwidth for 64, 128, 256 and 384 MiB local shards.
- [ ] Run the original all-to-all probe after the new all-gather cases.

**Gate:** 256 MiB local shards pass sync and async correctness on both ranks, with no CPU/Gloo fallback, no deadlock and no regression in the existing all-to-all path.

---

### Task 4: Validate dtype and uneven-tail behavior

**Files:**
- Modify: `ComfyUI/custom_nodes/raylight/tests/windows_p2p_ray_probe.py`
- Modify: `ComfyUI/custom_nodes/raylight/tests/test_windows_p2p.py`

**Interfaces:**
- Consumes: chunked all-gather.
- Produces: evidence for FP32, FP16, BF16 transport and uint8 byte-packed payloads, including final chunks not aligned to capacity.

- [ ] Add failing live cases whose final chunk is smaller than the staging buffer.
- [ ] Verify literal boundary values for FP32, FP16, BF16 and uint8.
- [ ] Verify contiguous CUDA and same-dtype rejection remains active.
- [ ] Verify output size must still equal exactly two input shards.
- [ ] Run sync and async cases at least 20 consecutive iterations per dtype.

**Gate:** Every dtype and uneven-tail case is correct on both ranks; transport tests do not require the V100 to execute BF16 arithmetic.

---

### Task 5: Run a large-parameter FSDP2 stress test

**Files:**
- Modify: `ComfyUI/custom_nodes/raylight/tests/windows_p2p_fsdp_probe.py`
- Create: `ComfyUI/custom_nodes/raylight/docs/windows-v100-fsdp-phase-f2-results.md`

**Interfaces:**
- Consumes: chunked P2P all-gather and the clean Gloo/DeviceMesh setup from Tasks 1?4.
- Produces: a repeatable synthetic model case with a 256 MiB local shard and compact memory/performance results.

- [ ] Add a mode that constructs an FP16 linear weight with approximately 512 MiB full parameters and approximately 256 MiB local FSDP shards.
- [ ] Record per-rank full/local parameter bytes, allocated/reserved VRAM before sharding, after sharding, peak forward and after teardown.
- [ ] Record process RSS, system committed memory and page-file usage before/after.
- [ ] Run one cold iteration and five warm iterations.
- [ ] Confirm output against a deterministic analytical or unsharded reference without retaining an unnecessary full baseline copy during measurement.
- [ ] Run the stress test five times in fresh Ray sessions.
- [ ] Confirm no rank drift, deadlock, NaN/Inf, growing VRAM or orphan process.
- [ ] Summarize results in the F2 results document; keep raw logs outside the repository only while debugging.

**Gate:** Both ranks retain approximately half the full weight, complete five fresh-session runs, release resources to the baseline, and never use Gloo/CPU for CUDA weight payloads.

---

### Task 6: F2 regression and LTX-entry decision

**Files:**
- Modify: `ComfyUI/custom_nodes/raylight/docs/windows-v100-fsdp-phase-f0.md`
- Modify: `ComfyUI/custom_nodes/raylight/docs/windows-v100-fsdp-phase-f2-results.md`
- Modify: `docs/raylight/2026-08-15-windows-v100-fsdp-next-stage-plan.md`

**Interfaces:**
- Produces: an explicit GO/NO-GO decision for the first 5-second LTX FSDP workflow.

- [ ] Run all Windows P2P and RayWorker unit tests.
- [ ] Run native NVIDIA P2P, project all-to-all/all-gather and minimal/large FSDP probes.
- [ ] Run `git diff --check`.
- [ ] Confirm both GPUs remain TCC and return to baseline memory.
- [ ] Confirm no Ray/Python child process remains.
- [ ] Compare every F2 gate above against fresh evidence.
- [ ] Mark GO only if every gate passes; otherwise document the exact blocker without relaxing timeouts or memory thresholds.

**Gate:** A written GO decision is required before any LTX model load.
