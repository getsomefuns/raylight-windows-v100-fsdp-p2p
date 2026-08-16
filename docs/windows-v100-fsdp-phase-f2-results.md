# Windows V100 FSDP Phase F2 Results

Date: 2026-08-16

## Decision summary

Phase F2 synthetic acceptance passed. The Windows CUDA IPC/P2P data plane now
supports FSDP2 all-gather shards larger than the fixed 128 MiB staging buffer.
No CUDA weight payload was routed through Gloo or host memory.

This is a GO for the next, separately gated 5-second LTX FSDP integration test.
It is not yet a claim that the complete LTX workflow is stable or faster than
the existing Raylight Ulysses path.

## Test conditions

- Native Windows, two Tesla V100-SXM2-16GB GPUs in TCC mode.
- NVIDIA driver 577.00.
- Python 3.10.11, PyTorch 2.7.0+cu126, Ray 2.57.0.
- Gloo/TCPStore control plane; custom CUDA IPC/P2P data plane.
- One fixed 128 MiB local CUDA staging buffer per rank.
- FSDP2 inference only; no training reduce-scatter.

## Results

| Gate | Result |
|---|---|
| Unit regression | 22 tests passed |
| Existing all-to-all | Correct; about 59.3 GiB/s per rank |
| All-gather bandwidth | 64 MiB: about 60.0 GiB/s; 128/256/384 MiB: about 61.5-61.6 GiB/s per rank |
| 256 MiB shard over 128 MiB buffer | Sync and async passed; exactly two operation ids consumed |
| Chunk boundaries | First byte, 128 MiB boundary - 1, boundary, and final byte correct for both ranks |
| Dtypes and tail | FP32, FP16, BF16 and uint8 passed with a 128 MiB + 4 KiB shard |
| Repetition | Every dtype passed 20 sync and 20 async iterations per rank |
| Large FSDP2 parameters | 512 MiB full parameters; 256 MiB local parameters per rank |
| Numerical result | Five fresh sessions; finite output and zero maximum error |
| Page file | 0 MiB used during recorded snapshots and after the five sessions |
| Teardown | No orphan Ray/Python process; both GPUs returned to 10 MiB |

BF16 was tested as raw transported bytes only. This does not claim V100 BF16
compute support.

## Large FSDP2 memory and timing

The synthetic model uses a 16384 x 16384 FP16 linear weight. Its values and
input make the expected output analytically equal to one, so validation does
not retain a second full baseline model.

Across five fresh Python/Ray sessions (two ranks per session):

- Sharded steady allocation: 384.03 MiB per rank, including the 128 MiB P2P
  staging buffer and 256 MiB local parameter shard.
- Allocation after each forward: 392.19 MiB per rank. The approximately 8 MiB
  difference is output/runtime state; the 512 MiB full parameter is not kept.
- Peak forward allocation: about 1416.22 MiB per rank.
- Actor teardown allocation: 136.13 MiB per rank.
- Process exit baseline: 10 MiB per physical GPU.
- Cold forward range: 0.121-0.389 seconds.
- Stable warm forwards: typically 6.83-7.27 ms. A few first-warm samples
  reached 8-27 ms; later samples in the same session returned to the stable
  band, with no cross-session growth.
- Per-actor committed-memory increase before actor exit: approximately
  555-633 MiB. After all five fresh sessions exited, system committed memory
  was approximately 13.0 GiB and the page file remained unused.

## Important FSDP2 inference finding

PyTorch 2.7 deliberately forces the outermost/root FSDP state to keep
unsharded parameters after forward, even when `reshard_after_forward=True`.
A root-only synthetic wrapping therefore retained about 904 MiB per rank after
the first forward and made later forwards unrealistically fast.

The accepted inference structure shards the weight-bearing child first and
then shards an outer root with no direct parameters. The child is no longer
the FSDP root, so it honors post-forward resharding. This changed the
post-forward allocation from about 904 MiB to 392 MiB and made every warm
forward exercise the P2P all-gather path.

Any LTX integration must preserve this nested/submodule wrapping property.
Applying `fully_shard()` only once to the top-level transformer would defeat
the memory-saving goal after its first forward.

## c10d warning classification

The recurring `<hostname>:<port>` warning is independently reproducible with a
single-process, world-size-one legacy TCPStore. PyTorch first tries an IPv6
candidate for the supplied IPv4 address, receives Winsock 10049, then
immediately connects over IPv4. See
`docs/windows-v100-fsdp-phase-f2-diagnostics.md` for the isolated evidence.

The warning is not caused by Raylight, Gloo payload transport, FSDP or CUDA
P2P. It is not globally suppressed because suppressing c10d warnings could
hide a real rendezvous failure.

## Next gate

The next phase may load only the 5-second LTX test workflow first. Acceptance
must separately prove:

- nested FSDP wrapping actually selects the weight-bearing LTX submodules;
- model weights stay sharded between forward calls;
- both ranks remain synchronized;
- output is finite and visually non-black;
- host committed memory and VRAM return to their expected baselines;
- performance is compared against the previously measured single-card and
  Raylight baselines without relaxing timeout or memory criteria.

## Subsequent integration status

The later LTX integration, failed visual-output gate, root-cause fixes, and final
visual acceptance are maintained in [WINDOWS_V100_FSDP_TESTING.md](WINDOWS_V100_FSDP_TESTING.md).
