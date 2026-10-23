# 2026-10-23 Long-reference OOM and memory recording plan

## Scope

- Fix the MiniMax H3 Director 15-second REF2VA CUDA OOM raised by a full-size Ulysses `all_to_all` output allocation during the high-resolution SelfLift pass.
- Separate genuine input working-set growth from Ray request copies, worker-private memory, and Windows commit usage.

## Implementation

1. Project MiniMax H3 QKV in bounded 4,096-token chunks into independently owned Q/K/V tensors, with one FSDP unshard/reshard pair per attention call.
2. On the two-rank Windows P2P path, reuse the contiguous communication layout as both all-to-all input and output. The existing 128 MiB staging protocol preserves outbound bytes before destination ranges are overwritten.
3. Preserve the original yunchang path on other backends.
4. Write dense xFormers attention results into consumed Query storage.
5. Record one-second host and CUDA samples for the main process and both workers, then write a JSONL summary on success or failure under `ComfyUI/user/raylight_memory/<date>/`.

## Acceptance

- Automated tests cover bounded QKV projection, independent storage, in-place P2P exchange, and failure-safe memory summaries.
- The original 15-second, two-image, 354-frame reference-video workflow must pass the former OOM point and complete with finite output and no quality regression.
- Do not modify ComfyUI or MiniMax H3 Director sources.
