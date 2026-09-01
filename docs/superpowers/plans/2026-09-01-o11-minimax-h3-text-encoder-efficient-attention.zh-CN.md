# O11 MiniMax H3 Text Encoder 高效注意力实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在原生 Windows 双 V100 TP2 Text Encoder 中加入 xFormers GQA 与 PyTorch Efficient 可选路由，删除平方级因果遮罩，并让 15 秒参考视频、7 张不同图片、7000 字符提示词完整通过 Text Encoder。

**Architecture:** 保留 GPU0 视觉预处理、CPU 中转 prepared conditioning、双 GPU 语言主干的 O10 生命周期。新增独立注意力后端模块，在加载 50 层前一次性选择 AUTO/XFORMERS_GQA/TORCH_EFFICIENT/MATH_DEBUG；xFormers 使用五维 GQA 与结构化因果 bias，Torch Efficient 显式扩展 K/V 并强制 Efficient kernel，任何正式路径都不得回退 Math。

**Tech Stack:** Windows 11、2× Tesla V100-SXM2-16GB TCC、Python 3.10.11、PyTorch 2.7.0+cu126、xFormers 0.0.30、ComfyUI 0.31、pytest、CUDA P2P TP2。

**Spec:** `docs/superpowers/specs/2026-09-01-o11-minimax-h3-text-encoder-efficient-attention-design.zh-CN.md`

## Global Constraints

- 所有源码只修改 `E:\ComfyUI-py310\raylight-windows-v100-fsdp-p2p`，不得直接修改部署目录。
- 不升级 Python、PyTorch、CUDA、xFormers、驱动或模型格式。
- 不修改 ComfyUI 核心和 MiniMaxH3Director。
- `XFuser_attention` 继续只控制 Diffusion 采样；Text Encoder 使用独立字段。
- `AUTO` 不允许选择 Math；`MATH_DEBUG` 超过 2048 token 必须拒绝。
- 不创建实体 `[sequence, sequence]` 浮点因果遮罩。
- 每个生产代码变化前必须先运行一个因缺失行为而失败的测试。
- 真实最大输入 Gate 使用 ComfyUI 工作流，Python 微基准不能替代。
- 只有 Gate A–F、审查和文档全部通过后才部署；部署验证后才允许推送 GitHub。

---

## File Map

- Create `src/raylight/text_encoder_parallel/attention_backends.py`: 后端枚举、能力探针、AUTO 选择和统一 attention 调用。
- Modify `src/raylight/text_encoder_parallel/qwen3vl_tp.py`: TP2 block 注入统一 attention callable，移除直接 SDPA。
- Modify `src/raylight/text_encoder_parallel/clip_proxy.py`: 删除 dense causal mask，向 backbone 传递后端并记录 token/峰值摘要。
- Modify `src/raylight/text_encoder_parallel/nodes.py`: 新增向后兼容的可选 `text_encoder_attention` widget。
- Modify `src/raylight/text_encoder_parallel/__init__.py`: 导出稳定后端接口。
- Create `tests/test_text_encoder_tp_attention.py`: 路由、GQA shape、Math 禁止、因果语义和数值测试。
- Modify `tests/test_text_encoder_tp_qwen3vl.py`: block 使用注入后端，不再要求 dense mask。
- Modify `tests/test_text_encoder_tp_clip_proxy.py`: backbone 生命周期、后端传播和无 dense mask 测试。
- Modify `tests/test_text_encoder_tp_nodes.py`: 节点默认 AUTO、旧调用兼容和显式选择测试。
- Create `scripts/minimax-h3/benchmark_text_encoder_attention.py`: 6992/18000/28000 V100 容量与数值探针。
- Create `tests/test_text_encoder_tp_attention_benchmark.py`: 探针 CLI/JSON 契约测试。
- Create `example_workflows/MMH3_TextEncoder_O11_MaxContext.json`: 最终 ComfyUI conditioning-only 验收工作流。
- Create `docs/testing/minimax-h3/TEXT_ENCODER_EFFICIENT_ATTENTION_O11_2026-09.zh-CN.md`: 所有成功与失败实验记录。
- Modify `README.md`, `README_EN.md`, `pyproject.toml`: Gate F 后才更新功能、限制和版本。

---

### Task 1: 冻结当前 OOM 回归和后端公共契约

**Files:**
- Create: `tests/test_text_encoder_tp_attention.py`
- Create: `src/raylight/text_encoder_parallel/attention_backends.py`
- Modify: `src/raylight/text_encoder_parallel/__init__.py`

**Interfaces:**
- Produces: `TextEncoderAttentionBackend(str, Enum)` values `AUTO`, `XFORMERS_GQA`, `TORCH_EFFICIENT`, `MATH_DEBUG`.
- Produces: `AttentionBackendUnavailable(RuntimeError)`.
- Produces: `resolve_attention_backend(requested: str, *, probe_device=None, probe=None) -> TextEncoderAttentionBackend`；`probe` 仅用于单元测试注入，生产环境使用真实 CUDA 小张量探针。
- Produces: `run_text_encoder_attention(query, key, value, *, backend, is_causal=True, xformers_ops=None) -> torch.Tensor`；`xformers_ops` 仅用于依赖注入测试，正常调用延迟导入本地 xFormers。

- [x] **Step 1: Write failing enum and policy tests**

```python
def test_auto_never_resolves_to_math():
    with pytest.raises(AttentionBackendUnavailable):
        resolve_attention_backend("AUTO", probe=lambda _: False)

def test_math_debug_rejects_long_sequence():
    q, k, v = cpu_gqa(sequence=2049)
    with pytest.raises(ValueError, match="2048"):
        run_text_encoder_attention(q, k, v, backend="MATH_DEBUG")
```

- [x] **Step 2: Run RED test**

Run: `E:\ComfyUI-py310\Python310\python.exe -m pytest tests/test_text_encoder_tp_attention.py -q`

Expected: collection fails because `attention_backends` does not exist.

- [x] **Step 3: Implement enum, strict parsing, dependency-injected resolver, and Math ceiling**

The module must not import xFormers at module import time. `AUTO` probes xFormers first, Torch Efficient second, and raises with both rejection reasons.

- [x] **Step 4: Run GREEN test and lint**

Run: `E:\ComfyUI-py310\Python310\python.exe -m pytest tests/test_text_encoder_tp_attention.py -q`

Run: `E:\ComfyUI-py310\Python310\python.exe -m ruff check src/raylight/text_encoder_parallel/attention_backends.py tests/test_text_encoder_tp_attention.py`

- [x] **Step 5: Commit**

```text
git commit -m "feat: add strict text attention backend policy"
```

### Task 2: 实现 xFormers GQA 和 Torch Efficient

**Files:**
- Modify: `tests/test_text_encoder_tp_attention.py`
- Modify: `src/raylight/text_encoder_parallel/attention_backends.py`

**Interfaces:**
- Consumes: Task 1 enum and `run_text_encoder_attention`.
- Produces: `_xformers_gqa_forward(query, key, value, *, is_causal) -> torch.Tensor`.
- Produces: `_torch_efficient_forward(query, key, value, *, is_causal) -> torch.Tensor`.

- [x] **Step 1: Write failing shape, causality, and numerical tests**

Tests use `[B,Hq,S,D]` with `Hq=32`, `Hkv=4`; compare at 64/221/256 tokens against forced Math GQA. FP16 uses the measured quantization-step bound `max_abs <= 1.25e-4`, plus `mean_abs <= 1e-5` and finite output; FP32 keeps `max_abs <= 1e-4`. Add a spy proving xFormers K/V use expand views and Torch physically expands to equal heads.

- [x] **Step 2: Run RED test**

Expected: backend functions raise unavailable/not implemented.

- [x] **Step 3: Implement xFormers five-dimensional GQA**

Reshape Q to `[B,S,Hkv,Hq/Hkv,D]`, reshape K/V to `[B,S,Hkv,1,D]`, use `expand`, call `memory_efficient_attention`, and restore `[B,Hq,S,D]`. Use `LowerTriangularMask()` when causal.

- [x] **Step 4: Implement Torch Efficient with explicit K/V expansion**

Use `repeat_interleave(Hq // Hkv, dim=1)`, `sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)`, and `is_causal=True`. Reject non-divisible head counts.

- [x] **Step 5: Run GREEN tests on CPU policy and V100 kernel cases**

Run the focused file, then repeat V100-marked tests with the local Python runtime.

- [x] **Step 6: Commit**

```text
git commit -m "feat: add V100 efficient GQA attention backends"
```

### Task 3: 注入 TP2 block 并删除直接 SDPA

**Files:**
- Modify: `tests/test_text_encoder_tp_qwen3vl.py`
- Modify: `src/raylight/text_encoder_parallel/qwen3vl_tp.py`

**Interfaces:**
- `TP2Qwen3VLBlock.__init__(..., attention_forward: Callable[..., torch.Tensor])`.
- `TP2Qwen3VLBlock.forward(..., operation_id: int)` no longer consumes a dense `attention_mask` in normal execution.

- [x] **Step 1: Change block tests first**

Inject a recording CPU Math callable and assert both ranks call it once with `is_causal=True`; assert no `[S,S]` argument is passed. Preserve the full-block numerical baseline and two P2P sums.

- [x] **Step 2: Run RED test**

Expected: constructor rejects `attention_forward` or forward still requires `attention_mask`.

- [x] **Step 3: Replace direct `F.scaled_dot_product_attention` with injected callable**

Keep Q/K/V, RMSNorm, MRoPE, output reshape, P2P sums, MLP, and operation IDs unchanged.

- [x] **Step 4: Run focused and quantized block tests**

Run: `pytest tests/test_text_encoder_tp_qwen3vl.py -q`

- [x] **Step 5: Commit**

```text
git commit -m "fix: route Qwen TP2 blocks through efficient attention"
```

### Task 4: Backbone 路由、无平方 mask 和生命周期

**Files:**
- Modify: `tests/test_text_encoder_tp_clip_proxy.py`
- Modify: `src/raylight/text_encoder_parallel/clip_proxy.py`

**Interfaces:**
- `load_minimax_h3_tp2_backbone(checkpoint, *, attention_backend="AUTO") -> TP2Qwen3VLBackbone`.
- `TP2Qwen3VLBackbone.selected_attention_backend: TextEncoderAttentionBackend`.
- `MiniMaxH3TP2ClipProxy(..., attention_backend="AUTO")`.

- [x] **Step 1: Write failing tests**

Assert backend selection happens before backbone weights load, all blocks receive the same callable, `_causal_mask` no longer exists/is never invoked, failure cleanup closes resources, and selected backend survives clone.

- [x] **Step 2: Run RED test**

Expected: unexpected keyword or dense-mask assertion fails.

- [x] **Step 3: Resolve backend once and remove `_causal_mask`**

Pass only `position_ids`, `operation_id`, and the injected backend through 50 layers. Preserve visual-to-CPU cleanup before `backbone_load`.

- [x] **Step 4: Run GREEN tests**

Run clip proxy and qwen focused suites together.

- [x] **Step 5: Commit**

```text
git commit -m "fix: remove quadratic TP2 causal mask"
```

### Task 5: 暴露网页端选择且保持旧工作流兼容

**Files:**
- Modify: `tests/test_text_encoder_tp_nodes.py`
- Modify: `src/raylight/text_encoder_parallel/nodes.py`

**Interfaces:**
- Optional widget: `text_encoder_attention` with choices `AUTO`, `XFORMERS_GQA`, `TORCH_EFFICIENT`, `MATH_DEBUG`.
- `load_clip(..., text_encoder_attention="AUTO")` preserves old positional/serialized workflows.

- [x] **Step 1: Write failing schema and propagation tests**

Assert the field is optional/default AUTO, explicit values reach proxy/backbone factories, invalid values fail before GPU lease, and RayInitializer schema remains unchanged.

- [x] **Step 2: Run RED test**

- [x] **Step 3: Implement optional widget and propagation**

Do not rename `XFuser_attention`; update only the dedicated Text Encoder node description/tooltip.

- [x] **Step 4: Run GREEN and workflow schema tests**

Run node, workflow, and Director task-9 workflow tests.

- [x] **Step 5: Commit**

```text
git commit -m "feat: expose Text Encoder attention routing"
```

### Task 6: 添加容量探针和聚合诊断

**Files:**
- Create: `scripts/minimax-h3/benchmark_text_encoder_attention.py`
- Create: `tests/test_text_encoder_tp_attention_benchmark.py`
- Modify: `src/raylight/text_encoder_parallel/baseline.py`
- Modify: `src/raylight/text_encoder_parallel/clip_proxy.py`

**Interfaces:**
- CLI: `--backend`, `--sequence`, `--device`, `--json-output`.
- JSON fields: backend, sequence, q_heads, kv_heads, dtype, input_mib, peak_mib, elapsed_seconds, finite, math_fallback_detected.
- Aggregate encode record includes token breakdown, stage timings, GPU peaks, and cleanup usage.

- [x] **Step 1: Write failing CLI contract and aggregate-record tests**

- [x] **Step 2: Run RED tests**

- [x] **Step 3: Implement probe without importing ComfyUI server state**

The probe uses actual local-head shapes 32/4/128 and refuses Math for sequence above 2048.

- [x] **Step 4: Implement one-line normal diagnostics**

Detailed data remains behind existing stage-timing diagnostics; do not log full prompts.

- [x] **Step 5: Run 6992/18000/28000 probes on both V100 backends**

Check design Gate B thresholds and save JSON under the canonical O11 artifact directory.

- [x] **Step 6: Commit**

```text
git commit -m "test: add O11 Text Encoder capacity probes"
```

### Task 7: 完整回归与代码审查

**Files:**
- Review all Task 1–6 files.

- [x] **Step 1: Run focused suites**

Run all `test_text_encoder_tp_*` files and record exact pass counts.

- [x] **Step 2: Run full repository suite**

Run `python -m pytest -q`, followed by Ruff on all modified Python files.

- [x] **Step 3: Complete Python-focused code review**

专项自审覆盖 backend dispatch、tensor shapes/strides、异常清理、imports 和向后兼容；发现的问题均先补失败测试再修复。

- [x] **Step 4: Commit review fixes**

```text
git commit -m "fix: harden O11 attention routing"
```

### Task 8: 冻结真实最大输入与工作流

**Files:**
- Create: `example_workflows/MMH3_TextEncoder_O11_MaxContext.json`
- Create: `docs/testing/minimax-h3/o11-boundary-input-manifest.local.json` (gitignored/local evidence only)
- Modify: `.gitignore` only if the local manifest pattern is not already ignored.

**Interfaces:**
- Workflow uses 1056×608, length 362, one 15.0-second reference video, seven distinct images, exactly 7000 normalized characters, and terminates at conditioning.

- [x] **Step 1: Inventory candidate media and validate metadata/hashes**

Reject duplicated image hashes, short images, non-15-second video, black/static capacity dummy, and meaningless prompt padding.

- [x] **Step 2: Build and validate workflow JSON**

Use real ComfyUI nodes, load the workflow in the UI/API parser, and save UI/API JSON hashes.

- [x] **Step 3: Verify fixture counters before running GPU work**

Assert video duration tolerance, seven hashes, `len(normalized_prompt)==7000`, and record tokenizer token count.

- [x] **Step 4: Commit distributable workflow only**

```text
git commit -m "test: freeze O11 maximum-context workflow"
```

### Task 9: Gate C/D 真实 Text Encoder 验收

**Files:**
- Create: `docs/testing/minimax-h3/TEXT_ENCODER_EFFICIENT_ATTENTION_O11_2026-09.zh-CN.md`

- [x] **Step 1: Stop stale ComfyUI/Ray processes and record idle GPU/RAM/commit/pagefile**

- [x] **Step 2: Deploy the reviewed worktree atomically**

Use the existing deployment script and verify tracked-file hashes.

- [x] **Step 3: Run AUTO/XFORMERS_GQA cold and warm-file-cache conditioning-only workflow**

Require exact frozen input hashes and record visual/language/cleanup timings and peaks.

- [x] **Step 4: Run forced TORCH_EFFICIENT once with the same workflow**

- [x] **Step 5: Run the workflow twice in one ComfyUI session**

Verify post-run commit growth is no more than 512 MiB and language weights are released.

- [x] **Step 6: Evaluate Gate D without lowering inputs**

If Attention passes but MLP OOMs, stop and record the MLP tensor/lifetime root cause. Do not shorten video, remove images, or truncate prompt.

- [x] **Step 7: Commit the factual test summary**

```text
git commit -m "test: record O11 maximum-context acceptance"
```

### Task 10: Gate E sampling smoke and regressions

**Files:**
- Modify: `docs/testing/minimax-h3/TEXT_ENCODER_EFFICIENT_ATTENTION_O11_2026-09.zh-CN.md`

- [x] **Step 1: Re-run the original 6992-token OOM case**

- [x] **Step 2: Run O10 221-token conditioning smoke**

- [x] **Step 3: Run one 3-second/480p/four-step sampling integration smoke**

This validates the conditioning contract but does not replace maximum Text Encoder acceptance.

- [x] **Step 4: Run O6/O8/O9/LTX/MiniMax/Director regression set**

- [x] **Step 5: Record failures and accepted results without deleting failed experiments**

- [x] **Step 6: Commit**

```text
git commit -m "test: complete O11 integration regression"
```

### Task 11: Gate F 文档、版本、部署和发布准备

**Files:**
- Modify: `README.md`
- Modify: `README_EN.md`
- Modify: `pyproject.toml`
- Create: `docs/releases/2026-09-o11-text-encoder-efficient-attention.zh-CN.md`
- Create: `docs/releases/2026-09-o11-text-encoder-efficient-attention.en.md`
- Modify: O11 plan checkboxes and acceptance summary.

- [x] **Step 1: Update bilingual user guidance**

Document AUTO default, forced backends, Text Encoder versus XFuser attention, capacity boundary, and sampling non-guarantee.

- [x] **Step 2: Update version only after Gate D/E**

Use the next minor version after 1.12.0 and keep every version source consistent.

- [x] **Step 3: Run final tests, import smoke, V100 probe, and deployed hash comparison**

- [x] **Step 4: Commit release preparation**

```text
git commit -m "release: complete O11 efficient Text Encoder attention"
```

- [x] **Step 5: Do not push until the user requests or confirms GitHub synchronization**

## Plan Self-Review

- Spec sections 1–13 map to Tasks 1–11.
- Backend names and public signatures are consistent across tasks.
- The real 15-second/seven-image/7000-character Gate D cannot be replaced by microbenchmarks.
- Vision preprocessing, language attention, MLP stop condition, cleanup, regressions, documentation, and deployment all have explicit gates.
- No placeholder tasks or unspecified fallback remain.

## Acceptance Summary

- Gate A/B：两种正式高效后端数值有限且无 Math；28k attention 峰值为 xFormers 1314.13 MiB、Torch Efficient 1861.38 MiB。
- Gate C/D：15 秒视频、7 张不同图片、7000 有效字符展开 24,236 序列；AUTO 冷/热 112.94/112.55 秒，强制 Torch Efficient 117.50 秒，全部通过。
- 用户真实门槛：7 图、1 视频、1 音频、5802 字符展开 12,302 序列，Text Encoder 62.88 秒，双卡分配峰值 9244/9451 MiB，完整通过并有余量。
- Gate E初始记录：3秒、864×480、Turbo4一采直出完成；当时15秒的0.6MP和0.4MP均于Diffusion第一步OOM。Task12追加优化随后让相同15秒0.4MP固定输入完整通过，0.6MP仍未验收。
- 回归：Task11为全仓664 passed；Task12收尾为全仓695 passed、35 subtests passed。修改范围Ruff、导入自检和双V100/TCC/P2P探针通过，128–384MiB all-gather约61.54–61.62GiB/s。
- GitHub：本计划明确保持未推送，等待用户单独要求同步。

---

## O11 追加优化：15 秒 Diffusion 安全 FP16 生命周期

本项是 O11 的追加优化，不建立 O12 或其他新阶段。它只修改唯一 Raylight 工作树，不修改 ComfyUI 核心、MiniMaxH3Director、模型文件或工作流接口。

### 固定基线与验收

- 基线工作流：用户真实人物迁移 MMH3Director.json，REF2VA Turbo4、一采直出、15 秒、365 帧、24 fps、0.4 MP、7 图、1 参考视频、1 音频和原提示词。
- 基线失败点：Diffusion 第一步安全 FP16 MLP fc2 前，每 rank PyTorch active 为 13.99 GiB，追加 1.40 GiB FP16 输入失败。
- 硬性显存验收：以PyTorch active同一口径比较，每卡完整采样峰值至少下降6GiB；即不高于7.99GiB。整卡NVIDIA物理占用单独报告，不与active混算。
- 努力目标：PyTorch active每卡下降8GiB；即不高于5.99GiB。
- 功能验收：固定 15 秒人物迁移工作流完成 Text Encoder、4 步双卡 Diffusion、VAE/音频解码和视频保存，输出有限且可正常解码。
- 性能与质量：不得通过全局FP16、CPU逐层错峰、减少输入、降低时长或关闭双卡采样换取通过；允许在已证明不降低速度且不改变数值的块边界回收CUDA allocator cache。固定种子不得出现NaN、黑屏或噪声回归。

### Task 12：安全 FP16 临时张量生命周期

- [x] 先用失败测试锁定 x / scale FP32 副本、FC1 投影存活、Attention 输出存活和 rank 切片共享完整底层存储。
- [x] 在 Raylight 安全 FP16 补丁中使用原位二次幂缩放，FC1 激活后释放 projected/gate/up，门控累积后释放 Attention 输出。
- [x] 让 rank 本地 hidden/rope 切片拥有独立存储，不长期引用完整序列。
- [x] 运行聚焦回归：49 passed，2 subtests passed。
- [x] 部署唯一工作树并完成文件一致性核对（候选提交部署验证464个文件一致）。
- [x] 运行固定15秒人物迁移工作流，记录每rank active/reserved峰值、NVIDIA峰值、每步速度、RAM/Commit/Pagefile和输出媒体质量。
- [x] active峰值7.42GiB，相对13.99GiB下降约6.57GiB，达到硬门槛；11.76GiB reserved与13005/12873MiB物理峰值单独报告，8GiB active努力目标未达到。
- [x] 将成功和失败实验追加到 O11 测试汇总及中英文发布说明。

### Task 12验收结果

- 固定Turbo4完整采样1456.12秒、端到端1823.77秒；原完整4步对照为1999.41/2332.86秒，分别快27.17%/21.82%。
- 采样窗口GPU0/GPU1物理峰值由16215/16230MiB降至13005/12873MiB；物理降幅约3.14/3.28GiB，没有冒充达到6GiB。
- `XFORMERS_QUERY_CHUNK_TOKENS=16384` 的实际形状数值对照max/mean abs diff均为0；50层稳态间隔8.08→6.72秒。
- 正式输出为864×480、365帧、15.209秒，H.264/AAC全量解码通过、无黑段；与此前正常4步结果逐帧SSIM为0.9416。
- safe-FP16 worker默认启用验收过的块边界cache trim，显式环境值0仍被保留；普通启动不再依赖额外脚本参数。
