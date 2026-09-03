# Windows 双 V100 DynamicVRAM 运行时升级实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `E:\ComfyUI-py310` 的独立运行时升级为在 Tesla V100 上真实可用的 PyTorch 2.8/CUDA 12.8 DynamicVRAM 环境，同时保持 Raylight CUDA P2P、FSDP、safe-FP16 和真实 MiniMax H3 工作流可用且可完整回退。

**Architecture:** 使用“原运行时完整备份 + 规范路径候选副本”的方式隔离风险。先用自动探针验证版本、`sm_70`、xFormers、AIMDO、CUDA IPC，再启动 ComfyUI/Ray 做双进程验证；最后才修改正式版本锁、文档并部署。新版 MiniMaxH3Director 采样入口适配不进入本计划。

**Tech Stack:** Windows 11、PowerShell、Python 3.10.11、PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128、torchaudio 2.8.0+cu128、xFormers 0.0.32.post2、comfy-aimdo 0.4.13、Ray 2.57.0、pytest、双 Tesla V100-SXM2-16GB TCC、CUDA IPC/P2P/NVLink。

**Spec:** `docs/superpowers/specs/2026-09-03-dynamic-vram-runtime-upgrade-design.md`

## Global Constraints

- 只修改 `E:\ComfyUI-py310`；不得影响其他 Python、CUDA 或 ComfyUI 环境。
- 唯一可编辑 Git 工作树为 `E:\ComfyUI-py310\raylight-windows-v100-fsdp-p2p`；不得直接编辑部署副本。
- NVIDIA 驱动保持 577.00；两张 V100 必须保持 TCC。
- 正式候选为 Python 3.10.11、torch 2.8.0+cu128、torchvision 0.23.0+cu128、torchaudio 2.8.0+cu128、xformers 0.0.32.post2、comfy-aimdo 0.4.13。
- 保留 `--disable-cuda-malloc`；不得用 `--enable-dynamic-vram` 绕过不满足要求的旧环境。
- 不把 Gloo 用作 CUDA tensor 数据面回退；CUDA collective 必须继续走项目 P2P 路径。
- 不把 DynamicVRAM 描述为透明 32 GiB 显存池。
- 任一 Gate A-F 失败即停止推广并恢复 PyTorch 2.7.0+cu126 备份。
- 新环境完成 Gate A-F 前，不删除旧运行时，不推送 GitHub，不修改正式 README 环境结论。

---

### Task 1: 固化 PyTorch 2.7 基线与运行时探针接口

**Files:**
- Create: `scripts/runtime/verify_dynamic_vram_runtime.py`
- Create: `tests/test_dynamic_vram_runtime_probe.py`
- Create: `docs/testing/runtime/DYNAMIC_VRAM_TORCH28_2026-09.zh-CN.md`

**Interfaces:**
- Produces: `collect_runtime_facts() -> dict[str, object]`
- Produces: `validate_runtime_facts(facts: dict[str, object], require_dynamic_vram: bool) -> list[str]`
- CLI: `python scripts/runtime/verify_dynamic_vram_runtime.py --comfyui-root PATH [--require-dynamic-vram] [--json PATH]`
- JSON 必须包含 Torch 四件套、CUDA runtime、设备、架构、xFormers、AIMDO、ComfyUI patcher 类型和每项 probe 结果。

- [x] **Step 1: 写版本、架构和 DynamicVRAM 判定的失败测试**

```python
def test_candidate_requires_exact_stack_sm70_and_dynamic_patcher():
    facts = {
        "torch": "2.7.0+cu126",
        "torchvision": "0.22.0+cu126",
        "torchaudio": "2.7.0+cu126",
        "xformers": "0.0.30",
        "cuda": "12.6",
        "arch_list": ["sm_70"],
        "device_capabilities": [[7, 0], [7, 0]],
        "dynamic_vram": False,
    }
    errors = validate_runtime_facts(facts, require_dynamic_vram=True)
    assert "torch must be 2.8.0+cu128" in errors
    assert "DynamicVRAM is not active" in errors
```

- [x] **Step 2: 运行测试并确认因探针模块不存在而失败**

Run: `E:\ComfyUI-py310\Python310\python.exe -m pytest tests/test_dynamic_vram_runtime_probe.py -q`

Expected: FAIL during import because `scripts.runtime.verify_dynamic_vram_runtime` does not exist.

- [x] **Step 3: 实现最小探针和 JSON 输出**

```python
EXPECTED = {
    "torch": "2.8.0+cu128",
    "torchvision": "0.23.0+cu128",
    "torchaudio": "2.8.0+cu128",
    "xformers": "0.0.32.post2",
    "cuda": "12.8",
}

def validate_runtime_facts(facts, require_dynamic_vram):
    errors = []
    for name, expected in EXPECTED.items():
        if facts.get(name) != expected:
            errors.append(f"{name} must be {expected}")
    if "sm_70" not in facts.get("arch_list", []):
        errors.append("torch wheel does not contain sm_70")
    if facts.get("device_capabilities") != [[7, 0], [7, 0]]:
        errors.append("both devices must be compute capability 7.0")
    if require_dynamic_vram and facts.get("dynamic_vram") is not True:
        errors.append("DynamicVRAM is not active")
    return errors
```

- [x] **Step 4: 增加纯 CPU 可测的 xFormers/AIMDO/IPC 结果字段校验**

```python
def test_release_probe_rejects_missing_gpu_subprobes():
    facts = candidate_facts()
    facts.update(xformers_forward=False, aimdo_devices=[], cuda_ipc_round_trip=False)
    errors = validate_runtime_facts(facts, require_dynamic_vram=True)
    assert "xFormers V100 forward failed" in errors
    assert "AIMDO did not register both devices" in errors
    assert "CUDA IPC round-trip failed" in errors
```

- [x] **Step 5: 运行探针单测和当前 2.7 基线采集**

Run:

```powershell
E:\ComfyUI-py310\Python310\python.exe -m pytest tests/test_dynamic_vram_runtime_probe.py -q
E:\ComfyUI-py310\Python310\python.exe scripts/runtime/verify_dynamic_vram_runtime.py `
  --comfyui-root E:\ComfyUI-py310\ComfyUI `
  --json E:\ComfyUI-py310\runtime-results\dynamic-vram-torch27-baseline.json
```

Expected: unit tests PASS; baseline probe exits nonzero only for candidate requirements and records the current working 2.7 stack without changing it.

- [x] **Step 6: 在专项测试汇总写入基线版本和“DynamicVRAM 未启用”的事实，不写长日志**

- [x] **Step 7: 提交探针与基线记录**

```powershell
git add scripts/runtime/verify_dynamic_vram_runtime.py tests/test_dynamic_vram_runtime_probe.py docs/testing/runtime/DYNAMIC_VRAM_TORCH28_2026-09.zh-CN.md
git commit -m "test: add DynamicVRAM runtime gate"
```

---

### Task 2: 下载候选 wheel 并验证可安装集合

**Files:**
- Modify: `docs/testing/runtime/DYNAMIC_VRAM_TORCH28_2026-09.zh-CN.md`
- Runtime artifact only: `E:\ComfyUI-py310\runtime-downloads\torch-2.8-cu128\`

**Interfaces:**
- Consumes: Task 1 runtime probe and exact version table.
- Produces: an offline-installable wheel directory and SHA256 manifest under `runtime-downloads`; these are not committed.

- [x] **Step 1: 确认 E 盘至少有 80 GiB 可用，ComfyUI 和 Ray 未运行**

Run: `Get-PSDrive E` and inspect only Python processes whose command line belongs to `E:\ComfyUI-py310`.

Expected: free space >= 80 GiB and no active ComfyUI/Ray worker before runtime mutation.

- [x] **Step 2: 仅下载、不安装候选 wheel**

Run:

```powershell
E:\ComfyUI-py310\Python310\python.exe -m pip download `
  torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 xformers==0.0.32.post2 `
  --index-url https://download.pytorch.org/whl/cu128 `
  --dest E:\ComfyUI-py310\runtime-downloads\torch-2.8-cu128
```

Expected: one Windows/Python-compatible wheel per requested package plus dependencies; no installed version changes.

- [x] **Step 3: 生成 SHA256 清单并用 pip dry-run 验证解析结果**

Run: `Get-FileHash` for every wheel and `pip install --dry-run --no-index --find-links ...` with the exact four pins.

Expected: resolver selects exactly the requested Torch stack and does not propose CPU-only torch.

- [x] **Step 4: 记录 wheel 文件名、大小、哈希和解析结论**

- [x] **Step 5: 不提交二进制，只提交更新后的专项测试记录**

```powershell
git add docs/testing/runtime/DYNAMIC_VRAM_TORCH28_2026-09.zh-CN.md
git commit -m "docs: record DynamicVRAM candidate wheels"
```

---

### Task 3: 创建可验证的完整运行时备份并升级规范路径

**Files:**
- Runtime backup: `E:\ComfyUI-py310\runtime-backups\Python310-torch2.7.0-cu126-20260903\`
- Runtime manifest: `E:\ComfyUI-py310\runtime-backups\Python310-torch2.7.0-cu126-20260903.manifest.json`
- Modify: `docs/testing/runtime/DYNAMIC_VRAM_TORCH28_2026-09.zh-CN.md`

**Interfaces:**
- Consumes: offline wheel directory from Task 2.
- Produces: canonical `E:\ComfyUI-py310\Python310` candidate plus byte-count/file-count verified rollback copy.

- [x] **Step 1: 保存原环境 freeze、关键包信息和目录统计**

Record `pip freeze`, `pip check`, `torch.utils.collect_env`, `xformers.info`, file count, byte count, and timestamp into the backup manifest.

- [x] **Step 2: 将原 `Python310` 移到精确备份目录，再复制回规范路径作为候选**

Use native PowerShell path checks before move/copy. Verify both resolved paths remain direct children of `E:\ComfyUI-py310`; do not use a wildcard or unresolved environment variable.

- [x] **Step 3: 核对候选副本与备份的文件数和总字节数一致**

Expected: both counts match before any package installation.

- [x] **Step 4: 从离线 wheel 目录安装精确 Torch 栈**

Run:

```powershell
E:\ComfyUI-py310\Python310\python.exe -m pip install --no-index `
  --find-links E:\ComfyUI-py310\runtime-downloads\torch-2.8-cu128 `
  --force-reinstall torch==2.8.0+cu128 torchvision==0.23.0+cu128 `
  torchaudio==2.8.0+cu128 xformers==0.0.32.post2
```

- [x] **Step 5: 若 flash-attn 导入出现 Torch ABI 错误，则从候选环境移除**

Expected: Raylight 不依赖它；移除原因和结果写入测试记录。

- [x] **Step 6: 运行 `pip check`，只允许保留升级前已经记录的无关冲突**

- [x] **Step 7: 运行 Task 1 探针的静态部分**

Expected: exact versions、CUDA 12.8、`sm_70` 和两张 capability 7.0 通过。

- [ ] **Step 8: 提交更新后的环境迁移记录；备份目录不入库**

```powershell
git add docs/testing/runtime/DYNAMIC_VRAM_TORCH28_2026-09.zh-CN.md
git commit -m "docs: record torch 2.8 runtime migration"
```

---

### Task 4: 验证 xFormers、AIMDO 和 CUDA IPC/P2P

**Files:**
- Modify only if RED exposes a real incompatibility: `src/raylight/distributed_worker/ray_worker.py`
- Modify only if RED exposes a real incompatibility: `src/raylight/distributed_worker/windows_p2p.py`
- Test: `tests/test_dynamic_vram_runtime_probe.py`
- Test: `tests/test_ray_worker_windows_p2p.py`
- Test: `tests/test_windows_p2p.py`
- Modify: `docs/testing/runtime/DYNAMIC_VRAM_TORCH28_2026-09.zh-CN.md`

**Interfaces:**
- Consumes: Torch 2.8 candidate and Task 1 probes.
- Produces: verified V100 attention, AIMDO two-device registration, CUDA IPC metadata and collective behavior.

- [ ] **Step 1: 在 GPU0 和 GPU1 分别运行 FP16 GEMM、stream/event 和 xFormers attention forward**

Use the tensor layout exercised by O11 GQA. Assert device placement, shape, dtype and `torch.isfinite(output).all()`.

- [ ] **Step 2: 运行 AIMDO 独立初始化探针**

Call `comfy_aimdo.control.init()` and `init_devices()` in a fresh process, reload `host_buffer`, and assert `host_buffer.lib is not None` and both device indices are registered.

- [ ] **Step 3: 运行现有 CUDA IPC、P2P all-to-all、all-gather 与大 payload 自动分块探针**

Run:

```powershell
E:\ComfyUI-py310\Python310\python.exe -m pytest tests/test_windows_p2p.py tests/test_ray_worker_windows_p2p.py -q
E:\ComfyUI-py310\Python310\python.exe tests/windows_p2p_probe.py
E:\ComfyUI-py310\Python310\python.exe tests/windows_p2p_full_correctness_probe.py
E:\ComfyUI-py310\Python310\python.exe tests/windows_p2p_ray_probe.py
```

Expected: payload correctness PASS on both ranks, no CPU/Gloo CUDA fallback, no CUDA IPC reducer signature error.

- [ ] **Step 4: 若 Torch 2.8 改变 CUDA IPC metadata 签名，先写精确失败测试**

```python
def test_cuda_ipc_export_import_round_trip_matches_torch28_reducer_contract():
    source = torch.arange(4096, device="cuda", dtype=torch.float16)
    rebuilt = import_cuda_tensor_ipc(export_cuda_tensor_ipc(source))
    assert torch.equal(source, rebuilt)
```

Run it and confirm failure points to tuple parsing or rebuild arguments before editing production code.

- [ ] **Step 5: 只实现测试要求的 Torch 2.8 兼容分支，并保留 2.7 测试夹具**

- [ ] **Step 6: 若官方 xFormers wheel 缺少 `sm_70` kernel，停止 Gate C 并评估源码构建**

Do not silently fall back to SDPA Math. Only continue after a V100-targeted xFormers build passes the same forward test; otherwise execute Task 8 rollback.

- [ ] **Step 7: 运行相关自动测试并提交实际需要的代码修复**

```powershell
E:\ComfyUI-py310\Python310\python.exe -m pytest tests/test_dynamic_vram_runtime_probe.py tests/test_windows_p2p.py tests/test_ray_worker_windows_p2p.py -q
git add src/raylight/distributed_worker tests docs/testing/runtime/DYNAMIC_VRAM_TORCH28_2026-09.zh-CN.md
git commit -m "fix: support torch 2.8 DynamicVRAM runtime"
```

If no source fix is needed, commit only the updated test record and do not create a no-op code commit.

---

### Task 5: 启动 ComfyUI 与双 Ray worker，证明 DynamicVRAM 真实启用

**Files:**
- Modify only after a failing test: `scripts/start-comfyui-windows-p2p.ps1`
- Modify only after a failing test: `src/raylight/distributed_worker/ray_worker.py`
- Test: `tests/test_windows_release_profile.py`
- Modify: `docs/testing/runtime/DYNAMIC_VRAM_TORCH28_2026-09.zh-CN.md`

**Interfaces:**
- Consumes: candidate runtime and existing launcher.
- Produces: main-process and both-worker DynamicVRAM evidence without forcing the version gate.

- [ ] **Step 1: 用现有启动脚本启动，不添加 `--enable-dynamic-vram`**

Run: `E:\ComfyUI-py310\start_raylight_p2p.ps1`

Expected: main process logs `DynamicVRAM support detected and enabled` and retains `--disable-cuda-malloc`.

- [ ] **Step 2: 通过轻量 Ray 初始化工作流创建两个 worker**

Expected: rank 0 and rank 1 each log `[Raylight][AIMDO] DynamicVRAM enabled in worker`, with `host_buffer_lib=True`.

- [ ] **Step 3: 采集主进程和 worker 内的 patcher 类型及 AIMDO 状态**

Assert main `CoreModelPatcher is ModelPatcherDynamic`; record worker `comfy.memory_management.aimdo_enabled=True` and device registration.

- [ ] **Step 4: 制造受控模型权重压力，记录动态驻留变化**

Use an existing loader-only/conditioning workflow, sample GPU/RAM/commit at one-second intervals, and confirm model weights can move without OOM or invalid output. A startup string alone is not acceptance.

- [ ] **Step 5: 若 worker 初始化失败，先在测试中复现环境传播或重复 init 问题，再做最小修复**

Expected test covers the exact failure: missing CLI values, repeated init, missing `host_buffer.lib`, or device list mismatch.

- [ ] **Step 6: 重启一次并重复 Gate D，排除只在首进程有效的偶然状态**

- [ ] **Step 7: 提交必要的代码/启动脚本修复和测试记录**

```powershell
git add scripts/start-comfyui-windows-p2p.ps1 src/raylight/distributed_worker/ray_worker.py tests/test_windows_release_profile.py docs/testing/runtime/DYNAMIC_VRAM_TORCH28_2026-09.zh-CN.md
git commit -m "fix: enable DynamicVRAM in Ray workers"
```

Skip unchanged paths and do not create a source change without a preceding RED test.

---

### Task 6: FSDP、O6、O9、O11 与真实工作流回归

**Files:**
- Modify only for proven regressions: relevant file under `src/raylight/`
- Add a targeted test beside the affected subsystem before every fix.
- Modify: `docs/testing/runtime/DYNAMIC_VRAM_TORCH28_2026-09.zh-CN.md`
- Runtime artifacts: `E:\ComfyUI-py310\runtime-results\dynamic-vram-torch28\`

**Interfaces:**
- Consumes: Gate A-D passing candidate.
- Produces: Gate E-F performance, resource and media-quality evidence.

- [ ] **Step 1: 运行完整自动测试套件**

Run: `E:\ComfyUI-py310\Python310\python.exe -m pytest -q`

Expected: no regression from the last accepted suite; every failure is classified before any fix.

- [ ] **Step 2: 运行 FSDP quant、host registration、mixed precision 和 O9 大 payload 探针**

Run targeted test modules and `tests/windows_p2p_fsdp_probe.py`; assert both ranks register, shard, gather and release successfully.

- [ ] **Step 3: 运行不依赖 Director 的短 MiniMax H3 I2V**

Record cold/warm load, Text Encoder, sampling s/it, VAE/audio decode, save time, GPU0/GPU1 board memory, process memory, RAM and commit/pagefile peaks.

- [ ] **Step 4: 运行不依赖 Director 的短 MiniMax H3 REF2VA**

Use existing source material and prompt. Validate frame count, resolution, FPS, unique-frame ratio, black-frame interval, audio finite values and output decodability.

- [ ] **Step 5: 比较 PyTorch 2.7 基线**

Expected: identical workflow sampling median slowdown <=10%; no new OOM, black/noise output, NaN/Inf, rank divergence or worker leak.

- [ ] **Step 6: 停止 worker 并验证显存回收，再进行一次热启动复用测试**

- [ ] **Step 7: 更新专项测试汇总，正文只保留关键表格，原始 JSON/CSV 作为本地附件路径**

- [ ] **Step 8: 提交真实回归记录及任何经 TDD 验证的最小修复**

```powershell
git add src tests docs/testing/runtime/DYNAMIC_VRAM_TORCH28_2026-09.zh-CN.md
git commit -m "test: validate DynamicVRAM on dual V100"
```

---

### Task 7: 正式版本锁、双语文档、部署与最终核验

**Files:**
- Modify: `requirements-windows-v100.txt`
- Modify: `README.md`
- Modify: `README_EN.md`
- Create: `docs/releases/2026-09-03-dynamic-vram-torch28.zh-CN.md`
- Create: `docs/releases/2026-09-03-dynamic-vram-torch28.en.md`
- Modify: `docs/testing/runtime/DYNAMIC_VRAM_TORCH28_2026-09.zh-CN.md`
- Test: `tests/test_windows_release_profile.py`
- Test: `tests/test_dynamic_vram_runtime_probe.py`

**Interfaces:**
- Consumes: all Gate A-F evidence.
- Produces: reproducible version lock, bilingual user documentation and a verified deployment commit.

- [ ] **Step 1: 写失败测试，要求正式版本锁为 cu128/Torch 2.8 组合**

```python
def test_windows_v100_release_profile_pins_dynamic_vram_stack():
    text = REQUIREMENTS.read_text(encoding="utf-8")
    assert "--extra-index-url https://download.pytorch.org/whl/cu128" in text
    assert "torch==2.8.0+cu128" in text
    assert "torchvision==0.23.0+cu128" in text
    assert "torchaudio==2.8.0+cu128" in text
    assert "xformers==0.0.32.post2" in text
```

- [ ] **Step 2: 运行测试确认旧 cu126/2.7 版本锁导致失败**

- [ ] **Step 3: 更新版本锁和中英文 README**

README must explain automatic activation, `--vram-headroom`, continued `--disable-cuda-malloc`, V100 limits, rollback location, and the fact that Director update compatibility remains pending.

- [ ] **Step 4: 写中英文升级说明和最终测试结论**

Include shipped changes, exact environment, measured gains/regressions, resource peaks, rejected experiments, known limits, and rollback instructions; do not paste long logs.

- [ ] **Step 5: 运行文档/版本锁测试、Ruff 和全量 pytest**

Run:

```powershell
E:\ComfyUI-py310\Python310\python.exe -m pytest tests/test_windows_release_profile.py tests/test_dynamic_vram_runtime_probe.py -q
E:\ComfyUI-py310\Python310\python.exe -m ruff check src tests
E:\ComfyUI-py310\Python310\python.exe -m pytest -q
```

- [ ] **Step 6: 提交正式版本与文档**

```powershell
git add requirements-windows-v100.txt README.md README_EN.md docs/releases docs/testing/runtime tests
git commit -m "release: enable DynamicVRAM on Windows V100"
```

- [ ] **Step 7: 从干净 HEAD 部署到 ComfyUI**

Run:

```powershell
.\scripts\deploy-to-comfyui.ps1 -ComfyUIRoot E:\ComfyUI-py310\ComfyUI
.\scripts\deploy-to-comfyui.ps1 -ComfyUIRoot E:\ComfyUI-py310\ComfyUI -VerifyOnly
```

Expected: deployment metadata equals source HEAD and no tracked/extra-file divergence is reported.

- [ ] **Step 8: 从部署副本启动并重跑轻量 Gate D/E 冒烟测试**

Expected: main/worker DynamicVRAM active, P2P/FSDP active, no direct source-tree imports.

- [ ] **Step 9: 检查 Git 工作树干净并保留旧运行时备份**

Do not push GitHub unless the user explicitly requests it.

---

### Task 8: 任一门禁失败时执行完整回退

**Files:**
- Modify: `docs/testing/runtime/DYNAMIC_VRAM_TORCH28_2026-09.zh-CN.md`
- Failed candidate archive: `E:\ComfyUI-py310\runtime-failed\Python310-torch2.8.0-cu128-<timestamp>\`

**Interfaces:**
- Consumes: exact backup/candidate paths and failure evidence.
- Produces: restored canonical PyTorch 2.7 runtime and a concise failure decision record.

- [x] **Step 1: 停止属于 `E:\ComfyUI-py310` 的 ComfyUI 和 Ray 进程并确认 GPU 空闲**

- [x] **Step 2: 解析并验证候选、备份、失败归档三个绝对路径均位于 `E:\ComfyUI-py310` 下**

- [x] **Step 3: 将失败候选移到 `runtime-failed`，把备份恢复为规范 `Python310`**

No recursive deletion is allowed during rollback.

- [x] **Step 4: 用恢复环境运行版本、xFormers、CUDA、P2P 和 ComfyUI 启动冒烟测试**

Expected: torch 2.7.0+cu126, xformers 0.0.30, `sm_70`, P2P pass, and only the known DynamicVRAM fallback warning remains.

- [x] **Step 5: 在专项记录中写明失败 Gate、根因证据、未推广理由和后续可行路径**

- [x] **Step 6: 提交失败结论，但不修改正式版本锁或 README 的已支持环境**

```powershell
git add docs/testing/runtime/DYNAMIC_VRAM_TORCH28_2026-09.zh-CN.md
git commit -m "docs: record DynamicVRAM upgrade rollback"
```

## Final Completion Checklist

- [ ] Gate A-F 全部通过，或 Task 8 已完整恢复基线。
- [ ] `E:\ComfyUI-py310` 外没有任何环境改动。
- [ ] 唯一工作树干净；部署副本来源提交号可核验。
- [ ] 旧运行时备份仍存在且 manifest 完整。
- [ ] GitHub 未在没有明确授权时推送。
- [ ] 新版 MiniMaxH3Director 适配仍作为下一项独立工作，没有混入本次验证。

## 2026-09-03 TCC 兼容候选续行

官方 0.4.13 wheel 的失败与回退记录仍然有效，但不再是本计划的最终终止点。经用户确认后，本计划增加受约束的源码构建分支：

- 固定 comfy-aimdo v0.4.13、Detours v4.0.1 和 Windows SDK NuGet 10.0.26100.1；
- 本地版本为 `0.4.13+tcc1`，WDDM 仍优先，TCC 仅在 NVML 压力设备成功映射时启用；
- 禁用或缺失 NVML 时必须失败关闭；
- wheel、DLL、补丁均进入 SHA-256 构建清单，安装脚本默认拒绝修改规范 `Python310`；
- 候选已通过双 V100、sm_70、xFormers FP16、CUDA IPC、AIMDO 双卡注册、无 NVML 失败关闭和真实 ComfyUI DynamicVRAM 启动门槛；
- Task 5/6 的双 worker、P2P/FSDP、真实 I2V/REF2VA、性能、质量和回收验证继续执行。完成前不修改正式版本锁，不推广候选，不处理新版 Director 入口适配。
