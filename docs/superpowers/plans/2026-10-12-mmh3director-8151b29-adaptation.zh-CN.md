# O15 MiniMaxH3Director 8151b29 适配实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把本机 MiniMaxH3Director 从 `576424f` 升级到官方 `8151b29`，让原有 Raylight 双 V100 FSDP/P2P 工作流继续工作，并让新增的分段 LoRA 真正进入两个远端 worker。

**Architecture:** Director 保持官方干净工作树；Raylight 在现有 `sample_single_stage` 代理边界上继续接管首采、SelfLift 与 Refine。新增的分段 LoRA 不在无权重的主进程代理上调用标准 `add_patches`，而是转换成已校验的路径/强度描述，随每次采样请求送往两个 worker，在各自的临时模型克隆上应用并在本次采样结束后释放，绝不累积到基础模型。

**Tech Stack:** Python 3.12.10、PyTorch 2.12.0+cu126、ComfyUI、Ray 2.57.0、FSDP2、CUDA IPC/P2P、pytest、PowerShell、MiniMaxH3Director `8151b29`。

**Spec:** 本文同时作为 O15 版本规格；英文对应文档为 `docs/superpowers/plans/2026-10-12-mmh3director-8151b29-adaptation.en.md`。

## Global Constraints

- 唯一允许修改的源码工作树是 `E:\ComfyUI-py310\raylight-windows-v100-fsdp-p2p`。
- `E:\ComfyUI-py310\ComfyUI\custom_nodes\raylight` 只允许由正式提交部署生成，禁止直接修改。
- 不修改 ComfyUI 核心，不把兼容代码写进 MiniMaxH3Director。
- MiniMaxH3Director 必须保持官方 `8151b29` 的干净状态，并保留 `576424f` 可恢复 bundle。
- 保留普通 Director、SelfLift、Semantic Bridge、Refine、FaceRefine、TE-Speed、SOL、实时预览和连续分段能力。
- 不允许远端采样失败后静默回退 GPU0；错误必须 fail closed。
- 分段 LoRA 必须只作用于当前片段，不能污染下一片段、基础模型或下一工作流。
- 所有生产代码按 RED→GREEN TDD 实现；完成后运行全量自动化、双卡 P2P 门禁和真实 Director 工作流。

## 已确认的版本差异

本机 `576424f` 到官方 `8151b29` 共 25 个提交、32 个文件：

| 上游变化 | Raylight 影响 | 处理 |
|---|---|---|
| Refine 画布变化后重对齐 keyframe/cond/PackedLayout | 仍走同一 `sample_single_stage` 契约 | 锁定最新真实签名和 `refine_sampling` 别名 |
| FL2V 选择运行、首尾帧、连续性修复 | Director 内部 | 升级后工作流回归，不复制代码 |
| FFV1 无损缓存及 raw/ffv1 选项 | Director 内部，依赖现有 ffmpeg | 节点导入与缓存冒烟验证 |
| 分段导出最后运行段一帧占位修复 | Director 内部 | 三段选择运行媒体检查 |
| 公共参数、混合模式与 UI 重构 | Director 内部 | 节点 schema/旧工作流加载检查 |
| 分段 LoRA | 当前无权重 Ray MODEL 代理无法执行标准 `add_patches` | 新增描述符桥和 worker 临时 LoRA 栈 |
| 自定义缓存目录，目录名加节点 ID | 解决跨工作流缓存污染 | 检查两个工作流获得不同缓存命名空间 |
| 解码音频真实采样率 | Director 内部 | 媒体探针检查音频采样率与可解码性 |
| 段间清理预取队列 | 主进程显存清理变化 | 验证不会销毁 Ray worker 基础模型或造成后段回退 |

## Review Focus

- 分段 LoRA 与 RayUNETLoader 全局 LoRA 同时启用时应叠加，而不是覆盖任一方；Task 3 的 worker 测试覆盖。
- SelfLift 低清、高清和 Refine 三次请求必须携带同一片段 LoRA；Task 2/3 的请求测试覆盖。
- 从带 LoRA 片段切到无 LoRA 或不同 LoRA 片段时不得残留；Task 3 的连续请求测试覆盖。
- 新版 `refine_sampling` 导入的函数别名必须被桥接并可卸载恢复；Task 1 的模块别名测试覆盖。
- Director 段间清理和 Raylight 采样后清理同时发生时不得清空新活动 patcher；Task 4 重跑 1.16.1 生命周期回归与真实多段工作流。

---

### Task 1: 锁定 8151b29 契约与所有采样别名

**Files:**
- Modify: `src/raylight/integrations/mmh3director/bridge.py`
- Modify: `tests/test_mmh3director_bridge.py`

**Interfaces:**
- Consumes: Director `director.core_sampling.sample_single_stage` 的 25 个 keyword-only 参数。
- Produces: `SUPPORTED_DIRECTOR_COMMIT = "8151b291fcacee5ddb70c6493eaed4f831bc6867"`，以及对 `executor_core`、`selflift.sample`、`refine_sampling` 已导入别名的统一桥接。

- [ ] **Step 1: 先写失败测试**

```python
def test_latest_director_contract_and_refine_alias_are_pinned(isolated_sample_bridge):
    assert bridge.SUPPORTED_DIRECTOR_COMMIT == "8151b291fcacee5ddb70c6493eaed4f831bc6867"
    refine = ModuleType("ComfyUI_MiniMaxH3_Director.director.refine_sampling")
    refine.__file__ = "C:/nodes/ComfyUI_MiniMaxH3_Director/director/refine_sampling.py"
    refine.sample_single_stage = _supported_sample_single_stage
    with mock.patch.dict(sys.modules, {refine.__name__: refine}, clear=False):
        bridge.install_mmh3director_sample_bridge(require_director=True)
        assert refine.sample_single_stage is not _supported_sample_single_stage
        bridge.uninstall_mmh3director_sample_bridge_for_test()
        assert refine.sample_single_stage is _supported_sample_single_stage
```

- [ ] **Step 2: 验证 RED**

Run: `E:\ComfyUI-py310\Python312\python.exe -m pytest tests/test_mmh3director_bridge.py::test_latest_director_contract_and_refine_alias_are_pinned -vv`

Expected: FAIL，当前支持提交仍为 `576424f`。

- [ ] **Step 3: 最小实现**

把 `SUPPORTED_DIRECTOR_COMMIT` 更新为完整 `8151b29...`；保留签名的逐字段 fail-closed 校验。现有别名扫描已按模块身份工作，不增加硬编码模块列表，只让测试锁定新版 `refine_sampling`。

- [ ] **Step 4: 验证 GREEN**

Run: `E:\ComfyUI-py310\Python312\python.exe -m pytest tests/test_mmh3director_bridge.py -q`

Expected: 全部通过。

- [ ] **Step 5: Commit**

```powershell
git add src/raylight/integrations/mmh3director/bridge.py tests/test_mmh3director_bridge.py
git commit -m "test: pin Director 8151b29 sampling contract"
```

### Task 2: 主进程分段 LoRA 描述符桥

**Files:**
- Create: `src/raylight/integrations/mmh3director/segment_lora.py`
- Modify: `src/raylight/integrations/mmh3director/proxy.py`
- Modify: `src/raylight/integrations/mmh3director/bridge.py`
- Create: `tests/test_mmh3director_segment_lora.py`
- Modify: `tests/test_mmh3director_proxy.py`
- Modify: `tests/test_mmh3director_bridge.py`

**Interfaces:**
- Consumes: Director `apply_segment_loras(model, rows, state_dict_cache=None)`。
- Produces: `RayMMH3DirectorModelProxy.segment_loras: tuple[dict[str, object], ...]`、`attach_segment_loras(model, rows)`、请求字段 `segment_loras`。

- [ ] **Step 1: 先写代理隔离失败测试**

```python
def test_segment_lora_descriptors_are_clone_local():
    parent = _proxy()
    child = parent.with_segment_loras([{"path": "a.safetensors", "strength_model": 0.7}])
    assert parent.segment_loras == ()
    assert child.segment_loras == ({"path": "a.safetensors", "strength_model": 0.7},)
```

- [ ] **Step 2: 先写 Director 包装器失败测试**

```python
def test_segment_lora_wrapper_uses_descriptors_for_proxy_and_original_for_normal_model(tmp_path):
    proxy = _proxy()
    wrapped = make_segment_lora_wrapper(mock.Mock(return_value="local"))
    with mock.patch("folder_paths.get_full_path_or_raise", return_value=str(tmp_path / "x.safetensors")):
        adapted = wrapped(proxy, [{"name": "x.safetensors", "strength": 0.5, "active": True}])
    assert adapted.segment_loras[0]["strength_model"] == 0.5
```

- [ ] **Step 3: 验证 RED**

Run: `E:\ComfyUI-py310\Python312\python.exe -m pytest tests/test_mmh3director_segment_lora.py tests/test_mmh3director_proxy.py -q`

Expected: FAIL，代理没有 `segment_loras`/`with_segment_loras`，包装器不存在。

- [ ] **Step 4: 实现最小描述符路径**

```python
def with_segment_loras(self, descriptors):
    cloned = self.clone()
    cloned.segment_loras = tuple(copy.deepcopy(tuple(descriptors)))
    return cloned
```

包装器对普通 MODEL 原样调用官方函数；只对 Ray 代理解析最多 8 条 active、非零 LoRA，用 `folder_paths.get_full_path_or_raise("loras", name)` 得到绝对路径，不在主进程加载权重。安装器同时替换 `segment_loras` 模块和 `executor_core` 中指向原函数的精确别名，卸载测试必须恢复。

- [ ] **Step 5: 把描述符加入每个远端请求**

```python
request["segment_loras"] = [dict(item) for item in model.segment_loras]
```

无分段 LoRA 时发送空列表；首采、SelfLift、Refine 都从当前 `seg_model` 读取同一描述符。

- [ ] **Step 6: 验证 GREEN**

Run: `E:\ComfyUI-py310\Python312\python.exe -m pytest tests/test_mmh3director_segment_lora.py tests/test_mmh3director_proxy.py tests/test_mmh3director_bridge.py -q`

Expected: 全部通过；普通 MODEL 路径仍调用官方原函数。

- [ ] **Step 7: Commit**

```powershell
git add src/raylight/integrations/mmh3director tests/test_mmh3director_segment_lora.py tests/test_mmh3director_proxy.py tests/test_mmh3director_bridge.py
git commit -m "feat: bridge Director segment LoRAs to Ray workers"
```

### Task 3: worker 临时 LoRA 栈与无污染清理

**Files:**
- Modify: `src/raylight/integrations/mmh3director/segment_lora.py`
- Modify: `src/raylight/distributed_worker/ray_worker.py`
- Modify: `tests/test_mmh3director_worker.py`
- Modify: `tests/test_mmh3director_segment_lora.py`

**Interfaces:**
- Consumes: 请求字段 `segment_loras=[{"path": str, "strength_model": float}]`。
- Produces: `apply_worker_segment_loras(model, descriptors, *, is_fsdp, is_quant, dynamic_sidecar)`；返回临时克隆，不改变 `self.model`。

- [ ] **Step 1: 写失败测试：基础模型不被替换**

```python
def test_worker_segment_lora_applies_to_temporary_sampling_clone_only():
    base = object()
    patched = object()
    with mock.patch.object(segment_lora, "_load_lora_state", return_value={"lora": 1}), mock.patch(
        "raylight.comfy_dist.sd.load_lora_for_models_quantized", return_value=patched
    ):
        out = segment_lora.apply_worker_segment_loras(
            base, [{"path": "x.safetensors", "strength_model": 0.8}],
            is_fsdp=True, is_quant=True, dynamic_sidecar=False,
        )
    assert out is patched
    assert base is not patched
```

- [ ] **Step 2: 写失败测试：连续三次请求不累积**

第一次 `A`、第二次空列表、第三次 `B`；断言第二次采样拿到基础克隆，第三次只加载 B，不能包含 A。

- [ ] **Step 3: 验证 RED**

Run: `E:\ComfyUI-py310\Python312\python.exe -m pytest tests/test_mmh3director_segment_lora.py tests/test_mmh3director_worker.py -q`

Expected: FAIL，worker 尚未消费 `segment_loras`。

- [ ] **Step 4: 实现 worker 应用顺序**

在 `sample_mmh3director_request` 中完成 SigmaShift 临时克隆后、重放 SOL/TE 前调用：

```python
model_for_sampling = apply_worker_segment_loras(
    model_for_sampling,
    request.get("segment_loras") or (),
    is_fsdp=bool(self.parallel_dict.get("is_fsdp")),
    is_quant=bool(self.parallel_dict.get("is_quant")),
    dynamic_sidecar=bool(comfy_memory_management.aimdo_enabled),
)
```

FSDP 量化模型复用现有 `load_lora_for_models_quantized(..., defer_device_move=True)`；非量化 FSDP 使用 `load_lora_for_models`；非 FSDP 使用 ComfyUI 官方加载器。`finally` 只清理临时 `model_for_sampling`，基础 `self.model` 不变。

- [ ] **Step 5: 增加最近 LoRA state 缓存**

每个 worker 只缓存最近两份按 `(path, size, mtime_ns)` 标识的 CPU state dict；第三份进入时逐出最旧项。`_reset_active_model()`、`set_lora_list()` 和 `kill()` 清空该缓存，避免跨模型/跨工作流无限占用内存。

- [ ] **Step 6: 验证 GREEN**

Run: `E:\ComfyUI-py310\Python312\python.exe -m pytest tests/test_mmh3director_segment_lora.py tests/test_mmh3director_worker.py tests/test_mmh3director_lifecycle.py -q`

Expected: 全部通过；失败请求后的下一热请求仍不污染。

- [ ] **Step 7: Commit**

```powershell
git add src/raylight/integrations/mmh3director/segment_lora.py src/raylight/distributed_worker/ray_worker.py tests/test_mmh3director_segment_lora.py tests/test_mmh3director_worker.py
git commit -m "feat: apply per-segment LoRAs inside FSDP workers"
```

### Task 4: 升级官方 Director 并执行接口/回归门禁

**Files:**
- Create (ignored local evidence): `.local/backups/director/ComfyUI_MiniMaxH3_Director-576424f.bundle`
- External clean checkout update: `E:\ComfyUI-py310\ComfyUI\custom_nodes\ComfyUI_MiniMaxH3_Director`
- Test: `tests/test_mmh3director_bridge.py`

**Interfaces:**
- Consumes: Tasks 1–3 的 `8151b29` 契约与分段 LoRA桥。
- Produces: 官方干净 `8151b29` 本机节点和可恢复的 `576424f` bundle。

- [ ] **Step 1: 保存旧版本并校验 bundle**

Run:

```powershell
git -C <Director> bundle create <Raylight>/.local/backups/director/ComfyUI_MiniMaxH3_Director-576424f.bundle 576424f
git bundle verify <bundle>
```

Expected: bundle 包含 `576424f`，验证通过。

- [ ] **Step 2: 快进官方节点**

Run: `git -C <Director> merge --ff-only origin/main`

Expected: `HEAD=8151b291fcacee5ddb70c6493eaed4f831bc6867`，`git status --short` 为空。

- [ ] **Step 3: 运行真实节点导入和 schema 测试**

Run:

```powershell
& 'E:\ComfyUI-py310\Python312\python.exe' -c "import pathlib,sys; sys.path.insert(0,'E:/ComfyUI-py310/ComfyUI'); sys.path.insert(0,'E:/ComfyUI-py310/ComfyUI/custom_nodes'); import ComfyUI_MiniMaxH3_Director as p; from ComfyUI_MiniMaxH3_Director.nodes.director import MiniMaxH3Director as D; from ComfyUI_MiniMaxH3_Director.director.core_sampling import sample_single_stage; from raylight.integrations.mmh3director.bridge import validate_mmh3director_sample_single_stage; required={'MiniMaxH3Director','MiniMaxH3DirectorRefine','MiniMaxH3DirectorSelfLift','MiniMaxH3DirectorFaceRefine','MiniMaxH3DirectorSemanticBridge'}; assert required <= set(p.NODE_CLASS_MAPPINGS); req=D.INPUT_TYPES()['required']; assert {'cache_frames_codec','cache_name','export_pre_face_refine'} <= set(req); assert D.RETURN_NAMES[-2:]==('images_pre_refine','images_pre_face_refine'); assert pathlib.Path('E:/ComfyUI-py310/ComfyUI/custom_nodes/ComfyUI_MiniMaxH3_Director/web/js/minimax_segment_loras.js').is_file(); assert validate_mmh3director_sample_single_stage(sample_single_stage)=='stage'; print('DIRECTOR_8151B29_SCHEMA_OK')"
```

Expected: 输出 `DIRECTOR_8151B29_SCHEMA_OK`。这同时确认 Director/Refine/SelfLift/FaceRefine/Semantic Bridge 节点注册、新版缓存参数、`images_pre_face_refine` 输出、分段 LoRA 前端资源和真实采样签名。

- [ ] **Step 4: 运行兼容回归**

Run: `E:\ComfyUI-py310\Python312\python.exe -m pytest tests/test_mmh3director_bridge.py tests/test_mmh3director_worker.py tests/test_mmh3director_lifecycle.py tests/test_mmh3director_proxy.py tests/test_mmh3director_preview_transport.py tests/test_mmh3director_segment_lora.py tests/test_fsdp_mixed_precision.py -q`

Expected: 全部通过，包含 1.16.1 多段 patcher 生命周期四个保护测试。

- [ ] **Step 5: Commit**

只提交 Raylight 测试变化；Director 保持独立官方仓库，不进入 Raylight commit。

### Task 5: 双 V100 实机验收、版本和文档

**Files:**
- Modify: `__init__.py`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `README_EN.md`
- Create: `docs/releases/2026-10-12-o15-mmh3director-8151b29.zh-CN.md`
- Create: `docs/releases/2026-10-12-o15-mmh3director-8151b29.en.md`
- Create: `docs/testing/minimax-h3/O15_MMH3DIRECTOR_8151B29_2026-10.zh-CN.md`
- Create: `docs/testing/minimax-h3/O15_MMH3DIRECTOR_8151B29_2026-10.en.md`

**Interfaces:**
- Consumes: 官方 `8151b29` + Tasks 1–4 的适配实现。
- Produces: Raylight `1.17.0`、部署副本、可追溯性能/质量记录。

- [ ] **Step 1: 全量自动化与静态检查**

Run:

```powershell
E:\ComfyUI-py310\Python312\python.exe -m ruff check --no-cache src/raylight/integrations/mmh3director src/raylight/distributed_worker/ray_worker.py tests/test_mmh3director_*.py
E:\ComfyUI-py310\Python312\python.exe -m pytest tests -q -p no:cacheprovider
```

Expected: Ruff 和全部 pytest 通过。

- [ ] **Step 2: 部署候选提交并跑硬件门禁**

停止 ComfyUI/Ray，提交候选版本，再执行 `scripts/deploy-to-comfyui.ps1`；部署 SHA 必须等于正式工作树 HEAD。运行 `verify-windows-v100.ps1 -RunP2PProbe`，双卡 TCC、包版本和 P2P ≥50 GiB/s 全部通过。

- [ ] **Step 3: 真实工作流基线回归**

使用 O14 固定 864×480、124 帧、24 fps、Turbo4 + SOL + SelfLift + Refine 工作流，记录端到端、各采样阶段、显存/物理内存/Commit、帧唯一性和音频 NaN/Inf。验收：画面/音频不劣化，采样入口全部远端双 rank，不能本地回退。

- [ ] **Step 4: 分段 LoRA 实机回归**

运行两段短工作流：第一段启用实际 Turbo LoRA，第二段使用不同栈或空栈。验收：日志/请求显示每 rank 使用相同描述符；两段均输出；第二段不继承第一段 LoRA；worker 热复用且无 OOM/actor death。

- [ ] **Step 5: Director 新功能冒烟**

分别验证自定义缓存目录隔离、raw/ffv1 缓存、选择运行分段导出、音频真实采样率、Refine 放大和段间清理。内部功能不宣称 Raylight 加速，但必须与代理共存。

- [ ] **Step 6: 写入 1.17.0 中英文发布与测试记录**

文档区分“Director 升级直接获得”“Raylight 新增适配”“未改变的能力”；性能只写本轮实测，不推测加速。

- [ ] **Step 7: Commit、部署、远端同步**

```powershell
git add __init__.py pyproject.toml README.md README_EN.md docs src tests
git commit -m "release: support MiniMaxH3Director 8151b29"
git push fsdp HEAD:main
```

Expected: 正式工作树、`custom_nodes/raylight/.raylight-deployed-commit` 与 GitHub `main` SHA 三方一致；ComfyUI/Ray 无遗留进程。

## 完成标准

- 官方 Director 本机版本为干净 `8151b29`，旧 `576424f` bundle 可验证恢复。
- 原有 O14 SelfLift/Refine/FaceRefine/实时预览/TE/SOL 路径不回退。
- 新增分段 LoRA 真实在两个 FSDP worker 内应用，并能在片段之间完全撤销。
- 自定义缓存目录解决两个工作流节点 ID 相同导致的缓存互相污染。
- 全量自动化、P2P 门禁和至少两组真实双卡工作流通过。
- 版本、README、发布说明、测试记录、部署副本和 GitHub 保持一致。
