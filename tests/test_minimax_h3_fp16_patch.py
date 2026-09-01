from types import SimpleNamespace
import json
import weakref

import pytest
import torch

import raylight.comfy_dist.minimax_h3_fp16 as h3fp16

from raylight.comfy_dist.minimax_h3_fp16 import (
    K_FC2,
    K_OUT_PROJ,
    SAFE_FP16_OPTION,
    activate_minimax_h3_safe_fp16_model,
    install_minimax_h3_safe_fp16_patch,
    prepare_minimax_h3_safe_fp16_worker,
    ray_unet_model_options,
    resolve_minimax_h3_safe_fp16_manual_cast,
    safe_fp16_requested,
    safe_linear_projection,
    validate_minimax_h3_api,
)


def test_safe_fp16_request_requires_flag_and_fp16_dtype():
    assert safe_fp16_requested({}) is False
    assert safe_fp16_requested({SAFE_FP16_OPTION: True, "dtype": torch.float32}) is False
    assert safe_fp16_requested({SAFE_FP16_OPTION: True, "dtype": torch.bfloat16}) is False
    assert safe_fp16_requested({SAFE_FP16_OPTION: True, "dtype": torch.float16}) is True


def test_ray_unet_safe_fp16_loader_mapping_is_explicit():
    assert ray_unet_model_options("default") == {}
    assert ray_unet_model_options("fp16") == {"dtype": torch.float16}
    assert ray_unet_model_options("fp16_h3_safe") == {
        "dtype": torch.float16,
        SAFE_FP16_OPTION: True,
    }


def test_worker_preparation_installs_only_on_v100():
    calls = []
    options = ray_unet_model_options("fp16_h3_safe")

    assert prepare_minimax_h3_safe_fp16_worker(
        options,
        compute_capability=70,
        device_name="Tesla V100-SXM2-16GB",
        rank=1,
        install_fn=lambda: calls.append("install") or True,
        emit_fn=calls.append,
    ) is True

    assert calls[0] == "install"
    assert "rank=1" in calls[1]
    assert "compute_capability=7.0" in calls[1]
    with pytest.raises(RuntimeError, match="Tesla V100"):
        prepare_minimax_h3_safe_fp16_worker(
            options,
            compute_capability=80,
            device_name="NVIDIA A100-SXM4-40GB",
            rank=0,
            install_fn=lambda: True,
        )


def test_worker_preparation_rejects_inconsistent_safe_mode():
    with pytest.raises(ValueError, match="torch.float16"):
        prepare_minimax_h3_safe_fp16_worker(
            {SAFE_FP16_OPTION: True, "dtype": torch.float32},
            compute_capability=70,
            device_name="Tesla V100-SXM2-16GB",
            rank=0,
            install_fn=lambda: True,
        )


def test_worker_preparation_rejects_non_fsdp_safe_mode():
    with pytest.raises(RuntimeError, match="requires Raylight FSDP"):
        prepare_minimax_h3_safe_fp16_worker(
            ray_unet_model_options("fp16_h3_safe"),
            compute_capability=70,
            device_name="Tesla V100-SXM2-16GB",
            rank=0,
            is_fsdp=False,
            install_fn=lambda: True,
        )


def test_safe_fp16_overrides_manual_cast_only_for_minimax_h3():
    class MiniMaxH3:
        pass

    safe_options = ray_unet_model_options("fp16_h3_safe")
    assert resolve_minimax_h3_safe_fp16_manual_cast(
        safe_options,
        MiniMaxH3(),
        torch.float32,
    ) is torch.float16
    assert resolve_minimax_h3_safe_fp16_manual_cast(
        {"dtype": torch.float16},
        MiniMaxH3(),
        torch.float32,
    ) is torch.float32

    class OtherModel:
        pass

    with pytest.raises(RuntimeError, match="MiniMaxH3"):
        resolve_minimax_h3_safe_fp16_manual_cast(
            safe_options,
            OtherModel(),
            torch.float32,
        )


def test_ray_worker_prepares_safe_fp16_before_fsdp_model_construction(monkeypatch):
    import comfy.model_management as model_management
    import raylight.comfy_dist.sd as fsdp_sd
    import raylight.distributed_worker.ray_worker as worker_module

    events = []
    monkeypatch.setattr(
        worker_module,
        "prepare_minimax_h3_safe_fp16_worker",
        lambda *args, **kwargs: events.append("prepare") or True,
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_management, "soft_empty_cache", lambda: None)
    monkeypatch.setattr(worker_module.gc, "collect", lambda: 0)

    loaded = SimpleNamespace(model=SimpleNamespace(manual_cast_dtype=torch.float16))

    def fake_fsdp_loader(*args, **kwargs):
        events.append("load")
        return loaded, {"loaded": True}

    monkeypatch.setattr(fsdp_sd, "fsdp_load_diffusion_model", fake_fsdp_loader)

    worker = worker_module.RayWorker.__new__(worker_module.RayWorker)
    worker.parallel_dict = {"is_fsdp": True, "is_xdit": True, "use_mmap": True}
    worker.compute_capability = 70
    worker.device = torch.device("cpu")
    worker.local_rank = 0
    worker.model = None
    worker.state_dict = None
    worker.active_request_key = None
    worker.lora_list = None
    worker.device_mesh = object()
    worker.is_cpu_offload = True
    worker._active_model_key = lambda *_args: "safe-key"
    worker._free_cached_aux_models = lambda: None

    worker.load_unet("model.safetensors", ray_unet_model_options("fp16_h3_safe"))

    assert events == ["prepare", "load"]
    assert worker.is_model_loaded is True


@pytest.mark.parametrize("scale", [K_OUT_PROJ, K_FC2])
def test_scaled_linear_projection_preserves_values_above_fp16_range(scale):
    layer = torch.nn.Linear(1, 1, bias=False, dtype=torch.float16)
    with torch.no_grad():
        layer.weight.fill_(2.0)

    result = safe_linear_projection(layer, torch.tensor([[60_000.0]], dtype=torch.float16), scale)

    assert result.dtype is torch.float32
    assert torch.isfinite(result).all()
    assert result.item() == pytest.approx(120_000.0, rel=2e-3)


def test_scaled_linear_projection_consumes_ephemeral_fp32_input_in_place():
    source = torch.tensor([[1024.0, -512.0]], dtype=torch.float32)

    result = safe_linear_projection(lambda value: value, source, K_FC2)

    assert source.tolist() == [[4.0, -2.0]]
    assert result.dtype is torch.float32
    assert result.tolist() == [[1024.0, -512.0]]


class _RecordingProjection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_dtype = None

    def forward(self, x):
        self.seen_dtype = x.dtype
        return x.to(torch.float32)


def test_model_activation_creates_fp32_condition_and_safe_out_projection():
    condition_proj = _RecordingProjection()
    out_proj = torch.nn.Linear(1, 1, bias=False, dtype=torch.float16)
    with torch.no_grad():
        out_proj.weight.fill_(2.0)
    block = SimpleNamespace(attn=SimpleNamespace(out_proj=out_proj), mlp=SimpleNamespace())
    model = SimpleNamespace(dtype=torch.float16, condition_proj=condition_proj, blocks=[block])

    assert activate_minimax_h3_safe_fp16_model(model) is True
    condition_proj(torch.ones(1, 1, dtype=torch.float16))
    result = h3fp16.safe_attention_output_projection(
        out_proj,
        torch.tensor([[60_000.0]], dtype=torch.float16),
    )

    assert condition_proj.seen_dtype is torch.float32
    assert block._raylight_h3_safe_fp16 is True
    assert block.mlp._raylight_h3_safe_fp16 is True
    assert result.dtype is torch.float32
    assert torch.isfinite(result).all()
    assert result.item() == pytest.approx(120_000.0, rel=2e-3)
    assert activate_minimax_h3_safe_fp16_model(model) is False


class _IdentityNorm(torch.nn.Module):
    def forward(self, x):
        return x.clone()


class _RecordingBranch(torch.nn.Module):
    def __init__(self, output_value):
        super().__init__()
        self.output_value = output_value
        self.seen_dtype = None

    def forward(self, x, **_kwargs):
        self.seen_dtype = x.dtype
        return torch.full_like(x, self.output_value, dtype=torch.float32)


class _Adaln(torch.nn.Module):
    def forward(self, _t_emb):
        zeros = torch.zeros(1, 1, dtype=torch.float32)
        ones = torch.ones(1, 1, dtype=torch.float32)
        return zeros, zeros, ones, zeros, zeros, ones


def _mod_scale_shift(h, shift, scale, segments):
    for start, stop, row in segments:
        h[start:stop].mul_(1.0 + scale[row]).add_(shift[row])
    return h


def _mod_gate(x, gate, other, segments):
    for start, stop, row in segments:
        x[start:stop].addcmul_(other[start:stop], gate[row])
    return x


class _FakeModel:
    def __init__(self, dtype=None):
        self.dtype = dtype
        self.condition_proj = _RecordingProjection()
        self.blocks = []


class _FakeMLP(torch.nn.Module):
    def forward(self, x):
        return x


class _WeakTensorState:
    tensor_ref = None
    alive_at_consumer = None


class _RecordingFC1(torch.nn.Module):
    def __init__(self, state):
        super().__init__()
        self.state = state

    def forward(self, x):
        projected = torch.cat((x, x), dim=-1)
        self.state.tensor_ref = weakref.ref(projected)
        return projected


class _ProjectionCheckingSourceLifetime(torch.nn.Module):
    def __init__(self, state):
        super().__init__()
        self.state = state

    def forward(self, x):
        self.state.alive_at_consumer = self.state.tensor_ref() is not None
        return x


class _ManualFSDPLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False, dtype=torch.float16)
        self.unshard_calls = 0
        self.reshard_calls = 0
        self.chunk_sizes = []

    def unshard(self):
        self.unshard_calls += 1

    def reshard(self):
        self.reshard_calls += 1

    def forward(self, value):
        self.chunk_sizes.append(value.shape[0])
        return super().forward(value)


class _AttentionWithWeakResult(torch.nn.Module):
    def __init__(self, state):
        super().__init__()
        self.state = state

    def forward(self, x, **_kwargs):
        result = torch.ones_like(x, dtype=torch.float32)
        self.state.tensor_ref = weakref.ref(result)
        return result


class _MLPCheckingAttentionLifetime(torch.nn.Module):
    def __init__(self, state):
        super().__init__()
        self.state = state

    def forward(self, x):
        self.state.alive_at_consumer = self.state.tensor_ref() is not None
        return torch.ones_like(x, dtype=torch.float32)


class _FakeBlock(torch.nn.Module):
    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):
        return x


def _fake_minimax_module(block_cls=_FakeBlock):
    return SimpleNamespace(
        MiniMaxH3Model=_FakeModel,
        MLP=_FakeMLP,
        DiTBlock=block_cls,
        _mod_scale_shift=_mod_scale_shift,
        _mod_gate=_mod_gate,
    )


def test_safe_block_keeps_residual_fp32_and_branches_fp16():
    fake_module = _fake_minimax_module()
    assert install_minimax_h3_safe_fp16_patch(fake_module) is True
    assert install_minimax_h3_safe_fp16_patch(fake_module) is False

    block = _FakeBlock()
    block._raylight_h3_safe_fp16 = True
    block.norm1 = _IdentityNorm()
    block.norm2 = _IdentityNorm()
    block.attn = _RecordingBranch(1.0)
    block.mlp = _RecordingBranch(2.0)
    block.adaln_proj = _Adaln()

    result = block(
        torch.tensor([[100_000.0]], dtype=torch.float32),
        torch.ones(1, 1),
        [(0, 1, 0)],
        None,
    )

    assert block.attn.seen_dtype is torch.float16
    assert block.mlp.seen_dtype is torch.float16
    assert result.dtype is torch.float32
    assert torch.isfinite(result).all()
    assert result.item() == pytest.approx(100_003.0)


def test_safe_mlp_releases_fc1_projection_before_fc2():
    fake_module = _fake_minimax_module()
    install_minimax_h3_safe_fp16_patch(fake_module)
    state = _WeakTensorState()
    mlp = _FakeMLP()
    mlp._raylight_h3_safe_fp16 = True
    mlp.fc1 = _RecordingFC1(state)
    mlp.fc2 = _ProjectionCheckingSourceLifetime(state)

    result = mlp(torch.ones(1, 1, dtype=torch.float16))

    assert state.alive_at_consumer is False
    assert result.item() == pytest.approx(0.73095703125, abs=1e-6)


def test_safe_mlp_chunks_activations_with_one_unshard_per_projection():
    mlp = _FakeMLP()
    mlp.fc1 = _ManualFSDPLinear(3, 8)
    mlp.fc2 = _ManualFSDPLinear(4, 3)
    with torch.no_grad():
        mlp.fc1.weight.copy_(torch.arange(24).reshape(8, 3).mul(0.01))
        mlp.fc2.weight.copy_(torch.arange(12).reshape(3, 4).mul(0.01))
    source = torch.arange(21, dtype=torch.float16).reshape(7, 3).mul(0.1)
    projected = mlp.fc1.forward(source)
    gate, up = projected.chunk(2, dim=-1)
    activated = torch.nn.functional.silu(gate.float()).mul(up.float())
    expected = h3fp16.safe_linear_projection(mlp.fc2.forward, activated, K_FC2)
    mlp.fc1.chunk_sizes.clear()
    mlp.fc2.chunk_sizes.clear()

    actual = h3fp16._fsdp_chunked_safe_mlp(mlp, source, chunk_tokens=3)

    torch.testing.assert_close(actual, expected)
    assert mlp.fc1.unshard_calls == 1
    assert mlp.fc2.unshard_calls == 1
    assert mlp.fc1.reshard_calls == 1
    assert mlp.fc2.reshard_calls == 1
    assert mlp.fc1.chunk_sizes == [3, 3, 1]
    assert mlp.fc2.chunk_sizes == [3, 3, 1]


def test_safe_block_releases_attention_output_before_mlp():
    fake_module = _fake_minimax_module()
    install_minimax_h3_safe_fp16_patch(fake_module)
    state = _WeakTensorState()
    block = _FakeBlock()
    block._raylight_h3_safe_fp16 = True
    block.norm1 = _IdentityNorm()
    block.norm2 = _IdentityNorm()
    block.attn = _AttentionWithWeakResult(state)
    block.mlp = _MLPCheckingAttentionLifetime(state)
    block.adaln_proj = _Adaln()

    result = block(torch.ones(1, 1), torch.ones(1, 1), [(0, 1, 0)], None)

    assert state.alive_at_consumer is False
    assert result.tolist() == [[3.0]]


def test_safe_block_reports_internal_memory_phases(monkeypatch):
    fake_module = _fake_minimax_module()
    events = []
    monkeypatch.setattr(h3fp16, "_trace_safe_fp16_memory", events.append)
    install_minimax_h3_safe_fp16_patch(fake_module)
    block = _FakeBlock()
    block._raylight_h3_safe_fp16 = True
    block.norm1 = _IdentityNorm()
    block.norm2 = _IdentityNorm()
    block.attn = _RecordingBranch(1.0)
    block.mlp = _RecordingBranch(2.0)
    block.adaln_proj = _Adaln()

    block(torch.ones(1, 1), torch.ones(1, 1), [(0, 1, 0)], None)

    assert events == [
        "block_residual_ready",
        "block_attention_branch_ready",
        "block_attention_ready",
        "block_attention_merged",
        "block_mlp_branch_ready",
        "block_mlp_ready",
        "block_mlp_merged",
    ]


def test_safe_block_transfers_branch_ownership_to_compatible_attention():
    class OwnedAttention(torch.nn.Module):
        _raylight_accepts_owned_input = True

        def __init__(self):
            super().__init__()
            self.received_list = False

        def forward(self, owned, **_kwargs):
            self.received_list = isinstance(owned, list)
            value = owned.pop()
            return torch.ones_like(value, dtype=torch.float32)

    fake_module = _fake_minimax_module()
    install_minimax_h3_safe_fp16_patch(fake_module)
    block = _FakeBlock()
    block._raylight_h3_safe_fp16 = True
    block.norm1 = _IdentityNorm()
    block.norm2 = _IdentityNorm()
    block.attn = OwnedAttention()
    block.mlp = _RecordingBranch(2.0)
    block.adaln_proj = _Adaln()

    result = block(torch.ones(2, 1), torch.ones(1, 1), [(0, 2, 0)], None)

    assert block.attn.received_list is True
    assert result.tolist() == [[4.0], [4.0]]


def test_safe_block_trims_cuda_cache_at_branch_boundaries(monkeypatch):
    fake_module = _fake_minimax_module()
    trims = []
    monkeypatch.setattr(
        h3fp16,
        "_trim_safe_fp16_cuda_cache",
        lambda: trims.append("trim"),
    )
    install_minimax_h3_safe_fp16_patch(fake_module)
    block = _FakeBlock()
    block._raylight_h3_safe_fp16 = True
    block.norm1 = _IdentityNorm()
    block.norm2 = _IdentityNorm()
    block.attn = _RecordingBranch(1.0)
    block.mlp = _RecordingBranch(2.0)
    block.adaln_proj = _Adaln()

    block(torch.ones(2, 1), torch.ones(1, 1), [(0, 2, 0)], None)

    assert trims == ["trim", "trim"]


def test_safe_fp16_memory_trace_includes_reserved_high_water(monkeypatch, capsys):
    monkeypatch.setenv("RAYLIGHT_H3_MEMORY_TRACE", "1")
    monkeypatch.setattr(h3fp16.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(h3fp16.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(h3fp16.torch.cuda, "memory_allocated", lambda _device: 50)
    monkeypatch.setattr(h3fp16.torch.cuda, "max_memory_allocated", lambda _device: 100)
    monkeypatch.setattr(h3fp16.torch.cuda, "memory_reserved", lambda _device: 150)
    monkeypatch.setattr(h3fp16.torch.cuda, "max_memory_reserved", lambda _device: 200)
    h3fp16._SAFE_TRACE_HIGH_WATER.clear()

    h3fp16._trace_safe_fp16_memory("unit_phase")

    line = capsys.readouterr().out.strip()
    payload = json.loads(line.removeprefix("[RAYLIGHT_H3_MEMORY] "))
    assert payload["allocated_bytes"] == 50
    assert payload["peak_allocated_bytes"] == 100
    assert payload["reserved_bytes"] == 150
    assert payload["peak_reserved_bytes"] == 200


def test_class_patch_activates_only_inside_explicit_safe_construction_context():
    fake_module = _fake_minimax_module()
    install_minimax_h3_safe_fp16_patch(fake_module)

    with h3fp16.minimax_h3_safe_fp16_construction(
        ray_unet_model_options("fp16_h3_safe")
    ):
        safe_model = fake_module.MiniMaxH3Model(dtype=torch.float16)
    ordinary_fp16_model = fake_module.MiniMaxH3Model(dtype=torch.float16)

    assert safe_model._raylight_h3_safe_fp16_active is True
    assert not hasattr(ordinary_fp16_model, "_raylight_h3_safe_fp16_active")


def test_signature_guard_rejects_incompatible_comfyui_api():
    class ChangedBlock:
        def forward(self, hidden_states):
            return hidden_states

    with pytest.raises(RuntimeError, match="ComfyUI MiniMax H3 API"):
        validate_minimax_h3_api(_fake_minimax_module(ChangedBlock))
