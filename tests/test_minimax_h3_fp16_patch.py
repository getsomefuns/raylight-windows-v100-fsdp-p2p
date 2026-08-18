from types import SimpleNamespace

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
