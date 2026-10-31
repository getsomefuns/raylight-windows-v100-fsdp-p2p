"""Real USP forward regression for curve checkpoints and legacy time embedders."""
import copy
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def parallel(monkeypatch):
    import raylight.distributed_modules.attention as attention
    monkeypatch.setattr(attention, "_ATTN_TYPE", "TORCH")
    monkeypatch.setattr(attention, "_SYNC_ULYSSES", True)
    monkeypatch.setattr(attention, "make_xfuser_attention", lambda *_args: None)
    import raylight.diffusion_models.minimax.xdit_context_parallel as module
    # Replace only distributed transport; real embedding and output code runs.
    monkeypatch.setattr(module, "get_sequence_parallel_world_size", lambda: 1)
    monkeypatch.setattr(module, "get_sequence_parallel_rank", lambda: 0)
    monkeypatch.setattr(module, "pad_to_world_size", lambda value, dim: (value, 0))
    monkeypatch.setattr(module, "get_sp_group", lambda: SimpleNamespace())
    return module


def make_model(curve_dtype=None):
    from comfy.ldm.minimax.model import MiniMaxH3Model
    operations = SimpleNamespace(Linear=torch.nn.Linear, RMSNorm=torch.nn.RMSNorm)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = MiniMaxH3Model(
            hidden_size=96, num_layers=0, token_refiner_num_layers=0,
            num_attention_heads=1, attention_head_dim=96, ffn_hidden_size=128,
            text_dim=96, timestep_input_dim=8, time_embed_hidden_size=16,
            time_embed_dim=8, adaln_curve_grid=3 if curve_dtype is not None else None,
            dtype=torch.float32, device="cpu", operations=operations,
        )
        model.rope.inv_freq.zero_()
        if curve_dtype is not None:
            # Reproduce assign=True replacing the constructor's FP32 buffer.
            model.adaln_t_table = torch.arange(24).reshape(3, 8).to(curve_dtype)
    return model.eval()


def inputs():
    return [torch.ones(1, 24, 1, 2, 2), torch.ones(1, 32, 2, 1)]


@pytest.mark.parametrize("table_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("sigma", [1000.0, 750.0, 0.0])
def test_curve_forward_matches_fp32_reference(parallel, table_dtype, sigma):
    model = make_model(table_dtype)
    reference = copy.deepcopy(model)
    reference.adaln_t_table = reference.adaln_t_table.float()
    context = torch.ones(1, 3, 96)
    options = {"minimax_h3_sigma_shift_audio": 12.0}
    table_before = model.adaln_t_table.clone()
    captured = []
    model.final_layer.adaln_proj.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
    )
    with torch.inference_mode():
        expected = reference._forward(inputs(), torch.tensor([sigma]), context, transformer_options=options)
        actual = parallel.usp_dit_forward(model, inputs(), torch.tensor([sigma]), context, transformer_options=options)
    for result, wanted in zip(actual, expected):
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, wanted, rtol=1e-5, atol=1e-5)
    assert captured[0].dtype is torch.float32
    offset = {1000.0: 0.0, 750.0: 4.0, 0.0: 16.0}[sigma]
    torch.testing.assert_close(captured[0], (torch.arange(8) + offset).reshape(1, 8))
    assert model.adaln_t_table.dtype is table_dtype
    assert torch.equal(model.adaln_t_table, table_before)


def test_legacy_time_embedder_forward_remains_equivalent(parallel):
    model = make_model()
    context = torch.ones(1, 3, 96)
    with torch.inference_mode():
        expected = model._forward(inputs(), torch.tensor([650.0]), context)
        actual = parallel.usp_dit_forward(model, inputs(), torch.tensor([650.0]), context)
    assert not model.use_adaln_curves
    for result, wanted in zip(actual, expected):
        torch.testing.assert_close(result, wanted, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", ["cuda:0", "cuda:1"])
def test_bf16_curve_full_embedding_and_output_on_each_v100(parallel, device):
    if not torch.cuda.is_available() or torch.cuda.device_count() <= int(device[-1]):
        pytest.skip("requires the selected CUDA device")
    model = make_model(torch.bfloat16).to(device)
    reference = copy.deepcopy(model)
    reference.adaln_t_table = reference.adaln_t_table.float()
    context = torch.ones(1, 3, 96, device=device)
    timestep = torch.tensor([750.0], device=device)
    with torch.inference_mode():
        expected = reference._forward([v.to(device) for v in inputs()], timestep, context)
        actual = parallel.usp_dit_forward(model, [v.to(device) for v in inputs()], timestep, context)
    for result, wanted in zip(actual, expected):
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, wanted, rtol=1e-5, atol=1e-5)


def test_curve_auxiliaries_are_not_runtime_parameters(parallel):
    model = make_model(torch.bfloat16)
    before = copy.deepcopy(model)
    auxiliary = {
        "adaln_basis": torch.randn(8, 2688, dtype=torch.bfloat16),
        "adaln_mean": torch.randn(2688, dtype=torch.bfloat16),
        "silu_t_emb_grid": torch.randn(3, 2688, dtype=torch.bfloat16),
    }
    checkpoint = dict(model.state_dict(), **auxiliary)
    result = model.load_state_dict(checkpoint, strict=False, assign=True)
    assert not result.missing_keys
    assert set(result.unexpected_keys) == set(auxiliary)
    with torch.inference_mode():
        expected = parallel.usp_dit_forward(before, inputs(), torch.tensor([750.0]), torch.ones(1, 3, 96))
        actual = parallel.usp_dit_forward(model, inputs(), torch.tensor([750.0]), torch.ones(1, 3, 96))
    for output, wanted in zip(actual, expected):
        torch.testing.assert_close(output, wanted, rtol=0, atol=0)
