from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "minimax-h3" / "benchmark_text_encoder_attention.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("benchmark_text_encoder_attention", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_contract_exposes_backend_sequence_device_and_json_output() -> None:
    module = _load_script()

    args = module.build_parser().parse_args(
        ["--backend", "XFORMERS_GQA", "--sequence", "6992", "--json-output", "result.json"]
    )

    assert args.backend == "XFORMERS_GQA"
    assert args.sequence == 6992
    assert args.device == "cuda:0"
    assert args.dtype == "float32"
    assert args.json_output == Path("result.json")


def test_math_probe_rejects_long_sequence_before_cuda_allocation() -> None:
    module = _load_script()

    with pytest.raises(ValueError, match="2048"):
        module.benchmark_attention("MATH_DEBUG", 2049, device="cuda:0")


def test_capacity_peak_excludes_random_fixture_construction_workspace() -> None:
    module = _load_script()
    source = inspect.getsource(module.benchmark_attention)

    assert source.rfind("torch.randn") < source.rfind("torch.cuda.reset_peak_memory_stats")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="capacity probe contract requires CUDA")
def test_probe_returns_stable_json_contract_and_real_finite_output() -> None:
    module = _load_script()

    result = module.benchmark_attention("AUTO", 64, device="cuda:0")

    assert set(result) == {
        "backend",
        "requested_backend",
        "sequence",
        "q_heads",
        "kv_heads",
        "head_dim",
        "dtype",
        "input_mib",
        "peak_mib",
        "elapsed_seconds",
        "finite",
        "math_fallback_detected",
    }
    assert result["backend"] in {"XFORMERS_GQA", "TORCH_EFFICIENT"}
    assert result["requested_backend"] == "AUTO"
    assert result["sequence"] == 64
    assert result["q_heads"] == 32
    assert result["kv_heads"] == 4
    assert result["head_dim"] == 128
    assert result["dtype"] == "float32"
    assert result["input_mib"] > 0
    assert result["peak_mib"] >= result["input_mib"]
    assert result["elapsed_seconds"] > 0
    assert result["finite"] is True
    assert result["math_fallback_detected"] is False
    json.dumps(result, allow_nan=False)
