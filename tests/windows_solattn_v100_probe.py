"""Execute the distributed MiniMax H3 Sol-Attn layout on a real V100."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

try:
    from .path_helpers import comfy_root as resolve_comfy_root
except ImportError:
    from path_helpers import comfy_root as resolve_comfy_root


def _load_sol_package(package_path: Path):
    module_name = "_raylight_solattn_v100_probe"
    spec = importlib.util.spec_from_file_location(
        module_name,
        package_path / "__init__.py",
        submodule_search_locations=[str(package_path)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Sol-Attn from {package_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path)
    args = parser.parse_args()

    comfy_root = resolve_comfy_root()
    sys.path.insert(0, str(comfy_root))
    package = _load_sol_package(
        comfy_root / "custom_nodes" / "ComfyUI-MiniMaxH3-SolAttn-V100"
    )
    nodes = sys.modules[f"{package.__name__}.nodes"]
    if args.library is None:
        nodes._load_extension()
    else:
        torch.ops.load_library(str(args.library.resolve()))
        nodes._EXTENSION_LOADED = True

    torch.cuda.set_device(0)
    generator = torch.Generator(device="cuda:0").manual_seed(20260910)
    shape = (1, 28, 1024, 128)
    q = torch.randn(shape, generator=generator, device="cuda:0", dtype=torch.float16)
    k = torch.randn(shape, generator=generator, device="cuda:0", dtype=torch.float16)
    v = torch.randn(shape, generator=generator, device="cuda:0", dtype=torch.float16)
    options = {
        nodes.TAU_KEY: 0.75,
        nodes.PREFIX_TOKENS_KEY: 128,
        nodes.TOPK_KEY: 8,
    }
    output = nodes._sparse_attn(q, k, v, 28, None, options)
    torch.cuda.synchronize()
    if output is None:
        raise RuntimeError("Sol-Attn sparse CUDA path returned no output")
    if tuple(output.shape) != (1, 1024, 28 * 128):
        raise RuntimeError(f"unexpected Sol-Attn output shape {tuple(output.shape)}")
    if not torch.isfinite(output).all().item():
        raise RuntimeError("Sol-Attn output contains NaN or Inf")
    print(
        {
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "input_shape": shape,
            "output_shape": tuple(output.shape),
            "output_abs_mean": float(output.abs().mean().item()),
            "finite": True,
        },
        flush=True,
    )
