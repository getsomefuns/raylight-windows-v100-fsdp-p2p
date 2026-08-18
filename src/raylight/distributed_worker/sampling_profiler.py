from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any

import torch


def _safe_emit(emit_fn, message: str) -> None:
    """Keep optional diagnostics from changing a valid sampling result."""
    try:
        emit_fn(message)
    except Exception:
        pass


def run_with_optional_profile(
    sample_fn: Callable[[], Any],
    *,
    rank: int,
    invocation: int,
    profile_factory=None,
    activities=None,
    synchronize_fn=None,
    emit_fn=print,
):
    """Profile one rank-zero sampling call when explicitly requested."""
    if os.environ.get("RAYLIGHT_TORCH_PROFILE", "0") != "1" or rank != 0:
        return sample_fn()

    if profile_factory is None:
        profile_factory = torch.profiler.profile
    if activities is None:
        activities = (
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        )
    if synchronize_fn is None:
        synchronize_fn = torch.cuda.synchronize

    stack = ExitStack()
    try:
        profile_context = profile_factory(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        profile = stack.enter_context(profile_context)
    except Exception as exc:
        try:
            stack.close()
        except Exception:
            pass
        _safe_emit(
            emit_fn,
            f"[RAYLIGHT_TORCH_PROFILE] rank={rank} invocation={invocation} "
            f"setup_failed={type(exc).__name__}: {exc}",
        )
        return sample_fn()

    try:
        result = sample_fn()
    except BaseException:
        try:
            stack.close()
        except Exception:
            pass
        raise

    try:
        synchronize_fn()
    except BaseException:
        try:
            stack.close()
        except Exception:
            pass
        raise

    try:
        stack.close()
    except Exception as exc:
        _safe_emit(
            emit_fn,
            f"[RAYLIGHT_TORCH_PROFILE] rank={rank} invocation={invocation} "
            f"report_failed={type(exc).__name__}: {exc}",
        )
        return result

    try:
        _safe_emit(
            emit_fn,
            f"[RAYLIGHT_TORCH_PROFILE] rank={rank} invocation={invocation} "
            "sort=self_cuda_time_total",
        )
        table = profile.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=40,
        )
        _safe_emit(emit_fn, table)
    except Exception as exc:
        _safe_emit(
            emit_fn,
            f"[RAYLIGHT_TORCH_PROFILE] rank={rank} invocation={invocation} "
            f"report_failed={type(exc).__name__}: {exc}",
        )
    return result


__all__ = ["run_with_optional_profile"]
