from __future__ import annotations

from unittest import mock

import torch

from raylight.distributed_worker import sampling_lifecycle
from raylight.integrations.mmh3director import bridge, nodes


def _metadata(rank):
    return {
        "rank": rank,
        "model_family": "minimax_h3",
        "sampling_settings": {"shift": 12.0, "audio_shift": 3.0},
        "latent_channels": 32,
        "latent_dimensions": 3,
        "spacial_downscale_ratio": 16,
        "temporal_downscale_ratio": 4,
        "noise_scale": 1.0,
    }


class _MetadataRemote:
    def __init__(self, actor):
        self.actor = actor

    def remote(self):
        self.actor.metadata_calls += 1
        return _metadata(self.actor.rank)


class _SampleRemote:
    def __init__(self, actor):
        self.actor = actor

    def remote(self, request):
        if hasattr(self.actor, "trace"):
            self.actor.trace.append(f"sample-rank-{self.actor.rank}")
        self.actor.sample_calls += 1
        self.actor.operation_ids.append(request["operation_id"])
        return {
            "rank": self.actor.rank,
            "operation_id": request["operation_id"],
            "samples": request["latent_image"].clone(),
        }


class _ParallelRemote:
    def __init__(self, actor):
        self.actor = actor

    def remote(self):
        return {"clear_vram_after_sampling": self.actor.clear_after_sampling}


class _ClearRemote:
    def __init__(self, actor):
        self.actor = actor

    def remote(self):
        self.actor.clear_calls += 1
        return True


class _Actor:
    def __init__(self, rank, clear_after_sampling=False):
        self.rank = rank
        self.pid = 41000 + rank
        self.clear_after_sampling = clear_after_sampling
        self.metadata_calls = 0
        self.sample_calls = 0
        self.clear_calls = 0
        self.operation_ids = []
        self.get_mmh3director_proxy_metadata = _MetadataRemote(self)
        self.sample_mmh3director_request = _SampleRemote(self)
        self.get_parallel_dict = _ParallelRemote(self)
        self.clear_sampling_vram = _ClearRemote(self)


def _build_proxy(actors):
    ray_actors = {"workers": actors, "world_size": len(actors)}
    with mock.patch.object(nodes.ray, "get", side_effect=lambda value: value):
        return nodes.RayMMH3DirectorModelAdapter().adapt_model(ray_actors)[0]


def _dispatch(proxy, callback=None):
    latent = torch.zeros(1, 32, 2, 2, 2)
    with (
        mock.patch.object(bridge, "ensure_mmh3director_compatibility"),
        mock.patch.object(bridge.ray, "get", side_effect=lambda value: value),
        mock.patch.object(bridge.comfy.model_management, "intermediate_device", return_value=torch.device("cpu")),
        mock.patch.object(bridge.comfy.model_management, "intermediate_dtype", return_value=torch.float32),
        mock.patch.object(bridge.comfy.model_management, "unload_all_models"),
        mock.patch.object(bridge.comfy.model_management, "soft_empty_cache"),
        mock.patch.object(sampling_lifecycle.gc, "collect"),
    ):
        return bridge.dispatch_mmh3director_sample(
            proxy,
            torch.ones_like(latent),
            4,
            1.0,
            "euler",
            "simple",
            [],
            [],
            latent,
            callback=callback,
            seed=7,
        )


def test_two_segments_reuse_workers_and_issue_one_remote_call_per_rank_per_pass(capsys):
    actors = [_Actor(0), _Actor(1)]
    proxy = _build_proxy(actors)
    original_pids = [actor.pid for actor in actors]

    _dispatch(proxy)
    _dispatch(proxy)

    assert [actor.metadata_calls for actor in actors] == [1, 1]
    assert [actor.sample_calls for actor in actors] == [2, 2]
    assert [actor.pid for actor in actors] == original_pids
    assert len(set(actors[0].operation_ids)) == 2
    assert actors[0].operation_ids == actors[1].operation_ids
    output = capsys.readouterr().out
    assert output.count("MMH3Director remote sampling started") == 2
    assert output.count("MMH3Director remote sampling completed") == 2


def test_same_proxy_refine_passes_each_dispatch_exactly_once():
    actors = [_Actor(0), _Actor(1)]
    proxy = _build_proxy(actors)

    for _pass_index in range(3):
        _dispatch(proxy)

    assert [actor.sample_calls for actor in actors] == [3, 3]
    assert len(set(actors[0].operation_ids)) == 3
    assert actors[0].operation_ids == actors[1].operation_ids


def test_clear_vram_policy_uses_existing_worker_cleanup_only_when_enabled():
    keep_actors = [_Actor(0, clear_after_sampling=False), _Actor(1, clear_after_sampling=False)]
    _dispatch(_build_proxy(keep_actors))
    assert [actor.clear_calls for actor in keep_actors] == [0, 0]

    clear_actors = [_Actor(0, clear_after_sampling=True), _Actor(1, clear_after_sampling=True)]
    _dispatch(_build_proxy(clear_actors))
    assert [actor.clear_calls for actor in clear_actors] == [1, 1]


def test_director_callback_runs_once_on_the_final_finite_latent():
    actors = [_Actor(0), _Actor(1)]
    proxy = _build_proxy(actors)
    callback = mock.Mock()

    output = _dispatch(proxy, callback=callback)

    callback.assert_called_once()
    step, x0, current, total_steps = callback.call_args.args
    assert (step, total_steps) == (3, 4)
    assert torch.equal(x0, output)
    assert torch.equal(current, output)


def test_preprocessing_models_are_released_before_remote_sampling():
    actors = [_Actor(0), _Actor(1)]
    trace = []
    for actor in actors:
        actor.trace = trace
    proxy = _build_proxy(actors)
    latent = torch.zeros(1, 32, 2, 2, 2)

    with (
        mock.patch.object(bridge, "ensure_mmh3director_compatibility"),
        mock.patch.object(bridge.ray, "get", side_effect=lambda value: value),
        mock.patch.object(bridge.comfy.model_management, "intermediate_device", return_value=torch.device("cpu")),
        mock.patch.object(bridge.comfy.model_management, "intermediate_dtype", return_value=torch.float32),
        mock.patch.object(
            bridge.comfy.model_management,
            "unload_all_models",
            side_effect=lambda: trace.append("unload-main-models"),
        ) as unload_all_models,
        mock.patch.object(
            bridge.comfy.model_management,
            "soft_empty_cache",
            side_effect=lambda: trace.append("empty-main-cache"),
        ) as soft_empty_cache,
        mock.patch.object(bridge.torch.cuda, "is_available", return_value=False),
        mock.patch.object(sampling_lifecycle.gc, "collect"),
    ):
        bridge.dispatch_mmh3director_sample(
            proxy,
            torch.ones_like(latent),
            4,
            1.0,
            "euler",
            "simple",
            [],
            [],
            latent,
            seed=7,
        )

    unload_all_models.assert_called_once_with()
    soft_empty_cache.assert_called_once_with()
    assert trace[:2] == ["unload-main-models", "empty-main-cache"]
    assert trace[2:] == ["sample-rank-0", "sample-rank-1"]
