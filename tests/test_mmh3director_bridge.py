from __future__ import annotations

import sys
from types import ModuleType
from unittest import mock

import pytest
import torch

import comfy.sample
import comfy.nested_tensor
from raylight.integrations.mmh3director import bridge
from raylight.integrations.mmh3director.metadata import MMH3DirectorModelMetadata
from raylight.integrations.mmh3director.proxy import RayMMH3DirectorModelProxy


EXPECTED_SAMPLE_SINGLE_STAGE_PARAMETERS = (
    "model",
    "positive",
    "negative",
    "latent",
    "seed",
    "cfg",
    "steps",
    "sampler_name",
    "scheduler",
    "shift_video",
    "shift_audio",
    "on_phase",
    "on_step_preview",
    "preview_every",
    "denoise",
    "phase_name",
    "sigmas",
    "apply_shift",
    "after_shift",
    "enable_tiling",
    "tile_count",
    "tile_overlap",
    "shift_cache",
    "on_step_state",
    "zero_noise",
    "noise_override",
)


def _supported_sample_single_stage(
    *,
    model,
    positive,
    negative,
    latent,
    seed: int,
    cfg: float,
    steps: int,
    sampler_name: str,
    scheduler: str,
    shift_video: float = 12.0,
    shift_audio: float = 3.0,
    on_phase=None,
    on_step_preview=None,
    preview_every: int = 1,
    denoise: float = 1.0,
    phase_name: str = "sample",
    sigmas=None,
    apply_shift: bool = True,
    after_shift=None,
    enable_tiling: bool = False,
    tile_count: int = 2,
    tile_overlap: int = 128,
    shift_cache=None,
    on_step_state=None,
    zero_noise: bool = False,
    noise_override=None,
):
    del (
        model,
        positive,
        negative,
        latent,
        seed,
        cfg,
        steps,
        sampler_name,
        scheduler,
        shift_video,
        shift_audio,
        on_phase,
        on_step_preview,
        preview_every,
        denoise,
        phase_name,
        sigmas,
        apply_shift,
        after_shift,
        enable_tiling,
        tile_count,
        tile_overlap,
        shift_cache,
        on_step_state,
        zero_noise,
        noise_override,
    )


def _supported_continue_remask(model, latent, sigmas):
    del model, latent, sigmas


_supported_continue_remask.__name__ = "install_continue_prefix_remask"
_supported_continue_remask.__module__ = "ComfyUI_MiniMaxH3_Director.director.h3_latent_continue"


def _metadata(model_family="minimax_h3"):
    return MMH3DirectorModelMetadata(
        model_family=model_family,
        sampling_settings={"shift": 12.0, "audio_shift": 3.0},
        latent_channels=32,
        latent_dimensions=3,
        spacial_downscale_ratio=16,
        temporal_downscale_ratio=4,
        noise_scale=1.0,
    )


class _SampleRemote:
    def __init__(self, rank, samples=None):
        self.rank = rank
        self.samples = samples
        self.requests = []

    def remote(self, request):
        self.requests.append(request)
        source = request["latent_image"] if self.samples is None else self.samples
        samples = source._copy() if getattr(source, "is_nested", False) else source.clone()
        result = {
            "rank": self.rank,
            "operation_id": request["operation_id"],
            "samples": samples,
        }
        if request.get("capture_step_state") and self.rank == 0:
            result["step_state"] = {
                "step": int(request["steps"]) - 1,
                "total_steps": int(request["steps"]),
                "x0": samples,
                "x": samples + 1,
            }
        return result


class _ParallelRemote:
    def remote(self):
        return {"clear_vram_after_sampling": False}


class _ClearRemote:
    def remote(self):
        return True


class _Actor:
    def __init__(self, rank, samples=None):
        self.sample_mmh3director_request = _SampleRemote(rank, samples=samples)
        self.get_parallel_dict = _ParallelRemote()
        self.clear_sampling_vram = _ClearRemote()


def _proxy(model_family="minimax_h3", samples_by_rank=None):
    samples_by_rank = samples_by_rank or (None, None)
    actors = [_Actor(rank, samples_by_rank[rank]) for rank in range(2)]
    proxy = RayMMH3DirectorModelProxy(
        ray_actors={"workers": actors, "world_size": 2},
        metadata=_metadata(model_family),
    )
    return proxy, actors


@pytest.fixture
def isolated_sample_bridge():
    bridge.uninstall_mmh3director_sample_bridge_for_test()
    original = comfy.sample.sample
    yield original
    bridge.uninstall_mmh3director_sample_bridge_for_test()
    comfy.sample.sample = original
    bridge.install_mmh3director_sample_bridge()


def test_normal_model_calls_original_once_with_identical_arguments(isolated_sample_bridge):
    sentinel_model = object()
    args = (object(), 8, 4.5, "euler", "simple", ["positive"], ["negative"], object())
    kwargs = {"denoise": 0.7, "seed": 123, "disable_pbar": True}
    original = mock.Mock(return_value="local-result")
    comfy.sample.sample = original

    assert bridge.install_mmh3director_sample_bridge() is True
    assert comfy.sample.sample(sentinel_model, *args, **kwargs) == "local-result"
    original.assert_called_once_with(sentinel_model, *args, **kwargs)


def test_install_is_idempotent_and_uninstall_restores_same_function(isolated_sample_bridge):
    original = mock.Mock()
    comfy.sample.sample = original

    assert bridge.install_mmh3director_sample_bridge() is True
    wrapper = comfy.sample.sample
    assert bridge.install_mmh3director_sample_bridge() is False
    assert comfy.sample.sample is wrapper

    bridge.uninstall_mmh3director_sample_bridge_for_test()
    assert comfy.sample.sample is original


def test_proxy_dispatch_failure_never_falls_back_to_local_sampler(isolated_sample_bridge):
    original = mock.Mock()
    comfy.sample.sample = original
    bridge.install_mmh3director_sample_bridge()
    proxy, _actors = _proxy()

    with (
        mock.patch.object(bridge, "dispatch_mmh3director_sample", side_effect=RuntimeError("remote failed")),
        pytest.raises(RuntimeError, match="remote failed"),
    ):
        comfy.sample.sample(proxy, object(), 1, 1.0, "euler", "simple", [], [], object())

    original.assert_not_called()


def test_supported_director_signature_is_fixed_to_the_accepted_commit():
    assert bridge.EXPECTED_SAMPLE_SINGLE_STAGE_PARAMETERS == EXPECTED_SAMPLE_SINGLE_STAGE_PARAMETERS
    assert bridge.SUPPORTED_DIRECTOR_COMMIT == "576424fb252b2b4a538e792027f7db1a5ef376e4"
    bridge.validate_mmh3director_sample_single_stage(_supported_sample_single_stage)


def test_after_shift_serializes_only_the_supported_continue_remask():
    assert bridge.serialize_director_after_shift(None) is None
    assert bridge.serialize_director_after_shift(_supported_continue_remask) == "continue_prefix_remask_v1"


def test_after_shift_rejects_unknown_callbacks_instead_of_silently_dropping_them():
    def unknown_callback(model, latent, sigmas):
        del model, latent, sigmas

    with pytest.raises(RuntimeError, match="Unsupported MiniMaxH3Director after_shift"):
        bridge.serialize_director_after_shift(unknown_callback)


def test_missing_director_parameter_reports_version_commit_and_field():
    def missing_scheduler(*, model, positive, negative, latent, seed, cfg, steps, sampler_name, **kwargs):
        del model, positive, negative, latent, seed, cfg, steps, sampler_name, kwargs

    with pytest.raises(RuntimeError, match=r"scheduler.*576424f"):
        bridge.validate_mmh3director_sample_single_stage(missing_scheduler)


def test_renamed_director_parameter_reports_actionable_error():
    def renamed_scheduler(*, model, positive, negative, latent, seed, cfg, steps, sampler_name, schedule, **kwargs):
        del model, positive, negative, latent, seed, cfg, steps, sampler_name, schedule, kwargs

    with pytest.raises(RuntimeError, match=r"scheduler.*update.*adapter"):
        bridge.validate_mmh3director_sample_single_stage(renamed_scheduler)


def test_proxy_request_preserves_director_inputs_and_uses_every_rank():
    proxy, actors = _proxy()
    noise = torch.randn(1, 32, 2, 4, 4)
    latent = torch.randn_like(noise)
    noise_mask = torch.ones(1, 1, 2, 4, 4)
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    positive = [["positive", {"pooled": True}]]
    negative = [["negative", {}]]

    with (
        mock.patch.object(bridge, "ensure_mmh3director_compatibility"),
        mock.patch.object(bridge.ray, "get", side_effect=lambda futures: futures),
        mock.patch.object(bridge.comfy.model_management, "intermediate_device", return_value=torch.device("cpu")),
        mock.patch.object(bridge.comfy.model_management, "intermediate_dtype", return_value=torch.float32),
    ):
        output = bridge.dispatch_mmh3director_sample(
            proxy,
            noise,
            8,
            4.5,
            "euler",
            "simple",
            positive,
            negative,
            latent,
            denoise=0.65,
            disable_noise=True,
            start_step=2,
            last_step=7,
            force_full_denoise=True,
            noise_mask=noise_mask,
            sigmas=sigmas,
            callback=None,
            disable_pbar=True,
            seed=12345,
        )

    assert output.dtype == torch.float32
    assert output.device.type == "cpu"
    assert len({actor.sample_mmh3director_request.requests[0]["operation_id"] for actor in actors}) == 1
    for actor in actors:
        request = actor.sample_mmh3director_request.requests[0]
        assert request["noise"] is noise
        assert request["latent_image"] is latent
        assert request["positive"] is positive
        assert request["negative"] is negative
        assert request["noise_mask"] is noise_mask
        assert request["sigmas"] is sigmas
        assert request["steps"] == 8
        assert request["cfg"] == 4.5
        assert request["sampler_name"] == "euler"
        assert request["scheduler"] == "simple"
        assert request["denoise"] == 0.65
        assert request["disable_noise"] is True
        assert request["start_step"] == 2
        assert request["last_step"] == 7
        assert request["force_full_denoise"] is True
        assert request["disable_pbar"] is True
        assert request["seed"] == 12345
        assert "callback" not in request


def test_proxy_preview_creates_one_queue_and_drains_it_while_workers_run():
    proxy, actors = _proxy()
    callback = mock.Mock()
    preview_queue = mock.Mock()

    with (
        mock.patch.object(bridge, "ensure_mmh3director_compatibility"),
        mock.patch.object(bridge, "_release_main_process_models_before_remote_sampling"),
        mock.patch.object(bridge.preview_transport, "create_preview_queue", return_value=preview_queue) as create_queue,
        mock.patch.object(
            bridge.preview_transport,
            "wait_for_worker_results",
            side_effect=lambda futures, queue, cb: (bridge.preview_transport.drain_preview_queue(queue, cb), futures)[1],
        ) as wait_results,
        mock.patch.object(bridge.preview_transport, "close_preview_queue") as close_queue,
        mock.patch.object(bridge.preview_transport, "drain_preview_queue", return_value=0),
        mock.patch.object(bridge.ray, "get", side_effect=lambda value: value),
        mock.patch.object(bridge.comfy.model_management, "intermediate_device", return_value=torch.device("cpu")),
        mock.patch.object(bridge.comfy.model_management, "intermediate_dtype", return_value=torch.float32),
    ):
        bridge.dispatch_mmh3director_sample(
            proxy,
            torch.zeros(1, 24, 2, 2, 2),
            4,
            1.0,
            "euler",
            "simple",
            [],
            [],
            torch.zeros(1, 24, 2, 2, 2),
            callback=callback,
            preview_every=1,
        )

    create_queue.assert_called_once_with()
    wait_results.assert_called_once()
    close_queue.assert_called_once_with(preview_queue)
    for actor in actors:
        request = actor.sample_mmh3director_request.requests[0]
        assert request["preview_queue"] is preview_queue
        assert request["preview_every"] == 1


def test_proxy_without_preview_does_not_create_preview_transport():
    proxy, _actors = _proxy()
    with (
        mock.patch.object(bridge, "ensure_mmh3director_compatibility"),
        mock.patch.object(bridge, "_release_main_process_models_before_remote_sampling"),
        mock.patch.object(bridge.preview_transport, "create_preview_queue") as create_queue,
        mock.patch.object(bridge.ray, "get", side_effect=lambda futures: futures),
        mock.patch.object(bridge.comfy.model_management, "intermediate_device", return_value=torch.device("cpu")),
        mock.patch.object(bridge.comfy.model_management, "intermediate_dtype", return_value=torch.float32),
    ):
        bridge.dispatch_mmh3director_sample(
            proxy,
            torch.zeros(1),
            1,
            1.0,
            "euler",
            "simple",
            [],
            [],
            torch.zeros(1),
            callback=None,
        )

    create_queue.assert_not_called()


def test_new_director_stage_routes_proxy_before_basic_guider_and_preserves_latent_metadata():
    proxy, actors = _proxy()
    latent_samples = torch.zeros(1, 32, 2, 4, 4)
    latent = {
        "samples": latent_samples,
        "noise_mask": torch.ones(1, 1, 2, 4, 4),
        "batch_index": [7],
        "director_marker": "preserved",
        "_director_continue_prefix_steps": 3,
        "_director_continue_seam_min": 0.1,
    }
    sigmas = torch.tensor([[1.0, 0.4, 0.0]])
    prepared_noise = torch.full_like(latent_samples, 0.25)
    phases = []

    with (
        mock.patch.object(bridge.comfy.sample, "prepare_noise", return_value=prepared_noise) as prepare_noise,
        mock.patch.object(bridge.comfy.sample, "fix_empty_latent_channels", return_value=latent_samples),
        mock.patch.object(bridge, "ensure_mmh3director_compatibility"),
        mock.patch.object(bridge, "_release_main_process_models_before_remote_sampling"),
        mock.patch.object(bridge.ray, "get", side_effect=lambda futures: futures),
        mock.patch.object(bridge.comfy.model_management, "intermediate_device", return_value=torch.device("cpu")),
        mock.patch.object(bridge.comfy.model_management, "intermediate_dtype", return_value=torch.float32),
    ):
        output = bridge.dispatch_mmh3director_stage(
            model=proxy,
            positive=[["positive", {}]],
            negative=[],
            latent=latent,
            seed=12345,
            cfg=1.0,
            steps=8,
            sampler_name="res_multistep",
            scheduler="simple",
            shift_video=11.0,
            shift_audio=2.5,
            on_phase=lambda name, value: phases.append((name, value)),
            on_step_preview=None,
            preview_every=1,
            denoise=0.8,
            phase_name="sample",
            sigmas=sigmas,
            apply_shift=True,
            after_shift=_supported_continue_remask,
            enable_tiling=True,
            tile_count=3,
            tile_overlap=96,
            shift_cache=object(),
        )

    prepare_noise.assert_called_once_with(latent_samples, 12345, [7])
    assert output["director_marker"] == "preserved"
    assert output["samples"].dtype == torch.float32
    assert output is not latent
    assert phases == [("sample", 0), ("sample", 1)]
    for actor in actors:
        request = actor.sample_mmh3director_request.requests[0]
        assert request["noise"] is prepared_noise
        assert torch.equal(request["sigmas"], torch.tensor([1.0, 0.4, 0.0]))
        assert request["apply_shift"] is True
        assert request["steps"] == 8
        assert request["sampler_name"] == "res_multistep"
        assert request["scheduler"] == "simple"
        assert request["after_shift_capability"] == "continue_prefix_remask_v1"
        assert request["director_latent_metadata"] == {
            "_director_continue_prefix_steps": 3,
            "_director_continue_seam_min": 0.1,
        }
        assert request["enable_tiling"] is True
        assert request["tile_count"] == 3
        assert request["tile_overlap"] == 96
        assert request["model_options"]["transformer_options"] == {
            "minimax_h3_sigma_shift_video": 11.0,
            "minimax_h3_sigma_shift_audio": 2.5,
        }


def test_selflift_stage_preserves_fixed_noise_and_returns_final_full_state():
    proxy, actors = _proxy()
    latent_samples = torch.zeros(1, 32, 2, 4, 4)
    fixed_noise = torch.full_like(latent_samples, 0.75)
    states = []

    with (
        mock.patch.object(bridge.comfy.sample, "prepare_noise") as prepare_noise,
        mock.patch.object(bridge.comfy.sample, "fix_empty_latent_channels", return_value=latent_samples),
        mock.patch.object(bridge, "ensure_mmh3director_compatibility"),
        mock.patch.object(bridge, "_release_main_process_models_before_remote_sampling"),
        mock.patch.object(bridge.ray, "get", side_effect=lambda futures: futures),
        mock.patch.object(bridge.comfy.model_management, "intermediate_device", return_value=torch.device("cpu")),
        mock.patch.object(bridge.comfy.model_management, "intermediate_dtype", return_value=torch.float32),
    ):
        bridge.dispatch_mmh3director_stage(
            model=proxy,
            positive=[["positive", {}]],
            negative=[],
            latent={"samples": latent_samples},
            seed=123,
            cfg=1.0,
            steps=3,
            sampler_name="euler",
            scheduler="simple",
            apply_shift=False,
            on_step_state=lambda step, total, x0, x: states.append((step, total, x0, x)),
            noise_override=fixed_noise,
        )

    prepare_noise.assert_not_called()
    assert len(states) == 1
    step, total, x0, x = states[0]
    assert (step, total) == (2, 3)
    assert torch.equal(x0, latent_samples)
    assert torch.equal(x, latent_samples + 1)
    for actor in actors:
        request = actor.sample_mmh3director_request.requests[0]
        assert request["noise"] is fixed_noise
        assert request["capture_step_state"] is True


def test_selflift_zero_noise_uses_exact_latent_structure():
    proxy, actors = _proxy()
    video = torch.ones(1, 24, 2, 4, 4)
    audio = torch.ones(1, 32, 2, 10)
    nested = comfy.nested_tensor.NestedTensor((video, audio))

    with (
        mock.patch.object(bridge.comfy.sample, "prepare_noise") as prepare_noise,
        mock.patch.object(bridge.comfy.sample, "fix_empty_latent_channels", return_value=nested),
        mock.patch.object(bridge, "ensure_mmh3director_compatibility"),
        mock.patch.object(bridge, "_release_main_process_models_before_remote_sampling"),
        mock.patch.object(bridge.ray, "get", side_effect=lambda futures: futures),
        mock.patch.object(bridge.comfy.model_management, "intermediate_device", return_value=torch.device("cpu")),
        mock.patch.object(bridge.comfy.model_management, "intermediate_dtype", return_value=torch.float32),
    ):
        bridge.dispatch_mmh3director_stage(
            model=proxy,
            positive=[["positive", {}]],
            negative=[],
            latent={"samples": nested},
            seed=123,
            cfg=1.0,
            steps=1,
            sampler_name="euler",
            scheduler="simple",
            apply_shift=False,
            zero_noise=True,
        )

    prepare_noise.assert_not_called()
    for actor in actors:
        parts = actor.sample_mmh3director_request.requests[0]["noise"].unbind()
        assert len(parts) == 2
        assert all(torch.count_nonzero(part).item() == 0 for part in parts)


def test_new_director_stage_accepts_real_minimax_av_nested_latent():
    proxy, actors = _proxy()
    video = torch.zeros(1, 24, 2, 4, 4)
    audio = torch.zeros(1, 32, 2, 10)
    nested = comfy.nested_tensor.NestedTensor((video, audio))
    latent = {"samples": nested, "director_marker": "preserved"}
    prepared_noise = comfy.nested_tensor.NestedTensor((torch.ones_like(video), torch.ones_like(audio)))

    with (
        mock.patch.object(bridge.comfy.sample, "prepare_noise", return_value=prepared_noise),
        mock.patch.object(bridge.comfy.sample, "fix_empty_latent_channels", return_value=nested),
        mock.patch.object(bridge, "ensure_mmh3director_compatibility"),
        mock.patch.object(bridge, "_release_main_process_models_before_remote_sampling"),
        mock.patch.object(bridge.ray, "get", side_effect=lambda futures: futures),
        mock.patch.object(bridge.comfy.model_management, "intermediate_device", return_value=torch.device("cpu")),
        mock.patch.object(bridge.comfy.model_management, "intermediate_dtype", return_value=torch.float32),
    ):
        output = bridge.dispatch_mmh3director_stage(
            model=proxy,
            positive=[["positive", {}]],
            negative=[],
            latent=latent,
            seed=12345,
            cfg=1.0,
            steps=1,
            sampler_name="res_multistep",
            scheduler="simple",
            apply_shift=False,
        )

    assert output["director_marker"] == "preserved"
    assert getattr(output["samples"], "is_nested", False) is True
    assert len(output["samples"].unbind()) == 2
    for actor in actors:
        assert actor.sample_mmh3director_request.requests[0]["latent_image"] is nested


def test_new_director_stage_passes_normal_model_to_original_unchanged():
    original = mock.Mock(return_value={"samples": "local"})
    normal_model = object()
    latent = {"samples": torch.zeros(1)}
    kwargs = {
        "model": normal_model,
        "positive": ["positive"],
        "negative": ["negative"],
        "latent": latent,
        "seed": 9,
        "cfg": 1.0,
        "steps": 4,
        "sampler_name": "euler",
        "scheduler": "simple",
        "sigmas": torch.tensor([1.0, 0.0]),
        "apply_shift": False,
    }

    wrapper = bridge._make_mmh3director_stage_wrapper(original)

    assert wrapper(**kwargs) == {"samples": "local"}
    original.assert_called_once_with(**kwargs)


def test_new_director_stage_installer_patches_only_exact_imported_aliases(isolated_sample_bridge):
    original = _supported_sample_single_stage
    core = ModuleType("ComfyUI_MiniMaxH3_Director.director.core_sampling")
    core.__file__ = "C:/nodes/ComfyUI_MiniMaxH3_Director/director/core_sampling.py"
    core.sample_single_stage = original
    executor = ModuleType("ComfyUI_MiniMaxH3_Director.director.executor_core")
    executor.__file__ = "C:/nodes/ComfyUI_MiniMaxH3_Director/director/executor_core.py"
    executor.sample_single_stage = original
    unrelated = ModuleType("unrelated.executor_core")
    unrelated.__file__ = "C:/nodes/unrelated/executor_core.py"
    unrelated.sample_single_stage = original

    with mock.patch.dict(
        sys.modules,
        {
            core.__name__: core,
            executor.__name__: executor,
            unrelated.__name__: unrelated,
        },
        clear=False,
    ):
        bridge.install_mmh3director_sample_bridge(require_director=True)

        assert core.sample_single_stage is executor.sample_single_stage
        assert core.sample_single_stage is not original
        assert unrelated.sample_single_stage is original

        bridge.uninstall_mmh3director_sample_bridge_for_test()
        assert core.sample_single_stage is original
        assert executor.sample_single_stage is original


@pytest.mark.parametrize(
    "model_family, workers, message",
    [
        ("ltx", [_Actor(0), _Actor(1)], "requires model_family=minimax_h3"),
        ("minimax_h3", [], "requires live Ray workers"),
    ],
)
def test_proxy_dispatch_fails_closed_for_wrong_family_or_dead_actors(model_family, workers, message):
    proxy = RayMMH3DirectorModelProxy(
        ray_actors={"workers": workers, "world_size": len(workers)},
        metadata=_metadata(model_family),
    )

    with pytest.raises(RuntimeError, match=message):
        bridge.dispatch_mmh3director_sample(proxy, torch.zeros(1), 1, 1.0, "euler", "simple", [], [], torch.zeros(1))


def test_actor_submission_failure_is_reported_without_local_fallback():
    class BrokenRemote:
        def remote(self, _request):
            raise RuntimeError("actor is dead")

    class BrokenActor:
        sample_mmh3director_request = BrokenRemote()

    proxy = RayMMH3DirectorModelProxy(
        ray_actors={"workers": [BrokenActor()], "world_size": 1},
        metadata=_metadata(),
    )

    with (
        mock.patch.object(bridge, "ensure_mmh3director_compatibility"),
        pytest.raises(
            RuntimeError,
            match="failed before rank validation: RuntimeError: actor is dead",
        ) as error,
    ):
        bridge.dispatch_mmh3director_sample(proxy, torch.zeros(1), 1, 1.0, "euler", "simple", [], [], torch.zeros(1))

    assert isinstance(error.value.__cause__, RuntimeError)
    assert "actor is dead" in str(error.value.__cause__)


def test_rank_tensor_mismatch_is_rejected():
    proxy, _actors = _proxy(samples_by_rank=(torch.zeros(2), torch.ones(2)))

    with (
        mock.patch.object(bridge, "ensure_mmh3director_compatibility"),
        mock.patch.object(bridge.ray, "get", side_effect=lambda futures: futures),
        pytest.raises(ValueError, match="rank 1 samples differ from rank 0"),
    ):
        bridge.dispatch_mmh3director_sample(proxy, torch.zeros(2), 1, 1.0, "euler", "simple", [], [], torch.zeros(2))


def test_non_finite_rank_tensor_is_rejected():
    proxy, _actors = _proxy(samples_by_rank=(torch.tensor([float("nan")]), torch.tensor([float("nan")])))

    with (
        mock.patch.object(bridge, "ensure_mmh3director_compatibility"),
        mock.patch.object(bridge.ray, "get", side_effect=lambda futures: futures),
        pytest.raises(ValueError, match="rank 0 returned NaN/Inf"),
    ):
        bridge.dispatch_mmh3director_sample(proxy, torch.zeros(1), 1, 1.0, "euler", "simple", [], [], torch.zeros(1))


def test_missing_or_misranked_worker_result_is_rejected():
    proxy, _actors = _proxy()

    with (
        mock.patch.object(bridge, "ensure_mmh3director_compatibility"),
        mock.patch.object(bridge.ray, "get", side_effect=lambda futures: [futures[0], None]),
        pytest.raises(RuntimeError, match="missing result for rank 1"),
    ):
        bridge.dispatch_mmh3director_sample(proxy, torch.zeros(1), 1, 1.0, "euler", "simple", [], [], torch.zeros(1))

    with (
        mock.patch.object(bridge, "ensure_mmh3director_compatibility"),
        mock.patch.object(
            bridge.ray,
            "get",
            side_effect=lambda futures: [futures[0], dict(futures[1], rank=9)],
        ),
        pytest.raises(ValueError, match="expected rank 1.*rank 9"),
    ):
        bridge.dispatch_mmh3director_sample(proxy, torch.zeros(1), 1, 1.0, "euler", "simple", [], [], torch.zeros(1))
