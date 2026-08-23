from __future__ import annotations

from unittest import mock

import pytest
import torch

import comfy.sample
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
    )


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
        samples = request["latent_image"].clone() if self.samples is None else self.samples.clone()
        return {
            "rank": self.rank,
            "operation_id": request["operation_id"],
            "samples": samples,
        }


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
    bridge.validate_mmh3director_sample_single_stage(_supported_sample_single_stage)


def test_missing_director_parameter_reports_version_commit_and_field():
    def missing_scheduler(*, model, positive, negative, latent, seed, cfg, steps, sampler_name, **kwargs):
        del model, positive, negative, latent, seed, cfg, steps, sampler_name, kwargs

    with pytest.raises(RuntimeError, match=r"scheduler.*1\.0\.0.*d655912"):
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
    callback = mock.Mock()

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
            callback=callback,
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
        pytest.raises(RuntimeError, match="failed before rank validation") as error,
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
