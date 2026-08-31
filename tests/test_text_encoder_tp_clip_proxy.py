from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from types import SimpleNamespace

import pytest
import torch

from raylight.text_encoder_parallel.linear import LinearExecutionStats

from raylight.text_encoder_parallel.clip_proxy import (
    MiniMaxH3TP2ClipProxy,
    PreparedMiniMaxConditioning,
    TP2Qwen3VLBackbone,
    prepare_minimax_token_sections,
)


def _embedding_lookup(token_ids: torch.Tensor) -> torch.Tensor:
    values = token_ids.to(torch.float32).unsqueeze(-1)
    return torch.cat((values, values + 1, values + 2, values + 3), dim=-1)


def _vision_encode(entry: dict) -> tuple[torch.Tensor, dict]:
    size = int(entry["data"]["tokens"])
    value = 20.0 if entry.get("minimax_video_block") else 10.0
    merged = torch.full((size, 4), value)
    grid = torch.tensor([[1, 2, max(2, size * 2)]], dtype=torch.long)
    deepstack = [torch.full((size, 4), value + layer) for layer in range(3)]
    return merged, {"grid": grid, "deepstack": deepstack}


def _vision(tokens: int, *, video: bool = False) -> dict:
    return {
        "type": "image",
        "data": {"tokens": tokens},
        "original_type": "image",
        "minimax_video_block": video,
    }


def test_prepare_pure_text_preserves_shape_and_text_tags() -> None:
    prepared = prepare_minimax_token_sections(
        [[101, 102, 103]],
        embedding_lookup=_embedding_lookup,
        vision_encode=_vision_encode,
    )

    assert prepared.embeddings.shape == (1, 3, 4)
    assert prepared.attention_mask.tolist() == [[1, 1, 1]]
    assert prepared.token_tags.tolist() == [1, 1, 1]
    assert prepared.position_ids.tolist() == [[0, 1, 2]]
    assert prepared.visual_position_mask is None
    assert prepared.deepstack_embeddings is None


@pytest.mark.parametrize(
    ("entries", "expected_length", "expected_tags", "video_flags"),
    [
        (
            [151652, _vision(2), 151653, 200],
            5,
            [0, 0, 0, 0, 1],
            [False],
        ),
        (
            [151652, _vision(2), 151653, 201, 151652, _vision(3), 151653, 202],
            11,
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [False, False],
        ),
        (
            [151652, _vision(4, video=True), 151653, 203],
            7,
            [0, 0, 0, 0, 0, 0, 1],
            [True],
        ),
    ],
)
def test_prepare_i2v_ref2va_and_video_reference_contracts(
    entries, expected_length, expected_tags, video_flags
) -> None:
    seen_video_flags: list[bool] = []

    def vision_encode(entry):
        seen_video_flags.append(bool(entry.get("minimax_video_block")))
        return _vision_encode(entry)

    prepared = prepare_minimax_token_sections(
        [entries],
        embedding_lookup=_embedding_lookup,
        vision_encode=vision_encode,
    )

    assert prepared.embeddings.shape == (1, expected_length, 4)
    assert prepared.token_tags.tolist() == expected_tags
    assert prepared.visual_position_mask.shape == (1, expected_length)
    assert int(prepared.visual_position_mask.sum()) == sum(
        int(entry["data"]["tokens"]) for entry in entries if isinstance(entry, dict)
    )
    assert len(prepared.deepstack_embeddings) == 3
    assert seen_video_flags == video_flags
    assert torch.isfinite(prepared.embeddings).all()


def test_ref2va_mrope_positions_match_frozen_comfyui_implementation() -> None:
    from comfy.text_encoders.qwen_vl import qwen2vl_mrope_position_ids

    entries = [151652, _vision(2), 151653, 201, 151652, _vision(3), 151653, 202]
    prepared = prepare_minimax_token_sections(
        [entries], embedding_lookup=_embedding_lookup, vision_encode=_vision_encode
    )
    expected = qwen2vl_mrope_position_ids(
        prepared.embeds_info, prepared.embeddings.shape[1], torch.device("cpu")
    )

    assert torch.equal(prepared.position_ids, expected.to(torch.long))


class _Tokenizer:
    def __init__(self) -> None:
        self.calls = []

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        self.calls.append((text, return_word_ids, kwargs))
        return {"qwen3vl_32b": [[(101, 1.0), (102, 1.0)]]}


@dataclass
class _ResourceState:
    preprocess_active: int = 0
    backbone_active: int = 0
    preprocess_closed: int = 0
    backbone_closed: int = 0


class _Preprocessor:
    def __init__(self, state: _ResourceState) -> None:
        self.state = state
        state.preprocess_active += 1

    def prepare(self, sections):
        return prepare_minimax_token_sections(
            sections,
            embedding_lookup=_embedding_lookup,
            vision_encode=_vision_encode,
        )

    def close(self):
        self.state.preprocess_active -= 1
        self.state.preprocess_closed += 1


class _Backbone:
    def __init__(self, state: _ResourceState, *, fail=False) -> None:
        self.state = state
        self.fail = fail
        state.backbone_active += 1

    def encode(self, prepared: PreparedMiniMaxConditioning):
        if self.fail:
            raise RuntimeError("injected TP2 failure")
        return prepared.embeddings + 0.5

    def profile_snapshot(self):
        return {"layers": 50, "linear_pair_operations": 350}

    def close(self):
        self.state.backbone_active -= 1
        self.state.backbone_closed += 1


def _proxy(state: _ResourceState, *, fail=False) -> MiniMaxH3TP2ClipProxy:
    return MiniMaxH3TP2ClipProxy(
        tokenizer=_Tokenizer(),
        preprocessor_factory=lambda: _Preprocessor(state),
        backbone_factory=lambda: _Backbone(state, fail=fail),
    )


def test_clip_proxy_preserves_comfy_clip_contract_and_metadata() -> None:
    state = _ResourceState()
    proxy = _proxy(state)
    tokens = proxy.tokenize("a cinematic robot")

    encoded = proxy.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
    scheduled = proxy.encode_from_tokens_scheduled(tokens, add_dict={"source": "test"})

    assert encoded["cond"].shape == (1, 2, 4)
    assert encoded["pooled_output"] is None
    assert encoded["minimax_token_tags"].tolist() == [1, 1]
    assert scheduled[0][0].shape == (1, 2, 4)
    assert scheduled[0][1]["source"] == "test"
    assert scheduled[0][1]["minimax_token_tags"].tolist() == [1, 1]
    assert state == _ResourceState(preprocess_closed=2, backbone_closed=2)


def test_clip_proxy_clone_is_compatible_and_rejects_unsplit_lora() -> None:
    state = _ResourceState()
    proxy = _proxy(state)
    clone = proxy.clone()

    assert clone is not proxy
    assert clone.tokenizer is proxy.tokenizer
    assert clone.tokenize("x") == proxy.tokenize("x")
    assert proxy.add_patches({}) == []
    with pytest.raises(ValueError, match="Text Encoder LoRA"):
        proxy.add_patches({"model.layers.0.self_attn.q_proj.weight": object()})


def test_encode_exception_releases_preprocessor_and_both_gpu_backbone() -> None:
    state = _ResourceState()
    proxy = _proxy(state, fail=True)
    tokens = {"qwen3vl_32b": [[(101, 1.0), (102, 1.0)]]}

    with pytest.raises(RuntimeError, match="injected TP2 failure"):
        proxy.encode_from_tokens(tokens)

    assert state.preprocess_active == 0
    assert state.backbone_active == 0
    assert state.preprocess_closed == 1
    assert state.backbone_closed == 1


def test_proxy_wraps_preprocessor_and_backbone_in_execution_context() -> None:
    state = _ResourceState()
    events = []

    @contextmanager
    def execution_context():
        events.append("lease-enter")
        try:
            yield
        finally:
            events.append("lease-exit")

    proxy = _proxy(state)
    proxy._execution_context_factory = execution_context
    proxy.encode_from_tokens(proxy.tokenize("prompt"), return_dict=True)

    assert events == ["lease-enter", "lease-exit"]
    assert state.preprocess_closed == 1
    assert state.backbone_closed == 1


def test_proxy_records_detailed_benchmark_stages(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from raylight.text_encoder_parallel import baseline

    event_path = tmp_path / "events.jsonl"
    monkeypatch.setenv(baseline.BASELINE_ENV, "1")
    monkeypatch.setenv(baseline.BASELINE_EVENT_PATH_ENV, str(event_path))
    state = _ResourceState()

    _proxy(state).encode_from_tokens(_proxy(state).tokenize("prompt"), return_dict=True)

    events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    assert [event["stage"] for event in events if event["event"] == "text_encoder_stage"] == [
        "preprocessor_load",
        "preprocess_and_vision",
        "preprocessor_cleanup",
        "backbone_load",
        "language_backbone",
        "backbone_cleanup",
        "conditioning_finalize",
    ]
    profiles = [event for event in events if event["event"] == "text_encoder_tp2_profile"]
    assert profiles == [
        {
            "event": "text_encoder_tp2_profile",
            "profile": {"layers": 50, "linear_pair_operations": 350},
        }
    ]


def test_backbone_profile_aggregates_linear_overlap_and_collectives() -> None:
    stats = [
        LinearExecutionStats(2.0, 2.0, 2.2, 1.8, 0.9),
        LinearExecutionStats(2.0, 2.0, 3.0, 1.0, 0.5),
    ]

    class Collective:
        def profile_snapshot(self):
            return {"calls": 4, "total_bytes_sent": 1024, "total_elapsed_ms": 1.5}

    class Block:
        def __init__(self, offset):
            names = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
            for index, name in enumerate(names):
                setattr(self, name, SimpleNamespace(last_execution=stats[(index + offset) % 2]))
            self.execution_profiles = [("attention", stats[offset % 2])]
            self.collective = collective

    collective = Collective()
    backbone = TP2Qwen3VLBackbone(
        [Block(0), Block(1)],
        devices=(torch.device("cpu"), torch.device("cpu")),
    )

    profile = backbone.profile_snapshot()

    assert profile["layers"] == 2
    assert profile["linear_pair_operations"] == 14
    assert profile["paired_operations"] == 16
    assert profile["linear_overlap_ge_80_percent"] == pytest.approx(50.0)
    assert profile["layers_with_linear_overlap_percent"] == 100.0
    assert profile["collective"]["calls"] == 4


class _TPBlock:
    def __init__(self) -> None:
        self.operation_ids = []

    def forward(self, hidden0, hidden1, position_ids, *, attention_mask, operation_id):
        self.operation_ids.append(operation_id)
        assert position_ids.shape == (1, 3)
        assert attention_mask.shape == (3, 3)
        return hidden0 + 1, hidden1 + 1


def test_backbone_injects_deepstack_after_matching_decoder_layers() -> None:
    blocks = [_TPBlock(), _TPBlock(), _TPBlock()]
    prepared = PreparedMiniMaxConditioning(
        embeddings=torch.zeros((1, 3, 4)),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
        num_tokens=(3,),
        embeds_info=(),
        token_tags=torch.tensor([1, 0, 1]),
        position_ids=torch.arange(3).unsqueeze(0),
        visual_position_mask=torch.tensor([[False, True, False]]),
        deepstack_embeddings=(torch.full((1, 4), 10.0), torch.full((1, 4), 20.0)),
    )
    closed = []
    backbone = TP2Qwen3VLBackbone(
        blocks,
        devices=(torch.device("cpu"), torch.device("cpu")),
        close_callbacks=(lambda: closed.append("executor"), lambda: closed.append("collective")),
    )

    output = backbone.encode(prepared)
    backbone.close()

    assert output[:, [0, 2]].eq(3).all()
    assert output[:, 1].eq(33).all()
    assert [block.operation_ids for block in blocks] == [[1], [3], [5]]
    assert closed == ["executor", "collective"]
