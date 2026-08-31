"""MiniMax H3 CLIP-compatible proxy for the native-Windows TP2 encoder.

The public proxy deliberately owns the tokenizer but not a normal ComfyUI
``CLIP``/``ModelPatcher``.  Loading the latter would materialize the complete
Qwen3-VL-32B language model on GPU0 before TP2 had a chance to shard it.
Instead, a short-lived preprocessor produces token/vision embeddings and a
short-lived TP2 backbone consumes them.  Both resources are closed in
``finally`` blocks so a failed encode cannot strand either V100's weights.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
import gc
import math
import numbers
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import torch

from .collectives import CudaPeerGroup2
from .baseline import timed_stage
from .linear import ColumnParallelLinear, RowParallelLinear, TorchDualCudaLinearExecutor
from .nvfp4_sharding import shard_nvfp4_linear
from .qwen3vl_tp import MINIMAX_H3_QWEN3VL_CONFIG, TP2Qwen3VLBlock
from .topology import validate_windows_v100_topology


MINIMAX_PAD_TOKEN = 151643
MINIMAX_CLIP_NAME = "qwen3vl_32b"


@dataclass(frozen=True)
class PreparedMiniMaxConditioning:
    """Inputs needed by the 50-layer TP2 language backbone."""

    embeddings: torch.Tensor
    attention_mask: torch.Tensor
    num_tokens: tuple[int, ...]
    embeds_info: tuple[dict[str, Any], ...]
    token_tags: torch.Tensor
    position_ids: torch.Tensor
    visual_position_mask: torch.Tensor | None
    deepstack_embeddings: tuple[torch.Tensor, ...] | None

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 3:
            raise ValueError("MiniMax embeddings must have shape [batch, sequence, hidden]")
        if tuple(self.attention_mask.shape) != tuple(self.embeddings.shape[:2]):
            raise ValueError("attention mask must match embedding batch and sequence dimensions")
        if len(self.num_tokens) != self.embeddings.shape[0]:
            raise ValueError("num_tokens must contain one entry per batch row")
        if tuple(self.token_tags.shape) != (self.embeddings.shape[1],):
            raise ValueError("minimax_token_tags must contain one tag per sequence position")
        if self.visual_position_mask is not None and tuple(self.visual_position_mask.shape) != (
            1,
            self.embeddings.shape[1],
        ):
            raise ValueError("visual position mask must have shape [1, sequence]")


def _token_tags(seq_len: int, embeds_info: Sequence[dict[str, Any]]) -> torch.Tensor:
    tags = torch.ones(seq_len, dtype=torch.long)
    for entry in embeds_info:
        if entry.get("type") != "image":
            continue
        start = int(entry["index"])
        end = start + int(entry["size"])
        tags[max(0, start - 1) : min(seq_len, end + 1)] = 0
    return tags


def _mrope_position_ids(
    embeds_info: Sequence[dict[str, Any]], seq_len: int, device: torch.device
) -> torch.Tensor:
    images = [entry for entry in embeds_info if entry.get("type") == "image"]
    if not images:
        return torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)

    position_ids = torch.zeros((3, seq_len), device=device, dtype=torch.long)
    offset = 0
    cursor = 0
    for entry in images:
        start = int(entry["index"])
        end = start + int(entry["size"])
        if start > cursor:
            text_start = cursor + offset
            position_ids[:, cursor:start] = torch.arange(
                text_start, text_start + start - cursor, device=device
            )
        extra = entry.get("extra")
        grid = extra["grid"] if isinstance(extra, dict) else extra
        if grid is None:
            raise ValueError("image embedding metadata must contain a Qwen3-VL grid")
        grid = grid.to(device="cpu")
        len_max = int(grid.max().item()) // 2
        position_ids[0, start:end] = start + offset
        height = max(1, int(grid[0][1].item()) // 2)
        width = max(1, int(grid[0][2].item()) // 2)
        position_ids[1, start:end] = (
            torch.arange(start + offset, start + height + offset, device=device)
            .unsqueeze(1)
            .repeat(1, math.ceil((end - start) / height))
            .flatten()[: end - start]
        )
        position_ids[2, start:end] = (
            torch.arange(start + offset, start + width + offset, device=device)
            .unsqueeze(0)
            .repeat(math.ceil((end - start) / width), 1)
            .flatten()[: end - start]
        )
        offset += len_max - (end - start)
        cursor = end
    if cursor < seq_len:
        text_start = cursor + offset
        position_ids[:, cursor:] = torch.arange(
            text_start, text_start + seq_len - cursor, device=device
        )
    return position_ids


def _integer_tokens_and_mask(entries: Sequence[Any], pad_token: int) -> tuple[list[int], list[int], list[tuple[int, Any]]]:
    token_ids: list[int] = []
    attention_mask: list[int] = []
    other_embeds: list[tuple[int, Any]] = []
    eos = False
    left_pad = False
    for index, entry in enumerate(entries):
        if isinstance(entry, numbers.Integral):
            token = int(entry)
            if index == 0 and token == pad_token:
                left_pad = True
            if eos or (left_pad and token == pad_token):
                attention_mask.append(0)
            else:
                attention_mask.append(1)
                left_pad = False
            token_ids.append(token)
            if not eos and token == pad_token and not left_pad:
                attention_mask[-1] = 0
                eos = True
        else:
            other_embeds.append((index, entry))
    return token_ids, attention_mask, other_embeds


def prepare_minimax_token_sections(
    sections: Sequence[Sequence[Any]],
    *,
    embedding_lookup: Callable[[torch.Tensor], torch.Tensor],
    vision_encode: Callable[[dict[str, Any]], tuple[torch.Tensor, dict[str, Any]]],
    pad_token: int = MINIMAX_PAD_TOKEN,
) -> PreparedMiniMaxConditioning:
    """Expand MiniMax tokenizer entries without instantiating the full CLIP model."""

    if not sections:
        raise ValueError("MiniMax token sections cannot be empty")
    embeddings_out: list[torch.Tensor] = []
    masks_out: list[list[int]] = []
    num_tokens: list[int] = []
    reference_info: list[dict[str, Any]] | None = None

    for section in sections:
        token_ids, attention_mask, other_embeds = _integer_tokens_and_mask(section, pad_token)
        ids = torch.tensor([token_ids], dtype=torch.long)
        token_embeddings = embedding_lookup(ids)
        if token_embeddings.ndim != 3 or token_embeddings.shape[:2] != ids.shape:
            raise ValueError("embedding_lookup must return [1, token_count, hidden]")

        insertion_offset = 0
        embeds_info: list[dict[str, Any]] = []
        for source_index, raw_entry in other_embeds:
            entry = raw_entry
            if torch.is_tensor(entry):
                entry = {"type": "embedding", "data": entry}
            if not isinstance(entry, dict):
                raise ValueError("non-token MiniMax entries must be tensors or dictionaries")
            embed_type = entry.get("type")
            extra = None
            if embed_type == "embedding":
                embedding = entry.get("data")
            else:
                embedding, extra = vision_encode(entry)
            if embedding is None:
                continue
            embedding = embedding.reshape(1, -1, embedding.shape[-1]).to(
                device=token_embeddings.device, dtype=token_embeddings.dtype
            )
            if embedding.shape[-1] != token_embeddings.shape[-1]:
                raise ValueError(
                    f"embedding width mismatch: {embedding.shape[-1]} != {token_embeddings.shape[-1]}"
                )
            insertion_index = source_index + insertion_offset
            token_embeddings = torch.cat(
                (
                    token_embeddings[:, :insertion_index],
                    embedding,
                    token_embeddings[:, insertion_index:],
                ),
                dim=1,
            )
            size = int(embedding.shape[1])
            attention_mask[insertion_index:insertion_index] = [1] * size
            insertion_offset += size - 1
            embeds_info.append(
                {"type": embed_type, "index": insertion_index, "size": size, "extra": extra}
            )

        if reference_info is None:
            reference_info = embeds_info
        elif embeds_info:
            raise ValueError("batched MiniMax vision sections are not supported by the TP2 proxy")
        embeddings_out.append(token_embeddings)
        masks_out.append(attention_mask)
        num_tokens.append(sum(attention_mask))

    lengths = {int(item.shape[1]) for item in embeddings_out}
    if len(lengths) != 1:
        raise ValueError("all MiniMax token sections must expand to the same sequence length")
    embeddings = torch.cat(embeddings_out, dim=0)
    device = embeddings.device
    mask = torch.tensor(masks_out, device=device, dtype=torch.long)
    info = reference_info or []
    seq_len = int(embeddings.shape[1])
    visual_mask = None
    deepstack: tuple[torch.Tensor, ...] | None = None
    image_info = sorted(
        (entry for entry in info if entry.get("type") == "image"), key=lambda entry: entry["index"]
    )
    if image_info:
        visual_mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
        deepstack_lists: list[list[torch.Tensor]] | None = None
        for entry in image_info:
            start = int(entry["index"])
            end = start + int(entry["size"])
            visual_mask[0, start:end] = True
            values = entry["extra"]["deepstack"]
            if deepstack_lists is None:
                deepstack_lists = [[] for _ in values]
            if len(values) != len(deepstack_lists):
                raise ValueError("all visual entries must contain the same DeepStack depth")
            for layer, value in enumerate(values):
                deepstack_lists[layer].append(value.to(device=device, dtype=embeddings.dtype))
        assert deepstack_lists is not None
        deepstack = tuple(torch.cat(parts, dim=0) for parts in deepstack_lists)

    return PreparedMiniMaxConditioning(
        embeddings=embeddings,
        attention_mask=mask,
        num_tokens=tuple(num_tokens),
        embeds_info=tuple(info),
        token_tags=_token_tags(seq_len, info),
        position_ids=_mrope_position_ids(info, seq_len, device),
        visual_position_mask=visual_mask,
        deepstack_embeddings=deepstack,
    )


def _prepared_to_cpu(prepared: PreparedMiniMaxConditioning) -> PreparedMiniMaxConditioning:
    def cpu_extra(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().to("cpu")
        if isinstance(value, dict):
            return {key: cpu_extra(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(cpu_extra(item) for item in value)
        return value

    return PreparedMiniMaxConditioning(
        embeddings=prepared.embeddings.detach().to("cpu"),
        attention_mask=prepared.attention_mask.detach().to("cpu"),
        num_tokens=prepared.num_tokens,
        embeds_info=tuple(cpu_extra(entry) for entry in prepared.embeds_info),
        token_tags=prepared.token_tags.detach().to("cpu"),
        position_ids=prepared.position_ids.detach().to("cpu"),
        visual_position_mask=(
            None
            if prepared.visual_position_mask is None
            else prepared.visual_position_mask.detach().to("cpu")
        ),
        deepstack_embeddings=(
            None
            if prepared.deepstack_embeddings is None
            else tuple(value.detach().to("cpu") for value in prepared.deepstack_embeddings)
        ),
    )


class MiniMaxH3CheckpointPreprocessor:
    """GPU0-only INT8 token embedding and Qwen3-VL vision preprocessing."""

    def __init__(self, checkpoint: str | Path, *, device: int = 0) -> None:
        from safetensors import safe_open

        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        self.device = torch.device("cuda", device)
        self._closed = False
        self._visual = None
        with safe_open(self.checkpoint, framework="pt", device="cpu") as handle:
            qdata = handle.get_tensor("model.embed_tokens.weight")
            scale = handle.get_tensor("model.embed_tokens.weight_scale")
        if tuple(qdata.shape) != (151936, MINIMAX_H3_QWEN3VL_CONFIG.hidden_size):
            raise ValueError(f"unsupported MiniMax H3 embedding shape: {tuple(qdata.shape)}")
        if tuple(scale.shape) != (151936, 1):
            raise ValueError(f"unsupported MiniMax H3 embedding scale shape: {tuple(scale.shape)}")
        self._embedding_qdata = qdata.to(self.device, non_blocking=False)
        self._embedding_scale = scale.to(self.device, dtype=torch.float32, non_blocking=False)
        del qdata, scale

    def _embedding_lookup(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("MiniMax H3 preprocessor is closed")
        ids = token_ids.to(self.device, dtype=torch.long)
        return self._embedding_qdata[ids].to(torch.float32) * self._embedding_scale[ids]

    def _load_visual(self):
        if self._visual is not None:
            return self._visual
        from safetensors import safe_open
        from comfy.text_encoders.qwen3vl import (
            QWEN3VL_VISION,
            QWEN3VL_VISION_COMMON,
            Qwen3VLVisionModel,
        )

        config = {
            **QWEN3VL_VISION_COMMON,
            **QWEN3VL_VISION[MINIMAX_CLIP_NAME],
            "out_hidden_size": MINIMAX_H3_QWEN3VL_CONFIG.hidden_size,
        }
        visual = Qwen3VLVisionModel(
            config,
            device=torch.device("meta"),
            dtype=torch.float32,
            ops=torch.nn,
        )
        with safe_open(self.checkpoint, framework="pt", device="cpu") as handle:
            state = {
                key.removeprefix("visual."): handle.get_tensor(key).to(
                    self.device, dtype=torch.float32, non_blocking=False
                )
                for key in handle.keys()
                if key.startswith("visual.")
            }
        incompatible = visual.load_state_dict(state, strict=True, assign=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"MiniMax H3 vision checkpoint mismatch: missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        visual = visual.to(self.device).eval()
        del state
        self._visual = visual
        return visual

    def _vision_encode(self, entry: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        visual = self._load_visual()
        if entry.get("type") != "image":
            raise ValueError(f"unsupported MiniMax embedding type: {entry.get('type')!r}")
        if entry.get("minimax_video_block"):
            from comfy.text_encoders.minimax import process_video_block

            flattened, grid = process_video_block(entry["data"])
        else:
            from comfy.text_encoders.qwen_vl import process_qwen2vl_images

            flattened, grid = process_qwen2vl_images(
                entry["data"],
                patch_size=16,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
            )
        with torch.inference_mode():
            merged, deepstack = visual(flattened.to(self.device, dtype=torch.float32), grid)
        return merged, {"grid": grid.to(self.device), "deepstack": deepstack}

    def prepare(self, sections: Sequence[Sequence[Any]]) -> PreparedMiniMaxConditioning:
        prepared = prepare_minimax_token_sections(
            sections,
            embedding_lookup=self._embedding_lookup,
            vision_encode=self._vision_encode,
        )
        torch.cuda.synchronize(self.device)
        return _prepared_to_cpu(prepared)

    def close(self) -> None:
        if self._closed:
            return
        self._visual = None
        self._embedding_qdata = None
        self._embedding_scale = None
        self._closed = True
        gc.collect()
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()


class TP2Qwen3VLBackbone:
    """Execute a sequence of TP2 decoder blocks and own their GPU lifetime."""

    def __init__(
        self,
        blocks: Sequence[Any],
        *,
        devices: tuple[torch.device, torch.device] = (torch.device("cuda", 0), torch.device("cuda", 1)),
        close_callbacks: Sequence[Callable[[], None]] = (),
    ) -> None:
        if not blocks:
            raise ValueError("TP2 language backbone requires at least one block")
        if len(devices) != 2:
            raise ValueError("TP2 language backbone requires exactly two devices")
        self.blocks = list(blocks)
        self.devices = devices
        self._close_callbacks = tuple(close_callbacks)
        self._closed = False

    @staticmethod
    def _causal_mask(sequence: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.full(
            (sequence, sequence), torch.finfo(dtype).min / 4, dtype=dtype
        ).triu_(1)

    def encode(self, prepared: PreparedMiniMaxConditioning) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("TP2 language backbone is closed")
        if prepared.embeddings.shape[0] != 1:
            raise ValueError("MiniMax H3 TP2 currently supports one conditioning batch at a time")
        hidden = (
            prepared.embeddings.to(self.devices[0], dtype=torch.float32),
            prepared.embeddings.to(self.devices[1], dtype=torch.float32),
        )
        sequence = int(prepared.embeddings.shape[1])
        mask = self._causal_mask(sequence, torch.float32)
        for layer, block in enumerate(self.blocks):
            hidden = block.forward(
                hidden[0],
                hidden[1],
                prepared.position_ids,
                attention_mask=mask,
                operation_id=layer * 2 + 1,
            )
            if prepared.deepstack_embeddings is not None and layer < len(prepared.deepstack_embeddings):
                if prepared.visual_position_mask is None:
                    raise RuntimeError("DeepStack embeddings require a visual position mask")
                outputs = []
                for rank in (0, 1):
                    value = hidden[rank]
                    visual_mask = prepared.visual_position_mask.to(self.devices[rank])
                    deepstack = prepared.deepstack_embeddings[layer].to(
                        self.devices[rank], dtype=value.dtype
                    )
                    value[visual_mask] = value[visual_mask] + deepstack
                    outputs.append(value)
                hidden = outputs[0], outputs[1]

        for device in self.devices:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        rank0 = hidden[0].detach().to("cpu", dtype=torch.float32)
        rank1 = hidden[1].detach().to("cpu", dtype=torch.float32)
        if not torch.isfinite(rank0).all().item() or not torch.isfinite(rank1).all().item():
            raise RuntimeError("MiniMax H3 TP2 language backbone produced NaN or Inf")
        if not torch.equal(rank0, rank1):
            max_abs = float((rank0 - rank1).abs().max().item())
            raise RuntimeError(f"MiniMax H3 TP2 ranks diverged (max_abs={max_abs})")
        return rank0

    def close(self) -> None:
        if self._closed:
            return
        self.blocks.clear()
        for callback in self._close_callbacks:
            callback()
        self._close_callbacks = ()
        self._closed = True
        gc.collect()
        for device in self.devices:
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()


_LINEAR_LAYOUT = {
    "q_proj": ("self_attn.q_proj", "row", (8192, 5120)),
    "k_proj": ("self_attn.k_proj", "row", (1024, 5120)),
    "v_proj": ("self_attn.v_proj", "row", (1024, 5120)),
    "o_proj": ("self_attn.o_proj", "column", (5120, 8192)),
    "gate_proj": ("mlp.gate_proj", "row", (25600, 5120)),
    "up_proj": ("mlp.up_proj", "row", (25600, 5120)),
    "down_proj": ("mlp.down_proj", "column", (5120, 25600)),
}


def _load_nvfp4_weight(handle, prefix: str, shape: tuple[int, int]):
    from comfy_kitchen.tensor import QuantizedTensor, TensorCoreNVFP4Layout

    params = TensorCoreNVFP4Layout.Params(
        scale=handle.get_tensor(f"{prefix}.weight_scale_2"),
        block_scale=handle.get_tensor(f"{prefix}.weight_scale"),
        orig_dtype=torch.float32,
        orig_shape=shape,
        transposed=False,
    )
    return QuantizedTensor(
        handle.get_tensor(f"{prefix}.weight"), "TensorCoreNVFP4Layout", params
    )


def _gpu_shards(weight, *, axis: str):
    shards = []
    for rank in (0, 1):
        shard = shard_nvfp4_linear(weight, axis=axis, rank=rank)
        shards.append(replace(shard, qweight=shard.qweight.to(f"cuda:{rank}")))
    return tuple(shards)


def load_minimax_h3_tp2_backbone(checkpoint: str | Path) -> TP2Qwen3VLBackbone:
    """Load 50 NVFP4 decoder layers directly into two V100-local shards."""

    import comfy_kitchen as ck
    from safetensors import safe_open

    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    validate_windows_v100_topology(torch_module=torch, devices=(0, 1))
    ck.registry.disable("cuda")
    ck.registry.disable("triton")
    executor = TorchDualCudaLinearExecutor()
    collective = CudaPeerGroup2()
    blocks: list[TP2Qwen3VLBlock] = []
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if "model.layers.49.mlp.down_proj.weight" not in keys or any(
                key.startswith("model.layers.50.") for key in keys
            ):
                raise ValueError("checkpoint must be the 50-layer MiniMax H3 Qwen3-VL-32B model")
            for layer in range(50):
                layer_prefix = f"model.layers.{layer}"
                shards = {}
                for name, (suffix, axis, shape) in _LINEAR_LAYOUT.items():
                    weight = _load_nvfp4_weight(handle, f"{layer_prefix}.{suffix}", shape)
                    shards[name] = _gpu_shards(weight, axis=axis)
                    del weight

                def norm_pair(suffix: str):
                    value = handle.get_tensor(f"{layer_prefix}.{suffix}").to(torch.float32)
                    pair = value.to("cuda:0"), value.to("cuda:1")
                    del value
                    return pair

                def scale_pair(suffix: str):
                    value = handle.get_tensor(f"{layer_prefix}.{suffix}").to(torch.float32)
                    parts = value.chunk(2)
                    pair = parts[0].contiguous().to("cuda:0"), parts[1].contiguous().to("cuda:1")
                    del value, parts
                    return pair

                blocks.append(
                    TP2Qwen3VLBlock(
                        MINIMAX_H3_QWEN3VL_CONFIG,
                        q_proj=ColumnParallelLinear(shards["q_proj"], executor=executor),
                        k_proj=ColumnParallelLinear(shards["k_proj"], executor=executor),
                        v_proj=ColumnParallelLinear(shards["v_proj"], executor=executor),
                        o_proj=RowParallelLinear(
                            shards["o_proj"],
                            pre_quant_scale=scale_pair("self_attn.o_proj.pre_quant_scale"),
                            executor=executor,
                            collective=collective,
                        ),
                        gate_proj=ColumnParallelLinear(shards["gate_proj"], executor=executor),
                        up_proj=ColumnParallelLinear(shards["up_proj"], executor=executor),
                        down_proj=RowParallelLinear(
                            shards["down_proj"],
                            pre_quant_scale=scale_pair("mlp.down_proj.pre_quant_scale"),
                            executor=executor,
                            collective=collective,
                        ),
                        input_layernorm=norm_pair("input_layernorm.weight"),
                        post_attention_layernorm=norm_pair("post_attention_layernorm.weight"),
                        q_norm=norm_pair("self_attn.q_norm.weight"),
                        k_norm=norm_pair("self_attn.k_norm.weight"),
                    )
                )
        return TP2Qwen3VLBackbone(
            blocks,
            close_callbacks=(executor.close, collective.close),
        )
    except BaseException:
        blocks.clear()
        executor.close()
        collective.close()
        gc.collect()
        for rank in (0, 1):
            with torch.cuda.device(rank):
                torch.cuda.empty_cache()
        raise


class MiniMaxH3TP2ClipProxy:
    """The subset of ComfyUI's CLIP contract used by MiniMax H3 workflows."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        preprocessor_factory: Callable[[], Any],
        backbone_factory: Callable[[], Any],
        execution_context_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self._preprocessor_factory = preprocessor_factory
        self._backbone_factory = backbone_factory
        self._execution_context_factory = execution_context_factory or nullcontext
        self.tokenizer_options: dict[str, Any] = {}
        self.layer_idx = None
        self.use_clip_schedule = False
        self.apply_hooks_to_conds = None
        self.patcher = SimpleNamespace(forced_hooks=None, load_device=torch.device("cuda", 0))
        self.cond_stage_model = self

    def clone(self) -> "MiniMaxH3TP2ClipProxy":
        clone = type(self)(
            tokenizer=self.tokenizer,
            preprocessor_factory=self._preprocessor_factory,
            backbone_factory=self._backbone_factory,
            execution_context_factory=self._execution_context_factory,
        )
        clone.tokenizer_options = self.tokenizer_options.copy()
        clone.layer_idx = self.layer_idx
        clone.use_clip_schedule = self.use_clip_schedule
        clone.apply_hooks_to_conds = self.apply_hooks_to_conds
        return clone

    def tokenize(self, text: str, return_word_ids: bool = False, **kwargs: Any) -> dict:
        tokenizer_options = kwargs.get("tokenizer_options", {})
        if self.tokenizer_options:
            kwargs["tokenizer_options"] = {**self.tokenizer_options, **tokenizer_options}
        return self.tokenizer.tokenize_with_weights(text, return_word_ids, **kwargs)

    def set_tokenizer_option(self, option_name: str, value: Any) -> None:
        self.tokenizer_options[option_name] = value

    def clip_layer(self, layer_idx: int) -> None:
        if layer_idx not in (-1, 49, None):
            raise ValueError("MiniMax H3 TP2 exposes only the checkpoint's final truncated layer")
        self.layer_idx = layer_idx

    def reset_clip_options(self) -> None:
        return None

    def set_clip_options(self, options: dict[str, Any]) -> None:
        layer = options.get("layer")
        if layer is not None:
            self.clip_layer(layer)

    def add_patches(self, patches: dict, strength_patch: float = 1.0, strength_model: float = 1.0):
        del strength_patch, strength_model
        if patches:
            raise ValueError(
                "Text Encoder LoRA is not supported until every adapter can be split identically across TP2 ranks"
            )
        return []

    @staticmethod
    def _sections_and_weights(tokens: dict) -> tuple[list[list[Any]], list[list[float]]]:
        if not isinstance(tokens, dict) or MINIMAX_CLIP_NAME not in tokens:
            raise ValueError(f"MiniMax H3 tokens must contain {MINIMAX_CLIP_NAME!r}")
        weighted_sections = tokens[MINIMAX_CLIP_NAME]
        if not weighted_sections:
            raise ValueError("MiniMax H3 token sections cannot be empty")
        sections: list[list[Any]] = []
        weights: list[list[float]] = []
        for weighted in weighted_sections:
            sections.append([item[0] for item in weighted])
            weights.append([float(item[1]) for item in weighted])
        return sections, weights

    def _encode_token_weights(self, tokens: dict) -> tuple[torch.Tensor, None, dict[str, Any]]:
        sections, weights = self._sections_and_weights(tokens)
        if any(weight != 1.0 for section in weights for weight in section):
            raise ValueError("weighted MiniMax prompts are not supported by the TP2 proxy")

        with self._execution_context_factory():
            with timed_stage("preprocessor_load"):
                preprocessor = self._preprocessor_factory()
            try:
                with timed_stage("preprocess_and_vision"):
                    prepared = preprocessor.prepare(sections)
            finally:
                with timed_stage("preprocessor_cleanup"):
                    preprocessor.close()

            with timed_stage("backbone_load"):
                backbone = self._backbone_factory()
            try:
                with timed_stage("language_backbone"):
                    encoded = backbone.encode(prepared)
            finally:
                with timed_stage("backbone_cleanup"):
                    backbone.close()
        with timed_stage("conditioning_finalize"):
            if encoded.ndim != 3 or encoded.shape[:2] != prepared.embeddings.shape[:2]:
                raise RuntimeError("TP2 backbone returned an invalid conditioning shape")
            if not torch.isfinite(encoded).all().item():
                raise RuntimeError("TP2 backbone returned NaN or Inf conditioning")
        return encoded, None, {"minimax_token_tags": prepared.token_tags}

    def encode_token_weights(self, tokens: dict):
        return self._encode_token_weights(tokens)

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        cond, pooled, extra = self._encode_token_weights(tokens)
        if return_dict:
            return {"cond": cond, "pooled_output": pooled, **extra}
        if return_pooled:
            return cond, pooled
        return cond

    def encode_from_tokens_scheduled(
        self, tokens, unprojected=False, add_dict: dict[str, Any] | None = None, show_pbar=True
    ):
        del unprojected, show_pbar
        result = self.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = result.pop("cond")
        result.update(add_dict or {})
        return [[cond, result]]

    def encode(self, text: str):
        return self.encode_from_tokens(self.tokenize(text))


__all__ = [
    "MINIMAX_CLIP_NAME",
    "MINIMAX_PAD_TOKEN",
    "MiniMaxH3TP2ClipProxy",
    "MiniMaxH3CheckpointPreprocessor",
    "PreparedMiniMaxConditioning",
    "TP2Qwen3VLBackbone",
    "load_minimax_h3_tp2_backbone",
    "prepare_minimax_token_sections",
]
