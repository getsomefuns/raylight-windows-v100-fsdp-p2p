import ast
import json
import torch
from .path_helpers import comfy_root, repo_root


ROOT = comfy_root()
CORE = ROOT / "comfy/ldm/minimax/model.py"
RAYLIGHT = repo_root() / "src/raylight/diffusion_models/minimax/xdit_context_parallel.py"


def _module_symbols(path):
    symbols = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            symbols.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            symbols.update(target.id for target in targets if isinstance(target, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            symbols.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
    return symbols


def _function(path, function_name, class_name=None):
    nodes = ast.parse(path.read_text()).body
    if class_name is not None:
        nodes = next(node.body for node in nodes if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == function_name)


def _signature(function):
    args = function.args
    parameters = []

    def add(kind, argument, default=None):
        parameters.append((kind, argument.arg, ast.dump(default, include_attributes=False) if default is not None else None))

    for argument in args.posonlyargs:
        add("positional-only", argument)
    defaults = [None] * (len(args.args) - len(args.defaults)) + list(args.defaults)
    for argument, default in zip(args.args, defaults):
        add("positional", argument, default)
    if args.vararg is not None:
        add("var-positional", args.vararg)
    for argument, default in zip(args.kwonlyargs, args.kw_defaults):
        add("keyword-only", argument, default)
    if args.kwarg is not None:
        add("var-keyword", args.kwarg)
    return parameters


def test_minimax_core_imports_exist():
    imports = next(
        node.names
        for node in ast.parse(RAYLIGHT.read_text()).body
        if isinstance(node, ast.ImportFrom) and node.module == "comfy.ldm.minimax.model"
    )
    missing = {alias.name for alias in imports} - _module_symbols(CORE)
    assert not missing


def test_minimax_usp_forward_signature_matches_core():
    assert _signature(_function(RAYLIGHT, "usp_dit_forward")) == _signature(
        _function(CORE, "_forward", "MiniMaxH3Model")
    )


def test_minimax_sequence_split_owns_only_local_storage(monkeypatch):
    import raylight.distributed_modules.attention as attention

    monkeypatch.setattr(attention, "_ATTN_TYPE", "TORCH")
    monkeypatch.setattr(attention, "_SYNC_ULYSSES", True)
    monkeypatch.setattr(attention, "make_xfuser_attention", lambda *_args: None)
    import raylight.diffusion_models.minimax.xdit_context_parallel as parallel

    monkeypatch.setattr(parallel, "get_sequence_parallel_world_size", lambda: 2)
    monkeypatch.setattr(parallel, "get_sequence_parallel_rank", lambda: 1)
    hidden = torch.arange(48, dtype=torch.float16).reshape(8, 6)
    rope = torch.arange(32, dtype=torch.float16).reshape(1, 8, 2, 2)

    local_hidden, local_rope, segments = parallel._split_packed_sequence(
        hidden,
        rope,
        [(0, 3, 0), (3, 8, 1)],
    )

    assert torch.equal(local_hidden, hidden[4:8])
    assert torch.equal(local_rope, rope[:, 4:8])
    assert segments == [(0, 4, 1)]
    assert local_hidden.untyped_storage().data_ptr() != hidden.untyped_storage().data_ptr()
    assert local_rope.untyped_storage().data_ptr() != rope.untyped_storage().data_ptr()


def test_minimax_packed_inputs_are_released_before_transformer_blocks():
    function = _function(RAYLIGHT, "usp_dit_forward")
    deleted = {
        target.id
        for node in ast.walk(function)
        if isinstance(node, ast.Delete)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    assert {
        "video_x",
        "audio_x",
        "context",
        "video_rows",
        "audio_rows",
        "cond_video_rows",
        "cond_audio_rows",
        "all_video_rows",
        "all_audio_rows",
        "video_embed",
        "audio_embed",
        "text_states",
    } <= deleted


def test_minimax_qk_norm_is_written_back_into_projection_storage(monkeypatch):
    import raylight.distributed_modules.attention as attention

    monkeypatch.setattr(attention, "_ATTN_TYPE", "TORCH")
    monkeypatch.setattr(attention, "_SYNC_ULYSSES", True)
    monkeypatch.setattr(attention, "make_xfuser_attention", lambda *_args: None)
    import raylight.diffusion_models.minimax.xdit_context_parallel as parallel

    q = torch.arange(12, dtype=torch.float16).reshape(2, 2, 3)
    k = torch.arange(12, 24, dtype=torch.float16).reshape(2, 2, 3)
    q_ptr = q.untyped_storage().data_ptr()
    k_ptr = k.untyped_storage().data_ptr()

    normalized_q, normalized_k = parallel._normalize_qk_in_place(
        lambda value: value.float().add(1),
        lambda value: value.float().sub(1),
        q,
        k,
    )

    assert normalized_q.untyped_storage().data_ptr() == q_ptr
    assert normalized_k.untyped_storage().data_ptr() == k_ptr
    torch.testing.assert_close(normalized_q, torch.arange(12).reshape(2, 2, 3).half() + 1)
    torch.testing.assert_close(normalized_k, torch.arange(12, 24).reshape(2, 2, 3).half() - 1)


def test_minimax_qk_norm_chunks_with_one_unshard_per_norm(monkeypatch):
    import raylight.distributed_modules.attention as attention

    monkeypatch.setattr(attention, "_ATTN_TYPE", "TORCH")
    monkeypatch.setattr(attention, "_SYNC_ULYSSES", True)
    monkeypatch.setattr(attention, "make_xfuser_attention", lambda *_args: None)
    import raylight.diffusion_models.minimax.xdit_context_parallel as parallel

    class FakeFSDPNorm:
        def __init__(self, offset):
            self.offset = offset
            self.unshard_calls = 0
            self.reshard_calls = 0
            self.chunk_sizes = []

        def unshard(self):
            self.unshard_calls += 1

        def reshard(self):
            self.reshard_calls += 1

        def forward(self, value):
            self.chunk_sizes.append(value.shape[0])
            return value.float().add(self.offset)

    q_norm = FakeFSDPNorm(1)
    k_norm = FakeFSDPNorm(-1)
    q = torch.arange(21, dtype=torch.float16).reshape(7, 1, 3)
    k = torch.arange(21, 42, dtype=torch.float16).reshape(7, 1, 3)

    parallel._normalize_qk_in_place(q_norm, k_norm, q, k, chunk_tokens=3)

    assert q_norm.chunk_sizes == [3, 3, 1]
    assert k_norm.chunk_sizes == [3, 3, 1]
    assert (q_norm.unshard_calls, q_norm.reshard_calls) == (1, 1)
    assert (k_norm.unshard_calls, k_norm.reshard_calls) == (1, 1)
    torch.testing.assert_close(q, torch.arange(21).reshape(7, 1, 3).half() + 1)
    torch.testing.assert_close(k, torch.arange(21, 42).reshape(7, 1, 3).half() - 1)


def test_minimax_rope_is_written_back_in_sequence_chunks(monkeypatch):
    import raylight.distributed_modules.attention as attention

    monkeypatch.setattr(attention, "_ATTN_TYPE", "TORCH")
    monkeypatch.setattr(attention, "_SYNC_ULYSSES", True)
    monkeypatch.setattr(attention, "make_xfuser_attention", lambda *_args: None)
    import raylight.diffusion_models.minimax.xdit_context_parallel as parallel

    chunk_sizes = []

    def fake_rope(query, key, rope_freqs):
        chunk_sizes.append((query.shape[0], rope_freqs.shape[1]))
        return query.float().add(1), key.float().sub(1)

    monkeypatch.setattr(
        parallel.comfy.quant_ops.ck,
        "apply_rope_split_half",
        fake_rope,
    )
    q = torch.arange(28, dtype=torch.float16).reshape(7, 1, 4)
    k = torch.arange(28, 56, dtype=torch.float16).reshape(7, 1, 4)
    rope = torch.zeros(1, 7, 1, 2, dtype=torch.float16)
    q_ptr = q.untyped_storage().data_ptr()
    k_ptr = k.untyped_storage().data_ptr()

    parallel._apply_rope_in_place(q, k, rope, rot=4, chunk_tokens=3)

    assert chunk_sizes == [(3, 3), (3, 3), (1, 1)]
    assert q.untyped_storage().data_ptr() == q_ptr
    assert k.untyped_storage().data_ptr() == k_ptr
    torch.testing.assert_close(q, torch.arange(28).reshape(7, 1, 4).half() + 1)
    torch.testing.assert_close(k, torch.arange(28, 56).reshape(7, 1, 4).half() - 1)


class _RecordingIdentity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.chunk_sizes = []

    def forward(self, value):
        self.chunk_sizes.append(value.shape[0])
        return value.clone()


class _RecordingLinear(torch.nn.Linear):
    def __init__(self):
        super().__init__(3, 2, bias=False, dtype=torch.float32)
        self.chunk_sizes = []
        with torch.no_grad():
            self.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

    def forward(self, value):
        self.chunk_sizes.append(value.shape[0])
        return super().forward(value)


class _FakeGatherGroup:
    def __init__(self, gathered):
        self.gathered = gathered
        self.inputs = []

    def all_gather(self, value, dim=0):
        self.inputs.append((value.clone(), dim))
        return self.gathered.clone()


def test_minimax_final_projection_uses_only_rank_local_target_rows_in_chunks(monkeypatch):
    import raylight.distributed_modules.attention as attention

    monkeypatch.setattr(attention, "_ATTN_TYPE", "TORCH")
    monkeypatch.setattr(attention, "_SYNC_ULYSSES", True)
    monkeypatch.setattr(attention, "make_xfuser_attention", lambda *_args: None)
    import raylight.diffusion_models.minimax.xdit_context_parallel as parallel

    norm = _RecordingIdentity()
    projector = _RecordingLinear()
    final_layer = type("FinalLayer", (), {"norm": norm})()
    local_hidden = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    shift = torch.tensor([[3.0, 3.0, 3.0]])
    scale = torch.tensor([[1.0, 1.0, 1.0]])

    projected, counts = parallel._project_local_final_segment(
        final_layer,
        local_hidden,
        shift,
        scale,
        (2, 9, 0),
        projector,
        rank=1,
        world_size=2,
        chunk_tokens=2,
    )

    expected_hidden = local_hidden[:4] * 2.0 + 3.0
    assert counts == [3, 4]
    assert norm.chunk_sizes == [2, 2]
    assert projector.chunk_sizes == [2, 2]
    torch.testing.assert_close(projected, expected_hidden[:, :2])


def test_minimax_final_projection_pads_short_rank_to_collective_chunk_count(monkeypatch):
    import raylight.distributed_modules.attention as attention

    monkeypatch.setattr(attention, "_ATTN_TYPE", "TORCH")
    monkeypatch.setattr(attention, "_SYNC_ULYSSES", True)
    monkeypatch.setattr(attention, "make_xfuser_attention", lambda *_args: None)
    import raylight.diffusion_models.minimax.xdit_context_parallel as parallel

    norm = _RecordingIdentity()
    projector = _RecordingLinear()
    final_layer = type("FinalLayer", (), {"norm": norm})()
    local_hidden = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    shift = torch.tensor([[3.0, 3.0, 3.0]])
    scale = torch.tensor([[1.0, 1.0, 1.0]])

    projected, counts = parallel._project_local_final_segment(
        final_layer,
        local_hidden,
        shift,
        scale,
        (2, 9, 0),
        projector,
        rank=0,
        world_size=2,
        chunk_tokens=2,
    )

    expected_hidden = local_hidden[2:5] * 2.0 + 3.0
    assert counts == [3, 4]
    assert norm.chunk_sizes == [2, 2]
    assert projector.chunk_sizes == [2, 2]
    assert projected.shape == (3, 2)
    torch.testing.assert_close(projected, expected_hidden[:, :2])


def test_minimax_final_projection_uses_empty_slice_when_segment_is_on_peer_rank(monkeypatch):
    import raylight.distributed_modules.attention as attention

    monkeypatch.setattr(attention, "_ATTN_TYPE", "TORCH")
    monkeypatch.setattr(attention, "_SYNC_ULYSSES", True)
    monkeypatch.setattr(attention, "make_xfuser_attention", lambda *_args: None)
    import raylight.diffusion_models.minimax.xdit_context_parallel as parallel

    norm = _RecordingIdentity()
    projector = _RecordingLinear()
    final_layer = type("FinalLayer", (), {"norm": norm})()
    local_hidden = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    shift = torch.tensor([[3.0, 3.0, 3.0]])
    scale = torch.tensor([[1.0, 1.0, 1.0]])

    projected, counts = parallel._project_local_final_segment(
        final_layer,
        local_hidden,
        shift,
        scale,
        (0, 4, 0),
        projector,
        rank=1,
        world_size=2,
        chunk_tokens=2,
    )

    assert counts == [4, 0]
    assert norm.chunk_sizes == [2, 2]
    assert projector.chunk_sizes == [2, 2]
    assert projected.shape == (0, 2)


def test_minimax_gathers_only_small_projected_rows_and_removes_rank_padding(monkeypatch):
    import raylight.distributed_modules.attention as attention

    monkeypatch.setattr(attention, "_ATTN_TYPE", "TORCH")
    monkeypatch.setattr(attention, "_SYNC_ULYSSES", True)
    monkeypatch.setattr(attention, "make_xfuser_attention", lambda *_args: None)
    import raylight.diffusion_models.minimax.xdit_context_parallel as parallel

    local = torch.tensor([[20.0], [21.0], [22.0], [23.0]])
    gathered = torch.tensor([[10.0], [11.0], [0.0], [0.0], [20.0], [21.0], [22.0], [23.0]])
    group = _FakeGatherGroup(gathered)

    result = parallel._gather_projected_rows(local, [2, 4], group)

    assert group.inputs[0][0].shape == (4, 1)
    assert group.inputs[0][1] == 0
    assert result.flatten().tolist() == [10.0, 11.0, 20.0, 21.0, 22.0, 23.0]


def test_minimax_cuda_memory_trace_emits_only_new_high_watermarks(monkeypatch, capsys):
    import raylight.distributed_modules.attention as attention

    monkeypatch.setattr(attention, "_ATTN_TYPE", "TORCH")
    monkeypatch.setattr(attention, "_SYNC_ULYSSES", True)
    monkeypatch.setattr(attention, "make_xfuser_attention", lambda *_args: None)
    import raylight.diffusion_models.minimax.xdit_context_parallel as parallel

    state = {"allocated": 100, "peak": 200}
    monkeypatch.setenv("RAYLIGHT_H3_MEMORY_TRACE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: state["allocated"])
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: state["peak"])
    parallel._TRACE_HIGH_WATER.clear()

    parallel._trace_cuda_memory("block_0_end")
    parallel._trace_cuda_memory("block_1_end")
    state.update(allocated=150, peak=300)
    parallel._trace_cuda_memory("final_video_local")

    records = [json.loads(line.split(" ", 1)[1]) for line in capsys.readouterr().out.splitlines()]
    assert [record["event"] for record in records] == ["block_0_end", "final_video_local"]
    assert records[-1]["peak_allocated_bytes"] == 300
