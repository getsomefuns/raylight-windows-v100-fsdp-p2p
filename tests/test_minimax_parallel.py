import ast
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
