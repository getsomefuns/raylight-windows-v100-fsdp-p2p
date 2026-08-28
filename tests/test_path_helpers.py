from pathlib import Path

from . import path_helpers


def _make_comfy_root(path: Path) -> Path:
    (path / "comfy").mkdir(parents=True)
    (path / "main.py").write_text("# fixture\n", encoding="utf-8")
    return path


def test_discover_comfy_root_supports_sibling_worktree_layout(tmp_path: Path) -> None:
    comfy = _make_comfy_root(tmp_path / "ComfyUI")
    repo = tmp_path / "raylight-windows-v100-fsdp-p2p"
    repo.mkdir()

    assert path_helpers.discover_comfy_root(repo) == comfy.resolve()


def test_discover_comfy_root_supports_custom_node_deployment_layout(tmp_path: Path) -> None:
    comfy = _make_comfy_root(tmp_path / "ComfyUI")
    repo = comfy / "custom_nodes" / "raylight"
    repo.mkdir(parents=True)

    assert path_helpers.discover_comfy_root(repo) == comfy.resolve()


def test_python_executable_is_resolved_from_comfy_environment(tmp_path: Path) -> None:
    comfy = _make_comfy_root(tmp_path / "ComfyUI")
    python = tmp_path / "Python310" / "python.exe"
    python.parent.mkdir()
    python.write_bytes(b"")

    assert path_helpers.discover_python_executable(comfy) == python.resolve()


def test_test_modules_do_not_use_ambiguous_absolute_tests_package() -> None:
    tests_root = Path(__file__).resolve().parent
    offenders = []
    for path in tests_root.glob("*.py"):
        if path.name == "test_path_helpers.py":
            continue
        if "from tests.path_helpers import" in path.read_text(encoding="utf-8"):
            offenders.append(path.name)

    assert offenders == []
