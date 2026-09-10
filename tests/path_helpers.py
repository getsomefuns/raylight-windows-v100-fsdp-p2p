import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def discover_comfy_root(root: Path, configured: str | os.PathLike[str] | None = None) -> Path:
    root = root.resolve()
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))
    candidates.append(root.parent / "ComfyUI")
    candidates.extend((root, *root.parents))

    searched: list[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in searched:
            continue
        searched.append(candidate)
        if (candidate / "comfy").is_dir() and (candidate / "main.py").is_file():
            return candidate
    raise RuntimeError(
        "Unable to locate ComfyUI. Set RAYLIGHT_COMFY_ROOT or place the "
        "worktree beside ComfyUI or below ComfyUI/custom_nodes. Searched: "
        + ", ".join(str(candidate) for candidate in searched)
    )


def comfy_root() -> Path:
    return discover_comfy_root(repo_root(), os.environ.get("RAYLIGHT_COMFY_ROOT"))


def discover_python_executable(
    comfy: Path,
    configured: str | os.PathLike[str] | None = None,
) -> Path:
    candidates = [Path(configured)] if configured else []
    environment_root = comfy.resolve().parent
    candidates.extend(
        (
            environment_root / "Python312" / "python.exe",
            environment_root / "Python310" / "python.exe",
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(
        "Unable to locate the Raylight Python executable. Set RAYLIGHT_PYTHON. "
        "Searched: " + ", ".join(str(candidate) for candidate in candidates)
    )


def python_executable() -> Path:
    return discover_python_executable(comfy_root(), os.environ.get("RAYLIGHT_PYTHON"))
