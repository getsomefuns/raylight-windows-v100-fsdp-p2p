import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def comfy_root() -> Path:
    root = repo_root()
    candidates = []
    configured = os.environ.get("RAYLIGHT_COMFY_ROOT")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        (
            root.parent / "ComfyUI",
            root.parents[1],
        )
    )
    for candidate in candidates:
        if (candidate / "comfy").is_dir():
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(
        "Unable to locate ComfyUI. Set RAYLIGHT_COMFY_ROOT or place the "
        f"worktree beside the ComfyUI directory. Searched: {searched}"
    )
