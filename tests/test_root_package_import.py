from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_NODES_ROOT = REPO_ROOT.parent
COMFYUI_ROOT = REPO_ROOT.parents[1]


def test_custom_node_root_package_can_import_src_nodes():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CUSTOM_NODES_ROOT)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import raylight.nodes; import comfy; "
            "assert hasattr(comfy, 'sample'); print(raylight.nodes.__file__)",
        ],
        cwd=tempfile.gettempdir(),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "src" in completed.stdout
    assert "raylight" in completed.stdout


def test_nodes_explicitly_imports_comfy_sample():
    source = (REPO_ROOT / "src" / "raylight" / "nodes.py").read_text(encoding="utf-8")
    assert "import comfy.sample" in source


if __name__ == "__main__":
    test_custom_node_root_package_can_import_src_nodes()