from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from tests.path_helpers import comfy_root

REPO_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_ROOT = comfy_root()


def test_custom_node_root_package_can_import_src_nodes():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(COMFYUI_ROOT)
    bootstrap = (
        "import importlib.util, pathlib, sys; "
        f"root=pathlib.Path({str(REPO_ROOT)!r}); "
        "spec=importlib.util.spec_from_file_location("
        "'raylight', root/'__init__.py', submodule_search_locations=[str(root)]); "
        "package=importlib.util.module_from_spec(spec); "
        "sys.modules['raylight']=package; spec.loader.exec_module(package); "
        "import raylight.nodes; import comfy; "
        "assert hasattr(comfy, 'sample'); print(raylight.nodes.__file__)"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            bootstrap,
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