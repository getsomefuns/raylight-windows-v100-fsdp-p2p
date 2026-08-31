from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .path_helpers import comfy_root


REPO_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_ROOT = comfy_root()


def test_root_import_registers_adapter_without_importing_director_package():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(COMFYUI_ROOT)
    blocker_source = (
        "class BlockDirector(importlib.abc.MetaPathFinder):\n"
        " def find_spec(self, fullname, path=None, target=None):\n"
        "  if fullname.startswith('ComfyUI_MiniMaxH3_Director'):\n"
        "   raise ImportError('Director import blocked by regression test')\n"
        "  return None\n"
    )
    bootstrap = (
        "import importlib.abc, importlib.util, pathlib, sys; "
        f"exec({blocker_source!r}); "
        "sys.meta_path.insert(0, BlockDirector()); "
        f"root=pathlib.Path({str(REPO_ROOT)!r}); "
        "spec=importlib.util.spec_from_file_location("
        "'raylight', root/'__init__.py', submodule_search_locations=[str(root)]); "
        "package=importlib.util.module_from_spec(spec); "
        "sys.modules['raylight']=package; spec.loader.exec_module(package); "
        "node=package.NODE_CLASS_MAPPINGS['RayMMH3DirectorModelAdapter']; "
        "assert package.NODE_DISPLAY_NAME_MAPPINGS['RayMMH3DirectorModelAdapter'] == "
        "'Ray MODEL Adapter (MMH3Director)'; "
        "assert node.RETURN_TYPES == ('MODEL',); "
        "tp=package.NODE_CLASS_MAPPINGS['RayMiniMaxH3DualGPUCLIPLoader']; "
        "assert package.NODE_DISPLAY_NAME_MAPPINGS['RayMiniMaxH3DualGPUCLIPLoader'] == "
        "'Load MiniMax H3 Text Encoder (Dual V100 TP2)'; "
        "assert tp.RETURN_TYPES == ('CLIP', 'RAY_ACTORS_INIT'); print('PASS')"
    )

    completed = subprocess.run(
        [sys.executable, "-c", bootstrap],
        cwd=tempfile.gettempdir(),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "PASS" in completed.stdout
