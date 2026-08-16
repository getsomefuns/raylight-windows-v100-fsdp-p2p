import os
import sys
from pathlib import Path

from tests.path_helpers import comfy_root


REPO_ROOT = Path(__file__).resolve().parents[1]
# Prefer this checkout over any installed custom-node copy and expose ComfyUI's
# namespace package for tests that import runtime modules during collection.
sys.path[:0] = [str(comfy_root()), str(REPO_ROOT / "src"), str(REPO_ROOT)]
