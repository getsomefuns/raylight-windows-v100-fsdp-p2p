"""Keep ComfyUI node metadata aligned with the installable package."""
import ast
from pathlib import Path
import re


def test_node_and_package_release_versions_agree():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / '__init__.py').read_text(encoding='utf-8'))
    node_version = next(
        ast.literal_eval(item.value)
        for item in tree.body if isinstance(item, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == '__version__'
                for target in item.targets)
    )
    package_version = re.search(
        r'^version\s*=\s*"([^"]+)"',
        (root / 'pyproject.toml').read_text(encoding='utf-8'), re.MULTILINE,
    ).group(1)
    assert node_version == package_version
