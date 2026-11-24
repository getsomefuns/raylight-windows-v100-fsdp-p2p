"""Recovery copies must be verified, non-destructive and media-free."""
import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / 'scripts/runtime_snapshot.py'
    spec = importlib.util.spec_from_file_location('runtime_snapshot_test', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_snapshot_restore_preserves_code_without_model_or_user_media(tmp_path):
    tool = module()
    core = tmp_path / 'ComfyUI'
    core.mkdir()
    (core / 'main.py').write_bytes(b'official source\n')
    for name in ('models', 'input', 'output', 'user', '.git', 'custom_nodes'):
        (core / name).mkdir()
        (core / name / 'keep').write_bytes(b'never copy media')
    snapshot = tmp_path / 'backup'
    tool.create_snapshot({'core': core}, snapshot)
    assert tool.verify_snapshot(snapshot)['file_count'] == 1
    restored = tmp_path / 'restored'
    tool.restore_snapshot(snapshot, restored)
    assert (restored / 'core/main.py').read_bytes() == b'official source\n'
    assert sorted(p.name for p in (restored / 'core').iterdir()) == ['main.py']
    assert (core / 'models/keep').read_bytes() == b'never copy media'
    with pytest.raises(FileExistsError):
        tool.restore_snapshot(snapshot, restored)


def test_corrupt_or_traversing_backup_is_refused_before_restore(tmp_path):
    tool = module()
    original = tmp_path / 'settings.ps1'
    original.write_bytes(b'original')
    snapshot = tmp_path / 'backup'
    tool.create_snapshot({'launcher': original}, snapshot)
    (snapshot / 'launcher/settings.ps1').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='hash|integrity'):
        tool.restore_snapshot(snapshot, tmp_path / 'restored')
    assert not (tmp_path / 'restored').exists()
    manifest = json.loads((snapshot / 'manifest.json').read_text())
    manifest['files'][0]['path'] = '../escape'
    (snapshot / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='path'):
        tool.verify_snapshot(snapshot)


def test_partial_failure_is_not_a_complete_recovery_point(tmp_path, monkeypatch):
    tool = module()
    original = tmp_path / 'settings.ps1'
    original.write_bytes(b'original')
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(tool.shutil, 'copy2', fail)
    with pytest.raises(OSError, match='disk full'):
        tool.create_snapshot({'launcher': original}, tmp_path / 'backup')
    assert not (tmp_path / 'backup').exists()
    assert original.read_bytes() == b'original'


def test_snapshot_cannot_copy_itself_or_overwrite_an_existing_destination(tmp_path):
    tool = module()
    source = tmp_path / 'source'
    source.mkdir()
    with pytest.raises(ValueError, match='inside|overlap'):
        tool.create_snapshot({'python': source}, source / 'recursive')
    with pytest.raises(FileExistsError):
        tool.create_snapshot({'python': source}, source)


def test_snapshot_rechecks_all_files_before_publishing_complete_status(tmp_path, monkeypatch):
    tool = module()
    source = tmp_path / 'python'
    source.mkdir()
    (source / 'first').write_bytes(b'one')
    (source / 'second').write_bytes(b'two')
    original = tool.shutil.copy2
    copied = []
    def copy_then_corrupt(first, second):
        result = original(first, second)
        copied.append(second)
        if len(copied) == 2:
            copied[0].write_bytes(b'changed after copy')
        return result
    monkeypatch.setattr(tool.shutil, 'copy2', copy_then_corrupt)
    with pytest.raises(ValueError, match='integrity|hash'):
        tool.create_snapshot({'python': source}, tmp_path / 'backup')
    assert not (tmp_path / 'backup').exists()
