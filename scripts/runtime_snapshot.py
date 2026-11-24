#!/usr/bin/env python3
"""Verified runtime copies and recovery rehearsals; never overwrite or delete.

Create: runtime_snapshot.py create DEST core=COMFY python=PYTHON nodes=NODES
Verify: runtime_snapshot.py verify SNAPSHOT
Restore to a NEW rehearsal directory: runtime_snapshot.py restore SNAPSHOT DEST
Core media/user/model directories are deliberately not copied. Git bundles and
machine-specific inventories can be kept beside, not inside, this snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import uuid


CORE_EXTERNAL = frozenset({'models', 'input', 'output', 'user', 'temp', 'custom_nodes', '.git'})


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def _is_link(path):
    return path.is_symlink() or getattr(path, 'is_junction', lambda: False)()


def _safe_relative(value):
    normalized = value.replace('\\', '/')
    if (not normalized or ':' in normalized or PureWindowsPath(normalized).is_absolute()
            or any(part in ('', '.', '..') for part in normalized.split('/'))):
        raise ValueError(f'Unsafe snapshot path: {value!r}')
    return Path(normalized)


def _ignored(component, relative):
    parts = relative.parts
    return (any(part in ('.git', '__pycache__') for part in parts)
            or component == 'core' and bool(parts) and parts[0] in CORE_EXTERNAL)


def _source_files(component, source):
    if source.is_file():
        yield source, Path(source.name)
        return
    for folder, directories, files in os.walk(source, followlinks=False):
        base = Path(folder)
        directories[:] = [name for name in directories if not _ignored(component, (base / name).relative_to(source))]
        for name in directories:
            if _is_link(base / name):
                raise ValueError(f'Capture this link separately; it will not be followed: {base / name}')
        for name in sorted(files):
            path = base / name
            relative = path.relative_to(source)
            if _ignored(component, relative):
                continue
            if _is_link(path):
                raise ValueError(f'Capture this link separately; it will not be followed: {path}')
            yield path, relative


def _new_stage(destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_name(destination.name + '.partial-' + uuid.uuid4().hex)
    stage.mkdir()
    return stage


def create_snapshot(components, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if not components:
        raise ValueError('At least one explicit component is required')
    sources = {}
    for name, value in components.items():
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', name):
            raise ValueError('Invalid component name')
        original = Path(value)
        if _is_link(original):
            raise ValueError('Component must name a real source, not a link')
        source = original.resolve(strict=True)
        if source == Path(source.anchor) or destination.is_relative_to(source) or source.is_relative_to(destination):
            raise ValueError('Snapshot destination overlaps or is inside its source')
        sources[name] = source
    stage = _new_stage(destination)
    records = []
    # On error the partial directory is retained as evidence, never published
    # as a complete restore point and never used to overwrite the source.
    for component, source in sources.items():
        for path, relative in _source_files(component, source):
            target = stage / component / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            expected = digest(path)
            shutil.copy2(path, target)
            if digest(target) != expected:
                raise ValueError(f'Backup integrity failure: {path}')
            records.append({'path': target.relative_to(stage).as_posix(),
                            'bytes': target.stat().st_size, 'sha256': expected})
    manifest = {'schema': 1, 'status': 'complete', 'file_count': len(records),
                'components': {key: str(value) for key, value in sources.items()},
                'files': records, 'excluded_core_roots': sorted(CORE_EXTERNAL)}
    (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    verify_snapshot(stage)
    stage.rename(destination)
    return manifest


def verify_snapshot(snapshot):
    snapshot = Path(snapshot).resolve(strict=True)
    try:
        manifest = json.loads((snapshot / 'manifest.json').read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise ValueError('Incomplete or invalid recovery snapshot') from error
    if manifest.get('schema') != 1 or manifest.get('status') != 'complete':
        raise ValueError('Incomplete recovery snapshot')
    expected = set()
    for row in manifest['files']:
        relative = _safe_relative(row['path'])
        path = snapshot / relative
        if (not path.resolve().is_relative_to(snapshot) or _is_link(path)
                or not path.is_file() or path.stat().st_size != row['bytes'] or digest(path) != row['sha256']):
            raise ValueError(f'Snapshot integrity/hash failure: {relative}')
        expected.add(relative.as_posix())
    actual = {path.relative_to(snapshot).as_posix() for path in snapshot.rglob('*') if path.is_file()}
    if len(expected) != manifest['file_count'] or actual != expected | {'manifest.json'}:
        raise ValueError('Snapshot integrity inventory differs from manifest')
    return manifest


def restore_snapshot(snapshot, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    manifest = verify_snapshot(snapshot)
    source = Path(snapshot).resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('Restore destination overlaps its snapshot')
    stage = _new_stage(destination)
    for row in manifest['files']:
        relative = _safe_relative(row['path'])
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    shutil.copy2(source / 'manifest.json', stage / 'manifest.json')
    verify_snapshot(stage)
    stage.rename(destination)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    create = commands.add_parser('create')
    create.add_argument('destination', type=Path)
    create.add_argument('components', nargs='+', metavar='NAME=PATH')
    verify = commands.add_parser('verify')
    verify.add_argument('snapshot', type=Path)
    restore = commands.add_parser('restore')
    restore.add_argument('snapshot', type=Path)
    restore.add_argument('destination', type=Path)
    args = parser.parse_args(argv)
    if args.action == 'create':
        manifest = create_snapshot(dict(value.split('=', 1) for value in args.components), args.destination)
    elif args.action == 'verify':
        manifest = verify_snapshot(args.snapshot)
    else:
        manifest = restore_snapshot(args.snapshot, args.destination)
    print(json.dumps({'status': 'verified', 'files': manifest['file_count']}))


if __name__ == '__main__':
    main()
