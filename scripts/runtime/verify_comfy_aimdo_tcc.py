from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from packaging.specifiers import SpecifierSet
from packaging.version import Version


WINDOWS_PRESSURE_HEADROOM_BYTES = 512 * 1024 * 1024
UPSTREAM_VERSION = "0.4.13"
UPSTREAM_REPOSITORY = "https://github.com/Comfy-Org/comfy-aimdo.git"
UPSTREAM_TAG = "v0.4.13"
UPSTREAM_COMMIT = "3f4259a502e7df7508dca7c30bd07eb936edf5a6"
DETOURS_REPOSITORY = "https://github.com/microsoft/Detours.git"
DETOURS_TAG = "v4.0.1"
DETOURS_COMMIT = "e4bfd6b03e50de46b47abfbd1e46b384f0c5f833"


def evaluate_backend_policy(*, wddm_available: bool, nvml_available: bool) -> dict[str, object]:
    if wddm_available:
        backend = "wddm"
    elif nvml_available:
        backend = "nvml"
    else:
        backend = None
    return {
        "allowed": backend is not None,
        "backend": backend,
        "pressure_headroom_bytes": WINDOWS_PRESSURE_HEADROOM_BYTES,
    }


def evaluate_version(local_version: str) -> dict[str, object]:
    satisfies = SpecifierSet(f"=={UPSTREAM_VERSION}").contains(Version(local_version))
    return {
        "upstream_version": UPSTREAM_VERSION,
        "local_version": local_version,
        "satisfies_exact_pin": satisfies,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def validate_build_manifest(path: Path) -> list[str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    expected_scalars = {
        "schema": 1,
        "version": "0.4.13+tcc1",
        "pressure_headroom_bytes": WINDOWS_PRESSURE_HEADROOM_BYTES,
    }
    for key, expected in expected_scalars.items():
        if manifest.get(key) != expected:
            errors.append(f"{key} must be {expected}")

    for name, expected in {
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "tag": UPSTREAM_TAG,
            "commit": UPSTREAM_COMMIT,
        },
        "detours": {
            "repository": DETOURS_REPOSITORY,
            "tag": DETOURS_TAG,
            "commit": DETOURS_COMMIT,
        },
    }.items():
        if manifest.get(name) != expected:
            errors.append(f"{name} source pin mismatch")

    manifest_root = path.resolve().parent
    for name, entry in manifest.get("artifacts", {}).items():
        artifact = (manifest_root / entry["path"]).resolve()
        try:
            artifact.relative_to(manifest_root)
        except ValueError:
            errors.append(f"artifact {name} is outside manifest directory")
            continue
        if not artifact.is_file():
            errors.append(f"artifact {name} is missing")
        elif _sha256(artifact) != str(entry.get("sha256", "")).upper():
            errors.append(f"artifact {name} SHA-256 mismatch")
    return errors


def _binary_flag(value: str) -> bool:
    if value not in {"0", "1"}:
        raise argparse.ArgumentTypeError("expected 0 or 1")
    return value == "1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the comfy-aimdo Windows TCC build contract")
    subparsers = parser.add_subparsers(dest="command", required=True)

    policy = subparsers.add_parser("policy")
    policy.add_argument("--wddm", required=True, type=_binary_flag)
    policy.add_argument("--nvml", required=True, type=_binary_flag)

    metadata = subparsers.add_parser("metadata")
    metadata.add_argument("--version", required=True)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--manifest", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "policy":
        payload = evaluate_backend_policy(wddm_available=args.wddm, nvml_available=args.nvml)
        result = 0 if payload["allowed"] else 1
    elif args.command == "metadata":
        payload = evaluate_version(args.version)
        result = 0 if payload["satisfies_exact_pin"] else 1
    else:
        errors = validate_build_manifest(args.manifest)
        payload = {"errors": errors, "passed": not errors}
        result = 0 if not errors else 1
    print(json.dumps(payload, sort_keys=True))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
