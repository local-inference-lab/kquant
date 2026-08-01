#!/usr/bin/env python3
"""Prepare the pinned GLM-5.2 EXL3 shared-H calibration encoder."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path


INPUT_SHA256 = {
    "encode_tr3_v31.py": (
        "e9a85a47e165c8d8644354cef611efbb81dfd9ba88544ca59f0c80ee6bc75032"
    ),
    "encode_b300.py": (
        "f378817b212dc9f4a8c9dc049803542e7c91748283f6e8ec1ebe0427be96aaf1"
    ),
    "calibration/reap_recall_calib.jsonl": (
        "cf247acc7c5da9f0600c7d6ab3b7c2fcfc54ec30b794e3b6047559285fa44df4"
    ),
}
OUTPUT_SHA256 = {
    "encode_tr3_v31.py": (
        "400c0df1c95c81c30a2ce31e060f0445a798fd29ad9339923d4e02e3ee40f6f7"
    ),
    "encode_b300.py": (
        "b41f9397a1754e67f41b2356db413b5da18228bcd3961c97023b4e1cabd01010"
    ),
}
PATCHES = {
    "encode_tr3_v31.py": "patches/encode_tr3_v31-shared-h-v1.patch",
    "encode_b300.py": "patches/encode_b300-shared-h-v1.patch",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(64 << 20):
            digest.update(block)
    return digest.hexdigest()


def verify_files(root: Path, expected: dict[str, str], label: str) -> None:
    for relative, digest in expected.items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"{label} file is missing: {path}")
        actual = sha256_file(path)
        if actual != digest:
            raise RuntimeError(
                f"{label} SHA mismatch for {relative}: {actual}, expected {digest}"
            )


def prepare(bundle: Path, output: Path) -> None:
    bundle = bundle.resolve()
    output = output.resolve()
    verify_files(bundle, INPUT_SHA256, "input bundle")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise RuntimeError(f"output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for source in bundle.iterdir():
        destination = output / source.name
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)

    recipe_root = Path(__file__).resolve().parent
    for target, relative_patch in PATCHES.items():
        patch_path = recipe_root / relative_patch
        subprocess.run(
            ["patch", "--batch", "--forward", "-p1", "-i", str(patch_path)],
            cwd=output,
            check=True,
        )
        if not (output / target).is_file():
            raise RuntimeError(f"patch did not produce {target}")
    verify_files(output, OUTPUT_SHA256, "patched bundle")
    subprocess.run(
        [
            "python3",
            "-m",
            "py_compile",
            str(output / "encode_tr3_v31.py"),
            str(output / "encode_b300.py"),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="Original calibration_encoder directory from the published model",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.bundle, args.output)
    print(f"Prepared verified shared-H encoder: {args.output.resolve()}")


if __name__ == "__main__":
    main()
