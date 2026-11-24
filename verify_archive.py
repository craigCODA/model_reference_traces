"""Verify the original archive files after cloning and fetching Git LFS assets."""

import argparse
import hashlib
import json
import sqlite3
import sys
import zlib
from contextlib import closing
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", action="store_true", help="also check SQLite integrity")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "ORIGINAL_ARCHIVE_MANIFEST.json").read_text(encoding="utf-8"))
    errors = []
    total = 0
    for entry in manifest["files"]:
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid manifest path: {relative}")
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"Missing: {relative}")
            continue
        if path.stat().st_size != entry["bytes"]:
            with path.open("rb") as source:
                pointer = source.read(42).startswith(b"version https://git-lfs.github.com/spec/v1")
            hint = " (Git LFS pointer; run git lfs pull)" if pointer else ""
            errors.append(f"Size mismatch: {relative}{hint}")
            continue
        digest = hashlib.sha256()
        crc = 0
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
                crc = zlib.crc32(block, crc)
        if digest.hexdigest() != entry["sha256"] or f"{crc:08x}" != entry["crc32"]:
            errors.append(f"Checksum mismatch: {relative}")
        total += entry["bytes"]
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Verified all {len(manifest['files'])} original files ({total:,} bytes).")
    if args.sqlite:
        uri = (ROOT / "raw/language.sqlite").resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as database:
            result = database.execute("PRAGMA quick_check").fetchall()
        if result != [("ok",)]:
            print(f"SQLite integrity check failed: {result}", file=sys.stderr)
            return 1
        print("SQLite integrity check: ok.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
