"""Extract the stored values used by the checkpoint probes and index the files."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw_sets"
EXTRACTORS = (
    "extract_graph_inkling.py",
    "extract_neural.py",
    "extract_classic.py",
    "extract_language.py",
)
MODELS = (
    "inkling-2-mini", "mamba-130m", "squeezenet1_1", "encodec-24khz",
    "graphconv", "opencv-haar", "langid", "vmaf", "jieba", "riva-arpa", "glove",
)


def build_manifest():
    models = []
    position = 0
    for name in MODELS:
        directory = RAW / name
        index = directory / "index.json"
        data = json.loads(index.read_text(encoding="utf-8"))
        count = data["scalar_count"]
        if type(count) is not int or count < 1:
            raise ValueError(f"Invalid scalar count in {index}")
        files = [
            {"path": path.relative_to(RAW).as_posix(), "bytes": path.stat().st_size}
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path != index
        ]
        if not files or any(not file["bytes"] for file in files):
            raise ValueError(f"An export in {directory} is missing data")
        models.append({
            "id": name,
            "model": data["model"],
            "index": index.relative_to(RAW).as_posix(),
            "scalar_count": count,
            "global_scalar_start": position,
            "global_scalar_stop_exclusive": position + count,
            "files": files,
        })
        position += count
    report = {
        "format": "checkpoint-scalars/1",
        "scope": "Stored values in the local weight sets selected for the traces, including the original Inkling example.",
        "indexing": "All scalar positions are zero-based. Each model index describes its file format, source keys, and stored scalar order.",
        "model_count": len(models),
        "scalar_count": position,
        "models": models,
    }
    pending = RAW / "manifest.json.pending"
    pending.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pending.replace(RAW / "manifest.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-only", action="store_true", help="Index the raw sets already on disk")
    args = parser.parse_args()
    if not args.index_only:
        for script in EXTRACTORS:
            subprocess.run([sys.executable, "-B", script], cwd=ROOT, check=True)
    report = build_manifest()
    print(f"Indexed {report['scalar_count']:,} stored values across {report['model_count']} checkpoint exports.")


if __name__ == "__main__":
    main()
