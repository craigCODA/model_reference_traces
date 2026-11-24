"""Replay the ten CPU weight probes and collect the observations in one file."""

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GROUPS = (
    ("graph_probe.py", "graph_results.json", 1),
    ("neural_probe.py", "neural_results.json", 3),
    ("classic_probe.py", "classic_results.json", 3),
    ("language_probe.py", "language_results.json", 3),
)


def read_models(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if "model" in data:
        return [data]
    for key in ("models", "traces", "results"):
        if isinstance(data.get(key), list):
            return data[key]
    return [entry for entry in data.values() if isinstance(entry, dict) and "model" in entry]


def main():
    results = []
    for script, filename, expected in GROUPS:
        completed = subprocess.run(
            [sys.executable, "-B", script], cwd=ROOT, capture_output=True, text=True, errors="replace",
        )
        if completed.returncode:
            raise RuntimeError(f"{script} failed. {completed.stderr.strip() or completed.stdout.strip()}")
        entries = read_models(ROOT / filename)
        if len(entries) != expected:
            raise ValueError(f"{filename} should contain {expected} model traces")
        for entry in entries:
            name = entry.get("model") or entry.get("id")
            if not isinstance(name, str) or not name:
                raise ValueError(f"A trace in {filename} lacks a model name")
            results.append({**entry, "model": name})
        print(f"{script} verified {expected} model traces", flush=True)

    names = {entry["model"] for entry in results}
    if len(results) != 10 or len(names) != 10:
        raise ValueError("Expected ten distinct trained-checkpoint traces")
    report = {
        "scope": "One selected computation per published trained checkpoint, replayed on CPU.",
        "models": results,
    }
    temporary = ROOT / "all_results.json.pending"
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(ROOT / "all_results.json")
    print("Verified 10 distinct checkpoint traces and wrote all_results.json.")


if __name__ == "__main__":
    main()
