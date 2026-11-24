"""Extract local Jieba, Riva and GloVe weights without changing their literals."""

import ast
from collections import Counter
import gzip
import json
from pathlib import Path
import re
from uuid import uuid4

import numpy as np


ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights"
OUTPUT = ROOT / "raw_sets"


def source_path(path):
    return path.relative_to(ROOT).as_posix()


def jieba_literals(node, keys=()):
    if isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values, strict=True):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                raise ValueError("Expected a string key in the Jieba table")
            yield from jieba_literals(value, (*keys, key.value))
        return

    magnitude = node
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        magnitude = node.operand
    if not isinstance(magnitude, ast.Constant) or type(magnitude.value) not in (int, float):
        raise ValueError("Expected a numeric literal in the Jieba table")
    yield keys, node


def extract_jieba(folder):
    weights = WEIGHTS / "jieba-v0.42.1"
    values = folder / "values.jsonl"
    tables = {}
    scalar_count = 0
    with values.open("w", encoding="utf-8", newline="\n") as output:
        for table in ("start", "trans", "emit"):
            path = weights / f"prob_{table}.py"
            source = path.read_text(encoding="utf-8-sig")
            lines = source.encode("utf-8").splitlines()
            [assignment] = [
                statement for statement in ast.parse(source).body
                if isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == "P"
            ]
            first = scalar_count
            for keys, node in jieba_literals(assignment.value):
                if node.lineno != node.end_lineno:
                    raise ValueError("Expected a single-line numeric literal")
                literal = lines[node.lineno - 1][node.col_offset:node.end_col_offset].decode("ascii")
                record = {
                    "scalar_index": scalar_count,
                    "table": table,
                    "keys": keys,
                    "source_line": node.lineno,
                    "source_byte_column": node.col_offset,
                    "value_literal": literal,
                }
                output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                scalar_count += 1
            tables[table] = {
                "source": source_path(path),
                "first_scalar_index": first,
                "scalar_count": scalar_count - first,
            }

    with values.open(encoding="utf-8") as output:
        observed = sum(1 for index, line in enumerate(output) if json.loads(line)["scalar_index"] == index)
    if observed != scalar_count or tables["start"]["scalar_count"] != 4 or tables["trans"]["scalar_count"] != 8:
        raise ValueError("Jieba scalar counts failed validation")
    return {
        "model": "Jieba v0.42.1 finalseg",
        "scalar_count": scalar_count,
        "sources": [table["source"] for table in tables.values()],
        "tables": tables,
        "formats": {"values.jsonl": "UTF-8 JSON Lines, one original numeric literal string per record"},
        "lookup": {
            "record_index": "Zero-based JSONL record index equals scalar_index in source dictionary order.",
            "keys_by_table": {
                "start": ["state"],
                "trans": ["previous_state", "next_state"],
                "emit": ["state", "character"],
            },
            "source_position": "One-based source line and zero-based UTF-8 byte column in the table's source file.",
        },
    }


def extract_riva(folder):
    path = WEIGHTS / "nvidia-riva-arpa" / "3-gram.pruned.3e-7.arpa"
    values = folder / "values.txt.gz"
    records = folder / "records.tsv.gz"
    declared = {}
    counts = Counter()
    backoffs = Counter()
    scalar_count = 0
    record_index = 0
    order = 0
    ended = False
    with (
        path.open(encoding="utf-8") as source,
        gzip.open(values, "wt", encoding="ascii", newline="\n", compresslevel=1) as output,
        gzip.open(records, "wt", encoding="utf-8", newline="\n", compresslevel=1) as keys,
    ):
        keys.write("source_record_index\tsection_record_index\tsource_line\torder\tngram\tprobability_scalar_index\tbackoff_scalar_index\n")
        for line_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line or line == "\\data\\":
                continue
            if line == "\\end\\":
                ended = True
                break
            if line.startswith("ngram "):
                label, count = line.split("=", maxsplit=1)
                declared[int(label.split()[1])] = int(count)
                continue
            section = re.fullmatch(r"\\(\d+)-grams:", line)
            if section:
                order = int(section[1])
                continue

            fields = line.split()
            if order == 0 or len(fields) not in (order + 1, order + 2):
                raise ValueError(f"Invalid ARPA record on line {line_number}")
            gram = " ".join(fields[1:order + 1])
            probability_index = scalar_count
            output.write(fields[0] + "\n")
            scalar_count += 1

            backoff_index = ""
            if len(fields) == order + 2:
                backoff_index = str(scalar_count)
                output.write(fields[-1] + "\n")
                scalar_count += 1
                backoffs[order] += 1
            keys.write(f"{record_index}\t{counts[order]}\t{line_number}\t{order}\t{gram}\t{probability_index}\t{backoff_index}\n")
            record_index += 1
            counts[order] += 1

    if not ended or declared != dict(counts) or scalar_count != record_index + sum(backoffs.values()):
        raise ValueError("ARPA source counts failed validation")
    with gzip.open(values, "rb") as output, gzip.open(records, "rb") as keys:
        if sum(1 for _ in output) != scalar_count or sum(1 for _ in keys) != record_index + 1:
            raise ValueError("ARPA output counts failed validation")
    return {
        "model": "NVIDIA Riva ASR English LM deployable_v1.0 pruned trigram",
        "scalar_count": scalar_count,
        "sources": [source_path(path)],
        "ngram_count_by_order": dict(counts),
        "probability_scalar_count": record_index,
        "backoff_scalar_count": sum(backoffs.values()),
        "backoff_count_by_order": {order: backoffs[order] for order in counts},
        "formats": {
            "values.txt.gz": "Gzip-compressed ASCII, one original base-10 logarithm literal per line.",
            "records.tsv.gz": "Gzip-compressed UTF-8 TSV, one n-gram per source record plus a header.",
        },
        "lookup": {
            "keys": ["order", "ngram"],
            "source_record_index": "Zero-based among all n-gram rows in source file order.",
            "section_record_index": "Zero-based within the source section for the n-gram's order.",
            "source_line": "One-based line in the ARPA source file.",
            "scalar_indices": "The probability and backoff indices are zero-based line indices in values.txt.gz after decompression. A blank backoff index means that field is absent from the source record.",
        },
    }


def extract_glove(folder):
    path = WEIGHTS / "glove-wiki-gigaword-50" / "glove-wiki-gigaword-50.gz"
    value_path = folder / "values.npy"
    vocab_path = folder / "vocabulary.tsv"
    dtype = np.dtype("S16")
    batch_size = 4096
    maximum_length = 0
    samples = {}
    buffer = []
    written = 0
    observed = 0

    with gzip.open(path, "rb") as source, vocab_path.open("wb") as vocabulary:
        rows, dimensions = map(int, next(source).split())
        scalar_count = rows * dimensions
        sample_rows = {0, rows // 2, rows - 1}
        values = np.lib.format.open_memmap(value_path, mode="w+", dtype=dtype, shape=(scalar_count,))
        vocabulary.write(b"word_row\tword\n")
        for row, line in enumerate(source):
            word, *literals = line.split()
            if len(literals) != dimensions:
                raise ValueError(f"Wrong GloVe dimension on vocabulary row {row}")
            maximum_length = max(maximum_length, max(map(len, literals)))
            if maximum_length > dtype.itemsize:
                raise ValueError(f"GloVe literal on vocabulary row {row} exceeds .npy element width")
            vocabulary.write(str(row).encode("ascii") + b"\t" + word + b"\n")
            buffer.extend(literals)
            observed += 1
            if row in sample_rows:
                samples[row] = (word, literals)
            if observed % batch_size == 0 or observed == rows:
                end = written + len(buffer)
                values[written:end] = np.asarray(buffer, dtype=dtype)
                written = end
                buffer.clear()
        values.flush()
        del values

    if (rows, dimensions, observed, written) != (400_000, 50, 400_000, 20_000_000):
        raise ValueError("GloVe source or output shape failed validation")
    values = np.load(value_path, mmap_mode="r", allow_pickle=False)
    if values.shape != (scalar_count,) or values.dtype != dtype:
        raise ValueError("GloVe .npy shape or dtype failed validation")
    for row, (_, literals) in samples.items():
        start = row * dimensions
        if values[start:start + dimensions].tolist() != literals:
            raise ValueError(f"GloVe literals differ on vocabulary row {row}")
    del values

    with vocab_path.open("rb") as vocabulary:
        next(vocabulary)
        observed = 0
        for row, record in enumerate(vocabulary):
            observed += 1
            if row in samples:
                expected = str(row).encode("ascii") + b"\t" + samples[row][0] + b"\n"
                if record != expected:
                    raise ValueError(f"GloVe vocabulary key differs at row {row}")
        if observed != rows:
            raise ValueError("GloVe vocabulary row count failed validation")

    return {
        "model": "Stanford GloVe 6B 50d, Gensim word2vec-format repack",
        "scalar_count": scalar_count,
        "sources": [source_path(path)],
        "source_matrix_shape": [rows, dimensions],
        "array_shape": [scalar_count],
        "dtype": dtype.str,
        "max_source_literal_bytes": maximum_length,
        "formats": {
            "values.npy": "Flat NumPy array of fixed-width ASCII byte strings. Each element preserves one original decimal literal and has NUL padding up to 16 bytes.",
            "vocabulary.tsv": "UTF-8 TSV with a header and one zero-based row index per vocabulary token in source order.",
        },
        "lookup": {
            "scalar_index": "50 * word_row + dimension, with zero-based word_row and dimension.",
            "source_line": "word_row + 2 in the decompressed source text.",
            "element_example": "values[50 * word_row + dimension].decode('ascii')",
        },
    }


def main():
    for name, extract in (("jieba", extract_jieba), ("riva-arpa", extract_riva), ("glove", extract_glove)):
        folder = OUTPUT / name
        folder.mkdir(exist_ok=True, parents=True)
        staging = folder / f".extract-{uuid4().hex}"
        staging.mkdir()
        try:
            index = extract(staging)
            manifest = staging / "index.json"
            manifest.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            for filename in index["formats"]:
                (staging / filename).replace(folder / filename)
            manifest.replace(folder / manifest.name)
        finally:
            for file in staging.iterdir():
                file.unlink()
            staging.rmdir()
        print(f"{name}: extracted and verified {index['scalar_count']:,} stored literals", flush=True)


if __name__ == "__main__":
    main()
