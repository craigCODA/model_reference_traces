# model_reference_traces

Experimental checkpoint traces, extraction scripts, and addressable vocabulary and continuation indexes.

This repository preserves all **80 original files** from `model-reference-traces.zip`, including the SQLite vocabulary database, compiled reference data, extracted values, and bundled model assets. The original files total 1,138,987,093 bytes. Large assets use Git LFS; all original paths and file contents are preserved.

## Get the complete repository

Install Git, Git LFS, and Python 3.12 or newer. Then run:

```sh
git lfs install
git clone https://github.com/craigCODA/model_reference_traces.git
cd model_reference_traces
git lfs pull
python verify_archive.py --sqlite
```

Use `python3` instead of `python` where appropriate, or `py -3.12` on Windows. Clone with Git LFS to retrieve the large assets. A source ZIP download may contain LFS pointers instead of the asset bytes.

`verify_archive.py` checks the size, CRC32, and SHA-256 of every original file. `--sqlite` also runs the database integrity check. [ORIGINAL_ARCHIVE_MANIFEST.json](ORIGINAL_ARCHIVE_MANIFEST.json) records the complete inventory and original ZIP checksum.

## Explore the vocabulary and continuations

These commands use the included database and Python's standard library:

```sh
python vocabulary_links.py links '[[hello]]'
python vocabulary_links.py combine '[[want]]' '[[to]]' '[[go]]'
python response_links.py respond '[[hello]]'
python response_links.py respond '[[hello]]' --reply '{@[[hello]]}'
python response_links.py respond '[[hello]]' --length 2 --offset 278642537400 --limit 1
```

Start with [raw/Index.md](raw/Index.md) and [raw/Responses.md](raw/Responses.md). The database contains 2,298,793 shared vocabulary keys and 2,376,696 continuation evidence links. Combination mode addresses possible ordered sequences; continuation mode returns matches supported by stored phrases.

## Tests and checkpoint traces

Run the existing vocabulary and response tests:

```sh
python -B -m unittest test_vocabulary_links test_response_links
```

[all_results.json](all_results.json) contains the saved checkpoint observations. [raw_sets/manifest.json](raw_sets/manifest.json) indexes the extracted values for 11 model sets. The `weights/` directory contains the source assets included in the original archive.

To replay the ten checkpoint probes in `run_traces.py`, install their additional dependencies:

```sh
python -m pip install numpy pillow torch
python run_traces.py
```

Replaying traces writes result files. Rebuilding the vocabulary additionally requires the sibling `inkling-disk-vault` project and the tokenizer/dictionary inputs accepted by `vocabulary_links.py build`. Re-extracting the Inkling values also requires that sibling project and its model assets. Those external inputs were not part of the original ZIP; the included database and extracted values can be used without rebuilding them.

## Source preservation

Original archive SHA-256:

```text
97ae761d42fb5eabf49831d3c7265b5ccc42c2865eb815198dfbbaaca5a36e6b
```

The import adds this README, Git attributes/ignore rules, the inventory, and the verification script. It does not alter the 80 original files. Bundled third-party assets retain their upstream terms; the included VMAF license remains at [weights/vmaf/LICENSE](weights/vmaf/LICENSE).
