"""Export the selected GraphConv and Inkling weight values without inference."""

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
INKLING = ROOT.parent / "inkling-disk-vault"
OUTPUT = ROOT / "raw_sets"
GRAPH_KEYS = {
    "input_proj.weight",
    "input_proj.bias",
    "blocks.0.conv.lin_rel.weight",
    "blocks.0.conv.lin_rel.bias",
    "blocks.0.conv.lin_root.weight",
}


def write_index(folder, index):
    pending = folder / "index.json.pending"
    pending.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    pending.replace(folder / "index.json")


def extract_graph():
    checkpoint = ROOT / "weights" / "graphconv" / "best_model.pt"
    folder = OUTPUT / "graphconv"
    folder.mkdir(parents=True, exist_ok=True)
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")["model_state_dict"]
    if not GRAPH_KEYS.issubset(state):
        raise ValueError("The GraphConv checkpoint is missing a traced parameter")

    arrays = []
    components = []
    position = 0
    for key, tensor in state.items():
        if key not in GRAPH_KEYS:
            continue
        array = tensor.numpy()
        if array.dtype != np.dtype("<f4"):
            raise ValueError(f"Unexpected dtype {array.dtype} for {key}")
        arrays.append(array.ravel())
        components.append({
            "original_key": key,
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "scalar_start": position,
            "scalar_stop_exclusive": position + array.size,
        })
        position += array.size

    path = folder / "values.npy"
    np.save(path, np.concatenate(arrays), allow_pickle=False)
    exported = np.load(path, mmap_mode="r", allow_pickle=False)
    for array, component in zip(arrays, components):
        actual = exported[component["scalar_start"]:component["scalar_stop_exclusive"]]
        if actual.tobytes() != array.tobytes():
            raise ValueError(f"Raw GraphConv export differs at {component['original_key']}")
    write_index(folder, {
        "model": "Cora GraphConv",
        "scope": "The five parameters of the input projection and first GraphConv used in the trace.",
        "checkpoint": {
            "path": checkpoint.relative_to(ROOT).as_posix(),
            "source": "https://huggingface.co/Anormalmaniac/cora_graphconv",
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
        "scalar_count": position,
        "array": {"path": path.name, "format": "npy", "dtype": "<f4", "shape": [position]},
        "components": components,
    })
    return position


def inkling_entries(graph, parameters, token_ids):
    embedding = graph["nodes"][0]["weights"]
    layer = graph["nodes"][1]
    if layer["attrs"]["layer"] != 0:
        raise ValueError("The first layer of Inkling's saved graph changed")
    names = [embedding["norm"]]
    names.extend(name for name in layer["weights"].values() if name.startswith("blk.0.attn_"))
    entries = [(name, 0, math.prod(parameters[name]["shape"][:-1])) for name in names]
    entries.extend((embedding["table"], token_id, 1) for token_id in sorted(set(token_ids)))
    return sorted(entries, key=lambda entry: (parameters[entry[0]]["offset"], entry[1]))


def extract_inkling():
    sys.path.insert(0, str(INKLING))
    from inkling_store import decode_rows
    from inkling_vault import load

    checkpoint, _, graph, parameters, tokenizer = load()
    prompt = " Craig told Alice. Craig"
    token_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    entries = inkling_entries(graph, parameters, token_ids)
    total = sum(rows * parameters[key]["shape"][-1] for key, _, rows in entries)
    folder = OUTPUT / "inkling-2-mini"
    folder.mkdir(parents=True, exist_ok=True)
    array_path = folder / "values.npy"
    packed_path = folder / "stored_bytes.bin"
    values = np.lib.format.open_memmap(array_path, mode="w+", dtype="<f4", shape=(total,))
    components = []
    scalar_position = 0
    byte_position = 0
    with checkpoint.open("rb") as source, packed_path.open("wb") as packed:
        for key, first_row, rows in entries:
            tensor = parameters[key]
            shape = tensor["shape"]
            columns = shape[-1]
            stride = tensor["size"] // math.prod(shape[:-1])
            source_offset = tensor["offset"] + first_row * stride
            byte_count = rows * stride
            component = {
                "original_key": key,
                "original_shape": shape,
                "source_flat_scalar_start": first_row * columns,
                "gguf_tensor_type": tensor["encoding"],
                "source_byte_start": source_offset,
                "source_byte_stop_exclusive": source_offset + byte_count,
                "packed_byte_start": byte_position,
                "packed_byte_stop_exclusive": byte_position + byte_count,
                "scalar_start": scalar_position,
                "scalar_stop_exclusive": scalar_position + rows * columns,
                "row_count": rows,
            }
            components.append(component)
            source.seek(source_offset)
            for first in range(0, rows, 128):
                count = min(128, rows - first)
                encoded = source.read(count * stride)
                if len(encoded) != count * stride:
                    raise ValueError(f"Short source read for {key}")
                packed.write(encoded)
                decoded = decode_rows(encoded, tensor["encoding"], columns).ravel()
                values[scalar_position:scalar_position + decoded.size] = decoded
                scalar_position += decoded.size
            byte_position += byte_count

    values.flush()
    del values
    values = np.load(array_path, mmap_mode="r", allow_pickle=False)
    with checkpoint.open("rb") as source, packed_path.open("rb") as packed:
        for component in components:
            source.seek(component["source_byte_start"])
            packed.seek(component["packed_byte_start"])
            rows = component["row_count"]
            width = component["original_shape"][-1]
            stride = (component["source_byte_stop_exclusive"] - component["source_byte_start"]) // rows
            for first in range(0, rows, 128):
                count = min(128, rows - first)
                encoded = source.read(count * stride)
                position = component["scalar_start"] + first * width
                if packed.read(count * stride) != encoded:
                    raise ValueError("An exported packed GGUF byte range differs from the source")
                decoded = decode_rows(encoded, component["gguf_tensor_type"], width)
                if values[position:position + count * width].tobytes() != decoded.tobytes():
                    raise ValueError("An exported Inkling coefficient differs from the decoded source")
    if scalar_position != total or packed_path.stat().st_size != byte_position:
        raise ValueError("The Inkling export lengths do not match its index")

    write_index(folder, {
        "model": "Inkling 2 Mini IQ4_NL GGUF",
        "scope": "Layer 0 attention parameters, the embedding normalizer, and the token embedding rows for the observed prompt.",
        "checkpoint": {
            "path": checkpoint.relative_to(ROOT.parent).as_posix(),
            "path_base": "research",
            "byte_size": checkpoint.stat().st_size,
        },
        "token_ids": token_ids,
        "scalar_count": total,
        "array": {"path": array_path.name, "format": "npy", "dtype": "<f4", "shape": [total]},
        "stored_bytes": {"path": packed_path.name, "size_bytes": byte_position},
        "components": components,
    })
    return total


if __name__ == "__main__":
    print(f"GraphConv extracted and verified {extract_graph():,} scalar values.")
    print(f"Inkling extracted and verified {extract_inkling():,} scalar values.")
