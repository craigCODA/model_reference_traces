"""Extract raw stored values from the local OpenCV, langid, and VMAF artifacts."""

from collections import deque
import hashlib
import io
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from classic_probe import ROOT, langid_parameters, read_pinned


DESTINATION = ROOT / "raw_sets"


def publish_bytes(path, payload):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(payload)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if path.read_bytes() != payload:
        raise ValueError(f"Read-back verification failed for {path}")
    return hashlib.sha256(payload).hexdigest()


def publish_json(path, value):
    publish_bytes(path, (json.dumps(value, ensure_ascii=True, indent=2) + "\n").encode("utf-8"))


def publish_literals(directory, values):
    name = "values.txt"
    return {
        "path": name,
        "format": "text, one exact source literal per line",
        "scalar_count": len(values),
        "sha256": publish_bytes(directory / name, ("\n".join(values) + "\n").encode("ascii")),
    }


def publish_array(directory, name, array, source_field, role):
    flat = array.ravel(order="C")
    buffer = io.BytesIO()
    np.save(buffer, flat, allow_pickle=False)
    digest = publish_bytes(directory / name, buffer.getvalue())
    stored = np.load(directory / name, allow_pickle=False)
    if stored.ndim != 1 or stored.dtype != flat.dtype or stored.tobytes() != flat.tobytes():
        raise ValueError(f"Array verification failed for {name}")
    return {
        "path": name,
        "format": "npy",
        "dtype": flat.dtype.str,
        "source_field": source_field,
        "role": role,
        "scalar_count": flat.size,
        "sha256": digest,
    }


def publish_index(directory, model, checkpoint, streams, lookup, **counts):
    publish_json(directory / "lookup.json", lookup)
    index = {
        "model": model,
        "checkpoint": checkpoint,
        "scalar_count": sum(stream["scalar_count"] for stream in streams),
        **counts,
        "streams": streams,
        "lookup_mapping": {
            "path": "lookup.json",
            "index_base": 0,
            "offset_convention": "Offsets address scalar items in their indicated stream. End offsets are exclusive.",
        },
    }
    publish_json(directory / "index.json", index)
    print(f"{directory.name} {index['scalar_count']} stored scalars")


def extract_haar():
    payload, checkpoint = read_pinned(
        "weights/opencv-haar/haarcascade_frontalface_default.xml",
        "https://raw.githubusercontent.com/opencv/opencv/4.12.0/data/haarcascades/haarcascade_frontalface_default.xml",
        "0f7d4527844eb514d4a4948e822da90fbb16a34a0bbbbc6adc6498747a5aafb0",
    )
    cascade = ET.fromstring(payload).find("cascade")
    source_fields = ("height", "width", "stageParams/maxWeakCount", "featureParams/maxCatCount", "stageNum")
    values = [cascade.findtext(field).strip() for field in source_fields]
    stage_offsets = []
    for stage in cascade.findall("stages/_"):
        stumps = stage.findall("weakClassifiers/_")
        assert len(stumps) == int(stage.findtext("maxWeakCount"))
        stage_offsets.append(len(values))
        values.extend((stage.findtext("maxWeakCount").strip(), stage.findtext("stageThreshold").strip()))
        for stump in stumps:
            nodes = stump.findtext("internalNodes").split()
            leaves = stump.findtext("leafValues").split()
            assert len(nodes) == 4 and len(leaves) == 2
            values.extend(nodes)
            values.extend(leaves)
    stage_offsets.append(len(values))

    feature_offsets = []
    for feature in cascade.findall("features/_"):
        feature_offsets.append(len(values))
        for rectangle in feature.findall("rects/_"):
            fields = rectangle.text.split()
            assert len(fields) == 5
            values.extend(fields)
    feature_offsets.append(len(values))

    # Check the schema-specific offset builder against the complete leaf order.
    source_literals = [
        literal
        for node in cascade.iter()
        if len(node) == 0 and node.tag not in {"stageType", "featureType"}
        for literal in node.text.split()
    ]
    assert source_literals == values
    lookup = {
        "stream": "values.txt",
        "xml_container_path": "/opencv_storage/cascade",
        "xml_non_numeric_fields": {
            "@type_id": cascade.get("type_id"),
            "stageType": cascade.findtext("stageType"),
            "featureType": cascade.findtext("featureType"),
        },
        "model_metadata_scalar_offsets_by_xpath": dict(zip(source_fields, range(len(source_fields)))),
        "stages": {
            "xml_path": "stages/_",
            "scalar_offsets": stage_offsets,
            "header_fields": ["maxWeakCount", "stageThreshold"],
            "weak_classifier_fields_in_order": [
                "internalNodes.left_node_or_leaf", "internalNodes.right_node_or_leaf",
                "internalNodes.feature_index", "internalNodes.threshold", "leafValues.left", "leafValues.right",
            ],
            "index_rule": "The first weak classifier begins two scalars after the stage offset. Its fields and subsequent classifiers repeat in groups of six.",
        },
        "features": {
            "xml_path": "features/_/rects/_",
            "scalar_offsets": feature_offsets,
            "rectangle_fields_in_order": ["x", "y", "width", "height", "weight"],
            "index_rule": "Consecutive feature offsets delimit the rectangles belonging to that feature. Each rectangle occupies five scalars.",
        },
    }
    directory = DESTINATION / "opencv-haar"
    directory.mkdir(parents=True, exist_ok=True)
    publish_index(directory, "OpenCV 4.12.0 frontal-face Haar cascade", checkpoint,
                  [publish_literals(directory, values)], lookup)


def langid_feature_keys(transitions, emissions, feature_count):
    """Get each feature key from its shortest path in the compiled scanner."""
    keys = [None] * feature_count
    prefixes = {0: b""}
    pending = deque([0])
    byte_values = tuple(bytes([byte]) for byte in range(256))
    while pending:
        state = pending.popleft()
        prefix = prefixes[state]
        for feature_id in emissions.get(state, ()):
            key = keys[feature_id]
            if key is None:
                keys[feature_id] = prefix
            else:
                assert prefix.endswith(key)
        base = state * 256
        for byte, following in zip(byte_values, transitions[base:base + 256]):
            if following not in prefixes:
                prefixes[following] = prefix + byte
                pending.append(following)
    assert len(prefixes) == len(transitions) // 256
    assert all(key is not None and 1 <= len(key) <= 4 for key in keys)
    assert len(set(keys)) == len(keys)
    return [key.hex() for key in keys]


def extract_langid():
    payload, checkpoint = read_pinned(
        "weights/langid/langid-1.1.6.tar.gz",
        "https://files.pythonhosted.org/packages/ea/4c/0fb7d900d3b0b9c8703be316fbddffecdab23c64e1b46c7a83561d78bd43/langid-1.1.6.tar.gz",
        "044bcae1912dab85c33d8e98f2811b8f4ff1213e5e9a9e9510137b84da2cb293",
    )
    likelihoods, priors, classes, transitions, emissions = langid_parameters(payload)
    feature_keys = langid_feature_keys(transitions, emissions, len(likelihoods))
    transition_values = np.asarray(transitions)
    emission_state_ids = list(emissions)
    emission_feature_ids = []
    emission_offsets = [0]
    for feature_ids in emissions.values():
        emission_feature_ids.extend(feature_ids)
        emission_offsets.append(len(emission_feature_ids))
    directory = DESTINATION / "langid"
    directory.mkdir(parents=True, exist_ok=True)
    parameter_streams = [
        publish_array(directory, "feature_likelihoods.npy", likelihoods, "nb_ptc", "parameter"),
        publish_array(directory, "class_prior_terms.npy", priors, "nb_pc", "parameter"),
    ]
    lookup_streams = [
        publish_array(directory, "dfa_transitions.npy", transition_values, "tk_nextmove", "stored_lookup"),
        publish_array(directory, "dfa_emission_states.npy", np.asarray(emission_state_ids, dtype="uint16"),
                      "tk_output keys", "stored_lookup"),
        publish_array(directory, "dfa_emission_features.npy", np.asarray(emission_feature_ids, dtype="uint16"),
                      "tk_output tuple values", "stored_lookup"),
    ]
    assert np.load(directory / "dfa_emission_states.npy", allow_pickle=False).tolist() == emission_state_ids
    assert np.load(directory / "dfa_emission_features.npy", allow_pickle=False).tolist() == emission_feature_ids

    lookup = {
        "source_member": "langid-1.1.6/langid/langid.py",
        "stored_model_field_order": ["nb_ptc", "nb_pc", "nb_classes", "tk_nextmove", "tk_output"],
        "class_labels_by_original_index": classes,
        "byte_feature_keys_hex_by_original_index": feature_keys,
        "byte_feature_key_recovery": "Each key is the shortest byte path from DFA state 0 emitting the feature ID. These derived keys are lookup information.",
        "original_likelihood_shape_feature_then_class": list(likelihoods.shape),
        "feature_likelihoods_index_rule": "feature_index * class_count + class_index",
        "class_prior_terms_index_rule": "class_index",
        "class_prior_terms_interpretation": "The stored term is the unnormalized natural logarithm of the class document count.",
        "original_transition_shape_state_then_byte": [len(transitions) // 256, 256],
        "dfa_transitions_index_rule": "state_index * 256 + unsigned_byte_value",
        "dfa_emission_feature_offsets_by_stored_state_row": emission_offsets,
        "dfa_emission_index_rule": "Row i corresponds to item i of dfa_emission_states.npy. Consecutive offsets delimit its feature IDs in dfa_emission_features.npy in stored tuple order.",
    }
    publish_index(directory, "langid.py 1.1.6 pretrained multinomial naive Bayes", checkpoint,
                  parameter_streams + lookup_streams, lookup,
                  parameter_scalar_count=sum(stream["scalar_count"] for stream in parameter_streams),
                  stored_lookup_scalar_count=sum(stream["scalar_count"] for stream in lookup_streams))


class NumberLiteral(str):
    """Mark JSON numbers while retaining the source text."""


def append_libsvm_literals(model, values):
    header, rows = model.split("\nSV\n", maxsplit=1)
    numeric_header = {}
    text_header = {}
    for line in header.splitlines():
        name, value = line.split(maxsplit=1)
        if name in {"svm_type", "kernel_type"}:
            text_header[name] = value
        else:
            numeric_header[name] = len(values)
            values.append(value)
    vector_offsets = []
    coordinate_count = 0
    for line in rows.splitlines():
        coefficient, *coordinates = line.split()
        vector_offsets.append(len(values))
        values.append(coefficient)
        for coordinate in coordinates:
            index, value = coordinate.split(":", maxsplit=1)
            values.extend((index, value))
        coordinate_count += len(coordinates)
    assert len(vector_offsets) == int(values[numeric_header["total_sv"]])
    vector_offsets.append(len(values))
    return {
        "numeric_header_scalar_offsets_by_field": numeric_header,
        "text_header_fields": text_header,
        "support_vector_scalar_offsets": vector_offsets,
        "support_vector_count": len(vector_offsets) - 1,
        "stored_coordinate_count": coordinate_count,
        "row_layout": "Signed dual coefficient, followed by stored pairs of one-based feature index and coordinate literal. Omitted sparse coordinates have no item.",
    }


def iter_json_leaves(value, pointer=""):
    if isinstance(value, dict):
        for key, child in value.items():
            segment = key.replace("~", "~0").replace("/", "~1")
            yield from iter_json_leaves(child, f"{pointer}/{segment}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_json_leaves(child, f"{pointer}/{index}")
    else:
        yield pointer, value


def extract_vmaf():
    payload, checkpoint = read_pinned(
        "weights/vmaf/vmaf_v0.6.1.json",
        "https://raw.githubusercontent.com/Netflix/vmaf/v3.0.0/model/vmaf_v0.6.1.json",
        "5950d61fa1f861bd45d8149d80539ed9f3376cfc2495b8f0fa8e9f57cb131ee3",
    )
    model = json.loads(payload, parse_float=NumberLiteral, parse_int=NumberLiteral)
    values = []
    metadata_offsets = {}
    metadata_text = {}
    libsvm_pointer = "/model_dict/model"
    for pointer, value in iter_json_leaves(model):
        if pointer == libsvm_pointer:
            libsvm = append_libsvm_literals(value, values)
        elif isinstance(value, NumberLiteral):
            metadata_offsets[pointer] = len(values)
            values.append(value)
        else:
            metadata_text[pointer] = value

    offsets = libsvm["support_vector_scalar_offsets"]
    header, rows = model["model_dict"]["model"].split("\nSV\n", maxsplit=1)
    for name, offset in libsvm["numeric_header_scalar_offsets_by_field"].items():
        assert f"{name} {values[offset]}" in header.splitlines()
    assert [" ".join(values[start:stop]).split() for start, stop in zip(offsets, offsets[1:])] == [
        row.replace(":", " ").split() for row in rows.splitlines()
    ]
    lookup = {
        "stream": "values.txt",
        "json_scalar_offsets_by_pointer": metadata_offsets,
        "json_other_values_by_pointer": metadata_text,
        "embedded_libsvm": {"source_json_pointer": libsvm_pointer, **libsvm},
        "order_rule": "JSON scalar items follow source field order, with the LIBSVM header and support-vector rows inserted at their source field.",
    }
    directory = DESTINATION / "vmaf"
    directory.mkdir(parents=True, exist_ok=True)
    publish_index(directory, "Netflix VMAF v0.6.1 LIBSVM RBF nu-SVR", checkpoint,
                  [publish_literals(directory, values)], lookup)


if __name__ == "__main__":
    extract_haar()
    extract_langid()
    extract_vmaf()
