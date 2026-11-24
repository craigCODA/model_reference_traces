"""Reproduce fixed-window Haar, language NB, and RBF-SVR traces offline.

Run with Python 3.10 or later, NumPy, and Pillow 9.1+. The weight and probe paths
are relative to this script. Each downloaded artifact is checked by SHA-256.
"""

import array
import ast
import base64
import bz2
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import tarfile
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
OPENCV_REVISION = "https://github.com/opencv/opencv/blob/4.12.0"
VMAF_REVISION = "https://github.com/Netflix/vmaf/blob/v3.0.0"
LANGID_RELEASE = "https://pypi.org/project/langid/1.1.6/"
ASTRONAUT_PATH = "weights/opencv-haar/astronaut.png"
VMAF_REFERENCE_TOLERANCE = 0.0002


def read_pinned(path, source_url, expected_hash):
    payload = (ROOT / path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_hash:
        raise ValueError(f"SHA-256 mismatch for {path}: {digest}")
    return payload, {
        "path": path,
        "source_url": source_url,
        "sha256": digest,
        "size_bytes": len(payload),
    }


def haar_parameters(model):
    cascade = ET.fromstring(model).find("cascade")
    assert cascade.findtext("stageType") == "BOOST"
    assert cascade.findtext("featureType") == "HAAR"
    assert cascade.findtext("height") == cascade.findtext("width") == "24"
    features = []
    for feature in cascade.findall("features/_"):
        assert feature.findtext("tilted", "0") == "0"
        rects = []
        for rect in feature.findall("rects/_"):
            x, y, width, height, weight = rect.text.split()
            rects.append((int(x), int(y), int(width), int(height), np.float32(weight)))
        features.append(rects)

    stages = []
    for stage in cascade.findall("stages/_"):
        stumps = []
        for stump in stage.findall("weakClassifiers/_"):
            left, right, feature_index, threshold = stump.findtext("internalNodes").split()
            assert (left, right) == ("0", "-1")
            leaves = tuple(map(np.float32, stump.findtext("leafValues").split()))
            stumps.append((int(feature_index), np.float32(threshold), leaves))
        stages.append((np.float32(stage.findtext("stageThreshold")), stumps))
    return features, stages


def haar_stage_trace(pixels, features, stages):
    # OpenCV 4.12.0 HaarEvaluator::setWindow normalizes on the inner 22x22.
    inner = pixels[1:-1, 1:-1].astype(np.int64)
    area = inner.size
    total = int(inner.sum())
    square_sum = int((inner * inner).sum())
    variance_term = area * square_sum - total * total
    factor = np.float32(1.0 / math.sqrt(variance_term)) if variance_term > 0 else np.float32(1.0)
    usable = variance_term > 0 and area * float(factor) < 0.1
    normalization = {
        "window_xywh": [1, 1, 22, 22],
        "pixel_sum": total,
        "squared_pixel_sum": square_sum,
        "population_standard_deviation": math.sqrt(variance_term) / area,
        "reciprocal_of_area_times_standard_deviation": float(factor),
        "area_times_reciprocal": area * float(factor),
        "area_times_reciprocal_cutoff_strict_upper_bound": 0.1,
        "passed_variance_gate": usable,
    }
    if not usable:
        return normalization, [], [], False

    integral = np.pad(pixels.cumsum(axis=0, dtype=np.int64).cumsum(axis=1), ((1, 0), (1, 0)))
    stage_traces = []
    first_stage_stumps = []
    for stage_index, (threshold, stumps) in enumerate(stages):
        score = 0.0
        for feature_index, weak_threshold, leaves in stumps:
            weighted_sum = np.float32(0.0)
            rect_traces = []
            for x, y, width, height, weight in features[feature_index]:
                x2, y2 = x + width, y + height
                rect_sum = int(integral[y2, x2] - integral[y, x2] - integral[y2, x] + integral[y, x])
                assert rect_sum == pixels[y:y2, x:x2].sum(dtype=np.int64)
                weighted = weight * np.float32(rect_sum)
                weighted_sum += weighted
                if stage_index == 0:
                    rect_traces.append({
                        "rectangle_xywh": [x, y, width, height],
                        "checkpoint_rectangle_weight": float(weight),
                        "pixel_sum": rect_sum,
                        "weighted_pixel_sum": float(weighted),
                    })
            normalized_response = weighted_sum * factor
            leaf_index = 0 if normalized_response < weak_threshold else 1
            leaf_score = float(leaves[leaf_index])
            score += leaf_score
            if stage_index == 0:
                first_stage_stumps.append({
                    "feature_index_zero_based": feature_index,
                    "rectangles": rect_traces,
                    "weighted_pixel_sum": float(weighted_sum),
                    "normalized_response": float(normalized_response),
                    "learned_threshold": float(weak_threshold),
                    "learned_leaf_values": list(map(float, leaves)),
                    "selected_leaf": "left" if leaf_index == 0 else "right",
                    "selected_leaf_score": leaf_score,
                })
        cutoff = float(threshold - np.float32(1e-5))
        accepted = score >= cutoff
        stage_traces.append({
            "stage_index_zero_based": stage_index,
            "weak_classifier_count": len(stumps),
            "score": score,
            "learned_threshold": float(threshold),
            "opencv_effective_cutoff": cutoff,
            "margin": score - cutoff,
            "passed": accepted,
        })
        if not accepted:
            break
    return normalization, stage_traces, first_stage_stumps, all(row["passed"] for row in stage_traces)


def trace_haar():
    model, checkpoint = read_pinned(
        "weights/opencv-haar/haarcascade_frontalface_default.xml",
        "https://raw.githubusercontent.com/opencv/opencv/4.12.0/data/haarcascades/haarcascade_frontalface_default.xml",
        "0f7d4527844eb514d4a4948e822da90fbb16a34a0bbbbc6adc6498747a5aafb0",
    )
    image_bytes, image_source = read_pinned(
        ASTRONAUT_PATH,
        "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/astronaut.png",
        "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5",
    )
    features, stages = haar_parameters(model)
    with Image.open(io.BytesIO(image_bytes)) as image:
        grayscale = image.convert("L")
    probes = {}
    for name, box in {
        "face_crop": (176, 63, 272, 159),
        "background_crop": (32, 32, 128, 128),
    }.items():
        pixels = np.asarray(grayscale.crop(box).resize((24, 24), Image.Resampling.BILINEAR))
        normalization, stage_traces, first_stumps, accepted = haar_stage_trace(pixels, features, stages)
        assert not first_stumps or math.isclose(
            math.fsum(row["selected_leaf_score"] for row in first_stumps), stage_traces[0]["score"], abs_tol=1e-12
        )
        probes[name] = {
            "crop_left_top_right_bottom_exclusive": list(box),
            "input_pixel_bytes_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
            "normalization": normalization,
            "stages": stage_traces,
            "first_stage_stumps": first_stumps,
            "window_accepted": accepted,
        }
    assert probes["face_crop"]["window_accepted"] and len(probes["face_crop"]["stages"]) == len(stages) == 25
    assert not probes["background_crop"]["window_accepted"] and len(probes["background_crop"]["stages"]) == 2
    return {
        "model": "OpenCV 4.12.0 frontal-face Haar cascade (haarcascade_frontalface_default.xml)",
        "checkpoint": checkpoint,
        "probe_image": image_source,
        "probe_image_documentation": "https://scikit-image.org/docs/0.25.x/api/skimage.data.html#skimage.data.astronaut",
        "evaluator_sources": [
            f"{OPENCV_REVISION}/modules/objdetect/src/cascadedetect.cpp",
            f"{OPENCV_REVISION}/modules/objdetect/src/cascadedetect.hpp",
        ],
        "scope": "Fixed windows from a real photograph. Pillow RGB-to-L and bilinear resizing to 24x24 precede the cascade evaluation. The trace covers the OpenCV variance gate and evaluated stages for each window.",
        "window_size": [24, 24],
        "stage_count": len(stages),
        "weak_classifier_count": sum(len(stumps) for _, stumps in stages),
        "stage_cutoff_rule": "Reject if sum of selected leaf scores is below the float32 stored stage threshold minus 1e-5, as in OpenCV 4.12.0.",
        "probes": probes,
    }


class LangidWeightsReader(pickle.Unpickler):
    """Decode the hash-pinned release's legacy arrays without importing it."""

    def find_class(self, module, name):
        if (module, name) != ("array", "array"):
            raise pickle.UnpicklingError(f"unexpected model global {module}.{name}")
        return array.array

    def persistent_load(self, identifier):
        raise pickle.UnpicklingError("unexpected model persistent object")


def langid_parameters(release_bytes):
    with tarfile.open(fileobj=io.BytesIO(release_bytes), mode="r:gz") as release:
        with release.extractfile("langid-1.1.6/langid/langid.py") as source:
            tree = ast.parse(source.read())
    model_assignment = next(
        statement for statement in tree.body
        if isinstance(statement, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "model" for target in statement.targets)
    )
    compressed_model = base64.b64decode(ast.literal_eval(model_assignment.value))
    weights = bz2.decompress(compressed_model)
    likelihoods, priors, classes, transitions, emissions = LangidWeightsReader(io.BytesIO(weights)).load()
    return np.array(likelihoods).reshape(-1, len(classes)), np.array(priors), classes, transitions, emissions


def byte_feature_counts(data, transitions, emissions):
    counts = Counter()
    state = 0
    for byte in data:
        state = transitions[state * 256 + byte]
        counts.update(emissions.get(state, ()))
    return counts


def active_byte_ngrams(data, transitions, emissions):
    # A production may be emitted by many states through failure links.
    # Recover each active feature's shortest triggering byte suffix.
    ngrams = {}
    for start in range(len(data)):
        state = 0
        for end in range(start, min(len(data), start + 4)):
            state = transitions[state * 256 + data[end]]
            substring = data[start:end + 1]
            for feature_id in emissions.get(state, ()):
                known = ngrams.get(feature_id)
                if known is None or len(substring) < len(known):
                    ngrams[feature_id] = substring
    return ngrams


def language_probe(text, likelihoods, priors, classes, transitions, emissions):
    data = text.encode("utf-8")
    counts = byte_feature_counts(data, transitions, emissions)
    ngrams = active_byte_ngrams(data, transitions, emissions)
    assert set(counts) == set(ngrams)
    for feature_id, count in counts.items():
        assert count == sum(data.startswith(ngrams[feature_id], start) for start in range(len(data)))

    feature_counts = np.zeros(len(likelihoods), dtype=np.uint32)
    for feature_id, count in counts.items():
        feature_counts[feature_id] = count
    weighted_scores = feature_counts @ likelihoods
    raw_scores = priors + weighted_scores
    for class_index, score in enumerate(raw_scores):
        check = float(priors[class_index]) + math.fsum(
            count * float(likelihoods[feature_id, class_index]) for feature_id, count in counts.items()
        )
        assert math.isclose(score, check, rel_tol=1e-12)
    ranking = sorted(range(len(classes)), key=lambda index: -raw_scores[index])
    winner, runner_up = ranking[:2]
    ratios = np.exp(raw_scores - raw_scores[winner])
    probabilities = ratios / ratios.sum()
    prior_weights = np.exp(priors.astype(np.float64))
    prior_probabilities = prior_weights / prior_weights.sum()
    comparison_order = [
        index for rank, index in enumerate(ranking)
        if rank < 5 or classes[index] in {"en", "fr", "es", "de"}
    ]
    candidate_scores = [
        {
            "language": classes[index],
            "class_index_zero_based": index,
            "rank_among_all_languages": ranking.index(index) + 1,
            "stored_log_class_prior_term": float(priors[index]),
            "normalized_class_prior_probability": float(prior_probabilities[index]),
            "sum_of_count_times_log_likelihood": float(weighted_scores[index]),
            "raw_log_score": float(raw_scores[index]),
            "normalized_probability_across_all_languages": float(probabilities[index]),
        }
        for index in comparison_order
    ]

    def evidence_gap(feature_id):
        row = likelihoods[feature_id]
        return counts[feature_id] * (float(row[winner]) - float(row[runner_up]))

    feature_details = []
    for feature_id in sorted(counts, key=lambda index: (-abs(evidence_gap(index)), index))[:8]:
        count = counts[feature_id]
        candidates = {}
        for index in comparison_order:
            log_likelihood = float(likelihoods[feature_id, index])
            candidates[classes[index]] = {
                "learned_log_likelihood": log_likelihood,
                "contribution": count * log_likelihood,
            }
        feature_details.append({
            "feature_id_zero_based": feature_id,
            "byte_ngram_hex": ngrams[feature_id].hex(" "),
            "byte_ngram_display": ngrams[feature_id].decode("utf-8", errors="backslashreplace"),
            "count": count,
            "winner_minus_runner_up_contribution": evidence_gap(feature_id),
            "candidate_contributions": candidates,
        })

    return {
        "text": text,
        "utf8_hex": data.hex(" "),
        "active_feature_counts": [[index, counts[index]] for index in sorted(counts)],
        "total_feature_hits": sum(counts.values()),
        "predicted_language": classes[winner],
        "runner_up": classes[runner_up],
        "winner_minus_runner_up_raw_log_score": float(raw_scores[winner] - raw_scores[runner_up]),
        "candidate_scores": candidate_scores,
        "top_features_by_absolute_winner_runner_up_contribution_gap": feature_details,
    }


def trace_langid():
    release, checkpoint = read_pinned(
        "weights/langid/langid-1.1.6.tar.gz",
        "https://files.pythonhosted.org/packages/ea/4c/0fb7d900d3b0b9c8703be316fbddffecdab23c64e1b46c7a83561d78bd43/langid-1.1.6.tar.gz",
        "044bcae1912dab85c33d8e98f2811b8f4ff1213e5e9a9e9510137b84da2cb293",
    )
    likelihoods, priors, classes, transitions, emissions = langid_parameters(release)
    assert likelihoods.shape[1] == len(priors) == len(classes) == 97
    probes = {
        "english_upstream_example": language_probe("This is a test", likelihoods, priors, classes, transitions, emissions),
        "french_utf8_example": language_probe("C'est déjà l'été à Montréal.", likelihoods, priors, classes, transitions, emissions),
    }
    first = probes["english_upstream_example"]
    assert first["predicted_language"] == "en"
    assert math.isclose(first["candidate_scores"][0]["raw_log_score"], -54.41310358047485, rel_tol=0.0, abs_tol=1e-12)
    probability = first["candidate_scores"][0]["normalized_probability_across_all_languages"]
    assert math.isclose(probability, 0.9999999909903544, rel_tol=0.0, abs_tol=1e-12)
    assert probes["french_utf8_example"]["predicted_language"] == "fr"
    return {
        "model": "langid.py 1.1.6 pretrained multinomial naive Bayes",
        "checkpoint": checkpoint,
        "model_release": LANGID_RELEASE,
        "method_sources": ["https://github.com/saffsd/langid.py", "https://aclanthology.org/P12-3005/"],
        "scope": "UTF-8 byte counts from the checkpoint's DFA, then log evidence, class priors, and argmax across all 97 classes. The listed probabilities use optional normalization across the full class set.",
        "class_count": len(classes),
        "selected_byte_ngram_count": len(likelihoods),
        "dfa_state_count": len(transitions) // 256,
        "class_prior_parameterization": "The stored prior term is the natural logarithm of the class document count. The training implementation in langid/train/NBtrain.py leaves the common prior normalizer out of the stored term.",
        "decision_rule": "Choose the highest stored class prior term plus the sum of active byte-feature counts times learned log likelihoods.",
        "upstream_english_example_raw_log_score": -54.41310358047485,
        "upstream_english_example_normalized_probability": 0.9999999909903544,
        "probes": probes,
    }


def read_libsvm_rbf_regressor(serialized, dimension):
    header, body = serialized.split("\nSV\n", maxsplit=1)
    fields = dict(line.split(maxsplit=1) for line in header.splitlines())
    assert fields["svm_type"] == "nu_svr" and fields["kernel_type"] == "rbf"
    rows = [line.split() for line in body.splitlines() if line.strip()]
    assert len(rows) == int(fields["total_sv"])
    vectors = np.zeros((len(rows), dimension), dtype=np.float64)
    coefficients = []
    for row_index, (coefficient, *components) in enumerate(rows):
        coefficients.append(float(coefficient))
        for component in components:
            feature_index, value = component.split(":")
            vectors[row_index, int(feature_index) - 1] = float(value)
    return float(fields["gamma"]), float(fields["rho"]), np.array(coefficients), vectors


def vmaf_probe(raw_features, expected, slope, intercept, gamma, rho, coefficients, vectors, bounds):
    normalized_features = np.array(raw_features) * slope[1:] + intercept[1:]
    distances = np.sum(np.square(vectors - normalized_features), axis=1)
    kernel_values = np.exp(-gamma * distances)
    terms = kernel_values * coefficients
    total = math.fsum(terms)
    svr_output = total - rho
    denormalized = (svr_output - intercept[0]) / slope[0]
    score = min(bounds[1], max(bounds[0], denormalized))
    assert math.isclose(score, expected, rel_tol=0.0, abs_tol=VMAF_REFERENCE_TOLERANCE)

    # Include some concrete terms while aggregating every support vector.
    sample_indices = list(dict.fromkeys([0, 1, 2, int(np.argmin(distances)), int(np.argmax(terms)), int(np.argmin(terms))]))
    sample_terms = []
    for index in sample_indices:
        sample_terms.append({
            "support_vector_index_zero_based": index,
            "learned_support_vector": vectors[index].tolist(),
            "learned_signed_dual_coefficient": float(coefficients[index]),
            "squared_distance": float(distances[index]),
            "rbf_kernel_value": float(kernel_values[index]),
            "coefficient_times_kernel": float(terms[index]),
        })
    return {
        "input_features_in_model_order": list(raw_features),
        "normalized_input_features": normalized_features.tolist(),
        "support_vector_terms_evaluated": len(terms),
        "example_support_vector_terms": sample_terms,
        "positive_term_sum": math.fsum(terms[terms > 0]),
        "negative_term_sum": math.fsum(terms[terms < 0]),
        "all_support_vector_terms_sum": total,
        "normalized_svr_output_after_subtracting_rho": svr_output,
        "denormalized_score": denormalized,
        "score_after_clipping": score,
        "published_reference_score": expected,
        "absolute_difference_from_reference": abs(score - expected),
    }


def trace_vmaf():
    artifact, checkpoint = read_pinned(
        "weights/vmaf/vmaf_v0.6.1.json",
        "https://raw.githubusercontent.com/Netflix/vmaf/v3.0.0/model/vmaf_v0.6.1.json",
        "5950d61fa1f861bd45d8149d80539ed9f3376cfc2495b8f0fa8e9f57cb131ee3",
    )
    artifact_data = json.loads(artifact)
    model = artifact_data["model_dict"]
    params = artifact_data["param_dict"]
    names = model["feature_names"]
    assert model["model_type"] == "LIBSVMNUSVR" and model["norm_type"] == "linear_rescale"
    gamma, rho, coefficients, vectors = read_libsvm_rbf_regressor(model["model"], len(names))
    slope = np.array(model["slopes"])
    intercept = np.array(model["intercepts"])
    bounds = model["score_clip"]
    common = (slope, intercept, gamma, rho, coefficients, vectors, bounds)
    distorted = vmaf_probe((0.962084, 0.0, 0.505712, 0.879061, 0.937873, 0.964301), 83.856285, *common)
    distorted["probe_source_url"] = f"{VMAF_REVISION}/libvmaf/tools/README.md#example"
    distorted["source_description"] = "Frame 0 from the published libvmaf CLI comparison of src01_hrc00_576x324.yuv and src01_hrc01_576x324.yuv. The feature inputs are printed to six decimal places."
    identical = vmaf_probe((1.000002, 0.0, 1.0, 1.0, 0.999999, 0.999999), 97.428478, *common)
    identical["probe_source_url"] = "https://github.com/Netflix/vmaf/issues/1431"
    identical["source_description"] = "Frame 0 of src01_hrc00_576x324.yuv compared with itself in the issue's libvmaf log. Motion is zero and the other features are close to one."
    moving = vmaf_probe((1.000002, 4.215680, 1.0, 1.0, 0.999999, 0.999999), 100.0, *common)
    moving["probe_source_url"] = identical["probe_source_url"]
    moving["source_description"] = "Frame 1 of the same identical-clips log, with positive motion. The score activates the upper clipping bound."
    assert moving["denormalized_score"] > bounds[1]
    return {
        "model": "Netflix VMAF v0.6.1 LIBSVM RBF nu-support vector regression",
        "checkpoint": checkpoint,
        "method_sources": [
            f"{VMAF_REVISION}/resource/doc/models.md",
            f"{VMAF_REVISION}/libvmaf/src/predict.c",
            "https://github.com/cjlin1/libsvm/blob/v337/svm.cpp",
        ],
        "scope": "Full support-vector regression over six published per-frame video features, including the default HDTV score clipping.",
        "input_feature_decimal_places_in_source": 6,
        "absolute_score_tolerance_for_rounded_inputs": VMAF_REFERENCE_TOLERANCE,
        "ordered_feature_names": names,
        "normalization_slopes_output_then_features": slope.tolist(),
        "normalization_intercepts_output_then_features": intercept.tolist(),
        "score_clip_bounds": bounds,
        "gamma": gamma,
        "training_parameters": {"C": params["C"], "nu": params["nu"]},
        "learned_rho_subtracted_from_kernel_sum": rho,
        "support_vector_count": len(coefficients),
        "coefficient_range": [float(coefficients.min()), float(coefficients.max())],
        "identical_minus_distorted_score": identical["score_after_clipping"] - distorted["score_after_clipping"],
        "probes": {
            "published_distorted_frame_0": distorted,
            "identical_reference_frame_0": identical,
            "identical_reference_frame_1": moving,
        },
    }


def main():
    results = {"opencv_haar": trace_haar(), "langid": trace_langid(), "vmaf_svr": trace_vmaf()}
    destination = ROOT / "classic_results.json"
    payload = json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(dir=ROOT, delete=False, mode="w", encoding="utf-8", newline="\n") as stream:
        temporary_path = Path(stream.name)
        try:
            stream.write(payload)
            stream.close()
            temporary_path.replace(destination)
        finally:
            temporary_path.unlink(missing_ok=True)
    print(f"Wrote {destination.name}")


if __name__ == "__main__":
    main()
