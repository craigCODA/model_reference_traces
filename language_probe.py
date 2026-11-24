"""Reproduce three pretrained language-model traces from local weight files."""

import ast
from collections import Counter
import gzip
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights"
JIEBA_TAG = "https://github.com/fxsjy/jieba/tree/v0.42.1/jieba/finalseg"
JIEBA_RAW = "https://raw.githubusercontent.com/fxsjy/jieba/v0.42.1/jieba/finalseg"
JIEBA_MISSING = -3.14e100
RIVA_CATALOG = "https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/speechtotext_english_lm/-/file-browser"
RIVA_RAW = (
    "https://api.ngc.nvidia.com/v2/models/nvidia/tao/speechtotext_english_lm/"
    "versions/deployable_v1.0/files/3-gram.pruned.3e-7.arpa"
)
GLOVE_RELEASE = "https://github.com/piskvorky/gensim-data/releases/tag/glove-wiki-gigaword-50"
GLOVE_RAW = (
    "https://github.com/piskvorky/gensim-data/releases/download/"
    "glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz"
)
GLOVE_MD5 = "c289bc5d7f2f02c6dc9f2f9b67641813"


def digest(path, algorithm="sha256"):
    with path.open("rb") as source:
        return hashlib.file_digest(source, algorithm).hexdigest()


def weight_record(path, source):
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "origin": source,
        "bytes": path.stat().st_size,
        "sha256": digest(path),
    }


def jieba_table(path):
    module = ast.parse(path.read_text(encoding="utf-8-sig"))
    [assignment] = [
        node for node in module.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "P"
    ]
    return ast.literal_eval(assignment.value)


def jieba_viterbi(text, start, transition, emissions):
    scores = {
        state: start[state] + emissions[state].get(text[0], JIEBA_MISSING)
        for state in start
    }
    paths = {state: state for state in start}
    for character in text[1:]:
        next_scores = {}
        next_paths = {}
        for state in start:
            emission = emissions[state].get(character, JIEBA_MISSING)
            score, previous = max(
                (scores[previous] + edges[state] + emission, previous)
                for previous, edges in transition.items() if state in edges
            )
            next_scores[state] = score
            next_paths[state] = paths[previous] + state
        scores, paths = next_scores, next_paths
    score, last = max((scores[state], state) for state in "ES")
    return score, paths[last]


def jieba_path_trace(text, tags, start, transition, emissions):
    trace = []
    running = 0.0
    for index, (character, state) in enumerate(zip(text, tags, strict=True)):
        prior = start[state] if index == 0 else transition[tags[index - 1]][state]
        emission = emissions[state].get(character, JIEBA_MISSING)
        running = running + prior + emission
        trace.append({
            "character": character,
            "tag": state,
            "start_or_transition_ln": prior,
            "emission_ln": emission,
            "cumulative_ln": running,
        })
    return trace


def jieba_trace():
    folder = WEIGHTS / "jieba-v0.42.1"
    names = ("prob_start.py", "prob_trans.py", "prob_emit.py")
    start, transition, emissions = (jieba_table(folder / name) for name in names)
    examples = []
    for text in ("南京市长江大桥", "研究生命起源", "我爱北京天安门"):
        score, tags = jieba_viterbi(text, start, transition, emissions)
        candidates = []
        for boundaries in product((False, True), repeat=len(text) - 1):
            cuts = [0, *(i for i, cut in enumerate(boundaries, start=1) if cut), len(text)]
            words = [text[left:right] for left, right in zip(cuts, cuts[1:])]
            path = "".join(
                "S" if len(word) == 1 else "B" + "M" * (len(word) - 2) + "E"
                for word in words
            )
            path_score = math.fsum([
                start[path[0]],
                *(transition[previous][current] for previous, current in zip(path, path[1:])),
                *(emissions[state].get(char, JIEBA_MISSING) for char, state in zip(text, path, strict=True)),
            ])
            candidates.append({"words": words, "tags": path, "path_ln_score": path_score})
        candidates.sort(key=lambda item: (item["path_ln_score"], item["tags"][::-1]), reverse=True)
        winner, runner_up = candidates[:2]
        difference = abs(score - winner["path_ln_score"])
        assert tags == winner["tags"] and difference < 1e-12
        trace = jieba_path_trace(text, tags, start, transition, emissions)
        assert math.isclose(trace[-1]["cumulative_ln"], score, rel_tol=0.0, abs_tol=1e-12)
        examples.append({
            "input": text,
            "viterbi_words": winner["words"],
            "viterbi_tags": tags,
            "viterbi_path_ln_score": score,
            "winning_path_terms": trace,
            "top_segmentations_by_exhaustive_score": candidates[:3],
            "lead_over_runner_up_ln": winner["path_ln_score"] - runner_up["path_ln_score"],
            "exhaustive_paths_checked": len(candidates),
            "exhaustive_winner_vs_viterbi_abs_error_ln": difference,
        })
    masses = {
        "initial": math.fsum(math.exp(logp) for logp in start.values()),
        "transition_by_state": {
            state: math.fsum(math.exp(logp) for logp in edges.values())
            for state, edges in transition.items()
        },
        "emission_by_state": {
            state: math.fsum(math.exp(logp) for logp in characters.values())
            for state, characters in emissions.items()
        },
    }
    assert all(
        abs(mass - 1.0) < 1e-12
        for mass in (masses["initial"], *masses["transition_by_state"].values())
    )
    return {
        "model": "Jieba v0.42.1 finalseg",
        "architecture": "Four-state character HMM with B/M/E/S tags, looked-up natural-log initial, transition and emission weights, and Viterbi decoding with E/S as terminal states.",
        "input_scope": "Direct finalseg HMM on contiguous Han text.",
        "scoring": "The path score uses the release tables as published. Exponentiated emission weights sum above one in all four states.",
        "sources": [JIEBA_TAG, "https://github.com/fxsjy/jieba/blob/v0.42.1/jieba/finalseg/__init__.py"],
        "weights": [weight_record(folder / name, f"{JIEBA_RAW}/{name}") for name in names],
        "emission_rows_by_state": {state: len(table) for state, table in emissions.items()},
        "weight_table_exp_sums": masses,
        "examples": examples,
    }


def read_arpa(path, histories):
    """Scan every record and retain unigrams plus rows needed by these histories."""
    declared = {}
    observed = Counter()
    tables = {}
    order = 0
    ended = False
    with path.open(encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if not line or line == "\\data\\":
                continue
            if line == "\\end\\":
                ended = True
                break
            if line.startswith("ngram "):
                left, count = line.split("=", maxsplit=1)
                declared[int(left.split()[1])] = int(count)
                continue
            section = re.fullmatch(r"\\(\d+)-grams:", line)
            if section:
                order = int(section[1])
                tables[order] = {}
                continue
            fields = line.split()
            if order == 0 or len(fields) not in (order + 1, order + 2):
                raise ValueError(f"Invalid ARPA row in section {order}: {line!r}")
            gram = tuple(fields[1:order + 1])
            if order == 1 or gram[:-1] in histories or gram in histories:
                logp = float(fields[0])
                backoff = float(fields[-1]) if len(fields) == order + 2 else 0.0
                tables[order][gram] = (logp, backoff)
            observed[order] += 1
    assert ended and declared == dict(observed) and set(tables) == {1, 2, 3}
    return tables, declared


def arpa_query(tables, history, word):
    mapped = word if (word,) in tables[1] else "<UNK>"
    history = tuple(token if (token,) in tables[1] else "<UNK>" for token in history[-2:])
    backoffs = []
    while True:
        gram = (*history, mapped)
        match = tables[len(gram)].get(gram)
        if match is not None:
            break
        history_entry = tables[len(history)].get(history)
        backoffs.append({
            "history": history,
            "log10_weight": history_entry[1] if history_entry is not None else 0.0,
            "history_exists": history_entry is not None,
        })
        history = history[1:]
    score = math.fsum([match[0], *(term["log10_weight"] for term in backoffs)])
    return {
        "token": word,
        "model_token": mapped,
        "matched_ngram": gram,
        "matched_order": len(gram),
        "matched_log10_probability": match[0],
        "backoff_terms": backoffs,
        "log10_probability": score,
    }


def riva_trace():
    path = WEIGHTS / "nvidia-riva-arpa" / "3-gram.pruned.3e-7.arpa"
    assert path.stat().st_size == 38_757_479
    prefix = "I want to"
    prefix_tokens = tuple(prefix.upper().split())
    candidates = ("go", "eat", "stay", "browse", "Alice", "Craig", "zzzxnonwordzzzx")
    sentences = (
        "Craig went to the store",
        "Alice went to the store",
        "Craig spoke with Alice",
        "Alice spoke with Craig",
    )
    streams = [("<s>", *sentence.upper().split(), "</s>") for sentence in sentences]
    normalization_contexts = ((), ("TO",), ("WANT", "TO"), ("WENT", "TO"))
    histories = {prefix_tokens[-2:], *normalization_contexts}
    for tokens in streams:
        histories.update(tokens[max(0, index - 2):index] for index in range(1, len(tokens)))
    histories.update(history[1:] for history in tuple(histories))
    tables, declared_counts = read_arpa(path, histories)
    assert {("<s>",), ("</s>",), ("<UNK>",)} <= tables[1].keys()

    next_word = [
        {"candidate": candidate, **arpa_query(tables, prefix_tokens, candidate.upper())}
        for candidate in candidates
    ]
    next_word.sort(key=lambda item: item["log10_probability"], reverse=True)
    assert {item["matched_order"] for item in next_word} == {1, 2, 3}
    assert any(item["model_token"] == "<UNK>" for item in next_word)

    sentence_results = []
    for text, tokens in zip(sentences, streams, strict=True):
        terms = [arpa_query(tables, tokens[:index], tokens[index]) for index in range(1, len(tokens))]
        sentence_results.append({
            "input": text,
            "model_tokens": tokens,
            "log10_probability_including_end_token": math.fsum(term["log10_probability"] for term in terms),
            "token_scores": terms,
        })

    normalizations = []
    for history in normalization_contexts:
        mass = math.fsum(
            10.0 ** arpa_query(tables, history, word)["log10_probability"]
            for (word,) in tables[1] if word != "<s>"
        )
        assert abs(mass - 1.0) < 1e-6
        normalizations.append({"history": history, "total_probability": mass})
    return {
        "model": "NVIDIA Riva ASR English LM deployable_v1.0 pruned trigram",
        "architecture": "Word-level backoff trigram. Query the longest stored suffix including the target word and add base-10 backoff weights for the skipped histories.",
        "training_data": "LibriSpeech, Switchboard and Fisher transcriptions, per NVIDIA's model listing.",
        "tokenization": "Whitespace splitting, uppercase model entries, <s> and </s> sentence markers, and <UNK> for words outside the vocabulary.",
        "sources": [RIVA_CATALOG, "https://kheafield.com/papers/avenue/kenlm.pdf"],
        "weights": [weight_record(path, RIVA_RAW)],
        "ngram_rows_by_order": declared_counts,
        "next_word_prefix": prefix,
        "next_word_effective_history": prefix_tokens[-2:],
        "next_word_candidates": next_word,
        "sentence_examples": sentence_results,
        "validation_probability_mass_excluding_start_token": normalizations,
    }


def read_glove(path, words):
    vectors = {}
    rows = 0
    with gzip.open(path, "rt", encoding="utf-8") as source:
        records, dimensions = map(int, next(source).split())
        for line in source:
            word, *values = line.split()
            if len(values) != dimensions:
                raise ValueError(f"Wrong embedding dimensionality for {word!r}")
            if word in words:
                vectors[word] = tuple(map(float, values))
            rows += 1
    assert (records, dimensions, rows) == (400_000, 50, 400_000)
    assert vectors.keys() == words and all(math.isfinite(v) for vector in vectors.values() for v in vector)
    return vectors


def glove_trace():
    path = WEIGHTS / "glove-wiki-gigaword-50" / "glove-wiki-gigaword-50.gz"
    archive_md5 = digest(path, "md5")
    assert path.stat().st_size == 69_182_535 and archive_md5 == GLOVE_MD5
    pairs = (
        ("Craig", ("Craig", "Alice", "Bob", "Carol", "Robert", "London", "dog")),
        ("Alice", ("Alice", "Craig", "Bob", "Carol", "Robert", "London", "dog")),
        ("Craig met Alice", ("Alice met Craig", "Craig greeted Alice", "Bob met Carol", "Alice and Bob talked", "The dog chased the cat")),
    )
    texts = {text for query, references in pairs for text in (query, *references)}
    words = {word for text in texts for word in text.lower().split()}
    vectors = read_glove(path, words)
    representations = {}
    for text in texts:
        tokens = text.lower().split()
        representations[text] = tuple(
            math.fsum(dimension) / len(tokens) for dimension in zip(*(vectors[token] for token in tokens), strict=True)
        )

    examples = []
    largest_error = 0.0
    for query, references in pairs:
        query_vector = representations[query]
        query_norm = math.hypot(*query_vector)
        scores = []
        for reference in references:
            reference_vector = representations[reference]
            reference_norm = math.hypot(*reference_vector)
            dot = math.fsum(a * b for a, b in zip(query_vector, reference_vector, strict=True))
            cosine = dot / (query_norm * reference_norm)
            normalized_distance_squared = math.fsum(
                (a / query_norm - b / reference_norm) ** 2
                for a, b in zip(query_vector, reference_vector, strict=True)
            )
            error = abs(cosine - (1.0 - normalized_distance_squared / 2.0))
            largest_error = max(largest_error, error)
            scores.append({
                "reference": reference,
                "cosine": cosine,
                "euclidean_distance": math.dist(query_vector, reference_vector),
                "dot_product": dot,
                "reference_vector_norm": reference_norm,
            })
        scores.sort(key=lambda item: item["cosine"], reverse=True)
        examples.append({"query": query, "query_vector_norm": query_norm, "candidate_scores": scores})

    assert largest_error < 1e-12
    assert representations["Craig met Alice"] == representations["Alice met Craig"]
    return {
        "model": "Stanford GloVe 6B 50d, Gensim word2vec-format repack",
        "architecture": "Learned static 50-dimensional word representations from weighted log-bilinear regression on global co-occurrence counts. Inference is vector lookup. The phrase probe uses a plain coordinate mean of the looked-up word vectors followed by cosine or Euclidean scoring.",
        "training_data": "Uncased Wikipedia 2014 and Gigaword 5, approximately six billion tokens.",
        "sources": [
            "https://nlp.stanford.edu/projects/glove/",
            "https://nlp.stanford.edu/pubs/glove.pdf",
            GLOVE_RELEASE,
            "https://github.com/piskvorky/gensim-data/blob/master/list.json",
        ],
        "weights": [{**weight_record(path, GLOVE_RAW), "md5": archive_md5}],
        "dimensions": 50,
        "vocabulary_size": 400_000,
        "required_rows_found": sorted(words),
        "examples_ranked_by_cosine": examples,
        "validation_cosine_vs_unit_distance_identity_max_abs_error": largest_error,
        "validation_phrase_reversal_vectors_equal": True,
    }


def main():
    results = {"jieba": jieba_trace(), "riva_arpa": riva_trace(), "glove": glove_trace()}
    output = ROOT / "language_results.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Validated {len(results)} pretrained traces and wrote {output.name}.")


if __name__ == "__main__":
    main()
