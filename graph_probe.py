"""Probe a published Cora GraphConv layer on a fixed graph and input."""

import hashlib
import json
import tarfile
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as functional


ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights" / "graphconv"
CHECKPOINT = WEIGHTS / "best_model.pt"
DATASET = WEIGHTS / "cora.tgz"


def read_graph(path):
    with tarfile.open(path, "r:gz") as archive:
        with archive.extractfile("cora/cora.content") as stream:
            content = [row.decode("ascii").split() for row in stream]
        with archive.extractfile("cora/cora.cites") as stream:
            citations = [row.decode("ascii").split() for row in stream]
    features = {row[0]: [int(value) for value in row[1:-1]] for row in content}
    neighbors = defaultdict(set)
    for left, right in citations:
        if left != right and left in features and right in features:
            neighbors[left].add(right)
            neighbors[right].add(left)
    return features, neighbors


@torch.inference_mode()
def probe():
    saved = torch.load(CHECKPOINT, weights_only=True, map_location="cpu")
    weights = saved["model_state_dict"]
    features, neighbors = read_graph(DATASET)
    target = next(node for node in features if 2 <= len(neighbors[node]) <= 5)
    incoming = sorted(neighbors[target], key=int)
    inputs = torch.tensor([features[node] for node in (target, *incoming)], dtype=torch.float32)
    hidden = functional.linear(inputs, weights["input_proj.weight"], weights["input_proj.bias"])

    neighbor_weight = weights["blocks.0.conv.lin_rel.weight"]
    neighbor_bias = weights["blocks.0.conv.lin_rel.bias"]
    root_weight = weights["blocks.0.conv.lin_root.weight"]
    root = functional.linear(hidden[0], root_weight)
    neighbor_sum = functional.linear(hidden[1:].sum(dim=0), neighbor_weight, neighbor_bias)
    output = root + neighbor_sum

    contributions = functional.linear(hidden[1:], neighbor_weight)
    decomposed = root + neighbor_bias + contributions.sum(dim=0)
    without_first = root + functional.linear(hidden[2:].sum(dim=0), neighbor_weight, neighbor_bias)
    torch.testing.assert_close(output, decomposed)
    torch.testing.assert_close(output - without_first, contributions[0], atol=1e-5, rtol=1e-5)

    return {
        "model": "Cora GraphConv",
        "family": "Graph message passing",
        "checkpoint_source": "https://huggingface.co/Anormalmaniac/cora_graphconv",
        "operator_source": "https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GraphConv.html",
        "dataset_source": "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "scope": "Standalone replay of checkpoint block 0 GraphConv with learned input projection",
        "input": {
            "node": target,
            "neighbors": incoming,
            "feature_width": inputs.shape[-1],
            "hidden_width": hidden.shape[-1],
            "graph": "Undirected Cora citation edges with self excluded from the neighbor sum",
        },
        "comparison": "Channel 0 of W_rel * sum(adjacent hidden states) + W_root * own hidden state + bias",
        "reference_contributions_channel_0": [
            {"node": node, "contribution": float(value)}
            for node, value in zip(incoming, contributions[:, 0])
        ],
        "root_contribution_channel_0": float(root[0]),
        "bias_channel_0": float(neighbor_bias[0]),
        "output_channel_0": float(output[0]),
        "edge_omission": {
            "omitted_node": incoming[0],
            "output_channel_0": float(without_first[0]),
            "change_channel_0": float((without_first - output)[0]),
        },
        "verification": {
            "channels_checked": output.numel(),
            "max_absolute_decomposition_error": float((output - decomposed).abs().max()),
            "max_absolute_edge_error": float((output - without_first - contributions[0]).abs().max()),
        },
    }


if __name__ == "__main__":
    result = probe()
    (ROOT / "graph_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
