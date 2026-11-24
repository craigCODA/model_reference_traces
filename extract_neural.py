"""Extract the stored scalars from the three local neural checkpoint sets.

Run with Python, NumPy, and CPU PyTorch: python -B extract_neural.py
Use --check to compare existing arrays and indexes with the checkpoint bytes.
"""

import argparse
import hashlib
import json

import numpy as np
import torch

from neural_probe import (
    ENCODEC_FILES,
    ENCODEC_URL,
    MAMBA_FILES,
    MAMBA_URL,
    ROOT,
    SQUEEZE_FILES,
    SQUEEZE_URL,
    check_cache,
    read_safetensors_ranges,
)


MAMBA_PREFIX = 'backbone.layers.0.mixer.'
MAMBA_TENSORS = {
    MAMBA_PREFIX + 'A_log': 'A_log_D.bin',
    MAMBA_PREFIX + 'D': 'A_log_D.bin',
    MAMBA_PREFIX + 'dt_proj.bias': 'dt_proj.bin',
    MAMBA_PREFIX + 'dt_proj.weight': 'dt_proj.bin',
    MAMBA_PREFIX + 'x_proj.weight': 'x_proj.bin',
}
ENCODEC_TENSORS = {
    'quantizer.layers.0.codebook.embed': 'codebook_0.bin',
    'quantizer.layers.1.codebook.embed': 'codebook_1.bin',
}


def safetensors_components(model_id, url, cache_files, tensor_files):
    caches = check_cache(model_id, url, cache_files, fetch=False)
    tensors, locations = read_safetensors_ranges(caches, tensor_files)
    for key, tensor in tensors.items():
        location = locations[key]
        cache = caches[location['cache_file']]
        first, last = location['checkpoint_byte_range_inclusive']
        cache_start = cache['checkpoint_byte_range_inclusive'][0]
        yield key, tensor, {
            'storage_dtype': location['dtype'],
            'source_path': cache['path'],
            'source_sha256': cache['sha256'],
            'source_byte_range_inclusive': [first - cache_start, last - cache_start],
            'checkpoint_byte_range_inclusive': [first, last],
            'source_header_path': caches['header.json']['path'],
        }


def squeezenet_components():
    caches = check_cache('squeezenet1_1', SQUEEZE_URL, SQUEEZE_FILES, fetch=False)
    cache = caches[SQUEEZE_FILES[0][0]]
    path = ROOT / cache['path']
    checkpoint = path.read_bytes()
    state = torch.load(path, map_location='cpu', weights_only=True)
    for key in ('features.0.weight', 'features.0.bias'):
        tensor = state[key]
        payload = tensor.numpy().tobytes(order='C')
        start = checkpoint.find(payload)
        if start < 0 or start != checkpoint.rfind(payload):
            raise ValueError(f'Could not uniquely locate the serialized values for {key}')
        byte_range = [start, start + len(payload) - 1]
        yield key, tensor, {
            'storage_dtype': str(tensor.dtype),
            'source_path': cache['path'],
            'source_sha256': cache['sha256'],
            'source_byte_range_inclusive': byte_range,
            'checkpoint_byte_range_inclusive': byte_range,
        }


def extract_model(model_id, model_name, checkpoint_url, expected_count, source_scope, components, check):
    folder = ROOT / 'raw_sets' / model_id
    array_path = folder / 'values.npy'
    index_path = folder / 'index.json'
    values = []
    indexed_components = []
    offset = 0
    for key, tensor, location in components:
        array = tensor.numpy()
        flat = array.reshape(-1, order='C')
        if flat.dtype.str != '<f4':
            raise ValueError(f'Unexpected dtype for {key}: {flat.dtype.str}')
        first, last = location['source_byte_range_inclusive']
        with (ROOT / location['source_path']).open('rb') as source:
            source.seek(first)
            raw_bytes = source.read(last - first + 1)
        if flat.tobytes() != raw_bytes:
            raise ValueError(f'Decoded scalars differ from their checkpoint bytes for {key}')
        stop = offset + flat.size
        indexed_components.append({
            'original_key': key,
            'shape': list(array.shape),
            'dtype': array.dtype.name,
            'scalar_count': flat.size,
            'scalar_start': offset,
            'scalar_stop_exclusive': stop,
            **location,
        })
        values.append(flat)
        offset = stop

    expected_values = np.concatenate(values)
    if offset != expected_count or expected_values.shape != (expected_count,):
        raise ValueError(f'Unexpected scalar count for {model_id}: {offset}')
    if not check:
        folder.mkdir(parents=True, exist_ok=True)
        np.save(array_path, expected_values, allow_pickle=False)

    output = np.load(array_path, allow_pickle=False)
    if (output.dtype != expected_values.dtype or output.shape != expected_values.shape
            or output.tobytes() != expected_values.tobytes()):
        raise ValueError(f'Saved scalars differ from the source values for {model_id}')
    with array_path.open('rb') as saved_array:
        array_sha = hashlib.file_digest(saved_array, 'sha256').hexdigest()

    index = {
        'model': model_name,
        'model_id': model_id,
        'scalar_count': expected_count,
        'array': array_path.name,
        'array_shape': [expected_count],
        'array_dtype': output.dtype.name,
        'array_dtype_descriptor': output.dtype.str,
        'array_sha256': array_sha,
        'checkpoint_url': checkpoint_url,
        'source_paths_relative_to': 'model-reference-traces project directory',
        'scalar_order': 'Zero-based offsets into the C-order flattened array. Scalar stops are exclusive.',
        'byte_order': 'Little-endian. Source byte ranges are inclusive and zero-based within each file.',
        'scope': source_scope,
        'components': indexed_components,
    }
    if check:
        if json.loads(index_path.read_text(encoding='utf-8')) != index:
            raise ValueError(f'The saved index differs from the source layout for {model_id}')
    else:
        index_path.write_text(json.dumps(index, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(f'{model_id}: verified {expected_count:,} scalars in {array_path.relative_to(ROOT).as_posix()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify the saved arrays and indexes without writing.')
    args = parser.parse_args()
    extract_model(
        'mamba-130m', 'State Spaces Mamba-130M', MAMBA_URL, 224_256,
        'The five complete layer-0 mixer tensors in the three weight caches used in the trace. '
        'The stored D values include untraced channels. The stored x projection includes columns whose '
        'controlled inputs were always zero. The header files hold tensor metadata.',
        safetensors_components('mamba-130m', MAMBA_URL, MAMBA_FILES, MAMBA_TENSORS), args.check,
    )
    extract_model(
        'squeezenet1_1', 'TorchVision SqueezeNet 1.1 ImageNet-1K V1', SQUEEZE_URL, 1_792,
        'The first convolution includes 64 learned RGB kernels and all 64 corresponding biases.',
        squeezenet_components(), args.check,
    )
    extract_model(
        'encodec-24khz', 'Meta EnCodec 24 kHz', ENCODEC_URL, 262_144,
        'The first two codebooks each contain 1,024 vectors of 128 scalars in code ID order.',
        safetensors_components('encodec-24khz', ENCODEC_URL, ENCODEC_FILES, ENCODEC_TENSORS), args.check,
    )


if __name__ == '__main__':
    main()
