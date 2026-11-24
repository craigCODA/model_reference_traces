"""Replay controlled Mamba-130M, SqueezeNet 1.1, and EnCodec 24 kHz traces.

Run with Python and CPU PyTorch: python -B neural_probe.py
Use --fetch to download missing public weight files with curl, or --check to
compare a replay with the saved JSON. The default run uses only local files.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
MAMBA_PAGE = 'https://huggingface.co/state-spaces/mamba-130m'
MAMBA_REVISION = 'ea8027ed54e00893f4f197d1a3af820b96e86fb9'
MAMBA_URL = f'{MAMBA_PAGE}/resolve/{MAMBA_REVISION}/model.safetensors'
SQUEEZE_URL = 'https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth'
ENCODEC_PAGE = 'https://huggingface.co/facebook/encodec_24khz'
ENCODEC_REVISION = '1d4c7cf1aca62eee005a2dd499a0cdfe5abf92b9'
ENCODEC_URL = f'{ENCODEC_PAGE}/resolve/{ENCODEC_REVISION}/model.safetensors'

# Cache file, inclusive byte range (or None for the full file), local SHA-256.
MAMBA_FILES = (
    ('header_length.bin', (0, 7), '844ce1b9750da94513c3e70f21219507ba87f01fb0c325f17bba34f7c1d1a5fb'),
    ('header.json', (8, 26151), 'e3e9edfc03fa3095203c306dba24801e3c6eae3cfa9b4d8e9a9607a261fa4818'),
    ('A_log_D.bin', (154486312, 154590759), '9be0db4acd41b13040f84d7ef83fe92c321b5a4ac584387531ac4339c59f3a3a'),
    ('dt_proj.bin', (154621480, 154922535), '18e0245acd7f233ee5420a148ca89a97996ed81fb6715819b7a23bd7e64dd60b'),
    ('x_proj.bin', (169078312, 169569831), 'f40d72782af94e2f7e41fd4fad73ec25f3b5ff8cebf7505a27501fdc36abfa14'),
)
SQUEEZE_FILES = (
    ('squeezenet1_1-b8a52dc0.pth', None, 'b8a52dc049b60e4b6ab68ad0df457362afab8b6304b2febdc1650a5dab4d7e7b'),
)
ENCODEC_FILES = (
    ('header_length.bin', (0, 7), 'ed5f2eaaca536bf9e2f8d9b0c9a1bb451b20d7e12d772c55b12fcf6f13b8c820'),
    ('header.json', (8, 26735), '2694648c41e7017546c83b4a88949d8dba618bb0c7047fc7c31635b9f1ddbf9b'),
    ('codebook_0.bin', (59438072, 59962359), 'e8dcfe68ad3f9a837dbe097da425a74a29bb2d8db9496e6dc7d502cf5ebb7adb'),
    ('codebook_1.bin', (60490748, 61015035), '4195180ccbfd177228d413cc329033bada564a4079fd4b3755eafdb36f5f1b65'),
)


def verify_file(path, span, expected_sha):
    with path.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    size = path.stat().st_size
    if digest != expected_sha or (span is not None and size != span[1] - span[0] + 1):
        raise ValueError(f'Unexpected checkpoint bytes in {path}')
    return size


def download_file(path, url, span, expected_sha):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as temp_dir:
        partial = Path(temp_dir, path.name)
        command = [
            'curl', '--fail', '--silent', '--show-error', '--location',
            '--connect-timeout', '15', '--max-time', '90',
            '--output', str(partial), '--write-out', '%{http_code}',
        ]
        expected_status = '200'
        if span is None:
            command += ['--max-filesize', '4958839']
        else:
            start, end = span
            command += ['--range', f'{start}-{end}', '--max-filesize', str(end - start + 1)]
            url = f'{url}?range_start={start}&probe_nonce={time.time_ns()}'
            expected_status = '206'
        result = subprocess.run(command + [url], check=True, capture_output=True, text=True)
        if result.stdout != expected_status:
            raise ValueError(f'Expected HTTP {expected_status} for {path}, got {result.stdout!r}')
        verify_file(partial, span, expected_sha)
        partial.replace(path)


def check_cache(model_id, url, files, fetch):
    records = {}
    for name, span, expected_sha in files:
        path = ROOT / 'weights' / model_id / name
        if fetch and not path.exists():
            download_file(path, url, span, expected_sha)
        records[name] = {
            'path': path.relative_to(ROOT).as_posix(),
            'bytes': verify_file(path, span, expected_sha),
            'sha256': expected_sha,
            'checkpoint_byte_range_inclusive': span,
        }
    return records


def read_safetensors_ranges(records, tensor_files):
    header_length = int.from_bytes((ROOT / records['header_length.bin']['path']).read_bytes(), 'little')
    header_bytes = (ROOT / records['header.json']['path']).read_bytes()
    if header_length != len(header_bytes):
        raise ValueError('Incomplete Safetensors header')
    header = json.loads(header_bytes)
    data_start = 8 + header_length
    tensors = {}
    tensor_sources = {}
    for key, file_name in tensor_files.items():
        description = header[key]
        tensor_start, tensor_end = [offset + data_start for offset in description['data_offsets']]
        cache_start, cache_end = records[file_name]['checkpoint_byte_range_inclusive']
        if description['dtype'] != 'F32' or not cache_start <= tensor_start < tensor_end <= cache_end + 1:
            raise ValueError(f'Invalid range or dtype for {key}')
        with (ROOT / records[file_name]['path']).open('rb') as source:
            source.seek(tensor_start - cache_start)
            raw = source.read(tensor_end - tensor_start)
        tensors[key] = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(description['shape'])
        tensor_sources[key] = {
            'cache_file': file_name,
            'checkpoint_byte_range_inclusive': [tensor_start, tensor_end - 1],
            'dtype': description['dtype'],
            'shape': description['shape'],
        }
    return tensors, tensor_sources


def mamba_trace(fetch):
    model_id = 'mamba-130m'
    artifacts = check_cache(model_id, MAMBA_URL, MAMBA_FILES, fetch)
    prefix = 'backbone.layers.0.mixer.'
    tensor_files = {
        'A_log': 'A_log_D.bin', 'D': 'A_log_D.bin',
        'dt_proj.bias': 'dt_proj.bin', 'dt_proj.weight': 'dt_proj.bin',
        'x_proj.weight': 'x_proj.bin',
    }
    tensors, tensor_sources = read_safetensors_ranges(
        artifacts, {prefix + name: file_name for name, file_name in tensor_files.items()},
    )
    weights = {name: tensors[prefix + name] for name in tensor_files}
    channels, state_size = weights['A_log'].shape
    rank = weights['dt_proj.weight'].shape[1]

    # These controlled x values enter after the mixer's convolution and SiLU.
    input_channels = [0, 17, 83, 257, 509, 1021, 1477]
    active_inputs = torch.tensor([
        [1.0, 0.75, -0.5, 0.25, 1.25, -0.75, 0.5],
        [-0.5, -0.875, 1.0, 1.5, -0.25, 0.625, -1.0],
        [1.25, 0.0, -0.25, -0.75, 0.625, 0.5, 0.875],
        [0.625, 1.5, -1.125, 0.25, -0.5, 1.25, 0.375],
    ])
    channel = 17
    z_inputs = torch.tensor([0.75, 1.25, -0.5, 1.0])
    inputs = torch.zeros(len(active_inputs), channels)
    inputs[:, input_channels] = active_inputs
    x_projections = F.linear(inputs, weights['x_proj.weight'])
    low_rank, input_weights, readout_weights = x_projections.split((rank, state_size, state_size), dim=1)
    time_step_logits = F.linear(low_rank, weights['dt_proj.weight']) + weights['dt_proj.bias']
    time_steps = F.softplus(time_step_logits)
    A = -weights['A_log'].exp()
    retention = (time_steps.unsqueeze(-1) * A).exp()
    write_weights = time_steps.unsqueeze(-1) * input_weights.unsqueeze(1)

    # A scalar float64 replay derives the same channel from the supplied weights.
    input_columns = weights['x_proj.weight'][:, input_channels].tolist()
    time_projection_row = weights['dt_proj.weight'][channel].tolist()
    time_projection_bias = weights['dt_proj.bias'][channel].item()
    log_rates = weights['A_log'][channel].tolist()
    direct_skip_weight = weights['D'][channel].item()
    independent_state = [0.0] * state_size
    state = torch.zeros_like(A)
    steps = []
    for index, values in enumerate(active_inputs.tolist()):
        prior = state
        retained_state = prior * retention[index]
        written_state = write_weights[index] * inputs[index].unsqueeze(-1)
        state = retained_state + written_state
        readout_terms = state[channel] * readout_weights[index]
        direct_skip = weights['D'][channel] * inputs[index, channel]
        gate = F.silu(z_inputs[index])
        gated_output = (readout_terms.sum() + direct_skip) * gate
        reset_output = ((written_state[channel] * readout_weights[index]).sum() + direct_skip) * gate

        independent_projection = [
            math.fsum(weight * value for weight, value in zip(row, values))
            for row in input_columns
        ]
        independent_time_logit = time_projection_bias + math.fsum(
            weight * value for weight, value in zip(time_projection_row, independent_projection[:rank])
        )
        independent_time = math.log1p(math.exp(independent_time_logit))
        independent_b = independent_projection[rank:rank + state_size]
        independent_c = independent_projection[rank + state_size:]
        x = inputs[index, channel].item()
        z = z_inputs[index].item()
        independent_state = [
            previous * math.exp(-independent_time * math.exp(log_rate)) + x * independent_time * b
            for previous, log_rate, b in zip(independent_state, log_rates, independent_b)
        ]
        independent_output = (
            math.fsum(value * c for value, c in zip(independent_state, independent_c)) + x * direct_skip_weight
        ) * z / (1 + math.exp(-z))
        torch.testing.assert_close(
            state[channel].double(), torch.tensor(independent_state, dtype=torch.float64), atol=1e-7, rtol=2e-5,
        )
        torch.testing.assert_close(float(gated_output), independent_output, atol=1e-7, rtol=2e-5)

        steps.append({
            'step': index,
            'x_on_active_channels': values,
            'x_on_traced_channel': x,
            'z_on_traced_channel': z,
            'dt_low_rank': low_rank[index].tolist(),
            'dt_logit_on_traced_channel': time_step_logits[index, channel].item(),
            'dt_on_traced_channel': time_steps[index, channel].item(),
            'B': input_weights[index].tolist(),
            'C': readout_weights[index].tolist(),
            'previous_state': prior[channel].tolist(),
            'retention_exp_dt_A': retention[index, channel].tolist(),
            'write_factor_dt_B': write_weights[index, channel].tolist(),
            'retained_state': retained_state[channel].tolist(),
            'written_state': written_state[channel].tolist(),
            'updated_state': state[channel].tolist(),
            'readout_terms': readout_terms.tolist(),
            'state_readout': readout_terms.sum().item(),
            'D_times_x': direct_skip.item(),
            'silu_z': gate.item(),
            'gated_output_before_out_proj': gated_output.item(),
            'gated_output_with_state_reset': reset_output.item(),
            'independent_scalar_gated_output': independent_output,
        })

    return {
        'id': model_id,
        'model': 'State Spaces Mamba-130M',
        'operation': 'Layer 0 selective state-space update over four controlled steps',
        'scope': 'The prescribed x enters after convolution and SiLU. The prescribed z is the gate input. '
                 'The trace covers the trained x and time-step projections, recurrence and gated readout '
                 'before the final output projection. It begins with zero state and exposes all 16 modes of one channel.',
        'sources': {
            'training_and_release': 'https://github.com/state-spaces/mamba#pretrained-models',
            'checkpoint_page': f'{MAMBA_PAGE}/blob/{MAMBA_REVISION}/model.safetensors',
            'checkpoint_url': MAMBA_URL,
            'publisher_listed_full_file_sha256': 'd2634ae0d405f2482574836b0d036372bd0f917aaeacbe026358a33138e63d6d',
            'implementation': 'https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py',
        },
        'artifacts': artifacts,
        'tensor_sources': tensor_sources,
        'mixer_channels': channels,
        'state_size_per_channel': state_size,
        'time_step_rank': rank,
        'traced_channel': channel,
        'input_channels_with_supplied_values': input_channels,
        'other_input_channels_value': 0.0,
        'x_projection_columns_for_input_channels': input_columns,
        'dt_projection_weight_row': time_projection_row,
        'dt_projection_bias': time_projection_bias,
        'A_log': log_rates,
        'A': A[channel].tolist(),
        'D': direct_skip_weight,
        'steps': steps,
    }


def squeezenet_trace(fetch):
    model_id = 'squeezenet1_1'
    artifacts = check_cache(model_id, SQUEEZE_URL, SQUEEZE_FILES, fetch)
    checkpoint = ROOT / artifacts[SQUEEZE_FILES[0][0]]['path']
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    filters = state['features.0.weight']
    biases = state['features.0.bias']
    if filters.shape != (64, 3, 3, 3) or biases.shape != (64,):
        raise ValueError('Unexpected SqueezeNet 1.1 first-layer shape')
    rows, cols = torch.meshgrid(torch.arange(7), torch.arange(7), indexing='ij')
    image = torch.stack([
        (3 * rows - cols - 12) / 8,
        ((2 * rows + 3 * cols) % 7 - 3) / 4,
        torch.where((rows // 2 + cols // 2) % 2 == 0, 0.75, -0.5),
    ])
    stride = 2
    output = F.conv2d(image.unsqueeze(0), filters, biases, stride=stride)[0]
    filter_index = 0
    kernel = filters[filter_index]
    bias = biases[filter_index]
    rows_out, cols_out = output.shape[1:]
    positions = []
    max_error = 0.0
    for row in range(rows_out):
        for col in range(cols_out):
            top, left = row * stride, col * stride
            patch = image[:, top:top + 3, left:left + 3]
            direct_output = (filters.double() * patch.double()).sum(dim=(1, 2, 3)) + biases.double()
            torch.testing.assert_close(output[:, row, col].double(), direct_output, atol=2e-6, rtol=2e-6)
            max_error = max(max_error, (output[:, row, col] - direct_output).abs().max().item())
            selected_value = output[filter_index, row, col].item()
            positions.append({
                'output_row_col': [row, col],
                'input_top_left_row_col': [top, left],
                'rgb_patch': patch.tolist(),
                'elementwise_products_by_channel': (patch * kernel).tolist(),
                'preactivation': selected_value,
                'relu_activation': max(0.0, selected_value),
                'independent_float64_preactivation': direct_output[filter_index].item(),
            })

    return {
        'id': model_id,
        'model': 'TorchVision SqueezeNet 1.1 ImageNet-1K V1',
        'operation': 'First-layer spatial convolution and ReLU',
        'scope': 'A controlled 7 by 7 RGB tensor enters features.0. All 64 filters are evaluated at nine '
                 'spatial positions. The complete patch operands are shown for filter 0, with every filter '
                 'and bias available in the checkpoint.',
        'sources': {
            'model_documentation': 'https://docs.pytorch.org/vision/main/models/generated/torchvision.models.squeezenet1_1.html',
            'checkpoint_url': SQUEEZE_URL,
            'implementation': 'https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py',
        },
        'artifacts': artifacts,
        'tensor_keys': ['features.0.weight', 'features.0.bias'],
        'input_rgb_channel_first': image.tolist(),
        'stride': stride,
        'padding': 0,
        'kernel_size': [3, 3],
        'filter_index': filter_index,
        'filter_kernel_rgb_channel_first': kernel.tolist(),
        'filter_bias': bias.item(),
        'filter_preactivation_map': output[filter_index].tolist(),
        'filter_relu_activation_map': output[filter_index].relu().tolist(),
        'preactivations_by_filter_and_position': output.tolist(),
        'positions': positions,
        'max_absolute_error_against_direct_float64_sum': max_error,
    }


def encodec_trace(fetch):
    model_id = 'encodec-24khz'
    artifacts = check_cache(model_id, ENCODEC_URL, ENCODEC_FILES, fetch)
    keys = [f'quantizer.layers.{stage}.codebook.embed' for stage in range(2)]
    tensors, tensor_sources = read_safetensors_ranges(
        artifacts, {key: f'codebook_{stage}.bin' for stage, key in enumerate(keys)},
    )
    first_book = tensors[keys[0]]
    perturbation = ((torch.arange(128) * 37 + 11) % 31 - 15) / 16
    latent = 0.625 * first_book[17] + 0.375 * first_book[341] + perturbation
    residual = latent.clone()
    reconstructed_latent = torch.zeros_like(latent)
    stage_records = []
    for stage, key in enumerate(keys):
        book = tensors[key]
        if book.shape != (1024, 128):
            raise ValueError(f'Unexpected EnCodec codebook shape for {key}')
        stage_input = residual
        squared_distances = stage_input.square().sum() - 2 * (book @ stage_input) + book.square().sum(dim=1)
        direct_distances = (book.double() - stage_input.double()).square().sum(dim=1)
        torch.testing.assert_close(squared_distances.double(), direct_distances, atol=3e-3, rtol=2e-6)
        winner = int(squared_distances.argmin())
        if winner != int(direct_distances.argmin()):
            raise ValueError(f'EnCodec stage {stage} selected a different codeword in the direct calculation')
        selected_codeword = book[winner]
        residual = stage_input - selected_codeword
        reconstructed_latent += selected_codeword
        ranked_ids = squared_distances.argsort(stable=True)[:5].tolist()
        stage_records.append({
            'stage': stage,
            'tensor_key': key,
            'stage_input': stage_input.tolist(),
            'winning_code_id': winner,
            'runner_up_code_id': ranked_ids[1],
            'squared_l2_by_code_id': squared_distances.tolist(),
            'top_five': [{
                'code_id': code_id,
                'squared_l2': squared_distances[code_id].item(),
                'independent_direct_squared_l2': direct_distances[code_id].item(),
                'codeword': book[code_id].tolist(),
            } for code_id in ranked_ids],
            'residual_after_subtracting_winner': residual.tolist(),
            'residual_squared_norm_after_subtracting_winner': residual.double().square().sum().item(),
            'max_absolute_error_against_direct_float64_distances': (
                squared_distances.double() - direct_distances
            ).abs().max().item(),
        })

    torch.testing.assert_close(latent - reconstructed_latent, residual, atol=1e-6, rtol=1e-6)
    return {
        'id': model_id,
        'model': 'Meta EnCodec 24 kHz',
        'operation': 'Two stages of residual vector quantization',
        'scope': 'A controlled 128-dimensional vector enters the quantizer in the place of one encoder frame. '
                 'The vector is a weighted blend of two first-stage codewords plus a deterministic perturbation. '
                 'Each stage scores all 1024 learned 128-dimensional candidates. Stage 1 receives the residual '
                 'after subtracting the stage 0 winner. All candidates are rows of the two stored codebook tensors.',
        'sources': {
            'publisher_project': 'https://github.com/facebookresearch/encodec',
            'checkpoint_page': f'{ENCODEC_PAGE}/blob/{ENCODEC_REVISION}/model.safetensors',
            'checkpoint_url': ENCODEC_URL,
            'publisher_listed_full_file_sha256': '37a7cb100f71a29e6c1d815aca8666a1d7ea8885ebe44c306c751a5103559d57',
            'implementation': 'https://github.com/facebookresearch/encodec/blob/main/encodec/quantization/core_vq.py',
        },
        'artifacts': artifacts,
        'tensor_sources': tensor_sources,
        'latent_construction': {
            'first_codebook_ids': [17, 341],
            'codeword_coefficients': [0.625, 0.375],
            'added_perturbation': perturbation.tolist(),
        },
        'latent': latent.tolist(),
        'latent_squared_norm': latent.double().square().sum().item(),
        'score_rule': 'The native implementation maximizes negative squared Euclidean distance. '
                      'The same code ID minimizes the squared distances reported here.',
        'stages': stage_records,
        'reconstructed_latent': reconstructed_latent.tolist(),
        'final_residual': residual.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true', help='Download missing checkpoint files or ranges with curl.')
    parser.add_argument('--check', action='store_true', help='Compare with the saved neural_results.json without writing.')
    args = parser.parse_args()
    torch.set_num_threads(1)
    with torch.inference_mode():
        result = {
            'schema_version': 1,
            'arithmetic': 'CPU PyTorch float32 with independent float64 arithmetic checks',
            'models': [mamba_trace(args.fetch), squeezenet_trace(args.fetch), encodec_trace(args.fetch)],
        }
    encoded = json.dumps(result, indent=2, allow_nan=False) + '\n'
    destination = ROOT / 'neural_results.json'
    if args.check:
        if destination.read_text(encoding='utf-8') != encoded:
            raise SystemExit('Saved neural_results.json differs from the replay.')
        print(f'Verified all three traces against {destination.name}')
    else:
        destination.write_text(encoded, encoding='utf-8', newline='\n')
        print(f'Wrote all three traces to {destination.name}')


if __name__ == '__main__':
    main()
