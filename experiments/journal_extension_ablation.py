"""Small reproducible ablation for the journal-extension modules.

This experiment is intentionally self-contained: it creates sine series with
localized spikes and level shifts, trains the conference-style point-prototype
model and the new multi-scale model for the same number of updates, then
compares fixed-step DDIM with adaptive reflection sampling.

It is an engineering sanity check, not a replacement for the full paper runs.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Models.KPD.k_diffusion import Model


def make_key_pattern_data(num_samples, length, channels, seed):
    generator = torch.Generator().manual_seed(seed)
    time_axis = torch.linspace(0, 1, length)
    samples = []
    for _ in range(num_samples):
        series = []
        for _ in range(channels):
            frequency = 1.0 + 3.0 * torch.rand(1, generator=generator).item()
            phase = 2.0 * math.pi * torch.rand(1, generator=generator).item()
            amplitude = 0.2 + 0.25 * torch.rand(1, generator=generator).item()
            channel = amplitude * torch.sin(2.0 * math.pi * frequency * time_axis + phase)
            channel += 0.025 * torch.randn(length, generator=generator)
            series.append(channel)
        sample = torch.stack(series, dim=-1)

        event_channel = int(torch.randint(channels, (1,), generator=generator))
        event_position = int(torch.randint(4, length - 6, (1,), generator=generator))
        if torch.rand(1, generator=generator).item() < 0.5:
            width = int(torch.randint(2, 5, (1,), generator=generator))
            height = 0.55 + 0.25 * torch.rand(1, generator=generator).item()
            sample[event_position:event_position + width, event_channel] += height
        else:
            width = int(torch.randint(5, 10, (1,), generator=generator))
            height = 0.35 + 0.2 * torch.rand(1, generator=generator).item()
            sample[event_position:event_position + width, event_channel] += height
        samples.append(sample.clamp(-1.0, 1.0))
    return torch.stack(samples)


def load_real_data(path, length, train_size, eval_size, train_proportion):
    frame = pd.read_csv(path).select_dtypes(include='number')
    values = torch.tensor(frame.to_numpy(dtype=np.float32))
    if train_proportion >= 1.0:
        train_values, eval_values = values, values
    else:
        split = int(train_proportion * len(values))
        train_values, eval_values = values[:split], values[split:]
    minimum = train_values.amin(dim=0, keepdim=True)
    scale = (train_values.amax(dim=0, keepdim=True) - minimum).clamp_min(1e-6)
    train_values = 2.0 * (train_values - minimum) / scale - 1.0
    eval_values = 2.0 * (eval_values - minimum) / scale - 1.0

    def windows(data, limit):
        result = data.unfold(0, length, 1).permute(0, 2, 1).contiguous()
        if len(result) > limit:
            indices = torch.linspace(0, len(result) - 1, limit).long()
            result = result[indices]
        return result

    return windows(train_values, train_size), windows(eval_values, eval_size)


def build_model(prototype_mode, args, device):
    return Model(
        seq_length=args.length,
        feature_size=args.channels,
        n_layer_enc=1,
        n_layer_dec=1,
        d_model=args.d_model,
        timesteps=args.diffusion_steps,
        sampling_timesteps=args.sample_nfe,
        loss_type='l2',
        beta_schedule='cosine',
        n_heads=4,
        mlp_hidden_times=2,
        attn_pd=0.0,
        resid_pd=0.0,
        kernel_size=1,
        padding_size=0,
        use_ff=True,
        fdm_mode='real',
        num_prototypes=args.num_prototypes,
        prototype_mode=prototype_mode,
        prototype_scales=args.prototype_scales,
        prototype_loss_weight=0.05,
        sampling_mode='adaptive_reflection' if prototype_mode == 'multiscale' else 'legacy',
        adaptive_sampling_timesteps=args.sample_nfe,
        reflection_strength=getattr(args, 'reflection_strength', 0.1),
        reflection_threshold=getattr(args, 'reflection_threshold', 0.15),
        reflection_topk_ratio=getattr(args, 'reflection_topk_ratio', 0.25),
        reflection_roughness_threshold=getattr(
            args, 'reflection_roughness_threshold', 1.25
        ),
    ).to(device)


def train_model(model, training_data, args, seed):
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train()
    losses = []
    started = time.perf_counter()
    for step in range(args.train_steps):
        indices = torch.randint(0, len(training_data), (args.batch_size,), device=training_data.device)
        batch = training_data[indices]
        optimizer.zero_grad(set_to_none=True)
        loss = model(batch, target=batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % max(1, args.train_steps // 5) == 0:
            print(f'  step {step + 1:4d}/{args.train_steps}: loss={np.mean(losses[-20:]):.6f}')
    return {
        'final_loss': float(np.mean(losses[-20:])),
        'training_seconds': time.perf_counter() - started,
        'loss_components': getattr(model, '_last_loss_components', None),
    }


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def sample_model(model, num_samples, mode, nfe, device, seed=2026):
    model.eval()
    shape = (num_samples, model.seq_length, model.feature_size)

    # Warm up the exact tensor shape before timing.  This keeps CUDA context and
    # kernel initialization out of the sampler comparison.
    torch.manual_seed(seed)
    warm_input = torch.randn(shape, device=device)
    warm_time = torch.full(
        (num_samples,), model.num_timesteps - 1, device=device, dtype=torch.long
    )
    _, warm_start = model.model_predictions(warm_input, warm_time, clip_x_start=True)
    if mode == 'adaptive':
        model._adaptive_reflect(warm_start, model.num_timesteps - 1)
    synchronize(device)

    # Every sampler receives the same initial noise for paired comparisons.
    torch.manual_seed(seed)
    synchronize(device)
    started = time.perf_counter()
    if mode == 'adaptive':
        generated = model.adaptive_reflection_sample(shape, max_nfe=nfe)
        actual_nfe = model.last_sampling_stats['nfe']
        sampler_stats = model.last_sampling_stats
    elif mode == 'ddim':
        original_steps = model.sampling_timesteps
        model.sampling_timesteps = nfe
        generated = model.fast_sample(shape)
        model.sampling_timesteps = original_steps
        actual_nfe = nfe
        sampler_stats = None
    elif mode == 'legacy_full':
        generated = model.sample(shape)
        reflected_steps = sum(
            timestep < model.lambda_step for timestep in range(model.num_timesteps)
        )
        actual_nfe = model.num_timesteps + 2 * model.T_max * reflected_steps
        sampler_stats = {'reflected_steps': reflected_steps}
    else:
        raise ValueError(mode)
    synchronize(device)
    return generated, {
        'sampling_seconds': time.perf_counter() - started,
        'nfe': actual_nfe,
        'sampler_stats': sampler_stats,
    }


def quality_metrics(real, generated):
    real, generated = real.cpu(), generated.cpu()
    real_diff = real[:, 1:] - real[:, :-1]
    fake_diff = generated[:, 1:] - generated[:, :-1]
    event_threshold = torch.quantile(real_diff.abs().flatten(), 0.95)
    real_event_density = (real_diff.abs() > event_threshold).float().mean()
    fake_event_density = (fake_diff.abs() > event_threshold).float().mean()
    real_q95 = torch.quantile(real_diff.abs().flatten(), 0.95)
    fake_q95 = torch.quantile(fake_diff.abs().flatten(), 0.95)

    def lag_one_correlation(data):
        left = data[:, :-1] - data[:, :-1].mean(dim=1, keepdim=True)
        right = data[:, 1:] - data[:, 1:].mean(dim=1, keepdim=True)
        numerator = (left * right).mean(dim=1)
        denominator = left.square().mean(dim=1).sqrt() * right.square().mean(dim=1).sqrt()
        return (numerator / denominator.clamp_min(1e-6)).mean(dim=0)

    real_acf = lag_one_correlation(real)
    fake_acf = lag_one_correlation(generated)
    return {
        'mean_error': float((real.mean(dim=(0, 1)) - generated.mean(dim=(0, 1))).abs().mean()),
        'std_error': float((real.std(dim=(0, 1)) - generated.std(dim=(0, 1))).abs().mean()),
        'lag1_acf_error': float((real_acf - fake_acf).abs().mean()),
        'derivative_q95_error': float((real_q95 - fake_q95).abs()),
        'event_density_error': float((real_event_density - fake_event_density).abs()),
        'real_event_density': float(real_event_density),
        'generated_event_density': float(fake_event_density),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--train-size', type=int, default=1024)
    parser.add_argument('--eval-size', type=int, default=128)
    parser.add_argument('--length', type=int, default=48)
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--d-model', type=int, default=32)
    parser.add_argument('--diffusion-steps', type=int, default=40)
    parser.add_argument('--sample-nfe', type=int, default=10)
    parser.add_argument('--train-steps', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--num-prototypes', type=int, default=6)
    parser.add_argument('--prototype-scales', type=int, nargs='+', default=[4, 8, 16])
    parser.add_argument('--reflection-strength', type=float, default=0.1)
    parser.add_argument('--reflection-threshold', type=float, default=0.15)
    parser.add_argument('--reflection-topk-ratio', type=float, default=0.25)
    parser.add_argument('--reflection-roughness-threshold', type=float, default=1.25)
    parser.add_argument('--output', default='OUTPUT/journal_extension_ablation.json')
    parser.add_argument('--dataset-path', default=None)
    parser.add_argument(
        '--train-proportion', type=float, default=1.0,
        help='1.0 reproduces the paper distribution-matching protocol; use 0.8 for temporal holdout',
    )
    parser.add_argument('--save-artifacts', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.dataset_path:
        training_data, evaluation_data = load_real_data(
            args.dataset_path, args.length, args.train_size, args.eval_size,
            args.train_proportion,
        )
        args.channels = training_data.shape[-1]
        print(
            f'Loaded {args.dataset_path}: train={tuple(training_data.shape)}, '
            f'eval={tuple(evaluation_data.shape)}'
        )
        training_data = training_data.to(device)
    else:
        training_data = make_key_pattern_data(
            args.train_size, args.length, args.channels, args.seed
        ).to(device)
        evaluation_data = make_key_pattern_data(
            args.eval_size, args.length, args.channels, args.seed + 1
        )

    results = {'settings': vars(args), 'models': {}}
    output_path = Path(args.output)
    artifact_root = output_path.parent / f'{output_path.stem}_artifacts'
    if args.save_artifacts:
        artifact_root.mkdir(parents=True, exist_ok=True)
    trained_models = {}
    for name, prototype_mode in [('point', 'point'), ('multiscale', 'multiscale')]:
        print(f'\nTraining {name} prototype model')
        # Reset before construction so the shared denoiser and KAN components
        # begin from identical weights in the paired ablation.
        torch.manual_seed(args.seed)
        model = build_model(prototype_mode, args, device)
        train_stats = train_model(model, training_data, args, args.seed)
        trained_models[name] = model
        results['models'][name] = {'training': train_stats, 'sampling': {}}
        if args.save_artifacts:
            torch.save(
                {'model': model.state_dict(), 'settings': vars(args)},
                artifact_root / f'{name}_model.pt',
            )

    sampling_runs = [
        ('point', 'legacy_full'),
        ('point', 'ddim'),
        ('multiscale', 'ddim'),
        ('multiscale', 'legacy_full'),
        ('multiscale', 'adaptive'),
    ]
    for model_name, sampler_name in sampling_runs:
        print(f'\nSampling {model_name}/{sampler_name}')
        generated, speed = sample_model(
            trained_models[model_name], args.eval_size, sampler_name,
            args.sample_nfe, device, seed=args.seed + 100,
        )
        results['models'][model_name]['sampling'][sampler_name] = {
            **speed,
            'quality': quality_metrics(evaluation_data, generated),
        }
        if args.save_artifacts:
            np.save(
                artifact_root / f'{model_name}_{sampler_name}_samples.npy',
                generated.detach().cpu().numpy(),
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'\nSaved results to {output_path}')
    print(json.dumps(results['models'], indent=2))


if __name__ == '__main__':
    main()
