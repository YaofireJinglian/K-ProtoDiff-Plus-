"""Evaluate adaptive-reflection settings from a saved ablation checkpoint."""

import argparse
import json
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.journal_extension_ablation import (
    build_model,
    load_real_data,
    quality_metrics,
    sample_model,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--strengths', type=float, nargs='+', default=[0.0, 0.1, 0.2, 0.3])
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.15, 0.25, 0.4])
    parser.add_argument('--budgets', type=int, nargs='+', default=[10, 20, 50])
    parser.add_argument('--seed', type=int, default=2126)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = argparse.Namespace(**checkpoint['settings'])
    model = build_model('multiscale', saved_args, device)
    model.load_state_dict(checkpoint['model'], strict=False)
    training_data, real = load_real_data(
        saved_args.dataset_path,
        saved_args.length,
        saved_args.train_size,
        saved_args.eval_size,
        saved_args.train_proportion,
    )
    model.calibrate_reflection(training_data[:min(1024, len(training_data))].to(device))

    results = []
    for budget in args.budgets:
        generated, speed = sample_model(
            model, saved_args.eval_size, 'ddim', budget, device, seed=args.seed
        )
        results.append({
            'sampler': 'ddim',
            'budget': budget,
            **speed,
            'quality': quality_metrics(real, generated),
        })
        for strength in args.strengths:
            for threshold in args.thresholds:
                model.reflection_strength = strength
                model.reflection_threshold = threshold
                generated, speed = sample_model(
                    model, saved_args.eval_size, 'adaptive', budget, device,
                    seed=args.seed,
                )
                results.append({
                    'sampler': 'adaptive',
                    'budget': budget,
                    'strength': strength,
                    'threshold': threshold,
                    **speed,
                    'quality': quality_metrics(real, generated),
                })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'Saved {len(results)} sweep runs to {output}')


if __name__ == '__main__':
    main()
