"""Aggregate evaluation script for K-ProtoDiff.

Wraps the metrics referenced in the README:
- Context-FID
- KL Divergence
- Discriminative Score (DS)
- Predictive Score (PS)
- Segment-wise DTW

NOTE: this is a minimal reference implementation reconstructed from the
scaffolding of Diffusion-TS (ICLR 2024) plus standard metric implementations.
The exact KL / segment-wise DTW formulations used in the K-ProtoDiff paper may
differ slightly; verify against the paper before reporting numbers.
"""
import os
import argparse
import numpy as np
from dtaidistance import dtw_ndim

def kl_divergence(ori, fake, n_bins=20, eps=1e-10):
    ori_flat = ori.reshape(-1)
    fake_flat = fake.reshape(-1)
    lo = min(ori_flat.min(), fake_flat.min())
    hi = max(ori_flat.max(), fake_flat.max())
    bins = np.linspace(lo, hi, n_bins + 1)
    p, _ = np.histogram(ori_flat, bins=bins, density=True)
    q, _ = np.histogram(fake_flat, bins=bins, density=True)
    p = p / (p.sum() + eps) + eps
    q = q / (q.sum() + eps) + eps
    return float(np.sum(p * np.log(p / q)))


def segment_wise_dtw(ori, fake, segment_len=8, stride=4):
    n = min(len(ori), len(fake))
    distances = []
    for i in range(n):
        x, y = ori[i], fake[i]
        T = x.shape[0]
        seg_dists = []
        for s in range(0, T - segment_len + 1, stride):
            xs = np.asarray(x[s:s + segment_len], dtype=np.double)
            ys = np.asarray(y[s:s + segment_len], dtype=np.double)
            seg_dists.append(dtw_ndim.distance_fast(xs, ys))
        if seg_dists:
            distances.append(np.mean(seg_dists))
    return float(np.mean(distances)) if distances else float('nan')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='output root for this run')
    parser.add_argument('--ori_path', type=str, required=True, help='path to .npy of real samples')
    parser.add_argument('--fake_path', type=str, required=True, help='path to .npy of generated samples')
    parser.add_argument('--n_iter', type=int, default=5, help='repetitions for DS / PS')
    parser.add_argument('--segment_lengths', type=int, nargs='+', default=[6, 8, 10])
    parser.add_argument('--segment_stride', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output-name', default='metrics.txt')
    parser.add_argument('--skip', nargs='*', default=[], help='metric names to skip: cfid, kl, ds, ps, dtw')
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    ori = np.load(args.ori_path)
    fake = np.load(args.fake_path)
    n = min(len(ori), len(fake))
    ori, fake = ori[:n], fake[:n]
    print(f'ori shape: {ori.shape}, fake shape: {fake.shape}')

    results = {}
    if 'cfid' not in args.skip:
        from Utils.context_fid import Context_FID
        results['Context-FID'] = Context_FID(ori, fake)
    if 'kl' not in args.skip:
        results['KL'] = kl_divergence(ori, fake)
    if 'ds' not in args.skip:
        from Utils.discriminative_metric import discriminative_score_metrics
        ds = [
            discriminative_score_metrics(ori, fake, seed=args.seed + index)[0]
            for index in range(args.n_iter)
        ]
        results['DS_mean'] = float(np.mean(ds))
        results['DS_std'] = float(np.std(ds))
    if 'ps' not in args.skip:
        from Utils.predictive_metric import predictive_score_metrics
        ps = [
            predictive_score_metrics(ori, fake, seed=args.seed + index)
            for index in range(args.n_iter)
        ]
        results['PS_mean'] = float(np.mean(ps))
        results['PS_std'] = float(np.std(ps))
    if 'dtw' not in args.skip:
        for segment_len in args.segment_lengths:
            results[f'Segment-DTW-L{segment_len}'] = segment_wise_dtw(
                ori, fake, segment_len=segment_len, stride=args.segment_stride
            )

    os.makedirs(args.root, exist_ok=True)
    out = os.path.join(args.root, args.output_name)
    with open(out, 'w') as f:
        for k, v in results.items():
            line = f'{k}: {v:.6f}'
            print(line)
            f.write(line + '\n')
    print(f'\nSaved to {out}')


if __name__ == '__main__':
    main()
