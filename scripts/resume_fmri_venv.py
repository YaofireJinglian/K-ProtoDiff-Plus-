"""Resume migrated fMRI runs using the repository venv and centralized checkpoints."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['diffusion_ts', 'pad_ts', 'k_protodiff'], required=True)
    parser.add_argument('--seed', type=int, choices=[2026, 2027, 2028], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    lockdir = ROOT / 'OUTPUT/run_locks'
    lockdir.mkdir(parents=True, exist_ok=True)
    run_lock = (lockdir / f'fmri_{args.method}_{args.seed}.lock').open('a')
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('This method/seed is already running; skipping duplicate launch.', flush=True)
        return
    completed_record = ROOT / 'OUTPUT/main_results_records' / f'fmri__{args.method}__seed{args.seed}.json'
    if completed_record.exists():
        completed = json.loads(completed_record.read_text())
        required = {'C-FID', 'KL', 'DS', 'PS', 'Seg-DTW-L6', 'Seg-DTW-L8', 'Seg-DTW-L10'}
        if required.issubset(completed.get('metrics', {})):
            print('All metrics already recorded; skipping completed run.', flush=True)
            return
    py = str(ROOT / '.venv/bin/python')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               TF_NUM_INTRAOP_THREADS='4', TF_NUM_INTEROP_THREADS='2')
    out = ROOT / 'OUTPUT/fmri_baselines'
    if args.method == 'diffusion_ts':
        name = f'fmri_diffusion_ts_seed{args.seed}'
        cwd = ROOT / 'baselines/Diffusion-TS'
        fake = out / name / f'ddpm_fake_{name}.npy'
        real = out / name / 'samples/fMRI_norm_truth_24_train.npy'
        cmd = [py, '-u', 'main.py', '--name', name, '--config_file', 'Config/fmri.yaml',
               '--output', str(out), '--gpu', '0', '--seed', str(args.seed),
               '--milestone', '10', 'solver.results_folder',
               str(ROOT / 'checkpoints/Diffusion-TS' / f'Checkpoints_{name}')]
    elif args.method == 'k_protodiff':
        cwd = ROOT
        name = f'fmri_k_protodiff_seed{args.seed}'
        fake = out / name / f'ddpm_fake_{name}.npy'
        real = out / name / 'samples/fMRI_norm_truth_24_train.npy'
        base = ROOT / 'checkpoints/conference' / f'Checkpoints_{name}'
        common = [py, '-u', 'run.py', '--name', name, '--config_file', 'Config/fmri.yaml',
                  '--output', str(out), '--gpu', '0', '--seed', str(args.seed)]
        if not Path(f'{base}_24/checkpoint-10.pt').exists():
            subprocess.run(common + ['--train', 'solver.results_folder', str(base)],
                           cwd=cwd, env=env, check=True)
        cmd = common + ['--milestone', '10', 'solver.results_folder', str(base)]
    else:
        cwd = ROOT / 'baselines/PaD-TS'
        ckptdir = ROOT / 'checkpoints/PaD-TS' / f'seed{args.seed}'
        fake = ckptdir / 'ddpm_fake_fmri_24.npy'
        real = ROOT / 'OUTPUT/formal/journal_fmri_seed2026/samples/fMRI_norm_truth_24_train.npy'
        ckpts = sorted(ckptdir.glob('model_*.pt'))
        cmd = [py, '-u', 'run.py', '-d', 'fmri', '--seed', str(args.seed),
               '--save-dir', str(ckptdir), '--skip-eval']
        if ckpts:
            cmd += ['--resume', str(ckpts[-1])]
    if not fake.exists():
        print('Running:', ' '.join(cmd), flush=True)
        subprocess.run(cmd, cwd=cwd, env=env, check=True)
    import numpy as np
    samples = np.load(fake)
    truth = np.load(real, mmap_mode='r')
    if samples.shape[1:] != truth.shape[1:] or len(samples) < len(truth):
        raise ValueError(f'Sample shape mismatch: {samples.shape} vs {truth.shape}')
    samples = samples[:len(truth)]
    if args.method == 'pad_ts':
        samples = (samples + 1) / 2
    metricdir = out / f'metrics_{args.method}_seed{args.seed}'
    metricdir.mkdir(parents=True, exist_ok=True)
    normalized = metricdir / 'fake_eval.npy'
    np.save(normalized, samples)
    subprocess.run([py, '-u', str(ROOT / 'metric_pytorch.py'), '--root', str(metricdir),
                    '--ori_path', str(real), '--fake_path', str(normalized), '--n_iter', '1',
                    '--seed', str(args.seed), '--output-name', 'metrics.txt'],
                   cwd=ROOT, env=env, check=True)
    values = dict(line.split(':', 1) for line in (metricdir / 'metrics.txt').read_text().splitlines())
    mapping = {'Context-FID': 'C-FID', 'KL': 'KL', 'DS_mean': 'DS', 'PS_mean': 'PS',
               **{f'Segment-DTW-L{n}': f'Seg-DTW-L{n}' for n in [6, 8, 10]}}
    record = {'dataset': 'fMRI', 'method': {'diffusion_ts': 'Diffusion-TS', 'pad_ts': 'PaD-TS', 'k_protodiff': 'K-ProtoDiff'}[args.method],
              'seed': args.seed, 'metrics': {dst: float(values[src]) for src, dst in mapping.items()}}
    if not all(np.isfinite(v) for v in record['metrics'].values()):
        raise ValueError('Non-finite evaluation result')
    with (ROOT / 'OUTPUT/.main-results.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = ROOT / 'OUTPUT/main_results_records' / f'fmri__{args.method}__seed{args.seed}.json'
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(path)
        for builder in ['build_main_results_table.py', 'build_segment_dtw_table.py']:
            subprocess.run([py, str(ROOT / 'scripts' / builder)], cwd=ROOT, check=True)
    print('Completed training, sampling, and metrics', flush=True)


if __name__ == '__main__':
    main()
