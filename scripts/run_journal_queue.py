"""Run journal training, sampling, and lightweight metrics as a persistent queue."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
LABELS = {
    'etth': 'ETTh',
    'electricity': 'Electricity',
    'energy': 'Energy',
    'traffic': 'Traffic',
    'weather': 'Weather',
    'illness': 'Illness',
    'exchange': 'Exchange',
    'stocks': 'Stocks',
    'eeg': 'EEG',
    'fmri': 'fMRI',
}


def wait_for_process(pid):
    if pid is None:
        return
    print(f'Waiting for PID {pid} to finish...', flush=True)
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(20)


def run_logged(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Running {" ".join(map(str, command))}', flush=True)
    with log_path.open('a', encoding='utf-8') as log:
        log.write(f'\nCOMMAND: {" ".join(map(str, command))}\n')
        log.flush()
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def update_record(dataset, seed, metrics_path):
    values = {}
    for line in metrics_path.read_text(encoding='utf-8').splitlines():
        key, value = line.split(':', 1)
        values[key.strip()] = float(value.strip())
    record = {
        'dataset': LABELS[dataset],
        'method': 'K-ProtoDiff-J',
        'seed': seed,
        'metrics': {
            'C-FID': values['Context-FID'],
            'KL': values['KL'],
            'DS': values.get('DS_mean'),
            'PS': values.get('PS_mean'),
            'Seg-DTW-L6': values.get('Segment-DTW-L6'),
            'Seg-DTW-L8': values.get('Segment-DTW-L8'),
            'Seg-DTW-L10': values.get('Segment-DTW-L10'),
        },
    }
    record_dir = ROOT / 'OUTPUT' / 'main_results_records'
    record_dir.mkdir(parents=True, exist_ok=True)
    destination = record_dir / f'{dataset}__k_protodiff_j__seed{seed}.json'
    temporary = destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, destination)

    lock_path = ROOT / 'OUTPUT' / '.main-results.lock'
    with lock_path.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        subprocess.run(
            [PYTHON, ROOT / 'scripts' / 'build_main_results_table.py'],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            check=True,
        )


def run_job(dataset, seed, log_root):
    config_path = ROOT / 'Config' / 'journal' / f'{dataset}.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    data_params = config['dataloader']['train_dataset']['params']
    data_name = data_params.get('name', dataset)
    window = data_params['window']
    run_name = f'journal_{dataset}_seed{seed}'
    checkpoint_base = str(ROOT / 'checkpoints' / 'journal' / f'Checkpoints_journal_{dataset}_seed{seed}')
    checkpoint = Path(f'{checkpoint_base}_{window}') / 'checkpoint-10.pt'
    output_dir = ROOT / 'OUTPUT' / 'formal' / run_name
    fake_path = output_dir / f'ddpm_fake_{run_name}.npy'
    real_path = output_dir / 'samples' / f'{data_name}_norm_truth_{window}_train.npy'
    metrics_dir = ROOT / 'OUTPUT' / 'main_metrics' / f'{dataset}_seed{seed}'
    metrics_path = metrics_dir / 'metrics.txt'

    common = [
        PYTHON, '-u', ROOT / 'run.py',
        '--name', run_name,
        '--config_file', config_path,
        '--output', ROOT / 'OUTPUT' / 'formal',
        '--gpu', '0',
        '--seed', str(seed),
    ]
    if not checkpoint.exists():
        run_logged(
            common + ['--train', 'solver.results_folder', checkpoint_base],
            log_root / f'{run_name}_train.log',
        )
    else:
        print(f'{run_name}: checkpoint already complete', flush=True)

    if not fake_path.exists() or not real_path.exists():
        run_logged(
            common + ['--milestone', '10', 'solver.results_folder', checkpoint_base],
            log_root / f'{run_name}_sample.log',
        )
    else:
        print(f'{run_name}: samples already complete', flush=True)

    if not metrics_path.exists():
        run_logged(
            [
                PYTHON, ROOT / 'metric_pytorch.py',
                '--root', metrics_dir,
                '--ori_path', real_path,
                '--fake_path', fake_path,
                '--skip', 'ds', 'ps', 'dtw',
            ],
            log_root / f'{run_name}_metrics.log',
        )
    update_record(dataset, seed, metrics_path)
    print(f'{run_name}: fully complete', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('jobs', nargs='+', help='Jobs formatted as dataset:seed')
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--worker', required=True)
    args = parser.parse_args()

    wait_for_process(args.wait_pid)
    log_root = ROOT / 'OUTPUT' / 'journal_queue_logs' / args.worker
    failures = []
    for job in args.jobs:
        dataset, raw_seed = job.split(':', 1)
        try:
            run_job(dataset, int(raw_seed), log_root)
        except Exception as error:
            failures.append((job, repr(error)))
            print(f'{job}: FAILED: {error!r}', flush=True)
    if failures:
        raise RuntimeError(f'Queue completed with failures: {failures}')
    print(f'Worker {args.worker}: queue complete', flush=True)


if __name__ == '__main__':
    main()
