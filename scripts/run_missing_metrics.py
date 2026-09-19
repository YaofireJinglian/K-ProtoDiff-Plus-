"""Evaluate DS, PS, and Segment-wise DTW for completed journal runs."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)


def parse_metric_file(path):
    values = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        key, value = line.split(':', 1)
        values[key.strip()] = float(value.strip())
    return values


def update_tables(record_path, supplemental_path):
    record = json.loads(record_path.read_text(encoding='utf-8'))
    values = parse_metric_file(supplemental_path)
    record['metrics'].update({
        'DS': values['DS_mean'],
        'PS': values['PS_mean'],
        'Seg-DTW-L6': values['Segment-DTW-L6'],
        'Seg-DTW-L8': values['Segment-DTW-L8'],
        'Seg-DTW-L10': values['Segment-DTW-L10'],
    })
    temporary = record_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, record_path)

    lock_path = ROOT / 'OUTPUT' / '.main-results.lock'
    with lock_path.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        subprocess.run(
            [PYTHON, ROOT / 'scripts' / 'build_main_results_table.py'],
            cwd=ROOT, stdout=subprocess.DEVNULL, check=True,
        )
        subprocess.run(
            [PYTHON, ROOT / 'scripts' / 'build_segment_dtw_table.py'],
            cwd=ROOT, stdout=subprocess.DEVNULL, check=True,
        )


def run_job(dataset, seed, worker):
    config_path = ROOT / 'Config' / 'journal' / f'{dataset}.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    params = config['dataloader']['train_dataset']['params']
    data_name = params.get('name', dataset)
    window = params['window']
    run_name = f'journal_{dataset}_seed{seed}'
    output_dir = ROOT / 'OUTPUT' / 'formal' / run_name
    real_path = output_dir / 'samples' / f'{data_name}_norm_truth_{window}_train.npy'
    fake_path = output_dir / f'ddpm_fake_{run_name}.npy'
    metric_dir = ROOT / 'OUTPUT' / 'main_metrics' / f'{dataset}_seed{seed}'
    supplemental = metric_dir / 'supplemental_metrics.txt'
    record = ROOT / 'OUTPUT' / 'main_results_records' / f'{dataset}__k_protodiff_j__seed{seed}.json'
    if not record.exists() or not real_path.exists() or not fake_path.exists():
        raise FileNotFoundError(f'{run_name}: training/sampling record is incomplete')

    if not supplemental.exists():
        log_dir = ROOT / 'OUTPUT' / 'full_metric_logs' / worker
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f'{run_name}.log'
        command = [
            PYTHON, ROOT / 'metric_pytorch.py',
            '--root', metric_dir,
            '--ori_path', real_path,
            '--fake_path', fake_path,
            '--n_iter', '1',
            '--seed', str(seed),
            '--output-name', supplemental.name,
            '--skip', 'cfid', 'kl',
        ]
        environment = os.environ.copy()
        environment.update({
            'TF_CPP_MIN_LOG_LEVEL': '2',
            'OMP_NUM_THREADS': '8',
            'TF_NUM_INTRAOP_THREADS': '8',
            'TF_NUM_INTEROP_THREADS': '2',
        })
        print(f'{run_name}: evaluating DS, PS, and Segment-wise DTW', flush=True)
        with log_path.open('a', encoding='utf-8') as log:
            subprocess.run(
                command, cwd=ROOT, env=environment,
                stdout=log, stderr=subprocess.STDOUT, check=True,
            )
    update_tables(record, supplemental)
    print(f'{run_name}: five-metric record complete', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('jobs', nargs='+', help='Jobs formatted as dataset:seed')
    parser.add_argument('--worker', required=True)
    args = parser.parse_args()
    failures = []
    for item in args.jobs:
        dataset, raw_seed = item.split(':', 1)
        try:
            run_job(dataset, int(raw_seed), args.worker)
        except Exception as error:
            failures.append((item, repr(error)))
            print(f'{item}: FAILED: {error!r}', flush=True)
    if failures:
        raise RuntimeError(f'Metric queue completed with failures: {failures}')


if __name__ == '__main__':
    main()
