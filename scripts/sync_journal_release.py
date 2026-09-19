"""Export the current compact snapshot and push it to the journal repository."""

import argparse
import fcntl
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ['fmri', 'energy', 'traffic', 'electricity', 'etth',
            'weather', 'illness', 'exchange', 'stocks', 'eeg']
SEEDS = [2026, 2027, 2028]


def require_complete():
    missing = []
    for dataset in DATASETS:
        for seed in SEEDS:
            path = ROOT / 'OUTPUT' / 'tuned_main_records' / f'{dataset}__k_protodiff_j__seed{seed}.json'
            if not path.exists():
                missing.append(f'{dataset}:{seed}')
    if missing:
        raise RuntimeError(f'Tuned formal records are incomplete: {missing}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', default='https://github.com/YaofireJinglian/KPD-Journal.git')
    parser.add_argument('--message', default='Update completed tuned journal results')
    parser.add_argument('--require-tuned-complete', action='store_true')
    args = parser.parse_args()
    if args.require_tuned_complete:
        require_complete()
    lock_path = ROOT / 'OUTPUT' / '.github-sync.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with tempfile.TemporaryDirectory(prefix='kpd-journal-sync-') as folder:
            temporary = Path(folder)
            checkout, snapshot = temporary / 'checkout', temporary / 'snapshot'
            subprocess.run(['git', 'clone', '--branch', 'main', '--single-branch',
                            args.repository, checkout], check=True)
            subprocess.run([ROOT / '.venv' / 'bin' / 'python',
                            ROOT / 'scripts' / 'export_journal_release.py', snapshot],
                           cwd=ROOT, check=True)
            oversized = [path for path in snapshot.rglob('*')
                         if path.is_file() and path.stat().st_size > 95 * 1024 * 1024]
            if oversized:
                raise ValueError(f'Refusing to push oversized files: {oversized}')
            subprocess.run(['rsync', '-a', '--delete', '--exclude=.git/',
                            f'{snapshot}/', f'{checkout}/'], check=True)
            subprocess.run(['git', 'add', '-A'], cwd=checkout, check=True)
            changed = subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=checkout)
            if changed.returncode == 0:
                print('GitHub snapshot is already current')
                return
            subprocess.run(['git', 'commit', '-m', args.message], cwd=checkout, check=True)
            subprocess.run(['git', 'push', 'origin', 'main'], cwd=checkout, check=True)


if __name__ == '__main__':
    main()
