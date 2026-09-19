"""Push a compact GitHub snapshot whenever an experiment completion appears."""

import argparse
import fcntl
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / 'OUTPUT' / 'github_result_sync'


def result_files():
    paths = []
    for folder in ['main_results_records', 'tuned_main_records']:
        root = ROOT / 'OUTPUT' / folder
        if root.exists():
            paths.extend(root.glob('*.json'))
    ablation = ROOT / 'checkpoints' / 'journal_ablation'
    if ablation.exists():
        paths.extend(ablation.glob('*/*/complete.json'))
    return sorted(set(paths))


def fingerprint(path):
    stat = path.stat()
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'size': stat.st_size, 'sha256': value}


def snapshot():
    return {str(path.relative_to(ROOT)): fingerprint(path) for path in result_files()}


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def sync(changed):
    labels = [Path(path).stem for path in changed]
    summary = ', '.join(labels[:4])
    if len(labels) > 4:
        summary += f' and {len(labels) - 4} more'
    message = f'Update completed experiment: {summary}'
    subprocess.run([ROOT / '.venv/bin/python', '-u',
                    ROOT / 'scripts/sync_journal_release.py', '--message', message],
                   cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poll-seconds', type=int, default=10)
    parser.add_argument('--initialize', action='store_true')
    args = parser.parse_args()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_path = STATE_DIR / 'state.json'
    with (STATE_DIR / 'watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = snapshot()
        if args.initialize or not state_path.exists():
            save(state_path, {'updated_utc': datetime.now(timezone.utc).isoformat(),
                              'files': current})
            print(f'Initialized with {len(current)} completed artifacts', flush=True)
            if args.initialize:
                return
        while True:
            state = json.loads(state_path.read_text())
            previous = state.get('files', {})
            current = snapshot()
            changed = [path for path, value in current.items()
                       if previous.get(path) != value]
            if changed:
                print('Completion detected:', changed, flush=True)
                try:
                    sync(changed)
                except Exception as error:
                    print(f'Git sync failed; will retry: {error!r}', flush=True)
                    time.sleep(max(10, args.poll_seconds))
                    continue
                save(state_path, {'updated_utc': datetime.now(timezone.utc).isoformat(),
                                  'last_synced': changed, 'files': current})
                print(f'Synced {len(changed)} completion(s)', flush=True)
            time.sleep(max(2, args.poll_seconds))


if __name__ == '__main__':
    main()
