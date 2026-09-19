"""Restore pinned upstream baseline repositories and apply journal patches."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    for entry in json.loads((ROOT / 'baselines.lock.json').read_text()):
        target = ROOT / 'baselines' / entry['name']
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite {target}')
        subprocess.run(['git', 'clone', entry['url'], str(target)], check=True)
        subprocess.run(['git', 'checkout', '--detach', entry['revision']], cwd=target, check=True)
        if entry.get('patch'):
            patch = str(ROOT / entry['patch'])
            subprocess.run(['git', 'apply', '--check', patch], cwd=target, check=True)
            subprocess.run(['git', 'apply', patch], cwd=target, check=True)


if __name__ == '__main__':
    main()
