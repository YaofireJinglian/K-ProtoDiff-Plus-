"""Remove reproducible artifacts while preserving selected formal weights and records."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ['fmri', 'energy', 'traffic', 'electricity', 'etth',
            'weather', 'illness', 'exchange', 'stocks', 'eeg']
SEEDS = [2026, 2027, 2028]


def selected_weights():
    import yaml
    manifest = json.loads((ROOT / 'Config/journal_tuned/manifest.json').read_text())
    keep = set()
    for dataset in DATASETS:
        meta = manifest['datasets'][dataset]
        config = yaml.safe_load((ROOT / 'Config/journal_tuned' / f'{dataset}.yaml').read_text())
        window = config['dataloader']['train_dataset']['params']['window']
        for seed in SEEDS:
            if meta['reuse_original_checkpoint']:
                path = (ROOT / 'checkpoints/journal' /
                        f'Checkpoints_journal_{dataset}_seed{seed}_{window}/checkpoint-10.pt')
            else:
                path = (ROOT / 'checkpoints/journal_tuned' /
                        f'Checkpoints_journal_tuned_{dataset}_seed{seed}_{window}/checkpoint-10.pt')
            keep.add(path.resolve())
    return keep


def stale_search_files():
    paths = []
    roots = list((ROOT / 'checkpoints').glob('journal_search*'))
    roots += list((ROOT / 'checkpoints').glob('journal_confirmation_seed*'))
    for root in roots:
        paths.extend(path for path in root.rglob('*')
                     if path.is_file() and path.suffix in {'.pt', '.pth', '.npy'})
    for path in (ROOT / 'checkpoints/journal').rglob('checkpoint-*.pt'):
        if path.name != 'checkpoint-10.pt':
            paths.append(path)
    return paths


def verified_release():
    manifest_path = ROOT / 'WEIGHTS_MANIFEST.json'
    if not manifest_path.exists():
        raise RuntimeError('WEIGHTS_MANIFEST.json is missing; weights were not verified as uploaded')
    manifest = json.loads(manifest_path.read_text())
    expected = {item['asset']: item['bytes'] for item in manifest['assets']}
    expected['WEIGHTS_MANIFEST.json'] = None
    gh = ROOT / '.tools/gh-apt/root/usr/bin/gh'
    import subprocess
    remote = json.loads(subprocess.check_output(
        [gh, 'release', 'view', manifest['release_tag'], '--repo', manifest['repository'],
         '--json', 'assets'], text=True))
    actual = {item['name']: item['size'] for item in remote['assets']}
    for name, size in expected.items():
        if name not in actual or (size is not None and actual[name] != size):
            raise RuntimeError(f'Release asset is absent or invalid: {name}')


def final_files():
    for dataset in DATASETS:
        for seed in SEEDS:
            if not (ROOT / 'OUTPUT/tuned_main_records' /
                    f'{dataset}__k_protodiff_j__seed{seed}.json').exists():
                raise RuntimeError('Tuned formal records are incomplete; final cleanup is unsafe')
    keep = selected_weights()
    paths = stale_search_files()
    for root in [ROOT / 'checkpoints/journal', ROOT / 'checkpoints/journal_tuned']:
        paths.extend(path for path in root.rglob('*.pt') if path.resolve() not in keep)
    for root in [ROOT / 'OUTPUT/formal', ROOT / 'OUTPUT/formal_tuned']:
        if root.exists():
            paths.extend(root.rglob('*.npy'))
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=['search', 'final'], required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if args.scope == 'final':
        verified_release()
        paths = final_files()
    else:
        paths = stale_search_files()
    unique = sorted({path.resolve() for path in paths if path.exists()})
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'scope': args.scope,
        'applied': args.apply, 'files': len(unique),
        'bytes': sum(path.stat().st_size for path in unique),
        'paths': [str(path.relative_to(ROOT)) for path in unique],
        'preserved': ('All active baseline/ablation artifacts and every selected checkpoint-10 '
                      'used by the frozen formal configuration.'),
    }
    report_path = ROOT / 'OUTPUT' / f'cleanup_{args.scope}_report.json'
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    if args.apply:
        for path in unique:
            path.unlink()
    print(json.dumps({key: report[key] for key in ['scope', 'applied', 'files', 'bytes']}))


if __name__ == '__main__':
    main()
