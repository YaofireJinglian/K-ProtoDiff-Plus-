import json
from pathlib import Path

import yaml

from scripts.prepare_tuned_main import DATASETS, prepare


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_tuned_configs_are_complete_and_keep_architecture():
    manifest_path = prepare(ROOT)
    manifest = json.loads(manifest_path.read_text())
    assert set(manifest['datasets']) == set(DATASETS)
    assert manifest['selection_split'] == 'validation'
    assert manifest['formal_seeds'] == [2026, 2027, 2028]
    for dataset, metadata in manifest['datasets'].items():
        base = yaml.safe_load((ROOT / 'Config/journal' / f'{dataset}.yaml').read_text())
        tuned = yaml.safe_load((ROOT / 'Config/journal_tuned' / f'{dataset}.yaml').read_text())
        assert tuned['solver']['max_epochs'] == base['solver']['max_epochs']
        assert tuned['solver']['save_cycle'] == base['solver']['save_cycle']
        assert metadata['screen_training_steps'] in {6000, 12000}
        for key in ['seq_length', 'feature_size', 'n_layer_enc', 'n_layer_dec',
                    'd_model', 'n_heads', 'prototype_scales', 'num_prototypes']:
            assert tuned['model']['params'][key] == base['model']['params'][key]


def test_only_energy_reuses_an_unchanged_record():
    manifest = json.loads(prepare(ROOT).read_text())
    reused = [dataset for dataset, value in manifest['datasets'].items()
              if value['reuse_original_record']]
    assert reused == ['energy']
