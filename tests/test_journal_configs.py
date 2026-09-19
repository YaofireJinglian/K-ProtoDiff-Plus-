from pathlib import Path
import unittest

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class JournalConfigTests(unittest.TestCase):
    DATASETS = {
        'etth': ('ETTh.csv', 7),
        'electricity': ('electricity.csv', 321),
        'traffic': ('traffic.csv', 862),
        'weather': ('weather.csv', 21),
        'illness': ('national_illness.csv', 7),
        'exchange': ('exchange_rate.csv', 8),
        'stocks': ('stock_data.csv', 6),
        'energy': ('energy_data.csv', 28),
        'eeg': ('EEG_Eye_State.arff', 14),
        'fmri': ('fMRI', 50),
    }

    def test_every_paper_dataset_has_a_valid_journal_config(self):
        config_root = REPOSITORY_ROOT / 'Config' / 'journal'
        self.assertEqual(
            {path.stem for path in config_root.glob('*.yaml')},
            set(self.DATASETS),
        )

        for name, (filename, feature_size) in self.DATASETS.items():
            with self.subTest(dataset=name):
                config = yaml.safe_load((config_root / f'{name}.yaml').read_text())
                model = config['model']['params']
                dataset = config['dataloader']['train_dataset']['params']
                data_path = REPOSITORY_ROOT / dataset['data_root']

                self.assertEqual(data_path.name, filename)
                self.assertEqual(model['feature_size'], feature_size)
                self.assertEqual(model['prototype_mode'], 'multiscale')
                self.assertEqual(model['sampling_mode'], 'adaptive_reflection')
                self.assertEqual(model['fdm_mode'], 'real')

                self.assertEqual(model['seq_length'], dataset['window'])
                self.assertTrue(all(0 < scale <= model['seq_length']
                                    for scale in model['prototype_scales']))
                self.assertGreater(config['solver']['save_cycle'], 0)
                self.assertEqual(config['solver']['max_epochs'] %
                                 config['solver']['save_cycle'], 0)
                for section in (config['model'],
                                config['dataloader']['train_dataset'],
                                config['solver']['scheduler']):
                    module = section['target'].rsplit('.', 1)[0]
                    self.assertTrue((REPOSITORY_ROOT /
                                     (module.replace('.', '/') + '.py')).is_file())


if __name__ == '__main__':
    unittest.main()
