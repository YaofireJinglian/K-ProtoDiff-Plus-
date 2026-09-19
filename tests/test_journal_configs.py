from pathlib import Path
import unittest

import pandas as pd
from scipy import io
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

                self.assertTrue(data_path.exists())
                self.assertEqual(data_path.name, filename)
                self.assertEqual(model['feature_size'], feature_size)
                self.assertEqual(model['prototype_mode'], 'multiscale')
                self.assertEqual(model['sampling_mode'], 'adaptive_reflection')
                self.assertEqual(model['fdm_mode'], 'real')

                if name == 'fmri':
                    fmri = io.loadmat(data_path / 'sim4.mat')['ts']
                    self.assertEqual(fmri.shape, (10000, feature_size))
                elif name != 'eeg':
                    frame = pd.read_csv(data_path, nrows=4)
                    self.assertEqual(
                        frame.select_dtypes(include='number').shape[1],
                        feature_size,
                    )


if __name__ == '__main__':
    unittest.main()
