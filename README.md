# K-ProtoDiff-Plus

Time-series generation with multiscale key prototypes and adaptive reflection sampling.

## Installation

Use Python 3.11 and an NVIDIA CUDA GPU for training and generation. Run commands
from the repository root. The dependency versions below follow the source
project's environment; GPU training has not been revalidated during repository cleanup.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell.
The CUDA wheel must be compatible with your GPU and driver. Legacy MuJoCo
and JacobiKAN helpers additionally require `dm-control` and `torchinfo`,
respectively; the standard configurations do not use them.

## Data

```bash
python scripts/download_paper_datasets.py
python scripts/download_fmri_dataset.py
```

The first command downloads the public CSV/EEG datasets; the second downloads
fMRI separately. Files are saved under `Data/datasets/`. Dataset access and use
remain subject to their upstream terms. Downloads require network access.

`Config/journal/` contains one standard configuration per dataset: stocks,
energy, EEG, ETTh, electricity, exchange, illness, traffic, weather, and fMRI.
`Config/sines_journal.yaml` generates synthetic sine data without a download.
These are standard configurations, not a promise to reproduce tuned paper results.

## Train and generate

For stocks:

```bash
python run.py --name stocks --config_file Config/journal/stocks.yaml --gpu 0 --seed 2026 --train
python run.py --name stocks --config_file Config/journal/stocks.yaml --gpu 0 --seed 2026 --milestone 10
```

The stocks configuration trains for 12,000 optimizer steps (`max_epochs` in the
configuration) and saves every 1,200 steps. Checkpoint 10 is therefore the final
checkpoint. For another configuration, use `max_epochs / save_cycle` as the final
milestone. Checkpoints go to `solver.results_folder` with the sequence length
appended, for example `checkpoints/journal/Checkpoints_journal_stocks_24/`.
Generated normalized samples are saved to `OUTPUT/stocks/ddpm_fake_stocks.npy`.

For a short synthetic-data smoke run:

```bash
python run.py --name sine_smoke --config_file Config/sines_journal.yaml --gpu 0 --train solver.max_epochs 2 solver.save_cycle 1 dataloader.train_dataset.params.num 256
python run.py --name sine_smoke --config_file Config/sines_journal.yaml --gpu 0 --milestone 2 dataloader.train_dataset.params.num 256
```

Change the configuration for a different dataset. Put dotted configuration
overrides at the end of a command. For independent seeds, change both `--name`
and `solver.results_folder` so runs do not overwrite each other's outputs or
checkpoints. `--seed` controls model randomness; dataset splitting uses the seed
in the dataset configuration where supported.

## Evaluate

```bash
python -m pip install -r requirements-eval.txt
python scripts/eval_seeded.py --root OUTPUT/stocks/evaluation --ori_path OUTPUT/stocks/samples/stocks_norm_truth_24_train.npy --fake_path OUTPUT/stocks/ddpm_fake_stocks.npy --seed 2026
```

Inputs must have shape `(samples, time, features)` and use the same normalization.
The example compares generated samples against training data; it is not a held-out
evaluation protocol. The evaluator reports Context-FID, KL, discriminative and
predictive scores, and segment-wise DTW. Context-FID uses CUDA device 0. DS and PS
require TensorFlow; omit them with `--skip ds ps` if it is not installed.
The evaluator's KL and segment-wise DTW implementations are reference versions
and may differ from the paper's definitions; check the protocol before comparing
published numbers.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests check configuration consistency, frequency reconstruction, prototype
gradients, and the adaptive sampler's evaluation budget. They do not require
downloaded datasets. Core model tests can run on CPU.

## Layout

- `Models/`, `Layer/`: model and training implementation.
- `Data/`, `Utils/`: data loading and evaluation utilities.
- `Config/`: standard dataset configurations and synthetic example.
- `scripts/`: dataset downloads, seeded evaluation, and baseline restoration.
- `tests/`: core model and configuration checks.
- `baseline_patches/`, `baselines.lock.json`: optional pinned baseline sources.

To restore external baselines, run `python scripts/restore_baselines.py` once.
This creates `baselines/`; use each upstream project's instructions and environment
to run it. The restore script refuses to overwrite existing repositories.

## Attribution and licensing

This implementation includes code derived from other research projects, including
Diffusion-TS and TS2Vec. Retain source attribution and consult the corresponding
upstream licenses. External baseline repositories retain their own licenses.
A project-wide license has not yet been selected by the maintainer.
