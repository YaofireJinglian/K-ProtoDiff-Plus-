# K-ProtoDiff+ benchmark data

Public source and download index for the ten datasets used by the K-ProtoDiff+ configurations. These are third-party reference datasets.

**Availability:** This directory contains source links and preparation instructions. Raw experiment-server files have not been deposited here. Fresh downloads have not been compared byte-for-byte with the files used for the reported experiments.

## Download

Run from the repository root:

```bash
python scripts/download_paper_datasets.py
python scripts/download_fmri_dataset.py
```

The first script prepares nine datasets and writes `Data/datasets/manifest.json` containing file sizes, CSV dimensions, and SHA-256 hashes. The fMRI script verifies the downloaded archive and MAT files against embedded hashes.

## Sources

| Dataset | Download | Expected local path |
| --- | --- | --- |
| ETTh | [Source file](https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv) | `Data/datasets/ETTh.csv` |
| Electricity | [Source file](https://huggingface.co/datasets/AutonLab/Timeseries-PILE/resolve/main/forecasting/autoformer/electricity.csv?download=true) | `Data/datasets/electricity.csv` |
| Traffic | [Source file](https://huggingface.co/datasets/AutonLab/Timeseries-PILE/resolve/main/forecasting/autoformer/traffic.csv?download=true) | `Data/datasets/traffic.csv` |
| Weather | [Source file](https://huggingface.co/datasets/AutonLab/Timeseries-PILE/resolve/main/forecasting/autoformer/weather.csv?download=true) | `Data/datasets/weather.csv` |
| Illness | [Source file](https://huggingface.co/datasets/AutonLab/Timeseries-PILE/resolve/main/forecasting/autoformer/national_illness.csv?download=true) | `Data/datasets/national_illness.csv` |
| Exchange | [Source file](https://huggingface.co/datasets/AutonLab/Timeseries-PILE/resolve/main/forecasting/autoformer/exchange_rate.csv?download=true) | `Data/datasets/exchange_rate.csv` |
| Stocks | [Source file](https://raw.githubusercontent.com/jsyoon0823/TimeGAN/master/data/stock_data.csv) | `Data/datasets/stock_data.csv` |
| Energy | [Source file](https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip) | `Data/datasets/energy_data.csv` |
| EEG | [Source file](https://archive.ics.uci.edu/static/public/264/eeg+eye+state.zip) | `Data/datasets/EEG_Eye_State.arff` |
| fMRI | [Source file](https://drive.google.com/uc?id=11DI22zKWtHjXMnNGPWNUbyGz-JiEtZy6) | `Data/datasets/fMRI/sim4.mat` |

Machine-readable index: [sources.json](sources.json).

## Preparation and attribution

- **ETTh:** Hourly ETTh1 from [ETDataset](https://github.com/zhouhaoyi/ETDataset). The loader excludes the timestamp.
- **Electricity, Traffic, Weather, Illness, Exchange:** Prepared [Autoformer](https://github.com/thuml/Autoformer) CSVs served through [Timeseries-PILE](https://huggingface.co/datasets/AutonLab/Timeseries-PILE).
- **Stocks:** Stock benchmark CSV from [TimeGAN](https://github.com/jsyoon0823/TimeGAN).
- **Energy:** Candanedo, L. (2017), [Appliances Energy Prediction](https://doi.org/10.24432/C5VC8G), UCI Machine Learning Repository. The script extracts `energydata_complete.csv`, retains 28 numeric columns, and saves without the timestamp or row index.
- **EEG:** Roesler, O. (2013), [EEG Eye State](https://doi.org/10.24432/C57G7J), UCI Machine Learning Repository. The script extracts the ARFF file; the model loader applies filtering and segmentation.
- **fMRI:** Simulated BOLD benchmark distributed by [Diffusion-TS](https://github.com/Y-debug-sys/Diffusion-TS#dataset-preparation). The configuration uses `sim4.mat`, whose `ts` array has shape `(10000, 50)`.

Energy and EEG are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Each dataset retains its upstream terms; the code license does not replace data licenses.
