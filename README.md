# NYC Taxi Demand Forecasting

Forecasting the next hour's yellow taxi pickups across **263 New York City taxi zones**, using 2024 and 2025 NYC TLC records.

The project investigates whether geographical neighbours, urban characteristics, and transport connections improve forecasts beyond a zone's own demand history. It compares temporal baselines, regression and XGBoost models, GRUs, and an ST-MGCN-inspired deep learning model.

## Approach

The notebooks cover data cleaning, complete hourly demand panels (including zero-count hours), feature engineering, model development, and error analysis. Spatial information is represented through:

- **Geography:** taxi zones whose boundaries intersect.
- **Functionality:** similarity between zones' restaurant and hotel counts.
- **Transport connectivity:** zones sharing a daytime subway route.

The final deep learning model combines these three graphs with contextual temporal gating, recurrent encoders, and learned graph filters. It is inspired by [Geng et al.'s ST-MGCN](https://ojs.aaai.org/index.php/AAAI/article/view/4247), rather than an exact reproduction. The final encoder uses a vanilla RNN; earlier experiments use GRUs.

Development trains on January–August 2024, validates on September, and tests on October–December. The selected configuration is then refitted on all of 2024 and evaluated on 2025. Predictions use the preceding observed demand, so this is **rolling one-hour-ahead forecasting**.

## Data

Inputs include [NYC TLC yellow taxi trip records and zone geography](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), historical OpenStreetMap restaurant/hotel counts, and the MTA's 2024 subway entrances and exits dataset.

Datasets, trained weights, generated results, and reports are kept locally and are **not included in this repository**. The standalone deep learning script expects:

```text
data/DL_Dataset.parquet
data/yellow_taxi_2025_features_neighbor_mean.parquet
taxi_zones/taxi_zones.shp              # with the accompanying shapefile files
MTA_Subway_Entrances_and_Exits__2024_20261021.csv
```

The notebooks document how the processed inputs and features are constructed. Some notebook paths reflect the original development machine and may need adjusting.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dl.txt
jupyter lab
```

Use a PyTorch build compatible with your machine's GPU/CUDA environment. Some exploratory notebooks also require XGBoost, `arch`, or OSMnx.

## Deep learning on Avon

The consolidated script handles development, validation-based epoch selection, full-2024 refitting, and 2025 evaluation:

```bash
python avon_deep_learning.py --stage all
```

You can also run `develop`, `refit`, or `evaluate` separately; see `python avon_deep_learning.py --help`. Results, learning histories, predictions, and checkpoints are written to `outputs/deep_learning/` by default.

For Avon's GPU scheduler, submit from the project directory:

```bash
sbatch avon_deep_learning.sbatch
```

The job file defaults to `.venv-avon-new/bin/python`; set `TAXI_PYTHON` to use another environment. Evaluation records both the original window starting on 2 January 2025 and the complete year, using December 2024 history for the first day's predictions.

## Repository guide

| Location | Contents |
|---|---|
| `notebooks/` | Data preparation, baselines, spatial experiments, and neural development |
| `notebooks/data_tools.py` | Shared data preparation and modelling helpers |
| `avon_deep_learning.py` | Standalone deep learning workflow |
| `avon_deep_learning.sbatch` | Avon GPU job configuration |
| `requirements*.txt` | Base and deep learning dependencies |
| `reports/research_report/*.py` | Saved-evidence verification and weighted-model reproduction |

## Research report

The detailed methodology, model comparisons, monthly and zone-level errors, uncertainty analysis, and limitations are documented in the research report. In the local project, it is available at `reports/research_report/taxi_demand_research_report.pdf`; the PDF and LaTeX sources are maintained separately from the code repository.

The report distinguishes validation results from later-year evaluation and discusses why improvements in citywide RMSE do not necessarily translate into better absolute errors or better forecasts for every zone.
