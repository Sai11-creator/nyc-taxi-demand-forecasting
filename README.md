# NYC Taxi Demand Forecasting

A reproducible machine learning project for forecasting hourly taxi pickup demand across New York City taxi zones, using real-world NYC TLC trip records.

## Research question

**How accurately can we forecast the next hour's taxi pickups in each NYC taxi zone using information available at prediction time, and which predictive signals provide robust out-of-sample improvements?**

## Data sources

- [Official NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- [Yellow taxi trips — January 2024 (Parquet)](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet)
- [Taxi zone lookup (CSV)](https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv)
- [Yellow taxi data dictionary](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf)

**Data is not committed to Git.** Download source files into `data/raw/`, and keep all intermediate and processed data under `data/`.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
jupyter lab
```

## Project roadmap

1. **Data audit** — inspect schema, missingness, invalid timestamps, implausible trips, and unusual values. Quantify issues before filtering.
2. **Exploratory analysis** — investigate hourly, weekly, and spatial seasonality; visualize variations by zone.
3. **Target creation** — aggregate pickups by `(taxi_zone, hour)` and explicitly represent zero-count hours.
4. **Validation and baselines** — split chronologically; compare seasonal naive forecasts and historical-average models.
5. **Predictive models** — evaluate count regression and tree-based models, using only features known at forecast time.
6. **Robustness** — test stability across months and spatial zones; assess uncertainty and distribution shift.

## Initial milestone

Use only the **January 2024** yellow taxi file at first. We will work on:

- dataset size and columns
- missingness per column
- distribution of trip durations and distances
- invalid and suspicious records with counts and percentages
- proposed cleaning rules and why each rule is justified


## Repository structure

```text
notebooks/        exploratory notebooks
src/              reusable data and modelling code
reports/figures/  selected small plots for reports/README
data/             locally stored raw/processed datasets (gitignored)
```
