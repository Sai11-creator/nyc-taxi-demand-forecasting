"""Reproduce the full-2024 / 2025 comparison from local processed data.

Run from any directory with the project's Python environment. No notebooks or
parquets are edited. Models are fitted in memory; results go to --output.
"""
import argparse
import json
import platform
import importlib.metadata
import os
from pathlib import Path
import sys

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/taxi-matplotlib")

import geopandas as gpd
import numpy as np
import pandas as pd
from arch.bootstrap import MovingBlockBootstrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--lambda-reg", type=float, default=0.3, help="Shrinkage toward zero (default: 0.3)")
    parser.add_argument("--output", type=Path, default=Path("/private/tmp/taxi_weighted_reproduction"))
    args = parser.parse_args()
    root = args.project.resolve()
    sys.path.insert(0, str(root / "notebooks"))
    from data_tools import model

    args.output.mkdir(parents=True, exist_ok=True)
    data = {
        year: pd.read_parquet(root / f"data/yellow_taxi_{year}_features_neighbor_mean.parquet")
        for year in [2024, 2025]
    }
    keys = ["tpep_pickup_datetime", "PULocationID"]
    for year, frame in data.items():
        assert not frame.duplicated(keys).any()
        assert frame.tpep_pickup_datetime.dt.year.eq(year).all()
        assert len(frame) == (366 if year == 2024 else 365) * 24 * 263

    tau, shrinkage = 1.0, args.lambda_reg
    if not np.isfinite(shrinkage) or shrinkage < 0:
        raise ValueError("lambda-reg must be finite and non-negative")
    means = data[2024].loc[data[2024].tpep_pickup_datetime.dt.month <= 8].groupby("PULocationID").trip_count.mean()
    zones = gpd.read_file(root / "taxi_zones/taxi_zones.shp").set_index("LocationID").sort_index()
    ids = pd.Index(range(1, 264), name="PULocationID")
    weights = pd.DataFrame(0.0, index=ids, columns=ids)
    for zone in ids:
        shared = zones.boundary.intersection(zones.loc[zone].geometry.boundary)
        neighbors = zones.index[(~shared.is_empty) & (zones.index != zone)]
        weights.loc[zone, neighbors] = np.exp(-np.abs(np.log1p(means[zone]) - np.log1p(means.loc[neighbors])) / tau)
    matrix = weights.to_numpy()
    totals = matrix.sum(axis=1)
    connected = totals > 0
    history = pd.concat([data[2024][keys + ["trip_count"]], data[2025][keys + ["trip_count"]]])
    counts = history.pivot(index=keys[0], columns=keys[1], values="trip_count").sort_index().reindex(columns=ids)
    weighted = {}
    for year, frame in data.items():
        hours = pd.DatetimeIndex(sorted(frame.tpep_pickup_datetime.unique()))
        previous = counts.reindex(hours - pd.Timedelta(hours=1)).to_numpy()
        signal = np.nan_to_num(previous, nan=0.0) @ matrix.T
        signal[:, connected] /= totals[connected] + shrinkage
        missing = np.isnan(previous).astype(float) @ (matrix > 0).T.astype(float) > 0
        signal[missing] = np.nan
        signal[:, ~connected] = 0
        weighted[year] = frame.copy()
        weighted[year]["weights"] = signal[hours.get_indexer(frame[keys[0]]), ids.get_indexer(frame[keys[1]])]

    y = data[2025].trip_count.to_numpy()
    errors = {}
    metadata = {"tau": tau, "lambda_reg": shrinkage, "formula": "sum(w * previous_demand) / (sum(w) + lambda)", "weighted_feature_adds_to_original_spatial": True, "mean_reference": "January-August 2024 (static)", "training": "January-December 2024", "evaluation": "January-December 2025", "raw_predictions": True, "bootstrap": {"blocks_days": [7, 14, 28], "reps": 5000, "seed": 42, "method": "percentile", "size": 0.95}, "python": platform.python_version(), "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "xgboost", "geopandas", "arch"]}, "models": {}}
    variants = {
        "basic": {year: frame.drop(columns="neighbors_mean_PU") for year, frame in data.items()},
        "spatial": data,
        "weights": weighted,
    }
    for name, frames in variants.items():
        train = frames[2024].drop(columns=["trip_count", keys[0]])
        test = frames[2025].drop(columns=["trip_count", keys[0]])
        assert train.columns.equals(test.columns)
        fitted = model(train).fit(train, frames[2024].trip_count)
        metadata["models"][name] = {"training_rows": len(train), "evaluation_rows": len(test), "input_columns": train.columns.tolist(), "transformed_columns": fitted.named_steps["columntransformer"].get_feature_names_out().tolist(), "booster_config": json.loads(fitted.named_steps["xgbregressor"].get_booster().save_config())}
        errors[name] = fitted.predict(test).astype(float) - y
        print(f"{name}: MAE={np.abs(errors[name]).mean():.6f}; RMSE={np.sqrt(np.square(errors[name]).mean()):.6f}", flush=True)

    seven = [2, 53, 124, 194, 199, 203, 219]
    group_ids = [("All zones", list(ids)), ("Seven zones", seven)] + [(str(z), [z]) for z in seven]
    metrics, intervals = [], []
    frame = data[2025]
    loss = pd.DataFrame({
        "date": frame[keys[0]].dt.normalize(), "zone": frame[keys[1]],
        "AE": np.abs(errors["spatial"]) - np.abs(errors["weights"]),
        "SE": errors["spatial"]**2 - errors["weights"]**2,
    })
    for label, members in group_ids:
        mask = frame[keys[1]].isin(members).to_numpy()
        row = {"group": label, "zone_name": zones.loc[int(label), "zone"] if label.isdigit() else label}
        for name, error in errors.items():
            row[name + "_mae"] = np.abs(error[mask]).mean()
            row[name + "_mse"] = np.square(error[mask]).mean()
            row[name + "_rmse"] = np.sqrt(row[name + "_mse"])
        row["mae_reduction_pct"] = 100 * (1 - row["weights_mae"] / row["spatial_mae"])
        row["mse_reduction_pct"] = 100 * (1 - row["weights_mse"] / row["spatial_mse"])
        metrics.append(row)
        daily = loss.loc[mask].groupby("date")[["AE", "SE"]].mean().sort_index()
        assert daily.index.equals(pd.date_range("2025-01-01", "2025-12-31"))
        for measure in ["AE", "SE"]:
            for block in [7, 14, 28]:
                values = daily[measure].to_numpy()
                lo, hi = MovingBlockBootstrap(block, values, seed=42).conf_int(np.mean, reps=5000, method="percentile", size=.95).ravel()
                intervals.append({"group": label, "loss": measure, "block_days": block, "difference": values.mean(), "lower": lo, "upper": hi})
    pd.DataFrame(metrics).to_csv(args.output / "weighted_2025_zone_metrics.csv", index=False)
    pd.DataFrame(intervals).to_csv(args.output / "weighted_2025_bootstrap.csv", index=False)
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    randalls = frame.PULocationID.eq(194).to_numpy()
    diagnostic = {}
    for name, error in errors.items():
        prediction = error + y
        diagnostic[name] = {"mean_prediction": float(prediction[randalls].mean()), "bias": float(error[randalls].mean()), "mean_prediction_zero_hours": float(prediction[randalls & (y == 0)].mean())}
    (args.output / "randalls_bias.json").write_text(json.dumps(diagnostic, indent=2))
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
