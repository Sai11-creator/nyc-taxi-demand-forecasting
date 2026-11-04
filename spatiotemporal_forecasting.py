"""Standalone ST-MGCN workflow for Avon (no other project Python modules required).

Run from the project directory after activating the Avon environment:
    python spatiotemporal_forecasting.py --stage all
    python spatiotemporal_forecasting.py --stage develop --epochs 100
    python spatiotemporal_forecasting.py --stage refit --epochs 92
    python spatiotemporal_forecasting.py --stage evaluate

Inputs: data/DL_Dataset.parquet, the 2025 neighbour-mean feature parquet,
        taxi_zones/taxi_zones.shp, and the 2024 MTA entrances CSV.
Outputs: outputs/deep_learning/ (override with --output-dir).
The all stage selects the refit epoch count using September validation only.
Evaluation saves both the original Jan 2-Dec 31 window and all of 2025.
The architecture and graph calculations match the verified Avon scripts.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def build_graphs(project):
    df_zones = gpd.read_file(project / "taxi_zones/taxi_zones.shp")
    df = pd.read_parquet(project / "data/DL_Dataset.parquet")
    df_subway = pd.read_csv(
        project / "MTA_Subway_Entrances_and_Exits__2024_20261021.csv"
    )
    if df_zones.LocationID.tolist() != list(range(1, 264)):
        raise ValueError("Zone geometry rows must be ordered by IDs 1-263")
    adjacency = np.zeros((263, 263))
    for i in range(263):
        boundary = df_zones.geometry.iloc[i].boundary
        shared_parts = df_zones.boundary.intersection(boundary)
        mask = ~shared_parts.is_empty
        neighbors = df_zones.loc[mask].drop(index=df_zones.index[i])
        for j in neighbors.LocationID:
            adjacency[i][j - 1] += 1
    adjacency = adjacency.astype(int)

    # Functional similarity from restaurant and hotel counts
    # One row per zone, ordered by zone ID
    poi = (
        df.groupby("PULocationID")[["restaurants", "hotels"]]
        .first()
        .reindex(range(1, 264))
        .to_numpy(dtype=float)
    )

    norms = np.linalg.norm(poi, axis=1, keepdims=True)

    # Normalize nonzero vectors; leave zero vectors as zeros
    poi_normalized = np.divide(
        poi,
        norms,
        out=np.zeros_like(poi),
        where=norms > 0,
    )

    functionnality = poi_normalized @ poi_normalized.T

    # Transportation connectivity

    zone_ids = pd.Index(range(1, 264), name="LocationID")

    # Coordinates use decimal commas in this CSV.
    subway = df_subway.copy()
    for column in ["Entrance Latitude", "Entrance Longitude"]:
        subway[column] = pd.to_numeric(
            subway[column].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )
    if subway[["Entrance Latitude", "Entrance Longitude"]].isna().any().any():
        raise ValueError("Missing or invalid subway entrance coordinates")

    entrances = gpd.GeoDataFrame(
        subway,
        geometry=gpd.points_from_xy(
            subway["Entrance Longitude"], subway["Entrance Latitude"]
        ),
        crs="EPSG:4326",
    ).to_crs(df_zones.crs)

    # Union geometry parts belonging to the same zone before the spatial join.
    zones = df_zones[["LocationID", "geometry"]].dissolve(by="LocationID").reset_index()
    assigned = gpd.sjoin(entrances, zones, how="left", predicate="intersects")
    unmatched = assigned.loc[assigned["LocationID"].isna()].copy()
    ambiguous = assigned.loc[assigned.index.duplicated(keep=False)].copy()
    if not ambiguous.empty:
        raise ValueError("Some entrances intersect multiple zones; inspect ambiguous")

    # Multiple entrances/stations on the same route in one zone count only once.
    zone_routes = assigned.dropna(subset=["LocationID", "Daytime Routes"]).copy()
    zone_routes["LocationID"] = zone_routes["LocationID"].astype(int)
    zone_routes["route"] = zone_routes["Daytime Routes"].str.split()
    zone_routes = (
        zone_routes.explode("route")
        .drop_duplicates(["LocationID", "route"])
        .reset_index(drop=True)
    )
    route_presence = pd.crosstab(zone_routes["LocationID"], zone_routes["route"])
    route_presence = (
        route_presence.reindex(zone_ids, fill_value=0).gt(0).astype("int32")
    )

    # B @ B.T counts shared routes; threshold to obtain binary connectivity.
    B = route_presence.to_numpy()
    transportation = (B @ B.T > 0).astype(int)
    np.fill_diagonal(transportation, 0)

    connectivity = np.maximum(0, transportation - adjacency).astype(int)

    # Normalise the three graph matrices
    adjacency_sc = adjacency + np.eye(263)
    degree_matrix = np.zeros((263, 263))
    for i in range(len(degree_matrix)):
        degree_matrix[i][i] = 1 / np.sqrt(np.sum(adjacency_sc[:, i]))

    adjacency_normalized = degree_matrix @ adjacency_sc @ degree_matrix

    functionnality_sc = functionnality.copy()
    np.fill_diagonal(functionnality_sc, 1)
    degree_matrix_1 = np.zeros((263, 263))
    for i in range(len(degree_matrix)):
        degree_matrix_1[i][i] = 1 / np.sqrt(np.sum(functionnality_sc[i, :]))
    functionnality_normalized = degree_matrix_1 @ functionnality_sc @ degree_matrix_1

    degree_matrix_2 = np.zeros((263, 263))
    for i in range(len(degree_matrix_2)):
        if np.sum(connectivity[i, :]) != 0:
            degree_matrix_2[i][i] = 1 / np.sqrt(np.sum(connectivity[:, i]))
    connectivity_normalized = degree_matrix_2 @ connectivity @ degree_matrix_2

    # Graph Laplacians
    L_adj = np.eye(263) - adjacency_normalized
    L_fun = np.eye(263) - functionnality_normalized
    L_conn = np.eye(263) - connectivity_normalized

    L_adj = (2 / np.max(np.linalg.eigvalsh(L_adj))) * L_adj - np.eye(len(L_adj))
    L_fun = (2 / np.max(np.linalg.eigvalsh(L_fun))) * L_fun - np.eye(len(L_fun))
    L_conn = (2 / np.max(np.linalg.eigvalsh(L_conn))) * L_conn - np.eye(len(L_conn))
    return dict(
        adjacency_normalized=adjacency_normalized,
        functionnality_normalized=functionnality_normalized,
        connectivity_normalized=connectivity_normalized,
        L_adj=L_adj,
        L_fun=L_fun,
        L_conn=L_conn,
    )


class CGRNN(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.RNN = nn.RNN(input_size=1, hidden_size=64, batch_first=True)
        self.K = 1
        self.alpha = nn.Parameter(torch.randn(self.K + 1))
        self.attention1 = nn.Linear(2, 8)
        self.attention2 = nn.Linear(8, 1)
        self.register_buffer("L", torch.as_tensor(L, dtype=torch.float32))

    def forward(self, x):
        B, T, N = x.shape
        x_copy = x
        alpha0, alpha1 = self.alpha
        f = self.sigmoid(alpha0 * x + alpha1 * (x_copy @ self.L))
        x_copy = torch.stack([x_copy, f], dim=-1)
        x_copy = x_copy.mean(dim=2)
        x_copy = self.relu(self.attention1(x_copy))
        x_copy = self.sigmoid(self.attention2(x_copy))
        x_copy = x * x_copy
        x_copy = x_copy.permute(0, 2, 1)
        x_copy = x_copy.reshape(B * N, T, 1)
        _, hidden = self.RNN(x_copy)
        x_copy = hidden[-1].reshape(B, N, 64)
        return x_copy


def cheb(k, L):
    L = torch.as_tensor(L, dtype=torch.float32)
    if k == 0:
        return torch.eye(len(L), device=L.device)
    elif k == 1:
        return L
    else:
        return 2 * L @ cheb(k - 1, L) - cheb(k - 2, L)


class ST_MGCN_init(nn.Module):
    def __init__(
        self,
        adjacency_normalized,
        functionnality_normalized,
        connectivity_normalized,
        input_dim,
        output_dim,
        L_fun,
        L_adj,
        L_conn,
    ):
        super().__init__()
        self.K = 2
        self.theta_adj = nn.Parameter(torch.randn(self.K + 1))
        self.theta_fun = nn.Parameter(torch.randn(self.K + 1))
        self.theta_conn = nn.Parameter(torch.randn(self.K + 1))
        self.relu = nn.ReLU()
        self.register_buffer(
            "adjacency", torch.as_tensor(adjacency_normalized, dtype=torch.float32)
        )
        self.register_buffer(
            "functionnality",
            torch.as_tensor(functionnality_normalized, dtype=torch.float32),
        )
        self.register_buffer(
            "connectivity",
            torch.as_tensor(connectivity_normalized, dtype=torch.float32),
        )
        self.register_buffer("L_adj", torch.as_tensor(L_adj, dtype=torch.float32))
        self.register_buffer("L_conn", torch.as_tensor(L_conn, dtype=torch.float32))
        self.register_buffer("L_fun", torch.as_tensor(L_fun, dtype=torch.float32))
        self.cgrnn_adj = CGRNN(L_adj)
        self.cgrnn_fun = CGRNN(L_fun)
        self.cgrnn_conn = CGRNN(L_conn)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x_adj = self.cgrnn_adj(x)
        x_fun = self.cgrnn_fun(x)
        x_conn = self.cgrnn_conn(x)
        theta0, theta1, theta2 = self.theta_adj
        theta00, theta11, theta22 = self.theta_fun
        theta000, theta111, theta222 = self.theta_conn
        adj = (
            theta0 * torch.eye(263, device=self.L_adj.device)
            + theta1 * self.L_adj
            + theta2 * (cheb(2, self.L_adj))
        )
        fun = (
            theta00 * torch.eye(263, device=self.L_fun.device)
            + theta11 * self.L_fun
            + theta22 * (cheb(2, self.L_fun))
        )
        conn = (
            theta000 * torch.eye(263, device=self.L_conn.device)
            + theta111 * self.L_conn
            + theta222 * (cheb(2, self.L_conn))
        )
        x_adj = self.linear(x_adj)
        x_fun = self.linear(x_fun)
        x_conn = self.linear(x_conn)
        x_adj = adj @ x_adj
        x_fun = fun @ x_fun
        x_conn = conn @ x_conn
        x_next = self.relu(x_adj + x_fun + x_conn)
        return x_next


class ST_MGCN_layer(nn.Module):
    def __init__(
        self,
        adjacency_normalized,
        functionnality_normalized,
        connectivity_normalized,
        input_dim,
        output_dim,
        L_fun,
        L_adj,
        L_conn,
    ):
        super().__init__()
        self.K = 2
        self.theta_adj = nn.Parameter(torch.randn(self.K + 1))
        self.theta_fun = nn.Parameter(torch.randn(self.K + 1))
        self.theta_conn = nn.Parameter(torch.randn(self.K + 1))
        self.relu = nn.ReLU()
        self.register_buffer("L_adj", torch.as_tensor(L_adj, dtype=torch.float32))
        self.register_buffer("L_conn", torch.as_tensor(L_conn, dtype=torch.float32))
        self.register_buffer("L_fun", torch.as_tensor(L_fun, dtype=torch.float32))
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        theta0, theta1, theta2 = self.theta_adj
        theta00, theta11, theta22 = self.theta_fun
        theta000, theta111, theta222 = self.theta_conn
        adj = (
            theta0 * torch.eye(263, device=self.L_adj.device)
            + theta1 * self.L_adj
            + theta2 * (cheb(2, self.L_adj))
        )
        fun = (
            theta00 * torch.eye(263, device=self.L_fun.device)
            + theta11 * self.L_fun
            + theta22 * (cheb(2, self.L_fun))
        )
        conn = (
            theta000 * torch.eye(263, device=self.L_conn.device)
            + theta111 * self.L_conn
            + theta222 * (cheb(2, self.L_conn))
        )
        x = self.linear(x)
        x_next = self.relu((adj + fun + conn) @ x)
        return x_next


class ST_MGCN(nn.Module):
    def __init__(
        self,
        adjacency_normalized,
        functionnality_normalized,
        connectivity_normalized,
        L_fun,
        L_adj,
        L_conn,
    ):
        super().__init__()
        self.stmgcn_i = ST_MGCN_init(
            adjacency_normalized,
            functionnality_normalized,
            connectivity_normalized,
            input_dim=64,
            output_dim=64,
            L_fun=L_fun,
            L_adj=L_adj,
            L_conn=L_conn,
        )
        self.layers = nn.ModuleList(
            [
                ST_MGCN_layer(
                    adjacency_normalized,
                    functionnality_normalized,
                    connectivity_normalized,
                    input_dim=64,
                    output_dim=64,
                    L_fun=L_fun,
                    L_adj=L_adj,
                    L_conn=L_conn,
                )
                for _ in range(2)
            ]
        )
        self.prediction = nn.Linear(64, 1)

    def forward(self, x):
        x = self.stmgcn_i(x)
        for layer in self.layers:
            x = layer(x)
        x = self.prediction(x)
        return x.squeeze(-1)


def save_json(output, name, value):
    (output / name).write_text(json.dumps(value, indent=2) + "\n")


def load_demand(project, year):
    name = (
        "DL_Dataset.parquet"
        if year == 2024
        else "yellow_taxi_2025_features_neighbor_mean.parquet"
    )
    frame = pd.read_parquet(project / "data" / name)
    timestamps = pd.to_datetime(frame["tpep_pickup_datetime"])
    if not timestamps.dt.year.eq(year).all():
        raise ValueError(f"Expected only {year} observations")
    frame["hour_of_year"] = (
        (timestamps.dt.day_of_year - 1) * 24 + timestamps.dt.hour + 1
    )
    demand = (
        frame.groupby(["hour_of_year", "PULocationID"])["trip_count"]
        .sum()
        .unstack("PULocationID")
    )
    expected_hours = 8784 if year == 2024 else 8760
    if demand.shape != (expected_hours, 263) or demand.isna().any().any():
        raise ValueError("Incomplete demand panel")
    if demand.columns.tolist() != list(range(1, 264)):
        raise ValueError("Demand columns must be ordered by IDs 1-263")
    return frame, demand.to_numpy()


def windows(demand):
    inputs = np.array([demand[i : i + 24, :] for i in range(len(demand) - 24)])
    targets = np.array([demand[i + 24] for i in range(len(demand) - 24)])
    return inputs, targets


def loader(inputs, targets, shuffle):
    dataset = TensorDataset(
        torch.tensor(inputs, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=64, shuffle=shuffle)


def evaluate(model, batches, device, mean, std, output, name):
    model.eval()
    predictions, actuals = [], []
    scaled_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for inputs, targets in batches:
            prediction = model(inputs.to(device))
            scaled_loss += criterion(
                prediction, targets.to(device)
            ).item() * inputs.size(0)
            predictions.append(prediction.cpu().numpy())
            actuals.append(targets.numpy())
    prediction = np.concatenate(predictions).astype(np.float64) * std + mean
    actual = np.concatenate(actuals).astype(np.float64) * std + mean
    error = prediction - actual
    # Retain the filenames consumed by the existing report-analysis tools.
    prediction_name = {
        "original_2025": "original_2025_jan02_dec31",
        "full_2025": "full_year_2025",
    }.get(name, name)
    np.savez_compressed(
        output / f"{prediction_name}_predictions.npz",
        prediction=prediction,
        actual=actual,
    )
    metrics = dict(
        mae=float(np.abs(error).mean()),
        mse=float(np.square(error).mean()),
        rmse=float(np.sqrt(np.square(error).mean())),
        bias=float(error.mean()),
        scaled_loss_rmse=float(np.sqrt(scaled_loss / len(batches.dataset)) * std),
        target_hours=len(actual),
        zones=263,
        zone_hours=int(actual.size),
        negative_predictions=int((prediction < 0).sum()),
    )
    metrics_name = (
        "development_test_metrics.json"
        if name == "development_oct_dec_2024"
        else f"{name}_metrics.json"
    )
    save_json(output, metrics_name, metrics)
    return metrics


def train(args, graphs, device):
    frame, demand = load_demand(args.project_dir, 2024)
    develop = args.stage == "develop"
    counts = (
        frame.loc[frame["tpep_pickup_datetime"].dt.month <= 8, "trip_count"]
        if develop
        else frame.trip_count
    )
    mean, std = float(np.mean(counts)), float(np.std(counts))
    inputs, targets = windows((demand - mean) / std)
    # Construct loaders in the same order as the verified scripts.
    if develop:
        training = loader(inputs[:5832], targets[:5832], True)
        validation = loader(inputs[5832:6552], targets[5832:6552], False)
        testing = loader(inputs[6552:], targets[6552:], False)
    else:
        training = loader(inputs, targets, True)
    model = ST_MGCN(**graphs).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = args.epochs or (100 if develop else 92)
    history, best = [], np.inf
    best_epoch = None
    for epoch in range(epochs):
        model.train()
        total = 0
        for inputs_batch, target_batch in training:
            inputs_batch, target_batch = inputs_batch.to(device), target_batch.to(
                device
            )
            optimizer.zero_grad()
            prediction = model(inputs_batch)
            loss = criterion(prediction, target_batch)
            total += loss.item() * inputs_batch.size(0)
            loss.backward()
            optimizer.step()
        row = dict(
            epoch=epoch + 1,
            training_rmse=float(np.sqrt(total / len(training.dataset)) * std),
        )
        if develop:
            model.eval()
            total_val = 0
            with torch.no_grad():
                for inputs_batch, target_batch in validation:
                    prediction = model(inputs_batch.to(device))
                    total_val += criterion(
                        prediction, target_batch.to(device)
                    ).item() * inputs_batch.size(0)
            row["validation_rmse"] = float(
                np.sqrt(total_val / len(validation.dataset)) * std
            )
            if row["validation_rmse"] < best:
                best, best_epoch = row["validation_rmse"], epoch + 1
                torch.save(model.state_dict(), args.output_dir / "best_ST_MGCN.pth")
        history.append(row)
        name = "development" if develop else "refit"
        pd.DataFrame(history).to_csv(
            args.output_dir / f"{name}_history.csv", index=False
        )
        print(json.dumps(row), flush=True)
    metadata = dict(
        seed=args.seed,
        epochs=epochs,
        mean_train=mean,
        std_train=std,
        training_target_hours=len(training.dataset),
    )
    if develop:
        metadata.update(best_epoch=best_epoch, best_validation_rmse=best)
        save_json(args.output_dir, "development_selection.json", metadata)
        model.load_state_dict(
            torch.load(
                args.output_dir / "best_ST_MGCN.pth",
                map_location=device,
                weights_only=True,
            )
        )
        print(
            evaluate(
                model,
                testing,
                device,
                mean,
                std,
                args.output_dir,
                "development_oct_dec_2024",
            ),
            flush=True,
        )
    else:
        torch.save(model.state_dict(), args.output_dir / "ST_MGCN_full_2024.pth")
        save_json(args.output_dir, "refit_metadata.json", metadata)


def evaluate_2025(args, graphs, device):
    metadata = json.loads((args.output_dir / "refit_metadata.json").read_text())
    mean, std = metadata["mean_train"], metadata["std_train"]
    model = ST_MGCN(**graphs).to(device)
    model.load_state_dict(
        torch.load(
            args.output_dir / "ST_MGCN_full_2024.pth",
            map_location=device,
            weights_only=True,
        )
    )
    _, demand_2025 = load_demand(args.project_dir, 2025)
    scaled_2025 = (demand_2025 - mean) / std
    inputs, targets = windows(scaled_2025)
    print(
        evaluate(
            model,
            loader(inputs, targets, False),
            device,
            mean,
            std,
            args.output_dir,
            "original_2025",
        ),
        flush=True,
    )
    del inputs, targets
    _, demand_2024 = load_demand(args.project_dir, 2024)
    history = np.concatenate([((demand_2024 - mean) / std)[-24:], scaled_2025])
    inputs, targets = windows(history)
    print(
        evaluate(
            model,
            loader(inputs, targets, False),
            device,
            mean,
            std,
            args.output_dir,
            "full_2025",
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stage", choices=["all", "develop", "refit", "evaluate"], default="all"
    )
    parser.add_argument(
        "--project-dir", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/deep_learning")
    )
    parser.add_argument(
        "--epochs", type=int, help="Training epochs for a single develop/refit stage"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    args.project_dir = args.project_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.epochs is not None and (
        args.epochs < 1 or args.stage in ["all", "evaluate"]
    ):
        parser.error("--epochs must be positive and used with develop or refit")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.stage == "all":
        # Separate processes preserve stage-specific random initialisation and free memory.
        manifest = dict(
            status="running",
            python=sys.version,
            platform=platform.platform(),
            torch=torch.__version__,
            numpy=np.__version__,
            pandas=pd.__version__,
            seed=args.seed,
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            source_sha256={
                Path(__file__)
                .name: hashlib.sha256(Path(__file__).read_bytes())
                .hexdigest()
            },
            started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        save_json(args.output_dir, "run_manifest.json", manifest)
        try:
            for stage in ["develop", "refit", "evaluate"]:
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--stage",
                    stage,
                    "--project-dir",
                    str(args.project_dir),
                    "--output-dir",
                    str(args.output_dir),
                    "--seed",
                    str(args.seed),
                    "--threads",
                    str(args.threads),
                    "--device",
                    args.device,
                ]
                if stage == "refit":
                    selection = json.loads(
                        (args.output_dir / "development_selection.json").read_text()
                    )
                    command += ["--epochs", str(selection["best_epoch"])]
                    manifest["refit_epochs_selected_on_validation"] = selection[
                        "best_epoch"
                    ]
                manifest["current_stage"] = stage
                save_json(args.output_dir, "run_manifest.json", manifest)
                subprocess.run(command, check=True)
                environment = json.loads(
                    (args.output_dir / f"{stage}_environment.json").read_text()
                )
                manifest["device"] = environment["device"]
        except BaseException as error:
            manifest.update(
                status=(
                    "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                ),
                error=str(error),
            )
            save_json(args.output_dir, "run_manifest.json", manifest)
            raise
        manifest.update(
            status="complete",
            finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        save_json(args.output_dir, "run_manifest.json", manifest)
        return
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    save_json(
        args.output_dir,
        f"{args.stage}_environment.json",
        dict(
            python=sys.version,
            platform=platform.platform(),
            torch=torch.__version__,
            numpy=np.__version__,
            pandas=pd.__version__,
            device=str(device),
            seed=args.seed,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        ),
    )
    graphs = build_graphs(args.project_dir)
    if args.stage == "evaluate":
        evaluate_2025(args, graphs, device)
    else:
        train(args, graphs, device)


if __name__ == "__main__":
    main()
