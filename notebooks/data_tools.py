import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import xgboost as xgb
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.compose import ColumnTransformer


def clean(df):
    return df[(df["PULocationID"] != 265) & (df["PULocationID"] != 264)]


def valid_date(df, m):
    start = pd.Timestamp(year=2024, month=m, day=1)
    end = start + pd.DateOffset(months=1)
    df_clean = df.copy()
    mask1 = pd.to_datetime(df_clean["tpep_pickup_datetime"]) >= start
    mask2 = pd.to_datetime(df_clean["tpep_pickup_datetime"]) < end
    df_clean = df_clean[mask1]
    df_clean = df_clean[mask2]
    return df_clean


def groupby_hour(df, m):
    df_hourly = (
        df.groupby([pd.Grouper(key="tpep_pickup_datetime", freq="h"), "PULocationID"])
        .size()
        .reset_index(name="trip_count")
    )
    start = pd.Timestamp(year=2024, month=m, day=1)
    end = start + pd.DateOffset(months=1)
    hours = pd.date_range(start=start, end=end, freq="h", inclusive="left")

    zones = range(1, 264)

    full_index = pd.MultiIndex.from_product(
        [hours, zones], names=["tpep_pickup_datetime", "PULocationID"]
    )

    df_hourly = (
        df_hourly.set_index(["tpep_pickup_datetime", "PULocationID"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    return df_hourly


def split(X):
    X_train = X[X["tpep_pickup_datetime"].dt.month <= 8]
    X_val = X[
        (X["tpep_pickup_datetime"].dt.month > 8)
        & (X["tpep_pickup_datetime"].dt.month <= 9)
    ]
    X_test = X[X["tpep_pickup_datetime"].dt.month > 9]
    y_train = X_train.trip_count
    y_val = X_val.trip_count
    y_test = X_test.trip_count

    train = X_train.drop(columns=["trip_count"])
    val = X_val.drop(columns=["trip_count"])
    test = X_test.drop(columns=["trip_count"])

    train = train.drop(columns=["tpep_pickup_datetime"])
    val = val.drop(columns=["tpep_pickup_datetime"])
    test = test.drop(columns=["tpep_pickup_datetime"])

    return X_train, X_val, X_test, y_train, y_val, y_test, train, val, test


def metrics(y_val, y_pred):
    mae = mean_absolute_error(y_val, y_pred)
    rsme = root_mean_squared_error(y_val, y_pred)
    m_pred = y_pred.mean()
    med_pred = np.median(y_pred)
    max_pred = np.max(y_pred)
    p95_pred = np.percentile(y_pred, 95)
    p99_pred = np.percentile(y_pred, 99)

    print("Validation performance")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rsme:.4f}")
    print("\nPredicted pickups per zone per hour")
    print(f"Mean:            {m_pred:.4f}")
    print(f"Median:          {med_pred:.4f}")
    print(f"95th percentile: {p95_pred:.4f}")
    print(f"99th percentile: {p99_pred:.4f}")
    print(f"Maximum:         {max_pred:.4f}")


def ranking_zones(X_train, X_val, val, model):
    # Ranking the highest demand on an hourly average
    rank = []
    for z in X_train.PULocationID.unique():
        rank.append((z, X_train["trip_count"][X_train["PULocationID"] == z].mean()))

    rank = sorted(rank, key=lambda x: x[1])
    rank.reverse()

    rank_zones = [rank[i][0] for i in range(len(rank))]  # keep only the zone number
    high_demand = rank_zones[: (263 // 3)]
    mid_demand = rank_zones[(263 // 3) : 2 * (263 // 3)]
    low_demand = rank_zones[2 * (263 // 3) :]

    val_high = val[val["PULocationID"].isin(high_demand)]
    val_mid = val[val["PULocationID"].isin(mid_demand)]
    val_low = val[val["PULocationID"].isin(low_demand)]

    y_high = X_val.trip_count[X_val["PULocationID"].isin(high_demand)]
    y_pred_high = model.predict(val_high)

    y_mid = X_val.trip_count[X_val.PULocationID.isin(mid_demand)]
    y__pred_mid = model.predict(val_mid)

    y_low = X_val.trip_count[X_val.PULocationID.isin(low_demand)]
    y_pred_low = model.predict(val_low)

    return (
        val_high,
        val_mid,
        val_low,
        y_high,
        y_pred_high,
        y_mid,
        y__pred_mid,
        y_low,
        y_pred_low,
    )


def model(train):
    train_ablation = train.drop(columns=["month"])

    categorical = ["PULocationID", "hours"]
    numerical = [col for col in train_ablation.columns if col not in categorical]

    preprocessor = ColumnTransformer(
        [
            ("categories", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numbers", StandardScaler(), numerical),
        ]
    )
    model = make_pipeline(
        preprocessor,
        xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
        ),
    )
    return model
