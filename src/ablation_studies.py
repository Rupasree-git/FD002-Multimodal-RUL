import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def add_rolling_features(df, sensors, window):
    """
    Adds per-engine rolling mean and standard deviation for a given window size
    with explicit index alignment.
    """
    df_new = df.copy()
    grouped = df_new.groupby("engine_id")

    for s in sensors:
        s_group = grouped[s]

        df_new[f"{s}_roll_mean_w{window}"] = (
            s_group.rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df_new[f"{s}_roll_std_w{window}"] = (
            s_group.rolling(window, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )

    return df_new


def evaluate_rolling_windows(
    train_df,
    useful_sensors,
    sensor_features,
    train_engines,
    val_engines,
    window_sizes,
    n_estimators=50,
    max_depth=12,
    seed=42,
):
    """
    Evaluates validation MAE across candidate rolling window sizes using an
    engine-disjoint train/validation split.
    """
    results = []
    engines = train_df["engine_id"].values
    y = train_df["RUL"].values

    train_mask = np.isin(engines, train_engines)
    val_mask = np.isin(engines, val_engines)

    for w in window_sizes:
        df_w = add_rolling_features(train_df, useful_sensors, w)
        rolling_features = [c for c in df_w.columns if c.endswith(f"_w{w}")]
        feature_cols = sensor_features + rolling_features

        X = df_w[feature_cols].values

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed,
            n_jobs=-1,
        )

        rf.fit(X[train_mask], y[train_mask])
        preds = rf.predict(X[val_mask])

        mae = mean_absolute_error(y[val_mask], preds)
        results.append([w, mae])

    return pd.DataFrame(
        results,
        columns=["Rolling Window Size", "MAE (Val)"],
    )