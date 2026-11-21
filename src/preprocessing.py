import numpy as np
import pandas as pd


# ---------------------------------------------------------
# FD002 SENSOR FILTERING
# ---------------------------------------------------------
def get_useful_sensors(train_df, var_threshold=0.01):
    """
    Filters out invalid C-MAPSS FD002 sensors (`sensor_20`, `sensor_21`)
    and near-constant low-variance sensors below `var_threshold`.
    """
    invalid_sensors = ["sensor_20", "sensor_21"]

    candidate_sensors = [
        f"sensor_{i}" for i in range(1, 22)
        if f"sensor_{i}" not in invalid_sensors
    ]

    sensor_var = train_df[candidate_sensors].var()
    low_var_sensors = sensor_var[sensor_var < var_threshold].index.tolist()

    useful_sensors = [
        s for s in candidate_sensors if s not in low_var_sensors
    ]

    meta = {
        "invalid_sensors": invalid_sensors,
        "low_variance_sensors": low_var_sensors,
    }

    return useful_sensors, meta


# ---------------------------------------------------------
# RUL COMPUTATION (FULL RUN-TO-FAILURE TRAJECTORIES)
# ---------------------------------------------------------
def compute_rul_full(df):
    """
    Computes run-to-failure Remaining Useful Life (`max(cycle) - cycle`)
    per engine. Use only on full-trajectory training/split data.
    """
    df = df.copy()
    df["RUL"] = (
        df.groupby("engine_id")["cycle"].transform("max") - df["cycle"]
    )
    return df


# ---------------------------------------------------------
# 3-SIGMA OUTLIER CAPPING (LEAK-SAFE FIT ON TRAIN / CLIP ALL)
# ---------------------------------------------------------
def apply_3sigma_capping(df, sensor_list, fit_df=None):
    """
    Computes 3-sigma bounds strictly from `fit_df` (defaults to `df` only
    if not provided) and clips `df[sensor_list]` in place on a copy.
    """
    df = df.copy()
    source_for_fit = fit_df if fit_df is not None else df
    cap_report = {}

    for s in sensor_list:
        mean = source_for_fit[s].mean()
        std = source_for_fit[s].std()

        lower = mean - 3 * std
        upper = mean + 3 * std

        outliers = int(((df[s] < lower) | (df[s] > upper)).sum())
        df[s] = df[s].clip(lower, upper)

        cap_report[s] = outliers

    return df, cap_report