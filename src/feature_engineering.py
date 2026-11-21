import numpy as np
from sklearn.decomposition import PCA


# ---------------------------------------------------------
# TEMPORAL FEATURES
# ---------------------------------------------------------
def add_temporal_features(df, sensors, roll_window=10, slope_window=20):
    """
    Adds per-engine rolling statistics, rolling linear slopes, and
    normalized cycle progression without cross-engine mixing.
    """
    df = df.copy()

    for s in sensors:
        grouped = df.groupby("engine_id")[s]

        df[f"{s}_roll_mean"] = (
            grouped.rolling(roll_window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df[f"{s}_roll_std"] = (
            grouped.rolling(roll_window, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )

        def slope(x):
            return float(np.polyfit(np.arange(len(x)), x, 1)[0]) if len(x) > 1 else 0.0

        df[f"{s}_slope"] = (
            grouped.apply(lambda g: g.rolling(slope_window, min_periods=2).apply(slope, raw=True))
            .reset_index(level=0, drop=True)
            .fillna(0)
        )

    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["cycle_norm"] = np.where(max_cycle > 0, df["cycle"] / max_cycle, 0.0)

    return df


# ---------------------------------------------------------
# PCA VISUAL PROXY (LEAK-SAFE FIT ON TRAIN / TRANSFORM ALL)
# ---------------------------------------------------------
def add_visual_proxy_features(df, sensors, n_components=2, fit_df=None):
    """
    Fits PCA strictly on `fit_df` (defaults to `df` only if not provided)
    and projects `df[sensors]` into standardized visual proxy features.
    """
    df = df.copy()
    source_for_fit = fit_df if fit_df is not None else df

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(source_for_fit[sensors])

    visual_data = pca.transform(df[sensors])
    n_comp = pca.n_components_
    visual_features = [f"visual_proxy_{i + 1}" for i in range(n_comp)]

    for i, col in enumerate(visual_features):
        df[col] = visual_data[:, i]

    meta = {
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": pca.explained_variance_ratio_.cumsum(),
        "n_components": n_comp,
    }

    return df, visual_features, meta


# ---------------------------------------------------------
# MODALITY DEFINITIONS
# ---------------------------------------------------------
def define_modalities(useful_sensors, visual_features):
    sensor_features = []
    for s in useful_sensors:
        sensor_features.extend([
            s,
            f"{s}_roll_mean",
            f"{s}_roll_std",
            f"{s}_slope",
        ])

    tabular_features = ["op_setting_1", "op_setting_2", "cycle_norm"]

    modalities = {
        "sensor": sensor_features,
        "visual": visual_features,
        "tabular": tabular_features,
    }

    all_features = sensor_features + visual_features + tabular_features

    return modalities, all_features