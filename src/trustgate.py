# =========================================================
# TrustGate Network
# =========================================================
# - Learns adaptive modality trust weights
# - Uses temporal reliability + operating context
# - NumPy / Torch safe
# =========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


# =========================================================
# TEMPORAL RELIABILITY
# =========================================================
def compute_temporal_reliability(df, sensor_list, window=20):
    """
    Computes simple temporal reliability using per-engine rolling std.
    Higher stability -> higher reliability in [0, 1].
    """
    reliability = pd.DataFrame(index=df.index)

    for sensor in sensor_list:
        roll_std = (
            df.groupby("engine_id")[sensor]
            .transform(lambda x: x.rolling(window=window, min_periods=1).std().fillna(0))
        )

        std_min, std_max = roll_std.min(), roll_std.max()
        if std_max - std_min > 1e-8:
            rel = 1.0 - (roll_std - std_min) / (std_max - std_min)
        else:
            rel = pd.Series(1.0, index=roll_std.index)

        reliability[f"{sensor}_reliability"] = rel

    return reliability


# =========================================================
# CONTEXT FEATURE PREPARATION
# =========================================================
def prepare_context_features(df, reliability_df):
    """
    Builds TrustGate context features:
    - Operating settings
    - Cycle position
    - Average sensor reliability
    - Operating-condition dummies
    """
    context = df[["op_setting_1", "op_setting_2", "cycle_norm"]].copy()

    rel_cols = [c for c in reliability_df.columns if "reliability" in c]
    context["avg_reliability"] = reliability_df[rel_cols].mean(axis=1)

    op_cond = (
        df[["op_setting_1", "op_setting_2"]]
        .round(1)
        .astype(str)
        .agg("_".join, axis=1)
    )
    op_dummies = pd.get_dummies(op_cond, prefix="cond")

    context = pd.concat([context, op_dummies], axis=1)
    return context


# =========================================================
# TRUSTGATE NETWORK
# =========================================================
class TrustGateNetwork(nn.Module):
    def __init__(self, input_dim, n_modalities=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_modalities),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# TRAIN TRUSTGATE
# =========================================================
def train_trustgate(
    train_df,
    val_mask,
    useful_sensors,
    MODALITIES,
    val_predictions,
    y_val,
    epochs=100,
    lr=0.001,
    seed=42,
):
    """
    Trains TrustGate Network on validation predictions and context features.
    Returns:
    - trained TrustGate model
    - scaler
    - fused validation MAE and predictions
    - trust weights DataFrame
    - context features DataFrame
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    val_df = train_df[val_mask].copy()

    reliability = compute_temporal_reliability(val_df, useful_sensors)
    context_df = prepare_context_features(val_df, reliability)

    scaler = StandardScaler()
    X_context = torch.tensor(
        scaler.fit_transform(context_df), dtype=torch.float32
    )
    y_true = torch.tensor(
        np.asarray(y_val, dtype=np.float32), dtype=torch.float32
    )

    modality_tensors = {
        i: torch.tensor(np.asarray(val_predictions[m], dtype=np.float32), dtype=torch.float32)
        for i, m in enumerate(MODALITIES)
    }

    tgn = TrustGateNetwork(X_context.shape[1], n_modalities=len(MODALITIES))
    optimizer = optim.Adam(tgn.parameters(), lr=lr)
    criterion = nn.L1Loss()

    tgn.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        weights = tgn(X_context)
        fused_pred = torch.zeros_like(y_true)

        for i in range(len(MODALITIES)):
            fused_pred = fused_pred + weights[:, i] * modality_tensors[i]

        loss = criterion(fused_pred, y_true)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

    tgn.eval()
    with torch.no_grad():
        final_weights = tgn(X_context)
        y_pred = torch.zeros_like(y_true)
        for i in range(len(MODALITIES)):
            y_pred = y_pred + final_weights[:, i] * modality_tensors[i]

    y_pred_np = y_pred.numpy()
    mae = mean_absolute_error(np.asarray(y_val), y_pred_np)

    weight_df = pd.DataFrame(
        final_weights.numpy(),
        columns=[f"{m}_weight" for m in MODALITIES],
        index=val_df.index,
    )

    return {
        "model": tgn,
        "scaler": scaler,
        "mae": mae,
        "y_pred": y_pred_np,
        "weights": weight_df,
        "context": context_df,
    }