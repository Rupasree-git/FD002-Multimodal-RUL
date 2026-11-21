# src/inference.py
# =========================================================
# Shared Inference & Batch Scoring Module
# =========================================================
# - Loads saved PCA, modality RFs, TrustGate, and metadata
# - Computes leak-safe preprocessing and multimodal features
# - Produces modality predictions, TrustGate weights, risk,
#   and inter-modality contradiction diagnostics
# - Supports both Python imports (e.g., Streamlit/Gradio)
#   and CLI batch scoring
# =========================================================

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

from src.contradiction import detect_contradictions
from src.data_loader import load_fd002_data
from src.evaluation import nasa_score
from src.feature_engineering import add_temporal_features, define_modalities
from src.preprocessing import compute_rul_full
from src.trustgate import TrustGateNetwork, compute_temporal_reliability, prepare_context_features


def rul_to_risk(rul: np.ndarray, alpha: float = 150.0, scale: float = 20.0) -> np.ndarray:
    """Converts RUL cycles into a normalized [0, 1] failure risk score."""
    rul_arr = np.asarray(rul, dtype=np.float64)
    return 1.0 / (1.0 + np.exp((rul_arr - alpha) / scale))


def load_artifacts(models_dir: str | Path = "models", device: str = "cpu") -> Dict[str, Any]:
    """Loads saved models, preprocessors, and pipeline metadata from disk."""
    models_path = Path(models_dir)
    metadata_path = models_path / "pipeline_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file at: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    pca_path = models_path / "pca_visual.joblib"
    if not pca_path.exists():
        raise FileNotFoundError(f"Missing PCA artifact at: {pca_path}")
    pca = joblib.load(pca_path)

    modalities: Dict[str, list] = metadata["modalities"]
    modality_models: Dict[str, Any] = {}
    for modality in modalities:
        rf_path = models_path / f"rf_{modality}.joblib"
        if not rf_path.exists():
            raise FileNotFoundError(f"Missing Random Forest model for '{modality}' at: {rf_path}")
        modality_models[modality] = joblib.load(rf_path)

    tgn_path = models_path / "trustgate.pt"
    scaler_path = models_path / "trustgate_scaler.joblib"
    if not tgn_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            f"Missing TrustGate artifacts ('trustgate.pt' or 'trustgate_scaler.joblib') in {models_path}"
        )

    checkpoint = torch.load(tgn_path, map_location=device)
    tgn_scaler = joblib.load(scaler_path)

    input_dim = int(checkpoint.get("input_dim", tgn_scaler.mean_.shape[0]))
    n_modalities = int(checkpoint.get("n_modalities", len(modalities)))

    tgn_model = TrustGateNetwork(input_dim=input_dim, n_modalities=n_modalities).to(device)
    tgn_model.load_state_dict(checkpoint["state_dict"])
    tgn_model.eval()

    return {
        "metadata": metadata,
        "pca": pca,
        "modality_models": modality_models,
        "tgn_model": tgn_model,
        "tgn_scaler": tgn_scaler,
        "device": device,
    }


def _ensure_cycle_norm(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures `cycle_norm` exists per engine trajectory without cross-engine leakage."""
    out = df.copy()
    if "cycle_norm" not in out.columns:
        max_cycles = out.groupby("engine_id")["cycle"].transform("max").replace(0, 1)
        out["cycle_norm"] = out["cycle"] / max_cycles
    return out


def _build_aligned_context(
    df: pd.DataFrame,
    useful_sensors: list,
    tgn_scaler: Any,
    context_columns: Optional[list],
    reliability_bounds: Optional[Dict[str, Dict[str, float]]] = None,
) -> np.ndarray:
    """Computes and column-aligns TrustGate context features for inference."""
    reliability, _ = compute_temporal_reliability(
        df,
        useful_sensors,
        reliability_bounds=reliability_bounds,
    )
    context_df = prepare_context_features(df, reliability)

    expected_cols = context_columns or list(getattr(tgn_scaler, "feature_names_in_", []))
    if expected_cols:
        context_df = context_df.reindex(columns=expected_cols, fill_value=0.0)

    return tgn_scaler.transform(context_df)


def predict_trajectories(
    df: pd.DataFrame,
    artifacts: Dict[str, Any],
    contradiction_threshold: Optional[float] = None,
    has_ground_truth_rul: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Runs end-to-end leak-safe feature generation and multimodal/TrustGate inference
    on a trajectory DataFrame containing `engine_id`, `cycle`, operating settings,
    and sensor columns.
    """
    if df.empty:
        raise ValueError("Input trajectory DataFrame is empty.")

    metadata = artifacts["metadata"]
    pca = artifacts["pca"]
    modality_models = artifacts["modality_models"]
    tgn_model = artifacts["tgn_model"]
    tgn_scaler = artifacts["tgn_scaler"]
    device = artifacts.get("device", "cpu")

    seed = int(metadata.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)

    threshold = (
        float(contradiction_threshold)
        if contradiction_threshold is not None
        else float(metadata.get("contradiction_threshold", 50.0))
    )

    useful_sensors = list(metadata["useful_sensors"])
    visual_features = list(metadata["visual_features"])
    all_features = list(metadata["all_features"])

    work_df = _ensure_cycle_norm(df)
    if has_ground_truth_rul and "RUL" not in work_df.columns:
        work_df = compute_rul_full(work_df)

    # Apply saved training 3-sigma bounds
    lower_bounds = pd.Series(metadata["outlier_lower"])
    upper_bounds = pd.Series(metadata["outlier_upper"])
    aligned_lower = lower_bounds.reindex(useful_sensors)
    aligned_upper = upper_bounds.reindex(useful_sensors)
    work_df[useful_sensors] = work_df[useful_sensors].clip(
        lower=aligned_lower, upper=aligned_upper, axis=1
    )

    # Compute per-engine temporal features
    work_df = add_temporal_features(work_df, useful_sensors)

    # Project saved PCA visual proxy features
    pca_transformed = pca.transform(work_df[useful_sensors])
    for i, col in enumerate(visual_features):
        work_df[col] = pca_transformed[:, i]

    modalities, _ = define_modalities(useful_sensors, visual_features)
    modalities["visual"] = visual_features

    missing_cols = [c for c in all_features if c not in work_df.columns]
    if missing_cols:
        raise ValueError(f"Missing engineered feature columns required for inference: {missing_cols}")

    # Modality-specific Random Forest predictions
    modality_preds: Dict[str, np.ndarray] = {}
    for modality, features in modalities.items():
        modality_preds[modality] = np.asarray(
            modality_models[modality].predict(work_df[features]), dtype=np.float64
        )

    # Preserve training modality order when mapping TrustGate weights
    saved_order = metadata.get("modality_order") or list(modalities.keys())
    modality_names = [m for m in saved_order if m in modalities] + [
        m for m in modalities.keys() if m not in saved_order
    ]

    # TrustGate adaptive weights & fused RUL
    X_context_scaled = _build_aligned_context(
        work_df,
        useful_sensors=useful_sensors,
        tgn_scaler=tgn_scaler,
        context_columns=metadata.get("context_columns"),
        reliability_bounds=metadata.get("reliability_bounds"),
    )
    context_tensor = torch.tensor(X_context_scaled, dtype=torch.float32, device=device)

    with torch.no_grad():
        gate_weights = tgn_model(context_tensor).cpu().numpy()

    y_pred_trustgate = np.zeros(len(work_df), dtype=np.float64)
    for idx, m_name in enumerate(modality_names):
        y_pred_trustgate += gate_weights[:, idx] * modality_preds[m_name]

    # Risk & contradiction diagnostics
    risk_score = rul_to_risk(y_pred_trustgate)
    contradiction_flags, disagreement_std = detect_contradictions(
        modality_preds, threshold=threshold
    )

    result_df = pd.DataFrame(
        {
            "engine_id": work_df["engine_id"].values,
            "cycle": work_df["cycle"].values,
            "cycle_norm": work_df["cycle_norm"].values,
            "pred_rul_trustgate": y_pred_trustgate,
            "failure_risk": risk_score,
            "modality_disagreement_std": disagreement_std,
            "contradiction_flag": contradiction_flags,
        },
        index=work_df.index,
    )

    for m_name in modality_names:
        result_df[f"pred_rul_{m_name}"] = modality_preds[m_name]
    for idx, m_name in enumerate(modality_names):
        result_df[f"trust_weight_{m_name}"] = gate_weights[:, idx]

    summary: Dict[str, Any] = {
        "num_rows": int(len(result_df)),
        "num_engines": int(result_df["engine_id"].nunique()),
        "contradiction_threshold": threshold,
        "contradiction_rate": float(np.mean(contradiction_flags)),
        "mean_pred_rul": float(np.mean(y_pred_trustgate)),
        "mean_failure_risk": float(np.mean(risk_score)),
    }

    if "RUL" in work_df.columns:
        y_true = np.asarray(work_df["RUL"], dtype=np.float64)
        result_df["true_rul"] = y_true
        result_df["abs_error"] = np.abs(y_true - y_pred_trustgate)
        summary["mae"] = float(np.mean(result_df["abs_error"]))
        summary["nasa_score"] = float(nasa_score(y_true, y_pred_trustgate))

    return result_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run batch RUL, TrustGate weight, risk, and contradiction inference on FD002."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing FD002 dataset files (used when --input-csv is not provided).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Which FD002 split to score when loading from --data-dir.",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Optional path to a custom trajectory CSV file.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing saved model and metadata artifacts.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="outputs/rul_predictions.csv",
        help="Path where prediction results will be saved.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional override for contradiction standard deviation threshold (cycles).",
    )
    args = parser.parse_args()

    artifacts = load_artifacts(models_dir=args.models_dir)

    if args.input_csv:
        input_df = pd.read_csv(args.input_csv)
        has_gt = "RUL" in input_df.columns
    else:
        train_df, test_df, _ = load_fd002_data(args.data_dir)
        if args.split == "train":
            input_df = compute_rul_full(train_df)
            has_gt = True
        else:
            input_df = test_df
            has_gt = False

    preds_df, summary = predict_trajectories(
        input_df,
        artifacts=artifacts,
        contradiction_threshold=args.threshold,
        has_ground_truth_rul=has_gt,
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preds_df.to_csv(output_path, index=False)

    print("Batch inference complete:", output_path.resolve())
    for k, v in summary.items():
        print(f"   - {k}: {v:.4f}" if isinstance(v, float) else f"   - {k}: {v}")


if __name__ == "__main__":
    main()