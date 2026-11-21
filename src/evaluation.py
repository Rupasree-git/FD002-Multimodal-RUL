# src/evaluation.py
# =========================================================
# Evaluation & Visual Analytics
# =========================================================
# - NASA official score
# - Global model performance visualizations
# - Static R² fusion weight analysis
# - Risk score distribution
# - SHAP explainability (sensor modality, validation set)
# =========================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score

sns.set(style="whitegrid")


# =========================================================
# NASA OFFICIAL SCORE
# =========================================================
def nasa_score(y_true, y_pred):
    """
    NASA C-MAPSS asymmetric scoring function.
    Late predictions (`d >= 0`) are penalized more heavily than early ones (`d < 0`).
    """
    d = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    scores = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(scores))


# =========================================================
# MAIN EVALUATION FUNCTION
# =========================================================
def run_evaluation(
    y_test,
    y_pred_fused,
    y_val,
    val_predictions,
    MODALITIES,
    risk_true,
    risk_pred,
    X_val,
    modality_models,
    seed=42,
):
    rng = np.random.default_rng(seed)

    y_test_arr = np.asarray(y_test, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred_fused, dtype=np.float64)
    y_val_arr = np.asarray(y_val, dtype=np.float64)
    risk_true_arr = np.asarray(risk_true, dtype=np.float64)
    risk_pred_arr = np.asarray(risk_pred, dtype=np.float64)

    mae_test = mean_absolute_error(y_test_arr, y_pred_arr)
    nasa_test = nasa_score(y_test_arr, y_pred_arr)

    print("🎯 FINAL TEST PERFORMANCE (GLOBAL MODEL)")
    print(f"   MAE:        {mae_test:.2f} cycles")
    print(f"   NASA Score: {nasa_test:,.0f}")
    print(f"   Risk range: [{risk_pred_arr.min():.3f}, {risk_pred_arr.max():.3f}]")

    # 1. True vs Predicted RUL
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test_arr, y_pred_arr, alpha=0.5, s=15)
    max_val = max(float(y_test_arr.max()), float(y_pred_arr.max()))
    plt.plot([0, max_val], [0, max_val], "r--", linewidth=2)
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title("True vs Predicted RUL (Global Multimodal Model)")
    plt.show()

    # 2. Residual Distribution
    residuals = y_pred_arr - y_test_arr
    plt.figure(figsize=(7, 4))
    plt.hist(residuals, bins=40, alpha=0.8)
    plt.axvline(0, color="red", linestyle="--")
    plt.xlabel("Prediction Error (Predicted − True RUL)")
    plt.ylabel("Frequency")
    plt.title("Residual Error Distribution (Test Set)")
    plt.show()

    # 3. Error vs True RUL
    plt.figure(figsize=(7, 4))
    plt.scatter(y_test_arr, residuals, alpha=0.5, s=15)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("True RUL")
    plt.ylabel("Prediction Error")
    plt.title("Prediction Error vs True RUL")
    plt.show()

    # 4. Modality Performance (Validation)
    modality_mae = {
        modality: mean_absolute_error(y_val_arr, val_predictions[modality])
        for modality in MODALITIES
    }
    plt.figure(figsize=(6, 4))
    plt.bar(modality_mae.keys(), modality_mae.values())
    plt.ylabel("MAE (cycles)")
    plt.title("Validation MAE per Modality")
    plt.show()

    # 5. Static R²-Based Fusion Weights
    reconstructed_weights = {
        modality: max(0.0, float(r2_score(y_val_arr, val_predictions[modality])))
        for modality in MODALITIES
    }
    if sum(reconstructed_weights.values()) == 0.0:
        reconstructed_weights = {m: 1.0 for m in MODALITIES}

    plt.figure(figsize=(6, 4))
    plt.bar(reconstructed_weights.keys(), reconstructed_weights.values())
    plt.ylabel("Fusion Weight (Trust)")
    plt.title("Static Modality Trust Weights (R²-based)")
    plt.show()

    # 6. Risk Score Distribution
    plt.figure(figsize=(7, 4))
    plt.hist(risk_true_arr, bins=30, alpha=0.6, label="True Risk")
    plt.hist(risk_pred_arr, bins=30, alpha=0.6, label="Predicted Risk")
    plt.xlabel("Risk Score")
    plt.ylabel("Frequency")
    plt.title("Risk Score Distribution (Test Set)")
    plt.legend()
    plt.show()

    # 7. SHAP Explainability (Sensor Modality – Validation Set)
    print("\n🔍 SHAP Explainability (Sensor Modality – Validation Set)")
    try:
        import shap

        n_samples = min(100, len(X_val))
        sample_pos = rng.choice(len(X_val), size=n_samples, replace=False)

        X_val_df = X_val if isinstance(X_val, pd.DataFrame) else pd.DataFrame(X_val)
        X_sample = X_val_df.iloc[sample_pos][MODALITIES["sensor"]].copy()

        explainer = shap.TreeExplainer(modality_models["sensor"])
        shap_values = explainer.shap_values(X_sample)

        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title("SHAP Summary Plot – Sensor Modality")
        plt.show()

    except ImportError:
        print("SHAP not installed. Run: pip install shap")
    except Exception as e:
        print(f"SHAP error: {e}")