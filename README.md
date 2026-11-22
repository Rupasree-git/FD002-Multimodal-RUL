# Trust-Aware Multimodal RUL Prediction (NASA C-MAPSS FD002)

An end-to-end Remaining Useful Life (RUL) prediction, adaptive fusion, and uncertainty-aware decision-support pipeline for multi-regime turbofan engine degradation on the **NASA C-MAPSS FD002** dataset.

## Key Highlights
- **Leak-Free Engine-Level Split:** Partitions engines (`80%` train / `15%` validation / `5%` test) prior to outlier capping, temporal aggregation, and PCA projection so trajectories never leak across splits.
- **Multimodal Feature Engineering (`77` features):**
  - **Sensor (`72`):** Useful FD002 sensors (`18` retained; drops invalid `sensor_20`, `sensor_21` and low-variance `sensor_16`) with per-engine rolling mean, rolling std, and rolling linear slope.
  - **Visual Proxy (`2`):** 2-component PCA (`visual_proxy_1`, `visual_proxy_2`, `99.49%` explained variance) fitted strictly on training engines.
  - **Tabular (`3`):** Operating settings (`op_setting_1`, `op_setting_2`) and normalized cycle progression (`cycle_norm`).
- **Adaptive Fusion & Decision Support:** Combines modality Random Forest regressors with **TrustGate** dynamic softmax weighting (`19.07` cycles MAE on unseen full-trajectory engines; `22.8%` error reduction), asymmetric **NASA scoring**, **SHAP** explainability, high-risk RUL screening (`6.19` high-risk MAE on official `test_FD002.txt`), and cross-modality disagreement screening (`detect_contradictions`).

## Repository Structure
- `notebooks/FD002_Clean_Pipeline.ipynb`: End-to-end execution across data loading, EDA, leak-free feature engineering, static & TrustGate fusion, diagnostics, SHAP, and decision support.
- `src/data_loader.py`: Loads `train_FD002.txt`, `test_FD002.txt`, and `RUL_FD002.txt`.
- `src/preprocessing.py`: Sensor filtering, run-to-failure RUL calculation, and 3-sigma outlier capping.
- `src/feature_engineering.py`: Per-engine temporal features, PCA visual proxy features, and modality definitions.
- `src/trustgate.py`: Rolling sensor reliability, operating context builder, and `TrustGateNetwork`.
- `src/contradiction.py`: Cross-modality prediction standard deviation and contradiction flagging.
- `src/evaluation.py`: Asymmetric NASA score, diagnostic plots, and SHAP summary visualization.
- `src/ablation_studies.py`: Rolling-window sensitivity evaluation across engine-disjoint splits.

## Quick Start
```bash
git clone https://github.com/Rupasree-git/FD002-Multimodal-RUL
cd FD002-Multimodal-RUL
pip install -r requirements.txt

## Live Demo & Deployment
- **Frontend (Next.js on Vercel):** https://fd-002-multimodal-rul.vercel.app
- **Backend API (FastAPI on Render):** https://fd002-multimodal-rul.onrender.com