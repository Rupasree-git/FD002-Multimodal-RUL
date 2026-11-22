# api/main.py
import gc
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.data_loader import load_fd002_data
from src.feature_engineering import add_temporal_features, define_modalities
from src.inference import load_artifacts, predict_trajectories
from src.preprocessing import compute_rul_full

torch.set_num_threads(1)

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"

app = FastAPI(title="C-MAPSS FD002 Prognostics API", version="1.0.0")

allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://fd-002-multimodal-rul.vercel.app",
]
extra_origin = os.getenv("FRONTEND_URL")
if extra_origin:
    allowed_origins.append(extra_origin.rstrip("/"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_ARTIFACTS: Optional[Dict[str, Any]] = None
_DATASETS: Optional[Dict[str, pd.DataFrame]] = None


def get_cached_artifacts() -> Dict[str, Any]:
    global _ARTIFACTS
    if _ARTIFACTS is None:
        _ARTIFACTS = load_artifacts(models_dir=MODELS_DIR, device="cpu")
        gc.collect()
    return _ARTIFACTS


def get_cached_datasets() -> Dict[str, pd.DataFrame]:
    global _DATASETS
    if _DATASETS is None:
        train_df, test_df, rul_df = load_fd002_data(DATA_DIR)
        train_with_rul = compute_rul_full(train_df)
        _DATASETS = {
            "train": train_with_rul,
            "test": test_df,
            "rul_reference": rul_df,
        }
        gc.collect()
    return _DATASETS


class InferenceRequest(BaseModel):
    split: str = "test"
    engine_id: int
    contradiction_threshold: Optional[float] = None


@app.get("/api/health")
def health() -> Dict[str, Any]:
    meta_path = MODELS_DIR / "pipeline_metadata.json"
    artifacts_ready = meta_path.exists()
    data_ready = (DATA_DIR / "train_FD002.txt").exists() and (DATA_DIR / "test_FD002.txt").exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if artifacts_ready else {}
    return {
        "status": "ready" if (artifacts_ready and data_ready) else "degraded",
        "artifacts_ready": artifacts_ready,
        "data_ready": data_ready,
        "seed": meta.get("seed", 42),
        "default_threshold": meta.get("contradiction_threshold", 50.0),
        "useful_sensors": meta.get("useful_sensors", []),
        "modalities": meta.get("modalities", {}),
        "total_features": len(meta.get("all_features", [])),
    }


@app.get("/api/engines")
def list_engines(split: str = Query("test", pattern="^(train|test)$")) -> Dict[str, Any]:
    try:
        datasets = get_cached_datasets()
        df = datasets[split]
        engines = sorted([int(x) for x in df["engine_id"].unique()])
        counts = df.groupby("engine_id")["cycle"].count().to_dict()
        return {
            "split": split,
            "engines": engines,
            "cycle_counts": {int(k): int(v) for k, v in counts.items()},
            "total_rows": int(len(df)),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/predict")
def run_engine_inference(req: InferenceRequest) -> Dict[str, Any]:
    if req.split not in ("train", "test"):
        raise HTTPException(status_code=400, detail="Split must be 'train' or 'test'.")
    try:
        artifacts = get_cached_artifacts()
        datasets = get_cached_datasets()
        split_df = datasets[req.split]
        engine_df = split_df[split_df["engine_id"] == req.engine_id].copy()
        if engine_df.empty:
            raise HTTPException(status_code=404, detail=f"Engine {req.engine_id} not found in {req.split}.")

        preds_df, summary = predict_trajectories(
            engine_df,
            artifacts=artifacts,
            contradiction_threshold=req.contradiction_threshold,
            has_ground_truth_rul=(req.split == "train"),
        )

        # Clean NaN/Inf values for JSON serialization
        clean_df = preds_df.replace([np.inf, -np.inf], np.nan)
        records = clean_df.where(pd.notna(clean_df), None).to_dict(orient="records")

        # Include raw sensor/op settings alongside processed features for inspection
        meta = artifacts["metadata"]
        useful_sensors = meta.get("useful_sensors", [])
        visual_features = meta.get("visual_features", [])

        capped_df = engine_df.copy()
        if useful_sensors and "outlier_lower" in meta and "outlier_upper" in meta:
            lower = pd.Series(meta["outlier_lower"]).reindex(useful_sensors)
            upper = pd.Series(meta["outlier_upper"]).reindex(useful_sensors)
            capped_df[useful_sensors] = capped_df[useful_sensors].clip(lower=lower, upper=upper, axis=1)

        proc_df = add_temporal_features(capped_df.copy(), useful_sensors)
        pca = artifacts["pca"]
        if useful_sensors and visual_features:
            pca_vals = pca.transform(proc_df[useful_sensors])
            for idx, col in enumerate(visual_features):
                proc_df[col] = pca_vals[:, idx]

        modalities, all_features = define_modalities(useful_sensors, visual_features)
        modalities["visual"] = visual_features

        raw_records = (
            engine_df[["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + useful_sensors]
            .replace([np.inf, -np.inf], np.nan)
            .where(pd.notna(engine_df), None)
            .to_dict(orient="records")
        )

        engineered_cols = [c for c in all_features if c in proc_df.columns]
        engineered_records = (
            proc_df[["engine_id", "cycle"] + engineered_cols]
            .replace([np.inf, -np.inf], np.nan)
            .where(pd.notna(proc_df[["engine_id", "cycle"] + engineered_cols]), None)
            .to_dict(orient="records")
        )

        return {
            "split": req.split,
            "engine_id": int(req.engine_id),
            "summary": summary,
            "predictions": records,
            "raw_rows": raw_records,
            "engineered_rows": engineered_records,
            "modalities": modalities,
            "useful_sensors": useful_sensors,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc