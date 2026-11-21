// frontend/lib/types.ts
export type NavSectionId =
  | "overview"
  | "engine-diagnostics"
  | "rul-analysis"
  | "multimodal-intelligence"
  | "trust-reliability"
  | "model-insights"
  | "dataset-explorer"
  | "system-settings";

export interface CyclePrediction {
  engine_id: number;
  cycle: number;
  cycle_norm: number;
  pred_rul_trustgate: number;
  pred_rul_sensor: number;
  pred_rul_visual: number;
  pred_rul_tabular: number;
  failure_risk: number;
  modality_disagreement_std: number;
  contradiction_flag: boolean;
  trust_weight_sensor: number;
  trust_weight_visual: number;
  trust_weight_tabular: number;
  true_rul?: number;
  abs_error?: number;
}

export interface InferenceSummary {
  num_rows: number;
  num_engines: number;
  contradiction_threshold: number;
  contradiction_rate: number;
  mean_pred_rul: number;
  mean_failure_risk: number;
  mae?: number;
  nasa_score?: number;
}

export interface PredictResponse {
  split: "train" | "test";
  engine_id: number;
  summary: InferenceSummary;
  predictions: CyclePrediction[];
  raw_rows: Record<string, number | null>[];
  engineered_rows: Record<string, number | null>[];
  modalities: Record<string, string[]>;
  useful_sensors: string[];
}

export interface EnginesResponse {
  split: "train" | "test";
  engines: number[];
  cycle_counts: Record<string, number>;
  total_rows: number;
}

export interface HealthResponse {
  status: "ready" | "degraded";
  artifacts_ready: boolean;
  data_ready: boolean;
  seed: number;
  default_threshold: number;
  useful_sensors: string[];
  modalities: Record<string, string[]>;
  total_features: number;
}