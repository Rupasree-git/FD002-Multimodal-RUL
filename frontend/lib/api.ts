// frontend/lib/api.ts
import { EnginesResponse, HealthResponse, PredictResponse } from "./types";

const DEFAULT_API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function fetchHealth(baseUrl = DEFAULT_API_BASE): Promise<HealthResponse> {
  const res = await fetch(`${baseUrl}/api/health`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Health check failed (${res.status})`);
  return res.json();
}

export async function fetchEngines(
  split: "train" | "test",
  baseUrl = DEFAULT_API_BASE
): Promise<EnginesResponse> {
  const res = await fetch(`${baseUrl}/api/engines?split=${split}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to load engines (${res.status})`);
  return res.json();
}

export async function postPredict(
  params: {
    split: "train" | "test";
    engine_id: number;
    contradiction_threshold?: number;
  },
  baseUrl = DEFAULT_API_BASE
): Promise<PredictResponse> {
  const res = await fetch(`${baseUrl}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const errBody = await res.json().catch(() => ({}));
    throw new Error(errBody.detail || `Inference request failed (${res.status})`);
  }
  return res.json();
}