// frontend/app/page.tsx
"use client";

import React, { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle, CheckCircle2, Copy, Check } from "lucide-react";
import { ShellHeader } from "@/components/ShellHeader";
import { NAV_ITEMS, SidebarNav } from "@/components/SidebarNav";
import { MetricStrip } from "@/components/MetricStrip";
import {
  DisagreementAndRiskChart,
  RulTrajectoryChart,
  TrustWeightsChart,
} from "@/components/TrajectoryCharts";
import { fetchEngines, fetchHealth, postPredict } from "@/lib/api";
import { CyclePrediction, NavSectionId, PredictResponse } from "@/lib/types";

export default function HomePage() {
  const [activeSection, setActiveSection] = useState<NavSectionId>("overview");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  const [apiBaseUrl, setApiBaseUrl] = useState(
    process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
  );
  const [apiStatus, setApiStatus] = useState<"ready" | "degraded" | "offline" | "loading">(
    "loading"
  );
  const [split, setSplit] = useState<"train" | "test">("test");
  const [engines, setEngines] = useState<number[]>([]);
  const [selectedEngine, setSelectedEngine] = useState<number>(1);
  const [threshold, setThreshold] = useState<number>(50);
  const [selectedCycle, setSelectedCycle] = useState<number>(1);

  const [payload, setPayload] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<boolean>(false);

  // Dataset Explorer local state
  const [tableQuery, setTableQuery] = useState("");
  const [onlyFlagged, setOnlyFlagged] = useState(false);
  const [sortKey, setSortKey] = useState<keyof CyclePrediction>("cycle");
  const [sortAsc, setSortAsc] = useState<boolean>(true);
  const [page, setPage] = useState<number>(1);
  const [explorerMode, setExplorerMode] = useState<"predictions" | "raw" | "engineered">(
    "predictions"
  );
  const pageSize = 15;

  useEffect(() => {
    let active = true;
    fetchHealth(apiBaseUrl)
      .then((h) => {
        if (!active) return;
        setApiStatus(h.status);
        if (typeof h.default_threshold === "number") {
          setThreshold(h.default_threshold);
        }
      })
      .catch(() => {
        if (active) setApiStatus("offline");
      });
    return () => {
      active = false;
    };
  }, [apiBaseUrl]);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    fetchEngines(split, apiBaseUrl)
      .then((res) => {
        if (!active) return;
        setEngines(res.engines);
        const nextEngine = res.engines.includes(selectedEngine)
          ? selectedEngine
          : res.engines[0] ?? 1;
        setSelectedEngine(nextEngine);
      })
      .catch((err: Error) => {
        if (active) {
          setError(err.message);
          setLoading(false);
        }
      });
    return () => {
      active = false;
    };
  }, [split, apiBaseUrl]);

  useEffect(() => {
    if (!engines.length) return;
    let active = true;
    setLoading(true);
    setError(null);
    postPredict(
      {
        split,
        engine_id: selectedEngine,
        contradiction_threshold: threshold,
      },
      apiBaseUrl
    )
      .then((res) => {
        if (!active) return;
        setPayload(res);
        setApiStatus("ready");
        const lastCycle = res.predictions[res.predictions.length - 1]?.cycle ?? 1;
        setSelectedCycle(lastCycle);
        setPage(1);
        setLoading(false);
      })
      .catch((err: Error) => {
        if (active) {
          setError(err.message);
          setApiStatus("offline");
          setLoading(false);
        }
      });
    return () => {
      active = false;
    };
  }, [split, selectedEngine, threshold, apiBaseUrl, engines.length]);

  const predictions = payload?.predictions ?? [];
  const activeCycleRow = useMemo(() => {
    if (!predictions.length) return null;
    return (
      predictions.find((r) => r.cycle === selectedCycle) ||
      predictions[predictions.length - 1]
    );
  }, [predictions, selectedCycle]);

  const flaggedCount = useMemo(
    () => predictions.filter((r) => r.contradiction_flag).length,
    [predictions]
  );

  const activeNavLabel =
    NAV_ITEMS.find((n) => n.id === activeSection)?.label ?? "Overview";

  const copyEngineTag = () => {
    navigator.clipboard?.writeText(`FD002-${split.toUpperCase()}-ENG-${selectedEngine}`);
    setCopiedId(true);
    setTimeout(() => setCopiedId(false), 1400);
  };

  const filteredPredictions = useMemo(() => {
    const q = tableQuery.trim().toLowerCase();
    const rows = predictions.filter((r) => {
      if (onlyFlagged && !r.contradiction_flag) return false;
      if (!q) return true;
      return (
        String(r.cycle).includes(q) ||
        r.pred_rul_trustgate.toFixed(1).includes(q) ||
        r.failure_risk.toFixed(3).includes(q)
      );
    });
    return [...rows].sort((a, b) => {
      const av = Number(a[sortKey] ?? 0);
      const bv = Number(b[sortKey] ?? 0);
      return sortAsc ? av - bv : bv - av;
    });
  }, [predictions, tableQuery, onlyFlagged, sortKey, sortAsc]);

  const totalPages = Math.max(1, Math.ceil(filteredPredictions.length / pageSize));
  const paginatedPredictions = filteredPredictions.slice(
    (page - 1) * pageSize,
    page * pageSize
  );

  return (
    <div className="min-h-screen flex flex-col bg-[#0B0D10] text-[#E6EDF3]">
      <ShellHeader
        split={split}
        apiStatus={apiStatus}
        onToggleMobileNav={() => setMobileNavOpen((v) => !v)}
        activeSectionLabel={activeNavLabel}
      />

      <div className="flex flex-1 min-w-0">
        <SidebarNav
          activeSection={activeSection}
          onSelectSection={setActiveSection}
          collapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed((v) => !v)}
          mobileOpen={mobileNavOpen}
          onCloseMobile={() => setMobileNavOpen(false)}
        />

        <main className="flex-1 min-w-0 max-w-[1440px] w-full mx-auto px-4 lg:px-7 py-6">
          {/* Workspace Context Header */}
          <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 pb-5 border-b border-slate-400/15">
            <div>
              <div className="text-[11px] font-mono uppercase tracking-wider text-slate-500">
                NASA C-MAPSS · FD002 / {activeNavLabel}
              </div>
              <h1 className="text-xl font-semibold tracking-tight text-white mt-1">
                {activeNavLabel}
              </h1>
              <p className="text-xs text-slate-400 mt-0.5">
                Multi-regime turbofan degradation inference, TrustGate modality weighting, and disagreement alerts.
              </p>
            </div>

            {/* Contextual Engine & Split Controls */}
            <div className="flex flex-wrap items-center gap-2.5">
              <div className="inline-flex rounded border border-slate-400/15 bg-[#12161C] p-0.5 text-xs">
                {(["test", "train"] as const).map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => setSplit(s)}
                    className={`px-2.5 py-1 rounded transition-colors duration-150 ${
                      split === s
                        ? "bg-[#38BDF8]/15 text-[#7DD3FC] font-medium"
                        : "text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    {s.toUpperCase()}
                  </button>
                ))}
              </div>

              <label className="flex items-center gap-2 rounded border border-slate-400/15 bg-[#12161C] px-2.5 py-1 text-xs">
                <span className="text-slate-400">Engine</span>
                <select
                  value={selectedEngine}
                  onChange={(e) => setSelectedEngine(Number(e.target.value))}
                  className="bg-transparent font-mono text-slate-100 focus:outline-none"
                >
                  {engines.map((id) => (
                    <option key={id} value={id} className="bg-[#12161C]">
                      #{id}
                    </option>
                  ))}
                </select>
              </label>

              <button
                type="button"
                onClick={copyEngineTag}
                className="flex items-center gap-1.5 rounded border border-slate-400/15 bg-[#12161C] hover:bg-[#171c24] px-2.5 py-1.5 text-xs font-mono text-slate-300 transition-colors duration-150"
              >
                {copiedId ? (
                  <Check className="w-3.5 h-3.5 text-[#22c55e]" />
                ) : (
                  <Copy className="w-3.5 h-3.5 text-slate-400" />
                )}
                <span>ENG-{selectedEngine}</span>
              </button>
            </div>
          </div>

          {/* Compact Status Bar */}
          {activeCycleRow && (
            <div className="mt-4 py-2 px-3 rounded border border-slate-400/15 bg-[#12161C] flex flex-wrap items-center justify-between gap-2 text-xs font-mono text-slate-400">
              <div className="flex flex-wrap items-center gap-4">
                <span>
                  ENGINE <strong className="text-slate-200">#{selectedEngine}</strong>
                </span>
                <span>
                  CYCLE{" "}
                  <strong className="text-[#38BDF8]">{activeCycleRow.cycle}</strong> /{" "}
                  {predictions[predictions.length - 1]?.cycle ?? activeCycleRow.cycle}
                </span>
                <span>
                  NORM <strong className="text-slate-200">{activeCycleRow.cycle_norm.toFixed(2)}</strong>
                </span>
                <span>
                  SOURCE <strong className="text-slate-200">FD002_{split}.txt</strong>
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-slate-500">SCRUB CYCLE:</span>
                <input
                  type="range"
                  min={predictions[0]?.cycle ?? 1}
                  max={predictions[predictions.length - 1]?.cycle ?? 1}
                  value={activeCycleRow.cycle}
                  onChange={(e) => setSelectedCycle(Number(e.target.value))}
                  className="w-28 accent-[#38BDF8]"
                />
              </div>
            </div>
          )}

          {/* Error / Loading States */}
          {error && (
            <div className="mt-5 rounded border border-[#ef4444]/40 bg-[#ef4444]/10 p-4 text-xs text-red-200">
              <div className="font-semibold font-mono uppercase mb-1">Backend Connection Error</div>
              <div>{error}</div>
              <div className="mt-2 text-slate-400 font-mono">
                Start API: uvicorn api.main:app --reload --port 8000
              </div>
            </div>
          )}

          {loading && !payload && (
            <div className="mt-6 space-y-4">
              <div className="h-24 rounded-md bg-[#12161C] animate-pulse" />
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <div className="h-80 rounded-md bg-[#12161C] animate-pulse" />
                <div className="h-80 rounded-md bg-[#12161C] animate-pulse" />
              </div>
            </div>
          )}

          {activeCycleRow && (
            <AnimatePresence mode="wait">
              <motion.div
                key={`${activeSection}-${split}-${selectedEngine}`}
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.18 }}
                className="mt-5 space-y-5"
              >
                {/* Contradiction Alert when active cycle exceeds threshold */}
                {activeCycleRow.contradiction_flag ? (
                  <div className="rounded-md border border-[#f59e0b]/40 bg-[#f59e0b]/10 p-3.5 flex items-start gap-3">
                    <AlertTriangle className="w-4 h-4 text-[#f59e0b] shrink-0 mt-0.5" />
                    <div className="text-xs text-slate-200 space-y-1">
                      <div className="font-medium text-[#fbbf24]">
                        Inter-Modality Contradiction Detected at Cycle {activeCycleRow.cycle}
                      </div>
                      <div className="font-mono text-slate-300">
                        Disagreement: {activeCycleRow.modality_disagreement_std.toFixed(2)} cycles
                        (Threshold: {threshold.toFixed(0)} cycles) · TrustGate RUL:{" "}
                        {activeCycleRow.pred_rul_trustgate.toFixed(1)} cycles
                      </div>
                      <div className="text-slate-400">
                        Recommended action: Inspect individual modality estimates before scheduling
                        maintenance or relying on point RUL.
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="rounded-md border border-[#22c55e]/25 bg-[#22c55e]/5 px-3.5 py-2.5 flex items-center gap-2.5 text-xs text-slate-300">
                    <CheckCircle2 className="w-4 h-4 text-[#22c55e] shrink-0" />
                    <span>
                      Modalities aligned at cycle{" "}
                      <strong className="font-mono text-white">{activeCycleRow.cycle}</strong>{" "}
                      (spread{" "}
                      <strong className="font-mono text-white">
                        {activeCycleRow.modality_disagreement_std.toFixed(2)}
                      </strong>{" "}
                      cycles vs.{" "}
                      <strong className="font-mono text-white">{threshold.toFixed(0)}</strong>{" "}
                      threshold).
                    </span>
                  </div>
                )}

                {/* OVERVIEW & RUL ANALYSIS */}
                {(activeSection === "overview" || activeSection === "rul-analysis") && (
                  <>
                    <MetricStrip
                      trustGateRul={activeCycleRow.pred_rul_trustgate}
                      failureRisk={activeCycleRow.failure_risk}
                      disagreementStd={activeCycleRow.modality_disagreement_std}
                      flaggedCount={flaggedCount}
                      totalCycles={predictions.length}
                      threshold={threshold}
                    />

                    <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                      <RulTrajectoryChart
                        data={predictions}
                        selectedCycle={activeCycleRow.cycle}
                        onSelectCycle={setSelectedCycle}
                      />
                      <TrustWeightsChart
                        data={predictions}
                        selectedCycle={activeCycleRow.cycle}
                      />
                    </div>

                    <DisagreementAndRiskChart
                      data={predictions}
                      selectedCycle={activeCycleRow.cycle}
                      threshold={threshold}
                    />
                  </>
                )}

                {/* ENGINE DIAGNOSTICS */}
                {activeSection === "engine-diagnostics" && (
                  <div className="space-y-4">
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                      <div className="lg:col-span-2">
                        <RulTrajectoryChart
                          data={predictions}
                          selectedCycle={activeCycleRow.cycle}
                          onSelectCycle={setSelectedCycle}
                        />
                      </div>
                      <div className="rounded-md border border-slate-400/15 bg-[#12161C] p-4 space-y-3">
                        <div className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                          Cycle {activeCycleRow.cycle} Modality Breakdown
                        </div>
                        {[
                          {
                            name: "Sensor Modality (RF)",
                            val: activeCycleRow.pred_rul_sensor,
                            w: activeCycleRow.trust_weight_sensor,
                          },
                          {
                            name: "Tabular Context (RF)",
                            val: activeCycleRow.pred_rul_tabular,
                            w: activeCycleRow.trust_weight_tabular,
                          },
                          {
                            name: "Visual Proxy PCA (RF)",
                            val: activeCycleRow.pred_rul_visual,
                            w: activeCycleRow.trust_weight_visual,
                          },
                        ].map((m) => (
                          <div
                            key={m.name}
                            className="p-2.5 rounded border border-slate-400/10 bg-[#171c24] text-xs"
                          >
                            <div className="flex justify-between text-slate-200 font-medium">
                              <span>{m.name}</span>
                              <span className="font-mono">{m.val.toFixed(1)} c</span>
                            </div>
                            <div className="text-[11px] font-mono text-slate-400 mt-1">
                              Gate Weight: {(m.w * 100).toFixed(1)}%
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    <DisagreementAndRiskChart
                      data={predictions}
                      selectedCycle={activeCycleRow.cycle}
                      threshold={threshold}
                    />
                  </div>
                )}

                {/* MULTIMODAL INTELLIGENCE */}
                {activeSection === "multimodal-intelligence" && (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    <TrustWeightsChart
                      data={predictions}
                      selectedCycle={activeCycleRow.cycle}
                    />
                    <div className="rounded-md border border-slate-400/15 bg-[#12161C] p-4 space-y-3">
                      <h3 className="text-sm font-medium text-white">
                        Feature Modalities (FD002)
                      </h3>
                      {Object.entries(payload?.modalities ?? {}).map(([group, cols]) => (
                        <div
                          key={group}
                          className="p-3 rounded border border-slate-400/10 bg-[#171c24]"
                        >
                          <div className="flex items-center justify-between text-xs">
                            <span className="font-mono uppercase text-[#7DD3FC]">{group}</span>
                            <span className="font-mono text-slate-400">{cols.length} features</span>
                          </div>
                          <div className="text-[11px] font-mono text-slate-400 mt-1.5 break-words">
                            {cols.slice(0, 8).join(", ")}
                            {cols.length > 8 ? ` +${cols.length - 8} more` : ""}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* TRUST & RELIABILITY CENTER */}
                {activeSection === "trust-reliability" && (
                  <div className="space-y-4">
                    <div className="rounded-md border border-[#38BDF8]/25 bg-[#12161C] p-4 space-y-3">
                      <h2 className="text-base font-semibold text-white">
                        Why does TrustGate trust this prediction?
                      </h2>
                      <p className="text-xs text-slate-400 leading-relaxed">
                        TrustGateNetwork evaluates standardized operating context, cycle progression,
                        and rolling sensor stability to output softmax modality weights. Below are
                        the exact computed values for Engine #{selectedEngine} at Cycle{" "}
                        {activeCycleRow.cycle}:
                      </p>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs font-mono">
                        <div className="p-3 rounded border border-slate-400/10 bg-[#171c24]">
                          <div className="text-slate-400">Sensor Trust Weight</div>
                          <div className="text-lg text-white mt-1">
                            {(activeCycleRow.trust_weight_sensor * 100).toFixed(1)}%
                          </div>
                        </div>
                        <div className="p-3 rounded border border-slate-400/10 bg-[#171c24]">
                          <div className="text-slate-400">Tabular Trust Weight</div>
                          <div className="text-lg text-white mt-1">
                            {(activeCycleRow.trust_weight_tabular * 100).toFixed(1)}%
                          </div>
                        </div>
                        <div className="p-3 rounded border border-slate-400/10 bg-[#171c24]">
                          <div className="text-slate-400">Visual Proxy Trust Weight</div>
                          <div className="text-lg text-white mt-1">
                            {(activeCycleRow.trust_weight_visual * 100).toFixed(1)}%
                          </div>
                        </div>
                      </div>
                      <div className="text-xs text-slate-300 border-t border-slate-400/10 pt-3">
                        <strong>Flag Reason:</strong>{" "}
                        {activeCycleRow.contradiction_flag
                          ? `Modality prediction standard deviation (${activeCycleRow.modality_disagreement_std.toFixed(
                              2
                            )} cycles) is above the configured ${threshold.toFixed(
                              0
                            )}-cycle threshold.`
                          : `Modality prediction standard deviation (${activeCycleRow.modality_disagreement_std.toFixed(
                              2
                            )} cycles) is within the ${threshold.toFixed(0)}-cycle threshold.`}
                      </div>
                    </div>

                    <TrustWeightsChart
                      data={predictions}
                      selectedCycle={activeCycleRow.cycle}
                    />
                  </div>
                )}

                {/* MODEL INSIGHTS */}
                {activeSection === "model-insights" && (
                  <div className="rounded-md border border-slate-400/15 bg-[#12161C] p-5 space-y-4">
                    <h2 className="text-base font-semibold text-white">
                      Pipeline Architecture &amp; Inference Flow
                    </h2>
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs font-mono">
                      {[
                        {
                          step: "01 · INGESTION",
                          desc: "FD002 trajectories; drops sensor_20, sensor_21, low-variance sensor_16.",
                        },
                        {
                          step: "02 · FEATURES",
                          desc: "Train-fitted 3-sigma bounds, rolling stats/slope, PCA visual proxy, regime context.",
                        },
                        {
                          step: "03 · FUSION",
                          desc: "Per-modality Random Forests weighted dynamically by TrustGateNetwork.",
                        },
                        {
                          step: "04 · DIAGNOSTICS",
                          desc: "Exponential risk score [0-1], NASA asymmetric score, modality std contradiction flag.",
                        },
                      ].map((s) => (
                        <div
                          key={s.step}
                          className="p-3.5 rounded border border-slate-400/15 bg-[#171c24]"
                        >
                          <div className="text-[#38BDF8] font-semibold">{s.step}</div>
                          <div className="text-slate-300 font-sans mt-1.5 leading-relaxed">
                            {s.desc}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* DATASET EXPLORER */}
                {activeSection === "dataset-explorer" && (
                  <div className="rounded-md border border-slate-400/15 bg-[#12161C] p-4 space-y-3">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-2 text-xs">
                        {(["predictions", "raw", "engineered"] as const).map((m) => (
                          <button
                            key={m}
                            type="button"
                            onClick={() => {
                              setExplorerMode(m);
                              setPage(1);
                            }}
                            className={`px-2.5 py-1 rounded border font-mono capitalize ${
                              explorerMode === m
                                ? "border-[#38BDF8]/40 bg-[#38BDF8]/15 text-[#7DD3FC]"
                                : "border-slate-400/15 text-slate-400"
                            }`}
                          >
                            {m}
                          </button>
                        ))}
                      </div>

                      <div className="flex flex-wrap items-center gap-2 text-xs">
                        <input
                          type="text"
                          value={tableQuery}
                          onChange={(e) => {
                            setTableQuery(e.target.value);
                            setPage(1);
                          }}
                          placeholder="Filter cycle or value..."
                          className="rounded border border-slate-400/15 bg-[#0B0D10] px-2.5 py-1 text-slate-200 focus:outline-none focus:border-[#38BDF8]"
                        />
                        <label className="flex items-center gap-1.5 text-slate-300">
                          <input
                            type="checkbox"
                            checked={onlyFlagged}
                            onChange={(e) => {
                              setOnlyFlagged(e.target.checked);
                              setPage(1);
                            }}
                          />
                          <span>Flagged only</span>
                        </label>
                      </div>
                    </div>

                    {explorerMode === "predictions" ? (
                      <div className="overflow-x-auto max-h-[440px]">
                        <table className="w-full text-left border-collapse text-xs font-mono">
                          <thead className="sticky top-0 bg-[#171c24] text-slate-400 border-b border-slate-400/15">
                            <tr>
                              {(
                                [
                                  ["cycle", "Cycle"],
                                  ["pred_rul_trustgate", "TrustGate RUL"],
                                  ["pred_rul_sensor", "Sensor RF"],
                                  ["pred_rul_tabular", "Tabular RF"],
                                  ["failure_risk", "Risk"],
                                  ["modality_disagreement_std", "Std"],
                                ] as [keyof CyclePrediction, string][]
                              ).map(([k, label]) => (
                                <th
                                  key={k}
                                  onClick={() => {
                                    if (sortKey === k) setSortAsc(!sortAsc);
                                    else {
                                      setSortKey(k);
                                      setSortAsc(true);
                                    }
                                  }}
                                  className="py-2 px-3 cursor-pointer hover:text-white select-none"
                                >
                                  {label}
                                </th>
                              ))}
                              <th className="py-2 px-3">Flag</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-400/10">
                            {paginatedPredictions.map((row) => (
                              <tr
                                key={row.cycle}
                                onClick={() => setSelectedCycle(row.cycle)}
                                className={`cursor-pointer transition-colors ${
                                  row.cycle === activeCycleRow.cycle
                                    ? "bg-[#38BDF8]/10"
                                    : "hover:bg-[#171c24]"
                                }`}
                              >
                                <td className="py-2 px-3 text-slate-200">{row.cycle}</td>
                                <td className="py-2 px-3 text-[#7DD3FC]">
                                  {row.pred_rul_trustgate.toFixed(2)}
                                </td>
                                <td className="py-2 px-3">{row.pred_rul_sensor.toFixed(2)}</td>
                                <td className="py-2 px-3">{row.pred_rul_tabular.toFixed(2)}</td>
                                <td className="py-2 px-3">{row.failure_risk.toFixed(3)}</td>
                                <td className="py-2 px-3">
                                  {row.modality_disagreement_std.toFixed(2)}
                                </td>
                                <td className="py-2 px-3">
                                  {row.contradiction_flag ? (
                                    <span className="text-[#fbbf24]">FLAGGED</span>
                                  ) : (
                                    <span className="text-slate-500">OK</span>
                                  )}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <div className="overflow-x-auto max-h-[440px]">
                        <table className="w-full text-left border-collapse text-xs font-mono">
                          <thead className="sticky top-0 bg-[#171c24] text-slate-400 border-b border-slate-400/15">
                            <tr>
                              {Object.keys(
                                (explorerMode === "raw"
                                  ? payload?.raw_rows?.[0]
                                  : payload?.engineered_rows?.[0]) ?? {}
                              )
                                .slice(0, 10)
                                .map((col) => (
                                  <th key={col} className="py-2 px-3">
                                    {col}
                                  </th>
                                ))}
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-400/10">
                            {(
                              (explorerMode === "raw"
                                ? payload?.raw_rows
                                : payload?.engineered_rows) ?? []
                            )
                              .slice((page - 1) * pageSize, page * pageSize)
                              .map((r, idx) => (
                                <tr key={idx} className="hover:bg-[#171c24]">
                                  {Object.values(r)
                                    .slice(0, 10)
                                    .map((val, cIdx) => (
                                      <td key={cIdx} className="py-2 px-3 text-slate-300">
                                        {typeof val === "number" ? val.toFixed(3) : String(val ?? "")}
                                      </td>
                                    ))}
                                </tr>
                              ))}
                          </tbody>
                        </table>
                      </div>
                    )}

                    <div className="flex items-center justify-between pt-2 text-xs font-mono text-slate-400">
                      <span>
                        Page {page} of {totalPages} ({filteredPredictions.length} rows)
                      </span>
                      <div className="flex gap-2">
                        <button
                          type="button"
                          disabled={page <= 1}
                          onClick={() => setPage((p) => Math.max(1, p - 1))}
                          className="px-2.5 py-1 rounded border border-slate-400/15 disabled:opacity-40"
                        >
                          Prev
                        </button>
                        <button
                          type="button"
                          disabled={page >= totalPages}
                          onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                          className="px-2.5 py-1 rounded border border-slate-400/15 disabled:opacity-40"
                        >
                          Next
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                {/* SYSTEM SETTINGS */}
                {activeSection === "system-settings" && (
                  <div className="rounded-md border border-slate-400/15 bg-[#12161C] p-5 space-y-4 max-w-2xl">
                    <h2 className="text-base font-semibold text-white">
                      Runtime &amp; Threshold Configuration
                    </h2>
                    <div className="space-y-1.5 text-xs">
                      <label className="block text-slate-400">Inference API Endpoint</label>
                      <input
                        type="text"
                        value={apiBaseUrl}
                        onChange={(e) => setApiBaseUrl(e.target.value)}
                        className="w-full rounded border border-slate-400/15 bg-[#0B0D10] px-3 py-2 font-mono text-slate-100 focus:outline-none focus:border-[#38BDF8]"
                      />
                    </div>
                    <div className="space-y-1.5 text-xs">
                      <div className="flex justify-between font-mono text-slate-300">
                        <span>Contradiction Threshold</span>
                        <span>{threshold.toFixed(0)} cycles</span>
                      </div>
                      <input
                        type="range"
                        min={10}
                        max={120}
                        step={5}
                        value={threshold}
                        onChange={(e) => setThreshold(Number(e.target.value))}
                        className="w-full accent-[#38BDF8]"
                      />
                    </div>
                  </div>
                )}
              </motion.div>
            </AnimatePresence>
          )}
        </main>
      </div>

      {/* Footer */}
      <footer className="border-t border-slate-400/15 bg-[#0B0D10] px-4 lg:px-7 py-3 text-[11px] font-mono text-slate-500 flex flex-wrap items-center justify-between gap-2">
        <div>C-MAPSS Intelligence · NASA C-MAPSS FD002 Multimodal RUL Platform</div>
        <div>Dataset: NASA Prognostics Center of Excellence (C-MAPSS FD002) · Status: {apiStatus}</div>
      </footer>
    </div>
  );
}