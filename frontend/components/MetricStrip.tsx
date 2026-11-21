// frontend/components/MetricStrip.tsx
"use client";

import React from "react";
import { motion } from "framer-motion";

interface MetricStripProps {
  trustGateRul: number;
  failureRisk: number;
  disagreementStd: number;
  flaggedCount: number;
  totalCycles: number;
  threshold: number;
}

export function MetricStrip({
  trustGateRul,
  failureRisk,
  disagreementStd,
  flaggedCount,
  totalCycles,
  threshold,
}: MetricStripProps) {
  const isFlagged = disagreementStd > threshold;
  const riskTier =
    failureRisk >= 0.7 ? "Critical" : failureRisk >= 0.4 ? "Elevated" : "Normal";
  const riskColor =
    failureRisk >= 0.7
      ? "text-[#ef4444]"
      : failureRisk >= 0.4
        ? "text-[#f59e0b]"
        : "text-[#22c55e]";

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-12 gap-4">
      {/* Primary Metric: TrustGate RUL */}
      <motion.div
        key={`rul-${trustGateRul.toFixed(1)}`}
        initial={{ opacity: 0.75 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.2 }}
        className="xl:col-span-5 rounded-md border border-[#38BDF8]/30 bg-[#12161C] p-4 flex flex-col justify-between"
      >
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-semibold uppercase tracking-[0.08em] text-[#7DD3FC]">
            TrustGate Fused RUL
          </span>
          <span className="text-[11px] font-mono text-slate-400">ADAPTIVE FUSION</span>
        </div>
        <div className="mt-2 flex items-baseline gap-2">
          <span className="text-3xl font-semibold font-mono tracking-tight text-white">
            {trustGateRul.toFixed(1)}
          </span>
          <span className="text-xs font-mono text-slate-400">cycles</span>
        </div>
        <div className="mt-2 text-xs text-slate-400">
          Dynamic softmax weighting across sensor, visual proxy, and tabular modalities
        </div>
      </motion.div>

      {/* Failure Risk Score */}
      <div className="xl:col-span-2 rounded-md border border-slate-400/15 bg-[#12161C] p-4 flex flex-col justify-between">
        <span className="text-[11px] font-semibold uppercase tracking-[0.08em] text-slate-400">
          Failure Risk
        </span>
        <div className="mt-2">
          <div className="text-2xl font-semibold font-mono text-slate-100">
            {failureRisk.toFixed(3)}
          </div>
          <div className={`text-xs font-mono mt-1 ${riskColor}`}>{riskTier}</div>
        </div>
      </div>

      {/* Modality Disagreement Std */}
      <div className="xl:col-span-3 rounded-md border border-slate-400/15 bg-[#12161C] p-4 flex flex-col justify-between">
        <span className="text-[11px] font-semibold uppercase tracking-[0.08em] text-slate-400">
          Modality Spread (Std)
        </span>
        <div className="mt-2 flex items-baseline justify-between gap-2">
          <div className="flex items-baseline gap-1.5">
            <span className="text-2xl font-semibold font-mono text-slate-100">
              {disagreementStd.toFixed(1)}
            </span>
            <span className="text-xs font-mono text-slate-400">cyc</span>
          </div>
          <span
            className={`text-[11px] font-mono px-1.5 py-0.5 rounded ${
              isFlagged
                ? "bg-[#f59e0b]/15 text-[#fbbf24]"
                : "bg-[#22c55e]/15 text-[#4ade80]"
            }`}
          >
            {isFlagged ? `>${threshold}` : `<=${threshold}`}
          </span>
        </div>
      </div>

      {/* Flagged Cycles */}
      <div className="xl:col-span-2 rounded-md border border-slate-400/15 bg-[#12161C] p-4 flex flex-col justify-between">
        <span className="text-[11px] font-semibold uppercase tracking-[0.08em] text-slate-400">
          Flagged Cycles
        </span>
        <div className="mt-2">
          <div className="text-2xl font-semibold font-mono text-slate-100">
            {flaggedCount}
            <span className="text-sm text-slate-500">/{totalCycles}</span>
          </div>
          <div className="text-xs font-mono text-slate-400 mt-1">
            {totalCycles > 0 ? `${((flaggedCount / totalCycles) * 100).toFixed(1)}%` : "0.0%"}
          </div>
        </div>
      </div>
    </div>
  );
}