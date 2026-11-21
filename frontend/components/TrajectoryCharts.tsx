// frontend/components/TrajectoryCharts.tsx
"use client";

import React, { useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { CyclePrediction } from "@/lib/types";

interface RulChartProps {
  data: CyclePrediction[];
  selectedCycle: number;
  onSelectCycle?: (cycle: number) => void;
}

export function RulTrajectoryChart({ data, selectedCycle, onSelectCycle }: RulChartProps) {
  const [visible, setVisible] = useState({
    pred_rul_trustgate: true,
    pred_rul_sensor: true,
    pred_rul_tabular: true,
    pred_rul_visual: false,
    true_rul: true,
  });

  const toggleSeries = (key: keyof typeof visible) =>
    setVisible((prev) => ({ ...prev, [key]: !prev[key] }));

  const hasGroundTruth = data.some((d) => typeof d.true_rul === "number" && d.true_rul !== null);

  return (
    <div className="rounded-md border border-slate-400/15 bg-[#12161C] p-4">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <div>
          <h3 className="text-sm font-medium text-slate-100">RUL Trajectory Across Cycles</h3>
          <p className="text-xs text-slate-400">
            Click or hover cycles to inspect modality spread; toggle series below
          </p>
        </div>
        <div className="flex flex-wrap gap-1.5 text-[11px] font-mono">
          {[
            { key: "pred_rul_trustgate", label: "TrustGate", color: "#38BDF8" },
            { key: "pred_rul_sensor", label: "Sensor RF", color: "#94A3B8" },
            { key: "pred_rul_tabular", label: "Tabular RF", color: "#22C55E" },
            { key: "pred_rul_visual", label: "Visual PCA", color: "#A78BFA" },
            ...(hasGroundTruth
              ? [{ key: "true_rul", label: "Ground Truth", color: "#F59E0B" }]
              : []),
          ].map((item) => {
            const k = item.key as keyof typeof visible;
            const active = visible[k];
            return (
              <button
                key={item.key}
                type="button"
                onClick={() => toggleSeries(k)}
                className={`px-2 py-0.5 rounded border transition-colors duration-150 flex items-center gap-1.5 ${
                  active
                    ? "border-slate-400/30 bg-[#171c24] text-slate-200"
                    : "border-slate-400/10 text-slate-500"
                }`}
              >
                <span
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: active ? item.color : "#475569" }}
                />
                {item.label}
              </button>
            );
          })}
        </div>
      </div>

      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 8, right: 12, left: 0, bottom: 4 }}
            onClick={(state) => {
              if (onSelectCycle && state?.activeLabel !== undefined) {
                const c = Number(state.activeLabel);
                if (!Number.isNaN(c)) onSelectCycle(c);
              }
            }}
          >
            <CartesianGrid stroke="rgba(148, 163, 184, 0.08)" strokeDasharray="3 3" />
            <XAxis
              dataKey="cycle"
              stroke="#64748b"
              tick={{ fill: "#94a3b8", fontSize: 11, fontFamily: "var(--font-mono)" }}
              name="Cycle"
            />
            <YAxis
              stroke="#64748b"
              tick={{ fill: "#94a3b8", fontSize: 11, fontFamily: "var(--font-mono)" }}
              unit="c"
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#0B0D10",
                borderColor: "rgba(148, 163, 184, 0.25)",
                borderRadius: "6px",
                fontSize: "12px",
                fontFamily: "var(--font-mono)",
              }}
              labelFormatter={(v) => `Cycle ${v}`}
            />
            <ReferenceLine
              x={selectedCycle}
              stroke="#38BDF8"
              strokeDasharray="4 3"
              strokeWidth={1.5}
            />
            {visible.pred_rul_trustgate && (
              <Line
                type="monotone"
                dataKey="pred_rul_trustgate"
                name="TrustGate RUL"
                stroke="#38BDF8"
                strokeWidth={2.2}
                dot={false}
                isAnimationActive={false}
              />
            )}
            {visible.pred_rul_sensor && (
              <Line
                type="monotone"
                dataKey="pred_rul_sensor"
                name="Sensor RF"
                stroke="#94A3B8"
                strokeWidth={1.5}
                dot={false}
                isAnimationActive={false}
              />
            )}
            {visible.pred_rul_tabular && (
              <Line
                type="monotone"
                dataKey="pred_rul_tabular"
                name="Tabular RF"
                stroke="#22C55E"
                strokeWidth={1.5}
                dot={false}
                isAnimationActive={false}
              />
            )}
            {visible.pred_rul_visual && (
              <Line
                type="monotone"
                dataKey="pred_rul_visual"
                name="Visual PCA"
                stroke="#A78BFA"
                strokeWidth={1.5}
                dot={false}
                isAnimationActive={false}
              />
            )}
            {hasGroundTruth && visible.true_rul && (
              <Line
                type="monotone"
                dataKey="true_rul"
                name="Ground Truth RUL"
                stroke="#F59E0B"
                strokeWidth={1.8}
                strokeDasharray="5 3"
                dot={false}
                isAnimationActive={false}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export function TrustWeightsChart({
  data,
  selectedCycle,
}: {
  data: CyclePrediction[];
  selectedCycle: number;
}) {
  const normalized = data.map((row) => {
    const s = Math.max(0, row.trust_weight_sensor ?? 0);
    const v = Math.max(0, row.trust_weight_visual ?? 0);
    const t = Math.max(0, row.trust_weight_tabular ?? 0);
    const total = s + v + t || 1;
    return {
      ...row,
      w_sensor_pct: Number(((s / total) * 100).toFixed(1)),
      w_visual_pct: Number(((v / total) * 100).toFixed(1)),
      w_tabular_pct: Number(((t / total) * 100).toFixed(1)),
    };
  });

  return (
    <div className="rounded-md border border-slate-400/15 bg-[#12161C] p-4">
      <div className="mb-3">
        <h3 className="text-sm font-medium text-slate-100">TrustGate Modality Weights</h3>
        <p className="text-xs text-slate-400">
          Actual softmax modality allocation across operating cycles (%)
        </p>
      </div>

      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={normalized} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
            <CartesianGrid stroke="rgba(148, 163, 184, 0.08)" strokeDasharray="3 3" />
            <XAxis
              dataKey="cycle"
              stroke="#64748b"
              tick={{ fill: "#94a3b8", fontSize: 11, fontFamily: "var(--font-mono)" }}
            />
            <YAxis
              domain={[0, 100]}
              stroke="#64748b"
              tick={{ fill: "#94a3b8", fontSize: 11, fontFamily: "var(--font-mono)" }}
              unit="%"
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#0B0D10",
                borderColor: "rgba(148, 163, 184, 0.25)",
                borderRadius: "6px",
                fontSize: "12px",
                fontFamily: "var(--font-mono)",
              }}
              labelFormatter={(v) => `Cycle ${v}`}
            />
            <ReferenceLine x={selectedCycle} stroke="#38BDF8" strokeDasharray="4 3" />
            <Area
              type="monotone"
              dataKey="w_sensor_pct"
              name="Sensor Weight (%)"
              stackId="1"
              stroke="#38BDF8"
              fill="#38BDF8"
              fillOpacity={0.35}
              isAnimationActive={false}
            />
            <Area
              type="monotone"
              dataKey="w_tabular_pct"
              name="Tabular Weight (%)"
              stackId="1"
              stroke="#22C55E"
              fill="#22C55E"
              fillOpacity={0.3}
              isAnimationActive={false}
            />
            <Area
              type="monotone"
              dataKey="w_visual_pct"
              name="Visual Proxy Weight (%)"
              stackId="1"
              stroke="#A78BFA"
              fill="#A78BFA"
              fillOpacity={0.3}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export function DisagreementAndRiskChart({
  data,
  selectedCycle,
  threshold,
}: {
  data: CyclePrediction[];
  selectedCycle: number;
  threshold: number;
}) {
  return (
    <div className="rounded-md border border-slate-400/15 bg-[#12161C] p-4">
      <div className="mb-3">
        <h3 className="text-sm font-medium text-slate-100">
          Inter-Modality Disagreement &amp; Failure Risk
        </h3>
        <p className="text-xs text-slate-400">
          Modality standard deviation against threshold ({threshold.toFixed(0)} cycles) and [0, 1] risk
        </p>
      </div>

      <div className="h-60 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
            <CartesianGrid stroke="rgba(148, 163, 184, 0.08)" strokeDasharray="3 3" />
            <XAxis
              dataKey="cycle"
              stroke="#64748b"
              tick={{ fill: "#94a3b8", fontSize: 11, fontFamily: "var(--font-mono)" }}
            />
            <YAxis
              yAxisId="left"
              stroke="#64748b"
              tick={{ fill: "#94a3b8", fontSize: 11, fontFamily: "var(--font-mono)" }}
              unit="c"
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              domain={[0, 1]}
              stroke="#64748b"
              tick={{ fill: "#94a3b8", fontSize: 11, fontFamily: "var(--font-mono)" }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#0B0D10",
                borderColor: "rgba(148, 163, 184, 0.25)",
                borderRadius: "6px",
                fontSize: "12px",
                fontFamily: "var(--font-mono)",
              }}
              labelFormatter={(v) => `Cycle ${v}`}
            />
            <ReferenceLine x={selectedCycle} yAxisId="left" stroke="#38BDF8" strokeDasharray="4 3" />
            <ReferenceLine
              y={threshold}
              yAxisId="left"
              stroke="#ef4444"
              strokeDasharray="5 4"
              label={{
                value: `Threshold (${threshold}c)`,
                fill: "#f87171",
                fontSize: 11,
                position: "insideTopRight",
              }}
            />
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="modality_disagreement_std"
              name="Modality Std (cycles)"
              stroke="#38BDF8"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="failure_risk"
              name="Failure Risk [0-1]"
              stroke="#f59e0b"
              strokeWidth={1.8}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}