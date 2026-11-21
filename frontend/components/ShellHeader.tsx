// frontend/components/ShellHeader.tsx
"use client";

import React from "react";
import { Activity, Cpu, Menu } from "lucide-react";

interface ShellHeaderProps {
  split: "train" | "test";
  apiStatus: "ready" | "degraded" | "offline" | "loading";
  onToggleMobileNav: () => void;
  activeSectionLabel: string;
}

export function ShellHeader({
  split,
  apiStatus,
  onToggleMobileNav,
  activeSectionLabel,
}: ShellHeaderProps) {
  const statusColor =
    apiStatus === "ready"
      ? "bg-[#22c55e]"
      : apiStatus === "degraded"
        ? "bg-[#f59e0b]"
        : apiStatus === "loading"
          ? "bg-[#38bdf8]"
          : "bg-[#ef4444]";

  return (
    <header className="sticky top-0 z-30 h-14 border-b border-slate-400/15 bg-[#0B0D10]/95 backdrop-blur-sm px-4 lg:px-6 flex items-center justify-between gap-4">
      <div className="flex items-center gap-3 min-w-0">
        <button
          type="button"
          onClick={onToggleMobileNav}
          aria-label="Toggle navigation menu"
          className="lg:hidden p-1.5 rounded border border-slate-400/15 text-slate-300 hover:text-white hover:bg-[#171c24] transition-colors duration-150"
        >
          <Menu className="w-4 h-4" />
        </button>

        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded border border-[#38BDF8]/35 bg-[#38BDF8]/10 flex items-center justify-center text-[#38BDF8]">
            <Cpu className="w-4 h-4" strokeWidth={1.75} />
          </div>
          <div className="leading-tight">
            <div className="text-sm font-semibold tracking-tight text-slate-100">
              C-MAPSS Intelligence
            </div>
            <div className="text-[11px] text-slate-400 hidden sm:block">
              Predictive Engine Health Platform
            </div>
          </div>
        </div>

        <span className="hidden md:inline-block text-slate-600">/</span>
        <span className="hidden md:inline-block text-xs font-medium text-slate-300 truncate">
          {activeSectionLabel}
        </span>
      </div>

      <div className="flex items-center gap-3 text-xs font-mono">
        <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded border border-slate-400/15 bg-[#12161C] text-slate-300">
          <Activity className="w-3.5 h-3.5 text-[#38BDF8]" strokeWidth={1.75} />
          <span>FD002 · {split.toUpperCase()}</span>
        </div>

        <div className="flex items-center gap-2 px-2.5 py-1 rounded border border-slate-400/15 bg-[#12161C]">
          <span className={`w-2 h-2 rounded-full ${statusColor}`} />
          <span className="uppercase text-[11px] tracking-wider text-slate-300">
            {apiStatus}
          </span>
        </div>
      </div>
    </header>
  );
}