"use client";

import React from "react";
import {
  Activity,
  BarChart3,
  ChevronLeft,
  ChevronRight,
  Cpu,
  Database,
  Layers,
  LayoutDashboard,
  ShieldCheck,
  Sliders,
  X,
} from "lucide-react";
import { NavSectionId } from "@/lib/types";

export const NAV_ITEMS: {
  id: NavSectionId;
  label: string;
  icon: React.ComponentType<{ className?: string; strokeWidth?: number }>;
}[] = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "engine-diagnostics", label: "Engine Diagnostics", icon: Activity },
  { id: "rul-analysis", label: "RUL Analysis", icon: BarChart3 },
  { id: "multimodal-intelligence", label: "Multimodal Intelligence", icon: Layers },
  { id: "trust-reliability", label: "Trust & Reliability", icon: ShieldCheck },
  { id: "model-insights", label: "Model Insights", icon: Cpu },
  { id: "dataset-explorer", label: "Dataset Explorer", icon: Database },
  { id: "system-settings", label: "System Settings", icon: Sliders },
];

interface SidebarNavProps {
  activeSection: NavSectionId;
  onSelectSection: (id: NavSectionId) => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
  mobileOpen: boolean;
  onCloseMobile: () => void;
}

export function SidebarNav({
  activeSection,
  onSelectSection,
  collapsed,
  onToggleCollapse,
  mobileOpen,
  onCloseMobile,
}: SidebarNavProps) {
  const navContent = (
    <div className="flex flex-col h-full justify-between py-3">
      <div className="px-2.5 space-y-0.5">
        {!collapsed && (
          <div className="px-2.5 pb-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-slate-500">
            Workspace
          </div>
        )}

        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const active = activeSection === item.id;
          return (
            <button
              key={item.id}
              type="button"
              title={collapsed ? item.label : undefined}
              onClick={() => {
                onSelectSection(item.id);
                onCloseMobile();
              }}
              className={`w-full flex items-center gap-2.5 px-2.5 py-2 rounded text-[13px] transition-colors duration-150 ${
                active
                  ? "bg-[#38BDF8]/12 text-[#7DD3FC] font-medium border border-[#38BDF8]/25"
                  : "text-slate-400 hover:text-slate-100 hover:bg-[#171c24] border border-transparent"
              } ${collapsed ? "justify-center" : "justify-start"}`}
            >
              <Icon
                className={`w-4 h-4 shrink-0 ${active ? "text-[#38BDF8]" : "text-slate-400"}`}
                strokeWidth={1.75}
              />
              {!collapsed && <span className="truncate">{item.label}</span>}
            </button>
          );
        })}
      </div>

      <div className="hidden lg:flex px-2.5 pt-3 border-t border-slate-400/15">
        <button
          type="button"
          onClick={onToggleCollapse}
          className="w-full flex items-center justify-center gap-2 py-1.5 rounded text-xs text-slate-400 hover:text-slate-200 hover:bg-[#171c24] transition-colors duration-150"
        >
          {collapsed ? (
            <ChevronRight className="w-4 h-4" />
          ) : (
            <>
              <ChevronLeft className="w-4 h-4" />
              <span>Collapse</span>
            </>
          )}
        </button>
      </div>
    </div>
  );

  return (
    <>
      <aside
        className={`hidden lg:block shrink-0 border-r border-slate-400/15 bg-[#12161C] transition-[width] duration-200 ${
          collapsed ? "w-16" : "w-60"
        }`}
      >
        <div className="sticky top-14 h-[calc(100vh-3.5rem)] overflow-y-auto">
          {navContent}
        </div>
      </aside>

      {mobileOpen && (
        <div className="fixed inset-0 z-40 lg:hidden flex">
          <div
            className="fixed inset-0 bg-black/60"
            onClick={onCloseMobile}
            aria-hidden="true"
          />
          <aside className="relative z-50 w-64 max-w-[80vw] bg-[#12161C] border-r border-slate-400/15 h-full">
            <div className="h-14 px-4 border-b border-slate-400/15 flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Navigation
              </span>
              <button
                type="button"
                onClick={onCloseMobile}
                aria-label="Close navigation"
                className="p-1 text-slate-400 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            {navContent}
          </aside>
        </div>
      )}
    </>
  );
}