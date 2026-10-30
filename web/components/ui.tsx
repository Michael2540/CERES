/* eslint-disable @next/next/no-img-element */
"use client";

import { useState } from "react";
import type { PointSource } from "@/lib/types";

export const SOURCE_COLOR: Record<string, string> = {
  dataset: "#34d399",
  fault_transform: "#fb923c",
  fault_timewarp: "#f472b6",
  simulated: "#38bdf8",
  derived: "#a78bfa",
  sensor: "#4ade80",
  replay: "#2dd4bf",
};

export function sourceLabel(source: PointSource | string): string {
  switch (source) {
    case "dataset": return "DATASET";
    case "fault_transform": return "SIMULATED FAULT · image transform";
    case "fault_timewarp": return "SIMULATED FAULT · time-warp";
    case "simulated": return "SIMULATED";
    case "derived": return "DERIVED";
    case "sensor": return "SENSOR";
    case "replay": return "REPLAY";
    default: return String(source).toUpperCase();
  }
}

export const isFault = (s: string) => s === "fault_transform" || s === "fault_timewarp";

export function SourceBadge({ source, testId }: { source: PointSource | string; testId?: string }) {
  const c = SOURCE_COLOR[source] ?? "#94a3b8";
  return (
    <span
      data-testid={testId}
      className="inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-[10px] font-semibold tracking-wide whitespace-nowrap"
      style={{ borderColor: c, color: c, background: `${c}18` }}
    >
      {sourceLabel(source)}
    </span>
  );
}

const STATUS: Record<string, string> = {
  nominal: "border-emerald-500 text-emerald-300",
  watch: "border-amber-500 text-amber-300",
  warning: "border-amber-500 text-amber-300",
  alert: "border-red-500 text-red-300",
  critical: "border-red-500 text-red-300",
};
export function StatusChip({ status }: { status: string }) {
  return (
    <span className={`rounded-full border px-2 py-0.5 text-[11px] font-semibold uppercase ${STATUS[status] ?? "border-slate-500 text-slate-300"}`}>
      {status}
    </span>
  );
}

export function Legend() {
  const items: [string, string][] = [
    ["dataset", "real ALED frame"],
    ["fault_transform", "real frame + image transform"],
    ["fault_timewarp", "real frames, time-warped"],
    ["simulated", "simulated sensor"],
    ["derived", "computed from others"],
  ];
  return (
    <details className="relative" data-testid="legend">
      <summary className="cursor-pointer list-none rounded border border-slate-600 px-2 py-1 text-xs text-slate-300 hover:bg-slate-800">
        Legend
      </summary>
      <div className="absolute right-0 z-40 mt-1 w-72 rounded-lg border border-slate-600 bg-slate-900 p-3 shadow-xl">
        <p className="mb-2 text-xs text-slate-400">Provenance badges. Images are real; sensors and faults are simulated.</p>
        <ul className="space-y-1.5">
          {items.map(([s, d]) => (
            <li key={s} className="flex items-center justify-between gap-2 text-xs text-slate-300">
              <SourceBadge source={s} />
              <span className="text-right">{d}</span>
            </li>
          ))}
        </ul>
      </div>
    </details>
  );
}

export function Img({ src, alt, className }: { src: string; alt: string; className?: string }) {
  const [failed, setFailed] = useState<string | null>(null);
  if (failed === src) {
    return (
      <div className={`flex items-center justify-center bg-slate-800 text-xs text-slate-500 ${className ?? ""}`} role="img" aria-label={alt}>
        image unavailable
      </div>
    );
  }
  return <img src={src} alt={alt} className={className} onError={() => setFailed(src)} />;
}

export function Panel({ title, children, right, className }: { title?: string; children: React.ReactNode; right?: React.ReactNode; className?: string }) {
  return (
    <section className={`rounded-lg border border-slate-700 bg-slate-900/70 p-3 ${className ?? ""}`}>
      {(title || right) && (
        <div className="mb-2 flex items-center justify-between gap-2">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400">{title}</h2>
          {right}
        </div>
      )}
      {children}
    </section>
  );
}

export function Loading({ what = "Loading" }: { what?: string }) {
  return <p className="animate-pulse text-sm text-slate-500" role="status">{what}...</p>;
}
export function ErrorNote({ error, retry }: { error: string; retry?: () => void }) {
  return (
    <p className="text-sm text-red-400" role="alert">
      {error}
      {retry && (
        <button onClick={retry} className="ml-2 rounded border border-red-500/60 px-1.5 text-xs hover:bg-red-950">retry</button>
      )}
    </p>
  );
}
export function Empty({ children }: { children: React.ReactNode }) {
  return <p className="text-sm text-slate-500">{children}</p>;
}

export const fmt = (v: number | null | undefined, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "n/a" : v.toFixed(d));
export const shortTs = (ts: string) => ts.replace("T", " ").replace(/:\d\dZ$/, "Z");
