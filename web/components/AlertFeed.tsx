"use client";

import { api } from "@/lib/api";
import { useAsync, useMission } from "@/lib/mission";
import type { AlertItem } from "@/lib/types";
import { Empty, ErrorNote, Loading, Panel, shortTs } from "./ui";

const SEV: Record<string, string> = {
  alert: "border-red-500 text-red-300",
  critical: "border-red-500 text-red-300",
  warning: "border-amber-500 text-amber-300",
  info: "border-sky-500 text-sky-300",
};

export function AlertRow({ a }: { a: AlertItem }) {
  const { openPanel } = useMission();
  return (
    <li>
      <button
        onClick={() => openPanel(a)}
        data-testid={`alert-${a.id}`}
        className={`w-full rounded border-l-4 bg-slate-900 px-2 py-1.5 text-left text-xs hover:bg-slate-800 ${SEV[a.severity] ?? "border-slate-500"} ${a.status === "resolved" ? "opacity-60" : ""}`}
      >
        <div className="flex items-center justify-between gap-2">
          <span className="font-semibold">
            {a.rule} · tray {a.tray_id}
          </span>
          <span className="text-[10px] uppercase text-slate-400">
            {a.severity} · {a.status}
          </span>
        </div>
        <div className="text-slate-300">{a.detail}</div>
        <div className="text-[10px] text-slate-500">
          {shortTs(a.ts)}
          {a.assessment_id ? " · assessed" : ""}
        </div>
      </button>
    </li>
  );
}

export default function AlertFeed({ trayId, title = "Alert feed" }: { trayId?: string; title?: string }) {
  const r = useAsync((a) => api.alerts(a, trayId), [trayId]);
  const items = r.data?.items ?? [];
  return (
    <Panel title={title}>
      {r.error && <ErrorNote error={r.error} retry={r.reload} />}
      {!r.data && r.loading && <Loading what="Loading alerts" />}
      {r.data && items.length === 0 && <Empty>No alerts at this mission time.</Empty>}
      <ul className="space-y-1.5" data-testid="alert-list">
        {items.map((a) => (
          <AlertRow key={a.id} a={a} />
        ))}
      </ul>
    </Panel>
  );
}
