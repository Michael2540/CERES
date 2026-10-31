"use client";

import { useState } from "react";
import Link from "next/link";
import { ApiError, api } from "@/lib/api";
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

/** Mars mode: while the link is not up, an open alert can be queued for on-board triage instead of going to Earth. */
function QueueActions({ items }: { items: AlertItem[] }) {
  const { link, asOf, refreshLink } = useMission();
  const [res, setRes] = useState<Record<string, { busy?: boolean; ok?: boolean; msg?: string }>>({});
  const open = items.filter((a) => a.status === "open");
  if (!link || link.state === "up" || open.length === 0) return null;
  const queue = async (id: string) => {
    if (res[id]?.busy) return;
    setRes((r) => ({ ...r, [id]: { busy: true } }));
    try {
      const it = await api.enqueue(id, asOf);
      setRes((r) => ({ ...r, [id]: { ok: true, msg: it.error ? `Queued, but: ${it.error}` : `Queued (${it.status}).` } }));
      refreshLink();
    } catch (e) {
      const msg = e instanceof ApiError && e.status === 409 ? `Not queued: ${e.message}` : e instanceof Error ? e.message : String(e);
      setRes((r) => ({ ...r, [id]: { msg } }));
    }
  };
  return (
    <div className="mt-3 border-t border-slate-700 pt-2" data-testid="queue-actions">
      <h3 className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-violet-300">Link not up: on-board triage</h3>
      <ul className="space-y-1.5">
        {open.map((a) => (
          <li key={a.id} className="text-[11px]">
            <button
              onClick={() => void queue(a.id)}
              disabled={res[a.id]?.busy || res[a.id]?.ok}
              data-testid={`queue-alert-${a.id}`}
              className="rounded border border-violet-500 px-2 py-0.5 text-violet-200 hover:bg-violet-950 disabled:opacity-50"
            >
              {res[a.id]?.busy ? "Queueing..." : `Queue ${a.rule} · tray ${a.tray_id} for on-board triage`}
            </button>
            {res[a.id]?.msg && (
              <span className={`ml-2 ${res[a.id].ok ? "text-emerald-300" : "text-red-300"}`} role={res[a.id].ok ? "status" : "alert"} data-testid="queue-result">
                {res[a.id].msg} {res[a.id].ok && <Link href="/mars" className="underline">Mars mode</Link>}
              </span>
            )}
          </li>
        ))}
      </ul>
    </div>
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
      <QueueActions items={items} />
    </Panel>
  );
}
