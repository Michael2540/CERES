"use client";

import Link from "next/link";
import { useState } from "react";
import { Line, LineChart, ResponsiveContainer, YAxis } from "recharts";
import { api, frameImagePath, imageUrl } from "@/lib/api";
import { useAsync, useMission } from "@/lib/mission";
import type { TrayListItem } from "@/lib/types";
import { Empty, Img, Panel, SourceBadge, StatusChip } from "./ui";

export default function TrayCard({ tray }: { tray: TrayListItem }) {
  const { asOf, openPanel } = useMission();
  const [pinned, setPinned] = useState(false);
  const [hover, setHover] = useState(false);
  const frames = useAsync((a) => api.frames(tray.id, a, 1), [tray.id]);
  const series = useAsync((a) => api.series(tray.id, "area_cm2", 168, a), [tray.id]);
  const frame = frames.data?.items[0];
  const fid = frame?.id ?? tray.latest_frame_id;
  const overlay = pinned || hover;
  const spark = (series.data?.items ?? []).map((p) => ({ t: p.ts, v: p.value }));

  return (
    <Panel className="flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2" data-testid={`tray-card-${tray.id}`}>
        <Link href={`/tray/${tray.id}`} className="text-base font-semibold text-slate-100 hover:text-emerald-300">
          Tray {tray.id}
        </Link>
        <StatusChip status={tray.status} />
      </div>

      <button
        type="button"
        aria-pressed={pinned}
        aria-label={`Toggle mask overlay for tray ${tray.id}`}
        onClick={() => setPinned((p) => !p)}
        onMouseEnter={() => setHover(true)}
        onMouseLeave={() => setHover(false)}
        className="relative block aspect-[4/3] w-full overflow-hidden rounded border border-slate-700 bg-slate-950"
      >
        {fid ? (
          <Img
            src={imageUrl(frameImagePath(tray.id, fid, overlay ? "overlay" : "image"), overlay ? "overlay" : "image", asOf)}
            alt={`Tray ${tray.id} ${overlay ? "mask overlay" : "latest frame"}`}
            className="h-full w-full object-cover"
          />
        ) : (
          <span className="flex h-full items-center justify-center text-sm text-slate-500">no frame yet</span>
        )}
        <span className="absolute bottom-1 left-1 rounded bg-black/60 px-1 text-[10px] text-slate-200">
          {overlay ? "SAM mask" : "click or hover for mask"}
        </span>
      </button>

      <div className="flex flex-wrap items-center gap-1.5">
        {frame ? <SourceBadge source={frame.source} testId={`prov-${tray.id}`} /> : null}
        <span className="rounded border border-sky-400 bg-sky-400/10 px-1.5 py-0.5 text-[10px] font-semibold text-sky-300">
          SENSORS: SIMULATED
        </span>
      </div>

      <div className="h-12" aria-label={`Tray ${tray.id} canopy area sparkline`}>
        {series.loading && !series.data ? (
          <Empty>loading trend...</Empty>
        ) : series.error ? (
          <p className="text-xs text-red-400">trend: {series.error}</p>
        ) : spark.length < 2 ? (
          <Empty>no trend yet</Empty>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={spark}>
              <YAxis hide domain={["auto", "auto"]} />
              <Line type="monotone" dataKey="v" stroke="#34d399" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-300">
        <span data-testid={`open-alerts-${tray.id}`}>
          {tray.open_alerts} open alert{tray.open_alerts === 1 ? "" : "s"}
        </span>
        <span>{tray.shared_alerts.length} shared sensor</span>
      </div>
      <div className="text-xs text-slate-400">Treatment: {tray.treatment.replace(/_/g, " ")}</div>
      {tray.shared_alerts.map((a) => (
        <button
          key={a.id}
          onClick={() => openPanel(a)}
          className="rounded border border-amber-600/60 bg-amber-950/40 px-2 py-1 text-left text-xs text-amber-200 hover:bg-amber-900/40"
        >
          shared sensor: {a.rule} · {a.detail}
        </button>
      ))}
    </Panel>
  );
}
