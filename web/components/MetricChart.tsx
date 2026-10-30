"use client";

import { useMemo } from "react";
import { CartesianGrid, ComposedChart, Line, ReferenceDot, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "@/lib/api";
import { downsample } from "@/lib/downsample";
import { useAsync, useMission } from "@/lib/mission";
import { Empty, ErrorNote, Loading, Panel, SOURCE_COLOR, isFault, sourceLabel } from "./ui";

type Row = { t: number; v?: number | null; ref?: number | null; src?: string; id?: string };

export type MetricSpec = { metric: string; label: string; threshold?: { y: number; label: string } };

const MAX_POINTS = 400;
const DOT_LIMIT = 300;
const SIMULATED_METRICS = new Set(["soil_vwc", "vpd_kpa", "temp_c", "rh_pct", "co2_ppm"]);

type DotProps = { cx?: number; cy?: number; payload?: Row; dense?: boolean };
function Dot({ cx, cy, payload, dense }: DotProps) {
  if (cx === undefined || cy === undefined || !payload || payload.v === undefined || payload.v === null) return <g />;
  if (dense && !(payload.src && isFault(payload.src))) return <g />;
  const c = SOURCE_COLOR[payload.src ?? ""] ?? "#94a3b8";
  if (payload.src && isFault(payload.src)) {
    return <path d={`M${cx},${cy - 6} L${cx + 6},${cy} L${cx},${cy + 6} L${cx - 6},${cy} Z`} fill={c} stroke="#fff" strokeWidth={1} />;
  }
  return <circle cx={cx} cy={cy} r={3.5} fill={c} stroke="#0b1220" strokeWidth={1} />;
}

export default function MetricChart({
  trayId, spec, hours, refTrayId, markerTs,
}: { trayId: string; spec: MetricSpec; hours: number; refTrayId: string | null; markerTs: string | null }) {
  const { focus } = useMission();
  const own = useAsync((a) => api.series(trayId, spec.metric, hours, a), [trayId, spec.metric, hours]);
  // The reference line is skipped for shared-air series (the same air sensor serves every tray).
  const shared = own.data?.sensor_tray;
  const ownLoaded = own.data !== null;
  const ref = useAsync(
    async (a) => (refTrayId && ownLoaded && !shared ? api.series(refTrayId, spec.metric, hours, a).catch(() => null) : null),
    [refTrayId, spec.metric, hours, ownLoaded, shared],
  );

  const focusId = focus?.evidenceId;
  const rows = useMemo<Row[]>(() => {
    const m = new Map<number, Row>();
    const ownPts = downsample(own.data?.items ?? [], (p) => p.value, MAX_POINTS, (p) => isFault(p.source) || p.id === focusId);
    for (const p of ownPts) {
      const t = Date.parse(p.ts);
      m.set(t, { ...(m.get(t) ?? { t }), v: p.value, src: p.source, id: p.id });
    }
    for (const p of downsample(ref.data?.items ?? [], (x) => x.value, MAX_POINTS, () => false)) {
      const t = Date.parse(p.ts);
      m.set(t, { ...(m.get(t) ?? { t }), ref: p.value });
    }
    return [...m.values()].sort((a, b) => a.t - b.t);
  }, [own.data, ref.data, focusId]);
  const dense = rows.filter((r) => r.v !== undefined).length > DOT_LIMIT;

  const focusPoint = focus ? rows.find((r) => r.id === focus.evidenceId) : undefined;
  const outside = Boolean(focus && !focusPoint && focus.metric === spec.metric && (own.data?.items.length ?? 0) > 0);
  const hasData = (own.data?.items.length ?? 0) > 0;
  const hasRef = (ref.data?.items.length ?? 0) > 0;
  const fmtT = (t: number) => new Date(t).toISOString().slice(5, 13).replace("T", " ") + "h";

  return (
    <Panel
      title={spec.label}
      right={
        <span className="flex items-center gap-2 text-[10px] text-slate-400">
          {SIMULATED_METRICS.has(spec.metric) && (
            <span className="rounded border border-sky-400 px-1 font-semibold text-sky-300" data-testid={`sim-badge-${spec.metric}`}>SIMULATED</span>
          )}
          {own.data?.sensor_tray === "air" && (
            <span className="rounded border border-sky-500 px-1 text-sky-300" data-testid={`air-note-${spec.metric}`}>sensor from shared air</span>
          )}
          {hasRef && <span className="flex items-center gap-1"><i className="inline-block h-0.5 w-4 bg-slate-500" />ref tray {refTrayId}</span>}
        </span>
      }
    >
      <div className="h-44" data-testid={`chart-${spec.metric}`}>
        {own.error && !own.data ? (
          <ErrorNote error={own.error} retry={own.reload} />
        ) : !own.data ? (
          <Loading what="Loading series" />
        ) : !hasData ? (
          <Empty>No {spec.metric} data in this window.</Empty>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={rows} margin={{ top: 6, right: 12, bottom: 0, left: -8 }}>
              <CartesianGrid stroke="#1e293b" />
              <XAxis dataKey="t" type="number" scale="time" domain={["dataMin", "dataMax"]} tickFormatter={fmtT} stroke="#64748b" fontSize={10} minTickGap={40} />
              <YAxis stroke="#64748b" fontSize={10} domain={spec.threshold ? [(min: number) => Math.min(min, spec.threshold!.y * 0.9), "auto"] : ["auto", "auto"]} width={46} />
              <Tooltip
                contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }}
                labelFormatter={(t) => new Date(Number(t)).toISOString().replace(".000", "")}
                formatter={(val, name, item) => {
                  const src = (item.payload as Row | undefined)?.src;
                  return name === "v" ? [`${val}${src ? ` · ${sourceLabel(src)}` : ""}`, trayId] : [String(val), `ref ${refTrayId}`];
                }}
              />
              {hasRef && <Line dataKey="ref" stroke="#64748b" strokeWidth={1.5} dot={false} connectNulls isAnimationActive={false} />}
              <Line dataKey="v" stroke="#94e2c0" strokeWidth={1.5} dot={<Dot dense={dense} />} activeDot={{ r: 5 }} connectNulls isAnimationActive={false} />
              {spec.threshold && (
                <ReferenceLine y={spec.threshold.y} stroke="#f87171" strokeDasharray="4 3" label={{ value: spec.threshold.label, fill: "#f87171", fontSize: 10, position: "insideTopRight" }} />
              )}
              {markerTs && <ReferenceLine x={Date.parse(markerTs)} stroke="#38bdf8" strokeOpacity={0.6} />}
              {focusPoint && focusPoint.v !== undefined && focusPoint.v !== null && (
                <ReferenceDot x={focusPoint.t} y={focusPoint.v} r={9} fill="none" stroke="#facc15" strokeWidth={2} data-testid="focus-dot" />
              )}
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </div>
      {own.data && own.data.unit && <p className="mt-1 text-[10px] text-slate-500">unit: {own.data.unit}</p>}
      {outside && <p className="text-[10px] text-yellow-300" data-testid={`outside-${spec.metric}`}>cited point is outside this window</p>}
    </Panel>
  );
}
