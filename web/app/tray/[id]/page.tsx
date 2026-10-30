"use client";

import { useParams } from "next/navigation";
import { useMemo, useState } from "react";
import { AlertRow } from "@/components/AlertFeed";
import FrameScrubber from "@/components/FrameScrubber";
import MetricChart, { type MetricSpec } from "@/components/MetricChart";
import { Empty, ErrorNote, Loading, Panel, SourceBadge, StatusChip, fmt } from "@/components/ui";
import { api } from "@/lib/api";
import { useAsync } from "@/lib/mission";

const METRICS: MetricSpec[] = [
  { metric: "area_cm2", label: "Canopy area (cm², nominal scale)" },
  { metric: "growth_rate", label: "Growth rate" },
  { metric: "exg", label: "Excess green (ExG)" },
  { metric: "chlorosis_frac", label: "Chlorosis fraction" },
  { metric: "soil_vwc", label: "Soil moisture (VWC, simulated)" },
  { metric: "vpd_kpa", label: "VPD (kPa)", threshold: { y: 0.4, label: "vpd_low 0.4 kPa" } },
];
const WINDOWS: [string, number][] = [["48 h", 48], ["7 d", 168], ["14 d", 336], ["All", 900]];

export default function TrayPage() {
  const { id } = useParams<{ id: string }>();
  const [hours, setHours] = useState(336);
  const [markerTs, setMarkerTs] = useState<string | null>(null);
  const tray = useAsync((a) => api.tray(id, a), [id]);
  const alerts = useAsync((a) => api.alerts(a, id), [id]);

  const timeline = useMemo(() => {
    const m = new Map<string, NonNullable<typeof alerts.data>["items"][number]>();
    for (const a of [...(alerts.data?.items ?? []), ...(tray.data?.shared_alerts ?? [])]) m.set(a.id, a);
    return [...m.values()].sort((a, b) => b.ts.localeCompare(a.ts));
  }, [alerts, tray.data]);

  const t = tray.data;
  const L = t?.latest;
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-xl font-semibold">Tray {id}</h1>
        {t && <StatusChip status={t.status} />}
        {t && <SourceBadge source={t.badge.startsWith("SIMULATED FAULT (image") ? "fault_transform" : t.badge.startsWith("SIMULATED FAULT") ? "fault_timewarp" : t.badge.startsWith("SIMULATED") ? "simulated" : "dataset"} />}
        {t && <span className="text-xs text-slate-400">{t.crop.replace(/_/g, " ")} · {t.treatment.replace(/_/g, " ")} · day {t.age_days}</span>}
      </div>
      {tray.error && <ErrorNote error={tray.error} retry={tray.reload} />}
      {!t && tray.loading && <Loading what="Loading tray" />}
      {t && (
        <div className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-4 lg:grid-cols-8" data-testid="latest-values">
          {L ? (
            [
              ["area px", L.area_px?.toLocaleString() ?? "n/a"], ["area cm²", fmt(L.area_cm2, 0)], ["growth", fmt(L.growth_rate, 3)],
              ["ExG", fmt(L.exg)], ["chlorosis", fmt(L.chlorosis_frac)], ["soil VWC", fmt(L.soil_vwc)],
              ["VPD kPa", fmt(L.vpd_kpa)], ["temp °C", fmt(L.temp_c, 1)],
            ].map(([k, v]) => (
              <div key={k} className="rounded border border-slate-700 bg-slate-900 px-2 py-1"><div className="text-slate-500">{k}</div><div className="font-mono text-slate-100">{v}</div></div>
            ))
          ) : (
            <Empty>No measurements yet for this tray.</Empty>
          )}
        </div>
      )}

      <div className="grid gap-4 xl:grid-cols-[minmax(0,420px)_minmax(0,1fr)]">
        <div className="space-y-4">
          <FrameScrubber trayId={id} onTs={setMarkerTs} />
          <Panel title="Alert timeline">
            {alerts.error && <ErrorNote error={alerts.error} retry={alerts.reload} />}
            {!alerts.data && alerts.loading && <Loading what="Loading alerts" />}
            {alerts.data && timeline.length === 0 && <Empty>No alert episodes for this tray.</Empty>}
            <ul className="space-y-1.5" data-testid="alert-timeline">
              {timeline.map((a) => <AlertRow key={a.id} a={a} />)}
            </ul>
          </Panel>
        </div>
        <div className="space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400">
            <span>
              Points coloured by source; diamonds = simulated fault.
              {t?.reference_tray_id && ` Grey line: reference tray ${t.reference_tray_id}.`}
            </span>
            <label className="flex items-center gap-1">
              Window
              <select value={hours} onChange={(e) => setHours(Number(e.target.value))} className="rounded border border-slate-600 bg-slate-900 px-1 py-0.5" aria-label="Chart window">
                {WINDOWS.map(([l, h]) => <option key={h} value={h}>{l}</option>)}
              </select>
            </label>
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            {METRICS.map((m) => (
              <MetricChart key={m.metric} trayId={id} spec={m} hours={hours} refTrayId={t?.reference_tray_id ?? null} markerTs={markerTs} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
