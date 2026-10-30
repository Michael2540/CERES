"use client";

import { ErrorNote, Loading, Panel } from "@/components/ui";
import { api } from "@/lib/api";
import { useAsync } from "@/lib/mission";
import type { AreaAgg, AreaValidation } from "@/lib/types";

const SOURCE_NAMES: Record<string, string> = {
  dataset: "Real frames (ALED, no fault)",
  simulated_fault: "Simulated fault (image transform or time-warp)",
  simulated: "Simulated sensors",
  logged: "Logged runs",
  replay: "Replay",
};

export default function TrustPage() {
  const ev = useAsync((a) => api.evalLatest(a), []);
  const d = ev.data;
  const av = useAsync(() => api.areaValidation(), []);
  return (
    <div className="mx-auto max-w-5xl space-y-4">
      <h1 className="text-xl font-semibold">Trust</h1>
      <p className="text-sm text-slate-400">How we test the agent like flight software, and what is real versus simulated.</p>

      <Panel title="Eval scoreboard">
        {ev.error && <ErrorNote error={ev.error} retry={ev.reload} />}
        {!d && ev.loading && <Loading what="Loading scoreboard" />}
        {!d && !ev.loading && !ev.error && <p className="text-sm text-slate-500">No eval run yet (M7 produces it).</p>}
        {d && (
          <div className="space-y-4" data-testid="scoreboard">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Stat label="Citation errors" value={String(d.citation_errors)} note="deterministic check" testId="stat-citation" />
              <Stat label="Mean latency" value={d.mean_latency_ms === null ? "n/a" : `${(d.mean_latency_ms / 1000).toFixed(1)} s`} note="per assessment" />
              <Stat label="Scenarios" value={String(d.n_scenarios)} note={`run ${d.run_id}`} />
              <Stat label="Generated" value={d.generated_at.slice(0, 10)} note="mission time" />
            </div>
            <div>
              <h3 className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-400">Top-1 cause, correct / scenarios per data source</h3>
              <table className="w-full text-sm" data-testid="top1-table">
                <tbody>
                  {Object.entries(d.top1).map(([k, v]) => (
                    <tr key={k} className="border-t border-slate-800">
                      <td className="py-1 text-slate-300">{SOURCE_NAMES[k] ?? k}</td>
                      <td className="py-1 text-right font-mono text-slate-100">{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {d.small_n && (
                <p className="mt-2 rounded border border-amber-600/50 bg-amber-950/30 p-2 text-xs text-amber-200" data-testid="small-n">
                  Small n: these are counts over a handful of scenarios, not rates. Read them as examples, not accuracy.
                </p>
              )}
            </div>
          </div>
        )}
      </Panel>

      <Panel title="Area validation">
        {av.error && <ErrorNote error={av.error} retry={av.reload} />}
        {!av.data && av.loading && <Loading what="Loading validation" />}
        {!av.data && !av.loading && !av.error && (
          <p className="text-sm text-slate-500" data-testid="area-empty">
            No validation numbers yet. Run scripts/validate_area.py to compare SAM 3.1 with the ALED hand-annotated ground truth.
          </p>
        )}
        {av.data && <AreaValidationView v={av.data} />}
      </Panel>

      <Panel title="What is real, what is simulated" className="text-sm">
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="rounded border border-emerald-700/60 bg-emerald-950/20 p-2">
            <h3 className="font-semibold text-emerald-300">Real</h3>
            <ul className="mt-1 list-disc pl-4 text-slate-300">
              <li>Plant images from the Aberystwyth Leaf Evaluation Dataset (ALED), CC BY 4.0.</li>
              <li>SAM 3.1 segments the frames (B and C partly transformed): the measurement is real; the input fault is simulated.</li>
              <li>Muse Spark&apos;s investigations, with every citation checked automatically.</li>
              <li data-testid="blind-note">The agent is deliberately not told which inputs are simulated, so it is tested blind. Every evidence chip it cites shows its provenance.</li>
            </ul>
          </div>
          <div className="rounded border border-sky-700/60 bg-sky-950/20 p-2">
            <h3 className="font-semibold text-sky-300">Simulated</h3>
            <ul className="mt-1 list-disc pl-4 text-slate-300">
              <li>All sensors (soil moisture, temperature, humidity, CO₂) come from a seeded model.</li>
              <li>Faults: tray B&apos;s drought is an image transform plus simulated soil drawdown; tray C&apos;s low light is a time-warp of real frames; tray L&apos;s humidity is a simulated sensor rise.</li>
              <li>Mission clock and the Earth-Mars light delay.</li>
            </ul>
          </div>
        </div>
        <p className="mt-2 text-xs text-slate-400">Every chart and card carries a badge that says which. Scale (cm² per pixel) is a nominal constant, not measured.</p>
      </Panel>

      <footer className="border-t border-slate-800 pt-3 text-xs text-slate-400">
        <p data-testid="footer-built">Built with Muse Spark · SAM 3.1</p>
        <p>Images: Aberystwyth Leaf Evaluation Dataset (CC BY 4.0)</p>
      </footer>
    </div>
  );
}

const f2 = (v: number | null | undefined) => (v === null || v === undefined ? "n/a" : v.toFixed(2));
const pct = (v: number | null | undefined) => (v === null || v === undefined ? "n/a" : `${Math.round(v * 100)}%`);

function AreaValidationView({ v }: { v: AreaValidation }) {
  const s = v.sam.overall;
  const small = v.sam.by_stage.small;
  const row = (label: string, a: AreaAgg, id: string) => (
    <tr className="border-t border-slate-800" data-testid={id}>
      <td className="py-1 text-slate-300">{label}</td>
      <td className="py-1 text-right font-mono">{f2(a.dice_mean)}</td>
      <td className="py-1 text-right font-mono">{pct(a.area_mape)}</td>
      <td className="py-1 text-right font-mono">{f2(a.pot_pearson_r)}</td>
      <td className="py-1 text-right font-mono">{f2(a.inst_pot_pearson_r)}</td>
      <td className="py-1 text-right font-mono">{pct(a.inst_plant_exact_frac)}</td>
    </tr>
  );
  return (
    <div className="space-y-3" data-testid="area-validation">
      <p className="text-sm text-slate-300">SAM 3.1 against hand annotation on real, untransformed ALED frames.</p>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <Stat label="Images" value={String(v.n_images)} note={`of ${v.n_gt} annotated`} testId="area-n" />
        <Stat label="Dice" value={f2(s.dice_mean)} note={`median ${f2(s.dice_median)}`} testId="area-dice" />
        <Stat label="IoU" value={f2(s.iou_mean)} note="mean" testId="area-iou" />
        <Stat label="Area error" value={pct(s.area_mape)} note={`mean; seedlings ${pct(small.area_mape)}`} testId="area-mape" />
        <Stat label="Per-pot r" value={f2(s.inst_pot_pearson_r)} note={`SAM instances, ${s.pot_n ?? 0} pots`} testId="area-pot-r" />
        <Stat label="Plant count" value={pct(s.inst_plant_exact_frac)} note={`exact; MAE ${f2(s.inst_plant_mae)}`} testId="area-plants" />
      </div>
      <table className="w-full text-sm" data-testid="area-compare">
        <thead>
          <tr className="text-[11px] uppercase tracking-wider text-slate-500">
            <th className="text-left font-normal">Method</th>
            <th className="text-right font-normal">Dice</th>
            <th className="text-right font-normal">Area error</th>
            <th className="text-right font-normal">Pot r (mask)</th>
            <th className="text-right font-normal">Pot r (instances)</th>
            <th className="text-right font-normal">Plants exact</th>
          </tr>
        </thead>
        <tbody>
          {row("SAM 3.1", s, "area-row-sam")}
          {row(`HSV (${v.baseline_hsv.label})`, v.baseline_hsv.overall, "area-row-hsv")}
        </tbody>
      </table>
      <p className="text-xs text-slate-400" data-testid="area-reading">
        Read honestly: on mask quality the colour threshold is as good as SAM or slightly better (Dice, area error,
        and per-pot area when each mask is simply split by pot). SAM traces the smallest seedlings generously, up to
        about twice the annotated area. SAM&apos;s advantage is instances: it keeps touching rosettes apart, so per-pot
        areas and plant counts built from its instances (as CERES does) hold up, while the threshold merges
        neighbours. &quot;Pot r (instances)&quot; and &quot;Plants exact&quot; use each method&apos;s instances; plant counts
        differ only on the larger, touching plants. Per-pot r pools pots of every growth stage.
      </p>
      <p className="text-xs text-slate-400" data-testid="area-trays">
        By tray: {Object.entries(v.sam.by_tray).map(([t, a]) => `${t} n=${a.n}, Dice ${f2(a.dice_mean)}`).join(" · ")}
      </p>
      {v.partial && (
        <p className="rounded border border-amber-600/50 bg-amber-950/30 p-2 text-xs text-amber-200" data-testid="area-partial">
          Partial: {v.n_images} of {v.n_gt} annotated frames are scored so far.
        </p>
      )}
      <p className="text-xs text-slate-500" data-testid="area-caption">{v.caption}</p>
    </div>
  );
}

function Stat({ label, value, note, testId }: { label: string; value: string; note: string; testId?: string }) {
  return (
    <div className="rounded border border-slate-700 bg-slate-900 p-2" data-testid={testId}>
      <div className="text-[11px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className="font-mono text-lg text-slate-100">{value}</div>
      <div className="text-[11px] text-slate-500">{note}</div>
    </div>
  );
}
