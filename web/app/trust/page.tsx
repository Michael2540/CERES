"use client";

import { ErrorNote, Loading, Panel } from "@/components/ui";
import { api } from "@/lib/api";
import { useAsync } from "@/lib/mission";

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
  const av = d?.area_validation;
  const avEmpty = !av || (av.dice === null && av.iou === null && av.pearson_r === null);
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
        {avEmpty ? (
          <p className="text-sm text-slate-500" data-testid="area-empty">
            No validation numbers yet. The ground-truth comparison (lettuce ground truth, not our trays) lands in M6.
          </p>
        ) : (
          <p className="font-mono text-sm">
            Dice {av!.dice ?? "n/a"} · IoU {av!.iou ?? "n/a"} · Pearson r {av!.pearson_r ?? "n/a"}
          </p>
        )}
      </Panel>

      <Panel title="What is real, what is simulated" className="text-sm">
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="rounded border border-emerald-700/60 bg-emerald-950/20 p-2">
            <h3 className="font-semibold text-emerald-300">Real</h3>
            <ul className="mt-1 list-disc pl-4 text-slate-300">
              <li>Plant images from the Aberystwyth Leaf Evaluation Dataset (ALED), CC BY 4.0.</li>
              <li>SAM 3.1 segmentation of those frames: the canopy measurements are computed, not guessed.</li>
              <li>Muse Spark&apos;s investigations, with every citation checked automatically.</li>
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

function Stat({ label, value, note, testId }: { label: string; value: string; note: string; testId?: string }) {
  return (
    <div className="rounded border border-slate-700 bg-slate-900 p-2" data-testid={testId}>
      <div className="text-[11px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className="font-mono text-lg text-slate-100">{value}</div>
      <div className="text-[11px] text-slate-500">{note}</div>
    </div>
  );
}
