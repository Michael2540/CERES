"use client";

import { usePathname, useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, api, streamAssessment } from "@/lib/api";
import { useMission } from "@/lib/mission";
import type { AlertItem, AssessmentResponse, Evidence, StreamEvent } from "@/lib/types";
import { Empty, ErrorNote, fmt, shortTs, sourceLabel } from "./ui";

const COST_ESTIMATE = "~$0.10–0.20 (2 measured runs)";

export default function AgentPanel() {
  const { panelAlert, openPanel, asOf, hourKey } = useMission();
  const isOpen = panelAlert !== null;
  const heading = useRef<HTMLHeadingElement>(null);
  const opener = useRef<HTMLElement | null>(null);

  // Focus moves into the panel on open and returns to the opener on close.
  useEffect(() => {
    if (isOpen) {
      opener.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
      heading.current?.focus();
    } else if (opener.current) {
      opener.current.focus?.();
      opener.current = null;
    }
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") openPanel(null);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [isOpen, openPanel]);

  // The panel never shows an alert from the mission's future.
  useEffect(() => {
    if (panelAlert && asOf && panelAlert.ts > asOf) openPanel(null);
  }, [hourKey, asOf, panelAlert, openPanel]);

  if (!panelAlert) return null;
  return (
    <aside
      role="dialog"
      aria-label="Agent panel"
      data-testid="agent-panel"
      className="fixed inset-y-0 right-0 z-50 flex w-full flex-col border-l border-slate-600 bg-slate-950 shadow-2xl sm:w-[480px]"
    >
      <div className="flex items-center justify-between border-b border-slate-700 px-3 py-2">
        <h2 ref={heading} tabIndex={-1} className="text-sm font-semibold tracking-wide text-slate-100 outline-none">
          Flight Agronomist
        </h2>
        <button onClick={() => openPanel(null)} className="rounded border border-slate-600 px-2 py-0.5 text-xs hover:bg-slate-800" aria-label="Close agent panel">
          Close
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-3">
        <PanelBody key={panelAlert.id} alert={panelAlert} />
      </div>
    </aside>
  );
}

type Row = { key: number; kind: "call" | "result"; round: number; tool: string; args?: Record<string, unknown>; n?: number };
type Phase = "idle" | "confirm" | "starting" | "streaming" | "loading" | "running" | "done" | "unavailable" | "error";

function PanelBody({ alert }: { alert: AlertItem }) {
  const { asOf, hourKey, refreshSpend, setFocus } = useMission();
  const router = useRouter();
  const path = usePathname();
  const [phase, setPhase] = useState<Phase>(alert.assessment_id ? "loading" : "idle");
  const [forceRun, setForceRun] = useState(false);
  const [rows, setRows] = useState<Row[]>([]);
  const [result, setResult] = useState<AssessmentResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [evDetail, setEvDetail] = useState<{ ev?: Evidence; error?: string; id: string } | null>(null);
  const cancel = useRef<(() => void) | null>(null);
  const busy = useRef(false);
  const keyN = useRef(0);
  const asOfRef = useRef(asOf);
  const phaseRef = useRef(phase);
  useEffect(() => {
    asOfRef.current = asOf;
    phaseRef.current = phase;
  });

  /** Read the run; a 404 (later than as_of) clears everything shown. Returns null when nothing can be shown. */
  const read = useCallback(async (id: string): Promise<AssessmentResponse | null> => {
    try {
      const r = await api.assessment(id, asOfRef.current);
      setResult(r);
      return r;
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) {
        setResult(null);
        setEvDetail(null);
        setRows([]);
        setPhase("unavailable");
      } else {
        setError(e instanceof Error ? e.message : String(e));
        setPhase("error");
      }
      return null;
    }
  }, []);

  const attach = useCallback(
    (id: string) => {
      cancel.current?.();
      setRows([]);
      setError(null);
      setPhase("streaming");
      cancel.current = streamAssessment(
        id,
        (e: StreamEvent) => {
          if (e.event === "tool_call") {
            setRows((r) => [...r, { key: keyN.current++, kind: "call", round: e.data.round, tool: e.data.tool, args: e.data.args }]);
          } else if (e.event === "tool_result") {
            setRows((r) => [...r, { key: keyN.current++, kind: "result", round: e.data.round, tool: e.data.tool, n: e.data.ids.length }]);
          }
        },
        async () => {
          // Whatever ended the stream (assessment, error, stalled, dropped), re-read the run once.
          const r = await read(id);
          if (!r) return;
          setPhase(r.valid === null ? "running" : "done");
          refreshSpend();
        },
      );
    },
    [read, refreshSpend],
  );

  useEffect(() => {
    if (alert.assessment_id) {
      const id = alert.assessment_id;
      void read(id).then((r) => {
        if (!r) return;
        if (r.valid === null) attach(id);
        else setPhase("done");
      });
    }
    return () => cancel.current?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [alert.id]);

  // Time-gate: re-read when the mission hour changes (unless a stream is attached).
  const firstHour = useRef(hourKey);
  useEffect(() => {
    if (firstHour.current === hourKey) return;
    firstHour.current = hourKey;
    const p = phaseRef.current;
    if (p === "streaming" || p === "starting" || p === "loading") return;
    const id = result?.assessment_id ?? alert.assessment_id;
    if (!id) return;
    void read(id).then((r) => r && setPhase(r.valid === null ? "running" : "done"));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hourKey]);

  const run = async (force: boolean) => {
    if (busy.current) return; // double-click guard: one paid run per confirm
    busy.current = true;
    setPhase("starting");
    setError(null);
    setResult(null);
    setRows([]);
    try {
      const { assessment_id } = await api.assess(alert.id, force);
      attach(assessment_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setPhase("error");
    } finally {
      busy.current = false;
    }
  };

  const decide = async (actionId: string, decision: "approve" | "reject") => {
    try {
      const r = await api.decide(actionId, decision);
      setResult((cur) => cur && { ...cur, actions: cur.actions.map((a) => (a.action_id === actionId ? { ...a, status: r.status } : a)) });
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        setError("That action was already decided or superseded; refreshed.");
        if (result) void read(result.assessment_id);
      } else {
        setError(e instanceof Error ? e.message : String(e));
      }
    }
  };

  const chip = async (id: string) => {
    try {
      const ev = await api.evidence(id, asOfRef.current);
      setEvDetail({ ev, id });
      if (ev.kind === "frame" || ev.kind === "metric" || ev.kind === "reading") {
        setFocus({ trayId: ev.tray_id, frameId: ev.frame_id, ts: ev.ts ?? undefined, evidenceId: id, metric: ev.metric });
        // Shared-air evidence is a time jump on the tray being viewed (or the alert's tray).
        const target = ev.tray_id === "air" ? (path.startsWith("/tray/") ? path : `/tray/${result?.tray_id ?? alert.tray_id}`) : `/tray/${ev.tray_id}`;
        if (path !== target) router.push(target);
      }
    } catch (e) {
      setEvDetail({ id, error: e instanceof ApiError && e.status === 404 ? "not available at this mission time" : String(e) });
    }
  };

  const a = result?.assessment ?? null;
  // Provenance of every cited id (the /evidence `source`, or its kind), shown as a small tag on each chip.
  const [prov, setProv] = useState<Record<string, string>>({});
  const provRef = useRef(prov);
  provRef.current = prov;
  const citedIds = useMemo(
    () =>
      Array.from(
        new Set([...(a?.findings ?? []), ...(a?.probable_causes ?? []), ...(a?.recommended_actions ?? [])].flatMap((x) => x.evidence_ids)),
      ),
    [a],
  );
  useEffect(() => {
    let alive = true;
    for (const id of citedIds) {
      if (id in provRef.current) continue;
      api
        .evidence(id, asOfRef.current)
        .then((ev) => alive && setProv((p) => ({ ...p, [id]: ev.source ?? ev.kind })))
        .catch(() => undefined);
    }
    return () => {
      alive = false;
    };
  }, [citedIds]);
  const terminal = result !== null && result.valid !== null;
  const failed = result !== null && result.valid !== null && (result.valid === false || a === null);
  const unverified = result?.valid === false;
  const showStart = !result && (phase === "idle" || phase === "confirm" || phase === "starting");
  const showForce = failed && phase === "done";
  const showConfirm = phase === "confirm" || phase === "starting";

  return (
    <div className="space-y-3 text-sm">
      <div className="rounded border border-slate-700 bg-slate-900 p-2">
        <div className="font-semibold text-slate-100">
          {alert.rule} · tray {alert.tray_id}
        </div>
        <div className="text-xs text-slate-400">
          {alert.detail} · {shortTs(alert.ts)} · {alert.severity} · {alert.status}
        </div>
      </div>

      {(showStart || showForce || showConfirm) && (
        <div>
          {showConfirm ? (
            <div className="rounded border border-amber-500/70 bg-amber-950/40 p-2" data-testid="confirm-box">
              <p className="text-amber-200">
                {forceRun ? "Force re-run: " : ""}This calls Meta&apos;s Muse Spark live. Estimated cost {COST_ESTIMATE}, about 30-60 s.
              </p>
              <div className="mt-2 flex gap-2">
                <button
                  onClick={() => void run(forceRun)}
                  disabled={phase === "starting"}
                  className="rounded bg-emerald-600 px-3 py-1 font-semibold text-white hover:bg-emerald-500 disabled:opacity-50"
                  data-testid="confirm-run"
                >
                  {phase === "starting" ? "Starting..." : "Confirm and run"}
                </button>
                <button
                  onClick={() => setPhase(result ? "done" : "idle")}
                  disabled={phase === "starting"}
                  className="rounded border border-slate-500 px-3 py-1 hover:bg-slate-800 disabled:opacity-50"
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : showForce ? (
            <button
              onClick={() => {
                setForceRun(true);
                setPhase("confirm");
              }}
              className="w-full rounded border border-amber-500 bg-amber-950 px-3 py-2 font-semibold text-amber-200 hover:bg-amber-900"
              data-testid="force-rerun"
            >
              Re-run with Muse Spark (force, {COST_ESTIMATE})
            </button>
          ) : (
            <>
              <button
                onClick={() => {
                  setForceRun(false);
                  setPhase("confirm");
                }}
                className="w-full rounded border border-emerald-500 bg-emerald-950 px-3 py-2 font-semibold text-emerald-200 hover:bg-emerald-900"
                data-testid="investigate"
              >
                Investigate with Muse Spark ({COST_ESTIMATE})
              </button>
              <p className="mt-1 text-xs text-slate-500">Nothing is sent until you confirm.</p>
            </>
          )}
        </div>
      )}

      {error && <ErrorNote error={error} />}

      {(phase === "streaming" || phase === "running") && (
        <p className="text-xs text-emerald-300" data-testid="running-note">
          Still running...
          {phase === "running" && (
            <>
              {" "}
              <span className="text-amber-300">stream lost; re-attach</span>{" "}
              <button
                onClick={() => result && attach(result.assessment_id)}
                className="rounded border border-emerald-500 px-2 py-0.5 text-emerald-200 hover:bg-emerald-950"
                data-testid="reattach"
              >
                Re-attach (free)
              </button>
            </>
          )}
        </p>
      )}

      {(phase === "streaming" || rows.length > 0) && (
        <div data-testid="tool-stream">
          <h3 className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-400">
            Tool calls {phase === "streaming" && <span className="animate-pulse text-emerald-300">streaming...</span>}
          </h3>
          <ol className="space-y-1">
            {rows.map((r) => (
              <li key={r.key} className="rounded border border-slate-700 bg-slate-900 px-2 py-1 font-mono text-[11px]" data-testid={`tool-${r.kind}`}>
                <span className="text-slate-500">r{r.round} </span>
                {r.kind === "call" ? (
                  <>
                    <span className="text-sky-300">{r.tool}</span>
                    <span className="break-all text-slate-400"> {JSON.stringify(r.args)}</span>
                  </>
                ) : (
                  <span className="text-emerald-300">
                    ← {r.tool}: {r.n} id{r.n === 1 ? "" : "s"}
                  </span>
                )}
              </li>
            ))}
          </ol>
        </div>
      )}
      {phase === "loading" && <p className="animate-pulse text-slate-500" role="status">Loading assessment...</p>}
      {phase === "unavailable" && (
        <p className="rounded border border-slate-700 bg-slate-900 p-2 text-slate-400" data-testid="unavailable">
          Not yet available at this mission time.
        </p>
      )}

      {terminal && result && (
        <div className={unverified ? "opacity-60 grayscale" : ""} data-testid="assessment">
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <span
              data-testid="live-badge"
              className={`rounded px-1.5 py-0.5 text-[11px] font-bold ${result.recorded ? "bg-violet-900 text-violet-200" : "bg-red-900 text-red-200"}`}
            >
              {result.recorded ? "RECORDED" : "LIVE"}
            </span>
            <span className="text-xs text-slate-300" data-testid="run-meta">
              {result.latency_ms !== null ? `${(result.latency_ms / 1000).toFixed(1)} s` : "latency n/a"} ·{" "}
              {typeof result.cost_usd === "number" ? `$${result.cost_usd.toFixed(3)}` : "cost n/a"} · {result.model}
            </span>
            {a && <span className="rounded border border-slate-500 px-1.5 text-[11px] uppercase" data-testid="assessment-status">{a.status}</span>}
            {unverified && <span className="rounded bg-slate-700 px-1.5 text-[11px] font-semibold text-slate-200">UNVERIFIED</span>}
          </div>
          {unverified && (
            <div className="mb-2 rounded border border-slate-600 bg-slate-900 p-2 text-xs text-slate-300">
              Citation check failed; treat this as unverified.
              <ul className="mt-1 list-disc pl-4 text-slate-400">
                {result.citation_errors.map((c) => <li key={c}>{c}</li>)}
              </ul>
            </div>
          )}
          {!a ? (
            <Empty>No assessment body (the run failed).</Empty>
          ) : (
            <div className="space-y-3">
              <section>
                <h3 className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-400">Crew brief</h3>
                <p className="rounded bg-slate-900 p-2 text-slate-100" data-testid="crew-brief">{a.crew_brief}</p>
              </section>
              <section>
                <h3 className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-400">Ranked causes</h3>
                <ul className="space-y-2">
                  {[...a.probable_causes].sort((x, y) => y.likelihood - x.likelihood).map((c) => (
                    <li key={c.cause} data-testid="cause">
                      <div className="flex justify-between text-xs"><span className="font-semibold">{c.cause.replace(/_/g, " ")}</span><span className="font-mono">{fmt(c.likelihood)}</span></div>
                      <div className="h-2 rounded bg-slate-800"><div className="h-2 rounded bg-emerald-500" style={{ width: `${Math.round(c.likelihood * 100)}%` }} /></div>
                      {c.discriminating_check && <p className="mt-0.5 text-[11px] text-slate-400">Check: {c.discriminating_check}</p>}
                      <Chips ids={c.evidence_ids} onClick={chip} prov={prov} />
                    </li>
                  ))}
                </ul>
              </section>
              <section>
                <h3 className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-400">Findings</h3>
                <ul className="space-y-2">
                  {a.findings.map((f, i) => (
                    <li key={i} className="text-slate-200">
                      {f.claim} <span className="text-[11px] text-slate-500">(confidence {fmt(f.confidence)})</span>
                      <Chips ids={f.evidence_ids} onClick={chip} prov={prov} />
                    </li>
                  ))}
                </ul>
                {evDetail && (
                  <p className="mt-2 rounded border border-yellow-600/50 bg-slate-900 p-1.5 font-mono text-[11px] text-yellow-200" data-testid="evidence-detail">
                    {evDetail.error ? `${evDetail.id}: ${evDetail.error}` : `${evDetail.id} · ${evDetail.ev?.kind}${evDetail.ev?.value !== undefined ? ` = ${evDetail.ev.value}` : ""}${evDetail.ev?.source ? ` · ${evDetail.ev.source}` : ""}`}
                  </p>
                )}
              </section>
              {a.setpoint_caveats.length > 0 && (
                <section>
                  <h3 className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-400">Setpoint caveats</h3>
                  <ul className="list-disc pl-4 text-xs text-amber-200" data-testid="caveats">{a.setpoint_caveats.map((c) => <li key={c}>{c}</li>)}</ul>
                </section>
              )}
              <section>
                <h3 className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-400">Recommended actions</h3>
                {a.recommended_actions.length === 0 && <Empty>None.</Empty>}
                <ul className="space-y-1">
                  {a.recommended_actions.map((r, i) => (
                    <li key={i} className="text-slate-200">
                      <b>{r.action}</b>{r.magnitude ? ` (${r.magnitude})` : ""}{r.rationale ? `: ${r.rationale}` : ""}
                      <Chips ids={r.evidence_ids} onClick={chip} prov={prov} />
                    </li>
                  ))}
                </ul>
                {result.actions.map((ac) => (
                  <div key={ac.action_id} className="mt-2 flex flex-wrap items-center gap-2 rounded border border-slate-700 bg-slate-900 p-2" data-testid="action-row">
                    <span className="flex-1 text-xs text-slate-200">Proposal: {ac.proposal.action} {ac.proposal.magnitude ?? ""}</span>
                    <span className="text-[11px] uppercase text-slate-400" data-testid="action-status">{ac.status}</span>
                    <button disabled={ac.status !== "pending"} onClick={() => void decide(ac.action_id, "approve")} className="rounded bg-emerald-700 px-2 py-0.5 text-xs font-semibold disabled:opacity-40">Approve</button>
                    <button disabled={ac.status !== "pending"} onClick={() => void decide(ac.action_id, "reject")} className="rounded bg-slate-700 px-2 py-0.5 text-xs font-semibold disabled:opacity-40">Reject</button>
                  </div>
                ))}
                <p className="mt-1 text-[11px] text-slate-500">The agent proposes; the crew decides.</p>
              </section>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function Chips({ ids, onClick, prov }: { ids: string[]; onClick: (id: string) => void; prov: Record<string, string> }) {
  if (!ids.length) return null;
  return (
    <div className="mt-1 flex flex-wrap gap-1">
      {ids.map((id) => (
        <button
          key={id}
          onClick={() => onClick(id)}
          title={id}
          data-testid="evidence-chip"
          className="max-w-full truncate rounded-full border border-yellow-700/60 bg-yellow-950/40 px-2 py-0.5 font-mono text-[10px] text-yellow-200 hover:bg-yellow-900/50"
        >
          {id}
          {prov[id] && (
            <span data-testid="evidence-prov" className="ml-1 rounded bg-slate-800 px-1 text-[9px] font-sans font-semibold text-slate-300">
              {sourceLabel(prov[id])}
            </span>
          )}
        </button>
      ))}
    </div>
  );
}
