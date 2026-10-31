"use client";

import { useEffect, useRef, useState } from "react";
import { ApiError, api } from "@/lib/api";
import { useAsync, useMission } from "@/lib/mission";
import type { Audit, DrainPlan, Evidence, QueueItem, QueueStatus } from "@/lib/types";
import { Chips } from "./AgentPanel";
import { Empty, ErrorNote, Loading, Panel, shortTs } from "./ui";

const STATUS_STYLE: Record<QueueStatus, string> = {
  queued: "border-slate-400 text-slate-200",
  edge_done: "border-violet-500 text-violet-200",
  edge_invalid: "border-red-500 text-red-300",
  sent: "border-sky-500 text-sky-300",
  audited: "border-emerald-500 text-emerald-300",
  failed: "border-red-500 text-red-300",
};
const URGENCY_STYLE: Record<string, string> = {
  now: "border-red-500 text-red-300",
  next_check: "border-amber-500 text-amber-300",
  log_only: "border-slate-500 text-slate-300",
};
const URGENCY_TEXT: Record<string, string> = { now: "now", next_check: "next check", log_only: "log only" };
const cause = (c: string | null) => (c ? c.replace(/_/g, " ") : "n/a");

function H3({ children }: { children: React.ReactNode }) {
  return <h3 className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-400">{children}</h3>;
}

/** The queue, in drain order, with a card per item. Refetches on every link poll so statuses move on their own. */
export function QueueView() {
  const { queueRev, link } = useMission();
  const r = useAsync((a) => api.linkQueue(a), [queueRev]);
  const items = r.data?.items ?? [];
  return (
    <Panel title={`Queue${r.data ? ` (${items.length})` : ""}`}>
      {r.error && <ErrorNote error={r.error} retry={r.reload} />}
      {!r.data && r.loading && <Loading what="Loading queue" />}
      {r.data && items.length === 0 && (
        <Empty>
          <span data-testid="queue-empty">
            Nothing queued at this mission time.
            {link?.state === "up"
              ? " The link is up, so alerts go straight to Earth (Investigate on an alert)."
              : " Queue an open alert for on-board triage from the alert list."}
          </span>
        </Empty>
      )}
      <ol className="space-y-3" data-testid="queue-list">
        {items.map((it) => (
          <QueueCard key={it.id} item={it} />
        ))}
      </ol>
    </Panel>
  );
}

function QueueCard({ item: it }: { item: QueueItem }) {
  const { asOf } = useMission();
  const e = it.edge_json;
  const unverified = it.edge_valid === false;
  const waiting = it.status === "queued" || it.status === "edge_done" || it.status === "edge_invalid";
  return (
    <li className="rounded border border-slate-700 bg-slate-900 p-2" data-testid={`queue-item-${it.id}`}>
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm font-semibold text-slate-100">{it.alert_id.replace(/^alert_/, "")}</span>
        <span
          className={`rounded-full border px-2 py-0.5 text-[11px] font-semibold uppercase ${STATUS_STYLE[it.status] ?? "border-slate-500 text-slate-300"}`}
          data-testid="queue-status"
        >
          {it.status}
        </span>
        <span className="text-[11px] text-slate-500">
          tray {it.tray_id} · queued as of {shortTs(it.as_of)}
        </span>
        <span className="ml-auto text-[11px] text-slate-400" data-testid="queue-deliver">
          {waiting
            ? it.deliverable
              ? "deliverable now"
              : it.deliver_in_s !== null
                ? `deliverable in ${Math.ceil(it.deliver_in_s)} s`
                : "waiting for the link"
            : ""}
        </span>
      </div>
      {it.error && (
        <p className="mt-1 rounded border border-red-600/50 bg-red-950/30 p-1.5 text-xs text-red-200" role="alert" data-testid="queue-error">
          {it.error}
        </p>
      )}
      {e && <EdgeCard item={it} asOf={asOf} unverified={unverified} />}
      {it.audit_json && <AuditCard audit={it.audit_json} earthId={it.earth_assessment_id} />}
      {!e && !it.error && <p className="mt-1 text-xs text-slate-500">No edge triage for this item.</p>}
      <p className="mt-1 text-[10px] text-slate-500" data-testid="perception-note">
        Perception is not re-run: {it.perception}.
      </p>
    </li>
  );
}

function EdgeCard({ item: it, asOf, unverified }: { item: QueueItem; asOf: string | undefined; unverified: boolean }) {
  const e = it.edge_json!;
  const p = it.provenance;
  const [prov, setProv] = useState<Record<string, string>>({});
  const [detail, setDetail] = useState<string | null>(null);
  const provRef = useRef(prov);
  provRef.current = prov;
  const ids = e.evidence_ids;
  useEffect(() => {
    let alive = true;
    for (const id of ids) {
      if (id in provRef.current) continue;
      api
        .evidence(id, asOf)
        .then((ev) => alive && setProv((x) => ({ ...x, [id]: ev.source ?? ev.kind })))
        .catch(() => undefined);
    }
    return () => {
      alive = false;
    };
  }, [ids, asOf]);
  const chip = async (id: string) => {
    try {
      const ev: Evidence = await api.evidence(id, asOf);
      setDetail(`${id} · ${ev.kind}${ev.value !== undefined ? ` = ${ev.value}` : ""}${ev.source ? ` · ${ev.source}` : ""}`);
    } catch (err) {
      setDetail(`${id}: ${err instanceof ApiError && err.status === 404 ? "not available at this mission time" : String(err)}`);
    }
  };
  const cites = it.citation_errors.filter((c) => !c.startsWith("grounding:"));
  return (
    <div className={`mt-2 space-y-2 rounded border border-violet-700/50 bg-slate-950 p-2 ${unverified ? "opacity-60 grayscale" : ""}`} data-testid="edge-card">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-violet-300">On-board triage</span>
        {p && (
          <>
            <span
              className={`rounded px-1.5 py-0.5 text-[11px] font-bold ${p.recorded ? "bg-violet-900 text-violet-200" : "bg-amber-900 text-amber-100"}`}
              data-testid="edge-recorded"
            >
              {p.recorded ? `RECORDED · EDGE (${p.model})` : `EDGE (${p.model}, remote API: stands in for an on-board model)`}
            </span>
            <span className="text-xs text-slate-300" data-testid="edge-model">
              {p.model} ({p.provider})
            </span>
            {p.recorded && p.recorded_as_of && <span className="text-[11px] text-slate-500">recorded as of {shortTs(p.recorded_as_of)}</span>}
          </>
        )}
        {unverified && (
          <span className="rounded bg-slate-700 px-1.5 text-[11px] font-semibold text-slate-200" data-testid="edge-unverified">
            UNVERIFIED
          </span>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span
          className={`rounded-full border px-2 py-0.5 text-[11px] font-semibold uppercase ${URGENCY_STYLE[e.urgency] ?? "border-slate-500"}`}
          data-testid="edge-urgency"
        >
          {URGENCY_TEXT[e.urgency] ?? e.urgency}
        </span>
        <span className="text-slate-400">likely cause</span>
        <b className="text-slate-100" data-testid="edge-cause">{cause(e.likely_cause)}</b>
      </div>
      <div>
        <H3>Crew instruction</H3>
        <p className="rounded bg-slate-900 p-2 text-slate-100" data-testid="edge-instruction">{e.crew_instruction}</p>
      </div>
      <div>
        <H3>Handoff note for Earth</H3>
        <p className="rounded bg-slate-900 p-2 text-slate-200" data-testid="edge-handoff">{e.handoff_note_for_earth}</p>
      </div>
      <div>
        <H3>Evidence</H3>
        <Chips ids={ids} onClick={(id) => void chip(id)} prov={prov} />
        {detail && (
          <p className="mt-1 rounded border border-yellow-600/50 bg-slate-900 p-1.5 font-mono text-[11px] text-yellow-200" data-testid="edge-evidence-detail">
            {detail}
          </p>
        )}
      </div>
      <div className="space-y-1 text-xs" data-testid="edge-checks">
        {cites.length === 0 ? (
          <p className="text-emerald-300" data-testid="edge-citations">citations checked against the run&apos;s tool log: ok</p>
        ) : (
          <div className="rounded border border-red-600/60 bg-red-950/30 p-1.5 text-red-200" data-testid="edge-citations">
            <span className="font-bold">UNVERIFIED</span> citations:
            <ul className="list-disc pl-4">{cites.map((c) => <li key={c}>{c}</li>)}</ul>
          </div>
        )}
        {it.grounding === null ? (
          <p className="text-slate-300" data-testid="edge-grounding">numbers: not evaluated (no usable answer to check)</p>
        ) : it.grounding.length === 0 ? (
          <p className="text-emerald-300" data-testid="edge-grounding">numbers checked against tool outputs: ok</p>
        ) : (
          <div className="rounded border border-red-600/60 bg-red-950/30 p-1.5 text-red-200" data-testid="edge-grounding">
            <span className="font-bold">UNVERIFIED</span> numbers not found in tool outputs:
            <ul className="list-disc pl-4">
              {it.grounding.map((g, i) => (
                <li key={i} data-testid="ungrounded">
                  <span className="font-mono font-semibold">{g.number}</span> in {g.field}: &ldquo;{g.context}&rdquo;
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

function AuditCard({ audit: a, earthId }: { audit: Audit; earthId: string | null }) {
  const delta = a.urgency_delta;
  const meaningless = a.agree === null;
  return (
    <div className="mt-2 space-y-1 rounded border border-sky-700/50 bg-slate-950 p-2" data-testid="audit-card">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-sky-300">Audit: edge vs Earth</span>
        <span
          className={`rounded-full border px-2 py-0.5 text-[11px] font-semibold ${meaningless ? "border-slate-500 text-slate-300" : `uppercase ${a.agree ? "border-emerald-500 text-emerald-300" : "border-red-500 text-red-300"}`}`}
          data-testid="audit-verdict"
        >
          {meaningless ? (a.notes.includes("audit not meaningful") ? "audit not meaningful (edge/Earth answer failed validation)" : "not compared") : a.agree ? "agree" : "disagree"}
        </span>
        <span className="text-[11px] italic text-slate-400" data-testid="audit-label">computed by code, not by a model</span>
      </div>
      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs">
        <dt className="text-slate-500">Cause</dt>
        <dd className="text-slate-200" data-testid="audit-cause">
          edge <b>{cause(a.edge_cause)}</b> vs Earth <b>{cause(a.earth_top_cause)}</b>{" "}
          {a.cause_match === null ? "" : a.cause_match ? "(match)" : "(differ)"}
        </dd>
        <dt className="text-slate-500">Urgency</dt>
        <dd className="text-slate-200" data-testid="audit-urgency">
          edge {a.edge_urgency ? (URGENCY_TEXT[a.edge_urgency] ?? a.edge_urgency) : "n/a"} vs Earth status {a.earth_status ?? "n/a"}
        </dd>
        <dt className="text-slate-500">Urgency delta</dt>
        <dd className="font-mono text-slate-200" data-testid="audit-delta">
          {delta === null ? "n/a" : delta > 0 ? `+${delta} (edge more urgent)` : delta < 0 ? `${delta} (edge less urgent)` : "0 (equal)"}
        </dd>
      </dl>
      <p className="text-xs text-slate-300" data-testid="audit-notes">{a.notes}</p>
      {earthId && <p className="font-mono text-[10px] text-slate-500">Earth run {earthId}</p>}
    </div>
  );
}

type Phase = "idle" | "planning" | "plan" | "starting" | "started";

/** Drain: plan first (free), then an explicit confirm click for the paid Earth-side runs. Never automatic. */
export function DrainBox() {
  const { link, refreshLink, refreshSpend } = useMission();
  const [phase, setPhase] = useState<Phase>("idle");
  const [plan, setPlan] = useState<DrainPlan | null>(null);
  const [error, setError] = useState<string | null>(null);
  const busy = useRef(false);
  const state = link?.state;
  // A plan is only valid for the link state it was made in.
  useEffect(() => {
    setPhase("idle");
    setPlan(null);
  }, [state]);

  const explain = (e: unknown, confirmed: boolean) => {
    if (e instanceof ApiError && e.status === 402) return `Budget guard: the estimate would exceed the Meta budget, nothing was sent. ${e.message}`;
    if (e instanceof ApiError && e.status === 409) return `Cannot drain: ${e.message}`;
    return `${confirmed ? "Drain failed" : "Planning failed"}: ${e instanceof Error ? e.message : String(e)}`;
  };

  const doPlan = async () => {
    if (busy.current) return;
    busy.current = true;
    setPhase("planning");
    setError(null);
    try {
      setPlan(await api.drain(false));
      setPhase("plan");
    } catch (e) {
      setError(explain(e, false));
      setPhase("idle");
    } finally {
      busy.current = false;
    }
  };
  const doConfirm = async () => {
    if (busy.current) return; // one paid drain per click
    busy.current = true;
    setPhase("starting");
    setError(null);
    try {
      const r = await api.drain(true);
      setPlan(r);
      setPhase(r.started ? "started" : "plan");
      refreshSpend();
      refreshLink();
    } catch (e) {
      setError(explain(e, true));
      setPhase("plan");
    } finally {
      busy.current = false;
    }
  };

  const n = plan?.item_ids.length ?? 0;
  return (
    <Panel title="Send to Earth (drain)">
      <p className="text-xs text-slate-400">
        Delivering the queue runs the Earth-side Flight Agronomist, a paid Meta call (about $0.10-0.20 per item). Planning is free; nothing is sent until you confirm.
      </p>
      {(phase === "idle" || phase === "planning") && (
        <button
          onClick={() => void doPlan()}
          disabled={phase === "planning"}
          className="mt-2 rounded border border-emerald-500 bg-emerald-950 px-3 py-1.5 text-sm font-semibold text-emerald-200 hover:bg-emerald-900 disabled:opacity-50"
          data-testid="drain-plan"
        >
          {phase === "planning" ? "Planning..." : "Plan drain"}
        </button>
      )}
      {plan && (phase === "plan" || phase === "starting" || phase === "started") && (
        <div className="mt-2 rounded border border-amber-500/70 bg-amber-950/40 p-2" data-testid="drain-plan-box">
          <p className="text-sm text-amber-100" data-testid="drain-plan-summary">
            {n} item{n === 1 ? "" : "s"} deliverable over a {plan.link_state} link. Estimated cost ${plan.est_cost_usd.low.toFixed(2)}-$
            {plan.est_cost_usd.high.toFixed(2)} (paid, Meta Muse Spark).
          </p>
          {plan.detail && <p className="mt-0.5 text-xs text-amber-200/80" data-testid="drain-detail">{plan.detail}</p>}
          {phase !== "started" && n > 0 && (
            <div className="mt-2 flex gap-2">
              <button
                onClick={() => void doConfirm()}
                disabled={phase === "starting"}
                className="rounded bg-emerald-600 px-3 py-1 text-sm font-semibold text-white hover:bg-emerald-500 disabled:opacity-50"
                data-testid="drain-confirm"
              >
                {phase === "starting" ? "Starting..." : `Confirm and send (up to $${plan.est_cost_usd.high.toFixed(2)})`}
              </button>
              <button
                onClick={() => {
                  setPhase("idle");
                  setPlan(null);
                }}
                disabled={phase === "starting"}
                className="rounded border border-slate-500 px-3 py-1 text-sm hover:bg-slate-800 disabled:opacity-50"
              >
                Cancel
              </button>
            </div>
          )}
          {phase === "started" && (
            <p className="mt-1 text-xs text-emerald-300" data-testid="drain-started">
              Started. Item statuses update in the queue below.
            </p>
          )}
        </div>
      )}
      {phase === "started" && (
        <button
          onClick={() => {
            setPhase("idle");
            setPlan(null);
          }}
          className="mt-2 rounded border border-slate-500 px-2 py-0.5 text-xs hover:bg-slate-800"
        >
          Plan another
        </button>
      )}
      {error && (
        <p className="mt-2 rounded border border-red-600/50 bg-red-950/30 p-1.5 text-sm text-red-200" role="alert" data-testid="drain-error">
          {error}
        </p>
      )}
    </Panel>
  );
}
