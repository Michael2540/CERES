"use client";

import { useEffect, useRef, useState } from "react";
import { ApiError, api } from "@/lib/api";
import { useMission } from "@/lib/mission";
import type { LinkInfo, LinkState } from "@/lib/types";

const LINK_STYLE: Record<LinkState, string> = {
  up: "border-emerald-600 bg-emerald-950 text-emerald-300",
  delayed: "border-amber-500 bg-amber-950 text-amber-300",
  down: "border-red-500 bg-red-950 text-red-300",
};

export function linkLabel(l: LinkInfo): string {
  if (l.state === "delayed") return `DELAYED (${l.delay_s !== null ? Number(l.delay_s.toFixed(1)) : "?"} s)`;
  return l.state.toUpperCase();
}

/** Which brain answers: the Earth-side agronomist while the link is up, the edge model otherwise. Nothing runs on the
 * habitat computer: an edge answer is a replayed recording or a remote API call standing in for an on-board model. */
export function brainLabel(l: LinkInfo): { text: string; recorded: boolean } {
  if (l.state === "up") return { text: `EARTH (${l.earth_model ?? "model unknown"})`, recorded: false };
  const model = l.edge_model ?? "no model";
  const recorded = l.edge_location === "recorded";
  return {
    text: recorded ? `RECORDED · EDGE (${model})` : `EDGE (${model}, remote API: stands in for an on-board model)`,
    recorded,
  };
}

/** Header: link state, which brain is answering, "simulated link" and the demo control. */
export default function LinkStatus() {
  const { link, linkError } = useMission();
  if (!link) {
    return (
      <span
        className={`rounded border px-2 py-0.5 text-[11px] font-semibold ${linkError ? "border-red-500 bg-red-950 text-red-300" : "border-slate-600 text-slate-400"}`}
        data-testid="link-chip"
        title={linkError ?? "Reading link state"}
      >
        {linkError ? "LINK: unavailable" : "LINK: ..."}
      </span>
    );
  }
  const brain = brainLabel(link);
  return (
    <div className="flex flex-wrap items-center gap-1.5" data-testid="link-status">
      <span
        className={`rounded border px-2 py-0.5 text-[11px] font-semibold ${LINK_STYLE[link.state]}`}
        data-testid="link-chip"
        title={link.light_delay_note}
      >
        LINK: {linkLabel(link)}
      </span>
      <span
        className={`rounded border px-2 py-0.5 text-[11px] font-semibold ${link.state === "up" ? "border-sky-600 bg-sky-950 text-sky-300" : "border-violet-500 bg-violet-950 text-violet-200"}`}
        data-testid="brain-badge"
        title={link.state === "up" ? "Earth-side Flight Agronomist" : link.edge_location === "recorded" ? "Edge triage: a recorded model response, replayed" : "Edge triage: a live remote API call, not a model on the habitat computer"}
      >
        {brain.text}
      </span>
      <span className="text-[10px] uppercase tracking-wide text-slate-400" data-testid="link-simulated">simulated link</span>
      <LinkControl link={link} />
    </div>
  );
}

function LinkControl({ link }: { link: LinkInfo }) {
  const { refreshLink } = useMission();
  const [delay, setDelay] = useState("30");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const box = useRef<HTMLDetailsElement>(null);
  // Closes on Escape or a click outside, so the popover never covers the page it controls.
  useEffect(() => {
    const close = () => box.current?.removeAttribute("open");
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && close();
    const onDown = (e: PointerEvent) => {
      if (box.current && !box.current.contains(e.target as Node)) close();
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("pointerdown", onDown);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("pointerdown", onDown);
    };
  }, []);

  const set = async (state: LinkState) => {
    if (busy) return;
    const d = Number(delay);
    if (state === "delayed" && !(d > 0 && d <= 3600)) {
      setError("Delay must be more than 0 and at most 3600 seconds.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      refreshLink(await api.setLink(state, state === "delayed" ? d : undefined));
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        setError(`Already ${state === "delayed" ? `delayed at ${d} s` : state}: nothing changed.`);
        refreshLink();
      } else {
        setError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setBusy(false);
    }
  };

  const btn = (state: LinkState, label: string) => (
    <button
      onClick={() => void set(state)}
      disabled={busy}
      aria-pressed={link.state === state}
      data-testid={`link-set-${state}`}
      className={`rounded border px-2 py-0.5 text-xs disabled:opacity-50 ${link.state === state ? "border-slate-300 bg-slate-700" : "border-slate-500 hover:bg-slate-800"}`}
    >
      {label}
    </button>
  );
  return (
    <details ref={box} className="relative" data-testid="link-control">
      <summary className="cursor-pointer list-none rounded border border-slate-600 px-2 py-0.5 text-xs text-slate-300 hover:bg-slate-800">
        Link control
      </summary>
      <div className="absolute right-0 z-40 mt-1 w-72 space-y-2 rounded-lg border border-slate-600 bg-slate-900 p-3 shadow-xl">
        <p className="text-xs text-slate-400">Demo control. The link is simulated; the delay below is a scaled wall-clock delay.</p>
        <p className="text-xs text-slate-300" data-testid="light-delay-note">{link.light_delay_note}</p>
        <div className="flex flex-wrap items-center gap-1.5">
          {btn("up", "Up")}
          {btn("down", "Down")}
        </div>
        <div className="flex items-center gap-1.5">
          <label className="text-xs text-slate-400" htmlFor="link-delay">Delay (s)</label>
          <input
            id="link-delay"
            type="number"
            min={1}
            max={3600}
            value={delay}
            onChange={(e) => setDelay(e.target.value)}
            data-testid="link-delay-input"
            className="w-20 rounded border border-slate-500 bg-slate-950 px-1 py-0.5 text-xs"
          />
          {btn("delayed", "Delayed")}
        </div>
        {error && <p className="text-xs text-red-400" role="alert" data-testid="link-error">{error}</p>}
      </div>
    </details>
  );
}
