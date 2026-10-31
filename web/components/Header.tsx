"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useMission } from "@/lib/mission";
import LinkStatus from "./LinkStatus";
import { Legend, shortTs } from "./ui";

const DAY0 = Date.UTC(2015, 11, 14); // mission day 1 (matches backend/sim/clock.py)
const LAST_DAY = 36;
const active = (p: string, h: string) => (h === "/" ? p === "/" : p.startsWith(h.startsWith("/tray") ? "/tray" : h));
const SPEEDS = [360, 720, 3600];

export default function Header() {
  const { clock, clockError, setClock, spend } = useMission();
  const path = usePathname();
  const nav = [
    ["/", "Overview"],
    ["/tray/B", "Tray B"],
    ["/mars", "Mars mode"],
    ["/trust", "Trust"],
  ] as const;
  return (
    <header className="sticky top-0 z-30 border-b border-slate-700 bg-slate-950/95 backdrop-blur">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 px-3 py-2 sm:px-4">
        <Link href="/" className="text-lg font-bold tracking-[0.25em] text-emerald-300">CERES</Link>
        <nav className="flex gap-1 text-sm" aria-label="Main">
          {nav.map(([href, label]) => (
            <Link
              key={href}
              href={href}
              className={`rounded px-2 py-1 ${active(path, href) ? "bg-slate-800 text-white" : "text-slate-400 hover:text-white"}`}
            >
              {label}
            </Link>
          ))}
        </nav>

        <div className="flex flex-wrap items-center gap-2" data-testid="mission-clock">
          {clock ? (
            <>
              <span className="font-mono text-sm text-slate-100" data-testid="clock-text">
                Mission day {clock.mission_day} · {shortTs(clock.as_of)}
              </span>
              <button
                onClick={() => setClock({ running: !clock.running })}
                className="rounded border border-slate-500 px-2 py-0.5 text-xs hover:bg-slate-800"
                aria-label={clock.running ? "Pause clock" : "Play clock"}
              >
                {clock.running ? "Pause" : "Play"}
              </button>
              <select
                aria-label="Clock speed"
                value={SPEEDS.includes(clock.speed) ? clock.speed : ""}
                onChange={(e) => setClock({ speed: Number(e.target.value) })}
                className="rounded border border-slate-500 bg-slate-900 px-1 py-0.5 text-xs"
              >
                {!SPEEDS.includes(clock.speed) && <option value="">×{clock.speed}</option>}
                {SPEEDS.map((s) => <option key={s} value={s}>×{s}</option>)}
              </select>
              <DayScrubber day={clock.mission_day} asOf={clock.as_of} onCommit={(iso) => setClock({ as_of: iso })} />
            </>
          ) : clockError ? (
            <span className="text-sm text-red-400" role="alert">Clock unavailable: {clockError}</span>
          ) : (
            <span className="text-sm text-slate-500">Loading clock...</span>
          )}
        </div>

        <div className="ml-auto flex flex-wrap items-center gap-2">
          <span
            className="font-mono text-xs text-slate-300"
            data-testid="spend-meter"
            title="Project cap $150 covers Muse Code + runtime"
          >
            {spend ? `Meta spend $${spend.total_usd.toFixed(2)} · runtime guard $${spend.budget_usd.toFixed(0)}` : "Meta spend $-- · runtime guard $--"}
          </span>
          <LinkStatus />
          <Legend />
        </div>
      </div>
    </header>
  );
}

function DayScrubber({ day, asOf, onCommit }: { day: number; asOf: string; onCommit: (iso: string) => void }) {
  const [draft, setDraft] = useDraft(day);
  const dragging = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const commit = (value: number) => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = null;
    if (value === day) return;
    const hh = asOf.slice(11, 19);
    const d = new Date(DAY0 + (value - 1) * 86400_000);
    onCommit(`${d.toISOString().slice(0, 10)}T${hh}Z`);
  };
  return (
    <input
      type="range"
      min={1}
      max={LAST_DAY}
      value={draft}
      aria-label="Mission day scrubber"
      aria-valuetext={`Mission day ${draft}`}
      onPointerDown={() => {
        dragging.current = true;
      }}
      onChange={(e) => {
        const v = Number(e.target.value);
        setDraft(v);
        if (timer.current) clearTimeout(timer.current);
        // Keyboard steps and programmatic changes commit once they settle; a drag waits for pointer-up.
        if (!dragging.current) timer.current = setTimeout(() => commit(v), 600);
      }}
      onPointerUp={(e) => {
        dragging.current = false;
        commit(Number((e.target as HTMLInputElement).value));
      }}
      onKeyDown={(e) => {
        if (e.key === "Enter") commit(draft);
      }}
      className="h-1 w-28 accent-emerald-400"
    />
  );
}

function useDraft(day: number) {
  const [draft, setDraft] = useState(day);
  useEffect(() => setDraft(day), [day]);
  return [draft, setDraft] as const;
}
