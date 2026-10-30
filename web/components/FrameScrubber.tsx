"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, imageUrl } from "@/lib/api";
import { useAsync, useMission } from "@/lib/mission";
import type { Frame } from "@/lib/types";
import { Empty, ErrorNote, Img, Loading, Panel, SourceBadge, shortTs } from "./ui";

export default function FrameScrubber({ trayId, onTs }: { trayId: string; onTs: (ts: string | null) => void }) {
  const { asOf, hourKey, focus } = useMission();
  const first = useAsync((a) => api.frames(trayId, a, 200), [trayId]);
  const [older, setOlder] = useState<{ trayId: string; frames: Frame[]; cursor: string | null | undefined }>({
    trayId, frames: [], cursor: undefined,
  });
  const [selId, setSelId] = useState<string | null>(null);
  const [overlay, setOverlay] = useState(false);
  const [olderError, setOlderError] = useState<string | null>(null);

  const olderForTray = older.trayId === trayId ? older : { trayId, frames: [] as Frame[], cursor: undefined };
  // Start over (newest frames, latest selected) whenever the mission hour or tray changes.
  useEffect(() => {
    setOlder({ trayId, frames: [], cursor: undefined });
    setSelId(null);
  }, [trayId, hourKey]);
  const frames = useMemo(() => {
    const m = new Map<string, Frame>();
    for (const f of [...olderForTray.frames, ...(first.data?.items ?? [])]) if (!asOf || f.ts <= asOf) m.set(f.id, f);
    return [...m.values()].sort((a, b) => a.ts.localeCompare(b.ts));
  }, [first.data, olderForTray.frames, asOf]);
  const cursor = olderForTray.cursor === undefined ? (first.data?.next_cursor ?? null) : olderForTray.cursor;

  const idx = selId ? frames.findIndex((f) => f.id === selId) : frames.length - 1;
  const cur = idx >= 0 ? frames[idx] : frames[frames.length - 1];
  const curIdx = cur ? frames.indexOf(cur) : -1;

  useEffect(() => {
    onTs(cur ? cur.ts : null);
  }, [cur, onTs]);

  const loadOlder = useCallback(async (): Promise<Frame[]> => {
    if (!cursor) return [];
    try {
      const page = await api.frames(trayId, asOf, 200, cursor);
      setOlder((o) => ({
        trayId,
        frames: [...(o.trayId === trayId ? o.frames : []), ...page.items],
        cursor: page.next_cursor,
      }));
      setOlderError(null);
      return page.items;
    } catch (e) {
      setOlderError(e instanceof Error ? e.message : String(e));
      return [];
    }
  }, [cursor, trayId, asOf]);

  // Jump to an evidence frame (or the nearest frame at or before its timestamp).
  const handled = useRef(0);
  useEffect(() => {
    if (!focus || (focus.trayId !== trayId && focus.trayId !== "air") || handled.current === focus.nonce || frames.length === 0) return;
    const pick = () => {
      if (focus.frameId) {
        const f = frames.find((x) => x.id === focus.frameId);
        if (f) return f;
      }
      if (focus.ts) {
        const before = frames.filter((x) => x.ts <= focus.ts!);
        if (before.length) return before[before.length - 1];
        if (!cursor) return frames[0];
      }
      return null;
    };
    const f = pick();
    if (f) {
      handled.current = focus.nonce;
      setSelId(f.id);
    }
  }, [focus, trayId, frames, cursor]);

  if (first.error && !first.data) return <Panel title="Frames"><ErrorNote error={first.error} retry={first.reload} /></Panel>;
  if (!first.data) return <Panel title="Frames"><Loading what="Loading frames" /></Panel>;
  if (!cur) return <Panel title="Frames"><Empty>No frames yet at this mission time.</Empty></Panel>;

  const useOverlay = overlay;
  const url = imageUrl(useOverlay ? cur.overlay_url : cur.image_url, useOverlay ? "overlay" : "image", asOf);
  return (
    <Panel title="Frame scrubber" right={<SourceBadge source={cur.source} testId="frame-source" />}>
      <div className="relative aspect-[4/3] w-full overflow-hidden rounded border border-slate-700 bg-slate-950">
        <Img src={url} alt={`Tray ${trayId} frame ${cur.ts} ${useOverlay ? "mask overlay" : ""}`} className="h-full w-full object-contain" />
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
        <button className="rounded border border-slate-600 px-2 py-1 hover:bg-slate-800 disabled:opacity-40" disabled={curIdx <= 0} onClick={() => setSelId(frames[curIdx - 1].id)} aria-label="Previous frame">Prev</button>
        <input
          type="range"
          min={0}
          max={Math.max(frames.length - 1, 0)}
          value={Math.max(curIdx, 0)}
          onChange={(e) => setSelId(frames[Number(e.target.value)].id)}
          className="min-w-24 flex-1 accent-emerald-400"
          aria-label="Frame slider"
          data-testid="frame-slider"
        />
        <button className="rounded border border-slate-600 px-2 py-1 hover:bg-slate-800 disabled:opacity-40" disabled={curIdx >= frames.length - 1} onClick={() => setSelId(frames[curIdx + 1].id)} aria-label="Next frame">Next</button>
        <button
          aria-pressed={overlay}
          onClick={() => setOverlay((o) => !o)}
          className={`rounded border px-2 py-1 ${overlay ? "border-cyan-400 bg-cyan-950 text-cyan-200" : "border-slate-600 hover:bg-slate-800"}`}
        >
          Mask overlay
        </button>
      </div>
      <div className="mt-1 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400">
        <span data-testid="frame-ts">{shortTs(cur.ts)} · frame {curIdx + 1}/{frames.length}</span>
        <span>scale {cur.cm2_per_px} cm²/px ({cur.calib_flag})</span>
        {cursor && (
          <button className="rounded border border-slate-600 px-2 py-0.5 hover:bg-slate-800" onClick={() => void loadOlder()}>Load older frames</button>
        )}
      </div>
      {olderError && <ErrorNote error={olderError} />}
    </Panel>
  );
}
