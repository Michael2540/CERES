"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import type { AlertItem, Clock, LinkInfo, Spend } from "./types";

export type Focus = { trayId: string; frameId?: string; ts?: string; evidenceId: string; metric?: string; nonce: number };

type Mission = {
  clock: Clock | null;
  clockError: string | null;
  asOf: string | undefined;
  /** Changes once per mission hour; data hooks refetch on it. */
  hourKey: string;
  setClock: (b: Partial<Pick<Clock, "as_of" | "speed" | "running">>) => Promise<void>;
  spend: Spend | null;
  refreshSpend: () => void;
  panelAlert: AlertItem | null;
  openPanel: (a: AlertItem | null) => void;
  focus: Focus | null;
  setFocus: (f: Omit<Focus, "nonce"> | null) => void;
  /** Simulated Earth link (null until the first read; linkError when /link is unreachable). */
  link: LinkInfo | null;
  linkError: string | null;
  /** Re-read /link now, optionally with a fresh value from a POST, and tell queue views to refetch. */
  refreshLink: (fresh?: LinkInfo) => void;
  /** Bumps on every link change or queue action; queue views refetch on it. */
  queueRev: number;
};

const Ctx = createContext<Mission | null>(null);

export function MissionProvider({ children }: { children: React.ReactNode }) {
  const [clock, setClockState] = useState<Clock | null>(null);
  const [clockError, setClockError] = useState<string | null>(null);
  const [spend, setSpend] = useState<Spend | null>(null);
  const [panelAlert, openPanel] = useState<AlertItem | null>(null);
  const [focus, setFocusState] = useState<Focus | null>(null);
  const [link, setLink] = useState<LinkInfo | null>(null);
  const [linkError, setLinkError] = useState<string | null>(null);
  const [queueRev, setQueueRev] = useState(0);

  // Responses that arrive out of order (an old poll finishing after a newer POST) are dropped.
  const seq = useRef(0);
  const applied = useRef(0);
  const pull = useCallback(async () => {
    const mine = ++seq.current;
    try {
      const c = await api.clock();
      if (mine < applied.current) return;
      applied.current = mine;
      setClockState(c);
      setClockError(null);
    } catch (e) {
      if (mine >= applied.current) setClockError(e instanceof Error ? e.message : String(e));
    }
  }, []);
  const refreshSpend = useCallback(() => {
    api.spend().then(setSpend).catch(() => setSpend(null));
  }, []);

  useEffect(() => {
    pull();
    refreshSpend();
  }, [pull, refreshSpend]);

  const running = clock?.running ?? false;
  useEffect(() => {
    const t = setInterval(() => {
      pull();
    }, running ? 2000 : 5000);
    return () => clearInterval(t);
  }, [running, pull]);

  // Data refetches are keyed on the mission hour, at most once per 1.5 s however fast the clock runs.
  const rawHour = clock ? clock.as_of.slice(0, 13) : "";
  const [hourKey, setHourKey] = useState("");
  const lastHour = useRef(0);
  useEffect(() => {
    if (!rawHour || rawHour === hourKey) return;
    const wait = Math.max(0, lastHour.current + 1500 - Date.now());
    const t = setTimeout(() => {
      lastHour.current = Date.now();
      setHourKey(rawHour);
    }, wait);
    return () => clearTimeout(t);
  }, [rawHour, hourKey]);
  useEffect(() => {
    if (hourKey) refreshSpend();
  }, [hourKey, refreshSpend]);

  const setClock = useCallback(async (b: Partial<Pick<Clock, "as_of" | "speed" | "running">>) => {
    const mine = ++seq.current;
    try {
      const c = await api.setClock(b);
      if (mine >= applied.current) {
        applied.current = mine;
        setClockState(c);
      }
      setClockError(null);
    } catch (e) {
      setClockError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  // The link is time-gated by as_of like everything else; polled every 4 s because a delayed link elapses in wall seconds.
  const asOfNow = clock?.as_of;
  const asOfLink = useRef(asOfNow);
  useEffect(() => {
    asOfLink.current = asOfNow;
  });
  const linkSeq = useRef(0);
  const refreshLink = useCallback((fresh?: LinkInfo) => {
    setQueueRev((n) => n + 1);
    if (fresh) {
      linkSeq.current++;
      setLink(fresh);
      setLinkError(null);
      return;
    }
    const a = asOfLink.current;
    if (!a) return;
    const mine = ++linkSeq.current;
    api
      .link(a)
      .then((l) => {
        if (mine === linkSeq.current) {
          setLink(l);
          setLinkError(null);
        }
      })
      .catch((e) => mine === linkSeq.current && setLinkError(e instanceof Error ? e.message : String(e)));
  }, []);
  useEffect(() => {
    if (hourKey) refreshLink();
  }, [hourKey, refreshLink]);
  useEffect(() => {
    if (!hourKey) return;
    const t = setInterval(() => refreshLink(), 4000);
    return () => clearInterval(t);
  }, [hourKey, refreshLink]);

  const nonce = useRef(0);
  const setFocus = useCallback((f: Omit<Focus, "nonce"> | null) => {
    setFocusState(f ? { ...f, nonce: ++nonce.current } : null);
  }, []);

  const value = useMemo<Mission>(
    () => ({
      clock, clockError, asOf: clock?.as_of, hourKey, setClock, spend, refreshSpend, panelAlert, openPanel, focus, setFocus,
      link, linkError, refreshLink, queueRev,
    }),
    [clock, clockError, hourKey, setClock, spend, refreshSpend, panelAlert, focus, setFocus, link, linkError, refreshLink, queueRev],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useMission(): Mission {
  const v = useContext(Ctx);
  if (!v) throw new Error("MissionProvider missing");
  return v;
}

export type Async<T> = { data: T | null; error: string | null; loading: boolean; reload: () => void };

/**
 * Fetch with the mission clock's as_of. Waits for the clock, refetches once per mission hour
 * (and when `deps` change). Keeps previous data while refetching.
 */
export function useAsync<T>(fn: (asOf: string) => Promise<T>, deps: unknown[]): Async<T> {
  const { asOf, hourKey, clockError } = useMission();
  const [state, setState] = useState<{ data: T | null; error: string | null; loading: boolean }>({
    data: null, error: null, loading: true,
  });
  const [nonce, setNonce] = useState(0);
  const fnRef = useRef(fn);
  const asOfRef = useRef(asOf);
  useEffect(() => {
    fnRef.current = fn;
    asOfRef.current = asOf;
  });
  const ready = Boolean(hourKey);
  useEffect(() => {
    if (!ready) {
      if (clockError) setState({ data: null, error: clockError, loading: false });
      return;
    }
    let live = true;
    setState((s) => ({ ...s, loading: true }));
    fnRef
      .current(asOfRef.current as string)
      .then((data) => live && setState({ data, error: null, loading: false }))
      .catch((e) => live && setState((s) => ({ data: s.data, error: e instanceof Error ? e.message : String(e), loading: false })));
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, hourKey, clockError, nonce, ...deps]);
  return { ...state, reload: () => setNonce((n) => n + 1) };
}
