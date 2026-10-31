import type {
  AlertItem, AreaValidation, AssessmentResponse, Clock, DrainPlan, EdgeEvalLatest, EvalLatest, Evidence, Frame, LinkInfo, LinkState, Page, QueueItem, Series, Spend, StreamEvent,
  TrayDetail, TrayListItem,
} from "./types";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
export const FIXTURES = process.env.NEXT_PUBLIC_FIXTURES === "1";
export const STREAM_GAP_MS = 300;

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

type Params = Record<string, string | number | undefined | null>;

const q = (params: Params) => {
  const s = Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== null && v !== "")
    .map(([k, v]) => `${k}=${encodeURIComponent(String(v))}`)
    .join("&");
  return s ? `?${s}` : "";
};

// ---- fixture adapter -------------------------------------------------------------------------
const fx = async <T>(file: string): Promise<T> => {
  const r = await fetch(`/fixtures/${file}`);
  if (!r.ok) throw new ApiError(r.status, `fixture ${file} missing`);
  return r.json();
};
let fixtureClock: Clock | null = null;
const FIXTURE_ASSESSMENT_TS = "2016-01-03T10:00:00Z";
const FRAME_SOURCE: Record<string, string> = { A: "dataset", B: "fault_transform", C: "fault_timewarp", L: "dataset" };

// Mars mode in fixtures: a small stateful stand-in so the whole flow can be clicked through without a backend.
// It mirrors the backend rules in backend/api/link.py (409 when up / already queued / down / same state).
let linkFx: LinkInfo | null = null;
let queueFx: QueueItem[] | null = null;
async function fixtureQueue(): Promise<QueueItem[]> {
  if (!queueFx) queueFx = (await fx<{ items: QueueItem[] }>("link_queue.json")).items;
  return queueFx;
}
async function fixtureLink(): Promise<LinkInfo> {
  if (!linkFx) linkFx = await fx<LinkInfo>("link.json");
  const depth = (await fixtureQueue()).filter((i) => ["queued", "edge_done", "sent"].includes(i.status)).length;
  return { ...linkFx, queue_depth: depth };
}

async function fixtureGet(path: string, params: Params): Promise<unknown> {
  const parts = path.split("/").filter(Boolean);
  if (path === "/health") return fx("health.json");
  if (path === "/clock") return fixtureClock ?? (fixtureClock = await fx<Clock>("clock.json"));
  if (path === "/spend") return fx("spend.json");
  if (path === "/trays") return fx("trays.json");
  if (path === "/alerts") {
    const d = await fx<{ items: AlertItem[] }>("alerts.json");
    return { items: d.items.filter((a) => !params.tray_id || a.tray_id === params.tray_id) };
  }
  if (path === "/eval/latest") return fx(params.role === "edge" ? "eval_latest_edge.json" : "eval_latest.json");
  if (path === "/eval/area-validation") return fx("area_validation.json");
  if (path === "/link") return fixtureLink();
  if (path === "/link/queue") return { items: await fixtureQueue() };
  if (parts[0] === "trays" && parts.length === 2) {
    if (parts[1] === "B") return fx("tray_B.json");
    const list = await fx<{ items: TrayListItem[] }>("trays.json");
    const t = list.items.find((x) => x.id === parts[1]);
    if (!t) throw new ApiError(404, "tray not found");
    return { ...t, age_days: 20, mission_day: 21, calib_flag: "nominal", latest: null, open_alerts: [] };
  }
  if (parts[0] === "trays" && parts[2] === "frames") {
    const d = await fx<Page<Frame>>("frames_B.json");
    const id = parts[1];
    return {
      next_cursor: null,
      items: d.items.map((f) => ({
        ...f,
        tray_id: id,
        id: f.id.replace(/^B/, id),
        source: FRAME_SOURCE[id] ?? f.source,
        image_url: f.image_url.replace(/trays\/B\/frames\/B/, `trays/${id}/frames/${id}`),
        overlay_url: f.overlay_url.replace(/trays\/B\/frames\/B/, `trays/${id}/frames/${id}`),
      })),
    };
  }
  if (parts[0] === "trays" && parts[2] === "series") {
    const file = `series_${parts[1]}_${params.metric}.json`;
    const index = await fx<string[]>("_index.json");
    if (index.includes(file)) return fx(file);
    return { tray_id: parts[1], metric: String(params.metric), unit: "", items: [] };
  }
  if (parts[0] === "assessments" && parts.length === 2) {
    const a = await fx<AssessmentResponse & { ts?: string }>("assessment.json");
    if (params.as_of && String(params.as_of) < FIXTURE_ASSESSMENT_TS) throw new ApiError(404, "assessment is later than as_of");
    return a;
  }
  if (parts[0] === "evidence") {
    const map = await fx<Record<string, Evidence>>("evidence.json");
    const id = decodeURIComponent(parts.slice(1).join("/"));
    if (!map[id]) throw new ApiError(404, "evidence not found");
    return map[id];
  }
  throw new ApiError(404, `no fixture for ${path}`);
}

async function fixturePost(path: string, body: Record<string, unknown>): Promise<unknown> {
  if (path === "/clock") {
    const cur = fixtureClock ?? (fixtureClock = await fx<Clock>("clock.json"));
    fixtureClock = { ...cur, ...(body as Partial<Clock>) };
    fixtureClock.mission_day = Math.floor((Date.parse(fixtureClock.as_of) - Date.UTC(2015, 11, 14)) / 86400000) + 1;
    return fixtureClock;
  }
  if (path.startsWith("/alerts/") && path.endsWith("/assess")) {
    const w = window as unknown as { __ceresAssessPosts?: number };
    w.__ceresAssessPosts = (w.__ceresAssessPosts ?? 0) + 1; // lets the smoke test count submissions
    const a = await fx<AssessmentResponse>("assessment.json");
    return { assessment_id: a.assessment_id };
  }
  if (path.startsWith("/actions/")) {
    return { action_id: path.split("/")[2], status: body.decision === "approve" ? "approved" : "rejected" };
  }
  if (path === "/link") {
    const cur = await fixtureLink();
    const st = body.state as LinkState;
    const delay = st === "delayed" ? Number(body.delay_s) : null;
    if (st === "delayed" && !(delay! > 0 && delay! <= 3600)) throw new ApiError(422, "delay_s must be in (0, 3600]");
    if (st === cur.state && delay === cur.delay_s) throw new ApiError(409, `the link is already ${st}`);
    linkFx = { ...cur, state: st, delay_s: delay };
    return fixtureLink();
  }
  if (path === "/link/queue") {
    const link = await fixtureLink();
    const items = await fixtureQueue();
    const id = String(body.alert_id);
    if (link.state === "up") throw new ApiError(409, "the link is up: nothing to queue (POST /alerts/{id}/assess reaches Earth directly)");
    if (items.some((i) => i.alert_id === id)) throw new ApiError(409, `alert ${id} is already queued`);
    const item: QueueItem = {
      ...items[0], id: `lq_${id}`, alert_id: id, status: "edge_done", audit_json: null, earth_assessment_id: null,
      deliverable: link.state !== "down", deliver_in_s: link.state === "delayed" ? link.delay_s : null,
    };
    queueFx = [...items, item];
    return item;
  }
  if (path === "/link/drain") {
    const link = await fixtureLink();
    if (link.state === "down") throw new ApiError(409, "the link is down: nothing can be delivered");
    const items = await fixtureQueue();
    const due = items.filter((i) => i.status === "queued" || i.status === "edge_done" || i.status === "edge_invalid");
    const plan: DrainPlan = {
      started: false, confirm_required: body.confirm !== true, link_state: link.state, item_ids: due.map((i) => i.id),
      est_cost_usd: { low: +(due.length * 0.1).toFixed(2), high: +(due.length * 0.2).toFixed(2) },
      detail: due.length ? 'send {"confirm": true} to run the Earth-side agronomist on these items (paid)' : "nothing is deliverable yet",
    };
    if (body.confirm !== true || !due.length) return plan;
    const w = window as unknown as { __ceresDrainConfirms?: number };
    w.__ceresDrainConfirms = (w.__ceresDrainConfirms ?? 0) + 1; // lets the smoke test count confirmed drains
    const audit = items.find((i) => i.audit_json)?.audit_json ?? null;
    queueFx = items.map((i) => (due.includes(i) ? { ...i, status: "audited", audit_json: audit, earth_assessment_id: "asmt_B_20160103T100000Z", deliverable: false } : i));
    return { ...plan, started: true, detail: "drain started; poll GET /link/queue for item statuses" };
  }
  throw new ApiError(404, `no fixture for POST ${path}`);
}

// ---- transport -------------------------------------------------------------------------------
async function request<T>(method: "GET" | "POST", path: string, params: Params = {}, body?: Record<string, unknown>): Promise<T> {
  if (FIXTURES) return (method === "GET" ? fixtureGet(path, params) : fixturePost(path, body ?? {})) as Promise<T>;
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}${q(params)}`, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new ApiError(0, `Backend unreachable at ${API_URL}`);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j);
    } catch {
      /* keep statusText */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json();
}
const get = <T>(path: string, params: Params = {}) => request<T>("GET", path, params);
const post = <T>(path: string, params: Params, body?: Record<string, unknown>) => request<T>("POST", path, params, body);

// ---- typed client (every read passes as_of) --------------------------------------------------
export const api = {
  health: () => get<{ status: string; demo_mode: boolean; edge_mode: string }>("/health"),
  clock: () => get<Clock>("/clock"),
  setClock: (b: Partial<Pick<Clock, "as_of" | "speed" | "running">>) => post<Clock>("/clock", {}, b),
  spend: () => get<Spend>("/spend"),
  trays: (asOf?: string) => get<{ items: TrayListItem[] }>("/trays", { as_of: asOf }),
  tray: (id: string, asOf?: string) => get<TrayDetail>(`/trays/${id}`, { as_of: asOf }),
  frames: (id: string, asOf?: string, limit = 200, cursor?: string | null) =>
    get<Page<Frame>>(`/trays/${id}/frames`, { limit, cursor, as_of: asOf }),
  series: (id: string, metric: string, hours: number, asOf?: string) =>
    get<Series>(`/trays/${id}/series`, { metric, hours, as_of: asOf }),
  alerts: (asOf?: string, trayId?: string) => get<{ items: AlertItem[] }>("/alerts", { tray_id: trayId, as_of: asOf }),
  assess: (alertId: string, force = false) =>
    post<{ assessment_id: string }>(`/alerts/${alertId}/assess`, { force: force ? 1 : undefined }, {}),
  assessment: (id: string, asOf?: string) => get<AssessmentResponse>(`/assessments/${id}`, { as_of: asOf }),
  evidence: (id: string, asOf?: string) => get<Evidence>(`/evidence/${id}`, { as_of: asOf }),
  decide: (actionId: string, decision: "approve" | "reject") =>
    post<{ action_id: string; status: string }>(`/actions/${actionId}`, {}, { decision }),
  /** Static validation report (not as_of-gated); null when it has not been generated (404). */
  areaValidation: () =>
    get<AreaValidation>("/eval/area-validation").catch((e) => {
      if (e instanceof ApiError && e.status === 404) return null;
      throw e;
    }),
  /** Static agent-eval report (not as_of-gated); null when no run has written it (404). */
  evalLatest: () =>
    get<EvalLatest>("/eval/latest").catch((e) => {
      if (e instanceof ApiError && e.status === 404) return null;
      throw e;
    }),
  /** Edge-triage eval (role=edge); null when no run has written it (404). */
  evalLatestEdge: () =>
    get<EdgeEvalLatest>("/eval/latest", { role: "edge" }).catch((e) => {
      if (e instanceof ApiError && e.status === 404) return null;
      throw e;
    }),
  link: (asOf?: string) => get<LinkInfo>("/link", { as_of: asOf }),
  setLink: (state: LinkState, delayS?: number) =>
    post<LinkInfo>("/link", {}, delayS !== undefined ? { state, delay_s: delayS } : { state }),
  linkQueue: (asOf?: string) => get<{ items: QueueItem[] }>("/link/queue", { as_of: asOf }),
  enqueue: (alertId: string, asOf?: string) => post<QueueItem>("/link/queue", { as_of: asOf }, { alert_id: alertId }),
  /** Without confirm: the plan only (free). With confirm: true: starts the PAID Earth-side runs. */
  drain: (confirm: boolean) => post<DrainPlan>("/link/drain", {}, { confirm }),
};

/** Absolute URL for an image path from the API (frame image or overlay). */
export function imageUrl(path: string, kind: "image" | "overlay", asOf?: string): string {
  if (FIXTURES) return kind === "overlay" ? "/placeholder-overlay.svg" : "/placeholder-frame.svg";
  // Quantised to the end of the mission hour so a polling clock does not re-download every frame.
  const hourEnd = asOf ? `${asOf.slice(0, 13)}:59:59Z` : undefined;
  return `${API_URL}${path}${q({ as_of: hourEnd })}`;
}
export const frameImagePath = (trayId: string, frameId: string, kind: "image" | "overlay") =>
  `/trays/${trayId}/frames/${frameId}/${kind}`;

/**
 * Stream an assessment run. Fixture mode replays assessment_stream.json at STREAM_GAP_MS per event.
 * Returns a cancel function. The stream ends after an assessment or error event.
 */
export function streamAssessment(id: string, onEvent: (e: StreamEvent) => void, onEnd: (err?: string) => void): () => void {
  let cancelled = false;
  if (FIXTURES) {
    (async () => {
      try {
        const events = await fx<StreamEvent[]>("assessment_stream.json");
        for (const e of events) {
          await new Promise((r) => setTimeout(r, STREAM_GAP_MS));
          if (cancelled) return;
          onEvent(e);
        }
        onEnd();
      } catch (err) {
        if (!cancelled) onEnd(String(err));
      }
    })();
    return () => {
      cancelled = true;
    };
  }
  const es = new EventSource(`${API_URL}/assessments/${id}/stream`);
  let finished = false;
  const finish = (err?: string) => {
    if (finished) return;
    finished = true;
    es.close();
    if (!cancelled) onEnd(err);
  };
  for (const name of ["tool_call", "tool_result", "assessment", "error"] as const) {
    es.addEventListener(name, (ev) => {
      if (cancelled) return;
      let data: unknown = {};
      try {
        data = JSON.parse((ev as MessageEvent).data);
      } catch {
        /* ignore malformed */
      }
      onEvent({ event: name, data } as StreamEvent);
      if (name === "assessment") finish();
      if (name === "error") finish((data as { message?: string }).message ?? "stream error");
    });
  }
  es.onerror = () => finish("stream connection lost");
  return () => {
    cancelled = true;
    es.close();
  };
}
