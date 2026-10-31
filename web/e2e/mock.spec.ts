import { readFileSync } from "node:fs";
import path from "node:path";
import { expect, test, type Page, type Route } from "@playwright/test";

// Real-API mode (no fixtures build) with every backend request answered by page.route.
// Nothing here can reach a real backend: unmatched requests to :8000 are aborted and counted as failures.

const FIX = path.join(__dirname, "..", "..", "tests", "fixtures", "api");
const fx = (n: string) => JSON.parse(readFileSync(path.join(FIX, n), "utf8"));
const ALERT = "alert_B_GROWTH_LAG_20160103T100000Z";
const AID = "asmt_B_20160103T100000Z";
type QItem = Record<string, unknown> & { id: string; alert_id: string; status: string };
const queueItem = (over: Partial<QItem> = {}): QItem => ({ ...fx("link_queue.json").items[0], ...over });
const EDGE_ONLY = { status: "edge_done", audit_json: null, earth_assessment_id: null, deliverable: true, deliver_in_s: null };
const CORS = { "access-control-allow-origin": "*", "access-control-allow-headers": "*", "access-control-allow-methods": "*" };

type Mode = {
  asOf: string;
  assessmentId: string | null; // set on the alert in GET /alerts
  assessment: "done" | "running" | "failed" | "404";
  stream: "events" | "abort";
  area: "ok" | "404" | "partial";
  evalRun: "ok" | "404";
  edgeEval: "ok" | "404";
  /** Mars mode */
  link: { state: "up" | "delayed" | "down"; delay_s: number | null; edge_model: string | null; edge_mode: string };
  queue: QItem[];
  drainStatus: 200 | 402 | 409; // what a confirmed drain answers
  drainPlans: number; // POST /link/drain without confirm
  drainConfirms: number; // POST /link/drain with confirm:true: must stay 0 until a click
  linkPosts: { state: string; delay_s?: number }[];
  queuePosts: string[];
  warnings: { field: string; number: string; context: string }[] | null;
  assessPosts: number;
  clockPosts: number;
  unexpected: string[];
};

async function mockApi(page: Page, init: Partial<Mode> = {}) {
  const m: Mode = {
    asOf: "2016-01-03T12:00:00Z", assessmentId: null, assessment: "done", stream: "events",
    area: "ok", evalRun: "ok", edgeEval: "ok",
    link: { state: "up", delay_s: null, edge_model: "muse-spark-1.3-contributor", edge_mode: "recorded" },
    queue: [], drainStatus: 200, drainPlans: 0, drainConfirms: 0, linkPosts: [], queuePosts: [], warnings: null,
    assessPosts: 0, clockPosts: 0, unexpected: [], ...init,
  };
  const json = (route: Route, body: unknown, status = 200) =>
    route.fulfill({ status, contentType: "application/json", headers: CORS, body: JSON.stringify(body) });
  const linkView = () => ({
    state: m.link.state, delay_s: m.link.delay_s, since: "2016-01-07T12:00:00Z",
    queue_depth: m.queue.filter((i) => ["queued", "edge_done", "sent"].includes(i.status)).length,
    simulated: true, light_delay_note: fx("link.json").light_delay_note, edge_mode: m.link.edge_mode,
    edge_model: m.link.edge_model, edge_recordings: [],
    edge_location: m.link.edge_mode === "recorded" ? "recorded" : "remote_api", earth_model: "muse-spark-1.3",
  });
  const clock = () => ({
    as_of: m.asOf, speed: 720, running: false,
    mission_day: Math.floor((Date.parse(m.asOf) - Date.UTC(2015, 11, 14)) / 86400000) + 1,
  });
  await page.route("http://localhost:8000/**", async (route) => {
    const req = route.request();
    const u = new URL(req.url());
    const p = u.pathname;
    if (req.method() === "OPTIONS") return route.fulfill({ status: 204, headers: CORS });
    if (/\/(image|overlay)$/.test(p)) {
      return route.fulfill({ contentType: "image/svg+xml", headers: CORS, body: readFileSync(path.join(__dirname, "..", "public", "placeholder-frame.svg")) });
    }
    if (p === "/clock" && req.method() === "GET") return json(route, clock());
    if (p === "/clock" && req.method() === "POST") {
      m.clockPosts++;
      m.asOf = req.postDataJSON().as_of ?? m.asOf;
      return json(route, clock());
    }
    if (p.startsWith("/evidence/")) {
      const e = fx("evidence.json")[decodeURIComponent(p.slice("/evidence/".length))];
      return e ? json(route, e) : json(route, { detail: "unknown or future evidence id" }, 404);
    }
    if (p === "/eval/latest" && u.searchParams.get("role") === "edge") {
      return m.edgeEval === "404" ? json(route, { detail: "no edge eval run yet" }, 404) : json(route, fx("eval_latest_edge.json"));
    }
    if (p === "/link" && req.method() === "GET") return json(route, linkView());
    if (p === "/link" && req.method() === "POST") {
      const b = req.postDataJSON();
      m.linkPosts.push(b);
      if (b.state === m.link.state && (b.delay_s ?? null) === m.link.delay_s) return json(route, { detail: `link is already ${b.state}` }, 409);
      m.link = { ...m.link, state: b.state, delay_s: b.state === "delayed" ? b.delay_s : null };
      return json(route, linkView());
    }
    if (p === "/link/queue" && req.method() === "GET") return json(route, { items: m.queue });
    if (p === "/link/queue" && req.method() === "POST") {
      const id = req.postDataJSON().alert_id;
      m.queuePosts.push(id);
      if (m.link.state === "up") return json(route, { detail: "the link is up: nothing to queue" }, 409);
      if (m.queue.some((i) => i.alert_id === id)) return json(route, { detail: `alert ${id} is already queued` }, 409);
      const it = queueItem({ ...EDGE_ONLY, id: `lq_${id}`, alert_id: id });
      m.queue.push(it);
      return json(route, it);
    }
    if (p === "/link/drain" && req.method() === "POST") {
      const confirm = req.postDataJSON().confirm === true;
      if (m.link.state === "down") return json(route, { detail: "the link is down: nothing can be delivered" }, 409);
      const due = m.queue.filter((i) => i.status === "queued" || i.status === "edge_done");
      const plan = {
        started: false, confirm_required: !confirm, link_state: m.link.state, item_ids: due.map((i) => i.id),
        est_cost_usd: { low: 0.1 * due.length, high: 0.2 * due.length }, detail: "send confirm to run (paid)",
      };
      if (!confirm) {
        m.drainPlans++;
        return json(route, plan);
      }
      m.drainConfirms++;
      if (m.drainStatus === 402) return json(route, { detail: "estimated $0.20 would exceed the budget guard" }, 402);
      if (m.drainStatus === 409) return json(route, { detail: "a drain is already running" }, 409);
      m.queue = m.queue.map((i) => (due.includes(i) ? { ...i, status: "audited", audit_json: fx("link_queue.json").items[0].audit_json } : i));
      return json(route, { ...plan, started: true, detail: "drain started" });
    }
    if (p === "/eval/latest") {
      return m.evalRun === "404" ? json(route, { detail: "no agent eval run yet" }, 404) : json(route, fx("eval_latest.json"));
    }
    if (p === "/eval/area-validation") {
      return m.area === "404" ? json(route, { detail: "area validation has not been run" }, 404) : json(route, m.area === "partial" ? { ...fx("area_validation.json"), partial: true } : fx("area_validation.json"));
    }
    if (p === "/spend") return json(route, fx("spend.json"));
    if (p === "/trays") return json(route, fx("trays.json"));
    if (/^\/trays\/[A-Z]$/.test(p)) return p.endsWith("B") ? json(route, fx("tray_B.json")) : json(route, { error: "n/a" }, 404);
    const fr = p.match(/^\/trays\/([A-Z])\/frames$/);
    if (fr) {
      const d = fx("frames_B.json");
      d.items = d.items.map((f: { id: string }) => ({ ...f, tray_id: fr[1], id: f.id.replace(/^B/, fr[1]) }));
      return json(route, d);
    }
    if (/^\/trays\/[A-Z]\/series$/.test(p)) return json(route, { tray_id: "B", metric: u.searchParams.get("metric"), unit: "", items: [] });
    if (p === "/alerts") {
      const d = fx("alerts.json");
      d.items[0].assessment_id = m.assessmentId;
      return json(route, d);
    }
    if (p === `/alerts/${ALERT}/assess` && req.method() === "POST") {
      m.assessPosts++;
      return json(route, { assessment_id: AID });
    }
    if (p === `/assessments/${AID}`) {
      const a = fx("assessment.json");
      if (m.assessment === "404") return json(route, { detail: "assessment is later than as_of" }, 404);
      if (m.assessment === "running") return json(route, { ...a, assessment: null, valid: null, latency_ms: null, cost_usd: null, actions: [] });
      if (m.assessment === "failed") {
        return json(route, { ...a, assessment: null, valid: false, citation_errors: ["error: timed out"], actions: [] });
      }
      return json(route, m.warnings === null ? a : { ...a, warnings: m.warnings });
    }
    if (p === `/assessments/${AID}/stream`) {
      if (m.stream === "abort") return route.abort();
      const body = (fx("assessment_stream.json") as { event: string; data: unknown }[])
        .map((e) => `event: ${e.event}\ndata: ${JSON.stringify(e.data)}\n\n`)
        .join("");
      return route.fulfill({ contentType: "text/event-stream", headers: CORS, body });
    }
    m.unexpected.push(`${req.method()} ${p}`);
    return route.abort();
  });
  return m;
}

test("double-clicking confirm sends exactly one POST /assess", async ({ page }) => {
  const m = await mockApi(page);
  await page.goto("/");
  await page.getByTestId(`alert-${ALERT}`).click();
  await page.getByTestId("investigate").click();
  expect(m.assessPosts).toBe(0);
  await page.getByTestId("confirm-run").dblclick();
  await expect(page.getByTestId("assessment")).toBeVisible({ timeout: 15_000 });
  expect(m.assessPosts).toBe(1);
  await expect(page.getByTestId("run-meta")).toContainText("$0.090");
  // every evidence chip carries a provenance tag from /evidence `source`
  await expect(page.getByTestId("evidence-prov").first()).toBeVisible();
  await expect(page.getByTestId("assessment")).toContainText("SIMULATED FAULT");
  expect(m.unexpected).toEqual([]);
});

test("a run still in flight is 'running', never unverified, never forceable; re-attach is free", async ({ page }) => {
  const m = await mockApi(page, { assessmentId: AID, assessment: "running", stream: "abort" });
  await page.goto("/");
  await page.getByTestId(`alert-${ALERT}`).click();
  const panel = page.getByTestId("agent-panel");
  await expect(panel.getByTestId("running-note")).toContainText("Still running");
  await expect(panel.getByTestId("running-note")).toContainText("stream lost; re-attach");
  await expect(panel.getByText("UNVERIFIED", { exact: true })).toHaveCount(0);
  await expect(panel.getByTestId("force-rerun")).toHaveCount(0);
  await expect(panel.getByTestId("investigate")).toHaveCount(0);
  // The run finishes while we were disconnected.
  m.assessment = "done";
  await panel.getByTestId("reattach").click();
  await expect(panel.getByTestId("assessment")).toBeVisible();
  expect(m.assessPosts).toBe(0);
  expect(m.unexpected).toEqual([]);
});

test("a terminal failed run shows UNVERIFIED and offers a confirmed force re-run", async ({ page }) => {
  const m = await mockApi(page, { assessmentId: AID, assessment: "failed" });
  await page.goto("/");
  await page.getByTestId(`alert-${ALERT}`).click();
  const panel = page.getByTestId("agent-panel");
  await expect(panel.getByText("UNVERIFIED", { exact: true })).toBeVisible();
  await expect(panel.getByText("error: timed out")).toBeVisible();
  await panel.getByTestId("force-rerun").click();
  await expect(panel.getByTestId("confirm-box")).toContainText("Force re-run");
  expect(m.assessPosts).toBe(0);
});

test("scrubbing back before the alert closes the panel and renders no assessment", async ({ page }) => {
  const m = await mockApi(page, { assessmentId: AID });
  await page.goto("/");
  await page.getByTestId(`alert-${ALERT}`).click();
  await expect(page.getByTestId("assessment")).toBeVisible();
  await page.getByLabel("Mission day scrubber").fill("10");
  await expect.poll(() => m.clockPosts).toBe(1);
  await expect(page.getByTestId("agent-panel")).toHaveCount(0);
  await expect(page.getByTestId("assessment")).toHaveCount(0);
});

test("an assessment later than as_of is cleared with a 'not yet available' note", async ({ page }) => {
  const m = await mockApi(page, { assessmentId: AID });
  await page.goto("/");
  await page.getByTestId(`alert-${ALERT}`).click();
  await expect(page.getByTestId("assessment")).toBeVisible();
  m.assessment = "404"; // as if the run were stamped after the new as_of, while the alert itself is still visible
  await page.getByLabel("Mission day scrubber").fill("22"); // 2016-01-04: a later mission hour, so the panel re-reads
  await expect(page.getByTestId("unavailable")).toContainText("Not yet available");
  await expect(page.getByTestId("assessment")).toHaveCount(0);
});

test("tray page renders in real-API mode without console errors", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  const m = await mockApi(page);
  await page.goto("/tray/B");
  await expect(page.getByTestId("chart-area_cm2")).toBeVisible();
  await expect(page.getByTestId("frame-ts")).toBeVisible();
  expect(errors).toEqual([]);
  expect(m.unexpected).toEqual([]);
});

test("trust page shows the area validation from /eval/area-validation", async ({ page }) => {
  const m = await mockApi(page, { area: "partial" });
  await page.goto("/trust");
  await expect(page.getByTestId("area-n")).toContainText("55");
  await expect(page.getByTestId("area-dice")).toContainText("0.94");
  await expect(page.getByTestId("area-mape")).toContainText("5%");
  await expect(page.getByTestId("area-pot-r")).toContainText("0.97");
  await expect(page.getByTestId("area-plants")).toContainText("75%");
  await expect(page.getByTestId("area-row-hsv")).toContainText("not a model");
  await expect(page.getByTestId("area-partial")).toBeVisible();
  await expect(page.getByTestId("area-empty")).toHaveCount(0);
  expect(m.unexpected).toEqual([]);
});

test("trust page keeps the empty state when the validation report is missing (404)", async ({ page }) => {
  const m = await mockApi(page, { area: "404" });
  await page.goto("/trust");
  await expect(page.getByTestId("area-empty")).toBeVisible();
  await expect(page.getByTestId("area-n")).toHaveCount(0);
  await expect(page.getByTestId("scoreboard")).toBeVisible();
  expect(m.unexpected).toEqual([]);
});

test("trust scoreboard leads with the deterministic metrics and shows fractions, latency, cost and per-scenario chips", async ({ page }) => {
  const m = await mockApi(page);
  await page.goto("/trust");
  const board = page.getByTestId("scoreboard");
  await expect(board).toBeVisible();
  await expect(page.getByTestId("fake-run")).toBeVisible();
  // the deterministic metrics come first in the DOM, before the top-1 table
  const order = await board.evaluate((el) => {
    const q = (id: string) => Array.from(el.querySelectorAll("*")).indexOf(el.querySelector(`[data-testid="${id}"]`) as Element);
    return [q("det-metrics"), q("top1-table"), q("eval-perf"), q("scenario-table")];
  });
  expect(order.every((n) => n >= 0)).toBe(true);
  expect([...order].sort((a, b) => a - b)).toEqual(order);
  await expect(page.getByTestId("det-citation")).toContainText("9/9");
  await expect(page.getByTestId("det-schema")).toContainText("9/9");
  await expect(page.getByTestId("det-sensor")).toContainText("9/9");
  await expect(page.getByTestId("det-injection")).toContainText("1/1");
  await expect(page.getByTestId("top1-table")).toContainText("2/3");
  await expect(page.getByTestId("top1-table")).toContainText("6/6");
  await expect(board).not.toContainText("%");
  await expect(page.getByTestId("small-n")).toBeVisible();
  await expect(page.getByTestId("eval-perf")).toContainText("p50");
  await expect(page.getByTestId("eval-perf")).toContainText("p95");
  await expect(page.getByTestId("eval-perf")).toContainText("$0.099");
  await expect(page.getByTestId("eval-model")).toContainText("low");
  await expect(page.getByTestId("stat-generated")).toContainText("UTC");
  await expect(page.getByTestId("stat-generated")).not.toContainText("mission");
  const rows = page.getByTestId("scenario-table").locator("tbody tr");
  await expect(rows).toHaveCount(10);
  await expect(page.getByTestId("scenario-lowlight_C")).toContainText("underwatering");
  await expect(page.getByTestId("scenario-lowlight_C").getByTestId("chip-fail")).toBeVisible();
  await expect(page.getByTestId("scenario-wilt_B").getByTestId("chip-pass")).toBeVisible();
  expect(m.unexpected).toEqual([]);
});

test("trust scoreboard shows an empty state when no eval run exists (404)", async ({ page }) => {
  const m = await mockApi(page, { evalRun: "404" });
  await page.goto("/trust");
  await expect(page.getByTestId("eval-empty")).toBeVisible();
  await expect(page.getByTestId("scoreboard")).toHaveCount(0);
  await expect(page.getByTestId("area-n")).toContainText("55");
  expect(m.unexpected).toEqual([]);
});

// ---- Mars mode (M8) ---------------------------------------------------------------------------

const MODEL = "test-edge-model-7"; // never "Glimmer": the badge must show whatever the API says, verbatim

test("header: link up shows EARTH, no edge badge, no drain traffic on load", async ({ page }) => {
  const m = await mockApi(page);
  await page.goto("/");
  await expect(page.getByTestId("link-chip")).toHaveText("LINK: UP");
  await expect(page.getByTestId("brain-badge")).toHaveText("EARTH (muse-spark-1.3)");
  await expect(page.getByTestId("link-simulated")).toHaveText("simulated link");
  expect(m.drainConfirms + m.drainPlans).toBe(0);
  expect(m.unexpected).toEqual([]);
});

test("link control: down shows the on-board edge badge with the API's model verbatim; 409 on a repeat", async ({ page }) => {
  const m = await mockApi(page, { link: { state: "up", delay_s: null, edge_model: MODEL, edge_mode: "recorded" } });
  await page.goto("/");
  await page.getByText("Link control", { exact: true }).click();
  await expect(page.getByTestId("light-delay-note")).toContainText("3-22 min");
  await page.getByTestId("link-set-down").click();
  await expect(page.getByTestId("link-chip")).toHaveText("LINK: DOWN");
  await expect(page.getByTestId("brain-badge")).toHaveText(`RECORDED · EDGE (${MODEL})`);
  await page.getByTestId("link-set-down").click(); // same state again: the API answers 409
  await expect(page.getByTestId("link-error")).toContainText("Already down");
  await page.getByTestId("link-delay-input").fill("45");
  await page.getByTestId("link-set-delayed").click();
  await expect(page.getByTestId("link-chip")).toHaveText("LINK: DELAYED (45 s)");
  expect(m.linkPosts).toEqual([{ state: "down" }, { state: "down" }, { state: "delayed", delay_s: 45 }]);
  await expect(page.locator("body")).not.toContainText(/glimmer/i);
  expect(m.unexpected).toEqual([]);
});

test("live (non-recorded) edge mode: remote API wording, never ON-BOARD or RECORDED", async ({ page }) => {
  await mockApi(page, { link: { state: "down", delay_s: null, edge_model: MODEL, edge_mode: "spark_contributor" } });
  await page.goto("/");
  await expect(page.getByTestId("brain-badge")).toHaveText(`EDGE (${MODEL}, remote API: stands in for an on-board model)`);
  await expect(page.getByTestId("brain-badge")).not.toContainText("RECORDED");
  await expect(page.getByTestId("brain-badge")).not.toContainText("ON-BOARD");
});

test("mars: edge card with checks and model; drain needs a plan, then a click; nothing paid on load", async ({ page }) => {
  const m = await mockApi(page, {
    link: { state: "delayed", delay_s: 30, edge_model: MODEL, edge_mode: "recorded" },
    queue: [queueItem({
      ...EDGE_ONLY, provenance: { recorded: true, model: MODEL, provider: "meta", recorded_as_of: "2016-01-07T12:00:00Z" },
      // a triage still ungrounded after its one repair call is stored invalid (backend/link/edge.py)
      edge_valid: false, grounding: [{ field: "crew_instruction", number: "0.45", context: "VPD 0.45 kPa" }],
    })],
  });
  await page.goto("/mars");
  const card = page.getByTestId("edge-card");
  await expect(card).toBeVisible();
  await expect(page.getByTestId("queue-status")).toHaveText("edge_done");
  await expect(card.getByTestId("edge-model")).toContainText(MODEL);
  await expect(card.getByTestId("edge-recorded")).toHaveText(`RECORDED · EDGE (${MODEL})`);
  await expect(card.getByTestId("edge-urgency")).toHaveText("now");
  await expect(card.getByTestId("edge-cause")).toHaveText("underwatering");
  await expect(card.getByTestId("edge-instruction")).toContainText("Check substrate moisture");
  await expect(card.getByTestId("edge-handoff")).toContainText("B shrinking vs control A");
  await expect(card.getByTestId("evidence-chip")).toHaveCount(2);
  await expect(card.getByTestId("edge-citations")).toContainText("ok");
  await expect(card.getByTestId("edge-grounding")).toContainText("UNVERIFIED"); // this item carries one ungrounded number
  await expect(card.getByTestId("ungrounded")).toContainText("0.45");
  await expect(page.getByTestId("perception-note")).toContainText("frames segmented before the link dropped");
  await expect(page.getByTestId("audit-card")).toHaveCount(0);
  await expect(page.locator("body")).not.toContainText(/glimmer/i);
  // page load made no drain request at all
  expect(m.drainPlans + m.drainConfirms).toBe(0);
  // the plan is free; nothing is confirmed by it
  await page.getByTestId("drain-plan").click();
  await expect(page.getByTestId("drain-plan-summary")).toContainText("1 item");
  await expect(page.getByTestId("drain-plan-summary")).toContainText("$0.10-$0.20");
  expect(m.drainPlans).toBe(1);
  expect(m.drainConfirms).toBe(0);
  await page.getByTestId("drain-confirm").dblclick();
  await expect(page.getByTestId("audit-card")).toBeVisible();
  expect(m.drainConfirms).toBe(1);
  await expect(page.getByTestId("queue-status")).toHaveText("audited");
  await expect(page.getByTestId("audit-label")).toHaveText("computed by code, not by a model");
  await expect(page.getByTestId("audit-verdict")).toHaveText("agree");
  await expect(page.getByTestId("audit-cause")).toContainText("underwatering");
  await expect(page.getByTestId("audit-urgency")).toContainText("alert");
  await expect(page.getByTestId("audit-delta")).toContainText("0 (equal)");
  await expect(page.getByTestId("audit-notes")).toContainText("Cause matches; urgency equal.");
  expect(m.unexpected).toEqual([]);
});

test("mars: an item whose edge run could not happen shows the error verbatim and can still be planned", async ({ page }) => {
  const m = await mockApi(page, {
    link: { state: "delayed", delay_s: 30, edge_model: null, edge_mode: "recorded" },
    queue: [queueItem({ ...EDGE_ONLY, status: "queued", edge_json: null, provenance: null, edge_valid: null, grounding: [], error: "no recorded edge run for this alert" })],
  });
  await page.goto("/mars");
  await expect(page.getByTestId("queue-error")).toHaveText("no recorded edge run for this alert");
  await expect(page.getByTestId("edge-card")).toHaveCount(0);
  await expect(page.getByTestId("brain-badge")).toContainText("RECORDED · EDGE (no model)");
  await page.getByTestId("drain-plan").click();
  await expect(page.getByTestId("drain-plan-summary")).toContainText("1 item");
  expect(m.drainConfirms).toBe(0);
});

test("mars: an invalid edge triage is edge_invalid and its audit says it is not meaningful", async ({ page }) => {
  await mockApi(page, {
    link: { state: "up", delay_s: null, edge_model: MODEL, edge_mode: "recorded" },
    queue: [queueItem({
      ...EDGE_ONLY, status: "audited", edge_valid: false, grounding: null,
      error: "edge triage invalid: grounding: number '0.45' in crew_instruction",
      audit_json: {
        agree: null, urgency_delta: null, cause_match: null, edge_cause: "underwatering", earth_top_cause: "underwatering",
        edge_urgency: "now", earth_status: "alert", notes: "edge triage invalid: grounding: number '0.45'; audit not meaningful",
      },
    })],
  });
  await page.goto("/mars");
  await expect(page.getByTestId("audit-verdict")).toHaveText("audit not meaningful (edge/Earth answer failed validation)");
  await expect(page.getByTestId("edge-grounding")).toContainText("not evaluated");
  await expect(page.getByTestId("queue-error")).toContainText("edge triage invalid");
});

test("mars: empty queue, then a failed item with clean grounding", async ({ page }) => {
  const m = await mockApi(page, { link: { state: "down", delay_s: null, edge_model: MODEL, edge_mode: "recorded" } });
  await page.goto("/mars");
  await expect(page.getByTestId("queue-empty")).toBeVisible();
  m.queue = [queueItem({ ...EDGE_ONLY, grounding: [], status: "failed" })];
  await page.getByText("Link control", { exact: true }).click();
  await page.getByTestId("link-set-delayed").click(); // refreshes the queue
  await expect(page.getByTestId("queue-status")).toHaveText("failed");
  await expect(page.getByTestId("edge-grounding")).toHaveText("numbers checked against tool outputs: ok");
});

test("drain: 402 (budget) and 409 are explained and the queue is untouched", async ({ page }) => {
  const m = await mockApi(page, {
    link: { state: "delayed", delay_s: 30, edge_model: MODEL, edge_mode: "recorded" },
    queue: [queueItem(EDGE_ONLY)], drainStatus: 402,
  });
  await page.goto("/mars");
  await page.getByTestId("drain-plan").click();
  await page.getByTestId("drain-confirm").click();
  await expect(page.getByTestId("drain-error")).toContainText("Budget guard");
  await expect(page.getByTestId("drain-error")).toContainText("would exceed the budget guard");
  await expect(page.getByTestId("queue-status")).toHaveText("edge_done");
  expect(m.drainConfirms).toBe(1);
  m.drainStatus = 409;
  await page.getByTestId("drain-confirm").click();
  await expect(page.getByTestId("drain-error")).toContainText("Cannot drain: a drain is already running");
  expect(m.drainConfirms).toBe(2); // one per click, never automatic
});

test("drain plan on a down link: 409 is shown, no confirm button", async ({ page }) => {
  const m = await mockApi(page, { link: { state: "down", delay_s: null, edge_model: MODEL, edge_mode: "recorded" }, queue: [queueItem(EDGE_ONLY)] });
  await page.goto("/mars");
  await page.getByTestId("drain-plan").click();
  await expect(page.getByTestId("drain-error")).toContainText("Cannot drain: the link is down");
  await expect(page.getByTestId("drain-confirm")).toHaveCount(0);
  expect(m.drainConfirms).toBe(0);
});

test("queueing an open alert from the alert feed: POST /link/queue, 409 on a repeat; hidden while the link is up", async ({ page }) => {
  const m = await mockApi(page, { link: { state: "up", delay_s: null, edge_model: MODEL, edge_mode: "recorded" } });
  await page.goto("/");
  await expect(page.getByTestId(`alert-${ALERT}`)).toBeVisible();
  await expect(page.getByTestId(`queue-alert-${ALERT}`)).toHaveCount(0);
  await page.getByText("Link control", { exact: true }).click();
  await page.getByTestId("link-set-down").click();
  await page.keyboard.press("Escape"); // the popover closes
  await expect(page.getByTestId("link-delay-input")).toBeHidden();
  await page.getByTestId(`queue-alert-${ALERT}`).click();
  await expect(page.getByTestId("queue-result")).toContainText("Queued (edge_done)");
  expect(m.queuePosts).toEqual([ALERT]);
  expect(m.drainConfirms + m.drainPlans).toBe(0);
  await page.goto("/mars");
  await expect(page.getByTestId("edge-card")).toBeVisible();
  await page.getByTestId(`queue-alert-${ALERT}`).click(); // already queued: the API answers 409
  await expect(page.getByTestId("queue-result")).toContainText("already queued");
  expect(m.unexpected).toEqual([]);
});

test("agent panel: ungrounded numbers show as an amber note, not an error; none when the list is empty", async ({ page }) => {
  const m = await mockApi(page, { assessmentId: AID });
  await page.goto("/");
  await page.getByTestId(`alert-${ALERT}`).click();
  const note = page.getByTestId("warnings-note");
  await expect(note).toBeVisible();
  await expect(note).toContainText("Numbers not found in tool outputs");
  await expect(note).toContainText("3.77");
  await expect(note).toContainText("crew_brief");
  await expect(note).toContainText("not an error");
  await expect(page.getByText("UNVERIFIED", { exact: true })).toHaveCount(0);
  m.warnings = [];
  await page.getByRole("button", { name: "Close agent panel" }).click();
  await page.getByTestId(`alert-${ALERT}`).click();
  await expect(page.getByTestId("assessment")).toBeVisible();
  await expect(page.getByTestId("warnings-note")).toHaveCount(0);
});

test("trust: edge triage eval leads with deterministic metrics, fractions with n, model and small-n; no percentages", async ({ page }) => {
  const m = await mockApi(page);
  await page.goto("/trust");
  const block = page.getByTestId("edge-eval");
  await expect(block).toBeVisible();
  const order = await block.evaluate((el) => {
    const q = (id: string) => Array.from(el.querySelectorAll("*")).indexOf(el.querySelector(`[data-testid="${id}"]`) as Element);
    return [q("edge-det-metrics"), q("edge-top1-table"), q("edge-eval-model")];
  });
  expect(order.every((n) => n >= 0)).toBe(true);
  expect([...order].sort((a, b) => a - b)).toEqual(order);
  await expect(page.getByTestId("edge-det-citation")).toContainText("6/6");
  await expect(page.getByTestId("edge-det-grounding")).toContainText("6/6");
  await expect(page.getByTestId("edge-det-injection")).toContainText("1/1");
  await expect(page.getByTestId("edge-det-urgency")).toContainText("6/6");
  await expect(page.getByTestId("edge-top1-table")).toContainText("1/1");
  await expect(page.getByTestId("edge-top1-table")).toContainText("5/5");
  await expect(block).not.toContainText("%");
  await expect(page.getByTestId("edge-small-n")).toBeVisible();
  await expect(page.getByTestId("edge-eval-model")).toContainText("fake-scripted");
  await expect(page.getByTestId("edge-fake-run")).toBeVisible();
  // the agronomist scoreboard gained the numeric-grounding entry
  await expect(page.getByTestId("det-grounding")).toContainText("9/9");
  expect(m.unexpected).toEqual([]);
});

test("trust: edge eval empty state (404) leaves the rest of the page alone", async ({ page }) => {
  const m = await mockApi(page, { edgeEval: "404" });
  await page.goto("/trust");
  await expect(page.getByTestId("edge-eval-empty")).toBeVisible();
  await expect(page.getByTestId("edge-eval")).toHaveCount(0);
  await expect(page.getByTestId("scoreboard")).toBeVisible();
  await expect(page.getByTestId("area-n")).toContainText("55");
  expect(m.unexpected).toEqual([]);
});
