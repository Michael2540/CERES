import { readFileSync } from "node:fs";
import path from "node:path";
import { expect, test, type Page, type Route } from "@playwright/test";

// Real-API mode (no fixtures build) with every backend request answered by page.route.
// Nothing here can reach a real backend: unmatched requests to :8000 are aborted and counted as failures.

const FIX = path.join(__dirname, "..", "..", "tests", "fixtures", "api");
const fx = (n: string) => JSON.parse(readFileSync(path.join(FIX, n), "utf8"));
const ALERT = "alert_B_GROWTH_LAG_20160103T100000Z";
const AID = "asmt_B_20160103T100000Z";
const CORS = { "access-control-allow-origin": "*", "access-control-allow-headers": "*", "access-control-allow-methods": "*" };

type Mode = {
  asOf: string;
  assessmentId: string | null; // set on the alert in GET /alerts
  assessment: "done" | "running" | "failed" | "404";
  stream: "events" | "abort";
  area: "ok" | "404" | "partial";
  evalRun: "ok" | "404";
  assessPosts: number;
  clockPosts: number;
  unexpected: string[];
};

async function mockApi(page: Page, init: Partial<Mode> = {}) {
  const m: Mode = {
    asOf: "2016-01-03T12:00:00Z", assessmentId: null, assessment: "done", stream: "events",
    area: "ok", evalRun: "ok", assessPosts: 0, clockPosts: 0, unexpected: [], ...init,
  };
  const json = (route: Route, body: unknown, status = 200) =>
    route.fulfill({ status, contentType: "application/json", headers: CORS, body: JSON.stringify(body) });
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
      return json(route, a);
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
