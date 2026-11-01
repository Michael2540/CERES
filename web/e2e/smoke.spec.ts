import { expect, test, type Page } from "@playwright/test";

function watchConsole(page: Page) {
  const errors: string[] = [];
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  page.on("pageerror", (e) => errors.push(String(e)));
  // The agent must never be triggered by a plain page load.
  page.on("request", (r) => {
    if (r.method() === "POST" && r.url().includes("/link/drain")) errors.push(`unexpected drain request ${r.url()}`);
    if (new RegExp("/alerts/[^/]+/assess").test(r.url())) errors.push(`unexpected assess request ${r.url()}`);
  });
  return errors;
}

test("overview shows header, four tray cards, provenance and alert feed", async ({ page }) => {
  const errors = watchConsole(page);
  await page.goto("/");
  await expect(page.getByTestId("clock-text")).toContainText("Mission day 21");
  await expect(page.getByTestId("spend-meter")).toContainText("Meta spend $4.37 · runtime guard $100");
  await expect(page.getByTestId("link-chip")).toHaveText("LINK: DELAYED (30 s)");
  await expect(page.getByTestId("link-simulated")).toHaveText("simulated link");
  await expect(page.getByTestId("brain-badge")).toHaveText("RECORDED · EDGE (muse-spark-1.3-contributor)");
  await expect(page.getByText("ALED plant images: real frames. Trays B and C carry labelled simulated faults")).toBeVisible();
  for (const id of ["A", "B", "C", "L"]) await expect(page.getByTestId(`tray-card-${id}`)).toBeVisible();
  await expect(page.getByTestId("prov-A")).toHaveText("DATASET");
  await expect(page.getByTestId("prov-B")).toHaveText("SIMULATED FAULT · image transform");
  await expect(page.getByTestId("prov-C")).toHaveText("SIMULATED FAULT · time-warp");
  await expect(page.getByText("SENSORS: SIMULATED").first()).toBeVisible();
  await expect(page.getByTestId("open-alerts-B")).toHaveText("1 open alert");
  await expect(page.getByTestId("alert-list").getByRole("button")).toHaveCount(1);
  await expect(page.getByTestId("alert-list")).toContainText("alert · open");
  await page.getByRole("button", { name: "Toggle mask overlay for tray B" }).click();
  await expect(page.getByAltText("Tray B mask overlay")).toBeVisible();
  await page.getByText("Legend").click();
  await expect(page.getByTestId("legend")).toContainText("DATASET");
  expect(errors).toEqual([]);
});

test("scrubbing back in time closes the panel and renders no assessment", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("alert-alert_B_GROWTH_LAG_20160103T100000Z").click();
  const panel = page.getByTestId("agent-panel");
  await panel.getByTestId("investigate").click();
  await panel.getByTestId("confirm-run").click();
  await expect(panel.getByTestId("assessment")).toBeVisible({ timeout: 15_000 });
  await page.getByLabel("Mission day scrubber").fill("10");
  await expect(page.getByTestId("clock-text")).toContainText("2015-12-23");
  await expect(page.getByTestId("agent-panel")).toHaveCount(0);
  await expect(page.getByTestId("assessment")).toHaveCount(0);
});

test("Escape closes the panel and restores focus to the opener", async ({ page }) => {
  await page.goto("/");
  const opener = page.getByTestId("alert-alert_B_GROWTH_LAG_20160103T100000Z");
  await opener.click();
  await expect(page.getByRole("heading", { name: "Flight Agronomist" })).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(page.getByTestId("agent-panel")).toHaveCount(0);
  await expect(opener).toBeFocused();
});

test("tray B shows scrubber, charts, shared-air note and timeline", async ({ page }) => {
  const errors = watchConsole(page);
  await page.goto("/tray/B");
  await expect(page.getByRole("heading", { name: "Tray B" })).toBeVisible();
  await expect(page.getByTestId("frame-source")).toBeVisible();
  await expect(page.getByTestId("frame-ts")).toContainText("frame");
  await page.getByRole("button", { name: "Previous frame" }).click();
  await page.getByRole("button", { name: "Mask overlay" }).click();
  await expect(page.getByTestId("chart-area_cm2").locator("svg").first()).toBeVisible();
  await expect(page.getByTestId("chart-soil_vwc").locator("svg").first()).toBeVisible();
  await expect(page.getByTestId("sim-badge-soil_vwc")).toBeVisible();
  await expect(page.getByTestId("sim-badge-area_cm2")).toHaveCount(0);
  await expect(page.getByTestId("alert-timeline").getByRole("button")).toHaveCount(2); // own + shared sensor alert
  expect(errors).toEqual([]);
});

test("trust page", async ({ page }) => {
  const errors = watchConsole(page);
  await page.goto("/trust");
  await expect(page.getByTestId("scoreboard")).toBeVisible();
  await expect(page.getByTestId("det-citation")).toContainText("9/9");
  await expect(page.getByTestId("top1-table")).toContainText("2/3");
  await expect(page.getByTestId("top1-table")).not.toContainText("%");
  await expect(page.getByTestId("small-n")).toBeVisible();
  await expect(page.getByTestId("area-empty")).toHaveCount(0);
  await expect(page.getByTestId("area-n")).toContainText("55");
  await expect(page.getByTestId("area-dice")).toContainText("0.94");
  await expect(page.getByTestId("area-mape")).toContainText("5%");
  await expect(page.getByTestId("area-row-hsv")).toContainText("colour-threshold baseline, not a model");
  await expect(page.getByTestId("area-trays")).toContainText("A n=43");
  await expect(page.getByTestId("area-partial")).toHaveCount(0);
  await expect(page.getByTestId("area-reading")).toContainText("advantage is instances");
  await expect(page.getByTestId("area-row-sam-raw")).toContainText("0.88");
  await expect(page.getByTestId("area-caption")).toContainText("Bell & Dee 2016");
  await expect(page.getByText(/lettuce/i)).toHaveCount(0);
  await expect(page.getByText("the measurement is real; the input fault is simulated.")).toBeVisible();
  await expect(page.getByTestId("blind-note")).toContainText("deliberately not told which inputs are simulated");
  await expect(page.getByText("Built with Muse Spark 1.3, SAM 3.1 (Meta Model API)")).toBeVisible();
  await expect(page.getByTestId("footer-built")).toHaveText("Built with Muse Spark 1.3, SAM 3.1 (Meta Model API) · plant images: ALED, CC BY 4.0");
  await expect(page.getByTestId("eval-recorded")).toContainText(/(RECORDED|SCRIPTED) RUN/);
  await expect(page.getByTestId("edge-eval-recorded")).toContainText(/(RECORDED|SCRIPTED) RUN/);
  await expect(page.locator("body")).not.toContainText(/glimmer|dinov3/i);
  expect(errors).toEqual([]);
});

test("agent panel replays the fixture stream to the assessment", async ({ page }) => {
  const errors = watchConsole(page);
  await page.goto("/");
  await page.getByTestId("alert-alert_B_GROWTH_LAG_20160103T100000Z").click();
  const panel = page.getByTestId("agent-panel");
  await expect(panel).toBeVisible();
  await panel.getByTestId("investigate").click();
  await expect(panel.getByTestId("confirm-box")).toContainText("~$0.10");
  await panel.getByTestId("confirm-run").dblclick();
  expect(await page.evaluate(() => (window as unknown as { __ceresAssessPosts?: number }).__ceresAssessPosts)).toBe(1);
  await expect(panel.getByTestId("tool-call").first()).toBeVisible();
  await expect(panel.getByTestId("tool-result").first()).toContainText("id");
  await expect(panel.getByTestId("assessment")).toBeVisible({ timeout: 15_000 });
  await expect(panel.getByTestId("tool-call")).toHaveCount(5);
  await expect(panel.getByTestId("live-badge")).toHaveText("LIVE");
  await expect(panel.getByTestId("assessment")).toContainText("41.2 s");
  await expect(panel.getByTestId("cause").first()).toContainText("underwatering");
  await expect(panel.getByTestId("crew-brief")).toContainText("Water tray B");
  await expect(panel.getByTestId("caveats")).toBeVisible();
  // Approve the proposed action.
  await panel.getByRole("button", { name: "Approve" }).click();
  await expect(panel.getByTestId("action-status")).toHaveText("approved");
  // Evidence chip jumps to the tray page and shows the evidence detail.
  await panel.getByTestId("evidence-chip").first().click();
  await expect(page).toHaveURL(/\/tray\/B$/);
  await expect(page.getByTestId("evidence-detail")).toContainText("0.11");
  await expect(page.getByTestId("chart-soil_vwc").locator(".recharts-reference-dot")).toHaveCount(1);
  expect(errors).toEqual([]);
});

const drainConfirms = (page: Page) =>
  page.evaluate(() => (window as unknown as { __ceresDrainConfirms?: number }).__ceresDrainConfirms ?? 0);

test("mars mode: link down, queue an alert, plan the drain, confirm, then the audit diff", async ({ page }) => {
  const errors = watchConsole(page);
  await page.goto("/mars");
  // The fixture queue already holds one audited item (edge card + audit card), model string verbatim from the fixture.
  const first = page.getByTestId("queue-item-lq_B_WILT_RISK_20160106T100000Z_20160107T120000Z");
  await expect(first.getByTestId("queue-status")).toHaveText("audited");
  await expect(first.getByTestId("edge-model")).toContainText("muse-spark-1.3-contributor (meta)");
  await expect(first.getByTestId("edge-recorded")).toHaveText("RECORDED · EDGE (muse-spark-1.3-contributor)");
  await expect(first.getByTestId("audit-label")).toHaveText("computed by code, not by a model");
  await expect(first.getByTestId("audit-verdict")).toHaveText("agree");
  await expect(first.getByTestId("perception-note")).toContainText("frames segmented before the link dropped");
  await expect(page.locator("body")).not.toContainText(/glimmer/i);
  expect(await drainConfirms(page)).toBe(0);

  // Link down: queue the open alert for on-board triage.
  await page.getByText("Link control", { exact: true }).click();
  await page.getByTestId("link-set-down").click();
  await expect(page.getByTestId("link-chip")).toHaveText("LINK: DOWN");
  await page.getByTestId("link-set-down").click();
  await expect(page.getByTestId("link-error")).toContainText("Already down");
  await page.keyboard.press("Escape");
  await page.getByTestId("queue-alert-alert_B_GROWTH_LAG_20160103T100000Z").click();
  const second = page.getByTestId("queue-item-lq_alert_B_GROWTH_LAG_20160103T100000Z");
  await expect(second.getByTestId("queue-status")).toHaveText("edge_done");
  await expect(second.getByTestId("edge-card")).toBeVisible();
  await expect(second.getByTestId("audit-card")).toHaveCount(0);

  // A down link cannot drain: the plan itself says so.
  await page.getByTestId("drain-plan").click();
  await expect(page.getByTestId("drain-error")).toContainText("Cannot drain");

  // Delayed: plan (free), then an explicit confirm click.
  await page.getByText("Link control", { exact: true }).click();
  await page.getByTestId("link-delay-input").fill("5");
  await page.getByTestId("link-set-delayed").click();
  await expect(page.getByTestId("link-chip")).toHaveText("LINK: DELAYED (5 s)");
  await page.keyboard.press("Escape");
  await page.getByTestId("drain-plan").click();
  await expect(page.getByTestId("drain-plan-summary")).toContainText("1 item");
  await expect(page.getByTestId("drain-plan-summary")).toContainText("$0.10-$0.20");
  expect(await drainConfirms(page)).toBe(0);
  await page.getByTestId("drain-confirm").dblclick();
  await expect(second.getByTestId("audit-card")).toBeVisible();
  await expect(second.getByTestId("queue-status")).toHaveText("audited");
  expect(await drainConfirms(page)).toBe(1);
  expect(errors).toEqual([]);
});

test("trust page shows the edge triage eval block and the numeric-grounding entry", async ({ page }) => {
  const errors = watchConsole(page);
  await page.goto("/trust");
  await expect(page.getByTestId("edge-eval")).toBeVisible();
  await expect(page.getByTestId("edge-det-grounding")).toContainText("6/6");
  await expect(page.getByTestId("edge-top1-table")).toContainText("5/5");
  await expect(page.getByTestId("edge-small-n")).toBeVisible();
  await expect(page.getByTestId("edge-eval-model")).toContainText("fake-scripted");
  await expect(page.getByTestId("det-grounding")).toContainText("9/9");
  for (const id of ["eval-note", "edge-eval-note"]) {
    await expect(page.getByTestId(id)).toContainText("Prompts were revised after earlier runs on these same scenarios");
    await expect(page.getByTestId(id)).toContainText("n = 1 per fault");
    await expect(page.getByTestId(id)).toContainText("9 of 11 scenarios raise an alert; 2 are no-alert checks");
  }
  expect(errors).toEqual([]);
});

test("agent panel shows the ungrounded-number warnings as a note", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("alert-alert_B_GROWTH_LAG_20160103T100000Z").click();
  const panel = page.getByTestId("agent-panel");
  await panel.getByTestId("investigate").click();
  await panel.getByTestId("confirm-run").click();
  await expect(panel.getByTestId("assessment")).toBeVisible({ timeout: 15_000 });
  await expect(panel.getByTestId("warnings-note")).toContainText("3.77");
});
