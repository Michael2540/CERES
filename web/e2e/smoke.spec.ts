import { expect, test, type Page } from "@playwright/test";

function watchConsole(page: Page) {
  const errors: string[] = [];
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  page.on("pageerror", (e) => errors.push(String(e)));
  // The agent must never be triggered by a plain page load.
  page.on("request", (r) => {
    if (new RegExp("/alerts/[^/]+/assess").test(r.url())) errors.push(`unexpected assess request ${r.url()}`);
  });
  return errors;
}

test("overview shows header, four tray cards, provenance and alert feed", async ({ page }) => {
  const errors = watchConsole(page);
  await page.goto("/");
  await expect(page.getByTestId("clock-text")).toContainText("Mission day 21");
  await expect(page.getByTestId("spend-meter")).toContainText("Meta spend $4.37 · runtime guard $100");
  await expect(page.getByTestId("link-chip")).toHaveText("LINK: UP (simulated)");
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
  await expect(page.getByTestId("top1-table")).toContainText("2/3");
  await expect(page.getByTestId("top1-table")).not.toContainText("%");
  await expect(page.getByTestId("small-n")).toBeVisible();
  await expect(page.getByTestId("area-empty")).toHaveCount(0);
  await expect(page.getByTestId("area-n")).toContainText("55");
  await expect(page.getByTestId("area-dice")).toContainText("0.88");
  await expect(page.getByTestId("area-mape")).toContainText("29%");
  await expect(page.getByTestId("area-row-hsv")).toContainText("colour-threshold baseline, not a model");
  await expect(page.getByTestId("area-trays")).toContainText("A n=43");
  await expect(page.getByTestId("area-partial")).toHaveCount(0);
  await expect(page.getByTestId("area-reading")).toContainText("SAM's advantage is instances");
  await expect(page.getByTestId("area-caption")).toContainText("Bell & Dee 2016");
  await expect(page.getByText(/lettuce/i)).toHaveCount(0);
  await expect(page.getByText("the measurement is real; the input fault is simulated.")).toBeVisible();
  await expect(page.getByTestId("blind-note")).toContainText("deliberately not told which inputs are simulated");
  await expect(page.getByText("Built with Muse Spark · SAM 3.1")).toBeVisible();
  await expect(page.getByText("Images: Aberystwyth Leaf Evaluation Dataset (CC BY 4.0)")).toBeVisible();
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
