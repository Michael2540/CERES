import { expect, test } from "@playwright/test";

// Requires `npm run build` with NEXT_PUBLIC_FIXTURES=1 (see `npm run build:fixtures`).
test("home shows health from fixtures", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "CERES" })).toBeVisible();
  await expect(page.getByTestId("health-status")).toContainText("status=ok");
});
