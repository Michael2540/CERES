import { defineConfig } from "@playwright/test";

// Smoke tests run against the production build with fixtures, so no backend is needed.
export default defineConfig({
  testDir: "./e2e",
  webServer: {
    command: "npm run start -- -p 3100",
    url: "http://localhost:3100",
    reuseExistingServer: true,
    env: { NEXT_PUBLIC_FIXTURES: "1" },
  },
  use: { baseURL: "http://localhost:3100" },
});
