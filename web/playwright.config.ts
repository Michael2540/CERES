import { defineConfig } from "@playwright/test";

// Two production builds, each in its own dist dir so the normal `.next` build is untouched:
//  - :3100 NEXT_PUBLIC_FIXTURES=1, no backend (smoke.spec.ts)
//  - :3101 real-API mode with every request mocked through page.route (mock.spec.ts; never reaches a backend)
const build = (port: number) => `npx next build --turbopack && npx next start -p ${port}`;

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  webServer: [
    {
      command: `node scripts/sync-fixtures.mjs && ${build(3100)}`,
      url: "http://localhost:3100",
      reuseExistingServer: false,
      timeout: 300_000,
      env: { NEXT_PUBLIC_FIXTURES: "1", NEXT_DIST_DIR: ".next-fixtures" },
    },
    {
      command: build(3101),
      url: "http://localhost:3101",
      reuseExistingServer: false,
      timeout: 300_000,
      env: { NEXT_PUBLIC_FIXTURES: "0", NEXT_DIST_DIR: ".next-mock", NEXT_PUBLIC_API_URL: "http://localhost:8000" },
    },
  ],
  projects: [
    { name: "fixtures", testMatch: /smoke\.spec\.ts/, use: { baseURL: "http://localhost:3100" } },
    { name: "mock", testMatch: /mock\.spec\.ts/, use: { baseURL: "http://localhost:3101" } },
  ],
});
