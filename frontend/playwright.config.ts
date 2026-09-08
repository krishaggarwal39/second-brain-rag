import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright E2E config for the Second Brain RAG frontend.
 *
 * Spins up the Vite dev server and runs real-browser tests against it.
 * Backend calls are intercepted/mocked per-test via page.route() so the E2E
 * flow (auth guard -> login -> upload -> chat streaming -> citations) is
 * verified in a real browser without needing the full backend stack running.
 */
export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: process.env.CI ? [['github'], ['list']] : 'list',
  use: {
    baseURL: 'http://localhost:5173',
    trace: 'on-first-retry',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: {
    command: 'npm run dev -- --port 5173',
    url: 'http://localhost:5173',
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
