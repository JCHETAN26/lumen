import { defineConfig } from "@playwright/test";

// End to end: the built page served by the real local server (lumen.app.server) with its fake
// answerer, in Chromium. `npm run build` first.
const PORT = 8799;

export default defineConfig({
  testDir: "e2e",
  timeout: 30_000,
  use: { baseURL: `http://localhost:${PORT}` },
  projects: [
    {
      name: "chromium",
      use: {
        browserName: "chromium",
        // A synthetic camera and auto-accepted permission, to test the camera flow.
        launchOptions: {
          args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"],
        },
        permissions: ["camera"],
      },
    },
  ],
  webServer: {
    command: `cd ../.. && uv run --group app python -m lumen.app.server --fake --port ${PORT}`,
    url: `http://localhost:${PORT}/`,
    reuseExistingServer: false,
    timeout: 120_000,
  },
});
