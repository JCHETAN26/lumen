/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The page is served by the local Python server (lumen.app.server) from dist/. In `npm run dev`,
// API calls are proxied to it (start it with --fake for UI work without a model).
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    proxy: { "/api": "http://127.0.0.1:8765" },
  },
  build: { sourcemap: false },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test-setup.ts"],
    include: ["src/**/*.test.tsx", "src/**/*.test.ts"],
  },
});
