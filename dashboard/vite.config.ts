import { loadEnv } from "vite";
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig(({ command, mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  // FastAPI (python run_api.py) — the dev server proxies /api there so the
  // browser stays same-origin (no CORS, no hard-coded host in the bundle).
  const apiTarget = env.VITE_API_PROXY_TARGET || "http://127.0.0.1:8000";
  return {
    plugins: [react()],
    // Production build is served by FastAPI at /dashboard/ (src/api/main.py).
    base: command === "build" ? "/dashboard/" : "/",
    server: {
      port: 5173,
      open: true,
      proxy: {
        "/api": { target: apiTarget, changeOrigin: true },
        "/health": { target: apiTarget, changeOrigin: true },
      },
    },
    test: {
      environment: "node",
      include: ["src/**/*.test.ts"],
    },
  };
});
