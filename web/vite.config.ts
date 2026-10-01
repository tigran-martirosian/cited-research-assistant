import { execSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/** The git commit the frontend is built from, compared with the running backend's. */
function buildCommit(): string {
  try {
    return execSync("git rev-parse --short HEAD", { encoding: "utf8" }).trim() || "unknown";
  } catch {
    return "unknown";
  }
}

/** The app version: the one line of VERSION at the repo root (the backend reads it too). */
function appVersion(): string {
  try {
    return readFileSync("../VERSION", "utf8").trim() || "unknown";
  } catch {
    return "unknown";
  }
}

// In development the API runs separately (python serve.py); Vite proxies /api to it
// (CRA_API_URL, default http://127.0.0.1:8765).
export default defineConfig({
  plugins: [react()],
  define: {
    __APP_COMMIT__: JSON.stringify(buildCommit()),
    __APP_VERSION__: JSON.stringify(appVersion()),
  },
  server: {
    proxy: { "/api": process.env.CRA_API_URL || "http://127.0.0.1:8765" },
  },
});
