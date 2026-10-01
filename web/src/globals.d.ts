/** The git short commit the frontend was built from (vite.config.ts `define`). */
declare const __APP_COMMIT__: string;
/** The app version from VERSION at the repo root (vite.config.ts `define`). */
declare const __APP_VERSION__: string;

// vite.config.ts runs under Node; the project has no @types/node, so what it uses is
// declared here.
declare const process: { env: Record<string, string | undefined> };
declare module "node:child_process" {
  export function execSync(command: string, options: { encoding: "utf8" }): string;
}
declare module "node:fs" {
  export function readFileSync(path: string, encoding: "utf8"): string;
}
