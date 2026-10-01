import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import type { Connection, ConnectionKey } from "../types";
import { XIcon, EyeIcon, EyeOffIcon } from "./Icons";

const ABOUT: Record<ConnectionKey, string> = {
  notebooklm: "Searches the source library notebook with your Google account.",
  claude: "Plans, selects evidence and reasons with your Claude Code subscription.",
  gemini: "Gemini through the Antigravity CLI, signed in with your Google account.",
};

type Action = "check" | "connect" | "cancel" | "disconnect";

const BUSY = new Set<Connection["state"]>(["unknown", "checking", "connecting"]);

/** Live connection states, polled while a check or login is in progress. */
export function useConnections() {
  const [connections, setConnections] = useState<Connection[] | null>(null);

  const refresh = useCallback(
    () =>
      api
        .connections()
        .then(setConnections)
        .catch(() => {}),
    [],
  );

  useEffect(() => {
    refresh();
    window.addEventListener("focus", refresh);
    return () => window.removeEventListener("focus", refresh);
  }, [refresh]);

  const busy = connections?.some((c) => BUSY.has(c.state)) ?? false;
  useEffect(() => {
    if (!busy) return;
    const timer = setInterval(refresh, 1500);
    return () => clearInterval(timer);
  }, [busy, refresh]);

  const act = useCallback(
    async (key: ConnectionKey, action: Action) => {
      // Show the new state at once; a check blocks until it finishes.
      setConnections((prev) =>
        prev?.map((c) =>
          c.key === key && (action === "check" || action === "connect")
            ? { ...c, state: action === "check" ? "checking" : "connecting", detail: null }
            : c,
        ) ?? null,
      );
      await api.connection(key, action).catch(() => {});
      await refresh();
    },
    [refresh],
  );

  // Only required providers block research; an optional one (Gemini) never does.
  const missing = connections?.filter((c) => c.required !== false && c.state !== "connected") ?? [];
  const ready = connections !== null && missing.length === 0;
  return { connections, refresh, act, missing, ready };
}

type Act = ReturnType<typeof useConnections>["act"];

function StatusBadge({ state }: { state: Connection["state"] }) {
  const label =
    state === "connected"
      ? "Connected"
      : state === "connecting"
        ? "Waiting for login"
        : state === "checking" || state === "unknown"
          ? "Checking…"
          : state === "reauth_required"
            ? "Sign-in expired"
            : state === "error"
              ? "Provider error"
              : "Not connected";
  return (
    <span className={`conn-status ${state}`}>
      <span className="conn-dot" aria-hidden="true" />
      {label}
    </span>
  );
}

function LoginCode({ connection }: { connection: Connection }) {
  const [code, setCode] = useState("");
  const [error, setError] = useState<string | null>(null);
  return (
    <form
      className="conn-code"
      onSubmit={(e) => {
        e.preventDefault();
        if (!code.trim()) return;
        api
          .loginCode(connection.key, code)
          .then(() => setCode(""))
          .catch((err: Error) => setError(err.message));
      }}
    >
      <input
        value={code}
        onChange={(e) => setCode(e.target.value)}
        placeholder="Paste the code shown after signing in"
        aria-label="Login code"
      />
      <button type="submit" className="ghost-button" disabled={!code.trim()}>
        Submit
      </button>
      {error && <p className="conn-error">{error}</p>}
    </form>
  );
}

/** m••••••••@gmail.com: first character and domain only. */
function mask(account: string): string {
  const at = account.indexOf("@");
  const name = at > 0 ? account.slice(0, at) : account;
  return `${name.slice(0, 1)}${"•".repeat(Math.max(4, Math.min(name.length - 1, 10)))}${at > 0 ? account.slice(at) : ""}`;
}

/** Signed-in account, masked until explicitly revealed. The reveal lives only in component
 * state, so reopening Connections or reloading the page masks it again. */
function MaskedAccount({ account }: { account: string }) {
  const [shown, setShown] = useState(false);
  return (
    <span className="masked-account">
      Signed in as <span className="account-value">{shown ? account : mask(account)}</span>
      <button
        type="button"
        className="reveal-button"
        onClick={() => setShown(!shown)}
        aria-label={shown ? "Hide email" : "Show email"}
        title={shown ? "Hide email" : "Show email"}
        aria-pressed={shown}
      >
        {shown ? <EyeOffIcon /> : <EyeIcon />}
      </button>
    </span>
  );
}

function ConnectionCard({ connection, act }: { connection: Connection; act: Act }) {
  const { key, name, state, account, detail, login_url } = connection;
  const busy = BUSY.has(state);
  const optional = connection.required === false;

  return (
    <div className={`conn-card ${state}`}>
      <div className="conn-head">
        <div className="conn-title">
          <span className="conn-name">{name}</span>
          {optional && <span className="conn-optional">Optional</span>}
          <StatusBadge state={state} />
        </div>
        <p className="conn-about">
          {state === "connected" && account ? <MaskedAccount account={account} /> : (connection.about ?? ABOUT[key])}
        </p>
      </div>

      {state === "connecting" && (
        <div className="conn-login">
          <p>
            <span className="spinner" aria-hidden="true" />{" "}
            {connection.console_login
              ? `A ${name} sign-in window opened. Choose Sign in with Google there and finish in your browser; the window closes by itself once you are signed in.`
              : "Finish signing in in the browser window that opened. This updates automatically."}
          </p>
          {login_url && (
            <>
              <p className="muted">
                No browser window?{" "}
                <a href={login_url} target="_blank" rel="noreferrer">
                  Open the login page
                </a>
                . If it shows a code, paste it here:
              </p>
              <LoginCode connection={connection} />
            </>
          )}
        </div>
      )}

      {detail && state !== "connected" && state !== "connecting" && <p className="conn-error">{detail}</p>}

      <div className="conn-actions">
        {state === "connecting" ? (
          <button className="ghost-button" onClick={() => act(key, "cancel")}>
            Cancel
          </button>
        ) : (
          <>
            {state === "connected" ? (
              <button className="ghost-button" onClick={() => act(key, "connect")} disabled={busy}>
                Reconnect
              </button>
            ) : state === "reauth_required" ? (
              <button className="primary-button" onClick={() => act(key, "connect")} disabled={busy}>
                Sign in again
              </button>
            ) : (
              <button className="primary-button" onClick={() => act(key, "connect")} disabled={busy}>
                Connect {name}
              </button>
            )}
            {!connection.disabled && (
              <button className="ghost-button small" onClick={() => act(key, "check")} disabled={busy}>
                {state === "checking" || state === "unknown" ? "Checking…" : "Check status"}
              </button>
            )}
            {!connection.disabled && (state === "connected" || state === "reauth_required" || state === "error") && (
              <button className="ghost-button small" onClick={() => act(key, "disconnect")} disabled={busy}>
                Disconnect
              </button>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export function ConnectionList({ connections, act }: { connections: Connection[] | null; act: Act }) {
  if (!connections) return <div className="loading" aria-label="Loading" />;
  return (
    <div className="conn-list">
      {connections.map((c) => (
        <ConnectionCard key={c.key} connection={c} act={act} />
      ))}
    </div>
  );
}

/** Shown instead of the welcome screen while a required connection is missing. */
export function Setup({ connections, act }: { connections: Connection[] | null; act: Act }) {
  return (
    <div className="welcome setup">
      <span className="welcome-mark" aria-hidden="true" />
      <h1>Connect to start researching</h1>
      <p>
        Research needs Claude and NotebookLM; Gemini is optional. Each one opens a browser sign-in; your
        login stays on this computer.
      </p>
      <ConnectionList connections={connections} act={act} />
    </div>
  );
}

/** The Connections dialog, opened from Settings. */
export function ConnectionsDialog({
  open,
  onClose,
  connections,
  act,
}: {
  open: boolean;
  onClose: () => void;
  connections: Connection[] | null;
  act: Act;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div className="modal-scrim" onClick={onClose}>
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="connections-title"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="modal-head">
          <h2 id="connections-title">Connections</h2>
          <button className="icon-button" onClick={onClose} aria-label="Close">
            <XIcon />
          </button>
        </header>
        <ConnectionList connections={connections} act={act} />
      </div>
    </div>
  );
}
