import type { BackendVersion, Connection, ConnectionKey, HistoryItem, ProgressEvent, ResearchItem } from "./types";

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) message = String(body.detail);
    } catch {
      /* not JSON */
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export const api = {
  history: () => call<HistoryItem[]>("/api/history"),
  /** The app version and the commit the backend started with. */
  version: () => call<BackendVersion>("/api/version"),
  get: (id: string) => call<ResearchItem>(`/api/research/${id}`),
  /** previousId: the answered question this one follows up (same conversation). fresh (the
   * "Research again" button): no reuse of any earlier result, ask or fact memory. */
  ask: (question: string, previousId?: string | null, fresh = false) =>
    call<ResearchItem>("/api/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, previous_id: previousId ?? null, fresh }),
    }),
  cancel: (id: string) => call<{ ok: boolean }>(`/api/research/${id}/cancel`, { method: "POST" }),
  remove: (id: string) => call<{ ok: boolean }>(`/api/research/${id}`, { method: "DELETE" }),
  connections: () => call<Connection[]>("/api/connections"),
  connection: (key: ConnectionKey, action: "check" | "connect" | "cancel" | "disconnect") =>
    call<Connection>(`/api/connections/${key}/${action}`, { method: "POST" }),
  loginCode: (key: ConnectionKey, code: string) =>
    call<Connection>(`/api/connections/${key}/code`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code }),
    }),
};

/** Stream a question's progress: every event so far, then live ones, ending with "end". */
export function streamEvents(
  id: string,
  onOpen: () => void,
  onEvent: (event: ProgressEvent) => void,
): () => void {
  const source = new EventSource(`/api/research/${id}/events`);
  // Every connection (including EventSource's automatic reconnects) replays from the start.
  source.onopen = onOpen;
  source.onmessage = (message) => {
    const event = JSON.parse(message.data) as ProgressEvent;
    onEvent(event);
    if (event.type === "end") source.close();
  };
  return () => source.close();
}
