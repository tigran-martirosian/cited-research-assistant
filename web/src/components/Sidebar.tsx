import { useMemo, useState } from "react";
import type { BackendVersion, HistoryItem } from "../types";
import { dateTime, historyGroup } from "../format";
import { APP_VERSION, BUILD_COMMIT } from "../version";
import { AlertIcon, ChevronIcon, PlugIcon, PlusIcon, SearchIcon, SidebarIcon, TrashIcon, XIcon } from "./Icons";
import { TextSize } from "./TextSize";
import { ThemeSwitch } from "./ThemeSwitch";

/** A conversation: its first question, the questions that followed it up, and its newest one. */
export interface Thread {
  id: string;
  first: HistoryItem;
  latest: HistoryItem;
  items: HistoryItem[]; // oldest first
}

/** The conversation's short title; its first question until the title is written. */
export function threadTitle(thread: Thread): string {
  return thread.first.title?.trim() || thread.first.question;
}

/** The history as conversations, newest activity first (history arrives newest first). */
export function threadsOf(history: HistoryItem[]): Thread[] {
  const byId = new Map<string, HistoryItem[]>();
  for (const item of history) {
    const key = item.thread_id ?? item.id;
    byId.set(key, [...(byId.get(key) ?? []), item]);
  }
  return [...byId.entries()].map(([id, newestFirst]) => {
    const items = [...newestFirst].sort((a, b) => a.created_at - b.created_at);
    return { id, first: items[0], latest: items[items.length - 1], items };
  });
}

/** Conversations that start with the same question (asked again, researched again), newest
 * first; the sidebar shows them as one row. */
export interface ThreadFamily {
  key: string;
  threads: Thread[]; // newest activity first
}

/** A question's identity for grouping repeats: case, spacing and trailing punctuation ignored. */
function questionKey(q: string): string {
  return q.toLowerCase().replace(/\s+/g, " ").replace(/[\s?.!]+$/, "").trim();
}

/** The threads grouped by their first question, in the order of each family's newest thread. */
export function familiesOf(threads: Thread[]): ThreadFamily[] {
  const byKey = new Map<string, ThreadFamily>();
  const out: ThreadFamily[] = [];
  for (const thread of threads) {
    const key = questionKey(thread.first.question);
    const family = byKey.get(key);
    if (family) family.threads.push(thread);
    else {
      const f = { key, threads: [thread] };
      byKey.set(key, f);
      out.push(f);
    }
  }
  return out;
}

interface Props {
  history: HistoryItem[];
  currentThread: string | null;
  open: boolean;
  onSelect: (id: string | null) => void;
  onDelete: (thread: Thread) => void;
  onClose: () => void;
  /** Hide the sidebar (wide screens; narrow screens use the drawer's close button). */
  onCollapse: () => void;
  onConnections: () => void;
  connectionsReady: boolean;
  /** The running backend's policy version and commit (null until loaded). */
  backendVersion: BackendVersion | null;
}

export function Sidebar({
  history,
  currentThread,
  open,
  onSelect,
  onDelete,
  onClose,
  onCollapse,
  onConnections,
  connectionsReady,
  backendVersion,
}: Props) {
  const [filter, setFilter] = useState("");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const groups = useMemo(() => {
    const needle = filter.trim().toLowerCase();
    const result: { label: string; families: ThreadFamily[] }[] = [];
    const threads = threadsOf(history).filter(
      (thread) =>
        !needle ||
        (thread.first.title ?? "").toLowerCase().includes(needle) ||
        thread.items.some((i) => i.question.toLowerCase().includes(needle)),
    );
    for (const family of familiesOf(threads)) {
      const label = historyGroup(family.threads[0].latest.created_at);
      const last = result[result.length - 1];
      if (last?.label === label) last.families.push(family);
      else result.push({ label, families: [family] });
    }
    return result;
  }, [history, filter]);

  const toggle = (key: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });

  const row = (thread: Thread, extra?: { family: ThreadFamily; open: boolean }) => (
    <div className={`history-item ${thread.id === currentThread ? "active" : ""} ${extra ? "" : "history-run"}`}>
      {extra && extra.family.threads.length === 1 && <span className="history-expand" aria-hidden="true" />}
      {extra && extra.family.threads.length > 1 && (
        <button
          className={`history-expand ${extra.open ? "open" : ""}`}
          onClick={() => toggle(extra.family.key)}
          aria-expanded={extra.open}
          aria-label={extra.open ? "Hide earlier runs" : "Show earlier runs"}
          title={`Asked ${extra.family.threads.length} times`}
        >
          <ChevronIcon />
        </button>
      )}
      <button
        className="history-link"
        onClick={() => onSelect(thread.latest.id)}
        title={thread.items.map((i) => i.question).join(" → ")}
      >
        {thread.latest.status === "running" && <span className="dot running" aria-label="Researching" />}
        {thread.latest.status === "error" && (
          <AlertIcon className="history-status error" aria-label="Failed" />
        )}
        <span className="history-text">{extra ? threadTitle(thread) : dateTime(thread.first.created_at)}</span>
        {extra && extra.family.threads.length > 1 && (
          <span className="history-runs" aria-label={`Asked ${extra.family.threads.length} times`}>
            ×{extra.family.threads.length}
          </span>
        )}
        {thread.items.length > 1 && (
          <span className="history-count" aria-label={`${thread.items.length - 1} follow-ups`}>
            +{thread.items.length - 1}
          </span>
        )}
      </button>
      <button
        className="history-delete"
        onClick={() => onDelete(thread)}
        aria-label="Delete from history"
        title="Delete"
      >
        <TrashIcon />
      </button>
    </div>
  );

  return (
    <>
      <div className={`scrim ${open ? "visible" : ""}`} onClick={onClose} />
      <aside className={`sidebar ${open ? "open" : ""}`} aria-label="Research history">
        <div className="sidebar-head">
          <div className="brand">
            <span className="brand-mark" aria-hidden="true" />
            Cited Research Assistant
            <span
              className="app-version"
              title={`Backend v${backendVersion?.app ?? "?"} (${backendVersion?.commit ?? "?"}); frontend build v${APP_VERSION} (${BUILD_COMMIT})`}
            >
              v{backendVersion?.app ?? APP_VERSION}
            </span>
          </div>
          <button className="icon-button mobile-only" onClick={onClose} aria-label="Close history">
            <XIcon />
          </button>
          <button className="icon-button desktop-only" onClick={onCollapse} aria-label="Hide history" title="Hide history">
            <SidebarIcon />
          </button>
        </div>

        <button className="new-question" onClick={() => onSelect(null)}>
          <PlusIcon />
          New question
        </button>

        {history.length > 4 && (
          <label className="history-filter">
            <SearchIcon />
            <input
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="Filter history"
              aria-label="Filter history"
            />
          </label>
        )}

        <nav className="history">
          {history.length === 0 && (
            <p className="history-empty">Your questions and answers will be kept here.</p>
          )}
          {history.length > 0 && groups.length === 0 && (
            <p className="history-empty">No matching questions.</p>
          )}
          {groups.map((group) => (
            <section key={group.label}>
              <h2 className="history-group">{group.label}</h2>
              <ul>
                {group.families.map((family) => {
                  // A family holding the open conversation stays expanded, so it is visible.
                  const open = expanded.has(family.key) ||
                    family.threads.slice(1).some((t) => t.id === currentThread);
                  return (
                    <li key={family.threads[0].id}>
                      {row(family.threads[0], { family, open })}
                      {open && family.threads.length > 1 && (
                        <ul className="history-runs-list">
                          {family.threads.slice(1).map((thread) => (
                            <li key={thread.id}>{row(thread)}</li>
                          ))}
                        </ul>
                      )}
                    </li>
                  );
                })}
              </ul>
            </section>
          ))}
        </nav>

        <div className="sidebar-foot">
          <button className="sidebar-connections" onClick={onConnections}>
            <PlugIcon />
            Connections
            <span
              className={`conn-dot ${connectionsReady ? "connected" : "disconnected"}`}
              aria-label={connectionsReady ? "All connected" : "Connection needed"}
            />
          </button>
          <TextSize />
          <ThemeSwitch />
        </div>
      </aside>
    </>
  );
}
