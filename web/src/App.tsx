import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { api, streamEvents } from "./api";
import type { BackendVersion, HistoryItem, ProgressEvent, ResearchItem } from "./types";
import { APP_VERSION, BUILD_COMMIT } from "./version";
import { dateTime, seconds } from "./format";
import { Sidebar, threadsOf, type Thread } from "./components/Sidebar";
import { Composer, type ComposerHandle } from "./components/Composer";
import { Progress } from "./components/Progress";
import { CostInfo } from "./components/CostInfo";
import { QuestionBubble } from "./components/QuestionBubble";
import { Answer } from "./components/Answer";
import { Sources } from "./components/Sources";
import { Details } from "./components/Details";
import { AlertIcon, MenuIcon, PlugIcon, PlusIcon, RetryIcon, SidebarIcon } from "./components/Icons";
import { ConnectionsDialog, Setup, useConnections } from "./components/Connections";

/** The open question lives in the URL hash (#/r/<id>), so reloads and back/forward work. */
function useRoute(): [string | null, (id: string | null) => void] {
  const read = () => window.location.hash.match(/^#\/r\/([\w-]+)/)?.[1] ?? null;
  const [id, setId] = useState(read);
  useEffect(() => {
    const onChange = () => setId(read());
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  const navigate = useCallback((next: string | null) => {
    window.location.hash = next ? `/r/${next}` : "/";
  }, []);
  return [id, navigate];
}

const SIDEBAR_KEY = "cra.sidebarCollapsed";
const NARROW = "(max-width: 860px)";

/** Whether the history sidebar is hidden on wide screens, remembered per viewer. Without a saved
 * choice it starts hidden when the page opens on a narrow screen (where it is a drawer). */
function useSidebarCollapsed(): [boolean, (collapsed: boolean) => void] {
  const [collapsed, setCollapsed] = useState(() => {
    try {
      const saved = window.localStorage.getItem(SIDEBAR_KEY);
      if (saved !== null) return saved === "1";
    } catch {
      /* storage unavailable */
    }
    return window.matchMedia?.(NARROW).matches ?? false;
  });
  const update = useCallback((next: boolean) => {
    setCollapsed(next);
    try {
      window.localStorage.setItem(SIDEBAR_KEY, next ? "1" : "0");
    } catch {
      /* storage unavailable: the choice lasts for this page only */
    }
  }, []);
  return [collapsed, update];
}

/** A stored question, kept live while it is being researched. */
function useResearch(id: string | null, onFinished: () => void) {
  const [item, setItem] = useState<ResearchItem | null>(null);
  const [events, setEvents] = useState<ProgressEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const finished = useRef(onFinished);
  finished.current = onFinished;

  useEffect(() => {
    setItem(null);
    setEvents([]);
    setError(null);
    if (!id) return;
    let alive = true;
    let close: (() => void) | undefined;

    const load = () =>
      api.get(id).then((record) => {
        if (!alive) return record;
        setItem(record);
        setEvents(record.events ?? []);
        return record;
      });

    load()
      .then((record) => {
        if (!alive || record.status !== "running") return;
        close = streamEvents(
          id,
          () => setEvents([]),
          (event) => {
            if (event.type === "end") {
              load().catch(() => {});
              finished.current();
            } else {
              setEvents((prev) => [...prev, event]);
            }
          },
        );
      })
      .catch((e: Error) => alive && setError(e.message));

    return () => {
      alive = false;
      close?.();
    };
  }, [id]);

  return { item, events, error };
}

/** Example chips: a short label, and the question it puts in the composer (not submitted). */
const EXAMPLES = [
  // These suit the sample corpus (sample_corpus/); change them for your own library.
  { label: "Vacation carry-over", question: "Can I carry over all my vacation days?" },
  { label: "Remote work", question: "How many days a week can I work from home?" },
  { label: "Expense claim", question: "What are the limits for an expense claim, and how is it filed?" },
];

function Welcome({ children, onExample }: { children: ReactNode; onExample: (q: string) => void }) {
  return (
    <div className="welcome">
      <span className="welcome-mark" aria-hidden="true" />
      <h1>What would you like to research?</h1>
      <p>
        Ask a question about the documents in your source library. Cited Research Assistant searches
        the source material, finds the most relevant passages, and explains what they say in context.
      </p>
      {children}
      <div className="examples" aria-label="Example questions">
        {EXAMPLES.map((e) => (
          <button key={e.label} type="button" className="example-chip" onClick={() => onExample(e.question)}>
            {e.label}
          </button>
        ))}
      </div>
    </div>
  );
}

/** The answer text streamed so far (answer_delta events; a reset starts it over). */
function streamedAnswer(events: ProgressEvent[]): string {
  let text = "";
  for (const e of events) if (e.type === "answer_delta") text = e.reset ? e.text : text + e.text;
  return text;
}

function ResearchView({
  item,
  events,
  onStop,
  onRetry,
  onResearchAgain,
  onConnections,
}: {
  item: ResearchItem;
  events: ProgressEvent[];
  onStop: () => void;
  onRetry: () => void;
  onResearchAgain?: () => void; // the open question only
  onConnections: () => void;
}) {
  const result = item.result;
  const running = item.status === "running";
  // One Answer in one place: the streamed text while running, then the final text, which is the
  // same markdown, so the answer stays exactly where it is when the stream ends.
  const answer = running ? streamedAnswer(events) : item.status === "done" && result ? result.answer : "";
  // A follow-up is researched as its standalone rewrite: show it under the question.
  const planned = events.find((e) => e.type === "plan");
  const standalone =
    result?.details.follow_up?.standalone ?? (planned?.type === "plan" ? planned.standalone : null) ?? null;
  const rewritten = standalone && standalone.trim().toLowerCase() !== item.question.trim().toLowerCase();
  // The citation last clicked in the answer, which the sources panel opens.
  const [focus, setFocus] = useState<{ n: number; seq: number } | null>(null);
  const onCite = useCallback((n: number) => setFocus((f) => ({ n, seq: (f?.seq ?? 0) + 1 })), []);
  return (
    <article className="research" id={`r-${item.id}`}>
      <header className="research-head">
        <QuestionBubble text={item.question} />
        {rewritten && <p className="research-rewrite">Researching: {standalone}</p>}
        <p className="research-meta">
          {dateTime(item.created_at)}
          {item.status === "done" && result && (
            <>
              <span aria-hidden="true"> · </span>
              Finished in {seconds(result.details.total_seconds)}
              {result.details.follow_up?.reuse === "answer_from_turn" && (
                <>
                  <span aria-hidden="true"> · </span>
                  Answered from earlier research (no new search)
                </>
              )}
              {!result.details.exact_reuse?.hit && <CostInfo details={result.details} />}
            </>
          )}
        </p>
      </header>

      <Progress
        events={events}
        status={item.status}
        createdAt={item.created_at}
        onStop={onStop}
        totalSeconds={result?.details.total_seconds}
        finishedNote={result?.details.exact_reuse?.hit ? "Reused an identical earlier result" : undefined}
      />

      {item.status === "done" && result?.details.exact_reuse?.hit && (
        <div className="notice">
          <div className="notice-body">
            <p>
              Reused the result of an identical earlier question ({(result.details.exact_reuse.age_hours ?? 0).toFixed(1)} h
              old, run {result.details.exact_reuse.source_run}). No research ran; use Research again for a fresh run.
            </p>
          </div>
        </div>
      )}

      {answer && (
        <Answer text={answer} streaming={running} citations={running ? undefined : result?.citations} onCite={onCite} />
      )}

      {(item.status === "error" || item.status === "cancelled") && (
        <div className={`notice ${item.status}`}>
          {item.status === "error" && <AlertIcon />}
          <div className="notice-body">
            <p>{item.status === "error" ? "The research did not finish." : "You stopped this research."}</p>
            {item.error && <pre className="notice-detail">{item.error}</pre>}
          </div>
          {item.error?.includes("Connections") && (
            <button className="ghost-button" onClick={onConnections}>
              <PlugIcon />
              Connections
            </button>
          )}
          <button className="ghost-button" onClick={onRetry}>
            <RetryIcon />
            Ask again
          </button>
        </div>
      )}

      {item.status === "done" && result && (
        <>
          <Sources key={item.id} sources={result.sources} citations={result.citations} focus={focus} idPrefix={`r-${item.id}`} />
          {onResearchAgain && (<div className="research-actions">
            <button
              className="ghost-button small"
              onClick={onResearchAgain}
              title="Research this question again, asking every fact question fresh (no reuse of earlier results or fact memory)"
            >
              <RetryIcon />
              Research again
            </button>
          </div>)}
          <Details result={result} />
        </>
      )}
    </article>
  );
}

/** An earlier question of the open conversation, shown finished above the current one. */
function EarlierItem({
  id,
  cached,
  onRetry,
  onConnections,
}: {
  id: string;
  cached?: ResearchItem;
  onRetry: (question: string) => void;
  onConnections: () => void;
}) {
  const [item, setItem] = useState<ResearchItem | null>(cached ?? null);
  useEffect(() => {
    if (cached) return;
    let alive = true;
    api.get(id).then((record) => alive && setItem(record)).catch(() => {});
    return () => {
      alive = false;
    };
  }, [id, cached]);
  if (!item) return <div className="loading" aria-label="Loading" />;
  return (
    <ResearchView
      item={item}
      events={item.events ?? []}
      onStop={() => {}}
      onRetry={() => onRetry(item.question)}
      onConnections={onConnections}
    />
  );
}

export default function App() {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [id, navigate] = useRoute();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useSidebarCollapsed();
  const [connectionsOpen, setConnectionsOpen] = useState(false);
  const conn = useConnections();
  const composer = useRef<ComposerHandle>(null);
  const scroller = useRef<HTMLDivElement>(null);

  const refreshHistory = useCallback(() => api.history().then(setHistory).catch(() => {}), []);
  // The running backend's app version and commit; a version or commit other than the one this
  // frontend was built from means the code changed since the app started.
  const [backendVersion, setBackendVersion] = useState<BackendVersion | null>(null);
  // Checked again whenever the window regains focus, so an edit made while the app is
  // open (prompts are read only at startup) shows up.
  useEffect(() => {
    const check = () => api.version().then(setBackendVersion).catch(() => {});
    check();
    window.addEventListener("focus", check);
    return () => window.removeEventListener("focus", check);
  }, []);
  const changedFiles = backendVersion?.changed ?? [];
  const codeChanged =
    backendVersion !== null &&
    (changedFiles.length > 0 ||
      (backendVersion.app !== undefined && backendVersion.app !== APP_VERSION) ||
      (backendVersion.commit !== "unknown" &&
        BUILD_COMMIT !== "unknown" &&
        backendVersion.commit !== BUILD_COMMIT));
  useEffect(() => {
    refreshHistory();
  }, [refreshHistory]);

  // Keep the sidebar's status markers current while any question is still running.
  const anyRunning = history.some((h) => h.status === "running");
  useEffect(() => {
    if (!anyRunning) return;
    const timer = setInterval(refreshHistory, 3000);
    return () => clearInterval(timer);
  }, [anyRunning, refreshHistory]);

  const { item, events, error } = useResearch(id, refreshHistory);

  // The open conversation: the questions before the open one are shown above it.
  const threads = useMemo(() => threadsOf(history), [history]);
  const thread = id ? threads.find((t) => t.items.some((i) => i.id === id)) : undefined;
  const openedAt = thread?.items.find((i) => i.id === id)?.created_at ?? Infinity;
  const earlier = thread ? thread.items.filter((i) => i.id !== id && i.created_at < openedAt) : [];
  // Finished items already on screen, so an earlier question does not reload (or flash) when a
  // follow-up opens below it.
  const known = useRef(new Map<string, ResearchItem>());
  if (item && item.status === "done") known.current.set(item.id, item);

  useEffect(() => {
    setSidebarOpen(false);
  }, [id]);

  // Opening a question shows its start: the top of the page, or, below earlier questions of its
  // conversation, the question itself. Once per question; nothing scrolls when it finishes.
  const scrolledFor = useRef<string | null>(null);
  const hasEarlier = earlier.length > 0;
  useEffect(() => {
    if (!id || !item || item.id !== id || scrolledFor.current === id) return;
    scrolledFor.current = id;
    const el = document.getElementById(`r-${id}`);
    if (hasEarlier && el) el.scrollIntoView({ block: "start" });
    else scroller.current?.scrollTo({ top: 0 });
  }, [id, item, hasEarlier]);
  useEffect(() => {
    if (!id) scroller.current?.scrollTo({ top: 0 });
  }, [id]);

  // No scroll when the research finishes; the reader stays where they are.

  async function ask(question: string, previousId: string | null = null, fresh = false) {
    if (conn.connections && !conn.ready) {
      navigate(null);
      throw new Error(`Connect ${conn.missing.map((c) => c.name).join(" and ")} first.`);
    }
    const record = await api.ask(question, previousId, fresh);
    await refreshHistory();
    navigate(record.id);
  }

  function select(next: string | null) {
    navigate(next);
    if (!next) setTimeout(() => composer.current?.focus(), 0);
  }

  /** The composer in an open conversation continues it: a follow-up to the open answer. */
  async function followUp(question: string) {
    if (!item || item.status === "running")
      throw new Error("Wait for this answer to finish, or start a new question.");
    // An answer that failed or was stopped cannot be followed up; ask it as a new question.
    return ask(question, item.status === "done" ? item.id : null);
  }

  async function remove(target: Thread) {
    const n = target.items.length;
    const prompt =
      n > 1
        ? `Delete this conversation (${n} questions) from history?\n\n${target.first.question}`
        : `Delete this question from history?\n\n${target.first.question}`;
    if (!window.confirm(prompt)) return;
    for (const i of target.items) await api.remove(i.id).catch(() => {});
    if (target.items.some((i) => i.id === id)) navigate(null);
    refreshHistory();
  }

  // The new-question screen places the composer under its intro; elsewhere it is docked.
  const welcome = !id && !(conn.connections && !conn.ready);

  return (
    <div className={`app ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
      <Sidebar
        history={history}
        currentThread={thread?.id ?? null}
        open={sidebarOpen}
        onSelect={select}
        onDelete={remove}
        onClose={() => setSidebarOpen(false)}
        onCollapse={() => setSidebarCollapsed(true)}
        onConnections={() => setConnectionsOpen(true)}
        connectionsReady={conn.connections === null || conn.ready}
        backendVersion={backendVersion}
      />
      <ConnectionsDialog
        open={connectionsOpen}
        onClose={() => setConnectionsOpen(false)}
        connections={conn.connections}
        act={conn.act}
      />

      <main className="main">
        {sidebarCollapsed && (
          <button
            className="icon-button desktop-only sidebar-expand"
            onClick={() => setSidebarCollapsed(false)}
            aria-label="Show history"
            title="Show history"
          >
            <SidebarIcon />
          </button>
        )}
        <header className="topbar">
          <button className="icon-button" onClick={() => setSidebarOpen(true)} aria-label="Open history">
            <MenuIcon />
          </button>
          <span className="topbar-title">Cited Research Assistant</span>
          <button className="icon-button" onClick={() => setConnectionsOpen(true)} aria-label="Connections">
            <PlugIcon />
          </button>
          <button className="icon-button" onClick={() => select(null)} aria-label="New question">
            <PlusIcon />
          </button>
        </header>
        {codeChanged && (
          <div className="code-changed" role="status">
            App code changed — restart the app
            <span className="muted">
              {" "}
              {changedFiles.length > 0
                ? `(edited since start: ${changedFiles.join(", ")})`
                : `(running v${backendVersion?.app} ${backendVersion?.commit}, built v${APP_VERSION} ${BUILD_COMMIT})`}
            </span>
          </div>
        )}

        <div className="scroller" ref={scroller}>
          <div className="column">
            {!id && (conn.connections && !conn.ready ? <Setup connections={conn.connections} act={conn.act} /> : (
              <Welcome onExample={(q) => composer.current?.fill(q)}>
                <Composer ref={composer} onSubmit={(q) => ask(q)} placeholder="Ask a question…" inline />
              </Welcome>
            ))}
            {id && error && (
              <div className="notice error">
                <AlertIcon />
                <div className="notice-body">
                  <p>This question could not be loaded.</p>
                  <pre className="notice-detail">{error}</pre>
                </div>
              </div>
            )}
            {id &&
              earlier.map((e) => (
                <EarlierItem
                  key={e.id}
                  id={e.id}
                  cached={known.current.get(e.id)}
                  onRetry={(q) => ask(q).catch(() => {})}
                  onConnections={() => setConnectionsOpen(true)}
                />
              ))}
            {id && !error && !item && <div className="loading" aria-label="Loading" />}
            {item && (
              <ResearchView
                key={item.id}
                item={item}
                events={events}
                onStop={() => api.cancel(item.id).catch(() => {})}
                onRetry={() => ask(item.question).catch(() => {})}
                onResearchAgain={() => ask(item.question, item.parent_id, true).catch(() => {})}
                onConnections={() => setConnectionsOpen(true)}
              />
            )}
          </div>
        </div>

        {!welcome && (
          <Composer
            ref={composer}
            onSubmit={id ? followUp : (q) => ask(q)}
            placeholder={id ? "Ask a follow-up…" : "Ask a question…"}
            onNewQuestion={id ? () => select(null) : undefined}
          />
        )}
      </main>
    </div>
  );
}
