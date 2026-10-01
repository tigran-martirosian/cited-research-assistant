import { useEffect, useState } from "react";
import type { ProgressEvent, StageKey, Status } from "../types";
import { seconds } from "../format";
import { CheckIcon, ChevronIcon, StopIcon, XIcon } from "./Icons";

/** The steps are the run's actual event timeline, in the order they happened. Pipeline
 * stages map to user steps; consecutive stages of one step join it, and a step that comes back
 * after another (a search after reading) is a new row ("Searching more"), so a step never shows
 * above one that happened before it. Runs saved before the Reading/Writing stages existed show
 * their reasoning stages as Writing. The normal path is Asking NotebookLM, (Asking again),
 * Reading sources, Writing; after a fallback event the search pipeline's stages join one
 * "Fallback: search" row. Planning comes first again, and the ask rows say how many fact
 * questions were asked and how many fact memory answered ("3 asked, 2 reused"). */
const GROUP: Partial<Record<StageKey, string>> = {
  auth: "connect",
  plan: "plan",
  search: "search",
  continuation: "search",
  select: "search",
  context: "search",
  repair: "search",
  check: "check",
  read: "read",
  write: "write",
  ask: "ask",
};
// The search pipeline's stages after a fallback event.
const FALLBACK_STAGES = new Set<StageKey>(["auth", "plan", "search", "continuation", "select", "context", "repair"]);
const LABEL: Record<string, [string, string]> = {
  // [first time, again]
  connect: ["Connecting", "Connecting"],
  plan: ["Planning", "Planning"],
  search: ["Searching", "Searching more"],
  check: ["Checking", "Checking"],
  read: ["Reading sources", "Reading sources"],
  write: ["Writing", "Writing"],
  ask: ["Asking NotebookLM", "Asking again"],
  fallback: ["Fallback: search", "Fallback: search"],
};

type StepState = "running" | "done" | "failed" | "stopped";

interface Step {
  group: string;
  label: string;
  state: StepState;
  startedAt: number;
  endedAt?: number;
  running: Set<StageKey>;
  note?: string; // e.g. "3 asked, 2 reused" on an ask row
}

function timeline(events: ProgressEvent[], status: Status): Step[] {
  const legacy = !events.some((e) => "stage" in e && (e.stage === "read" || e.stage === "write"));
  let fallback = false;
  const group = (stage: StageKey) =>
    fallback && FALLBACK_STAGES.has(stage)
      ? "fallback"
      : GROUP[stage] ?? (legacy && stage === "reason" ? "write" : undefined);
  const steps: Step[] = [];
  const seen = new Set<string>();
  // The open step each running stage belongs to (auth and plan run side by side).
  const owner = new Map<StageKey, Step>();
  for (const event of events) {
    if (event.type === "fallback") fallback = true;
    if (event.type === "ask_plan") {
      const step = [...steps].reverse().find((s) => s.group === "ask");
      if (step) step.note = `${event.asked} asked, ${event.reused} reused`;
      continue;
    }
    if (event.type !== "stage_start" && event.type !== "stage_end") continue;
    const g = group(event.stage);
    if (!g) continue;
    if (event.type === "stage_start") {
      const last = steps[steps.length - 1];
      // Consecutive stages of one step (search, continuation, select, context) join its row;
      // Connecting runs beside Planning and the steps after it.
      // The repair round's retrieval runs as "check" inside "repair": one Searching more row.
      const inRepair = g === "check" && last?.state === "running" && last.running.has("repair");
      let step =
        last && (last.group === g || inRepair) ? last : g === "connect" ? steps.find((s) => s.group === g) : undefined;
      if (!step) {
        // A new row: the rows still open before it are over (Connecting may still be waiting).
        for (const s of steps) {
          if (s.state !== "running" || s.group === "connect") continue;
          s.state = "done";
          s.endedAt ??= event.at;
          for (const k of s.running) owner.delete(k);
          s.running.clear();
        }
        step = { group: g, label: LABEL[g][seen.has(g) ? 1 : 0], state: "running", startedAt: event.at, running: new Set() };
        steps.push(step);
        seen.add(g);
      }
      step.state = "running";
      step.endedAt = undefined;
      step.running.add(event.stage);
      owner.set(event.stage, step);
    } else {
      const step = owner.get(event.stage);
      if (!step) continue;
      step.running.delete(event.stage);
      owner.delete(event.stage);
      step.endedAt = event.at;
      if (step.running.size === 0) step.state = "done";
    }
  }
  if (status !== "running")
    for (const s of steps)
      if (s.state === "running") s.state = status === "error" ? "failed" : status === "cancelled" ? "stopped" : "done";
  return steps;
}

function useNow(active: boolean) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const timer = setInterval(() => setNow(Date.now()), 100);
    return () => clearInterval(timer);
  }, [active]);
  return now / 1000;
}

interface Props {
  events: ProgressEvent[];
  status: Status;
  createdAt: number;
  onStop?: () => void;
  /** A finished research's total time; the panel then collapses to "Researched in Xs ▸". */
  totalSeconds?: number;
  /** Shown instead of the time for a finished research (e.g. an exact reuse). */
  finishedNote?: string;
}

export function Progress({ events, status, createdAt, onStop, totalSeconds, finishedNote }: Props) {
  const running = status === "running";
  const [expanded, setExpanded] = useState(false);
  const now = useNow(running);
  const steps = timeline(events, status);
  const current = [...steps].reverse().find((s) => s.state === "running");

  const list = (
    <ol className="stages">
      {steps.map((step, i) => (
        <li key={i} className={`stage ${step.state}`}>
          <span className="stage-icon" aria-hidden="true">
            {step.state === "done" && <CheckIcon />}
            {step.state === "failed" && <XIcon />}
            {step.state === "stopped" && <StopIcon />}
            {step.state === "running" && <span className="spinner" />}
          </span>
          <div className="stage-body">
            <div className="stage-line">
              <span className="stage-label">
                {step.label}
                {step.note && <span className="muted"> ({step.note})</span>}
              </span>
              <span className="stage-time">
                {step.state === "running" && seconds(Math.max(0, now - step.startedAt))}
                {step.state !== "running" && step.endedAt != null && seconds(Math.max(0, step.endedAt - step.startedAt))}
              </span>
            </div>
          </div>
        </li>
      ))}
    </ol>
  );

  if (status === "done") {
    const label = finishedNote ?? `Researched in ${seconds(totalSeconds ?? 0)}`;
    return (
      <section className={`progress collapsed ${expanded ? "open" : ""}`}>
        <button className="progress-toggle" onClick={() => setExpanded(!expanded)} aria-expanded={expanded}>
          <span>{label}</span>
          <ChevronIcon className="chevron" />
        </button>
        {expanded && list}
      </section>
    );
  }

  return (
    <section className={`progress ${running ? "live" : ""}`} aria-live="polite">
      <header className="progress-head">
        <div>
          <p className="progress-title">
            {running
              ? `${current?.label ?? (steps.length ? "Working" : "Starting")}…`
              : status === "cancelled"
                ? "Stopped"
                : "Research failed"}
          </p>
          {running && <p className="progress-sub">{seconds(Math.max(0, now - createdAt))} elapsed</p>}
        </div>
        {running && onStop && (
          <button className="ghost-button" onClick={onStop}>
            <StopIcon />
            Stop
          </button>
        )}
      </header>
      {list}
    </section>
  );
}
