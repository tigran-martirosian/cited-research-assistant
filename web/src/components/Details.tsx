import { useState } from "react";
import type {
  AskActivity,
  AnswerChecks,
  ClaudeUsage,
  Coverage,
  Details as RunDetails,
  FactQuestion,
  MapEntry,
  MapStatement,
  Result,
  SynthesisItem,
  TraceCandidate,
  TraceRaw,
} from "../types";
import { chars, cost, count, runTokens, seconds } from "../format";
import { CheckIcon, ChevronIcon, CopyIcon } from "./Icons";
import { SourceList } from "./Sources";

function tokens(usage: ClaudeUsage["usage"], key: string): string {
  const value = usage?.[key];
  return typeof value === "number" ? count(value) : "—";
}

/** A stage's model(s), with the Gemini fallback note when Haiku took over the selector. */
function modelLabel(u: ClaudeUsage): string {
  const models = u.models.join(", ") || "—";
  if (u.failed) return `${models} (failed: ${u.failed})`;
  return u.fallback ? `${models} (${u.fallback})` : models;
}

function isGemini(u: ClaudeUsage): boolean {
  return u.provider === "gemini";
}

/** Fresh (uncached) input. Antigravity's raw fields do not say whether input_tokens includes cache
 * reads; with no cache read both readings agree, otherwise fresh input is unavailable. */
function freshInput(u: ClaudeUsage): string {
  if (!isGemini(u) || !u.usage) return tokens(u.usage, "input_tokens");
  return u.usage["cache_read_tokens"] === 0 ? tokens(u.usage, "input_tokens") : "n/a";
}

const USAGE_COLUMNS = ["Stage", "Provider / model", "Origin", "Fresh input", "Cache write", "Cache read", "Output", "Thinking", "Time", "Cost"];

/** Per-stage accounting rows (see USAGE_COLUMNS): model stages, then each NotebookLM search.
 * Claude output includes thinking; Gemini output and thinking are the worker's raw fields. On an
 * exact reuse the rows are the source run's. */
function usageRows(d: RunDetails): string[][] {
  const reused = d.exact_reuse?.hit;
  const rows = d.claude_usage.map((u) => {
    const gemini = isGemini(u);
    return [
      u.stage,
      modelLabel(u),
      reused ? "exact reuse" : "fresh",
      freshInput(u),
      gemini ? "—" : tokens(u.usage, "cache_creation_input_tokens"),
      tokens(u.usage, gemini ? "cache_read_tokens" : "cache_read_input_tokens"),
      tokens(u.usage, "output_tokens"),
      gemini ? tokens(u.usage, "thinking_tokens") : u.usage ? "in output" : "—",
      seconds(u.wall_seconds ?? u.seconds),
      gemini ? "not computed" : cost(u.cost_usd),
    ];
  });
  for (const q of d.retrieval?.detail ?? []) {
    rows.push([`${q.round > 1 ? "follow-up" : "search"}: ${q.query}`, "NotebookLM", reused ? "exact reuse" : q.origin,
      "—", "—", "—", "—", "—", seconds(q.seconds), "—"]);
  }
  return rows;
}

/** The headline: total wall time and depth (cost only under Show usage). */
function headline(d: RunDetails): string {
  const e = d.exact_reuse;
  if (e?.hit) return `exact reuse · ${seconds(e.seconds ?? 0)} (source run: ${seconds(d.total_seconds)})`;
  return `${seconds(d.total_seconds)} · ${d.depth} depth`;
}

const USAGE_KEY = "cra.showUsage";

/** Show usage, remembered per viewer (browser storage may be unavailable: then off). */
function useShowUsage(): [boolean, (on: boolean) => void] {
  const [on, setOn] = useState(() => {
    try {
      return localStorage.getItem(USAGE_KEY) === "1";
    } catch {
      return false;
    }
  });
  const set = (value: boolean) => {
    setOn(value);
    try {
      localStorage.setItem(USAGE_KEY, value ? "1" : "0");
    } catch {
      /* storage unavailable */
    }
  };
  return [on, set];
}

/** Exact reuse of a completed result: hit or miss and why. */
function exactText(e: RunDetails["exact_reuse"]): string {
  if (!e) return "not recorded";
  if (!e.hit) return `miss — ${e.reason ?? "unknown"}`;
  return `hit — run ${e.source_run}, ${(e.age_hours ?? 0).toFixed(1)} h old`;
}

function exactNote(e: RunDetails["exact_reuse"]): string | undefined {
  if (!e) return undefined;
  if (!e.hit) return `key ${e.key}`;
  return `corpus identity ${e.corpus_strength ?? "?"} (epoch ${e.corpus?.epoch ?? "?"}) · policy ${e.policy_match ? "matches" : "differs"}`;
}

/** The earlier answer to this question that the run re-checked, or why none. */
function priorText(p: RunDetails["prior_answer"]): string {
  if (!p) return "none (follow-up)";
  if (!p.found) return `none — ${p.reason ?? "unknown"}`;
  const age = p.age_hours != null ? `, ${p.age_hours.toFixed(1)} h old` : "";
  return `re-checked run ${p.source_run}${age}; ${p.passages ?? 0} cited passages offered`
    + (p.policy_version ? ` (answered under policy ${p.policy_version})` : "");
}

function retrievalText(r: NonNullable<RunDetails["retrieval"]>): string {
  return `${r.from_cache} of ${r.queries} ${r.queries === 1 ? "search" : "searches"} from cache, ${r.fresh} fresh`;
}

function retrievalNote(r: NonNullable<RunDetails["retrieval"]>): string {
  return `passages: ${r.passages_from_cache} from cache, ${r.passages_fresh} fresh`;
}

/** Raw hits merged into another candidate of the same source: identical text, or contained. */
function collapsedText(trace: RunDetails["trace"]): string {
  const merges = (trace?.raw ?? []).filter((r) => r.merge);
  const identical = merges.filter((r) => r.merge?.includes("identical text")).length;
  return `${identical} identical, ${merges.length - identical} contained (same source)`;
}

const SELECTOR_REUSE = "all fresh (cross-question reuse disabled by design)";

function repairNote(r: NonNullable<RunDetails["repair"]>): string {
  if (r.error) return "search failed";
  return `${r.candidates} new candidates, ${r.selected} selected` + (r.forced ? " · exact detail missing" : "");
}

/** Answer diagnostics that found something, as [label, values] (the answer is never changed). */
function checkRows(c: AnswerChecks | undefined): [string, string][] {
  if (!c) return [];
  const rows: [string, string[] | undefined][] = [
    ["Grounding failures (quotes not in evidence)", c.grounding_failures?.map((q) => `"${q}"`)],
    ["Unverified numbers", c.unverified_numbers],
    ["Percentages not in evidence", c.ungrounded_percentages],
    ["Amounts not in evidence", c.ungrounded_quantities],
    ["Threshold wording not in evidence", c.threshold_phrases],
    ["Mechanism verbs not in evidence", c.unsourced_mechanism_verbs],
    ["Synthesis issues", c.synthesis_issues?.map((i) => `${i.issue.replace(/_/g, " ")} (${i.hit_id})`)],
    ["Citations of unknown passages", c.citation_unknown_ids],
    ["Quotes not in the passage they cite", c.citation_misattributed?.map(
      (m) => `"${m.quote}" cites ${m.cited.join(", ")}, found in ${m.found_in.join(", ")}`)],
    ["Quotes without a citation", c.citation_uncited_quotes?.map((q) => `"${q}"`)],
    ["Citations in the Short answer", c.citation_in_short_answer ? ["yes"] : undefined],
  ];
  return rows.filter(([, v]) => v && v.length > 0).map(([label, v]) => [label, (v as string[]).join(", ")]);
}

/** A community record's attribution check: the fact question that asked the corpus author's material about it. */
function checkText(c?: { fact?: string; origin?: string; passages?: number } | null): string {
  if (!c?.fact) return "";
  return ` · attribution checked by ${c.fact} (${c.origin ?? "not asked"}${c.passages != null ? `, ${c.passages} passages` : ""})`;
}

/** Research memory in one line: memory matches (added from memory vs. also found fresh) and secondary claims. */
function memoryText(m: NonNullable<RunDetails["memory"]>): string {
  if (m.status !== "ok") return m.status === "error" ? `unavailable (${m.errors.map((e) => e.stage).join(", ")})` : m.status.replace(/_/g, " ");
  const added = m.primary.filter((p) => !p.merged_into_fresh).length;
  return `${m.primary.length} memory matches (${added} memory-only passages added, ${m.primary.length - added} also found fresh), `
    + `${m.secondary.length} secondary claims`;
}

/** What was saved to memory in one line. */
function writtenText(w: NonNullable<RunDetails["memory"]>["written"]): string {
  return `${w.units_new} new passages saved (${w.units_seen} already known)`
    + (w.selections != null ? `, ${w.selections} selector decisions, ${w.relationships ?? 0} relations, ${w.derived ?? 0} derived` : "");
}

function statementText(s: MapStatement): string {
  const scope = s.scope + (s.applies_to ? `: ${s.applies_to}` : "");
  const pred = s.predicate ? `"${s.predicate}"${s.predicate_verbatim ? "" : " (not verbatim)"}` : "no predicate";
  return `${s.source}${s.date && !s.source.includes(s.date) ? ` (${s.date})` : ""} · ${scope} · ${pred}${s.use === "context" ? " · context only" : ""}`;
}

function relationText(r: MapEntry["relations"][number]): string {
  const note = r.check === "older_than_target" ? " (not supported by dates)" : r.check === "not_dated" ? " (dates unknown)"
    : r.check === "general_scope" ? " (scope mismatch)" : "";
  return `${r.from} ${r.type.replace(/_/g, " ")} ${r.to}${note}`;
}

function synthesisText(s: SynthesisItem): string {
  return `${s.hit_ids.join(", ")}: ${s.treatment.replace(/_/g, " ")}${s.qualifies ? ` of ${s.qualifies}` : ""} · applies to user: ${s.applies_to_user}`;
}

/** The evidence map (how kept statements relate per requirement) and the reasoner's synthesis. */
function EvidenceMap({ entries, synthesis }: { entries: MapEntry[]; synthesis?: SynthesisItem[] }) {
  const statements = entries.reduce((n, e) => n + e.statements.length, 0);
  return (
    <>
      <h3>Evidence map</h3>
      <details className="trace-group">
        <summary>
          Statements and relationships <span className="muted">{statements} kept statements</span>
        </summary>
        {entries.map((e) => (
          <div key={e.requirement_id} className="trace-query">
            <p className="trace-query-text">{e.requirement_id}</p>
            <ul className="trace-list">
              {e.statements.map((s) => (
                <li key={s.hit_id} className={s.use === "context" ? "dropped" : "kept"}>
                  <div className="trace-line">
                    <span className="trace-id">{s.hit_id}</span>
                    <span className="trace-fate">{statementText(s)}</span>
                  </div>
                </li>
              ))}
            </ul>
            {e.relations.length > 0 && <p className="trace-meta">{e.relations.map(relationText).join(" · ")}</p>}
            {e.gap && (
              <p className="trace-preview">
                gap ({e.gap.kind}): {e.gap.missing || "unspecified"}
                {e.gap.leads.length > 0 && ` · leads ${e.gap.leads.join(", ")}`}
              </p>
            )}
          </div>
        ))}
      </details>
      {synthesis && synthesis.length > 0 && (
        <details className="trace-group">
          <summary>
            Synthesis <span className="muted">how the answer combined the evidence</span>
          </summary>
          <ul className="trace-list">
            {synthesis.map((s, i) => (
              <li key={i} className={s.treatment === "not_used" ? "dropped" : "kept"}>
                <p className="trace-reason">{synthesisText(s)}</p>
              </li>
            ))}
          </ul>
        </details>
      )}
    </>
  );
}

function mapText(d: RunDetails): string[] {
  if (!d.evidence_map) return [];
  const lines = [``, `## Evidence map`];
  for (const e of d.evidence_map) {
    lines.push(`- ${e.requirement_id}`);
    for (const s of e.statements) lines.push(`  - ${s.hit_id}: ${statementText(s)}`);
    for (const r of e.relations) lines.push(`  - relation: ${relationText(r)}`);
    if (e.gap) lines.push(`  - gap (${e.gap.kind}): ${e.gap.missing}${e.gap.leads.length ? ` · leads ${e.gap.leads.join(", ")}` : ""}`);
  }
  if (d.synthesis?.length) {
    lines.push(``, `## Synthesis`);
    for (const s of d.synthesis) lines.push(`- ${synthesisText(s)}`);
  }
  return lines;
}

/** Coverage entries by requirement (or premise) id. */
function coverageById(entries: Coverage[] | undefined): Map<string, Coverage> {
  return new Map((entries ?? []).map((c) => [c.requirement_id, c]));
}

function coverageText(c: Coverage | undefined): string {
  if (!c) return "unassessed";
  const by = c.hit_ids.length ? ` by ${c.hit_ids.join(", ")}` : "";
  const gap = !c.missing ? "" : c.status === "missing" ? `: ${c.missing}` : ` — missing: ${c.missing}`;
  return `${c.status}${by}${gap}`;
}

/** The run answered from NotebookLM asks (no fallback), so the search-pipeline rows are empty. */
function askPath(d: RunDetails): boolean {
  return !!d.ask && !d.ask.fallback;
}

/** The planned searches with the requirements each covers (older results: queries only). */
function plannedSearches(result: Result): { query: string; covers: string[] }[] {
  return result.details.search_plan ?? result.search_queries.map((query) => ({ query, covers: [] }));
}

/** Follow-up premises as p1, p2 with the requirement each serves (older results: premise text only). */
function followUps(r: NonNullable<RunDetails["repair"]>) {
  return r.request.map((q, i) => ({ id: `p${i + 1}`, for: q.requirement_id ?? "", premise: q.premise, search: q.search }));
}

function CoverageBadge({ entry }: { entry: Coverage | undefined }) {
  const status = entry?.status ?? "unassessed";
  return <span className={`cov cov-${status}`}>{status}</span>;
}

function fate(c: TraceCandidate): string {
  if (!c.role && c.kept == null) return "not judged";
  return `${c.role ?? "?"}${c.kept ? " · kept" : " · dropped"}${c.context ? " · context" : ""}`;
}

function merged(r: TraceRaw): string {
  return r.merge ? `merged into ${r.candidate} (${r.merge})` : `→ ${r.candidate}`;
}

/** The evidence trace as Markdown: candidates with selector decisions, then raw hits by query. */
function traceText(d: RunDetails): string[] {
  if (!d.trace) return [];
  const lines = [``, `## Evidence trace`, ``, `### Candidates after merge/dedupe (selector decisions)`];
  for (const c of d.trace.candidates) {
    lines.push(`- ${c.id} [${fate(c)}] ${c.source} (raw ${c.raw_ids.join(", ")})`
      + (c.covers?.length ? ` — covers ${c.covers.join(", ")}` : "")
      + (c.reason ? ` — ${c.reason}` : "") + (c.continuation ? ` — continuation: ${c.continuation}` : "")
      + (c.predicate ? ` — "${c.predicate}" (${c.scope}${c.applies_to ? `: ${c.applies_to}` : ""})` : ""),
      `  > ${c.preview}`);
  }
  lines.push(``, `### Raw NotebookLM hits`);
  for (const [query, hits] of byQuery(d.trace.raw)) {
    lines.push(`- Query (round ${hits[0].round}): ${query}`);
    for (const r of hits) lines.push(`  - ${r.id} rank ${r.rank ?? "—"} ${r.source} ${merged(r)}`, `    > ${r.preview}`);
  }
  return lines;
}

function byQuery(raw: TraceRaw[]): [string, TraceRaw[]][] {
  const groups = new Map<string, TraceRaw[]>();
  for (const r of raw) {
    const key = `${r.round}\u0000${r.query}`;
    groups.set(key, [...(groups.get(key) ?? []), r]);
  }
  return [...groups.values()].map((hits) => [hits[0].query, hits]);
}

function Trace({ trace }: { trace: NonNullable<RunDetails["trace"]> }) {
  const kept = trace.candidates.filter((c) => c.kept).length;
  const merges = trace.raw.filter((r) => r.merge).length;
  return (
    <>
      <h3>Evidence trace</h3>
      <details className="trace-group">
        <summary>
          Selector decisions <span className="muted">{kept} kept of {trace.candidates.length} candidates</span>
        </summary>
        <ul className="trace-list">
          {trace.candidates.map((c) => (
            <li key={c.id} className={c.kept ? "kept" : "dropped"}>
              <div className="trace-line">
                <span className="trace-id">{c.id}</span>
                <span className={`role role-${(c.role ?? "none").toLowerCase()}`}>{c.role ?? "—"}</span>
                <span className="trace-fate">{c.kept ? "kept" : "dropped"}{c.context ? " · context" : ""}</span>
                {c.covers && c.covers.length > 0 && <span className="trace-covers">covers {c.covers.join(", ")}</span>}
                <span className="trace-source">{c.source}</span>
              </div>
              {c.reason && <p className="trace-reason">{c.reason}</p>}
              {c.predicate && (
                <p className="trace-meta">
                  “{c.predicate}” · {c.scope}
                  {c.applies_to ? `: ${c.applies_to}` : ""}
                  {c.use === "context" ? " · context only" : ""}
                  {c.relations?.length ? ` · ${c.relations.map((r) => `${r.type.replace(/_/g, " ")} ${r.hit_id}`).join(", ")}` : ""}
                </p>
              )}
              <p className="trace-preview">{c.preview}</p>
              <p className="trace-meta">
                raw {c.raw_ids.join(", ")}
                {c.continuation && ` · continuation ${c.continuation}`}
              </p>
            </li>
          ))}
        </ul>
      </details>
      <details className="trace-group">
        <summary>
          Raw retrieval <span className="muted">{trace.raw.length} hits · {merges} merged</span>
        </summary>
        {byQuery(trace.raw).map(([query, hits]) => (
          <div key={`${hits[0].round}-${query}`} className="trace-query">
            <p className="trace-query-text">
              {hits[0].round > 1 && <span className="muted">follow-up · </span>}
              {query}
            </p>
            <ul className="trace-list">
              {hits.map((r) => (
                <li key={r.id} className={r.merge ? "dropped" : "kept"}>
                  <div className="trace-line">
                    <span className="trace-id">{r.id}</span>
                    <span className="trace-fate">{merged(r)}</span>
                    <span className="trace-source">{r.source}</span>
                  </div>
                  <p className="trace-preview">{r.preview}</p>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </details>
    </>
  );
}

/** The NotebookLM ask rows (label, value, note); Fact questions, per-ask wall time
 * and fact memory. */
function askRows(a: AskActivity): [string, string, string?][] {
  const recovery = Object.entries(a.recovery).map(([k, v]) => `${v} ${k.replace(/_/g, " ")}`).join(", ");
  const facts = a.facts ?? [];
  const reused = facts.filter((f) => f.origin === "reused").length;
  const failed = facts.filter((f) => f.origin === "failed").length;
  const premise = a.reuse ? [...a.reuse.premise_asks, ...a.reuse.follow_up_asks] : [];
  return [
    ...(a.reuse
      ? ([["Answered from", "earlier research (no new search)",
          `${a.reuse.passages} passages from turn ${a.reuse.related_turn ?? "?"}` +
            (premise.length ? `; one missing premise asked: ${premise.join("; ")}` : "")]] as [string, string, string?][])
      : []),
    ...(facts.length
      ? ([["Fact questions", count(facts.length),
          `${facts.length - reused - failed} asked, ${reused} reused${failed ? `, ${failed} failed` : ""}`]] as [string, string, string?][])
      : []),
    ["Ask wall time", seconds(a.seconds), `${a.asks.length} ${a.asks.length === 1 ? "ask" : "asks"}`],
    ...a.asks.map((x): [string, string, string?] => [
      `Ask ${x.n}`,
      x.error ? "failed" : seconds(x.seconds),
      [x.questions?.join(", "), x.citations != null ? `${x.citations} citations` : null,
        x.cache ? "cached reply" : null, x.error].filter(Boolean).join("; ") || undefined,
    ]),
    ["Citations", count(a.citations)],
    ["Passages recovered", count(a.passages_recovered), recovery || undefined],
    ...(a.passages_from_ledger != null
      ? ([["Passages from fact memory", count(a.passages_from_ledger)]] as [string, string, string?][])
      : []),
    ["Passages from memory", count(a.passages_from_memory)],
    ...(a.ledger
      ? ([["Fact memory", a.ledger.status,
          `${a.ledger.offered} with candidates, ${a.ledger.matched} matched, ${a.ledger.written} written`]] as [string, string, string?][])
      : []),
    [
      "Follow-up ask",
      a.follow_up ? "yes" : "no",
      a.follow_up
        ? a.follow_up.error ?? `${a.follow_up.new_passages} new passages: ${a.follow_up.questions.join("; ")}`
        : undefined,
    ],
    ["Ask cache hit", a.cache_hit ? "yes" : "no"],
    ["Fallback used", a.fallback ? "yes" : "no", a.fallback ?? undefined],
  ];
}

/** How one fact question was answered, in a few words. */
function factNote(f: FactQuestion): string {
  if (f.origin === "reused") return `reused (fact memory L${f.ledger_id}${f.reused_from ? `, run ${f.reused_from}` : ""})`;
  if (f.origin === "failed") return `ask ${f.batch} failed`;
  return `asked in ask ${f.batch}${f.round > 1 ? " (follow-up)" : ""}, ${count(f.citations)} citations`;
}

/** Everything the panel shows, as Markdown (the answer itself is not included). */

function detailsText(result: Result): string {
  const d = result.details;
  const budget = d.evidence_budget ?? d.evidence_target?.[1];
  const usage = runTokens(d.claude_usage);
  const lines = [
    `# Research details`,
    ``,
    ...(d.version ? [`- Version: v${d.version.app ?? "?"}, code ${d.version.commit}`, ``] : []),
    `Question: ${result.question}`,
    `Summary: ${headline(d)}`,
    ``,
    ...(d.ask
      ? [`## NotebookLM ask`, ...askRows(d.ask).map(([l, v, n]) => `- ${l}: ${v}${n ? ` (${n})` : ""}`), ``]
      : []),
    ...(d.ask?.facts?.length
      ? [`## Fact questions`, ...d.ask.facts.map((f) => `- ${f.id}: ${f.question} — ${factNote(f)}, ${count(f.passages)} passages`), ``]
      : []),
    `## Retrieval`,
    ...(askPath(d) ? [] : [`- Depth: ${d.depth}`, `- Searches: ${count(d.searches)}`, `- Raw hits: ${count(d.raw_hits)}`]),
    `- Unique candidates: ${count(d.candidates)} (${chars(d.candidate_chars)})`,
    `- Cut-off passages: ${count(d.clipped)}` + (d.clipped ? ` (${d.continuation_incomplete} incomplete)` : ""),
    `- Selected passages: ${count(d.selected)}`,
    `- Context requests: ${count(d.context_hits)}` + (d.context_reduced ? ` (${d.context_reduced} reduced)` : ""),
  ];
  if (d.repair) lines.push(`- Evidence repair: ${d.repair.searches.length} follow-up searches (${repairNote(d.repair)})`);
  else if (d.repair === null) lines.push(`- Evidence repair: not needed`);
  lines.push(
    `- Evidence: ${chars(d.evidence_chars)}` + (budget != null ? ` (budget ${count(budget)})` : ""),
    `- Evidence blocks: ${count(d.evidence_blocks)} (${d.sources} sources)`,
    `- Fulltexts: ${d.fulltexts_fetched.length} fetched, ${d.fulltexts_from_cache.length} cached`,
  );
  for (const [label, value] of checkRows(d.checks)) lines.push(`- Answer check, ${label.toLowerCase()}: ${value}`);
  if (d.exact_reuse || d.retrieval) {
    const note = exactNote(d.exact_reuse);
    lines.push(`- Exact reuse: ${exactText(d.exact_reuse)}${note ? ` (${note})` : ""}`);
    if (d.prior_answer !== undefined) lines.push(`- Earlier answer: ${priorText(d.prior_answer)}`);
    if (d.retrieval && d.retrieval.queries > 0) {
      lines.push(`- Retrieval${d.exact_reuse?.hit ? " (source run)" : ""}: ${retrievalText(d.retrieval)}; ${retrievalNote(d.retrieval)}`);
      for (const q of d.retrieval.detail) {
        lines.push(`  - round ${q.round} ${q.origin}: ${q.query}`
          + (q.origin === "cache" ? ` (retrieved ${q.retrieved_at} by ${q.cached_from})` : q.miss ? ` (cache: ${q.miss})` : ""));
      }
    }
    lines.push(`- Duplicates collapsed: ${collapsedText(d.trace)}`, `- Selector: ${SELECTOR_REUSE}`);
  }
  if (d.memory) {
    lines.push(`- Research memory: ${memoryText(d.memory)}; ${writtenText(d.memory.written)}`);
    for (const p of d.memory.primary) lines.push(`  - memory match ${p.unit_id} → ${p.hit_id}${p.merged_into_fresh ? " (also found fresh)" : " (added from memory)"}`);
    for (const s of d.memory.secondary) lines.push(`  - community used ${s.id} ${s.unit_id} [${s.claim_type}, ${s.platform}${s.author ? `, ${s.author}` : ""}]${checkText(s.attribution_check)} ${s.preview}`);
    for (const s of d.memory.secondary_dropped ?? []) lines.push(`  - community dropped ${s.unit_id} (${s.reason}) ${s.preview}`);
  }
  if (d.requirements?.length) {
    const cov = coverageById(d.coverage);
    lines.push(``, `## Answer requirements (selector coverage)`);
    for (const r of d.requirements) {
      const c = cov.get(r.id);
      lines.push(`- ${r.id} [${r.kind}${r.exact ? ", exact" : ""}] ${r.text} — ${coverageText(c)}`
        + (c?.gap ? ` (gap: ${c.gap}${c.leads?.length ? `, leads ${c.leads.join(", ")}` : ""})` : ""));
    }
  }
  if (plannedSearches(result).length) lines.push(``, `## Search queries`);
  plannedSearches(result).forEach((s, i) =>
    lines.push(`${i + 1}. ${s.query}${s.covers.length ? ` (covers ${s.covers.join(", ")})` : ""}`));
  if (d.repair) {
    const cov = coverageById(d.repair.coverage);
    lines.push(``, `## Follow-up searches`);
    for (const f of followUps(d.repair)) {
      lines.push(`- ${f.id}${f.for ? ` (for ${f.for})` : ""}: ${f.search} — missing premise: ${f.premise}`
        + (d.repair.coverage ? ` — ${coverageText(cov.get(f.id))}` : ""));
    }
    if (d.repair.error) lines.push(``, `Error: ${d.repair.error}`);
  }
  lines.push(``, `## Timing`);
  for (const s of d.stage_seconds) lines.push(`- ${s.stage}: ${seconds(Math.max(0, s.seconds))}`);
  lines.push(`- Total: ${seconds(d.total_seconds)}`, ``, `## Usage by stage`,
    `| ${USAGE_COLUMNS.join(" | ")} |`, `|---|---|---|${"---:|".repeat(USAGE_COLUMNS.length - 3)}`);
  for (const row of usageRows(d)) lines.push(`| ${row.join(" | ")} |`);
  lines.push(`| Claude total | | | ${count(usage.input)} | ${count(usage.cacheWrite)} | ${count(usage.cacheRead)} | `
    + `${count(usage.output)} | | | ${cost(d.claude_cost_usd)} |`,
    ``, `Raw Claude tokens (not cost-weighted): ${count(usage.total)}. Gemini cost is not computed.`,
    ...mapText(d), ...traceText(d), ``, `## Logs`, result.run_dir);
  return lines.join("\n");
}

function Stat({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="stat">
      <dt>{label}</dt>
      <dd>
        {value}
        {note && <span className="muted"> {note}</span>}
      </dd>
    </div>
  );
}

export function Details({ result }: { result: Result }) {
  const [open, setOpen] = useState(false);
  const d = result.details;
  const budget = d.evidence_budget ?? d.evidence_target?.[1];
  const slowest = Math.max(...d.stage_seconds.map((s) => s.seconds), 0.1);
  const usage = runTokens(d.claude_usage);
  const [copied, setCopied] = useState(false);
  const [showUsage, setShowUsage] = useShowUsage();

  async function copy() {
    try {
      await navigator.clipboard.writeText(detailsText(result));
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable */
    }
  }

  return (
    <section className={`details ${open ? "open" : ""}`}>
      <button className="details-head" onClick={() => setOpen(!open)} aria-expanded={open}>
        <ChevronIcon className="chevron" />
        <span>Research details</span>
        <span className="muted details-summary">{headline(d)}</span>
      </button>

      {open && (
        <div className="details-body">
          <div className="details-actions">
            <button className="ghost-button small" onClick={copy}>
              {copied ? <CheckIcon /> : <CopyIcon />}
              {copied ? "Copied" : "Copy details"}
            </button>
          </div>
          {d.version && (
            <p className="muted details-version">
              Version v{d.version.app ?? "?"} · code {d.version.commit}
            </p>
          )}
          {d.ask && (
            <>
              <h3>NotebookLM ask</h3>
              <dl className="stats">
                {askRows(d.ask).map(([label, value, note]) => (
                  <Stat key={label} label={label} value={value} note={note} />
                ))}
              </dl>
            </>
          )}
          {d.ask?.facts && d.ask.facts.length > 0 && (
            <>
              <h3>Fact questions</h3>
              <ol className="detail-queries">
                {d.ask.facts.map((f) => (
                  <li key={f.id}>
                    {f.question}
                    <span className="muted">
                      {" "}
                      — {f.id}, {factNote(f)}, {count(f.passages)} {f.passages === 1 ? "passage" : "passages"}
                    </span>
                  </li>
                ))}
              </ol>
            </>
          )}
          <h3>Retrieval</h3>
          <dl className="stats">
            {!askPath(d) && (
              <>
                <Stat label="Depth" value={d.depth} />
                <Stat label="Searches" value={count(d.searches)} />
                <Stat label="Raw hits" value={count(d.raw_hits)} />
              </>
            )}
            <Stat label="Unique candidates" value={count(d.candidates)} note={chars(d.candidate_chars)} />
            <Stat
              label="Cut-off passages"
              value={count(d.clipped)}
              note={d.clipped ? `${d.continuation_incomplete} incomplete` : undefined}
            />
            <Stat label="Selected passages" value={count(d.selected)} />
            <Stat label="Context requests" value={count(d.context_hits)} note={d.context_reduced ? `${d.context_reduced} reduced` : undefined} />
            {d.repair && (
              <Stat
                label="Evidence repair"
                value={`${d.repair.searches.length} follow-up ${d.repair.searches.length === 1 ? "search" : "searches"}`}
                note={repairNote(d.repair)}
              />
            )}
            <Stat label="Evidence" value={chars(d.evidence_chars)} note={budget != null ? `budget ${count(budget)}` : undefined} />
            <Stat label="Evidence blocks" value={count(d.evidence_blocks)} note={`${d.sources} sources`} />
            <Stat
              label="Fulltexts"
              value={`${d.fulltexts_fetched.length} fetched`}
              note={`${d.fulltexts_from_cache.length} cached`}
            />
            {d.memory && (
              <Stat
                label="Research memory"
                value={memoryText(d.memory)}
                note={writtenText(d.memory.written)}
              />
            )}
            {checkRows(d.checks).map(([label, value]) => (
              <Stat key={label} label="Answer check" value={value} note={label.toLowerCase()} />
            ))}
          </dl>

          {(d.exact_reuse || d.retrieval) && (
            <>
              <h3>Reuse and provenance</h3>
              <dl className="stats">
                <Stat label="Exact reuse" value={exactText(d.exact_reuse)} note={exactNote(d.exact_reuse)} />
                {d.prior_answer !== undefined && <Stat label="Earlier answer" value={priorText(d.prior_answer)} />}
                {d.retrieval && d.retrieval.queries > 0 && (
                  <Stat
                    label={d.exact_reuse?.hit ? "Retrieval (source run)" : "Retrieval"}
                    value={retrievalText(d.retrieval)}
                    note={retrievalNote(d.retrieval)}
                  />
                )}
                <Stat label="Duplicates collapsed" value={collapsedText(d.trace)} />
                <Stat label="Selector" value={SELECTOR_REUSE} />
              </dl>
            </>
          )}

          {d.memory && (d.memory.secondary.length > 0 || (d.memory.secondary_dropped?.length ?? 0) > 0) && (
            <>
              <h3>Community records</h3>
              <ul className="trace-list">
                {d.memory.secondary.map((s) => (
                  <li key={`u-${s.unit_id}`}>
                    <div className="trace-line">
                      <span className="trace-id">{s.id}</span>
                      <span className="trace-fate">used</span>
                      <span className="trace-source">{s.unit_id} · {s.claim_type}, {s.platform}{s.author ? `, ${s.author}` : ""}{checkText(s.attribution_check)}</span>
                    </div>
                    <p className="trace-preview">{s.preview}</p>
                  </li>
                ))}
                {(d.memory.secondary_dropped ?? []).map((s) => (
                  <li key={`d-${s.unit_id}`}>
                    <div className="trace-line">
                      <span className="trace-id">{s.unit_id}</span>
                      <span className="trace-fate">dropped</span>
                      <span className="trace-source">{s.reason}</span>
                    </div>
                    <p className="trace-preview">{s.preview}</p>
                  </li>
                ))}
              </ul>
            </>
          )}

          {d.requirements && d.requirements.length > 0 && (
            <>
              <h3>Answer requirements</h3>
              <ul className="trace-list req-list">
                {d.requirements.map((r) => {
                  const c = coverageById(d.coverage).get(r.id);
                  return (
                    <li key={r.id}>
                      <div className="trace-line">
                        <span className="trace-id">{r.id}</span>
                        <span className="trace-fate">{r.kind}{r.exact ? " · exact" : ""}</span>
                        <CoverageBadge entry={c} />
                        {c && c.hit_ids.length > 0 && <span className="trace-source">{c.hit_ids.join(", ")}</span>}
                      </div>
                      <p className="trace-reason">{r.text}</p>
                      {c?.missing && (
                        <p className="trace-preview">
                          missing: {c.missing}
                          {c.gap && ` (${c.gap}${c.leads?.length ? `; leads ${c.leads.join(", ")}` : ""})`}
                        </p>
                      )}
                    </li>
                  );
                })}
              </ul>
            </>
          )}

          {plannedSearches(result).length > 0 && (
            <>
              <h3>Search queries</h3>
              <ol className="detail-queries">
                {plannedSearches(result).map((s) => (
                  <li key={s.query}>
                    {s.query}
                    {s.covers.length > 0 && <span className="muted"> → {s.covers.join(", ")}</span>}
                  </li>
                ))}
              </ol>
            </>
          )}

          {result.sources.length > 0 && (
            <>
              <h3>Evidence passages</h3>
              <p className="trace-meta">Every passage the answer pass read, by source (the Sources panel lists only the cited ones).</p>
              <SourceList sources={result.sources} />
            </>
          )}

          {d.repair && (
            <>
              <h3>Follow-up searches</h3>
              <ol className="detail-queries">
                {followUps(d.repair).map((f) => {
                  const c = coverageById(d.repair?.coverage).get(f.id);
                  return (
                    <li key={f.search}>
                      {f.search}
                      {f.for && <span className="muted"> → {f.for}</span>}{" "}
                      {d.repair?.coverage && <CoverageBadge entry={c} />}
                      <span className="muted"> — missing premise: {f.premise}</span>
                    </li>
                  );
                })}
              </ol>
            </>
          )}

          {d.evidence_map && d.evidence_map.length > 0 && <EvidenceMap entries={d.evidence_map} synthesis={d.synthesis} />}

          {d.trace && <Trace trace={d.trace} />}

          <h3>Timing</h3>
          <div className="timings">
            {d.stage_seconds.map((s) => (
              <div className="timing" key={s.stage}>
                <span className="timing-label">{s.stage}</span>
                <span className="timing-bar">
                  <span style={{ width: `${Math.max(0, (s.seconds / slowest) * 100)}%` }} />
                </span>
                <span className="timing-value">{seconds(Math.max(0, s.seconds))}</span>
              </div>
            ))}
            <div className="timing total">
              <span className="timing-label">Total</span>
              <span />
              <span className="timing-value">{seconds(d.total_seconds)}</span>
            </div>
          </div>

          <label className="usage-toggle">
            <input type="checkbox" checked={showUsage} onChange={(e) => setShowUsage(e.target.checked)} />
            Show usage
          </label>
          {showUsage && (
          <>
          <h3>Usage by stage</h3>
          <p className="trace-meta">
            Claude {cost(d.claude_cost_usd)} est.{d.exact_reuse?.hit ? " in the source run; $0 for this request" : ""}.
          </p>
          <div className="table-scroll">
            <table className="usage">
              <thead>
                <tr>
                  {USAGE_COLUMNS.map((c, i) => (
                    <th key={c} className={i > 2 ? "num" : undefined}>{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {usageRows(d).map((row, r) => (
                  <tr key={r}>
                    {row.map((cell, i) => (
                      <td key={i} className={i === 1 ? "mono" : i > 2 ? "num" : undefined}>{cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
              <tfoot>
                <tr>
                  <td colSpan={3}>Claude total</td>
                  <td className="num">{count(usage.input)}</td>
                  <td className="num">{count(usage.cacheWrite)}</td>
                  <td className="num">{count(usage.cacheRead)}</td>
                  <td className="num">{count(usage.output)}</td>
                  <td />
                  <td />
                  <td className="num">{cost(d.claude_cost_usd)}</td>
                </tr>
              </tfoot>
            </table>
          </div>
          <p className="trace-meta">
            Claude cost is the CLI's cache-weighted estimate; Gemini cost is not computed. Raw Claude tokens (not
            cost-weighted): {count(usage.total)}.
          </p>
          </>
          )}

          <h3>Logs</h3>
          <div className="log-path">
            <code>{result.run_dir}</code>
            <button
              className="icon-button"
              onClick={() => navigator.clipboard?.writeText(result.run_dir).catch(() => {})}
              aria-label="Copy log folder path"
              title="Copy path"
            >
              <CopyIcon />
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
