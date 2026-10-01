export type Status = "running" | "done" | "error" | "cancelled";

export type StageKey =
  | "auth"
  | "plan"
  | "search"
  | "continuation"
  | "select"
  | "context"
  | "reason"
  | "repair"
  // The coverage check and its follow-up retrieval; the answer from first text to done.
  | "check"
  | "write"
  // An answer call from its start to its first text.
  | "read"
  // A NotebookLM ask (the first, or the one follow-up ask).
  | "ask";

export type ProgressEvent =
  | { type: "stage_start"; stage: StageKey; label: string; at: number }
  | { type: "stage_end"; stage: StageKey; seconds: number; detail?: string | null; at: number }
  | { type: "stage_skip"; stage: StageKey; detail?: string | null; at: number }
  | { type: "plan"; depth: string; searches: string[]; standalone?: string | null; asks?: string[]; at: number }
  // How many of a round's fact questions go to NotebookLM and how many fact memory answers.
  | { type: "ask_plan"; round: number; asked: number; reused: number; at: number }
  // The ask could not be used; the search pipeline runs instead.
  | { type: "fallback"; reason: string; at: number }
  // A follow-up answered from the related turn's evidence (no new search).
  | { type: "reuse"; mode: ReuseMode; passages: number; premise_asks: string[]; at: number }
  // Live only (never stored): the answer's text as the reasoner writes it; reset starts it over.
  | { type: "answer_delta"; stage: string; text: string; reset: boolean; at: number }
  | { type: "end"; status: Status; at?: number };

/** How a follow-up was researched ("answer_from_turn": from the earlier turn's evidence). */
export type ReuseMode = "research" | "answer_from_turn";

/** The ask path's record of a follow-up answered from earlier research. */
export interface ReuseActivity {
  mode: ReuseMode;
  related_turn: string | null;
  passages: number;
  premise_asks: string[];
  follow_up_asks: string[];
}

export interface HistoryItem {
  id: string;
  question: string;
  status: Status;
  created_at: number;
  finished_at: number | null;
  /** The conversation (the first question's id) and the question this one follows up. */
  thread_id: string;
  parent_id: string | null;
  /** The conversation's short sidebar title (null until it is written). */
  title?: string | null;
}

export interface Passage {
  hit_id: string;
  text: string;
  queries: string[];
}

export interface Excerpt {
  source_id: string;
  kind: "passage" | "continuation" | "context";
  label?: string; // the readable source name (a compilation's heading or date included)
  priority: number;
  passages: Passage[];
  continuation: string | null;
  context: string | null;
  incomplete: boolean;
}

export interface Source {
  source_id: string;
  title: string | null;
  name?: string; // readable name ("Q&A, April 12, 2008")
  date?: string | null;
  excerpts: Excerpt[];
}

/** A passage the answer cites, numbered in order of first appearance. */
export interface Citation {
  n: number;
  id: string; // evidence id the answer cites ("h15", "s2")
  kind: "passage" | "community";
  source_id: string | null;
  title: string | null;
  name: string;
  date: string | null;
  text: string;
  preceding: string | null;
  continuation: string | null;
  context: string | null;
}

/** One model stage's usage. Claude rows (provider "claude" or absent) follow Anthropic semantics:
 * input_tokens excludes cache reads and writes, output_tokens includes thinking. Gemini rows
 * carry the Antigravity worker's raw fields (input_tokens, output_tokens, thinking_tokens,
 * cache_read_tokens, total_tokens) and never a cost. */
export interface ClaudeUsage {
  stage: string;
  models: string[];
  usage: Record<string, unknown> | null;
  cost_usd: number | null;
  seconds: number;
  provider?: "claude" | "gemini";
  wall_seconds?: number; // wall-clock time of the call
  /** Set on a Haiku selector row when Gemini was attempted first and failed. */
  fallback?: string;
  /** Set on a Gemini row whose attempt failed (Haiku then handled the stage). */
  failed?: string;
}

export interface StageTiming {
  stage: string;
  seconds: number;
  chars: number | null;
}

/** One thing the question asks the answer to cover, planned before retrieval. */
export interface Requirement {
  id: string;
  kind: string;
  text: string;
  exact?: boolean; // the answer needs exact details (amounts, components)
}

/** The selector's rating of how well the kept evidence covers a requirement (or, in a repair
 * round, a follow-up premise, with the requirement it serves in `for`). */
export interface Coverage {
  requirement_id: string;
  for?: string;
  status: "covered" | "partial" | "missing" | "unassessed";
  hit_ids: string[];
  missing: string;
  gap?: string; // kind of missing detail (quantity, component, ...), when the selector named one
  leads?: string[]; // hits pointing at the missing detail
}

/** A planned search and the requirements it covers. */
export interface PlannedSearch {
  query: string;
  covers: string[];
}

/** A follow-up premise searched in the repair round (p1, p2), for the requirement it serves. */
export interface Premise {
  id: string;
  for: string;
  text: string;
  search: string;
}

/** The optional follow-up search round the reasoner requested for missing premises. */
export interface Repair {
  request: { requirement_id?: string; premise: string; search: string }[];
  premises: (Premise | string)[]; // plain strings in results saved before requirements existed
  coverage?: Coverage[];
  searches: string[];
  raw_hits: number;
  candidates: number;
  selected: number;
  context_hits: number;
  error: string | null;
  forced?: boolean; // started by the pipeline for a missing exact detail, not by the reasoner
}

/** Per-run evidence trace: raw NotebookLM hits, the candidates left after merge/dedupe, and the
 * selector's decision on each candidate. Previews only. */
export interface TraceRaw {
  round: number;
  id: string;
  query: string;
  source: string;
  rank: number | null;
  preview: string;
  candidate: string | null;
  merge: string | null;
  origin?: "fresh" | "cache"; // where the search response came from
  cached_from?: string; // run that retrieved a cached response
  retrieved_at?: string;
}

export interface TraceCandidate {
  id: string;
  raw_ids: string[];
  source: string;
  preview: string;
  continuation: string | null;
  role: "CORE" | "CONTRAST" | "SUPPORT" | "DROP" | null;
  covers?: string[];
  reason: string | null;
  kept: boolean | null;
  context: boolean;
  // Claim structure of kept hits.
  predicate?: string;
  scope?: string;
  applies_to?: string;
  use?: "answer" | "context";
  relations?: { type: string; hit_id: string }[];
}

/** One kept statement in the evidence map. */
export interface MapStatement {
  hit_id: string;
  source: string;
  date: string | null;
  scope: string;
  applies_to: string;
  predicate: string;
  predicate_verbatim: boolean;
  use: "answer" | "context";
}

/** How the kept statements for one requirement relate (built before the reasoner runs). */
export interface MapEntry {
  requirement_id: string;
  statements: MapStatement[];
  relations: { from: string; type: string; to: string; check: string }[];
  changed: string[];
  unlinked_dated: [string, string][];
  predicates: { predicate: string; hit_ids: string[] }[];
  gap: { kind: string; missing: string; leads: string[] } | null;
}

/** How the reasoner says it combined the evidence. */
export interface SynthesisItem {
  hit_ids: string[];
  treatment: string;
  applies_to_user: string;
  qualifies: string;
}

/** Research-memory activity in one run. */
export interface MemoryActivity {
  status: string; // ok | no_database | disabled | error | not_used
  primary: { unit_id: string; hit_id: string; merged_into_fresh: boolean; source: string; score: number }[];
  secondary: { id: string; unit_id: string; claim_type: string; platform: string; author?: string | null; record_id: string; attribution_check?: { question: string; fact?: string; origin?: string; passages?: number } | null; score: number; preview: string }[];
  retained?: { primary: string[]; fresh_with_memory: string[]; secondary: string[] };
  /** Community records the lookup found but did not pass to the answer. */
  secondary_dropped?: { unit_id: string; score: number; key?: string; reason: string; preview: string }[];
  written: { units_new: number; units_seen: number; discoveries: number; selections?: number; relationships?: number; derived?: number };
  errors: { stage: string; error: string }[];
}

/** Exact reuse of a completed result. On a hit the stored result is shown unchanged:
 * its usage and timings are the source run's, and nothing ran for this request. */
export interface ExactReuse {
  hit: boolean;
  key: string;
  reason?: string | null; // why the lookup missed
  source_run?: string;
  source_run_dir?: string;
  completed_at?: string;
  age_hours?: number;
  corpus?: { notebooks: string[]; epoch: string; manifest: string | null };
  corpus_strength?: string;
  policy_match?: boolean;
  record_dir?: string;
  seconds?: number;
}

/** Where each NotebookLM search response came from: a fresh call or the exact-request cache. */
export interface RetrievalProvenance {
  queries: number;
  from_cache: number;
  fresh: number;
  passages_from_cache: number;
  passages_fresh: number;
  detail: {
    round: number;
    query: string;
    origin: "fresh" | "cache";
    passages: number;
    seconds: number;
    miss?: string;
    cached_from?: string;
    retrieved_at?: string;
  }[];
}

/** Related-question reuse (disabled; kept for older results). */
export interface ResearchReuse {
  exact_result: boolean;
  previous_run: string | null;
  previous_question?: string;
  previous_total_seconds?: number;
  requirements: { id: string; from_run: string; from_requirement: string }[];
  primary: { unit_id: string; hit_id: string; from_run: string; merged_into_fresh: boolean; source: string }[];
  searches_skipped: string[];
  searches_fresh: string[];
  selector_reused: string[];
  selector_fresh: string[];
}

export interface AnswerChecks {
  // Grounding: quoted strings not found verbatim in the evidence, and numbers with a
  // unit (amounts, temperatures, durations) not found in it. ungrounded_quantities is its older name.
  grounding_failures?: string[];
  unverified_numbers?: string[];
  ungrounded_percentages: string[];
  ungrounded_quantities?: string[];
  threshold_phrases?: string[];
  unsourced_mechanism_verbs?: string[];
  synthesis_issues?: { issue: string; hit_id: string; [key: string]: unknown }[];
  // Citation grounding
  citation_unknown_ids?: string[];
  citation_misattributed?: { quote: string; cited: string[]; found_in: string[] }[];
  citation_uncited_quotes?: string[];
  citation_in_short_answer?: boolean;
}

export interface PriorAnswer {
  found: boolean;
  reason?: string;
  source_run?: string;
  age_hours?: number | null;
  passages?: number;
  policy_version?: string | null;
}

/** GET /api/version, and a run's details.version. */
export interface BackendVersion {
  /** The app version (VERSION at the repo root); absent on runs recorded before 0.10. */
  app?: string;
  commit: string;
  /** Code and prompt files edited on disk since the backend started (absent before). */
  changed?: string[];
}

export interface Details {
  depth: string;
  // Absent in results saved before answer requirements existed.
  requirements?: Requirement[];
  search_plan?: PlannedSearch[];
  coverage?: Coverage[];
  checks?: AnswerChecks;
  evidence_map?: MapEntry[];
  synthesis?: SynthesisItem[];
  memory?: MemoryActivity;
  reuse?: ResearchReuse; // only
  exact_reuse?: ExactReuse;
  /** The earlier answer to this same question that the run re-checked (a draft for the
   * reasoner; its cited passages offered as evidence), or why there was none. */
  prior_answer?: PriorAnswer | null;
  version?: BackendVersion;
  /** Set for a follow-up; standalone is the rewrite research used. */
  follow_up?: {
    question: string;
    standalone: string;
    previous_question: string;
    previous_run: string;
    previous_passages: number;
    related_turn?: string | null; // the earlier turn the question builds on
    related_question?: string | null;
    turns_offered?: number;
    reuse?: ReuseMode;
  } | null;
  retrieval?: RetrievalProvenance;
  ask?: AskActivity | null; // the NotebookLM ask path (null: not run)
  searches: number;
  raw_hits: number;
  candidates: number;
  candidate_chars: number;
  selected: number;
  context_hits: number;
  repair?: Repair | null; // absent in results saved before evidence repair existed
  evidence_chars: number;
  evidence_budget?: number;
  evidence_target?: [number, number]; // results saved before the budget rename
  context_reduced: number;
  fulltexts_fetched: { source_id: string; title: string | null }[];
  fulltexts_from_cache: { source_id: string; title: string | null }[];
  stage_seconds: StageTiming[];
  total_seconds: number;
  claude_usage: ClaudeUsage[];
  claude_cost_usd: number | null;
  clipped: number;
  continuation_incomplete: number;
  evidence_blocks: number;
  sources: number;
  trace?: { raw: TraceRaw[]; candidates: TraceCandidate[] }; // absent in older results
}

export interface Result {
  question: string;
  answer: string;
  depth: string;
  search_queries: string[];
  sources: Source[];
  citations?: Citation[];
  run_dir: string;
  details: Details;
}

export interface ResearchItem extends HistoryItem {
  answer: string | null;
  error: string | null;
  result: Result | null;
  events: ProgressEvent[] | null;
}

export type ConnectionKey = "notebooklm" | "claude" | "gemini";

/** One provider connection. reauth_required: the saved sign-in expired or was rejected (by a
 * status check or a real call); error: a non-authentication failure (network, missing CLI). */
export interface Connection {
  key: ConnectionKey;
  name: string;
  about?: string;
  required?: boolean; // false for optional providers (Gemini); research needs only required ones
  state: "unknown" | "checking" | "connected" | "disconnected" | "connecting" | "reauth_required" | "error";
  account: string | null;
  detail: string | null;
  checked_at: number | null;
  disabled?: boolean; // disconnected in this app; the computer's sign-in is kept
  console_login?: boolean; // the sign-in runs in its own window (Antigravity on Windows)
  login_url: string | null;
  output: string[];
}

/** One planned fact question (asked of NotebookLM, reused from fact memory, or failed). */
export interface FactQuestion {
  id: string;
  question: string;
  round: number;
  origin: "asked" | "reused" | "failed";
  batch: number | null;
  citations: number;
  passages: number;
  ledger_id: number | null;
  reused_from?: string | null;
  error: string | null;
}

/** The NotebookLM ask path of a run (fact questions in numbered batches). */
export interface AskActivity {
  seconds: number;
  asks: {
    n: number;
    seconds: number;
    cache: boolean;
    citations?: number;
    error?: string | null;
    questions?: string[]; // the fact question ids in this ask
  }[];
  citations: number;
  passages_recovered: number;
  recovery: Record<string, number>;
  passages_from_memory: number;
  passages_from_ledger?: number;
  follow_up: {
    questions: string[];
    citations: number;
    new_passages: number;
    reused?: number;
    error: string | null;
  } | null;
  cache_hit: boolean;
  fallback: string | null;
  facts?: FactQuestion[];
  ledger?: { status: string; offered: number; matched: number; written: number };
  reuse?: ReuseActivity | null;
}
