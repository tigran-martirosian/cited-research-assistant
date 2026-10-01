"""The research pipeline: answers one question against the source library.

Planner, retrieval through NotebookLM, evidence assembly, the cited answer and the Python checks
all live here. See docs/architecture.md for the flow. Entry point: research(question, ...).
Used by ask.py and server/."""
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import settings

ROOT = Path(__file__).resolve().parent
PROMPTS_DIR = ROOT / "prompts"  # stage system prompts, loaded below
# The NotebookLM notebook that holds the source library (CRA_NOTEBOOK_ID); live runs need it.
NOTEBOOK = settings.env("CRA_NOTEBOOK_ID")

# Model names are passed to `claude --model`: an alias ("haiku", "opus") or a full model id.
FAST = (settings.env("CRA_FAST_MODEL", "haiku"), "low")
# Model mix: every call shares the Claude subscription's usage limit, so each stage uses the
# smallest model that held quality on the evals. The planner and both answer depths use Opus,
# chosen after trying Sonnet on a private library.
PLANNER_MODEL = settings.env("CRA_PLANNER_MODEL", "opus")
COVERAGE_MODEL = settings.env("CRA_COVERAGE_MODEL", "haiku")  # the requested-parts coverage pre-check (see coverage_check)
ANSWER_MODEL = settings.env("CRA_ANSWER_MODEL", "opus")  # normal and deep depth
QUICK_ANSWER_MODEL = settings.env("CRA_QUICK_ANSWER_MODEL", "opus")  # quick ("direct") depth
PLANNER = (PLANNER_MODEL, "medium")
SELECTOR = FAST
COVERAGE = (COVERAGE_MODEL, None)
# Reasoner effort by research depth: deep questions get more thinking.
REASONER = {"direct": (QUICK_ANSWER_MODEL, "medium"), "normal": (ANSWER_MODEL, "medium"),
            "deep": (ANSWER_MODEL, "high")}
# Cap on the answer pass's extended thinking (MAX_THINKING_TOKENS for its claude -p calls).
# Thinking stays on; uncapped it ran to ~4k tokens before the first answer character. The
# planner and coverage check already did the structuring, so a small budget is enough.
ANSWER_THINKING_BUDGET = 1024
# No stage waits longer than this on a failing dependency. Non-streaming Claude stages
# (planner, coverage check) are killed after it; a streaming answer pass after this long without
# any stream event (a stall; a live stream sends thinking and text deltas continuously).
STAGE_TIMEOUT = 30
# Output ceiling for the selector. Haiku writes a short analysis (~500-650 tokens for ~26
# candidates) before its structured-output call; a ceiling below that stops the first request at
# max_tokens and makes the CLI send a second request that re-reads the whole prompt. Keep headroom
# for deep runs with ~48 candidates; each kept hit also carries its claim structure (predicate,
# scope, relations), which roughly doubles the output of a large selection.
SELECTOR_MAX_OUTPUT_TOKENS = "8192"

CLAUDE = settings.env("CRA_CLAUDE_CMD") or shutil.which("claude") or "claude"
NOTEBOOKLM = settings.env_command("CRA_NOTEBOOKLM_CMD", [
    shutil.which("uv") or "uv", "run", "--no-project", "--quiet", "--with", "notebooklm-py",
    "notebooklm"])
# Claude stages run outside the project so CLAUDE.md is not auto-loaded; the reasoning rules
# each stage needs are in its own system prompt in prompts/.
STAGE_CWD = tempfile.gettempdir()
ENV = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}  # use the subscription login
# Seconds; hard limit for one NotebookLM call (no automatic retry), so no
# stage waits longer than that on a failing NotebookLM; searches take ~3-7s.
NOTEBOOKLM_TIMEOUT = 30
# Total searches per depth (component, vocabulary and community searches come on top of
# the planner prompt's literal/qualifier maximums of 2 / 4 / 6).
DEPTH_MAX_SEARCHES = {"direct": 5, "normal": 8, "deep": 12}
SEARCH_TYPES = ["literal", "qualifier", "component", "vocabulary", "community"]
TYPE_MAX_SEARCHES = {"component": 6, "vocabulary": 4}  # per question (4 with formula aliases)
# A follow-up query whose normalized terms overlap an earlier query's this much is a repeat.
QUERY_OVERLAP_MAX = 0.7
# Concurrent NotebookLM calls (searches, fulltext fetches); Claude calls stay serial. The
# search cap, so a planned round of searches runs in one wave.
WORKERS = max(DEPTH_MAX_SEARCHES.values())
MAX_SEARCHES = max(DEPTH_MAX_SEARCHES.values())
SEARCH_LIMIT = 8  # passages per search
# Top hits kept per NotebookLM search: a candidate is kept when some search ranked it
# within this many; the rest never reach continuation or the evidence. Tuned on a private
# library with tools/trim_sim.py; it equals SEARCH_LIMIT.
PER_QUERY_KEEP = 8
# Soft upper evidence-size budgets (chars, after context expansion). A budget is a maximum, never
# a minimum or a desired size, and never a truncation limit: when evidence runs far past it, the
# context windows of the lowest-priority context hits shrink first; no selected passage is ever
# dropped. Without the runtime selector the budget also caps the candidates: over it,
# they are trimmed deterministically (see trim_candidates).
# Tuned offline with tools/trim_sim.py on a private library, as the lowest budgets that keep
# the required eval facts in the evidence.
EVIDENCE_BUDGET_NORMAL = 43_000
EVIDENCE_BUDGET_DEEP = 57_000
EVIDENCE_BUDGET_DIRECT = 20_000  # the quick depth
EVIDENCE_BUDGET = {"direct": EVIDENCE_BUDGET_DIRECT, "normal": EVIDENCE_BUDGET_NORMAL,
                   "deep": EVIDENCE_BUDGET_DEEP}
FAR_OVER_BUDGET = 1.25  # shrink context only when evidence exceeds the budget by this factor
CONTEXT_WINDOW = (600, 1000)  # original text kept before / after a context hit
REDUCED_WINDOW = (250, 400)  # the same, for context hits shrunk to stay near the budget
ALIGN_SLACK = 300  # how far a window edge may move outward to reach a line/sentence boundary
QA_SLACK = 1200  # how far a window may reach to complete a clipped question/answer pair
MERGE_GAP = 200  # windows of one source closer than this are merged into one block
# Forward continuation of a search hit whose chunk stops before its answer: the immediate
# answer runs to the next speaker question or section boundary; past CONTINUATION_SOFT answer
# chars it stops at the next line end, and CONTINUATION_MAX is a last-resort cap.
CONTINUATION_SOFT = 2500
CONTINUATION_MAX = 6000
CONTINUATION_MAX_FETCHES = 4  # fulltexts fetched (not cached) for continuation per round
# Selector bypass: when every candidate (with its continuation) fits the depth's
# evidence budget, the reasoner gets them all, unjudged, and the selector does not run.
BYPASS_HIT_OVERHEAD = 150  # chars per hit for its PASSAGE/SOURCE/provenance header lines
# The runtime selector is switched off. Candidates over the budget are trimmed instead (see
# trim_candidates); the selector code stays intact behind this flag.
RUNTIME_SELECTOR_ENABLED = False
# Backward continuation: a chunk that starts mid-list gets the source text before it,
# back to its recipe/section heading, at most this many chars.
BACKWARD_MAX = 1500
# Runtime Gemini selector: one attempt is capped, and a network/timeout failure opens a
# process-wide circuit breaker so later attempts go straight to the Haiku fallback.
GEMINI_SELECTOR_TIMEOUT_SECONDS = 30
GEMINI_COOLDOWN_SECONDS = 600
# Research memory (research_memory.py): primary passages earlier runs retrieved, and community
# (secondary) claims. It supplements NotebookLM and never replaces a search. None: the memory
# module's default ($CRA_MEMORY_DB or data/research-memory.db); CRA_MEMORY=off disables it.
MEMORY_DB = None
MEMORY_PRIMARY_MAX = 4  # remembered primary passages added as candidates per run
MEMORY_CITED_MAX = 6  # Passages earlier answers cited for matching questions, added too
MEMORY_SECONDARY_MAX = 5  # top community claims from the memory lookup, shown to the reasoner
MEMORY_COMMUNITY_MAX = 3  # community claims per community-facet search key; all of them are shown
# Corpus identity recorded in memory's run metadata; only the disabled
# established-requirement reuse reads it. Reuse uses corpus_identity() (CORPUS_EPOCH).
CORPUS_VERSION = os.environ.get("CRA_CORPUS_VERSION") or f"{NOTEBOOK}:1"
RESULT_FILE = "result.json"  # the completed result, saved in the run folder for exact reuse
# Related-question reuse (established requirements: skipped searches, reused selector
# judgments and coverage). Off: fuzzy requirement matching must never suppress retrieval, and a
# selector judgment depends on the question and the whole candidate set. Code and data are kept.
LEGACY_RELATED_REUSE_ENABLED = False
# ---- reuse -------------------------------------------------------------------------------
# Exact reuse serves a completed result only for the same identity: normalized question, answer-
# affecting options, corpus identity and pipeline policy. The retrieval cache serves a stored
# NotebookLM response only for the identical request on the same corpus identity. Index entries
# point into run history (logs/<run>/result.json) and research memory (primary units).
REUSE_DIR = settings.DATA_DIR / "reuse"  # results/<key>.json, retrieval/<key>.json
# A stored result is never served as the answer (the pipeline changes between runs and
# the answer would not show it). The index still points at the latest run of each question, and
# that run's answer and cited passages feed the new run as a draft to re-check (prior_answer).
EXACT_REUSE_SERVE = False
# Bump after changing the notebook's sources: every exact and retrieval entry then misses.
CORPUS_EPOCH = os.environ.get("CRA_CORPUS_EPOCH", "").strip() or "0"
# No local source manifest exists, so corpus identity is "weak": entries older than this miss.
REUSE_MAX_AGE_HOURS_WEAK_CORPUS = float(os.environ.get("CRA_REUSE_MAX_AGE_HOURS") or 168)
# Bump for pipeline behavior that the prompt templates, schemas and models do not capture; a
# new value invalidates cached results, retrieval entries and ledger entries.
PIPELINE_POLICY_VERSION = "22"

# The app version shown in the UI and recorded on each run: the one line of VERSION at the repo
# root, which the frontend build also reads. Bump it for each release. PIPELINE_POLICY_VERSION
# above is internal (cache and ledger invalidation) and is not shown.
APP_VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip()


def code_commit():
    """The git short commit of the code this process loaded, or "unknown"."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# Read once at import, so it names the code the running app started with (the web app
# compares it with the commit its frontend was built from; Research Details shows both).
CODE_COMMIT = code_commit()


def code_files_digest():
    """{relative path: sha256} of the backend's code and prompt files as they are on disk now
   . Prompts are read once at import, so an edit takes effect only after a restart."""
    out = {}
    files = [*PROMPTS_DIR.glob("*.txt"), *ROOT.glob("*.py"), *(ROOT / "server").glob("*.py")]
    for p in sorted(files):
        try:
            out[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError:
            pass
    return out


LOADED_CODE = code_files_digest()  # What this process started with


def code_changed_since_start():
    """The code and prompt files that differ on disk from what this process loaded:
    committed or not, the running app does not use them until it is restarted."""
    now = code_files_digest()
    return sorted(k for k in LOADED_CODE.keys() | now.keys() if LOADED_CODE.get(k) != now.get(k))
RETRIEVAL_PARSE_VERSION = "1"  # bump when search() starts reading different response fields
# Primary memory recorded before discoveries carried a corpus identity came from this notebook,
# before epochs existed: it counts as epoch "0".
LEGACY_CORPUS = {"notebooks": [NOTEBOOK], "epoch": "0",
                 "manifest": None}
SECONDARY_CLAIM_CHARS = 600
# A record's method is often past its first few hundred characters, so the wording shown to
# the answer model is long enough to include its temperatures and timings.
SECONDARY_WORDING_CHARS = 2400
CACHE_DIR = settings.CACHE_DIR / "fulltext"  # <source_id>.txt: NotebookLM fulltext, keyed by source id
TITLES_FILE = settings.CACHE_DIR / "source-titles.json"  # source id -> title
TICK = 5  # seconds between wait-timer updates
POLL = 0.25  # seconds between child-process checks, so Ctrl+C is handled promptly
STREAM_FLUSH = 0.1  # seconds: streamed answer text is passed on at most this often
# Exit codes of a child killed by Ctrl+C (Windows STATUS_CONTROL_C_EXIT, POSIX SIGINT).
CTRL_C_EXIT = (0xC000013A, -2)
CANCELLED = "cancelled"
STOPPED = "stopped"  # a streamed call ended early on purpose (stream_child)

PLANNER_SYSTEM = (PROMPTS_DIR / "planner.txt").read_text(encoding="utf-8")

# Answer requirements: what the question itself asks the answer to cover, planned before any
# search (a simple question has one; a compound one has one per thing it asks). Every search
# names the requirements it covers; the selector reports coverage per requirement and the
# reasoner receives both, so a requirement the evidence leaves open is visible, not silent.
REQUIREMENT_KINDS = ["claim", "purpose", "procedure", "composition", "quantity", "applicability",
                     "chronology", "comparison"]
MAX_REQUIREMENTS = 8
ASKS_MAX = 8  # The planner's fact questions for NotebookLM (the prompt asks for 3-8)
VERIFY_MAX = 2  # The planner's verification questions for community attributions
# How a follow-up is researched. "answer_from_turn": the question is an
# inference from what the related turn already retrieved, so its passages (cited and uncited)
# go straight to the answer pass, with at most ANSWER_FROM_TURN_ASKS ask for one missing premise;
# "research": the normal ask path.
REUSE_MODES = ["research", "answer_from_turn"]
ANSWER_FROM_TURN = "answer_from_turn"
ANSWER_FROM_TURN_ASKS = 1
PLANNER_COMMUNITY_POOL = 15  # community records shown to the planner, which picks the used ones
COMMUNITY_PICK_MAX = 5  # records the planner may pick for the answer (verified ones always kept)

PLANNER_SCHEMA = {
    "type": "object",
    "properties": {
        # The earlier turn (t1..t3 of EARLIER TURNS) the question builds on, or null.
        "related_turn": {"type": ["string", "null"]},
        # The question rewritten to stand alone when it builds on an earlier turn;
        # otherwise the question exactly as written. Research uses it (see plan()).
        "standalone": {"type": "string"},
        # Answer a follow-up from the related turn's evidence, or research it anew.
        "reuse": {"type": "string", "enum": REUSE_MODES},
        "requirements": {"type": "array", "minItems": 1, "maxItems": MAX_REQUIREMENTS, "items": {
            "type": "object",
            "properties": {"id": {"type": "string"},
                           "kind": {"type": "string", "enum": REQUIREMENT_KINDS},
                           "text": {"type": "string"},
                           "exact": {"type": "boolean"}},
            "required": ["id", "kind", "text", "exact"]}},
        "depth": {"type": "string", "enum": list(DEPTH_MAX_SEARCHES)},
        "searches": {"type": "array", "minItems": 1, "maxItems": MAX_SEARCHES, "items": {
            "type": "object",
            "properties": {"query": {"type": "string"},
                           "covers": {"type": "array", "items": {"type": "string"}},
                           "type": {"type": "string", "enum": SEARCH_TYPES}},
            "required": ["query", "covers", "type"]}},
        # Precise fact questions for NotebookLM, the main retrieval (the searches above
        # are the fallback's). Empty (or one missing premise) with reuse answer_from_turn.
        "asks": {"type": "array", "maxItems": ASKS_MAX, "items": {"type": "string"}},
        # Self-contained questions (and a source-search string) that check what a
        # COMMUNITY RECORD attributes to the corpus author; asked with the asks, searched in the fallback.
        "verifications": {"type": "array", "maxItems": VERIFY_MAX, "items": {
            "type": "object",
            "properties": {"record": {"type": "string"}, "question": {"type": "string"},
                           "search": {"type": "string"}},
            "required": ["record", "question", "search"]}},
        # The COMMUNITY RECORDS (s1..) the answer uses; the others are left out.
        "community": {"type": "array", "maxItems": COMMUNITY_PICK_MAX, "items": {"type": "string"}},
        # Parts the user explicitly asks about individually (the reasoner's COVERAGE
        # line has one entry per part; a "no" drives the coverage follow-up). Empty otherwise.
        "requested_parts": {"type": "array", "maxItems": 8, "items": {
            "type": "object",
            "properties": {"part": {"type": "string"}, "per_component_of": {"type": "string"},
                           "attribute": {"type": "string"}},
            "required": ["part", "per_component_of", "attribute"]}},
        # Last, so it is generated after all search planning; used only for the research-need
        # signature (research_need.py), never for retrieval.
        "case_frame": {"type": "object", "properties": {
            "subject": {"type": "string"},
            "qualifiers": {"type": "array", "items": {"type": "string"}},
            "processing_action": {"type": "string"},
            "purpose": {"type": "string", "enum": ["source_statement", "practical_applicability",
                                                   "procedure", "mechanism", "comparison", "other"]},
            "context": {"type": "string"}},
            "required": ["subject", "qualifiers", "processing_action", "purpose", "context"]},
    },
    "required": ["related_turn", "standalone", "reuse", "requirements", "depth", "searches", "asks",
                 "verifications", "community", "requested_parts", "case_frame"],
}
# The planner writes its plan as plain JSON text against this schema (one model turn),
# not through the CLI's --json-schema tool, which costs a second turn; Sonnet also nested the first array
# ({"requirements": {"requirements": [...]}}), the CLI rejected it and the whole plan was
# generated again (+4-10s in 6 of 12 runs). Such nesting is unwrapped in Python instead; an
# unreadable plan falls back to one --json-schema call.
PLANNER_TEXT_SYSTEM = (
    PLANNER_SYSTEM + "\n\n<json_schema>\nWrite the plan as one JSON object and nothing else: no "
    "prose and no code fence. It must validate against this JSON Schema. Keep the property order "
    "related_turn, standalone, reuse, requirements, depth, searches, asks, verifications, community, "
    "requested_parts, case_frame, "
    "each "
    "directly at the "
    "top level "
    "of the object:\n" + json.dumps(PLANNER_SCHEMA) + "\n</json_schema>")

SELECTOR_SYSTEM = (PROMPTS_DIR / "selector.txt").read_text(encoding="utf-8")

# Evidence roles the selector assigns to every candidate (shown in Research Details):
# CORE is needed to establish or materially qualify an answer requirement; CONTRAST is an
# opposite condition or direction that materially changes interpretation; SUPPORT is needed to
# understand a CORE passage; DROP is related but unnecessary. DROP hits never reach the reasoner.
# "covers" names the requirements a kept hit serves; "coverage" rates every requirement.
SELECTOR_ROLES = ["CORE", "CONTRAST", "SUPPORT", "DROP"]
COVERAGE_STATUSES = ["covered", "partial", "missing"]
# Claim structure of every kept hit, so the relationships between statements survive to the
# reasoner instead of being flattened into one list of passages:
#   predicate  the source's own verb phrase for what the passage claims (kept verbatim, because
#              "requires", "creates demand for", "causes deficiency" and "pulls out" differ);
#   scope      who or when it applies to: everyone (general), a condition, a stage, one person or
#              case (individual), or unclear; applies_to names the condition/stage/case;
#   relations  how it relates to another kept hit (see RELATION_TYPES);
#   use        "answer" when the answer needs the passage itself, "context" when it only informs
#              how another passage is read and need not appear in the answer.
SCOPES = ["general", "condition", "stage", "individual", "unclear"]
# restates: same proposition, same scope. updates: a later statement that changes the same
# proposition (only with dates or an explicit correction). adds_to: adds a component or dimension.
# narrows: a condition/stage/case-specific version of it. qualifies: an exception or limit to that
# one statement only. contradicts: incompatible claim about the same proposition under comparable
# conditions. compatible: a different mechanism or predicate that can hold alongside it.
# context_for: needed only to interpret it.
RELATION_TYPES = ["restates", "updates", "adds_to", "narrows", "qualifies", "contradicts",
                  "compatible", "context_for"]
USES = ["answer", "context"]
# What a partial or missing requirement lacks. lead_hit_ids name kept or candidate hits that point
# at the missing detail (for example a component mentioned without its amount): such a gap is a
# candidate for targeted repair, never for an invented value.
GAP_KINDS = ["none", "quantity", "component", "condition", "chronology", "referent", "mechanism",
             "other"]
SELECTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {"type": "array", "items": {
            "type": "object",
            "properties": {"hit_id": {"type": "string"},
                           "role": {"type": "string", "enum": SELECTOR_ROLES},
                           "covers": {"type": "array", "items": {"type": "string"}},
                           "reason": {"type": "string"}},
            "required": ["hit_id", "role", "covers", "reason"]}},
        # One entry per kept hit only, so dropped hits cost no extra output.
        "claims": {"type": "array", "items": {
            "type": "object",
            "properties": {"hit_id": {"type": "string"},
                           "predicate": {"type": "string"},
                           "scope": {"type": "string", "enum": SCOPES},
                           "applies_to": {"type": "string"},
                           # "type:hit_id", e.g. "qualifies:h3" (flat strings keep the
                           # schema simple enough for the fast selector model)
                           "relations": {"type": "array", "items": {"type": "string"}},
                           "use": {"type": "string", "enum": USES}},
            "required": ["hit_id", "predicate", "scope", "applies_to", "relations", "use"]}},
        "coverage": {"type": "array", "items": {
            "type": "object",
            "properties": {"requirement_id": {"type": "string"},
                           "status": {"type": "string", "enum": COVERAGE_STATUSES},
                           "hit_ids": {"type": "array", "items": {"type": "string"}},
                           "missing": {"type": "string"},
                           "gap": {"type": "string", "enum": GAP_KINDS},
                           "lead_hit_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["requirement_id", "status", "hit_ids", "missing", "gap",
                         "lead_hit_ids"]}},
        "selected_hit_ids": {"type": "array", "items": {"type": "string"}},
        "context_hit_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["decisions", "claims", "coverage", "selected_hit_ids", "context_hit_ids"],
}

REASONER_SYSTEM = (PROMPTS_DIR / "reasoner.txt").read_text(encoding="utf-8")
# The requested-parts coverage pre-check (COVERAGE_MODEL) before the answer pass.
COVERAGE_SYSTEM = (PROMPTS_DIR / "coverage.txt").read_text(encoding="utf-8")
# The ask-ledger match (COVERAGE_MODEL): planned fact questions against earlier ones.
LEDGER_SYSTEM = (PROMPTS_DIR / "ledger.txt").read_text(encoding="utf-8")

# The reasoner states how it combined the evidence before writing the answer: each item names
# kept hits, how the answer treats them, whether they apply to the user, and for an exception the
# one statement it qualifies. Python checks this against the evidence map (see check_synthesis).
TREATMENTS = ["base", "update", "addition", "condition", "stage", "individual_example",
              "mechanism", "exception", "contrast", "conflict", "context_only", "not_used"]
APPLIES = ["yes", "no", "partly", "unknown"]
SYNTHESIS_SCHEMA = {"type": "array", "items": {
    "type": "object",
    "properties": {"hit_ids": {"type": "array", "items": {"type": "string"}},
                   "treatment": {"type": "string", "enum": TREATMENTS},
                   "applies_to_user": {"type": "string", "enum": APPLIES},
                   "qualifies": {"type": "string"}},
    "required": ["hit_ids", "treatment", "applies_to_user", "qualifies"]}}

# Answer first. A short decision field, then (first call only) the follow-up request,
# then the answer, then the synthesis bookkeeping, so the streamed answer starts right after the
# decision instead of after the synthesis.
REASONER_SCHEMA = {
    "type": "object",
    "properties": {"decision": {"type": "string", "enum": ["answer"]},
                   "answer": {"type": "string"}, "synthesis": SYNTHESIS_SCHEMA},
    "required": ["decision", "answer", "synthesis"],
}

# Evidence repair: the first reasoner call may, instead of answering, ask for one round of
# follow-up searches for missing premises. Then only the new hits go through continuation,
# selection and context, and the reasoner answers from the augmented evidence (no further repair).
REPAIR_MAX_SEARCHES = 4  # follow-up searches per repair round
REPAIRABLE_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["answer", "search"]},
        "research_request": {
            "type": "array", "maxItems": REPAIR_MAX_SEARCHES,
            "items": {"type": "object",
                      "properties": {"requirement_id": {"type": "string"},
                                     "premise": {"type": "string"},
                                     "search": {"type": "string"}},
                      "required": ["requirement_id", "premise", "search"]},
        },
        "answer": {"type": "string"},
        "synthesis": SYNTHESIS_SCHEMA,
    },
    "required": ["decision", "answer", "synthesis"],
}


class ResearchError(Exception):
    """The research failed. run_dir holds its logs (None when it failed before they existed).

    provider ("claude", "notebooklm" or None) is the service whose call failed, and kind how:
    "auth" (sign-in missing, expired or rejected: the connection needs reauthentication),
    "network" (unreachable, timed out, rate-limited or overloaded), "provider" (any other
    failure of that service), or None when no provider was involved."""

    def __init__(self, message, run_dir=None, provider=None, kind=None):
        super().__init__(message)
        self.run_dir = run_dir
        self.provider = provider
        self.kind = kind


# Failure classification for provider calls. Only unmistakable sign-in failures count as "auth";
# network trouble, rate limits, overload and model errors never do, so a flaky connection does
# not send the user to sign in again.
AUTH_FAILURE = re.compile(
    r"\b(?:not (?:logged|signed) in|log ?in again|please (?:run )?/login|run /login|sign in again|"
    r"reauthenticat\w*|re-authenticat\w*|authentication (?:expired|failed|required|error)|"
    r"auth_error|authentication_error|unauthenticated|unauthori[sz]ed|invalid (?:api key|token|"
    r"credentials?|grant)|oauth token (?:has )?(?:expired|revoked)|token (?:has )?(?:expired|been "
    r"revoked)|credentials? (?:expired|revoked|invalid|missing)|login (?:expired|required)|"
    r"storage file not found|no (?:saved |stored )?credentials|no active session|"
    r"(?:please|must|need to) (?:sign|log) ?in|sign-?in (?:is )?required|401\b)", re.I)
NETWORK_FAILURE = re.compile(
    r"\b(?:network|connection (?:error|refused|reset|timed out)|timed? ?out|timeout|getaddrinfo|"
    r"name resolution|dns|econn\w+|enotfound|etimedout|temporarily unavailable|unavailable|"
    r"overloaded|rate.?limit\w*|429|5\d\d|resource.?exhausted|deadline.?exceeded|quota)\b", re.I)


def notebooklm_auth_status(returncode, stdout, stderr=""):
    """Read `notebooklm auth check --test --json`; return (status, account email, detail).

    status is "connected"; "signed_out" (no saved session); "auth_failed" (a saved session the
    service rejects or that has expired); or "error" (the check could not tell, for example the
    network is down). Used by the pipeline's preflight and by the Connections status check."""
    try:
        out = json.loads(stdout)
    except (TypeError, ValueError):
        out = None
    out = out if isinstance(out, dict) else {}
    account = out.get("account") if isinstance(out.get("account"), dict) else {}
    email = account.get("email")
    if returncode == 0 and out.get("status") == "ok":
        return "connected", email, None
    checks = out.get("checks") if isinstance(out.get("checks"), dict) else {}
    details = out.get("details") if isinstance(out.get("details"), dict) else {}
    error = clean(details.get("error") or out.get("message") or (stderr or stdout or "")[-300:], 300)
    if checks.get("storage_exists") is False:
        return "signed_out", email, "Not signed in to NotebookLM."
    kind = classify_failure(error)
    if kind == "network":
        return "error", email, f"NotebookLM could not be reached: {error}"
    if kind == "auth" or checks.get("token_fetch") is False or False in checks.values():
        return "auth_failed", email, error or "The NotebookLM sign-in is no longer valid."
    return "error", email, error or "The NotebookLM status check failed."


# ---- preflight -----------------------------------------------------------------------
# One command logs in for the pipeline, the web app and the terminal, with the same package.
NOTEBOOKLM_LOGIN_COMMAND = settings.env(
    "CRA_NOTEBOOKLM_LOGIN_CMD",
    'uvx --from "notebooklm-py[browser]" notebooklm login --browser chrome')
PREFLIGHT_CACHE_SECONDS = 60  # a successful preflight is reused for this long
PREFLIGHT_TIMEOUT = 8  # hard cap per check; the NotebookLM token fetch normally takes ~2s
_PREFLIGHT = {"at": None, "result": None}


def preflight_check(cmd, timeout=PREFLIGHT_TIMEOUT):
    """(returncode, stdout, stderr) of one quick status command; returncode None on a timeout
    or a missing executable (stderr says which)."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=timeout, env=ENV, cwd=STAGE_CWD)
        return out.returncode, out.stdout, out.stderr
    except subprocess.TimeoutExpired:
        return None, "", f"timed out after {timeout}s"
    except OSError as e:
        return None, "", str(e)


def preflight_result(notebooklm_out, claude_out):
    """The preflight verdict (pure) from the NotebookLM `auth check --test --json` and the
    `claude --version` outputs ((returncode, stdout, stderr) each): {"ok", "message",
    "provider", "kind", "notebooklm", "claude"}. A failure message says what to run."""
    state, email, detail = notebooklm_auth_status(*notebooklm_out)
    rc, out, err = claude_out
    claude_ok = rc == 0
    result = {"ok": False, "provider": None, "kind": None, "message": None,
              "notebooklm": {"state": state, "account": email, "detail": detail},
              "claude": {"ok": claude_ok, "version": out.strip() if claude_ok else None,
                         "detail": None if claude_ok else (err or out).strip()[-300:]}}
    if state in ("signed_out", "auth_failed"):
        what = "NotebookLM is not logged in" if state == "signed_out" else "NotebookLM login expired"
        result.update(provider="notebooklm", kind="auth",
                      message=f"{what} — run: {NOTEBOOKLM_LOGIN_COMMAND}")
    elif state != "connected":
        result.update(provider="notebooklm", kind=classify_failure(detail) if detail else "provider",
                      message=f"NotebookLM login check failed: {detail or 'unknown error'}")
    elif not claude_ok:
        result.update(provider="claude", kind="provider",
                      message=f"The claude CLI is not callable ({result['claude']['detail']}) — "
                              "install Claude Code or fix PATH, then run: claude /login")
    else:
        result["ok"] = True
    return result


def preflight(force=False, now=None, checker=preflight_check):
    """Preflight before planning: a NotebookLM token fetch and `claude --version` (no model
    call), run concurrently. A success is cached for PREFLIGHT_CACHE_SECONDS; a failure is
    never cached, so a fresh login takes effect on the next question."""
    if not NOTEBOOK:
        detail = "CRA_NOTEBOOK_ID is not set"
        return {"ok": False, "provider": "notebooklm", "kind": "provider", "cached": False,
                "message": f"{detail} — set it to your NotebookLM notebook id (see .env.example)",
                "notebooklm": {"state": "error", "account": None, "detail": detail},
                "claude": {"ok": False, "version": None, "detail": "not checked"}}
    now = time.monotonic() if now is None else now
    cached = _PREFLIGHT["result"]
    if (not force and cached and cached["ok"]
            and now - _PREFLIGHT["at"] <= PREFLIGHT_CACHE_SECONDS):
        return {**cached, "cached": True}
    with ThreadPoolExecutor(max_workers=2) as pool:
        nlm = pool.submit(checker, NOTEBOOKLM + ["auth", "check", "--test", "--json"])
        cli = pool.submit(checker, [CLAUDE, "--version"])
        result = preflight_result(nlm.result(), cli.result())
    _PREFLIGHT.update(at=now, result=result)
    return {**result, "cached": False}


def classify_failure(text):
    """"auth", "network" or "provider" for a provider's error text (see AUTH_FAILURE)."""
    text = str(text or "")
    if NETWORK_FAILURE.search(text) and not re.search(r"\b(?:401|unauthenticated|not (?:logged|signed) in)\b", text, re.I):
        return "network"
    if AUTH_FAILURE.search(text):
        return "auth"
    return "provider"


class ResearchCancelled(Exception):
    def __init__(self, run_dir=None):
        super().__init__("Research cancelled.")
        self.run_dir = run_dir


class Run:
    """State of one question's research: its log directory, stage timings, Claude usage, the
    fulltexts loaded so far, running children and cancellation.

    Progress goes to on_event(dict). Event types: stage_start / stage_end / stage_skip (stage
    keys auth, plan, search, continuation, select, context, reason, repair; Adds check,
    the coverage check and its follow-up retrieval, and write, from the answer's first text to
    completion; Adds read, from an answer call's start to its first text), plan (depth,
    searches, requirements, standalone),
    waiting / waiting_end (NotebookLM wait timer), answer_delta (the reasoner's answer text as it
    streams: stage, text, reset), answer.
    """

    def __init__(self, question, on_event=None, cancel=None, fresh=False, previous=None):
        self.question = question
        # Follow-up: the previous exchange (see previous_exchange), or None; and the
        # question as research uses it (the planner's standalone rewrite of a follow-up).
        self.previous = previous
        self.standalone = question
        # The earlier turn the planner says this question builds on (see related_turn)
        self.related = None
        # The planner's reuse decision, "research" or ANSWER_FROM_TURN (normalize_reuse)
        self.reuse = "research"
        # The latest earlier answer to this same question, re-checked by this run (see
        # prior_answer), or None
        self.prior = None
        self.on_event = on_event
        self.cancel = cancel or threading.Event()
        # Fresh run: no exact reuse and no retrieval cache reads (entries are still written).
        self.fresh = fresh
        # Primary units the retrieval cache created in this run, which capture_primary still
        # counts as new.
        self.cache_units = set()
        self.dir = None
        self.usage = []  # one entry per Claude stage, for the run summary
        self.times = []  # (stage label, seconds, chars or None), measured by Python wall clock
        self.output_wait = 0.0  # seconds spent reporting progress between stages
        # Fulltexts loaded during this run, shared by continuation recovery and context: source id
        # -> content, source ids whose fetch failed (not retried), and locate()'s per-source data.
        self.sources = {"texts": {}, "failed": set(), "aux": {}}
        # Children running now, so a cancel seen by one thread can kill those started by workers.
        self.live = set()
        self.live_lock = threading.Lock()
        # Evidence trace for Research Details: raw hits (with their merge fate), the candidate
        # objects themselves, and one selector decision per candidate, across both rounds.
        self.trace = {"raw": [], "candidates": [], "selector": [],
                      "memory": {"status": "not_used", "primary": [], "secondary": [],
                                 "written": {"units_new": 0, "units_seen": 0, "discoveries": 0,
                                             "selections": 0, "relationships": 0, "derived": 0},
                                 "reuse": {"exact_result": False, "previous_run": None,
                                           "requirements": [], "primary": [],
                                           "searches_skipped": [], "searches_fresh": [],
                                           "selector_reused": [], "selector_fresh": []},
                                 "errors": []},
                      # Where each NotebookLM search response came from (fresh call or the
                      # exact-request cache), across both rounds.
                      "retrieval": {"queries": 0, "from_cache": 0, "fresh": 0,
                                    "passages_from_cache": 0, "passages_fresh": 0,
                                    "detail": []},
                      # Research-need classification and the shadow reuse gate (see
                      # need_shadow): recorded, never acted on.
                      "reuse_decision": {"memory_lookup": None, "need": None, "gate": None,
                                         "stages": None, "versions": None},
                      # Per selector round, "bypassed", "trimmed" (with the
                      # dropped hit ids) or "run" with candidate chars vs the evidence budget;
                      # per Gemini selector attempt, the breaker state.
                      "selector_mode": [], "gemini_breaker": [],
                      # The NotebookLM ask path (see ask_research); None before it runs
                      "ask": None}
        # Selector decisions (with claim structure) of every kept hit, across both rounds: the
        # evidence map is built from them.
        self.claims = {}
        # Why the last selector stage fell back from Gemini to Haiku (None: no fallback).
        self.selector_fallback = None
        self.gemini_fallback = None  # why the last gemini_task fell back to Haiku
        # A Gemini selector network/timeout failure in this run: skip Gemini for the rest of it.
        self.gemini_tripped = False
        # The planner's raw case_frame, and this run's research-need record for the ledger.
        self.case_frame_raw = None
        self.requested_parts = []  # the planner's requested_parts, normalized
        self.asks = []  # The planner's fact questions for NotebookLM (see normalize_asks)
        # The community records looked up before the planner (None: no lookup ran) and
        # the planner's verification questions for them (see normalize_verifications)
        self.community = None
        self.community_pool = []  # Unit ids of the records the planner was shown
        self.verifications = []
        self.planned = None  # (depth, requirements, searches) the fallback reuses
        self.need = None
        # When the run started (monotonic) and when the first streamed answer text arrived
        # (seconds after the start; None until then).
        self.started = time.monotonic()
        self.first_answer_seconds = None
        self.writing_since = None  # wall time the current answer's first text arrived
        self.checking_since = None  # wall time the coverage check / follow-up round began
        self.reading_since = None  # wall time the current answer call began
        self.emit_lock = threading.Lock()  # the preflight thread reports its end too

    def emit(self, kind, **data):
        """Report progress, timing the callback. A Windows console blocks writes while text is
        selected in it (QuickEdit), which once stalled a run for ~228s after the reasoner."""
        if self.on_event:
            with self.emit_lock:
                start = time.monotonic()
                self.on_event({"type": kind, "at": time.time(), **data})
                self.output_wait += time.monotonic() - start

    def writing(self, active):
        """The web app's Writing step: stage "write" starts when an answer's first text
        arrives and ends when the answer is complete, or when a follow-up round interrupts it."""
        if active and self.writing_since is None:
            self.writing_since = time.time()
            self.emit("stage_start", stage="write", label="Writing")
        elif not active and self.writing_since is not None:
            secs = round(time.time() - self.writing_since, 1)
            self.writing_since = None
            self.emit("stage_end", stage="write", seconds=secs, timed=f"{secs:.1f}s")

    def reading(self, active):
        """The web app's Reading sources step: stage "read" runs from the start of an
        answer call to its first text (or its end, when it stops to search instead)."""
        if active and self.reading_since is None:
            self.reading_since = time.time()
            self.emit("stage_start", stage="read", label="Reading sources")
        elif not active and self.reading_since is not None:
            secs = round(time.time() - self.reading_since, 1)
            self.reading_since = None
            self.emit("stage_end", stage="read", seconds=secs, timed=f"{secs:.1f}s")

    def checking(self, active):
        """The web app's Checking step: stage "check" covers the coverage check and any
        follow-up retrieval, up to the answer pass that follows it."""
        if active and self.checking_since is None:
            self.checking_since = time.time()
            self.emit("stage_start", stage="check", label="Checking")
        elif not active and self.checking_since is not None:
            secs = round(time.time() - self.checking_since, 1)
            self.checking_since = None
            self.emit("stage_end", stage="check", seconds=secs, timed=f"{secs:.1f}s")

    def log(self, event, **data):
        if self.dir is None:
            return
        entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **data}
        with open(self.dir / "run.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def save(self, name, text):
        (self.dir / name).write_text(text, encoding="utf-8")

    def fail(self, msg, provider=None, kind=None):
        self.log("error", msg=msg, provider=provider, kind=kind)
        raise ResearchError(msg, self.dir, provider, kind)

    def cancelled(self):
        self.log("cancelled")
        raise ResearchCancelled(self.dir)

    def begin(self, stage, label, cli=None):
        """Start a stage; return its start time. `cli` is the terminal's same-line stage text."""
        if self.cancel.is_set():
            self.cancelled()
        self.emit("stage_start", stage=stage, label=label, cli=cli)
        return time.monotonic()

    def timed(self, label, start, chars=None):
        """Record a finished stage and return its '12.3s, 4,567 chars' summary."""
        secs = time.monotonic() - start
        self.times.append((label, secs, chars))
        return f"{secs:.1f}s" + (f", {chars:,} chars" if chars is not None else "")

    def done(self, stage, timed, detail=None, summary=None):
        """Report a finished stage: `timed` from timed(), a short `detail` for the web app and
        the terminal's `summary` line."""
        self.emit("stage_end", stage=stage, seconds=round(self.times[-1][1], 1), timed=timed,
                  detail=detail, summary=summary)


def kill_tree(proc):
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            proc.kill()
    except OSError:
        pass


def stop(proc):
    """Kill a child (and its process tree) and collect whatever output it produced."""
    try:
        kill_tree(proc)
        out, err = proc.communicate(timeout=10)
    except (subprocess.TimeoutExpired, KeyboardInterrupt, OSError, ValueError):
        proc.kill()
        out, err = "", ""
    return out or "", err or ""


def run_child(run, cmd, input=None, cwd=None, env=None, timeout=None):
    """Run a child process; return (returncode, stdout, stderr, status).

    status is None on normal exit, "timeout" or "cancelled" (Ctrl+C or run.cancel) otherwise; in
    those cases the child has been killed and returncode is None.
    """
    if run.cancel.is_set():
        return None, "", "", CANCELLED
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE if input is not None else None,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            encoding="utf-8", errors="replace", cwd=cwd, env=env)
    with run.live_lock:
        run.live.add(proc)
    if run.cancel.is_set():  # cancelled while this child was starting
        kill_tree(proc)
    start = time.monotonic()
    try:
        while True:
            elapsed = time.monotonic() - start
            wait_s = POLL if timeout is None else min(POLL, max(timeout - elapsed, 0.01))
            try:
                out, err = proc.communicate(input, timeout=wait_s)
                # The child shares the console, so Ctrl+C may kill it before we see the interrupt.
                status = CANCELLED if proc.returncode in CTRL_C_EXIT or run.cancel.is_set() else None
                return proc.returncode, out or "", err or "", status
            except subprocess.TimeoutExpired:
                input = None  # already sent; communicate() keeps the output collected so far
                if run.cancel.is_set():
                    return (None, *stop(proc), CANCELLED)
                if timeout is not None and time.monotonic() - start >= timeout:
                    return (None, *stop(proc), "timeout")
    except KeyboardInterrupt:
        return (None, *stop(proc), CANCELLED)
    finally:
        with run.live_lock:
            run.live.discard(proc)


def stream_child(run, cmd, input, cwd=None, env=None, on_line=None, idle_timeout=None):
    """run_child for a child whose stdout is consumed line by line as it arrives: on_line(line)
    runs on the calling thread, which keeps polling run.cancel, so a cancel or Ctrl+C mid-stream
    kills the child promptly. Returns (returncode, stdout, stderr, status) like run_child.

    on_line returning True stops the call early (the reasoner decided to search before
    any answer text): the child is killed and status is "stopped". With idle_timeout,
    a child that prints no line for that many seconds is killed and status is "timeout"."""
    if run.cancel.is_set():
        return None, "", "", CANCELLED
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding="utf-8",
                            errors="replace", cwd=cwd, env=env)
    with run.live_lock:
        run.live.add(proc)
    lines, err, out = queue.Queue(), [], []

    def feed():
        try:
            proc.stdin.write(input)
            proc.stdin.close()
        except (OSError, ValueError):
            pass

    def read_out():
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    readers = [threading.Thread(target=f, daemon=True) for f in
               (feed, read_out, lambda: err.append(proc.stderr.read()))]
    for t in readers:
        t.start()
    status, last = None, time.monotonic()
    try:
        while True:
            if run.cancel.is_set():
                status = CANCELLED
                break
            try:
                line = lines.get(timeout=POLL)
            except queue.Empty:
                if idle_timeout is not None and time.monotonic() - last >= idle_timeout:
                    status = "timeout"
                    break
                continue
            last = time.monotonic()
            if line is None:
                break
            out.append(line)
            if on_line and on_line(line) is True:
                status = STOPPED
                break
    except KeyboardInterrupt:
        status = CANCELLED
    finally:
        if status in (CANCELLED, STOPPED, "timeout"):
            kill_tree(proc)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        for t in readers:
            t.join(timeout=5)
        with run.live_lock:
            run.live.discard(proc)
    if status is None and (proc.returncode in CTRL_C_EXIT or run.cancel.is_set()):
        status = CANCELLED
    return (None if status else proc.returncode), "".join(out), "".join(err), status


def stream_mark(t, line):
    """A compact timing record of one stream-json line (diagnostics: where the time before the
    first answer character goes): seconds since the stage started, the event and block/delta
    type, the delta's length, the CLI's thinking estimate, and the text of plain text deltas
    (a turn written as text instead of the structured-output call). None for unparsable lines."""
    try:
        e = json.loads(line)
    except json.JSONDecodeError:
        return None
    ev = e.get("event") or {}
    delta = ev.get("delta") or {}
    mark = {"t": round(t, 2), "type": e.get("type"), "sub": e.get("subtype") or ev.get("type")}
    if ev.get("type") == "content_block_start":
        mark["block"] = (ev.get("content_block") or {}).get("type")
    if delta:
        mark["delta"] = delta.get("type")
        mark["chars"] = len(delta.get("partial_json") or delta.get("text") or delta.get("thinking") or "")
        if delta.get("type") == "text_delta":
            mark["text"] = delta.get("text")
    if e.get("subtype") == "thinking_tokens":
        mark["thinking_est"] = e.get("estimated_tokens")
    if e.get("type") == "user":  # a tool result (e.g. a structured-output validation error)
        mark["text"] = json.dumps(e.get("message"), ensure_ascii=False)[:400]
    return mark


def stream_event_type(line):
    """The "type" of one stream-json line, or None when it is not a JSON object."""
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return None
    return event.get("type") if isinstance(event, dict) else None


class JsonStringField:
    """Incremental decoder of one top-level string field of a JSON object that arrives in chunks
    (feed(chunk) returns the field's newly decoded text, possibly ""). It tracks nesting and
    strings only as far as needed to find `"<field>": "` at depth 1, then decodes that string's
    escapes (including \\uXXXX surrogate pairs split across chunks) until its closing quote."""

    ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "/": "/", "\\": "\\",
               '"': '"'}

    def __init__(self, field):
        self.field = field
        self.depth = 0
        self.in_str = self.esc = self.after_colon = self.streaming = self.is_key = False
        self.key = []
        self.last_key = None
        self.uni = None  # hex digits of a \u escape being read
        self.high = None  # a pending high surrogate

    def feed(self, chunk):
        out = []
        for ch in chunk:
            if self.in_str:
                if self.uni is not None:
                    self.uni += ch
                    if len(self.uni) == 4:
                        code, self.uni = int(self.uni, 16), None
                        if 0xD800 <= code < 0xDC00:
                            self.high = code
                            continue
                        if 0xDC00 <= code < 0xE000 and self.high is not None:
                            code = 0x10000 + ((self.high - 0xD800) << 10) + (code - 0xDC00)
                        self.high = None
                        self._char(chr(code), out)
                    continue
                if self.esc:
                    self.esc = False
                    if ch == "u":
                        self.uni = ""
                    else:
                        self._char(self.ESCAPES.get(ch, ch), out)
                    continue
                if ch == "\\":
                    self.esc = True
                elif ch == '"':
                    self.in_str = False
                    if self.streaming:
                        self.streaming = False
                    elif self.is_key:
                        self.last_key = "".join(self.key)
                else:
                    self._char(ch, out)
                continue
            if ch == '"':
                self.in_str = True
                self.is_key = self.depth == 1 and not self.after_colon
                self.key = []
                self.streaming = (self.depth == 1 and self.after_colon
                                  and self.last_key == self.field)
            elif ch in "{[":
                self.depth += 1
            elif ch in "}]":
                self.depth -= 1
            elif self.depth == 1 and ch == ":":
                self.after_colon = True
            elif self.depth == 1 and ch == ",":
                self.after_colon = False
        return "".join(out)

    def _char(self, ch, out):
        if self.streaming:
            out.append(ch)
        elif self.is_key:
            self.key.append(ch)


class AnswerStream:
    """Streams the "answer" field of a structured-output Claude stage from the CLI's stream-json
    lines: the StructuredOutput tool input arrives as input_json_delta chunks. on_text(text,
    reset) receives decoded answer text, coalesced to at most one call per STREAM_FLUSH seconds;
    reset is True for the first text of each tool call (a structured-output retry starts over).

    pre_answer measures what the model produced before the first answer character:
    the CLI's running thinking-token estimate and the structured-output JSON chars (fields that
    precede the answer); None until answer text arrives."""

    def __init__(self, on_text, field="answer"):
        self.on_text, self.field = on_text, field
        self.parser = self.block = None
        self.pending, self.pending_reset, self.reset = [], False, False
        self.last = 0.0
        self.thinking_tokens = 0  # the CLI's cumulative thinking estimate so far
        self.json_chars = 0  # structured-output chars of the current tool call so far
        self.pre_answer = None

    def line(self, raw):
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            return
        if event.get("type") == "system" and event.get("subtype") == "thinking_tokens":
            self.thinking_tokens = event.get("estimated_tokens") or self.thinking_tokens
            return
        if event.get("type") != "stream_event":
            return
        ev = event.get("event") or {}
        kind = ev.get("type")
        if kind == "content_block_start" and (ev.get("content_block") or {}).get("type") == "tool_use":
            self.flush()
            self.parser, self.block, self.reset = JsonStringField(self.field), ev.get("index"), True
            self.json_chars = 0
        elif (kind == "content_block_delta" and self.parser is not None
              and ev.get("index") == self.block
              and (ev.get("delta") or {}).get("type") == "input_json_delta"):
            chunk = ev["delta"].get("partial_json") or ""
            text = self.parser.feed(chunk)
            if text and self.pre_answer is None:
                # JSON chars before this chunk plus this chunk's non-answer prefix.
                self.pre_answer = {"thinking_tokens_est": self.thinking_tokens,
                                   "pre_answer_json_chars": self.json_chars + len(chunk) - len(text)}
            self.json_chars += len(chunk)
            if text:
                if not self.pending:
                    self.pending_reset, self.reset = self.reset, False
                self.pending.append(text)
                if time.monotonic() - self.last >= STREAM_FLUSH:
                    self.flush()

    def flush(self):
        if self.pending:
            text, reset = "".join(self.pending), self.pending_reset
            self.pending, self.pending_reset = [], False
            self.last = time.monotonic()
            self.on_text(text, reset)


META_OPEN, META_CLOSE = "<<<META", "META>>>"
CONTROL_PREFIXES = ("COVERAGE:", "DECISION:")


def parse_coverage(line):
    """[(part, "yes" | "no")] from a "COVERAGE: a=yes; b=no" line (pure); [] for "n/a" or None."""
    body = (line or "").split(":", 1)[1] if line and ":" in line else ""
    out = []
    for item in body.split(";"):
        part, _, mark = item.rpartition("=")
        part, mark = clean(part), mark.strip().lower()
        if part and mark in ("yes", "no"):
            out.append((part, mark))
    return out


def parse_decision(line):
    """("answer" | "search", [queries]) from a DECISION line (pure); ("answer", []) when absent
    or unreadable."""
    body = (line or "").split(":", 1)[1] if line and ":" in line else ""
    head, *queries = [clean(x) for x in body.split("|")]
    if head.lower() == "search":
        return "search", [q for q in queries if q][:3]
    return "answer", []


def parse_meta(raw):
    """The META block's JSON object (pure), or None when missing or invalid."""
    if raw is None:
        return None
    text = raw.split(META_CLOSE, 1)[0].strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        out = json.loads(text)
    except (TypeError, ValueError):
        return None
    return out if isinstance(out, dict) else None


class ReasonerText:
    """The plain-text reasoner protocol, parsed as it streams (feed(text) with text
    deltas): control lines (COVERAGE, DECISION) first, then the markdown answer, then a
    <<<META ... META>>> block. Only answer text reaches on_text(text, reset) (coalesced to one
    call per STREAM_FLUSH seconds; reset on the first); control lines and META never do, and a
    possible partial "<<<META" at the end of a delta is held back until it is decided.

    on_control(self), called once when the control lines are complete (at DECISION, or when the
    answer starts without one), returns True to stop the call before any answer text; stopped
    is then True."""

    def __init__(self, on_text=None, on_control=None):
        self.on_text, self.on_control = on_text, on_control
        self.state = "control"  # control -> answer -> meta
        self.buf, self.answer, self.meta_raw = "", [], None
        self.coverage_line = self.decision_line = None
        self.stopped = self.controlled = False
        self.pending, self.started, self.last = [], False, 0.0

    def _control_done(self):
        if not self.controlled:
            self.controlled = True
            if self.on_control and self.on_control(self):
                self.stopped = True

    def _emit(self, text):
        if not text:
            return
        if not self.answer:
            text = text.lstrip("\n")
            if not text:
                return
        self.answer.append(text)
        self.pending.append(text)
        if time.monotonic() - self.last >= STREAM_FLUSH:
            self.flush()

    def flush(self):
        if self.pending and self.on_text:
            text, reset = "".join(self.pending), not self.started
            self.pending, self.started, self.last = [], True, time.monotonic()
            self.on_text(text, reset)
        self.pending = []

    def feed(self, text):
        if self.stopped:
            return
        self.buf += text
        while not self.stopped:
            if self.state == "control":
                nl = self.buf.find("\n")
                head = self.buf.lstrip()
                if nl == -1:
                    # An unfinished first line: wait while it may still be a control line.
                    if not head or any(p.startswith(head[:len(p)]) for p in CONTROL_PREFIXES):
                        return
                    self.state = "answer"
                    self._control_done()
                    continue
                line = self.buf[:nl].strip()
                if not line:
                    self.buf = self.buf[nl + 1:]
                    continue
                if line.startswith("COVERAGE:") and self.coverage_line is None:
                    self.coverage_line, self.buf = line, self.buf[nl + 1:]
                    continue
                if line.startswith("DECISION:") and self.decision_line is None:
                    self.decision_line, self.buf = line, self.buf[nl + 1:]
                    self.state = "answer"
                    self._control_done()
                    continue
                self.state = "answer"
                self._control_done()
                continue
            if self.state == "answer":
                i = self.buf.find(META_OPEN)
                if i != -1:
                    self._emit(self.buf[:i])
                    self.meta_raw, self.buf, self.state = self.buf[i + len(META_OPEN):], "", "meta"
                    continue
                keep = next((k for k in range(min(len(META_OPEN) - 1, len(self.buf)), 0, -1)
                             if META_OPEN.startswith(self.buf[-k:])), 0)
                self._emit(self.buf[:len(self.buf) - keep])
                self.buf = self.buf[len(self.buf) - keep:]
                return
            self.meta_raw += self.buf  # meta
            self.buf = ""
            return

    def finish(self):
        """End of the stream: release held-back text; return the parsed result."""
        if not self.stopped:
            if self.state == "control" and self.buf.strip():
                # A reply may end on its control line without a newline ("DECISION:
                # search | q1 | q2" and nothing else); complete that line so it is read as one.
                self.feed("\n")
            if self.state == "control" and self.buf.strip() and not self.stopped:
                self.state = "answer"
                self._control_done()
            if self.state == "answer" and not self.stopped:
                self._emit(self.buf)
            self.buf = ""
            self.flush()
        return self.result()

    def result(self):
        decision, queries = parse_decision(self.decision_line)
        meta = parse_meta(self.meta_raw)
        return {"decision": decision, "queries": queries,
                "coverage": parse_coverage(self.coverage_line),
                "coverage_line": self.coverage_line, "decision_line": self.decision_line,
                "answer": "".join(self.answer).strip(), "meta": meta,
                "meta_status": ("ok" if meta is not None else
                                "missing" if self.meta_raw is None else "invalid"),
                "stopped": self.stopped}


def parse_reasoner_text(text):
    """The plain-text protocol parsed from a complete reply (pure)."""
    reader = ReasonerText()
    reader.feed(text)
    return reader.finish()


def parallel(run, fn, items, waiting):
    """Run fn(item) for every item on at most WORKERS threads; return results in item order.

    Only NotebookLM children run here, never Claude. Workers return errors instead of raising or
    logging. The calling thread polls, so Ctrl+C or run.cancel is handled promptly: it kills every
    running child and cancels the question.
    """
    if not items:
        return []
    pool = ThreadPoolExecutor(max_workers=min(WORKERS, len(items)))
    futures = [pool.submit(fn, item) for item in items]
    start, next_tick, shown, interrupted = time.monotonic(), TICK, False, False
    try:
        while wait(futures, timeout=POLL).not_done:
            if run.cancel.is_set():
                interrupted = True
                break
            if time.monotonic() - start >= next_tick:
                run.emit("waiting", label=waiting, seconds=next_tick)
                shown = True
                next_tick += TICK
    except KeyboardInterrupt:
        interrupted = True
    finally:
        if shown:
            run.emit("waiting_end")
    if interrupted:
        run.cancel.set()
        with run.live_lock:
            running = list(run.live)
        for proc in running:
            kill_tree(proc)
        pool.shutdown(wait=False, cancel_futures=True)
        run.cancelled()
    pool.shutdown()
    return [f.result() for f in futures]


SYSTEM_PROMPT_DIR = settings.CACHE_DIR / "system-prompts"


def system_prompt_file(system):
    """The system prompt as a file for --system-prompt-file, named by its content hash (the
    answer prompt outgrew Windows' 32,767-character command line)."""
    path = SYSTEM_PROMPT_DIR / f"{hashlib.sha256(system.encode('utf-8')).hexdigest()[:24]}.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.part")
        part.write_text(system, encoding="utf-8")
        os.replace(part, path)
    return path


def claude(run, stage, model_effort, system, prompt, schema=None, on_answer=None,
           text_reader=None, timeout=STAGE_TIMEOUT, stream=False):
    """Run one tool-less Claude stage; return its text or parsed structured output.

    With text_reader or on_answer the call streams. A non-streaming call is killed after `timeout`
    seconds, a streaming one after that long without an event; either raises ResearchError."""
    model, effort = model_effort
    if "haiku" in model:
        effort = None  # Haiku 4.5 does not use the effort setting
    cmd = [CLAUDE, "-p", "--model", model]
    if effort:
        cmd += ["--effort", effort]
    cmd += ["--system-prompt-file", str(system_prompt_file(system)), "--tools", "",
            "--strict-mcp-config", "--setting-sources", "", "--no-session-persistence"]
    streaming = bool(on_answer or text_reader or stream)
    cmd += (["--output-format", "stream-json", "--verbose", "--include-partial-messages"]
            if streaming else ["--output-format", "json"])
    if schema:
        cmd += ["--json-schema", json.dumps(schema)]
    stage_env = ENV.copy()
    if stage.startswith("reasoner-"):
        stage_env["MAX_THINKING_TOKENS"] = str(ANSWER_THINKING_BUDGET)
    if stage.startswith(("selector-", "coverage-", "ledger-")):
        stage_env["MAX_THINKING_TOKENS"] = "0"
        stage_env["DISABLE_PROMPT_CACHING_HAIKU"] = "1"
        stage_env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = SELECTOR_MAX_OUTPUT_TOKENS
    run.log("claude_start", stage=stage, model=model, effort=effort,
            max_thinking_tokens=stage_env.get("MAX_THINKING_TOKENS"),
            disable_prompt_caching=stage_env.get("DISABLE_PROMPT_CACHING_HAIKU"),
            max_output_tokens=stage_env.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS"))
    started = time.monotonic()
    for attempt in (1, 2):
        if streaming:
            reader = AnswerStream(on_answer) if on_answer else None
            timeline, t0 = [], time.monotonic()

            def on_line(line):
                mark = stream_mark(time.monotonic() - t0, line)
                timeline.append(mark)
                if reader:
                    reader.line(line)
                if text_reader:
                    if mark and mark.get("delta") == "text_delta":
                        text_reader.feed(json.loads(line)["event"]["delta"].get("text") or "")
                    # Checked on every line: a coverage verdict may stop the call from outside.
                    return text_reader.stopped
            returncode, stdout, stderr, status = stream_child(run, cmd, prompt, cwd=STAGE_CWD,
                                                              env=stage_env, on_line=on_line,
                                                              idle_timeout=timeout)
            if reader:
                reader.flush()
            run.save(f"{stage}.stream.jsonl", "\n".join(
                json.dumps(m, ensure_ascii=False) for m in timeline if m))
            # The final "result" event is the object --output-format json prints.
            stdout = next((line for line in reversed(stdout.splitlines())
                           if stream_event_type(line) == "result"), stdout)
        else:
            returncode, stdout, stderr, status = run_child(run, cmd, input=prompt, cwd=STAGE_CWD,
                                                           env=stage_env, timeout=timeout)
        if status == STOPPED:  # the reader ended the call on purpose (DECISION: search)
            run.save(f"{stage}.claude.json", json.dumps({"stopped": True}))
            usage = {"stage": stage, "model": model, "effort": effort, "models": [],
                     "usage": None, "cost_usd": None,
                     "seconds": None, "provider": "claude", "stopped_early": True,
                     "wall_seconds": round(time.monotonic() - started, 1)}
            run.usage.append(usage)
            run.log("claude_done", **usage)
            return {**text_reader.result(), "turns": None}
        run.save(f"{stage}.claude.json", stdout or stderr)
        if status == CANCELLED:
            run.cancelled()
        if status == "timeout":
            run.usage.append({"stage": stage, "model": model, "effort": effort, "models": [],
                              "usage": None, "cost_usd": None,
                              "seconds": None, "provider": "claude", "timed_out": True,
                              "wall_seconds": round(time.monotonic() - started, 1)})
            run.fail(f"{stage}: no response from Claude within {timeout}s", provider="claude",
                     kind="network")
        try:
            out = json.loads(stdout)
        except json.JSONDecodeError:
            run.fail(f"{stage}: Claude returned no JSON (exit {returncode}): {stderr[:300]}",
                     provider="claude", kind=classify_failure(stdout + stderr))
        # The CLI gives up after several malformed structured outputs in one call; that is a
        # model slip, not a pipeline error, so the stage is run once more before failing.
        if out.get("subtype") == "error_max_structured_output_retries" and attempt == 1:
            run.log("claude_retry", stage=stage, reason=out.get("subtype"))
            continue
        break
    if out.get("is_error"):
        detail = out.get("result") or "; ".join(map(str, out.get("errors") or [])) or out.get("subtype")
        run.fail(f"{stage}: {detail}", provider="claude",
                 kind=classify_failure(f"{detail} {stderr}"))
    # usage follows Anthropic semantics: input_tokens excludes cache reads and writes, and
    # output_tokens already includes extended thinking. cost_usd is the CLI's own estimate.
    usage = {"stage": stage, "model": model, "effort": effort,
             "models": list(out.get("modelUsage", {})), "usage": out.get("usage"),
             "cost_usd": out.get("total_cost_usd"), "seconds": round(out.get("duration_ms", 0) / 1000),
             "provider": "claude", "wall_seconds": round(time.monotonic() - started, 1)}
    if on_answer:
        usage["pre_answer"] = reader.pre_answer
    if streaming:
        usage["max_thinking_tokens"] = stage_env.get("MAX_THINKING_TOKENS")
        usage["turns"] = out.get("num_turns")
    run.usage.append(usage)
    run.log("claude_done", **usage)
    if text_reader:
        parsed = text_reader.finish()
        if (not parsed["answer"] and parsed["decision_line"] is None and not parsed["stopped"]
                and (out.get("result") or "").strip()):
            # No text deltas arrived (or they could not be read): parse the final text instead.
            # Not when the stream held a DECISION line (a reparse loses "stopped").
            parsed = parse_reasoner_text(out["result"])
        return {**parsed, "turns": out.get("num_turns")}
    if schema:
        if not isinstance(out.get("structured_output"), dict):
            run.fail(f"{stage}: no structured output")
        return out["structured_output"]
    return out["result"].strip()


def notebooklm(run, label, args):
    """Run one NotebookLM CLI command that prints JSON; return (parsed output, error).

    Safe in worker threads: it never raises or logs. error is None on success, CANCELLED after
    a cancel, otherwise a message.
    """
    returncode, stdout, stderr, status = run_child(run, NOTEBOOKLM + args,
                                                   timeout=NOTEBOOKLM_TIMEOUT)
    if status == CANCELLED:
        return None, CANCELLED
    if status == "timeout":
        return None, f"NotebookLM {label} timed out after {NOTEBOOKLM_TIMEOUT}s"
    # Any non-zero exit is a failure, even if stdout happens to hold valid-looking JSON.
    if returncode != 0:
        return None, f"NotebookLM {label} failed (exit {returncode}): {(stderr or stdout)[-300:]}"
    try:
        return json.loads(stdout), None
    except json.JSONDecodeError:
        return None, f"NotebookLM {label} failed: output is not JSON: {stderr[-300:]}"


def clean(text, limit=None):
    text = " ".join(str(text or "").split())
    return text[:limit] if limit else text


def normalize_plan(out, question):
    """Clean the planner's output (pure: plan() and the tests use it). Returns (depth,
    requirements, searches, notes for the run log).

    Requirements are renumbered r1.. with a valid kind; a plan without any falls back to the
    question itself as one claim requirement. Searches become {"query", "covers", "type"} with
    known requirement ids only (type defaults to "literal"; TYPE_MAX_SEARCHES caps component and
    vocabulary searches, later ones dropped first) (a lone requirement is implied when a search names none); a repeated
    query merges into the first. Over the depth's limit, searches that add coverage are kept
    first, in plan order, so capping drops coverage only when the limit leaves no choice.
    """
    notes = {}
    depth = out.get("depth")
    if depth not in DEPTH_MAX_SEARCHES:
        notes["depth_invalid"] = depth
        depth = "normal"
    requirements, rename = [], {}
    for r in out.get("requirements") or []:
        text = clean(r.get("text")) if isinstance(r, dict) else ""
        if not text or len(requirements) >= MAX_REQUIREMENTS:
            continue
        rid = f"r{len(requirements) + 1}"
        rename.setdefault(clean(r.get("id")) or rid, rid)
        kind = r.get("kind") if r.get("kind") in REQUIREMENT_KINDS else "claim"
        requirements.append({"id": rid, "kind": kind, "text": text,
                             **({"exact": True} if r.get("exact") is True else {})})
    if not requirements:
        notes["requirements_fallback"] = True
        requirements = [{"id": "r1", "kind": "claim", "text": clean(question)}]

    by_key = {}
    for s in out.get("searches") or []:
        query = clean(s.get("query") if isinstance(s, dict) else s)
        if not query:
            continue
        named = (s.get("covers") if isinstance(s, dict) else None) or []
        covers = [rename[k] for k in (clean(c) for c in named if isinstance(c, str)) if k in rename]
        if not covers and len(requirements) == 1:
            covers = ["r1"]
        kind = s.get("type") if isinstance(s, dict) else None
        prior = by_key.setdefault(query.lower(), {"query": query, "covers": [],
                                                  "type": kind if kind in SEARCH_TYPES
                                                  else "literal"})
        prior["covers"] = list(dict.fromkeys(prior["covers"] + covers))
    searches = list(by_key.values())
    unlabeled = [s["query"] for s in searches if not s["covers"]]
    if unlabeled:
        notes["searches_without_covers"] = unlabeled
    for kind, most in TYPE_MAX_SEARCHES.items():  # per-type caps first, in plan order
        of_kind = [s for s in searches if s["type"] == kind]
        if len(of_kind) > most:
            notes.setdefault("searches_type_capped", {})[kind] = [s["query"] for s in of_kind[most:]]
            searches = [s for s in searches if s not in of_kind[most:]]

    limit = DEPTH_MAX_SEARCHES[depth]
    if len(searches) > limit:
        kept, covered = [], set()
        for s in searches:  # first the searches that add coverage, in plan order
            if len(kept) < limit and set(s["covers"]) - covered:
                kept.append(s)
                covered.update(s["covers"])
        for s in searches:  # then any slots left, in plan order
            if len(kept) < limit and s not in kept:
                kept.append(s)
        notes["searches_capped"] = {"limit": limit,
                                    "dropped": [s["query"] for s in searches if s not in kept]}
        searches = [s for s in searches if s in kept]
    covered = {c for s in searches for c in s["covers"]}
    uncovered = [r["id"] for r in requirements if r["id"] not in covered]
    if uncovered:
        notes["requirements_uncovered"] = uncovered
    return depth, requirements, searches, notes


def normalize_requested_parts(parts):
    """The planner's requested_parts cleaned (pure): entries with an attribute and exactly one of
    part / per_component_of, as {"part", "attribute"} or {"per_component_of", "attribute"}."""
    out = []
    for x in parts or []:
        if not isinstance(x, dict):
            continue
        part, whole, attr = (clean(x.get(k)) for k in ("part", "per_component_of", "attribute"))
        if attr and bool(part) != bool(whole):
            out.append({"per_component_of": whole, "attribute": attr} if whole
                       else {"part": part, "attribute": attr})
    return out


def normalize_verifications(items, community, asks):
    """The planner's verification questions cleaned (pure): each names a known community
    record (s1..), has a question that repeats no ask, and a search string (the question's key
    terms when the planner gave none); at most VERIFY_MAX. Returns [{"record", "unit_id",
    "question", "search"}]."""
    ids = {f"s{n}": s["unit_id"] for n, s in enumerate(community or [], 1)}
    out, seen = [], {a.lower() for a in asks or []}
    for x in items or []:
        if not isinstance(x, dict) or len(out) >= VERIFY_MAX:
            continue
        record = clean(x.get("record")).strip("[]")
        question = clean(x.get("question")).replace("|", "/")
        if record not in ids or not question or question.lower() in seen:
            continue
        seen.add(question.lower())
        out.append({"record": record, "unit_id": ids[record], "question": question,
                    "search": cap_query(clean(x.get("search")).replace("|", "/"))
                              or search_terms(question)})
    return out


def pick_community(items, community, verifications):
    """The community records the answer uses (pure): those the planner listed in
    "community" (ids s1.. of COMMUNITY RECORDS, in its order, at most COMMUNITY_PICK_MAX), then any
    record with a verification not listed. Without a "community" list (an older or unreadable
    reply) the top MEMORY_SECONDARY_MAX records are kept. Returns (picked records, planner_picked)."""
    community = list(community or [])
    if not isinstance(items, list):
        return community[:MEMORY_SECONDARY_MAX], False
    by_id = {f"s{n}": s for n, s in enumerate(community, 1)}
    out = []
    for x in items:
        s = by_id.get(clean(x if isinstance(x, str) else "").strip("[]"))
        if s is not None and s not in out and len(out) < COMMUNITY_PICK_MAX:
            out.append(s)
    verified = {v["unit_id"] for v in verifications or []}
    out += [s for s in community if s["unit_id"] in verified and s not in out]
    return out, True


def normalize_asks(asks, question, most=ASKS_MAX, allow_empty=False):
    """The planner's fact questions cleaned (pure): non-empty strings, "|" removed, repeats
    (case-insensitive) dropped, at most `most`. Without any, the question itself is the ask,
    unless allow_empty (a follow-up answered from the related turn's evidence)."""
    out, seen = [], set()
    for a in asks or []:
        text = clean(a if isinstance(a, str) else "").replace("|", "/")
        if text and text.lower() not in seen and len(out) < most:
            seen.add(text.lower())
            out.append(text)
    return out or ([] if allow_empty else [clean(question)])


def normalize_reuse(value, related):
    """The planner's reuse decision (pure): ANSWER_FROM_TURN only for a follow-up whose
    related turn has passages to answer from; anything else is "research"."""
    if (value == ANSWER_FROM_TURN and related is not None
            and (related.get("retrieved") or related.get("cited"))):
        return ANSWER_FROM_TURN
    return "research"


def coverage_followup(parsed, requested_parts, depth, pass_n, followup_done, earlier):
    """Whether the reasoner's COVERAGE line triggers the one follow-up round (pure); returns
    (fire, queries, reason). It fires when an entry is "no", on the first pass, before any
    follow-up, at a depth other than quick ("direct"), with at least one usable query. Queries:
    the reasoner's own DECISION queries first, then "<component> <attribute>" per "no" entry
    (the attribute of the named part, else of the per-component request), each dropped when it
    rewords an earlier query (repeats_query), at most REPAIR_MAX_SEARCHES."""
    if pass_n != 1 or followup_done:
        return False, [], "not the first pass"
    if depth == "direct":
        return False, [], "quick depth"
    missing = [part for part, mark in parsed.get("coverage") or [] if mark == "no"]
    if not missing:
        return False, [], "no coverage gap"
    named = {x["part"].lower(): x["attribute"] for x in requested_parts if x.get("part")}
    shared = next((x["attribute"] for x in requested_parts if x.get("per_component_of")), None)
    proposed = list(parsed.get("queries") or [])
    for part in missing:
        attr = named.get(part.lower()) or shared or next(iter(named.values()), "")
        proposed.append(clean(f"{part} {attr}") if attr and attr.lower() not in part.lower()
                        else part)
    queries = []
    for q in proposed:
        if q and not repeats_query(q, list(earlier) + queries):
            queries.append(q)
        if len(queries) >= REPAIR_MAX_SEARCHES:
            break
    if not queries:
        return False, [], "every follow-up query repeats an earlier search"
    return True, queries, f"coverage no: {', '.join(missing)}"


def validate_plan(out):
    """Problems with a raw planner output (pure; [] when valid): case_frame must be the last
    field, every search needs a known type, and the depth's total cap and the per-type caps
    (TYPE_MAX_SEARCHES) must hold. normalize_plan still enforces the caps; this reports them."""
    problems = []
    if not isinstance(out, dict):
        return ["plan is not an object"]
    keys = list(out)
    if "case_frame" in out and keys[-1] != "case_frame":
        problems.append(f"case_frame is not the last field (order: {keys})")
    depth = out.get("depth")
    searches = [s for s in out.get("searches") or [] if isinstance(s, dict)]
    limit = DEPTH_MAX_SEARCHES.get(depth)
    if limit is None:
        problems.append(f"unknown depth {depth!r}")
    elif len(searches) > limit:
        problems.append(f"{len(searches)} searches over the {depth} cap of {limit}")
    bad = [s.get("query") for s in searches if s.get("type") not in SEARCH_TYPES]
    if bad:
        problems.append(f"searches without a valid type: {bad}")
    for kind, most in TYPE_MAX_SEARCHES.items():
        n = sum(1 for s in searches if s.get("type") == kind)
        if n > most:
            problems.append(f"{n} {kind} searches over the cap of {most}")
    return problems


QUERY_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from he his how i in is it its of on or should "
    "the their to was what when which who why with you your".split())
# Words that name the kind of text sought, not its subject: they do not make a query new.
GENERIC_QUERY_WORDS = frozenset(
    "make making made prepare preparing preparation method methods recipe recipes way ways "
    "process procedure steps step instructions separate separating separation use using used "
    "about said say says author".split())


def query_terms_set(query, other=()):
    """A query's normalized subject terms (pure): lowercase words without stopwords or generic
    query words, with adjacent words joined when the joined form is a term of `other` (so
    "expense claim" matches "expenseclaim")."""
    words = [w for w in re.findall(r"[a-z0-9]+", (query or "").lower()) if w not in QUERY_STOPWORDS]
    other, terms, i = set(other), [], 0
    while i < len(words):
        if i + 1 < len(words) and words[i] + words[i + 1] in other:
            terms.append(words[i] + words[i + 1])
            i += 2
            continue
        terms.append(words[i])
        i += 1
    return {t for t in terms if t not in GENERIC_QUERY_WORDS}


def query_overlap(a, b):
    """Overlap of two queries' subject terms (pure): shared terms over the smaller set, 0..1."""
    raw_a = set(re.findall(r"[a-z0-9]+", (a or "").lower()))
    raw_b = set(re.findall(r"[a-z0-9]+", (b or "").lower()))
    ta, tb = query_terms_set(a, raw_b), query_terms_set(b, raw_a)
    if not ta or not tb:
        return 1.0 if ta == tb else 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def repeats_query(query, earlier):
    """The earlier query that `query` rewords (overlap >= QUERY_OVERLAP_MAX), else None."""
    return next((q for q in earlier if query_overlap(query, q) >= QUERY_OVERLAP_MAX), None)


def unwrap_plan(out):
    """A text plan made usable (pure), or None: a field nested inside an object of its own
    name ({"requirements": {"requirements": [...]}}) is unwrapped, and the plan must have a
    known depth and non-empty requirements and searches lists."""
    if not isinstance(out, dict):
        return None
    out = dict(out)
    for key in ("requirements", "searches", "requested_parts", "case_frame", "depth"):
        value = out.get(key)
        if isinstance(value, dict) and set(value) == {key}:
            out[key] = value[key]
    ok = (out.get("depth") in DEPTH_MAX_SEARCHES
          and all(isinstance(out.get(k), list) and out[k] for k in ("requirements", "searches")))
    return out if ok else None


FOLLOW_UP_PASSAGES_MAX = 10  # the previous run's kept passages offered to a follow-up
HEADING = re.compile(r"^#{1,6}\s+\S")


def short_answer(answer):
    """An answer's first section (pure): its text up to the next markdown heading, without the
    section's own heading; the whole answer when it has no heading."""
    lines, body = (answer or "").strip().splitlines(), []
    if lines and HEADING.match(lines[0]):
        lines = lines[1:]
    for line in lines:
        if HEADING.match(line):
            break
        body.append(line)
    return "\n".join(body).strip()


TURN_FILE = "turn.json"  # The compact record of an answered turn, in its run folder
THREAD_TURNS_MAX = 3  # earlier turns of the thread the planner sees
RELATED_PASSAGES_MAX = 20  # the related turn's cited passages offered as memory candidates


def answer_headings(answer):
    """The answer's section headings after the Short answer (pure)."""
    heads = [re.sub(r"^#{1,6}\s+", "", line).strip() for line in (answer or "").splitlines()
             if HEADING.match(line)]
    return [h for h in heads if h.lower() != "short answer"]


def retrieved_passages(sources, cited=()):
    """Every evidence passage of a run, from its result's sources (pure): id, source,
    title and text (the passage with any recovered continuation), the cited ones first. A later
    turn answered from this one's evidence (ANSWER_FROM_TURN) reads all of them, not only the
    cited ones."""
    out = []
    for src in sources or []:
        for e in src.get("excerpts") or []:
            for n, p in enumerate(e.get("passages") or []):
                if not p.get("text"):
                    continue
                text = p["text"]
                if n == 0 and e.get("continuation"):
                    text = f"{text}\n{e['continuation']}"
                out.append({"id": p.get("hit_id"), "source_id": src["source_id"],
                            "source_title": src.get("title"), "text": text})
    cited = set(cited)
    return ([p for p in out if p["id"] in cited]
            + [p for p in out if p["id"] not in cited])


def turn_record(question, standalone, answer, citations, run_dir, follow=None, sources=None):
    """Compact record of an answered turn, built from the answer (pure, no model call): the
    question, its standalone rewrite, the Short answer, the section headings and the cited
    passages (ids and texts, to offer again when a later turn builds on this one). `follow` is
    the run folder of the turn before it in the thread. "retrieved" holds every evidence
    passage of the run (see retrieved_passages)."""
    cited = [{"id": c["id"], "source_id": c["source_id"],
              "source_title": c.get("title"), "text": c["text"]}
             for c in citations or [] if c.get("kind") == "passage"]
    return {"question": question, "standalone": standalone,
            "short_answer": short_answer(answer), "headings": answer_headings(answer),
            "cited": cited, "retrieved": retrieved_passages(sources, [c["id"] for c in cited]),
            "run_dir": str(run_dir) if run_dir else None, "follow": follow}


def load_turn(folder):
    """The turn record in a run folder; for an older run without one, one built from its
    result.json (its passages stand in for cited ones). Raises ResearchError when the folder
    holds no completed result."""
    folder = Path(folder)
    try:
        record = json.loads((folder / TURN_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        record = None
    try:
        result = json.loads((folder / RESULT_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        if record is not None:
            return record
        raise ResearchError(f"no completed result to follow up in {folder}: {e}")
    if record is not None:
        if "retrieved" not in record:  # an older turn record: its result's evidence
            record["retrieved"] = retrieved_passages(
                result.get("sources"), [c["id"] for c in record.get("cited") or []])
        return record
    citations = result.get("citations")
    if citations is None:
        citations = [{"id": p.get("hit_id"), "kind": "passage", "source_id": src["source_id"],
                      "title": src.get("title"), "text": p["text"]}
                     for src in result.get("sources") or []
                     for e in src.get("excerpts") or [] for p in e.get("passages") or []
                     if p.get("text")][:FOLLOW_UP_PASSAGES_MAX]
    follow = ((result.get("details") or {}).get("follow_up") or {}).get("previous_run")
    return turn_record(result.get("question") or "", result.get("question") or "",
                       result.get("answer") or "", citations, folder, follow,
                       result.get("sources"))


def previous_exchange(run_dir):
    """The conversation a follow-up continues, from the run folder of the thread's latest
    answered turn: the last THREAD_TURNS_MAX turn records (oldest first, ids t1..tn), following
    each record's `follow` back. question, short_answer and passages describe the latest turn.
    Raises ResearchError when that folder holds no completed result."""
    turns, folder = [], run_dir
    while folder and len(turns) < THREAD_TURNS_MAX:
        try:
            turns.append(load_turn(folder))
        except ResearchError:
            if not turns:
                raise
            break
        folder = turns[-1].get("follow")
    turns.reverse()
    for n, t in enumerate(turns, 1):
        t["turn_id"] = f"t{n}"
    last = turns[-1]
    return {"run_dir": str(Path(run_dir)), "question": last["standalone"] or last["question"],
            "short_answer": last["short_answer"],
            "passages": [{"hit_id": c["id"], "source_id": c["source_id"],
                          "source_title": c["source_title"], "text": c["text"]}
                         for c in last["cited"]],
            "turns": turns}


def community_block(community):
    """The COMMUNITY RECORDS block of the planner's input (pure): each record's id (s1..,
    as the answer pass numbers them), claim type, claim and a context excerpt, without the
    member's name; None without records."""
    if not community:
        return None
    lines = []
    for n, s in enumerate(community, 1):
        claim = s["claim"]
        context = clean(claim.get("context"), PLANNER_RECORD_CONTEXT_CHARS)
        lines.append(f"[s{n}] ({claim.get('claim_type') or 'unclassified'}) "
                     f"{clean(claim.get('claim_text'), PLANNER_RECORD_CHARS)}"
                     + (f"\n  Context: {context}" if context else ""))
    return ("COMMUNITY RECORDS (what community members wrote; not evidence):\n"
            + "\n".join(lines))


def planner_prompt(question, previous=None, community=None, prior=None):
    """The planner's input (pure): the question, after the thread's earlier turns for a
    follow-up (each with its turn id, question, short answer and section headings), and then
    the community records found for it (see community_block). For a repeated
    question, the EARLIER ANSWER to it (short answer, headings, how many passages it cited)."""
    block = community_block(community)
    tail = f"\n\n{block}" if block else ""
    if prior and not previous:
        tail = (f"\n\nEARLIER ANSWER (this same question, answered before; its "
                f"{len(prior.get('cited') or [])} cited passages are already in the evidence):\n"
                f"Short answer: {prior.get('short_answer') or ''}"
                + (f"\nSections: {'; '.join(prior['headings'])}" if prior.get("headings") else "")
                + tail)
    if not previous:
        return f"USER QUESTION:\n{question}" + tail
    turns = previous.get("turns") or [{"turn_id": "t1", "question": previous["question"],
                                       "short_answer": previous["short_answer"], "headings": []}]
    lines = []
    for t in turns:
        lines.append(f"[{t['turn_id']}] Question: {t['question']}\n"
                     f"Short answer: {t['short_answer']}"
                     + (f"\nSections: {'; '.join(t['headings'])}" if t.get("headings") else "")
                     # What an answer_from_turn follow-up would answer from
                     + f"\nPassages retrieved: {len(t.get('retrieved') or t.get('cited') or [])}")
    return ("EARLIER TURNS (this conversation, oldest first):\n" + "\n\n".join(lines)
            + f"\n\nUSER QUESTION:\n{question}" + tail)


def related_turn(out, previous):
    """The earlier turn the question builds on, per the planner's related_turn (pure), or None.
    A plan without the field relates a follow-up to the latest turn."""
    if not previous:
        return None
    turns = previous.get("turns") or [{"turn_id": "t1", **previous, "headings": [],
                                       "cited": [{"id": p["hit_id"], "source_id": p["source_id"],
                                                  "source_title": p["source_title"],
                                                  "text": p["text"]}
                                                 for p in previous.get("passages") or []]}]
    if not isinstance(out, dict) or "related_turn" not in out:
        return turns[-1]
    wanted = out.get("related_turn")
    return next((t for t in turns if isinstance(wanted, str) and t["turn_id"] == wanted.strip()),
                None)


def standalone_question(question, out, previous=None):
    """The question research uses (pure): the planner's standalone rewrite when the question
    builds on an earlier turn, else the question itself (without earlier turns, or when
    related_turn is null, the planner's field is ignored)."""
    rewrite = out.get("standalone") if isinstance(out, dict) else None
    if (not previous or related_turn(out, previous) is None or not isinstance(rewrite, str)
            or not rewrite.strip()):
        return question
    return " ".join(rewrite.split())


def previous_candidates(previous, related=None, everything=False):
    """The related turn's cited passages as remembered primary passages (pure; see
    merge_memory_primary), at most RELATED_PASSAGES_MAX, in citation order. None without a
    related turn. (offered the previous answer's first FOLLOW_UP_PASSAGES_MAX passages.)
    `everything` (ANSWER_FROM_TURN): every passage the turn retrieved, cited first,
    uncapped (the evidence budget still applies in select)."""
    if related is None:
        return []
    passages = ((related.get("retrieved") or related.get("cited") or []) if everything
                else related.get("cited", [])[:RELATED_PASSAGES_MAX])
    return [{"unit_id": f"prev:{p['id']}", "score": 1.0, "matched": ["(earlier turn)"],
             "text": p["text"], "source_id": p["source_id"], "source_title": p["source_title"]}
            for p in passages]


def already_told(related, candidates, evidence):
    """The ALREADY TOLD THE USER block for the answer pass (pure), or None without a related
    turn: its question, short answer and headings, and the ids its cited passages carry in this
    run's evidence (when they made it in)."""
    if related is None:
        return None
    labeled = evidence_blocks(evidence)
    ids = [c["hit_id"] for c in candidates if c["hit_id"] in labeled
           and any(u.startswith("prev:") for u in c.get("memory_units") or [])]
    return (f"Question (the user's original text): {related['question']}\n"
            f"Short answer: {related['short_answer']}"
            + (f"\nSections: {'; '.join(related['headings'])}" if related.get("headings") else "")
            + (f"\nPassages it cited, as labeled in the evidence below: "
               f"{', '.join(f'[{i}]' for i in ids)}" if ids else ""))


def plan(run, question, community=None):
    """The planner (PLANNER_MODEL) lists the question's answer requirements, picks a research
    depth and writes distinct source-search queries, each naming the requirements it covers;
    Python cleans the plan and enforces the depth's search limit (see normalize_plan). Returns
    (depth, requirements, searches as {"query", "covers"}).

    the plan comes back as JSON text (PLANNER_TEXT_SYSTEM, see unwrap_plan); only an
    unreadable one is asked for again through --json-schema (planner-2)."""
    prompt = planner_prompt(question, run.previous, community, run.prior)
    # Streamed: a long question can need a plan that takes over STAGE_TIMEOUT to write.
    text = claude(run, "planner-1", PLANNER, PLANNER_TEXT_SYSTEM, prompt, stream=True)
    out = unwrap_plan(parse_json_object(str(text)))
    if out is None:
        run.log("planner_text_unreadable", reply=str(text)[:500])
        out = claude(run, "planner-2", PLANNER, PLANNER_SYSTEM, prompt, PLANNER_SCHEMA,
                     stream=True)
    run.save("planner-1.json", json.dumps(out, indent=2, ensure_ascii=False))
    run.standalone = standalone_question(question, out, run.previous)
    run.related = related_turn(out, run.previous)
    run.reuse = normalize_reuse(out.get("reuse"), run.related)
    if run.previous:
        run.log("related_turn", related_turn=(run.related or {}).get("turn_id"),
                offered=[t.get("turn_id") for t in run.previous.get("turns") or []])
    if run.standalone != question:
        run.log("standalone", question=question, standalone=run.standalone)
    question = run.standalone
    run.case_frame_raw = out.get("case_frame")
    run.requested_parts = normalize_requested_parts(out.get("requested_parts"))
    if run.reuse == ANSWER_FROM_TURN:  # At most one ask, for one missing premise
        run.asks = normalize_asks(out.get("asks"), question, ANSWER_FROM_TURN_ASKS, True)
        run.verifications = []
        run.log("reuse", mode=run.reuse, related_turn=run.related.get("turn_id"),
                passages=len(run.related.get("retrieved") or run.related.get("cited") or []),
                premise_asks=run.asks)
    else:
        run.asks = normalize_asks(out.get("asks"), question)
        run.verifications = normalize_verifications(out.get("verifications"), community,
                                                    run.asks)
    if community is not None:  # The planner picks the records the answer uses
        picked, by_planner = pick_community(out.get("community"), community, run.verifications)
        kept = {s["unit_id"] for s in picked}
        run.trace["memory"]["community_pick"] = {
            "pool": len(community), "picked": [s["unit_id"] for s in picked],
            "by_planner": by_planner}
        run.trace["memory"].setdefault("secondary_dropped", []).extend(
            {"unit_id": s["unit_id"], "score": s.get("score"), "key": s.get("key"),
             "reason": "not picked by the planner" if by_planner
                       else f"ranked below the top {MEMORY_SECONDARY_MAX}",
             "preview": preview(s["claim"].get("claim_text") or "")}
            for s in community if s["unit_id"] not in kept)
        run.community_pool = [s["unit_id"] for s in community]
        run.community = picked
    problems = validate_plan(out)
    if problems:
        run.log("planner_validation", problems=problems)
    depth, requirements, searches, notes = normalize_plan(out, question)
    if notes:
        run.log("planner_notes", **notes)
    if not searches:
        run.fail("planner returned no search queries")
    run.save("plan-1.json", json.dumps({"depth": depth, "requirements": requirements,
                                        "searches": searches, "asks": run.asks,
                                        "verifications": run.verifications},
                                       indent=2, ensure_ascii=False))
    return depth, requirements, searches


def search(run, searches, round_n=1, known=(), label=None):
    """Run `notebooklm source search` for every query concurrently; return (raw hits, candidates).

    Raw hits and candidates are ordered by planner query order, then result rank, whatever order
    the searches finish in. Each candidate keeps its source_id, exact text, rank, start/end and
    every query that found it. Exact repeats, and hits of one source covering substantially the
    same span, become one hit. For a repair round, `known` holds the earlier candidates: hits
    repeating one of them are dropped, and new hit ids continue after theirs.

    A query whose identical request has a stored response on the same corpus identity is served
    from the retrieval cache (see retrieval_lookup); only the others call NotebookLM, and their
    successful responses are cached. Provenance goes to the trace, never to a model prompt.
    label names an extra search step within a round (raw ids and the log file), e.g. "1b".
    """
    tag = label or round_n
    # New ids continue after the highest known one (ids are not contiguous after a collapse).
    base = max((int(c["hit_id"][1:]) for c in known if re.fullmatch(r"h\d+", c["hit_id"])),
               default=0)

    def one(item):
        qi, query = item
        start = time.monotonic()
        out = notebooklm(run, f"search {qi}/{len(searches)}", search_args(query))
        return out, time.monotonic() - start

    results, origin = [None] * len(searches), [None] * len(searches)
    for i, query in enumerate(searches):
        start = time.monotonic()
        hit, reason, key = retrieval_lookup(run, query)
        origin[i] = {"origin": "fresh" if hit is None else "cache", "key": key,
                     "seconds": time.monotonic() - start}
        if hit is None:
            origin[i]["reason"] = reason
        else:
            results[i] = (hit["result"], None)
            origin[i].update(run_id=hit["run_id"], retrieved_at=hit["retrieved_at"])
    pending = [(i + 1, q) for i, q in enumerate(searches) if results[i] is None]
    for (qi, _), (out, secs) in zip(pending, parallel(
            run, one, pending, f"Waiting for NotebookLM ({len(pending)} searches)...")):
        results[qi - 1], origin[qi - 1]["seconds"] = out, secs
    if any(err == CANCELLED for _, err in results):
        run.cancelled()
    for qi, (result, err) in enumerate(results, 1):
        if err:
            run.fail(err, provider="notebooklm", kind=classify_failure(err))
        if not isinstance(result, list):
            run.fail(f"NotebookLM search {qi} returned unexpected JSON", provider="notebooklm",
                     kind="provider")
    for qi, query in pending:
        retrieval_store(run, query, origin[qi - 1]["key"], results[qi - 1][0], round_n)

    raw, tally = [], run.trace["retrieval"]
    for qi, (query, (result, _)) in enumerate(zip(searches, results), 1):
        hits = [r for r in result if is_passage(r)]
        hits.sort(key=lambda r: r["rank"] if isinstance(r.get("rank"), int) else 10**9)  # stable
        prov = origin[qi - 1]
        cached = ({"cached_from": prov["run_id"], "retrieved_at": prov["retrieved_at"]}
                  if prov["origin"] == "cache" else {})
        raw += [{"raw_id": f"r{tag}.{len(raw) + i + 1}", "query": query,
                 "source_id": r["source_id"], "text": r["text"], "rank": r.get("rank"),
                 "start": r.get("start"), "end": r.get("end"), "origin": prov["origin"], **cached}
                for i, r in enumerate(hits)]
        run.log("search", round=round_n, n=qi, query=query, hits=len(result),
                origin=prov["origin"], key=prov["key"][:12], seconds=round(prov["seconds"], 2),
                **({"miss": prov["reason"]} if prov.get("reason") else cached))
        tally["queries"] += 1
        tally["from_cache" if cached else "fresh"] += 1
        tally["passages_from_cache" if cached else "passages_fresh"] += len(hits)
        tally["detail"].append({"round": round_n, "query": query, "origin": prov["origin"],
                                "passages": len(hits), "seconds": round(prov["seconds"], 2),
                                **({"miss": prov["reason"]} if prov.get("reason") else cached)})

    def repeats(c, r):
        """Why hit r adds nothing to candidate c (same source), else None. Only repeats that lose
        no text are merged; partially overlapping hits stay separate candidates."""
        return c["source_id"] == r["source_id"] and duplicate_reason(c, r)

    # Each raw hit records its fate: the candidate it became or merged into, and why it merged.
    fresh = []
    for r in raw:
        prior = next(((c, why) for c in known if (why := repeats(c, r))), None)
        if prior:
            r.update(candidate=prior[0]["hit_id"], merge=f"already found ({prior[1]})")
        else:
            fresh.append(r)
    if known:
        run.log("repair_hits", raw_hits=len(raw), already_known=len(raw) - len(fresh))
    candidates = []
    for r in fresh:
        found = {"query": r["query"], "rank": r["rank"], "start": r["start"], "end": r["end"]}
        for c in candidates:
            why = repeats(c, r)
            if why:
                if len(r["text"]) > len(c["text"]):  # keep the fuller text of a contained pair
                    c.update(text=r["text"], start=r["start"], end=r["end"])
                c["found_by"].append(found)
                c["raw_ids"].append(r["raw_id"])
                r.update(candidate=c["hit_id"], merge=why)
                break
        else:
            hit_id = f"h{base + len(candidates) + 1}"
            candidates.append({"hit_id": hit_id, "source_id": r["source_id"],
                               "text": r["text"], "start": r["start"], "end": r["end"],
                               "found_by": [found], "raw_ids": [r["raw_id"]]})
            r.update(candidate=hit_id, merge=None)
    for c in candidates:
        ranks = [f["rank"] for f in c["found_by"] if isinstance(f["rank"], int) and f["rank"] > 0]
        c["rank"] = min(ranks) if ranks else None
    candidates = keep_per_query(run, raw, candidates, base, round_n)
    for c in candidates:
        reason = continuation_reason(c["text"])
        c["needs_continuation"] = reason is not None
        c["continuation_reason"] = reason
    run.log("continuation_candidates",
            hit_ids=[c["hit_id"] for c in candidates if c["needs_continuation"]],
            reasons={c["hit_id"]: c["continuation_reason"]
                     for c in candidates if c["needs_continuation"]},
            starts_mid_list=[c["hit_id"] for c in candidates if fragment_start(c["text"])])
    run.trace["raw"] += [dict(r, round=round_n) for r in raw]
    run.trace["candidates"] += candidates
    run.save(f"search-results-{tag}.json", json.dumps({"queries": searches, "raw_hits": raw,
                                                           "candidates": candidates},
                                                          indent=1, ensure_ascii=False))
    return raw, candidates


def keep_per_query(run, raw, candidates, first, round_n):
    """The candidates some search ranked within PER_QUERY_KEEP (a hit without a rank is kept),
    renumbered h{first+1}... in order; raw hits of a dropped candidate get candidate None and
    merge "beyond PER_QUERY_KEEP"."""
    kept = [c for c in candidates if c["rank"] is None or c["rank"] <= PER_QUERY_KEEP]
    if len(kept) == len(candidates):
        return candidates
    ids = {c["hit_id"]: f"h{first + n}" for n, c in enumerate(kept, 1)}
    dropped = {c["hit_id"] for c in candidates} - set(ids)
    for r in raw:  # this round's raw hits only (earlier rounds' candidates are in `known`)
        if r.get("candidate") in ids:
            r["candidate"] = ids[r["candidate"]]
        elif r.get("candidate") in dropped:
            r.update(candidate=None, merge="beyond PER_QUERY_KEEP")
    for c in kept:
        c["hit_id"] = ids[c["hit_id"]]
    run.log("per_query_keep", round=round_n, keep=PER_QUERY_KEEP, dropped=len(dropped))
    return kept


QUESTION_LINE = re.compile(r"(?:Q|Question)\s*:", re.I)  # a questioner's line in a transcript
# Another labeled speaker's question line (not the answerer: A: or the corpus author's name).
SPEAKER_QUESTION = re.compile(rf"(?!A\s*:|(?:{settings.AUTHOR_NAMES})\b)"
                              r"[A-Z][\w .'()\[\]-]{0,30}:\s.*\?[\"')\]]*$", re.I)
ANSWER_ON_LINE = re.compile(r"\sA\s*:")
ENDS_WITH_QUESTION = re.compile(r"\?[\"')\]]*$")
SECTION_LINE = re.compile(r"#{1,6}\s|[=*_-]{3,}$")
# A final line ending like this was cut mid-sentence (no terminal punctuation).
DANGLING = re.compile(r"(?:[,:;(]|\b(?:and|or|but|because|that|which|the|a|an|of|to|with|for|if|"
                      r"when|than))$", re.I)


def continuation_reason(text):
    """Why a search hit's text appears to stop before its answer or explanation, else None.

    Only the final non-empty line is judged, so a complete passage with a question somewhere
    earlier is not marked: it ends on a Q:/Question: line (with no A: on it), on an empty A:, on
    another question (not an A: line), or mid-sentence after a comma/colon or a function word.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    last = lines[-1]
    if QUESTION_LINE.match(last) and not ANSWER_ON_LINE.search(last):
        return "ends_on_question_line"
    if re.fullmatch(r"A\s*:", last):
        return "ends_on_empty_answer"
    if ENDS_WITH_QUESTION.search(last) and not re.match(r"A\s*:", last):
        return "ends_on_question"
    if DANGLING.search(last):
        return "truncated_sentence"
    return list_fragment_end(lines)


# List fragments. Search chunks split recipes and lists mid-way: an ingredient, bulleted
# or numbered line has no dangling word, so the checks above never saw them.
QUANTITY_UNIT = (r"(?:cups?|tablespoons?|teaspoons?|tbsps?|tsps?|ounces?|oz|pounds?|lbs?|"
                 r"pinch(?:es)?|slices?|whole|cloves?|stalks?|inch(?:es)?|dash(?:es)?|drops?|"
                 r"grams?|ml|quarts?|pints?)")
LIST_LINE = re.compile(
    r"(?:[-*•·]\s+\S"  # bullet
    r"|\d{1,3}[.)]\s+\S"  # numbered item
    # a whole-bold line that starts with an amount (the recipe books' ingredient lines)
    r"|\*\*\s*(?:[\d½¼¾⅓⅔⅛�]|(?:one|two|three|half|a few|a pinch|a dash)\b)[^*]*\*\*$"
    r"|(?:\d+(?:[/.]\d+)?|[½¼¾⅓⅔])(?:\s*(?:-|to)\s*\d+(?:/\d+)?)?\s*" + QUANTITY_UNIT + r"\b)",
    re.I)
LIST_HEADING = re.compile(r"#{1,6}\s|\*\*[^*]+\*\*$|\d+\s+servings?\b|serves\s+\d|[=*_-]{3,}$", re.I)
# A method that ends like this is finished (storage or serving note).
METHOD_END = re.compile(r"\b(?:will keep|keeps? (?:in|for|up)|store[sd]? (?:in|for)|serve[sd]?\b|"
                        r"refrigerat)", re.I)
LIST_RUN = 3  # consecutive list lines that make a recipe's ingredient list


def is_list_line(line):
    return bool(LIST_LINE.match(line))


def list_fragment_end(lines):
    """Why a chunk (its non-empty, stripped lines) stops inside a list or recipe, else None:
    "ends_in_list" when its last two lines are list items; "recipe_method_open" when it ends in
    the method text after an ingredient list (LIST_RUN+ items) with no heading after it and no
    closing storage/serving note, so the rest of the method may follow in the source."""
    if len(lines) >= 2 and is_list_line(lines[-1]) and is_list_line(lines[-2]):
        return "ends_in_list"
    run_end, run = None, 0
    for n, line in enumerate(lines):
        run = run + 1 if is_list_line(line) else 0
        if run >= LIST_RUN:
            run_end = n
    if run_end is None or run_end == len(lines) - 1:
        return None
    method = lines[run_end + 1:]
    if any(is_list_line(x) or LIST_HEADING.match(x) or SECTION_LINE.match(x) for x in method):
        return None
    return None if METHOD_END.search(method[-1]) else "recipe_method_open"


def fragment_start(text):
    """"starts_mid_list" when a chunk begins inside a list: its first two non-empty lines are
    list items and no heading precedes them, else None. Only forward continuation exists, so the
    text before such a chunk (a recipe's title, its first items) is not recovered; the evidence
    says so instead."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 2 and is_list_line(lines[0]) and is_list_line(lines[1]):
        return "starts_mid_list"
    return None


CONTAIN_SLACK = 20  # offset tolerance when one hit's span lies inside another's


def duplicate_reason(a, b):
    """Why two hits of the same source are one passage, else None: identical text, one text
    inside the other, or one span inside the other. Hits that only partly overlap are not
    duplicates: merging them would drop the part of the shorter hit outside the longer one."""
    ta, tb = a["text"].strip(), b["text"].strip()
    if ta == tb:
        return "identical text"
    if (ta in tb) or (tb in ta):
        return "contained text"
    if None in (a["start"], a["end"], b["start"], b["end"]):
        return None
    inner, outer = (a, b) if a["end"] - a["start"] <= b["end"] - b["start"] else (b, a)
    if (inner["start"] >= outer["start"] - CONTAIN_SLACK
            and inner["end"] <= outer["end"] + CONTAIN_SLACK):
        return "contained span"
    return None


# Cross-source duplicates: normalized passages shorter than this are too generic to call
# the same passage when they appear in two source files.
CROSS_SOURCE_MIN_CHARS = 120


def source_specificity(title):
    """How specific a source is (pure): 2 for a dated one (a Q&A or workshop), 1 for a book or
    topical compilation, 0 for a general compilation (Misc, Questions And Answers, ...)."""
    if source_date(title):
        return 2
    base = re.sub(r"\.(md|txt)$", "", (title or "").strip(), flags=re.I)
    return 0 if base in COMPILATIONS else 1


def cross_source_relation(a, b, norm):
    """"identical" or "contained" when two candidates from different source files carry the
    same passage (normalized text equal, or one inside the other), else None (pure)."""
    if a["source_id"] == b["source_id"]:
        return None
    na, nb = norm[a["hit_id"]], norm[b["hit_id"]]
    if min(len(na), len(nb)) < CROSS_SOURCE_MIN_CHARS:
        return None
    if na == nb:
        return "identical"
    return "contained" if na in nb or nb in na else None


def collapse_cross_source(candidates, titles, known=()):
    """(kept candidates, collapsed records) (pure): the same passage found in different source
    files (a dated Q&A and a compilation that reprints it) becomes one candidate. The more
    specific source is kept (see source_specificity; on a tie the fuller text, then the earlier
    candidate); it takes over the other's search provenance and lists it under "alternates".
    Candidates that duplicate one in `known` (an earlier round's) are dropped in its favor."""
    norm = {c["hit_id"]: grounding_norm(c["text"]) for c in [*known, *candidates]}
    kept, collapsed = [], []

    def absorb(winner, loser, relation):
        winner.setdefault("alternates", []).append(
            {"hit_id": loser["hit_id"], "source_id": loser["source_id"],
             "title": titles.get(loser["source_id"])})
        winner["alternates"] += loser.get("alternates") or []
        winner["found_by"] = winner.get("found_by", []) + (loser.get("found_by") or [])
        winner["raw_ids"] = winner.get("raw_ids", []) + (loser.get("raw_ids") or [])
        for key in ("facts", "memory_units"):  # The fact questions it answers
            if loser.get(key):
                winner[key] = list(dict.fromkeys((winner.get(key) or []) + loser[key]))
        ranks =[r for r in (winner.get("rank"), loser.get("rank")) if isinstance(r, int)]
        winner["rank"] = min(ranks) if ranks else None
        collapsed.append({"kept": winner["hit_id"], "kept_source": titles.get(winner["source_id"]),
                          "dropped": loser["hit_id"],
                          "dropped_source": titles.get(loser["source_id"]), "relation": relation})

    for c in candidates:
        prior = next(((k, rel) for k in known if (rel := cross_source_relation(k, c, norm))), None)
        if prior:
            absorb(prior[0], c, prior[1])
            continue
        match = next(((n, k, rel) for n, k in enumerate(kept)
                      if (rel := cross_source_relation(k, c, norm))), None)
        if not match:
            kept.append(c)
            continue
        n, k, rel = match
        rank = lambda h: (source_specificity(titles.get(h["source_id"])), len(norm[h["hit_id"]]))
        if rank(c) > rank(k):
            absorb(c, k, rel)
            kept[n] = c
        else:
            absorb(k, c, rel)
    return kept, collapsed


def collapse_duplicates(run, candidates, titles, round_n=1, known=()):
    """collapse_cross_source on this round's candidates, in place, with the collapsed pairs and
    the candidate size before and after in the trace (evidence-trace.json "cross_source")."""
    before = sum(len(c["text"]) for c in candidates)
    kept, collapsed = collapse_cross_source(candidates, titles, known)
    if collapsed:
        dropped = {x["dropped"]: x["kept"] for x in collapsed}
        for r in run.trace["raw"]:
            if r.get("round") == round_n and r.get("candidate") in dropped:
                r.update(candidate=dropped[r["candidate"]], merge="cross-source duplicate")
        candidates[:] = kept
    info = {"round": round_n, "candidates_before": len(kept) + len(collapsed),
            "candidates_after": len(kept), "chars_before": before,
            "chars_after": sum(len(c["text"]) for c in kept), "collapsed": collapsed}
    run.trace.setdefault("cross_source", []).append(info)
    run.log("cross_source_duplicates", **info)
    return collapsed


# Referenced recipes: an ingredient line naming a sub-recipe in capitals ("1 tablespoon
# TRAVEL REQUEST", "2 ounces SPICE MIX"). The name is 1-4 capitalized words after an amount and at
# most three unit words.
# Ingredient lists are often flattened onto one line or wrapped in bold, so an amount is matched
# anywhere after a line start, whitespace or markup; "TRAVEL REQUEST, or SPICE MIX" names both.
CAPS_NAME = r"[A-Z][A-Z'&-]{2,}(?:[ \t]+[A-Z][A-Z'&-]+){0,3}"
SUBRECIPE_LINE = re.compile(
    r"(?:^|(?<=[\s*_(]))(?:\d+(?:[.,/]\d+)?|[½¼¾⅓⅔]|(?i:one|two|three|four|half|a|an))"
    r"(?:[ \t]*[-–][ \t]*\d+(?:[./]\d+)?)?[ \t]*[½¼¾⅓⅔]?[ \t]+(?:[A-Za-z.]{1,12}[ \t]+){0,3}?"
    rf"(?P<name>{CAPS_NAME})\b(?:,?[ \t]+or[ \t]+(?P<alt>{CAPS_NAME})\b)?", re.M)
UNIT_WORDS = {"TBSP", "TBS", "TSP", "OZ", "OZS", "LB", "LBS", "CUP", "CUPS", "QT", "PT", "ML",
              "TABLESPOON", "TABLESPOONS", "TEASPOON", "TEASPOONS", "OUNCE", "OUNCES", "POUND",
              "POUNDS", "GRAMS", "PINCH", "DASH", "DRIED", "FRESH", "THE", "AND", "OF", "OR"}
REFERENCED_RECIPES_MAX = 3


def referenced_subrecipes(texts):
    """Sub-recipe names written in capitals in ingredient lines of `texts` that no text has a
    heading for (pure), in order of first appearance. A heading is a short line that starts with
    the name (optionally markdown or bold, "recipe", or a date after it)."""
    names = []
    for text in texts:
        for m in SUBRECIPE_LINE.finditer(text or ""):
            for found in (m.group("name"), m.group("alt")):
                words = (found or "").split()
                while words and words[0] in UNIT_WORDS:
                    words = words[1:]
                name = " ".join(words)
                # a single word of 3 letters or fewer, or one without vowels, is an abbreviation
                # (ACV, TRFLWD), not a recipe
                abbreviation = len(words) == 1 and not re.search(r"[AEIOUY]", name)
                if (len(name) >= 4 and not abbreviation and name not in UNIT_WORDS
                        and name not in names):
                    names.append(name)

    def headed(name):
        heading = re.compile(rf"^[\s#*_>]*{re.escape(name)}\b[^\n]{{0,40}}$", re.I | re.M)
        return any(not SUBRECIPE_LINE.search(line.group(0))
                   for text in texts for line in heading.finditer(text or ""))
    return [n for n in names if not headed(n)]


def referenced_recipes(run, candidates, titles, depth, done_queries):
    """Search once for each sub-recipe an ingredient line names in capitals when the evidence
    has no recipe of that name (see referenced_subrecipes): "<NAME> recipe", at most
    REFERENCED_RECIPES_MAX in parallel, at any depth. New hits get their continuation and join
    `candidates` before selection. Returns the new raw hits (for memory capture)."""
    texts = [t for c in candidates for t in (c.get("backward_text"), c["text"],
                                             c.get("continuation_text"))]
    names = referenced_subrecipes(texts)
    queries = []
    for n in names:
        q = f"{n} recipe"
        if not repeats_query(q, done_queries + queries):
            queries.append(q)
    queries = queries[:REFERENCED_RECIPES_MAX]
    run.trace["referenced_recipes"] = {"names": names, "queries": queries, "new_candidates": []}
    if not queries:
        return []
    t = run.begin("search", "Searching referenced recipes")
    raw, fresh = search(run, queries, known=candidates, label="1b")
    collapse_duplicates(run, fresh, titles, known=candidates)
    if any(c["needs_continuation"] or fragment_start(c["text"]) for c in fresh):
        hydrate(run, fresh, titles, round_n="1b", budget=EVIDENCE_BUDGET[depth],
                prior_chars=sum(hit_chars(c) for c in candidates))
    candidates.extend(fresh)
    run.trace["referenced_recipes"]["new_candidates"] = [c["hit_id"] for c in fresh]
    run.trace["referenced_recipes"]["hits"] = {
        n: [c["hit_id"] for c in fresh if any(f["query"] == f"{n} recipe" for f in c["found_by"])]
        for n in names}
    secs = run.timed("Referenced recipes", t, sum(len(c["text"]) for c in fresh))
    run.done("search", secs, detail=f"{len(queries)} referenced "
             + ("recipe" if len(queries) == 1 else "recipes") + f", {len(fresh)} new passages",
             summary=f"Referenced recipes: {secs}; {', '.join(queries)}; {len(fresh)} new candidates")
    run.log("referenced_recipes", **run.trace["referenced_recipes"])
    return raw


def referenced_text(run, evidence):
    """The REFERENCED RECIPES lines for the answer pass (each sub-recipe searched for and the
    ids its search contributed to the evidence), or None."""
    hits = (run.trace.get("referenced_recipes") or {}).get("hits") or {}
    labeled = evidence_blocks(evidence)
    lines = [f"- {name}: " + ", ".join(f"[{i}]" for i in ids if i in labeled)
             for name, ids in hits.items() if any(i in labeled for i in ids)]
    return "\n".join(lines) or None


def requirement_line(r):
    """One requirement (or follow-up premise, tagged with the requirement it serves) as a line.
    A requirement whose answer needs exact details (amounts, components) says so."""
    tag = f"for {r['for']}" if r.get("for") else r["kind"]
    if r.get("exact"):
        tag += "; exact details needed"
    return f"- {r['id']} [{tag}] {r['text']}"


def squash(text):
    """Lowercase text with runs of whitespace and punctuation collapsed, for verbatim checks."""
    return " ".join(re.sub(r"[^\w%./-]+", " ", str(text or "").lower()).split())


def claim_structure(d, hit, kept_ids):
    """The claim structure of one kept hit's selector decision (pure): the source's predicate and
    whether it occurs verbatim in the passage or its continuation, scope, applies_to, use, and
    relations to other kept hits (unknown types, self-links, links to hits that were not kept, and
    repeated links are dropped and listed in `rejected`)."""
    predicate = clean(d.get("predicate"), 120)
    text = squash(hit["text"] + " " + (hit.get("continuation_text") or ""))
    scope = d.get("scope") if d.get("scope") in SCOPES else "unclear"
    relations, rejected, seen = [], [], set()
    for rel in d.get("relations") or []:
        if isinstance(rel, str):  # the selector's "type:hit_id" form
            kind, _, target = rel.partition(":")
            rel = {"type": kind.strip().lower(), "hit_id": target.strip()}
        if not isinstance(rel, dict):
            continue
        key = (rel.get("type"), rel.get("hit_id"))
        if (key[0] not in RELATION_TYPES or key[1] not in kept_ids or key[1] == hit["hit_id"]
                or key in seen):
            rejected.append({"type": key[0], "hit_id": key[1]})
            continue
        seen.add(key)
        relations.append({"type": key[0], "hit_id": key[1]})
    return {"predicate": predicate,
            "predicate_verbatim": bool(predicate) and squash(predicate) in text,
            "scope": scope, "applies_to": "" if scope == "general" else clean(d.get("applies_to"), 120),
            "use": d.get("use") if d.get("use") in USES else "answer",
            "relations": relations, "rejected": rejected}


def normalize_selection(out, candidates, requirements, follow_up=False):
    """Clean the selector's output (pure: select() and the tests use it). Returns (kept hit ids in
    priority order, context hit ids, decisions by hit id, coverage per requirement, notes as
    (event, data) pairs for the run log).

    A hit the selector itself classed DROP is never kept, and only a kept hit can cover a
    requirement: coverage lists kept hits only (falling back to the kept hits whose decisions
    name the requirement), and "covered" or "partial" with no kept hit behind it becomes
    "missing". A requirement the selector did not rate is "unassessed".

    Every kept hit's decision also carries its claim structure (see claim_structure). A partial or
    missing requirement keeps the kind of gap and any lead hits that point at the missing detail
    ("gap" and "leads", present only when the selector named a gap).
    """
    notes = []
    by_id = {c["hit_id"]: c for c in candidates}
    valid = [r["id"] for r in requirements]
    decisions = {}
    for d in out.get("decisions") or []:
        if not isinstance(d, dict) or d.get("hit_id") not in by_id or d["hit_id"] in decisions:
            continue
        role = d.get("role") if d.get("role") in SELECTOR_ROLES else None
        covers = [] if role == "DROP" else [c for c in dict.fromkeys(d.get("covers") or [])
                                            if c in valid]
        decisions[d["hit_id"]] = {"role": role, "covers": covers, "reason": clean(d.get("reason"), 200)}
    claims = {}
    for c in (out.get("claims") or []) + (out.get("decisions") or []):
        # Claim fields come from "claims"; a decision carrying them is accepted as well.
        if isinstance(c, dict) and c.get("hit_id") in by_id and "predicate" in c:
            claims.setdefault(c["hit_id"], c)
    ids = [i for i in dict.fromkeys(out.get("selected_hit_ids") or []) if i in by_id]
    dropped = [i for i in ids if (decisions.get(i) or {}).get("role") == "DROP"]
    if dropped:  # a hit the selector itself classed DROP does not reach the reasoner
        notes.append(("selector_drop_conflict", {"hit_ids": dropped}))
        ids = [i for i in ids if i not in dropped]
    if not ids and not follow_up:
        notes.append(("selector_fallback", {"reason": "no valid hit ids selected"}))
        ids = list(by_id)
    for i in ids:  # claim structure, now that the kept set is known
        structure = claim_structure(claims.get(i) or {}, by_id[i], set(ids))
        rejected = structure.pop("rejected")
        if rejected:
            notes.append(("selector_relations_rejected", {"hit_id": i, "relations": rejected}))
        decisions.setdefault(i, {"role": None, "covers": [], "reason": ""}).update(structure)
    wanted = set(out.get("context_hit_ids") or [])
    context_ids = [i for i in ids if i in wanted]
    ignored = sorted(wanted - set(context_ids))
    if ignored:
        notes.append(("selector_context_ignored",
                      {"reason": "not in selected_hit_ids", "hit_ids": ignored}))

    given = {}
    for c in out.get("coverage") or []:
        if isinstance(c, dict) and c.get("requirement_id") in valid:
            given.setdefault(c["requirement_id"], c)
    coverage = []
    for r in requirements:
        serving = [i for i in ids if r["id"] in (decisions.get(i) or {}).get("covers", [])]
        c = given.get(r["id"])
        if c is None:
            status, hit_ids, missing = "unassessed", serving, ""
        else:
            status = c.get("status") if c.get("status") in COVERAGE_STATUSES else "unassessed"
            hit_ids = [i for i in dict.fromkeys(c.get("hit_ids") or []) if i in ids] or serving
            missing = clean(c.get("missing"), 200)
            if status in ("covered", "partial") and not hit_ids:
                notes.append(("selector_coverage_unsupported",
                              {"requirement_id": r["id"], "status": status}))
                status, missing = "missing", missing or "no kept passage supports it"
        entry = {"requirement_id": r["id"], **({"for": r["for"]} if r.get("for") else {}),
                 "status": status, "hit_ids": hit_ids,
                 "missing": "" if status == "covered" else missing}
        gap = c.get("gap") if c is not None and c.get("gap") in GAP_KINDS else "none"
        if status in ("partial", "missing") and gap != "none":
            entry["gap"] = gap
            entry["leads"] = [i for i in dict.fromkeys(c.get("lead_hit_ids") or []) if i in by_id]
        coverage.append(entry)
    return ids, context_ids, decisions, coverage, notes


def parse_json_object(text):
    """The JSON object in a model's text reply (bare, or inside a ```json fence), or None."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        out = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return out if isinstance(out, dict) else None


def gemini_usage(worker, stage, task, since):
    """The Antigravity worker's own record of one runtime call (it logs every call it makes,
    matched here by stage, request size and start time), or None. Its worker_tokens are the raw
    Antigravity fields (input_tokens, output_tokens, thinking_tokens, cache_read_tokens,
    total_tokens); the worker code does not define how they overlap, so they are passed on as
    reported."""
    try:
        size = len(task.encode("utf-8"))
        for e in reversed(worker.read_usage()[-50:]):
            if (e.get("stage") == stage
                    and e.get("input_bytes") == size and isinstance(e.get("ts"), (int, float))
                    and e["ts"] >= since - 1):
                return e
    except Exception:  # noqa: BLE001 - usage is best effort
        pass
    return None


# Process-wide circuit breaker for the runtime Gemini selector. The ~2.5-minute waits
# came from Antigravity itself retrying a 429 quota error for about two minutes before exiting
# with code 3 (classified "network"), once per selector round. After such a failure, or a timeout
# (also "network"), Gemini is skipped for the rest of the run and for GEMINI_COOLDOWN_SECONDS.
GEMINI_BREAKER_FAILURES = ("network", "timeout")
GEMINI_BREAKER = {"open_until": 0.0, "failure": None}
GEMINI_BREAKER_LOCK = threading.Lock()


def gemini_breaker_trip(failure, now=None):
    with GEMINI_BREAKER_LOCK:
        GEMINI_BREAKER["open_until"] = (time.time() if now is None else now) + GEMINI_COOLDOWN_SECONDS
        GEMINI_BREAKER["failure"] = failure


def gemini_breaker_state(run=None, now=None):
    """{"state": "closed" | "open" (cooldown) | "open_run" (tripped earlier in this run), ...}."""
    now = time.time() if now is None else now
    with GEMINI_BREAKER_LOCK:
        until, failure = GEMINI_BREAKER["open_until"], GEMINI_BREAKER["failure"]
    if run is not None and getattr(run, "gemini_tripped", False):
        return {"state": "open_run", "failure": failure}
    if now < until:
        return {"state": "open", "failure": failure, "seconds_left": round(until - now)}
    return {"state": "closed"}


def gemini_selector(run, stage, prompt):
    """One Gemini Flash attempt at a selector stage (see gemini_task). Returns the selector
    output, or None, after which the caller runs the Haiku selector."""
    def parse(text):
        out = parse_json_object(text)
        if out is None:
            return None, "invalid_json"
        if not (isinstance(out.get("decisions"), list) and isinstance(out.get("coverage"), list)):
            return None, "unusable_output"
        return out, None
    task = (f"{SELECTOR_SYSTEM}\n\nReturn only one JSON object (no prose, no code fence) "
            f"matching this JSON schema:\n{json.dumps(SELECTOR_SCHEMA)}\n\n{prompt}")
    out = gemini_task(run, stage, "selector", task, parse)
    run.selector_fallback = run.gemini_fallback
    return out


def gemini_task(run, stage, mode, task, parse):
    """One Gemini Flash attempt at a mechanical stage (selector, fact-memory match, coverage
    check) through the product's Gemini runtime (server/connections.run_gemini). `parse(text)`
    returns (output, None) or (None, failure). Returns the output, or None when Gemini is
    unavailable or fails, after which the caller runs its Haiku fallback; run.gemini_fallback
    then says why. Never raises: Gemini is optional. Its usage row carries the worker's raw token
    fields (see gemini_usage) and no cost; a failed attempt that reached Antigravity keeps its
    row, marked failed."""
    run.gemini_fallback = None
    breaker = gemini_breaker_state(run)
    run.trace["gemini_breaker"].append({"stage": stage, **breaker})
    if breaker["state"] != "closed":
        run.gemini_fallback = f"Gemini skipped (breaker {breaker['state']}); Haiku handled"
        run.log("gemini_skipped", stage=stage, **breaker)
        return None
    start, since = time.monotonic(), time.time()
    model = worker = None
    try:
        from server import connections
        worker = connections.worker
        model = worker.model()
        run.log("gemini_start", stage=stage, model=model,
                timeout_seconds=GEMINI_SELECTOR_TIMEOUT_SECONDS)
        # The shared Antigravity worker reads its per-call timeout from the environment; this
        # call site caps the runtime attempt without changing the worker's default.
        previous = os.environ.get("GEMINI_WORKER_TIMEOUT_SECONDS")
        os.environ["GEMINI_WORKER_TIMEOUT_SECONDS"] = str(GEMINI_SELECTOR_TIMEOUT_SECONDS)
        try:
            text = connections.run_gemini(mode, task, {"stage": stage})
        finally:
            if previous is None:
                os.environ.pop("GEMINI_WORKER_TIMEOUT_SECONDS", None)
            else:
                os.environ["GEMINI_WORKER_TIMEOUT_SECONDS"] = previous
        run.save(f"{stage}.gemini.txt", text)
        out, failure = parse(text)
    except Exception as e:  # noqa: BLE001 - any Gemini failure falls back to Haiku
        out = None
        failure = getattr(e, "kind", None) or ("unavailable" if model is None else "error")
        run.log("gemini_error", stage=stage, error=repr(e)[:300])
    seconds = round(time.monotonic() - start, 1)
    record = gemini_usage(worker, stage, task, since) if worker is not None else None
    usage = {"stage": stage, "models": (record or {}).get("worker_models") or [model],
             "usage": (record or {}).get("worker_tokens"), "cost_usd": None,
             "seconds": seconds, "provider": "gemini", "wall_seconds": seconds}
    if out is None:
        run.gemini_fallback = f"Gemini attempted ({failure}); Haiku handled"
        run.log("gemini_fallback", stage=stage, model=model, failure=failure, seconds=seconds)
        if failure in GEMINI_BREAKER_FAILURES:
            run.gemini_tripped = True
            gemini_breaker_trip(failure)
            run.trace["gemini_breaker"][-1].update(tripped=failure,
                                                   open_until=GEMINI_BREAKER["open_until"])
            run.log("gemini_breaker_open", stage=stage, failure=failure,
                    cooldown_seconds=GEMINI_COOLDOWN_SECONDS)
        if record is not None:  # the call reached Antigravity: its time and tokens still count
            run.usage.append({**usage, "failed": failure})
        return None
    run.usage.append(usage)
    run.log("gemini_done", **usage)
    return out


def bypass_decision(candidates, depth, prior_chars=0):
    """("bypassed" | "run", candidate chars, budget) for one selector round (pure): bypassed when
    everything that would reach the reasoner (these candidates with their continuation text and
    headers, plus `prior_chars` already in the evidence) fits the depth's evidence budget."""
    chars = prior_chars + sum(hit_chars(c) for c in candidates)
    budget = EVIDENCE_BUDGET[depth]
    return ("bypassed" if candidates and chars <= budget else "run"), chars, budget


def bypass_order(candidates):
    """Candidates in retrieval order, grouped by source (sources in order of their first hit)."""
    ranked = sorted(candidates, key=lambda c: (c["rank"] is None, c["rank"] or 0))
    first = {}
    for n, c in enumerate(ranked):
        first.setdefault(c["source_id"], n)
    return sorted(ranked, key=lambda c: first[c["source_id"]])


def pass_through(candidates, requirements, follow_up=False):
    """The selector's result shape without a selector (pure): every candidate kept in bypass
    order, role "unjudged", no context requests, and every requirement "unassessed"."""
    ordered = bypass_order(candidates)
    ids, context_ids, decisions, coverage, _ = normalize_selection(
        {"selected_hit_ids": [c["hit_id"] for c in ordered]}, candidates, requirements, follow_up)
    for d in decisions.values():
        d["role"] = "unjudged"
    return ids, context_ids, decisions, coverage


def trim_order(candidates):
    """Candidates in trim priority (pure): round-robin by per-query rank (every query's rank 1,
    then rank 2, ...; a hit found by several queries counts at its best rank), fresh hits before
    memory-only ones at equal rank, then retrieval order. Memory-only hits (remembered or
    established passages, which have no search rank) are ranked by their own order."""
    memory_rank = {}
    for c in candidates:
        if c.get("origin") in ("memory", "established") or c["rank"] is None:
            memory_rank[c["hit_id"]] = len(memory_rank) + 1
    position = {c["hit_id"]: n for n, c in enumerate(candidates)}

    def key(c):  # A passage an earlier answer cited ranks like a fresh hit
        i = c["hit_id"]
        return (memory_rank.get(i, c["rank"]), i in memory_rank and not c.get("cited_before"),
                position[i])
    return sorted(candidates, key=key)


def trim_candidates(candidates, budget):
    """(kept, dropped hit ids) (pure): candidates in trim_order, dropped from the tail until what
    remains fits `budget`. A hit is measured with its continuation text (hit_chars), so the two
    are kept or dropped together."""
    ordered = trim_order(candidates)
    total = sum(hit_chars(c) for c in ordered)
    dropped = []
    while ordered and total > budget:
        c = ordered.pop()
        total -= hit_chars(c)
        dropped.append(c["hit_id"])
    return ordered, dropped


def bypassed(run, round_n=None):
    """Whether the selector was bypassed (in round `round_n`, or in any round): no selector ran,
    and the candidates passed through unjudged, all of them or trimmed to the budget."""
    return any(m["mode"] in ("bypassed", "trimmed") and round_n in (None, m["round"])
               for m in run.trace["selector_mode"])


def trimmed(run, round_n=None):
    """Whether candidates were dropped to fit the budget (in round `round_n`, or in any round)."""
    return any(m["mode"] == "trimmed" and round_n in (None, m["round"])
               for m in run.trace["selector_mode"])


def select(run, question, depth, candidates, titles, requirements, round_n=1, follow_up=False,
           prior_chars=0):
    """Judge the candidates against the answer requirements; return (selected hits, context hit ids, coverage).

    With RUNTIME_SELECTOR_ENABLED off, candidates over the budget are trimmed and the rest pass
    through unjudged."""
    mode, chars, budget = bypass_decision(candidates, depth, prior_chars)
    dropped = []
    if mode == "run" and not RUNTIME_SELECTOR_ENABLED:
        _, dropped = trim_candidates(candidates, budget - prior_chars)
        mode = "trimmed" if dropped else "bypassed"
        candidates = [c for c in candidates if c["hit_id"] not in dropped]
    run.trace["selector_mode"].append({"round": round_n, "mode": mode, "candidate_chars": chars,
                                       "budget": budget, "candidates": len(candidates) + len(dropped),
                                       "dropped": dropped})
    run.log("selector_mode", round=round_n, mode=mode, candidate_chars=chars, budget=budget,
            dropped=dropped)
    if mode in ("bypassed", "trimmed"):
        ids, context_ids, decisions, coverage = pass_through(candidates, requirements, follow_up)
        by_id = {c["hit_id"]: c for c in candidates}
        for i in ids:
            by_id[i]["unjudged"] = True
        run.claims.update({i: decisions[i] for i in ids})
        run.log("selector_coverage", round=round_n, coverage=coverage, bypassed=True)
        return [by_id[i] for i in ids], context_ids, coverage
    def entry(c):
        head = f"[{c['hit_id']}] source: {titles.get(c['source_id']) or c['source_id']}"
        text = c["text"].strip()
        if c.get("backward_text"):
            text = ("[preceding: exact source text immediately before this hit]\n"
                    + c["backward_text"] + "\n[hit]\n" + text)
        if c.get("continuation_text"):
            text += ("\n[continuation: exact source text immediately following this hit]\n"
                     + c["continuation_text"])
        elif c.get("continuation_status") == "source_ends":
            text += "\n[continuation: the source ends here]"
        elif c["needs_continuation"]:
            text += ("\n[continuation: this hit stops before its answer; the following source text "
                     "could not be retrieved]")
        return f"{head}\n{text}"

    listing = "\n\n".join(entry(c) for c in candidates)
    header = ("MISSING PREMISES (this is a follow-up search for them; covers and coverage use "
              "these ids)" if follow_up else
              "ANSWER REQUIREMENTS (covers and coverage use these ids)")
    needs = "\n".join(requirement_line(r) for r in requirements)
    prompt = (f"USER QUESTION:\n{question}\n\nRESEARCH DEPTH: {depth}\n\n{header}:\n{needs}\n\n"
              f"CANDIDATE SEARCH HITS:\n{listing}")
    stage = f"selector-{round_n}"
    run.save(f"{stage}.input.txt", prompt)
    out = gemini_selector(run, stage, prompt)
    if out is None:
        out = claude(run, stage, SELECTOR, SELECTOR_SYSTEM, prompt, SELECTOR_SCHEMA)
        if run.selector_fallback:
            run.usage[-1]["fallback"] = run.selector_fallback
    run.save(f"{stage}.json", json.dumps(out, indent=2, ensure_ascii=False))
    ids, context_ids, decisions, coverage, notes = normalize_selection(out, candidates,
                                                                     requirements, follow_up)
    for event, data in notes:
        run.log(event, **data)
    flagged = [c["hit_id"] for c in candidates if c["needs_continuation"]]
    run.log("selector_continuation", flagged=flagged, selected=[i for i in ids if i in flagged],
            context=[i for i in context_ids if i in flagged],
            rejected=[i for i in flagged if i not in ids])
    run.log("selector_coverage", round=round_n, coverage=coverage)
    run.trace["selector"] += [
        {"round": round_n, "hit_id": c["hit_id"], "kept": c["hit_id"] in ids,
         "context": c["hit_id"] in context_ids, **(decisions.get(c["hit_id"]) or {})}
        for c in candidates]
    run.claims.update({i: decisions.get(i) or {} for i in ids})
    by_id = {c["hit_id"]: c for c in candidates}
    return [by_id[i] for i in ids], context_ids, coverage


def cache_path(source_id):
    return CACHE_DIR / (re.sub(r"[^A-Za-z0-9_-]", "_", source_id) + ".txt")


def cached_fulltext(source_id):
    """The cached fulltext of a source, or None when there is no valid (non-empty) entry."""
    try:
        content = cache_path(source_id).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return content if content.strip() else None


def fetch_fulltext(run, source_id):
    """Download one source's fulltext into the cache (worker thread); return (content, title, error).

    The file is written under a temporary name and moved into place only when non-empty, so a
    failed or empty fetch never becomes a cache entry.
    """
    final = cache_path(source_id)
    part = final.with_name(f"{final.stem}.{os.getpid()}.{threading.get_ident()}.part.txt")
    meta, err = notebooklm(run, f"fulltext {source_id[:8]}",
                           ["source", "fulltext", source_id, "-n", NOTEBOOK, "-o", str(part),
                            "--force", "--json"])
    try:
        content = None if err else part.read_text(encoding="utf-8")
        if not err and not content.strip():
            err = f"NotebookLM fulltext {source_id[:8]} returned empty text"
        if err:
            return None, None, err
        os.replace(part, final)
    except (OSError, UnicodeDecodeError) as e:
        return None, None, f"NotebookLM fulltext {source_id[:8]}: {e}"
    finally:
        part.unlink(missing_ok=True)
    title = meta.get("title") if isinstance(meta, dict) else None
    return content, title if isinstance(title, str) and title.strip() else None, None


def list_titles(run):
    """Every source's title from `notebooklm source list` (worker thread); return (titles, error)."""
    out, err = notebooklm(run, "source list", ["source", "list", "-n", NOTEBOOK, "--json"])
    if err:
        return None, err
    rows = out.get("sources") if isinstance(out, dict) else out
    if not isinstance(rows, list):
        return None, "NotebookLM source list returned unexpected JSON"
    return {r["id"]: r["title"] for r in rows if isinstance(r, dict) and isinstance(r.get("id"), str)
            and isinstance(r.get("title"), str) and r["title"].strip()}, None


def load_titles():
    try:
        titles = json.loads(TITLES_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return titles if isinstance(titles, dict) else {}


def store_titles(titles):
    TITLES_FILE.parent.mkdir(parents=True, exist_ok=True)
    part = TITLES_FILE.with_name(f"{TITLES_FILE.name}.{os.getpid()}.{threading.get_ident()}.part")
    part.write_text(json.dumps(titles, indent=1, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    os.replace(part, TITLES_FILE)


def load_sources(run, sids, titles, title_sids=()):
    """Fulltext for every source in `sids` not yet loaded or failed in this run (cache first, then
    concurrent fetches) into run.sources, and a title for every source in `title_sids`. Updates
    `titles` in place; returns (source ids loaded from cache, source ids fetched) by this call."""
    sources = run.sources
    texts, from_cache, to_fetch = sources["texts"], [], []
    needed = [sid for sid in dict.fromkeys(sids) if sid not in texts and sid not in sources["failed"]]
    for sid in needed:
        content = cached_fulltext(sid)
        if content is None:
            to_fetch.append(sid)
        else:
            texts[sid] = content
            from_cache.append(sid)
    untitled = [sid for sid in dict.fromkeys(title_sids)
                if not titles.get(sid) and sid not in to_fetch]
    jobs = [("fulltext", sid) for sid in to_fetch] + ([("titles", None)] if untitled else [])
    if to_fetch:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results = parallel(run, lambda job: (fetch_fulltext(run, job[1]) if job[0] == "fulltext"
                                         else list_titles(run)),
                       jobs, f"Waiting for NotebookLM ({len(to_fetch)} fulltexts"
                       + (", source titles" if untitled else "") + ")...")
    if any(r[-1] == CANCELLED for r in results):
        run.cancelled()

    fetched, changed = [], False
    for (kind, sid), result in zip(jobs, results):
        if kind == "titles":
            listed, err = result
            if err:
                run.log("titles_failed", error=err, untitled=untitled)
                continue
            changed = changed or any(titles.get(k) != v for k, v in listed.items())
            titles.update(listed)
            continue
        content, title, err = result
        if err:  # the hit keeps its exact search passage; only its continuation/context is lost
            run.log("fulltext_failed", source_id=sid, title=titles.get(sid), error=err)
            sources["failed"].add(sid)
            continue
        texts[sid] = content
        fetched.append(sid)
        if title and titles.get(sid) != title:
            titles[sid] = title
            changed = True
    if changed:
        store_titles(titles)
    for sid in needed:
        if sid in texts:
            run.log("fulltext", source_id=sid, title=titles.get(sid), chars=len(texts[sid]),
                    origin="cache" if sid in from_cache else "fetched")
    return from_cache, fetched


MARKUP = frozenset("*#_�")  # markdown the search chunks carry and the fulltext may not
MARKUP_ANCHOR = 120  # compact chars of a hit's end (or start) matched when the whole hit is not


def compact_markup(content, src):
    """(the fulltext without whitespace and markup, index map into the fulltext), built once."""
    if "compact_markup" not in src:
        idx = [j for j, ch in enumerate(content) if not ch.isspace() and ch not in MARKUP]
        src["compact_markup"] = ("".join(content[j] for j in idx), idx)
    return src["compact_markup"]


def strip_markup(text):
    return "".join(ch for ch in text if not ch.isspace() and ch not in MARKUP)


def locate_ignoring_markup(content, text, src):
    """Spans of the hit in the fulltext ignoring whitespace and markdown markup (search chunks of
    formatted books carry **bold** and ##### headings their fulltext lacks). When the whole hit
    does not match (a garbled character, for example), its last MARKUP_ANCHOR chars are matched,
    else its first ones, and the span is estimated from the hit's length."""
    compact, idx = compact_markup(content, src)
    needle = strip_markup(text)
    if not needle:
        return []

    def find(part):
        found, i = [], compact.find(part)
        while i != -1:
            found.append((i, i + len(part)))
            i = compact.find(part, i + 1)
        return found

    spans = [(idx[a], idx[b - 1] + 1) for a, b in find(needle)]
    if spans or len(needle) <= MARKUP_ANCHOR:
        return spans
    tail = [(idx[max(b - len(needle), 0)], idx[b - 1] + 1) for a, b in find(needle[-MARKUP_ANCHOR:])]
    if tail:
        return tail
    return [(idx[a], idx[min(a + len(needle), len(idx)) - 1] + 1)
            for a, b in find(needle[:MARKUP_ANCHOR])]


def locate(content, hit, src):
    """Find the hit's exact text in the fulltext; return (start, end, method) or None.

    Search offsets index NotebookLM's document tree, while the fulltext file joins text runs with
    "\\n", so the raw offsets drift. The hit text is matched exactly, then ignoring whitespace;
    the search offset only picks the nearest of several matches, or is used as a last resort.
    """
    text, hint = hit["text"].strip(), hit["start"] or 0
    spans = []
    i = content.find(text)
    while i != -1:
        spans.append((i, i + len(text)))
        i = content.find(text, i + 1)
    method = "exact"
    if not spans:
        if "compact" not in src:  # whitespace-free copy of the source + index map, built once
            idx = [j for j, ch in enumerate(content) if not ch.isspace()]
            src["compact"] = ("".join(content[j] for j in idx), idx)
        compact, idx = src["compact"]
        needle = "".join(text.split())
        i = compact.find(needle) if needle else -1
        while i != -1:
            spans.append((idx[i], idx[i + len(needle) - 1] + 1))
            i = compact.find(needle, i + 1)
        method = "ignoring_whitespace"
    if not spans:
        spans, method = locate_ignoring_markup(content, text, src), "ignoring_markup"
    if spans:
        return (*min(spans, key=lambda sp: abs(sp[0] - hint)), method)
    if hit["start"] is not None and hit["start"] < len(content):
        return hit["start"], min(hit["end"], len(content)), "search_offsets"
    return None


SENTENCE_ENDS = (". ", "? ", "! ", ".\"", "?\"", "!\"")


def edge_back(content, pos):
    """Move a window start outward to a line start, else a sentence start, within ALIGN_SLACK."""
    if pos <= 0 or content[pos - 1] == "\n":
        return max(pos, 0)
    lo = max(0, pos - ALIGN_SLACK)
    nl = content.rfind("\n", lo, pos)
    if nl != -1:
        return nl + 1
    ends = [content.rfind(s, lo, pos) for s in SENTENCE_ENDS]
    best = max(ends)
    return best + 2 if best != -1 else pos


def edge_forward(content, pos):
    """Move a window end outward to a line end, else a sentence end, within ALIGN_SLACK."""
    if pos >= len(content) or content[pos] == "\n":
        return min(pos, len(content))
    hi = min(len(content), pos + ALIGN_SLACK)
    nl = content.find("\n", pos, hi)
    if nl != -1:
        return nl
    ends = [i for i in (content.find(s, pos, hi) for s in SENTENCE_ENDS) if i != -1]
    return min(ends) + 1 if ends else pos


def is_question_line(line):
    return bool(QUESTION_LINE.match(line) or SPEAKER_QUESTION.match(line))


def continuation(content, end):
    """End of the immediate answer after a hit (located to end at `end`) whose search chunk stops
    before it; return (end position, status).

    The hit's own line is finished first. If that line is a question, following question lines
    are kept too until the answer starts. The answer then runs until the next speaker question or
    section boundary, or, once it exceeds CONTINUATION_SOFT chars, the next line end. status:
    "recovered"; "source_ends" (the source ends with no answer); "incomplete" (cut at
    CONTINUATION_MAX, at a sentence end where possible).
    """
    line_start = content.rfind("\n", 0, end) + 1
    pos = content.find("\n", end)
    pos = len(content) if pos == -1 else pos
    own = content[line_start:pos].strip()
    asked = ((QUESTION_LINE.match(own) and not ANSWER_ON_LINE.search(own))
             or ENDS_WITH_QUESTION.search(own))
    answer = None if asked else line_start  # a clipped answer/statement continues on its own line

    def capped():
        base = answer if answer is not None else end
        cap = min(len(content), base + CONTINUATION_MAX)
        best = max(content.rfind(s, base, cap) for s in SENTENCE_ENDS)
        return (best + 1 if best != -1 else cap), "incomplete"

    before = None  # where the previous non-empty line starts (a recipe title before its yield)
    while pos < len(content):
        nxt = content.find("\n", pos + 1)
        line_end = len(content) if nxt == -1 else nxt
        line = content[pos + 1:line_end].strip()
        if line:
            if answer is None:
                if not is_question_line(line):
                    answer = pos + 1
            elif is_question_line(line) or SECTION_LINE.match(line) or PAGE_LINE.match(line):
                return pos, "recovered"
            elif RECIPE_YIELD.match(line) and before is not None and before > answer:
                return before, "recovered"  # the next recipe starts at its title
            before = pos
        if line_end - (answer if answer is not None else end) > CONTINUATION_MAX:
            return capped()
        if answer is not None and line_end - answer >= CONTINUATION_SOFT:
            return line_end, "recovered"
        pos = line_end
    return len(content), "recovered" if answer is not None else "source_ends"


CONTINUATION_DONE = ("recovered", "source_ends")
# Recipe-book boundaries in plain fulltext (its search chunks have ##### headings, it has none):
# a page marker, or a "4 Servings" yield line under the next recipe's title.
PAGE_LINE = re.compile(r"(?:\*\*)?PAGE \d+(?:\*\*)?$")
RECIPE_YIELD = re.compile(r"\d+(?:\s*(?:-|to)\s*\d+)?\s+servings?$", re.I)
BACKWARD_DONE = ("recovered", "source_start")


def locate_start(content, text, src):
    """Where a hit's text starts in the fulltext, only when that is certain: the whole text,
    ignoring whitespace and markup, occurs exactly once. Else None: search offsets drift, and an
    estimated or ambiguous span must not decide which text precedes the hit."""
    compact, idx = compact_markup(content, src)
    needle = strip_markup(text)
    first = compact.find(needle) if needle else -1
    if first == -1 or compact.find(needle, first + 1) != -1:
        return None
    return idx[first]


def backward(content, start):
    """Start of the text before a hit that begins mid-list (the hit starts at `start`), back to
    its heading; return (position, status).

    The walk goes back line by line from the hit's line. It stops at the nearest heading: a
    "4 Servings" yield line (the recipe's title, the line before it, is included), a markdown
    heading or a speaker question (included), or a page marker or rule (excluded). status:
    "recovered"; "source_start" (no heading before the source's start); "capped" (no heading
    within BACKWARD_MAX chars: the text back to that cap, from a line start, and the start is
    still missing)."""
    lo = max(0, start - BACKWARD_MAX)
    end = content.rfind("\n", 0, start)  # the newline that ends the line before the hit's line
    yield_at = None
    while end > 0:
        ls = content.rfind("\n", 0, end) + 1
        if ls < lo:
            break
        line = content[ls:end].strip()
        if line:
            if yield_at is not None:
                return ls, "recovered"  # the recipe title above its yield line
            if RECIPE_YIELD.match(line):
                yield_at = ls
            elif SECTION_LINE.match(line) and line.startswith("#") or is_question_line(line):
                return ls, "recovered"
            elif PAGE_LINE.match(line) or SECTION_LINE.match(line):
                return end + 1, "recovered"
        end = ls - 1
    if yield_at is not None:
        return yield_at, "recovered"  # a yield line with no title within reach
    if end <= 0 and lo == 0:
        return 0, "source_start"
    nl = content.find("\n", lo, start)
    return (nl + 1 if nl != -1 and lo > 0 else lo), "capped"


def hit_chars(c):
    """Chars a candidate would put into the evidence: its passage, its continuation (forward and
    backward) and a header."""
    return (len(c["text"].strip()) + len(c.get("continuation_text") or "")
            + len(c.get("backward_text") or "") + BYPASS_HIT_OVERHEAD)


def hydrate(run, candidates, titles, round_n=1, budget=None, prior_chars=0):
    """Add the text that continues each cut-off hit (forward) or leads into it (backward).

    Only flagged hits are fetched, within the fetch cap and the evidence budget. Each hit gets a
    status and the exact source text, or None.
    """
    clipped = [c for c in candidates if c["needs_continuation"]]
    headless = [c for c in candidates if fragment_start(c["text"])]
    backward_ids = {c["hit_id"] for c in headless}
    if not clipped and not headless:
        return clipped
    clipped.sort(key=lambda c: (c["rank"] is None, c["rank"] or 0))
    work = sorted({c["hit_id"]: c for c in clipped + headless}.values(),
                  key=lambda c: (c["rank"] is None, c["rank"] or 0))
    sids = list(dict.fromkeys(c["source_id"] for c in work))
    uncached = [s for s in sids if s not in run.sources["texts"] and s not in run.sources["failed"]
                and cached_fulltext(s) is None]
    capped = set(uncached[CONTINUATION_MAX_FETCHES:])
    if capped:
        run.log("continuation_fetch_cap", round=round_n, limit=CONTINUATION_MAX_FETCHES,
                skipped_sources=sorted(capped))
    load_sources(run, [s for s in sids if s not in capped], titles)
    base = prior_chars + sum(hit_chars({**c, "continuation_text": None, "backward_text": None})
                             for c in candidates)
    room = None if budget is None or base > budget else budget - base
    for c in work:
        sid = c["source_id"]
        content = run.sources["texts"].get(sid)
        aux = run.sources["aux"].setdefault(sid, {})
        if c["needs_continuation"]:
            span = locate(content, c, aux) if content else None
            c["continuation_end"] = c["continuation_text"] = None
            if sid in capped:
                c["continuation_status"] = "skipped_fetch_cap"
            elif content is None:
                c["continuation_status"] = "no_fulltext"
            elif not span or span[2] == "search_offsets":
                # Raw search offsets drift from the fulltext: text read from there is not this
                # hit's continuation.
                c["continuation_status"] = "unlocated"
            else:
                end, status = continuation(content, span[1])
                text = content[span[1]:end].strip() or None
                if room is not None and text and len(text) > room:
                    c["continuation_status"] = "skipped_budget"
                    run.log("continuation_skipped", hit_id=c["hit_id"], reason="evidence budget",
                            chars=len(text), room=room)
                else:
                    c["continuation_status"], c["continuation_end"] = status, end
                    c["continuation_text"] = text
                    if room is not None:
                        room -= len(text or "")
            run.log("continuation_hit", hit_id=c["hit_id"], source_id=sid,
                    reason=c["continuation_reason"], status=c["continuation_status"],
                    alignment=span[2] if span else None,
                    chars=len(c["continuation_text"] or ""))
        if c["hit_id"] in backward_ids:
            start = locate_start(content, c["text"].strip(), aux) if content else None
            c["backward_start"] = c["backward_text"] = None
            if sid in capped:
                c["backward_status"] = "skipped_fetch_cap"
            elif content is None:
                c["backward_status"] = "no_fulltext"
            elif start is None:
                c["backward_status"] = "unlocated"  # no certain match: the start stays missing
            else:
                pos, status = backward(content, start)
                text = content[pos:start].strip() or None
                if room is not None and text and len(text) > room:
                    c["backward_status"] = "skipped_budget"
                    run.log("continuation_skipped", hit_id=c["hit_id"], reason="evidence budget",
                            chars=len(text), room=room, direction="backward")
                else:
                    c["backward_status"], c["backward_start"], c["backward_text"] = status, pos, text
                    if room is not None:
                        room -= len(text or "")
            run.log("backward_hit", hit_id=c["hit_id"], source_id=sid,
                    status=c["backward_status"], chars=len(c["backward_text"] or ""))
    run.save(f"continuations-{round_n}.json", json.dumps(
        [{k: c[k] for k in ("hit_id", "source_id", "continuation_reason", "continuation_status",
                            "continuation_end", "continuation_text")} for c in clipped],
        indent=1, ensure_ascii=False))
    if headless:
        run.save(f"backward-{round_n}.json", json.dumps(
            [{k: c[k] for k in ("hit_id", "source_id", "backward_status", "backward_start",
                                "backward_text")} for c in headless],
            indent=1, ensure_ascii=False))
    return clipped


def expand(content, start, end, window, forward_end=None):
    """Context window around a located hit: the whole hit, about `window` chars before/after,
    moved outward to natural line/sentence boundaries. It reaches further (up to QA_SLACK) only
    to complete the immediate question/answer: back to the question of an answer it starts in,
    forward through the answer of a question it ends on. A hit with a recovered continuation
    instead runs forward at least to `forward_end`, the end of its immediate answer."""
    before, after = window
    ws = edge_back(content, max(0, start - before))
    we = edge_forward(content, min(len(content), end + after))
    if forward_end is not None:
        we = max(we, forward_end)
    if content.startswith("A:", ws):
        q = content.rfind("\nQ:", max(0, ws - QA_SLACK), ws)
        if q != -1:
            ws = q + 1
        elif ws - QA_SLACK <= 0 and content.startswith("Q:"):
            ws = 0
    last_line = max(content.rfind("\n", ws, we) + 1, ws)
    if forward_end is None and content.startswith("Q:", last_line):
        a = content.find("\nA:", we, we + QA_SLACK)
        if a != -1:
            nl = content.find("\n", a + 1, a + 1 + QA_SLACK)
            we = nl if nl != -1 else edge_forward(content, min(len(content), a + 1 + QA_SLACK))
    return ws, we


CONTINUATION_NOTE = ("[Note: this passage ends at a question or cut-off point; the source text "
                     "that follows it was not retrieved in full.]")
FRAGMENT_START_NOTE = ("[Note: this passage starts partway through a list; the text before it "
                       "(such as a recipe's title or first items) was not retrieved.]")


def provenance(hit):
    """A hit's provenance for unjudged (selector-bypassed) evidence: raw search ids and origin."""
    raw = ", ".join(str(r) for r in hit.get("raw_ids") or []) or "none"
    return f"raw: {raw}; origin: {hit.get('origin') or 'fresh'}"


def assemble(selected, context_ids, texts, spans, titles, reduced, with_provenance=False,
             groups=None):
    """Build the evidence blocks in priority order and return (evidence text, diagnostics, excerpts).

    Passages are grouped under the fact question they answer; overlapping windows of one source
    merge into one block, so no source text is repeated.
    """
    order = {h["hit_id"]: i for i, h in enumerate(selected)}
    by_id = {h["hit_id"]: h for h in selected}

    def cont_end(i):
        return by_id[i].get("continuation_end") if spans.get(i) else None

    def incomplete(i):
        return (by_id[i]["needs_continuation"]
                and by_id[i].get("continuation_status") not in CONTINUATION_DONE)

    blocks, placed = [], set()
    for sid, content in texts.items():
        located = [h["hit_id"] for h in selected if h["source_id"] == sid and spans.get(h["hit_id"])]
        windows = []
        for i in located:
            if i in context_ids:
                windows.append((*expand(content, *spans[i][:2],
                                        REDUCED_WINDOW if i in reduced else CONTEXT_WINDOW,
                                        cont_end(i)), i))
            elif cont_end(i) is not None:
                windows.append((spans[i][0], max(spans[i][1], cont_end(i)), i))
        merged = []  # [start, end, hit ids, has context]
        for ws, we, i in sorted(windows):
            is_context = i in context_ids
            if merged and ws <= merged[-1][1] + (MERGE_GAP if is_context or merged[-1][3] else 0):
                m = merged[-1]
                m[1], m[3] = max(m[1], we), m[3] or is_context
                m[2].append(i)
            else:
                merged.append([ws, we, [i], is_context])
        for i in located:
            if any(i in m[2] for m in merged):
                continue
            for m in merged:
                if m[0] <= spans[i][0] and spans[i][1] <= m[1]:
                    m[2].append(i)
                    break
        for ws, we, ids, is_context in merged:
            ids.sort(key=lambda i: spans[i][0])  # passages in document order within the block
            block = {"source_id": sid, "hit_ids": ids, "context": None, "continuation": None,
                     "text_start": ws, "text_end": we}
            if is_context:
                block["context"] = content[ws:we]
            else:  # the earliest passage plus the source text after it contains every hit here
                block["continuation"] = content[spans[ids[0]][1]:we]
            blocks.append(block)
            placed.update(ids)
    blocks += [{"source_id": h["source_id"], "hit_ids": [h["hit_id"]], "context": None,
                "continuation": None} for h in selected if h["hit_id"] not in placed]
    blocks.sort(key=lambda b: min(order[i] for i in b["hit_ids"]))
    heads, tail = group_headings(blocks, order, groups) if groups else ({}, [])

    def queries(i):
        return list(dict.fromkeys(f["query"] for f in by_id[i]["found_by"]))

    def passage(i):
        head = f"PASSAGE [{i}]" + (f" ({provenance(by_id[i])})" if with_provenance else "")
        text = by_id[i]["text"].strip()
        headless = (fragment_start(text)
                    and by_id[i].get("backward_status") not in BACKWARD_DONE)
        note = f"\n{FRAGMENT_START_NOTE}" if headless else ""
        before = by_id[i].get("backward_text")
        if before:  # the recovered text before a passage that starts mid-list
            return (f"PRECEDING (exact source text immediately before [{i}]):\n{before}\n"
                    f"{head}:{note}\n{text}")
        return f"{head}:{note}\n{text}"

    rendered, diagnostics, excerpts = [], [], []
    for n, b in enumerate(blocks):
        title = titles.get(b["source_id"])
        date = source_date(title) if with_provenance and title else None
        # The readable name (a compilation's heading or date comes from the block's text)
        label = readable_source(title, "\n".join(
            [by_id[i].get("backward_text") or "" for i in b["hit_ids"]]
            + [by_id[i]["text"] for i in b["hit_ids"]]
            + [b["continuation"] or "", b["context"] or ""]))
        parts = heads.get(n, []) + [f"SOURCE: {label}" + (f" (date: {date})" if date else "")]
        if b["continuation"] is not None:
            shown = b["hit_ids"][:1]
            parts.append(passage(b["hit_ids"][0]))
            if b["continuation"].strip():
                parts.append(f"CONTINUATION:\n{b['continuation'].strip()}")
            if any(incomplete(i) for i in b["hit_ids"]):
                parts.append(CONTINUATION_NOTE)
        else:
            shown = b["hit_ids"]
            for i in b["hit_ids"]:
                parts.append(passage(i))
                if incomplete(i):
                    parts.append(CONTINUATION_NOTE)
            if b["context"] is not None:
                parts.append(f"CONTEXT:\n{b['context'].strip()}")
        rendered.append("\n".join(parts))
        kind = ("context" if b["context"] is not None else
                "continuation" if b["continuation"] is not None else "passage")
        diagnostics.append({
            "source_id": b["source_id"], "source_title": titles.get(b["source_id"]),
            "kind": kind, "text_start": b.get("text_start"), "text_end": b.get("text_end"),
            "hits": [{"hit_id": i, "priority": order[i] + 1, "rank": by_id[i]["rank"],
                      "context_requested": i in context_ids, "context_reduced": i in reduced,
                      "needs_continuation": by_id[i]["needs_continuation"],
                      "continuation": by_id[i].get("continuation_status"),
                      "queries": queries(i),
                      "search_start": by_id[i]["start"], "search_end": by_id[i]["end"],
                      "text_start": spans[i][0] if spans.get(i) else None,
                      "text_end": spans[i][1] if spans.get(i) else None,
                      "alignment": spans[i][2] if spans.get(i) else
                      ("unlocated" if b["source_id"] in texts else "no_fulltext")}
                     for i in b["hit_ids"]]})
        excerpts.append({
            "source_id": b["source_id"], "kind": kind, "label": label,
            "priority": min(order[i] for i in b["hit_ids"]) + 1,
            "passages": [{"hit_id": i, "text": by_id[i]["text"].strip(), "queries": queries(i),
                          **({"preceding": by_id[i]["backward_text"]}
                             if by_id[i].get("backward_text") else {})}
                         for i in shown],
            "continuation": (b["continuation"] or "").strip() or None,
            "context": (b["context"] or "").strip() or None,
            "incomplete": any(incomplete(i) for i in b["hit_ids"])})
    if tail and rendered:
        rendered[-1] += "\n\n" + "\n".join(tail)
    elif tail:
        rendered.append("\n".join(tail))
    return "\n\n=====\n\n".join(rendered), diagnostics, excerpts


OTHER_PASSAGES_HEADING = ("OTHER PASSAGES (research memory and the earlier turn; not cited for a "
                          "fact question above)")


def group_headings(blocks, order, groups):
    """Sort `blocks` (in place) by fact group, then priority; return ({block index: heading
    lines placed before it}, heading lines left after the last block) (see assemble). A question
    whose passages all sit under an earlier question gets its heading before the next group's,
    naming them; one with no passage says so."""
    first_group = {}
    for k, (_, ids) in enumerate(groups):
        for i in ids:
            first_group.setdefault(i, k)
    other = len(groups)

    def group(b):
        return min(first_group.get(i, other) for i in b["hit_ids"])
    blocks.sort(key=lambda b: (group(b), min(order[i] for i in b["hit_ids"])))
    starts = {}
    for n, b in enumerate(blocks):
        starts.setdefault(group(b), n)
    heads, waiting = {}, []
    for k, (heading, ids) in enumerate(list(groups) + [(OTHER_PASSAGES_HEADING, [])]):
        elsewhere = [i for i in ids if first_group.get(i, k) != k]
        line = heading + (f"\n(also answered by {''.join(f'[{i}]' for i in elsewhere)}, under an "
                          "earlier question)" if elsewhere else "")
        if k not in starts:
            if k < other:
                waiting.append(line if ids else heading + "\n(no passage cited for it)")
            continue
        heads[starts[k]] = waiting + [line]
        waiting = []
    return heads, waiting


def build_evidence(run, selected, context_ids, depth, titles, round_n=1, groups=None):
    """Evidence for the reasoner: exact passages (with any recovered continuation) for all
    selected hits, exact surrounding text for context hits only, kept near or under the depth's
    soft size budget.

    Returns (evidence text, stats, sources). Retrieval metadata goes to diagnostic files only;
    sources is the evidence grouped by document (priority order) for display.
    """
    by_id = {h["hit_id"]: h for h in selected}
    from_cache, fetched = load_sources(run, [by_id[i]["source_id"] for i in context_ids], titles,
                                       [h["source_id"] for h in selected])
    texts = run.sources["texts"]
    spans = {h["hit_id"]: locate(texts[h["source_id"]], h,
                                 run.sources["aux"].setdefault(h["source_id"], {}))
             for h in selected if h["source_id"] in texts}

    budget = EVIDENCE_BUDGET[depth]
    reduced = []
    # Unjudged (selector-bypassed) hits carry their provenance: raw ids, origin, source date.
    marked = any(h.get("unjudged") for h in selected)
    evidence, diagnostics, excerpts = assemble(selected, context_ids, texts, spans, titles, set(),
                                               marked, groups)
    if len(evidence) > budget * FAR_OVER_BUDGET:
        # Shrink the context of the lowest-priority context hits first; passages are never dropped,
        # and a recovered continuation is never cut (the window still reaches its end).
        for i in reversed([i for i in context_ids if spans.get(i)]):
            if len(evidence) <= budget:
                break
            reduced.append(i)
            evidence, diagnostics, excerpts = assemble(selected, context_ids, texts, spans, titles,
                                                       set(reduced), marked, groups)

    clipped = [h for h in selected if h["needs_continuation"]]
    stats = {
        "depth": depth, "budget_chars": budget, "chars": len(evidence),
        "over_budget": len(evidence) > budget, "blocks": len(diagnostics),
        "selected": len(selected), "context_hits": len(context_ids),
        "context_reduced": reduced,
        "context_unlocated": [i for i in context_ids
                              if by_id[i]["source_id"] in texts and not spans.get(i)],
        "context_without_fulltext": [i for i in context_ids if by_id[i]["source_id"] not in texts],
        "continuation_selected": [h["hit_id"] for h in clipped],
        "continuation_status": {h["hit_id"]: h.get("continuation_status") for h in clipped},
        "continuation_incomplete": [h["hit_id"] for h in clipped
                                    if h.get("continuation_status") not in CONTINUATION_DONE],
        "fulltext_from_cache": [{"source_id": s, "title": titles.get(s)} for s in from_cache],
        "fulltext_fetched": [{"source_id": s, "title": titles.get(s)} for s in fetched],
    }
    run.log("evidence", **stats)
    run.save(f"evidence-{round_n}.txt", evidence)
    run.save(f"evidence-{round_n}.diagnostics.json", json.dumps({"stats": stats, "blocks": diagnostics},
                                                       indent=1, ensure_ascii=False))
    sources = {}
    for e in excerpts:
        title = titles.get(e["source_id"])
        sources.setdefault(e["source_id"], {"source_id": e["source_id"], "title": title,
                                            "name": readable_source(title),
                                            "date": source_date(title),
                                            "excerpts": []})["excerpts"].append(e)
    return evidence, stats, list(sources.values())


def preview(text, limit=160):
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


# ---- research memory --------------------------------------------------------------------------
# Every memory step is optional: a missing or empty database, or any memory failure, leaves the
# NotebookLM-only pipeline exactly as it was (failures are logged and shown in the trace).

def memory_disabled():
    return os.environ.get("CRA_MEMORY", "").strip().lower() in ("0", "off", "false", "no")


def open_memory(create):
    """The research memory store, or None when memory is disabled or (create=False) its database
    does not exist yet."""
    if memory_disabled():
        return None
    import research_memory
    path = Path(MEMORY_DB or os.environ.get("CRA_MEMORY_DB") or research_memory.DEFAULT_DB)
    if not create and not path.exists():
        return None
    return research_memory.MemoryStore(path)


def memory_failed(run, stage, error):
    run.log("memory_error", stage=stage, error=repr(error))
    run.trace["memory"]["errors"].append({"stage": stage, "error": repr(error)[:300]})


def memory_keys(question, searches=(), case_frame=None, community=()):
    """The memory lookup keys: the raw question, every planner search query (component and
    vocabulary searches included; community-facet ones keyed "community_search"), and the
    planner's case-frame subject with its qualifiers when present; [(source, text)] without
    repeats."""
    community = set(community or ())
    keys = [("question", question)] + [
        ("community_search" if q in community else "planner_search", q) for q in searches or ()]
    if isinstance(case_frame, dict):
        quals = [q for q in case_frame.get("qualifiers") or [] if isinstance(q, str)]
        frame = " ".join(x for x in [case_frame.get("subject")] + quals if isinstance(x, str) and x)
        if frame.strip():
            keys.append(("case_frame", frame))
    seen, out = set(), []
    for source, text in keys:
        norm = " ".join((text or "").lower().split())
        if norm and norm not in seen:
            seen.add(norm)
            out.append((source, text))
    return out


def best_across_keys(store, keys, layer, per_key, min_score=0.5):
    """store.relevant for every key; each unit ranked by its best score across the keys (ties:
    more matched terms, then unit id). Returns (hits, per-key hit counts)."""
    best, counts = {}, []
    for source, text in keys:
        hits = store.relevant(text, layer, per_key, min_score)
        counts.append(len(hits))
        for h in hits:
            if h["unit_id"] not in best or h["score"] > best[h["unit_id"]]["score"]:
                best[h["unit_id"]] = {**h, "key": text, "key_source": source}
    ranked = sorted(best.values(), key=lambda x: (-x["score"], -len(x["matched"]), x["unit_id"]))
    return ranked, counts


# Community-lookup aliases; a key naming one name of a group is also looked up under
# the others (the community records use them interchangeably). Corpus-specific: these groups
# suit the sample corpus (sample_corpus/); edit them, and the Aliases list in
# prompts/planner.txt, for your own library.
COMMUNITY_ALIASES = (("expense claim", "expense report", "reimbursement claim"),
                     ("remote work", "work from home"))


def community_keys(keys):
    """The secondary (community) lookup keys: every (source, text) key spelling-
    normalized (ledger_key; alias words are never respelled), then each key naming a
    COMMUNITY_ALIASES name repeated with every other name of its group; [(source, text)] without
    repeats. A key found under several sources keeps its first source, except that a
    "community_search" source wins (it marks a community-facet key)."""
    try:
        vocab, index = corpus_vocabulary()
    except Exception:  # noqa: BLE001 - normalization is best effort
        vocab, index = None, None
    if vocab:
        vocab = {**vocab, **{w: 1 for g in COMMUNITY_ALIASES for a in g for w in a.split()}}
    out = {}
    for source, text in keys:
        norm = ledger_key(text, vocab, index)
        variants = [norm]
        for group in COMMUNITY_ALIASES:
            for name in sorted(group, key=len, reverse=True):
                pattern = r"\b" + re.escape(name) + r"\b"
                if re.search(pattern, norm):
                    variants += [re.sub(pattern, other, norm) for other in group if other != name]
                    break
        for v in variants:
            if v and (v not in out or source == "community_search"):
                out[v] = source
    return [(source, text) for text, source in out.items()]


# Community records are ranked by their overlap with the question's subject before the
# top MEMORY_SECONDARY_MAX are taken (4 of 5 records of one run were about a neighboring topic).
COMMUNITY_POOL = 20  # community records considered per lookup key before the subject ranking
# A whole-question key carries words no record uses ("possible upgrades that can be
# done"), which held every on-subject record under the 0.5 floor. Community records get a lower floor
# (subject_overlap and the planner's pick filter them), and each name of an alias group the
# question names is a key of its own, so every record naming the subject is considered.
COMMUNITY_MIN_SCORE = 0.25


def alias_pattern(name):
    return r"\b" + re.escape(name) + r"\b"


def subject_profile(subject):
    """(alias groups the subject names, stems of its other content words) (pure); None when the
    subject has neither. An alias group counts as one subject term under any of its names."""
    text = " ".join(str(subject or "").lower().split())
    groups = [g for g in COMMUNITY_ALIASES if any(re.search(alias_pattern(n), text) for n in g)]
    for g in groups:
        for n in sorted(g, key=len, reverse=True):
            text = re.sub(alias_pattern(n), " ", text)
    stems = {stem(w) for w in query_terms_set(text) if w not in SUBJECT_GENERIC}
    return (groups, stems) if groups or stems else None


def subject_overlap(profile, text):
    """How much of the subject a record's text names (pure): 2 per alias group named (any of its
    names) plus 1 per other subject stem. 0 when the subject names an alias group and the text
    names none of them, since the record is then about something else. Without a profile, 1."""
    if profile is None:
        return 1
    groups, stems = profile
    text = str(text or "").lower()
    named = sum(1 for g in groups if any(re.search(alias_pattern(n), text) for n in g))
    if groups and not named:
        return 0
    return 2 * named + len(stems & {stem(w) for w in re.findall(r"[\w'’-]+", text)})


def secondary_lookup(store, keys, info, subject=None, keep=MEMORY_SECONDARY_MAX):
    """Community records for the keys: up to COMMUNITY_POOL per key (score floor
    COMMUNITY_MIN_SCORE; every name of an alias group the `subject` names is a key too), ranked
    by their overlap with the `subject` (the user's question; see subject_overlap), then by best
    score across all keys; records with no overlap are dropped; the top `keep` are kept, plus up
    to MEMORY_COMMUNITY_MAX per "community_search" key in their own slots. Dropped records are
    listed in info["secondary_dropped"]. Returns (records, per-key hit counts)."""
    import research_memory as rm
    profile = subject_profile(subject)
    anchors = [("subject", name) for g in (profile or ((), ()))[0] for name in g]
    keys = list(keys) + [k for k in anchors if k[1] not in {text for _, text in keys}]
    ranked, counts = best_across_keys(store, keys, rm.SECONDARY, COMMUNITY_POOL,
                                      COMMUNITY_MIN_SCORE)
    community = [k for k in keys if k[0] == "community_search"]
    facet = best_across_keys(store, community, rm.SECONDARY, MEMORY_COMMUNITY_MAX)[0] if community else []
    records, dropped = {}, []

    def load(hit):
        if hit["unit_id"] not in records:
            row = store.db.execute("SELECT * FROM units WHERE unit_id = ?", (hit["unit_id"],)).fetchone()
            claim = store.secondary_claim(hit["unit_id"]) or {
                "unit_id": row["unit_id"], "claim_text": row["text"], "platform": row["platform"],
                "author": row["author"], "url": row["url"], "source_date": row["source_date"],
                "community": row["source_title"], "record_id": row["source_id"],
                "claim_type": row["kind"] or "unclassified", "verification": "unverified",
                "links": [], "primary_refs": []}
            overlap = subject_overlap(profile, f"{row['text']} {claim.get('claim_text') or ''} "
                                               f"{claim.get('context') or ''}")
            records[hit["unit_id"]] = (row, claim, overlap)
            if not overlap:
                dropped.append({"unit_id": hit["unit_id"], "score": hit["score"],
                                "key": hit.get("key"),
                                "reason": "no overlap with the question's subject",
                                "preview": preview(row["text"] or "")})
        return records[hit["unit_id"]][2]

    ranked = [h for h in ranked if load(h)]
    ranked.sort(key=lambda h: -records[h["unit_id"]][2])  # stable: best score within a tie
    facet = [h for h in facet if load(h)]
    hits = ranked[:keep]
    # A community-facet search queries secondary memory on its own, with its own
    # slots, so the claims it finds do not compete with the other keys' matches.
    if community:
        have = {h["unit_id"] for h in hits}
        extra = [dict(h, community_facet=True) for h in facet if h["unit_id"] not in have]
        for h in hits:
            if any(f["unit_id"] == h["unit_id"] for f in facet):
                h["community_facet"] = True
        hits = hits + extra
        info["community_facet"] = {"keys": [t for _, t in community],
                                   "hits": [h["unit_id"] for h in facet]}
    kept = {h["unit_id"] for h in hits}
    info["secondary_dropped"] = dropped + [
        {"unit_id": h["unit_id"], "score": h["score"], "key": h.get("key"),
         "reason": f"ranked below the top {keep}",
         "preview": preview(records[h["unit_id"]][0]["text"] or "")}
        for h in ranked if h["unit_id"] not in kept]
    secondary = []
    for hit in hits:
        row, claim, overlap = records[hit["unit_id"]]
        secondary.append({**hit, "claim": claim, "subject_overlap": overlap,
                          "metadata": rm.loads(row["metadata"]) or {}})
    info["community_keys"] = [{"source": src, "key": text, "hits": n}
                              for (src, text), n in zip(keys, counts)]
    return secondary, counts


def lookup_keys_for_community(run, fact_questions, search_plan=()):
    """The community lookup keys for a run: the user's question, the standalone rewrite
    and the planner's community-facet searches as "community_search" keys, and every fact
    question (follow-up ones included), through community_keys."""
    keys = [("question", run.question), ("community_search", run.standalone)]
    keys += [("community_search", s["query"]) for s in search_plan or ()
             if s.get("type") == "community"]
    keys += [("fact_question", q) for q in fact_questions or () if q]
    return community_keys(keys)


def community_lookup(run, keys, subject=None, keep=MEMORY_SECONDARY_MAX):
    """The secondary lookup alone (again after follow-up fact questions, with the
    extended keys), ranked against `subject`, top `keep` (see secondary_lookup). Returns the
    records, or None when memory is unavailable or fails."""
    try:
        store = open_memory(create=False)
        if store is None:
            return None
        with store:
            return secondary_lookup(store, keys, run.trace["memory"], subject, keep)[0]
    except Exception as e:  # noqa: BLE001 - memory is optional
        memory_failed(run, "community_lookup", e)
        return None


# The planner writes the verification question for a community record that attributes a
# statement to the corpus author (normalize_verifications); it is asked with the planner's asks, and its
# search string also runs as a source search (fallback path, follow-up round).
PLANNER_RECORD_CHARS = 400  # a community record's claim in the planner's input
PLANNER_RECORD_CONTEXT_CHARS = 200  # and its context excerpt


def add_verification_facts(facts, verifications, secondary):
    """Link each verification to its fact question and community record (in place): the fact
    gains "verifies" [unit_id], the record a "check" {"question", "search", "fact"}."""
    by_q = {f["question"]: f for f in facts}
    for v in verifications or []:
        f = by_q.get(v["question"])
        if f:
            f["verifies"] = [v["unit_id"]]
        for s in secondary or []:
            if s["unit_id"] == v["unit_id"]:
                s["check"] = {"question": v["question"], "search": v["search"],
                              **({"fact": f["id"]} if f else {})}


def verification_searches(run, facts, candidates, titles, round_n, unanswered_only):
    """Run the verification questions' search strings as source searches, so a failed
    or uncited ask does not hide the passage where the corpus author said it: in round 1 only those whose
    fact question brought no passage, in the follow-up round every one not searched yet. A hit
    is tagged with the verification's fact id (a known passage it repeats gains the id too).
    Returns (raw hits, new candidates, collapsed against `candidates`); a failed search only
    logs."""
    todo = []
    for v in run.verifications:
        f = next((x for x in facts if x["question"] == v["question"]), None)
        if not v.get("searched") and not (unanswered_only and f and f.get("passages")):
            todo.append((v, f))
    if not todo:
        return [], []
    queries = list(dict.fromkeys(v["search"] for v, _ in todo))
    try:
        raw, new = search(run, queries, round_n, known=candidates, label=f"{round_n}v")
    except ResearchError as e:
        run.log("verification_search", round=round_n, queries=queries, error=str(e)[:300])
        return [], []
    collapse_duplicates(run, new, titles, round_n, known=candidates)
    for v, f in todo:
        v["searched"] = True
        if f is None:
            continue
        found = {r.get("candidate") for r in raw if r["query"] == v["search"]}
        for c in list(candidates) + new:
            if c["hit_id"] in found:
                c["facts"] = list(dict.fromkeys((c.get("facts") or []) + [f["id"]]))
        f["searched"] = v["search"]
        f["passages"] = sum(1 for c in list(candidates) + new if f["id"] in (c.get("facts") or []))
    run.log("verification_search", round=round_n, queries=queries, raw_hits=len(raw),
            new_candidates=len(new))
    return raw, new


def settle_checks(secondary, facts):
    """Record each check's outcome on its record (in place): the fact question's origin (asked,
    reused, failed) and how many passages it brought."""
    by_id = {f["id"]: f for f in facts}
    for s in secondary or []:
        f = by_id.get((s.get("check") or {}).get("fact"))
        if f:
            s["check"].update(origin=f["origin"], passages=f.get("passages", 0))


def memory_lookup(run, question, keys=None, secondary_keys=None, subject=None,
                  with_secondary=True):
    """Remembered primary passages and secondary claims relevant to the question (see
    MemoryStore.relevant: deterministic FTS + term-weight coverage, no model), looked up with
    every key of memory_keys (default: the question alone) and capped after ranking by best
    score; community records use `secondary_keys` when given (see community_keys). Returns
    (primary, secondary); both empty when memory is unavailable or fails."""
    info = run.trace["memory"]
    keys = keys or [("question", question)]
    try:
        store = open_memory(create=False)
        if store is None:
            info["status"] = "disabled" if memory_disabled() else "no_database"
            return [], []
        import research_memory as rm
        corpus = corpus_identity()
        with store:
            primary = []
            hits, p_counts = best_across_keys(store, keys, rm.PRIMARY, MEMORY_PRIMARY_MAX * 2)
            for hit in hits:
                if len(primary) >= MEMORY_PRIMARY_MAX:
                    break
                row = store.db.execute("SELECT * FROM units WHERE unit_id = ?", (hit["unit_id"],)).fetchone()
                if row["source_id"] is None:  # a passage needs its source to be located and cited
                    continue
                if not discovered_in_corpus(store, hit["unit_id"], corpus):
                    info["other_corpus"] = info.get("other_corpus", 0) + 1
                    continue
                primary.append({**hit, "text": row["text"], "source_id": row["source_id"],
                                "source_title": row["source_title"]})
            # Passages earlier answers cited for a matching question, rewrite or
            # requirement (keyed by the question and the planner's searches)
            cited_keys = [k for k in keys if k[0] in ("question", "planner_search")]
            best = {}
            for _, text in cited_keys:
                for h in store.cited_relevant(text, MEMORY_CITED_MAX * 2):
                    if h["unit_id"] not in best or h["score"] > best[h["unit_id"]]["score"]:
                        best[h["unit_id"]] = {**h, "key": text}
            have, cited = {p["unit_id"] for p in primary}, []
            for hit in sorted(best.values(), key=lambda x: (-x["score"], x["unit_id"])):
                if len(cited) >= MEMORY_CITED_MAX:
                    break
                row = store.db.execute("SELECT * FROM units WHERE unit_id = ?", (hit["unit_id"],)).fetchone()
                if (row is None or row["source_id"] is None or row["layer"] != rm.PRIMARY
                        or not discovered_in_corpus(store, hit["unit_id"], corpus)):
                    continue
                cited.append(hit["unit_id"])
                if hit["unit_id"] in have:
                    next(p for p in primary if p["unit_id"] == hit["unit_id"])["cited_before"] = True
                    continue
                primary.append({**hit, "text": row["text"], "source_id": row["source_id"],
                                "source_title": row["source_title"], "cited_before": True})
            info["cited_before"] = cited
            secondary, s_counts = ([], [None] * len(keys)) if not with_secondary else (
                secondary_lookup(store, secondary_keys or keys, info, subject))
            if secondary_keys or not with_secondary:
                s_counts = [None] * len(keys)
            info["lookup_keys"] = [
                {"source": src, "key": " ".join(rm.query_terms(text)), "primary_hits": pc,
                 "secondary_hits": sc} for (src, text), pc, sc in zip(keys, p_counts, s_counts)]
        info["status"] = "ok"
        return primary, secondary
    except Exception as e:  # noqa: BLE001 - memory is optional
        memory_failed(run, "lookup", e)
        info["status"] = "error"
        return [], []


def merge_memory_primary(run, candidates, remembered, titles):
    """Add remembered primary passages to the round-1 candidates. A remembered passage that a
    fresh hit already contains (same source; identical or contained text) is not added again: the
    fresh candidate records the memory unit instead. Others become candidates like search hits,
    with origin "memory", and go through continuation, selection and evidence unchanged: they are
    primary evidence."""
    info = run.trace["memory"]
    for m in remembered:
        probe = {"source_id": m["source_id"], "text": m["text"], "start": None, "end": None}
        same = next((c for c in candidates
                     if c["source_id"] == m["source_id"] and duplicate_reason(c, probe)), None)
        entry = {"unit_id": m["unit_id"], "score": m["score"], "matched": m["matched"],
                 "source": m["source_title"] or m["source_id"], "preview": preview(m["text"])}
        if same is not None:
            same.setdefault("memory_units", []).append(m["unit_id"])
            info["primary"].append({**entry, "hit_id": same["hit_id"], "merged_into_fresh": True})
            continue
        hit_id = f"h{len(candidates) + 1}"
        reason = continuation_reason(m["text"])
        c = {"hit_id": hit_id, "source_id": m["source_id"], "text": m["text"], "start": None,
             "end": None, "found_by": [{"query": "(research memory)", "rank": None, "start": None,
                                        "end": None}],
             "raw_ids": [], "rank": None, "needs_continuation": reason is not None,
             "continuation_reason": reason, "origin": "memory", "memory_units": [m["unit_id"]]}
        if m.get("cited_before"):
            c["cited_before"] = True
        candidates.append(c)
        run.trace["candidates"].append(c)
        if m["source_title"] and not titles.get(m["source_id"]):
            titles[m["source_id"]] = m["source_title"]
        info["primary"].append({**entry, "hit_id": hit_id, "merged_into_fresh": False})
    run.log("memory_primary", candidates=[(p["unit_id"], p["hit_id"], p["merged_into_fresh"])
                                          for p in info["primary"]])


def capture_primary(run, raw, candidates, titles, round_n, depth=None, requirements=(), searches=()):
    """Persist this round's exact NotebookLM passages into primary memory (research_memory's
    dedup: the same text from the same source stays one unit; each rediscovery is a discovery).
    Only raw search-hit text is stored; a recovered continuation goes into the discovery's
    metadata, never into the passage."""
    written = run.trace["memory"]["written"]
    try:
        store = open_memory(create=True)
        if store is None:
            return
        import research_memory as rm
        by_raw = {r: c for c in candidates for r in c.get("raw_ids", [])}
        seen = time.strftime("%Y-%m-%dT%H:%M:%S")
        corpus = {k: v for k, v in corpus_identity().items() if k != "strength"}
        with store:
            if round_n == 1:
                store.add_run(run.dir.name, run.question, depth, seen, str(run.dir), None,
                              requirements, [{"query": q} for q in searches])
            for r in raw:
                title = titles.get(r["source_id"])
                uid = rm.unit_id_for(rm.PRIMARY, rm.primary_key(r["source_id"], title),
                                     rm.fingerprint(r["text"]))
                existed = store.layer(uid) is not None and uid not in run.cache_units
                run.cache_units.discard(uid)
                c = by_raw.get(r["raw_id"]) or {}
                meta = {k: r[k] for k in ("start", "end") if r.get(k) is not None}
                meta.update({k: c[k] for k in ("hit_id", "continuation_status", "continuation_text")
                             if c.get(k)})
                meta["corpus"] = corpus
                if r.get("origin") == "cache":
                    meta.update(cached_from=r.get("cached_from"), retrieved_at=r.get("retrieved_at"))
                before = store.db.execute("SELECT count(*) FROM discoveries").fetchone()[0]
                uid = store.record_retrieval(r["text"], run.dir.name, r["query"], r.get("rank"), round_n,
                                             "notebooklm_cache" if r.get("origin") == "cache"
                                             else "notebooklm", r["raw_id"], r["source_id"], title,
                                             source_date(title), seen, meta)
                written["units_seen" if existed else "units_new"] += 1
                written["discoveries"] += store.db.execute(
                    "SELECT count(*) FROM discoveries").fetchone()[0] - before
                if c:
                    c.setdefault("memory_units", [])
                    if uid not in c["memory_units"]:
                        c["memory_units"].append(uid)
        run.log("memory_capture", round=round_n, **written)
    except Exception as e:  # noqa: BLE001 - memory is optional
        memory_failed(run, f"capture-{round_n}", e)


def secondary_evidence(run, secondary):
    """The secondary claims as reasoner text, each visibly SECONDARY with its claim type,
    verification, provenance and original ids (never as SOURCE/PASSAGE); None when there are none.
    without the member's name or handle (the answer and its citations stay anonymous;
    Research Details keeps the author)."""
    if not secondary:
        return None
    try:
        import research_memory as rm
        blocks = []
        for n, s in enumerate(secondary, 1):
            claim, md = dict(s["claim"], author=None), s["metadata"]
            # A record rewritten by tools/community_statements.py is shown as its
            # plain-English statement, without platform, record ids or the raw chat wording.
            statement = clean(md.get("statement"), SECONDARY_WORDING_CHARS)
            if statement:
                claim.update(claim_text=statement, platform="community member", community=None,
                             source_date=(claim.get("source_date") or "")[:10] or None)
            else:
                claim["claim_text"] = clean(claim["claim_text"], SECONDARY_CLAIM_CHARS)
            lines = [f"SECONDARY [s{n}]", rm.secondary_evidence_block(claim)]
            record = claim.get("record_id")
            ids = md.get("original_ids") if isinstance(md.get("original_ids"), dict) else {}
            other = "; ".join(f"{k}: {', '.join(map(str, v))}" for k, v in ids.items()
                              if isinstance(v, list) and v)
            if (record or other) and not statement:
                lines.append(f"RECORD: {record or '—'}" + (f" ({other})" if other else ""))
            if md.get("cross_platform_merged") and not statement:
                lines.append(f"PLATFORMS: {', '.join(md.get('platforms') or [])} (merged claim)")
            pv = md.get("primary_verification") if isinstance(md.get("primary_verification"), dict) else {}
            checks = [c.get("status") for c in pv.get("checks") or [] if isinstance(c, dict) and c.get("status")]
            if checks:
                lines.append("EARLIER VERIFICATION CHECKS (not a link to primary evidence): "
                             + ", ".join(checks))
            if claim.get("primary_refs"):
                lines.append(f"CITED PRIMARY REFERENCES (not in the evidence above): {len(claim['primary_refs'])}")
            check = s.get("check") or {}
            if check:  # A failed or missing check is inconclusive, never "not confirmed"
                how = []
                if check.get("fact") and check.get("origin") in ("asked", "reused"):
                    how.append(f"asked as FACT QUESTION {check['fact']} (see the passages under "
                               f"it and its NotebookLM finding)")
                elif check.get("fact"):
                    how.append("its NotebookLM ask " + ("failed" if check.get("origin") == "failed"
                                                        else "was not used"))
                if check.get("search"):
                    how.append(f"source search \"{check['search']}\"")
                lines.append(f"VERIFICATION: \"{check['question']}\" - " + "; ".join(how)
                             + ". Confirmed only when a PASSAGE in the evidence states it; "
                               "otherwise inconclusive: present it as reported in the community.")
            wording = None if statement else clean(claim.get("context"), SECONDARY_WORDING_CHARS)
            if wording:
                lines.append(f"SECONDARY WORDING (excerpt):\n{wording}")
            blocks.append("\n".join(lines))
        return "\n\n=====\n\n".join(blocks)
    except Exception as e:  # noqa: BLE001 - memory is optional
        memory_failed(run, "secondary", e)
        return None


# Community precision: words too generic to identify a case subject by themselves.
# Corpus-specific: these suit the sample corpus; list the words every record of your library uses.
SUBJECT_GENERIC = frozenset("employee employees company policy policies handbook author speaker".split())


def stem(word):
    """A crude English stem (pure): lowercase, common inflection suffixes removed."""
    w = word.lower().strip("'’-")
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 4 and w.endswith("es") and w[-3] in "sxzh":
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    elif len(w) > 5 and w.endswith(("ing", "ed")):
        w = w[:-3] if w.endswith("ing") else w[:-2]
    return w[:-1] if len(w) > 3 and w.endswith("e") else w


def subject_stems(subject):
    """The case subject's content-word stems (pure); generic words dropped unless nothing else
    is left."""
    words = [w for w in re.findall(r"[\w'’-]+", str(subject or "").lower())
             if w not in QUERY_STOPWORDS]
    specific = [w for w in words if w not in SUBJECT_GENERIC]
    return {stem(w) for w in (specific or words)}


def filter_secondary(secondary, case_frame):
    """(kept, dropped) community records (pure): a record is kept when a community-facet search
    found it, or when its claim text contains every stem of the case_frame subject. Without a
    usable subject every record is kept. The pipeline does not apply it: it drops relevant
    records whose wording differs from the subject."""
    subject = case_frame.get("subject") if isinstance(case_frame, dict) else None
    want = subject_stems(subject)
    if not want:
        return list(secondary), []
    kept, dropped = [], []
    for s in secondary:
        text = s["claim"].get("claim_text") or ""
        have = {stem(w) for w in re.findall(r"[\w'’-]+", text.lower())}
        (kept if s.get("community_facet") or want <= have else dropped).append(s)
    return kept, dropped


def memory_summary(run, selected, secondary, used):
    """Memory activity for Research Details: remembered primary candidates, secondary candidates,
    what reached the reasoner, and passages written."""
    info = run.trace["memory"]
    kept = {h["hit_id"] for h in selected}
    info["secondary"] = [{"id": f"s{n}", "unit_id": s["unit_id"], "score": s["score"],
                          "matched": s["matched"], "key": s.get("key"),
                          "claim_type": s["claim"].get("claim_type"),
                          "platform": s["claim"].get("platform"), "author": s["claim"].get("author"),
                          "record_id": s["claim"].get("record_id"),
                          "verification": s["claim"].get("verification"),
                          "attribution_check": s.get("check"),
                          "preview": preview(s["claim"].get("claim_text") or "")}
                         for n, s in enumerate(secondary, 1)]
    info["retained"] = {
        "primary": [p["hit_id"] for p in info["primary"] if p["hit_id"] in kept],
        "fresh_with_memory": [h["hit_id"] for h in selected if h.get("memory_units") and h.get("origin") != "memory"],
        "secondary": [s["unit_id"] for s in secondary] if used else []}
    return info


def evidence_trace(run, titles):
    """Compact per-run trace answering: was a passage retrieved (raw), did it survive merge/dedupe
    (candidate), and did the selector keep it (decision). Previews only, never full passages."""
    raw = [{"round": r["round"], "id": r["raw_id"], "query": r["query"],
            "source": titles.get(r["source_id"]) or r["source_id"], "rank": r["rank"],
            "preview": preview(r["text"]), "candidate": r.get("candidate"), "merge": r.get("merge"),
            **{k: r[k] for k in ("origin", "cached_from", "retrieved_at", "citation", "recovery")
               if r.get(k)}}
           for r in run.trace["raw"]]
    decisions = {d["hit_id"]: d for d in run.trace["selector"]}
    candidates = []
    for c in run.trace["candidates"]:
        d = decisions.get(c["hit_id"], {})
        candidates.append({
            "id": c["hit_id"], "raw_ids": c.get("raw_ids", []),
            "source": titles.get(c["source_id"]) or c["source_id"], "preview": preview(c["text"]),
            "continuation": c.get("continuation_status") if c["needs_continuation"] else None,
            "role": d.get("role"), "covers": d.get("covers") or [], "reason": d.get("reason"),
            "kept": d.get("kept"), "context": d.get("context", False),
            **({"origin": c["origin"]} if c.get("origin") else {}),
            **({"memory_units": c["memory_units"]} if c.get("memory_units") else {}),
            **({"alternates": c["alternates"]} if c.get("alternates") else {}),
            **{k: d[k] for k in ("predicate", "scope", "applies_to", "use", "relations",
                          "reused_from") if k in d}})
    return {"raw": raw, "candidates": candidates, "memory": run.trace["memory"],
            "reuse_decision": run.trace["reuse_decision"],
            "selector_mode": run.trace["selector_mode"],
            "coverage_followup": run.trace.get("coverage_followup"),
            "cross_source": run.trace.get("cross_source", []),
            "referenced_recipes": run.trace.get("referenced_recipes"),
            "gemini_breaker": run.trace["gemini_breaker"], "ask": run.trace["ask"]}


# ---- research reuse ------------------------------------------------------------------------------
# Exact reuse: a completed result is served only for the same identity, the normalized
# question plus answer-affecting options, corpus identity and pipeline policy (see exact_lookup),
# before auth or any provider or model client. The retrieval cache serves a stored NotebookLM
# response only for the identical request on the same corpus identity (see retrieval_lookup).
# Identity is always decided deterministically, never by a model. Cache read and write errors
# are logged and count as misses; they never fail a request.
#
# Related-question reuse (established requirements, below) is kept but disabled by
# LEGACY_RELATED_REUSE_ENABLED.

def question_key(text):
    """Deterministic identity of a question: Unicode NFKC, case-folded, whitespace collapsed,
    trailing punctuation ignored."""
    text = " ".join(unicodedata.normalize("NFKC", text or "").casefold().split())
    return text.rstrip(" ?!.。？！…").strip()


def requirement_key(r):
    """Deterministic identity of an answer requirement: kind, exact-details flag and normalized
    text."""
    return f"{r.get('kind')}|{bool(r.get('exact'))}|{question_key(r.get('text'))}"


def sha256(value):
    """Hex digest of a string, or of a JSON value serialized with sorted keys."""
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True,
                                                           ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fresh_requested(fresh=False):
    """A fresh run was asked for, by the request or by CRA_FRESH."""
    return fresh or os.environ.get("CRA_FRESH", "").strip().lower() in ("1", "on", "true", "yes")


def corpus_identity():
    """The corpus the answer is researched against: the notebook id(s) and CORPUS_EPOCH. There is
    no local source manifest (the titles cache grows during runs, so it is not one): the identity
    is "weak", and entries older than REUSE_MAX_AGE_HOURS_WEAK_CORPUS miss. NotebookLM is never
    called to establish it."""
    manifest = None
    return {"notebooks": sorted([NOTEBOOK]), "epoch": CORPUS_EPOCH, "manifest": manifest,
            "strength": "strong" if manifest else "weak"}


def corpus_mismatch(stored, current):
    """Why a stored corpus identity differs from the current one, else None."""
    if not isinstance(stored, dict):
        return "corpus identity missing"
    if stored.get("notebooks") != current["notebooks"]:
        return "corpus notebooks changed"
    if stored.get("epoch") != current["epoch"]:
        return f"corpus epoch changed ({stored.get('epoch')} -> {current['epoch']})"
    if stored.get("manifest") != current["manifest"]:
        return "corpus manifest changed"
    return None


def too_old(corpus, ts):
    """Why an entry recorded at `ts` is too old under a weak corpus identity, else None."""
    if corpus["strength"] != "weak":
        return None
    if not isinstance(ts, (int, float)):
        return "entry age unknown under weak corpus identity"
    hours = (time.time() - ts) / 3600
    if hours > REUSE_MAX_AGE_HOURS_WEAK_CORPUS:
        return (f"too old under weak corpus identity ({hours:.0f} h > "
                f"{REUSE_MAX_AGE_HOURS_WEAK_CORPUS:.0f} h)")
    return None


def answer_options():
    """Request options and configuration that change the answer, as one JSON value."""
    return {"notebooks": sorted([NOTEBOOK]), "memory": not memory_disabled()}


def policy_fingerprint():
    """Hash of what decides the pipeline's behavior: the planner, selector and reasoner prompt
    templates and output schemas (the repair round uses the reasoner's), the configured model
    ids (never response model strings), retrieval and evidence limits, and
    PIPELINE_POLICY_VERSION for behavior these do not capture."""
    gemini = {"model": None, "enabled": None}
    try:
        from server import connections
        gemini = {"model": connections.worker.model(),
                  "enabled": not connections.CONNECTIONS["gemini"].disabled}
    except Exception:  # noqa: BLE001 - Gemini is optional; its absence is part of the policy
        pass
    return sha256({
        "version": PIPELINE_POLICY_VERSION,
        "prompts": [PLANNER_TEXT_SYSTEM, SELECTOR_SYSTEM, REASONER_SYSTEM, COVERAGE_SYSTEM,
                    LEDGER_SYSTEM],
        "schemas": [PLANNER_SCHEMA, SELECTOR_SCHEMA, REASONER_SCHEMA, REPAIRABLE_SCHEMA],
        "models": {"planner": PLANNER, "selector": gemini, "selector_fallback": SELECTOR,
                   "reasoner": REASONER, "coverage": COVERAGE,
                   "answer_thinking": ANSWER_THINKING_BUDGET},
        "limits": {"searches": DEPTH_MAX_SEARCHES, "search_limit": SEARCH_LIMIT,
                   "repair": REPAIR_MAX_SEARCHES, "evidence": EVIDENCE_BUDGET,
                   "memory": [MEMORY_PRIMARY_MAX, MEMORY_SECONDARY_MAX, MEMORY_CITED_MAX],
                   "selector_output": SELECTOR_MAX_OUTPUT_TOKENS,
                   "retrieval_parse": RETRIEVAL_PARSE_VERSION}})


def write_json_atomic(path, data):
    """Write JSON under a temporary name, flush it to disk and move it into place, so a reader
    never sees a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.part")
    try:
        with open(part, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(part, path)
    finally:
        part.unlink(missing_ok=True)


def read_entry(path):
    """(index entry, None), or (None, miss reason)."""
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "no entry"
    except (OSError, ValueError) as e:
        return None, f"entry unreadable ({type(e).__name__})"
    return (entry, None) if isinstance(entry, dict) else (None, "entry unreadable")


def result_identity(question):
    """(index key, normalized question, answer-affecting options) of a question."""
    nq, options = question_key(question), answer_options()
    return sha256({"question": nq, "options": options}), nq, options


def exact_lookup(question, fresh=False):
    """Look the question up in the exact-result index, touching no provider or model client.
    Returns {"hit", "key" (prefix), "reason" (why it missed, None on a hit)} and, on a hit, the
    source run, its age, corpus identity and policy match, and the stored "result"."""
    key, nq, options = result_identity(question)
    out = {"hit": False, "key": key[:12], "reason": None}
    if fresh_requested(fresh):
        out["reason"] = "bypassed (fresh run)"
        return out
    try:
        entry, reason = read_entry(REUSE_DIR / "results" / f"{key}.json")
        corpus = corpus_identity()
        if entry is not None:
            if entry.get("question") != nq or entry.get("options") != options:
                reason = "no entry"
            else:
                reason = (corpus_mismatch(entry.get("corpus"), corpus)
                          or ("pipeline policy changed" if entry.get("policy")
                              != policy_fingerprint() else None)
                          or too_old(corpus, entry.get("completed_ts")))
        result = None
        if reason is None:
            try:
                result = json.loads(Path(entry["result_path"]).read_text(encoding="utf-8"))
            except (OSError, ValueError, KeyError, TypeError):
                pass
            if not (isinstance(result, dict) and (result.get("answer") or "").strip()
                    and isinstance(result.get("details"), dict)):
                reason = "source run missing or corrupt"
    except Exception as e:  # noqa: BLE001 - a failed lookup is a miss
        reason = f"lookup error ({e!r})"[:200]
    if reason is not None:
        out["reason"] = reason
        return out
    out.update(hit=True, source_run=entry.get("run_id"), source_run_dir=entry.get("run_dir"),
               completed_at=entry.get("completed_at"),
               age_hours=round((time.time() - entry["completed_ts"]) / 3600, 2),
               corpus=entry["corpus"], corpus_strength=corpus["strength"], policy_match=True,
               result=result)
    return out


def prior_answer(question, fresh=False):
    """The latest completed answer to the same question identity (the exact-result
    index entry), for this run to re-check instead of serving it: (record, None) or (None, why
    it missed). The record is the earlier run's turn record (question, short answer, headings,
    cited passages) plus its full answer, source run and age. Pipeline policy and age are not
    checked, since the answer is re-derived from evidence; the corpus notebooks and epoch must
    match, since its passages come from them."""
    if fresh_requested(fresh):
        return None, "bypassed (fresh run)"
    key, nq, options = result_identity(question)
    try:
        entry, reason = read_entry(REUSE_DIR / "results" / f"{key}.json")
        if entry is not None and (entry.get("question") != nq or entry.get("options") != options):
            reason = "no entry"
        if reason is None:
            reason = corpus_mismatch(entry.get("corpus"), corpus_identity())
        if reason is None:
            folder = Path(entry["run_dir"])
            turn = load_turn(folder)
            answer = (json.loads((folder / RESULT_FILE).read_text(encoding="utf-8"))
                      .get("answer") or "").strip()
            if not answer:
                reason = "source run missing or corrupt"
    except Exception as e:  # noqa: BLE001 - a failed lookup is a miss
        reason = f"lookup error ({e!r})"[:200]
    if reason is not None:
        return None, reason
    ts = entry.get("completed_ts")
    return {**turn, "answer": answer, "key": key[:12], "source_run": entry.get("run_id"),
            "run_dir": entry.get("run_dir"), "completed_at": entry.get("completed_at"),
            "age_hours": round((time.time() - ts) / 3600, 2) if isinstance(ts, (int, float))
            else None, "policy_version": entry.get("policy_version")}, None


def prior_block(prior):
    """The EARLIER ANSWER block for the answer pass (pure), or None: the earlier answer to this
    same question with its citation marks removed (they numbered another run's evidence; the
    passages it cited are in this run's evidence, marked as from the earlier turn)."""
    if not prior:
        return None
    text = re.sub(r"\s*\[(?:h|s)\d+(?:\s*[,;]\s*(?:h|s)?\d+)*\]", "", prior["answer"])
    when = f", {prior['age_hours']:.0f} h ago" if prior.get("age_hours") is not None else ""
    return f"(answered{when})\n\n{text.strip()}"


def start_log(run):
    """Create the run's log folder (its run-history record) and record the question."""
    slug = re.sub(r"[^a-z0-9]+", "-", run.question.lower()).strip("-")[:50]
    run.dir = settings.LOGS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}"
    run.dir.mkdir(parents=True)
    run.save("question.txt", run.question)
    run.log("start", question=run.question)


def exact_reuse_result(run, lookup, run_start):
    """The stored result of an identical earlier question, unchanged, with details.exact_reuse
    naming its source. Records a lightweight run-history entry of type exact_reuse (zero usage,
    never indexed). No provider or model client is constructed."""
    info = {k: lookup[k] for k in ("hit", "key", "source_run", "source_run_dir", "completed_at",
                                   "age_hours", "corpus", "corpus_strength", "policy_match")}
    for stage in ("auth", "plan", "search", "continuation", "select", "context", "reason",
                  "repair", "check", "read", "write"):
        run.emit("stage_skip", stage=stage, detail="Reused an identical earlier result")
    try:
        start_log(run)
        info["record_dir"] = str(run.dir)
        run.log("exact_reuse", key=info["key"], outcome="hit", source_run=info["source_run"],
                age_hours=info["age_hours"], corpus_strength=info["corpus_strength"])
        run.save("exact-reuse.json", json.dumps(
            {"type": "exact_reuse", **info, "usage": [], "claude_cost_usd": 0,
             "seconds": round(time.monotonic() - run_start, 2)}, indent=1, ensure_ascii=False))
    except OSError as e:  # the record is optional; the reused result is still served
        info["record_error"] = repr(e)[:200]
    info["seconds"] = round(time.monotonic() - run_start, 2)
    result = lookup["result"]
    result = {**result, "details": {**result["details"], "exact_reuse": info}}
    run.emit("answer", answer=result["answer"])
    return result


def index_result(run, lookup):
    """Point the exact-result index at this completed run, after its result.json is durably
    written. A newer completed run of the same key supersedes the older entry."""
    try:
        path = run.dir / RESULT_FILE
        if not path.is_file():
            run.log("exact_index", key=lookup["key"], written=False, reason="result not saved")
            return
        key, nq, options = result_identity(run.question)
        now = time.time()
        write_json_atomic(REUSE_DIR / "results" / f"{key}.json", {
            "version": 1, "question": nq, "options": options, "policy": policy_fingerprint(),
            "policy_version": PIPELINE_POLICY_VERSION, "corpus": corpus_identity(),
            "run_id": run.dir.name, "run_dir": str(run.dir), "result_path": str(path),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
            "completed_ts": now})
        run.log("exact_index", key=key[:12], written=True)
    except Exception as e:  # noqa: BLE001 - a failed index write only costs a later miss
        run.log("exact_index", key=lookup["key"], written=False, reason=repr(e)[:200])


def search_args(query):
    """The NotebookLM source-search request for one query ("--" so a query starting with "-" is
    not parsed as an option)."""
    return ["source", "search", "-n", NOTEBOOK, "--limit", str(SEARCH_LIMIT), "--json", "--",
            query]


def retrieval_key(query):
    """Identity of a NotebookLM search request: the full request with the query as sent (NFKC,
    whitespace collapsed), and the parse version of its response."""
    query = " ".join(unicodedata.normalize("NFKC", query).split())
    return sha256({"request": search_args(query), "parse": RETRIEVAL_PARSE_VERSION})


def is_passage(r):
    """A search response item search() uses: a source id and non-empty text."""
    return isinstance(r, dict) and bool(r.get("source_id")) and bool((r.get("text") or "").strip())


def retrieval_lookup(run, query):
    """A stored response to the identical search request on the same corpus identity:
    ({"result" (the parsed response, identical to the original), "run_id", "retrieved_at"}, None,
    key), or (None, miss reason, key). Every passage must still load from primary memory with
    its exact text."""
    key = retrieval_key(query)
    if run.fresh:
        return None, "bypassed (fresh run)", key
    try:
        entry, reason = read_entry(REUSE_DIR / "retrieval" / f"{key}.json")
        if entry is None:
            return None, reason, key
        corpus = corpus_identity()
        reason = (("parse version changed" if entry.get("parse") != RETRIEVAL_PARSE_VERSION
                   else None)
                  or corpus_mismatch(entry.get("corpus"), corpus)
                  or too_old(corpus, entry.get("retrieved_ts")))
        if reason:
            return None, reason, key
        store = open_memory(create=False)
        if store is None:
            return None, "research memory unavailable", key
        import research_memory as rm
        result = []
        with store:
            for item in entry["items"]:
                if "inline" in item:
                    result.append(item["inline"])
                    continue
                row = store.db.execute("SELECT layer, text FROM units WHERE unit_id = ?",
                                       (item["unit_id"],)).fetchone()
                if row is None or row["layer"] != rm.PRIMARY or sha256(row["text"]) != item["sha256"]:
                    return None, "passage missing from research memory", key
                result.append({k: row["text"] if k == "text" else v
                               for k, v in item["fields"].items()})
        return {"result": result, "run_id": entry.get("run_id"),
                "retrieved_at": entry.get("retrieved_at")}, None, key
    except Exception as e:  # noqa: BLE001 - a failed lookup is a miss
        return None, f"lookup error ({e!r})"[:200], key


def retrieval_store(run, query, key, result, round_n):
    """Cache one successful search response: its passages go to primary memory (the same units
    the run's capture records) and the entry stores references to them. Nothing is written for
    a response without passages, or when memory is off."""
    reason = None
    try:
        if not any(is_passage(r) for r in result):
            reason = "no passages"
        else:
            store = open_memory(create=True)
            if store is None:
                reason = "research memory disabled"
            else:
                import research_memory as rm
                items = []
                with store:
                    for r in result:
                        if not is_passage(r):
                            items.append({"inline": r})
                            continue
                        uid = rm.unit_id_for(rm.PRIMARY, rm.primary_key(r["source_id"], None),
                                             rm.fingerprint(r["text"]))
                        if store.layer(uid) is None:
                            run.cache_units.add(uid)
                        uid = store.add_primary(r["text"], r["source_id"])
                        stored = store.db.execute("SELECT text FROM units WHERE unit_id = ?",
                                                  (uid,)).fetchone()["text"]
                        if stored != r["text"]:  # an earlier whitespace variant of the passage
                            reason = "passage differs from its stored unit"
                            break
                        items.append({"unit_id": uid, "sha256": sha256(r["text"]),
                                      "fields": {k: None if k == "text" else v
                                                 for k, v in r.items()}})
                if reason is None:
                    now = time.time()
                    write_json_atomic(REUSE_DIR / "retrieval" / f"{key}.json", {
                        "version": 1, "query": query, "notebook": NOTEBOOK,
                        "parse": RETRIEVAL_PARSE_VERSION, "corpus": corpus_identity(),
                        "run_id": run.dir.name if run.dir else None, "round": round_n,
                        "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
                        "retrieved_ts": now, "items": items})
    except Exception as e:  # noqa: BLE001 - a failed cache write only costs a later miss
        reason = f"write error ({e!r})"[:200]
    run.log("retrieval_cache_write", key=key[:12], query=query, written=reason is None,
            **({"reason": reason} if reason else {}))


def discovered_in_corpus(store, unit_id, corpus):
    """Whether a primary unit was retrieved at least once under the current corpus identity.
    Discoveries recorded before they carried one count as LEGACY_CORPUS."""
    for row in store.db.execute("SELECT metadata FROM discoveries WHERE unit_id = ?", (unit_id,)):
        try:
            md = json.loads(row["metadata"]) if row["metadata"] else {}
        except ValueError:
            continue
        stored = md.get("corpus", LEGACY_CORPUS) if isinstance(md, dict) else LEGACY_CORPUS
        if corpus_mismatch(stored, corpus) is None:
            return True
    return False


def established_requirements(run, requirements):
    """The current requirements an earlier completed run already established: a requirement with
    the same key (see requirement_key) that its selector rated covered, where every passage
    behind that rating is a kept primary unit whose recorded judgment covers it. Returns
    {current requirement id: {"run_id", "requirement_id" (the earlier id), "rows" (the earlier
    selection records, with each unit's text and source)}}. Similar wording alone never counts."""
    found = {}
    try:
        store = open_memory(create=False)
        if store is None:
            return found
        import research_memory as rm
        wanted = {requirement_key(r): r["id"] for r in requirements}
        with store:
            for prior in store.completed_runs(CORPUS_VERSION):
                md = prior["metadata"]
                prior_reqs = {x.get("id"): x for x in md.get("requirements") or []
                              if isinstance(x, dict)}
                selections = None
                for cov in md.get("coverage") or []:
                    prior_req = prior_reqs.get(cov.get("requirement_id"))
                    if not prior_req or cov.get("status") != "covered":
                        continue
                    rid = wanted.get(requirement_key(prior_req))
                    if rid is None or rid in found:
                        continue
                    if selections is None:
                        selections = store.run_selections(prior["run_id"])
                    rows = [selections.get((1, h)) for h in cov.get("hit_ids") or []]
                    if not rows or any(s is None or s["layer"] != rm.PRIMARY or s["kept"] != 1
                                       or not s["source_id"] or prior_req["id"] not in s["covers"]
                                       or s["role"] not in ("CORE", "CONTRAST", "SUPPORT")
                                       for s in rows):
                        continue
                    found[rid] = {"run_id": prior["run_id"], "requirement_id": prior_req["id"],
                                  "rows": rows}
                if len(found) == len(wanted):
                    break
    except Exception as e:  # noqa: BLE001 - memory is optional; failure means fresh research
        memory_failed(run, "established", e)
        return {}
    run.log("established_requirements", requirements={
        rid: {"run_id": e["run_id"], "requirement_id": e["requirement_id"],
              "units": [s["unit_id"] for s in e["rows"]]} for rid, e in found.items()})
    return found


def add_established(run, candidates, established, titles):
    """Add the passages behind established requirements to the round-1 candidates, each with its
    earlier selector judgment (covers mapped to the current requirement ids, relations to the
    other reused hits of the same earlier run). A passage a fresh hit already contains is not
    added again: the fresh candidate takes the reused judgment. Returns {hit_id: {"decision",
    "context", "run_id"}}."""
    info = run.trace["memory"]["reuse"]
    current = {(e["run_id"], e["requirement_id"]): rid for rid, e in established.items()}
    units = {}
    for rid, e in established.items():
        info["requirements"].append({"id": rid, "from_run": e["run_id"],
                                     "from_requirement": e["requirement_id"]})
        for row in e["rows"]:
            units.setdefault(row["unit_id"], []).append((e["run_id"], row))
    reused, hit_of = {}, {}
    for uid, rows in units.items():
        run_id, row = rows[0]
        probe = {"source_id": row["source_id"], "text": row["text"], "start": None, "end": None}
        same = next((c for c in candidates if c["hit_id"] not in reused
                     and c["source_id"] == row["source_id"] and duplicate_reason(c, probe)), None)
        if same is None:
            reason = continuation_reason(row["text"])
            same = {"hit_id": f"h{len(candidates) + 1}", "source_id": row["source_id"],
                    "text": row["text"], "start": None, "end": None,
                    "found_by": [{"query": "(established research)", "rank": None, "start": None,
                                  "end": None}],
                    "raw_ids": [], "rank": None, "needs_continuation": reason is not None,
                    "continuation_reason": reason, "origin": "established",
                    "memory_units": [uid]}
            candidates.append(same)
            run.trace["candidates"].append(same)
            if row["source_title"] and not titles.get(row["source_id"]):
                titles[row["source_id"]] = row["source_title"]
        elif uid not in same.setdefault("memory_units", []):
            same["memory_units"].append(uid)
        md = row["metadata"] or {}
        covers = [current[(r, x)] for r, s in rows for x in s["covers"] if (r, x) in current]
        reused[same["hit_id"]] = {
            "run_id": run_id, "relations": md.get("relations") or [],
            "context": any(s["context_requested"] == 1 for _, s in rows),
            "decision": {"role": row["role"], "covers": list(dict.fromkeys(covers)),
                         "reason": row["reason"] or "", "predicate": md.get("predicate") or "",
                         "predicate_verbatim": bool(md.get("predicate_verbatim")),
                         "scope": md.get("scope") if md.get("scope") in SCOPES else "unclear",
                         "applies_to": md.get("applies_to") or "",
                         "use": md.get("use") if md.get("use") in USES else "answer",
                         "relations": [], "reused_from": run_id}}
        hit_of[uid] = same["hit_id"]
        info["primary"].append({"unit_id": uid, "hit_id": same["hit_id"], "from_run": run_id,
                                "merged_into_fresh": same.get("origin") != "established",
                                "source": row["source_title"] or row["source_id"],
                                "preview": preview(row["text"])})
    for hid, r in reused.items():
        rels = r["decision"]["relations"]
        for x in r.pop("relations"):
            target = hit_of.get(x.get("unit_id")) if isinstance(x, dict) else None
            if (target and target != hid and x.get("type") in RELATION_TYPES
                    and reused[target]["run_id"] == r["run_id"]
                    and {"type": x["type"], "hit_id": target} not in rels):
                rels.append({"type": x["type"], "hit_id": target})
    run.log("established_candidates", hits={h: r["run_id"] for h, r in reused.items()})
    return reused


def select_with_reuse(run, question, depth, candidates, titles, requirements, established, reused):
    """Selection when some candidates carry an established judgment: those keep it (no selector
    call for them) and the selector judges only the other candidates, against all requirements.
    An established requirement stays covered by its reused passages (plus any fresh ones the
    selector says cover it); the others take the selector's coverage. Returns (selected, context
    hit ids, coverage) like select()."""
    info = run.trace["memory"]["reuse"]
    pending = [c for c in candidates if c["hit_id"] not in reused]
    info["selector_fresh"] = [c["hit_id"] for c in pending]
    info["selector_reused"] = list(reused)
    if not reused:
        return select(run, question, depth, candidates, titles, requirements)
    fresh, fresh_context, fresh_coverage = [], [], []
    if pending:
        fresh, fresh_context, fresh_coverage = select(run, question, depth, pending, titles,
                                                      requirements)
    ids = list(reused)
    for i in ids:
        run.trace["selector"].append({"round": 1, "hit_id": i, "kept": True,
                                      "context": reused[i]["context"], **reused[i]["decision"]})
        run.claims[i] = reused[i]["decision"]
    by_fresh = {c["requirement_id"]: c for c in fresh_coverage}
    coverage = []
    for r in requirements:
        if r["id"] in established:
            hit_ids = [i for i in ids if r["id"] in reused[i]["decision"]["covers"]]
            more = (by_fresh.get(r["id"]) or {}).get("hit_ids") or []
            coverage.append({"requirement_id": r["id"], "status": "covered",
                             "hit_ids": hit_ids + [i for i in more if i not in hit_ids],
                             "missing": "", "reused_from": established[r["id"]]["run_id"]})
        elif r["id"] in by_fresh:
            coverage.append(by_fresh[r["id"]])
        else:
            coverage.append({"requirement_id": r["id"], "status": "missing", "hit_ids": [],
                             "missing": "no new passages were found for it"})
    run.log("selector_coverage", round=1, coverage=coverage, reused=ids)
    by_id = {c["hit_id"]: c for c in candidates}
    selected = [by_id[i] for i in ids] + fresh
    context_ids = [i for i in ids if reused[i]["context"]] + fresh_context
    return selected, context_ids, coverage


def persist_research(run, depth, requirements, candidates, coverage, final_selected, synthesis,
                     answer, result, titles):
    """After a successful run, record its established structure in memory: every selector
    judgment (role, covers, claim structure, relations by unit), the relations between kept
    primary units, the answer as a DERIVED unit linked to the primary passages it used, and the
    run's requirements, coverage (with unit ids) and reuse identity. Then save the result for
    exact reuse. PRIMARY units stay exact source text; nothing inferred is stored as primary."""
    written = run.trace["memory"]["written"]
    run_id = run.dir.name
    try:
        store = open_memory(create=True)
        if store is not None:
            import research_memory as rm
            with store:
                units = {}
                for c in candidates:
                    uid = rm.unit_id_for(rm.PRIMARY, rm.primary_key(c["source_id"],
                                                                     titles.get(c["source_id"])),
                                         rm.fingerprint(c["text"]))
                    if store.layer(uid) == rm.PRIMARY:
                        units[c["hit_id"]] = uid
                final_ids = {h["hit_id"] for h in final_selected}
                for d in run.trace["selector"]:
                    uid = units.get(d["hit_id"])
                    if uid is None:
                        continue
                    rels = [{"type": x["type"], "unit_id": units[x["hit_id"]]}
                            for x in d.get("relations") or [] if x.get("hit_id") in units
                            and units[x["hit_id"]] != uid]
                    meta = {k: d[k] for k in ("predicate", "predicate_verbatim", "scope",
                                              "applies_to", "use", "reused_from") if k in d}
                    if rels:
                        meta["relations"] = rels
                    store.add_selection(uid, run_id, d["round"], d["hit_id"], d.get("role"),
                                        d.get("covers"), d.get("reason"), d.get("kept"),
                                        d.get("context"), d["hit_id"] in final_ids, meta or None)
                    written["selections"] += 1
                    if d.get("kept"):
                        for x in rels:
                            store.add_relationship(uid, x["type"], x["unit_id"],
                                                   metadata={"run_id": run_id, "by": "selector"})
                            written["relationships"] += 1
                used = [i for it in synthesis if it["treatment"] != "not_used"
                        for i in it["hit_ids"]] or [h["hit_id"] for h in final_selected]
                supports = [units[i] for i in dict.fromkeys(used) if i in units]
                answer_unit = None
                if supports:  # a derived conclusion always links to its primary support
                    retained = run.trace["memory"].get("retained") or {}
                    inspired = [u for u in retained.get("secondary", [])
                                if store.layer(u) == rm.SECONDARY]
                    answer_unit = store.add_derived(
                        answer, "research_answer", supports, inspired,
                        producer=f"research:{run_id}",
                        metadata={"run_id": run_id, "question": run.question,
                                  "corpus_version": CORPUS_VERSION,
                                  "synthesis": [{**it, "unit_ids": [units.get(i) for i in it["hit_ids"]]}
                                                for it in synthesis]})
                    written["derived"] += 1
                # Each cited passage with the question and requirement it was cited for
                memory_units = {c["hit_id"]: [u for u in c.get("memory_units") or []
                                              if not u.startswith("prev:")] for c in candidates}
                for row in rm.citation_rows(result):
                    uid = units.get(row["hit_id"]) or rm.unit_id_for(
                        rm.PRIMARY, rm.primary_key(row["source_id"], row["title"]),
                        rm.fingerprint(row["text"]))
                    if store.layer(uid) != rm.PRIMARY:
                        uid = next((u for u in memory_units.get(row["hit_id"]) or []
                                    if store.layer(u) == rm.PRIMARY), None)
                    if uid:
                        written["citations"] = written.get("citations", 0) + store.add_citation(
                            uid, run_id, row["question"], row["standalone"], row["requirement"])
                reuse = run.trace["memory"]["reuse"]
                store.add_run(run_id, run.question, depth, None, str(run.dir), {
                    "completed_research": True, "corpus_version": CORPUS_VERSION,
                    "question_key": question_key(run.question), "result_file": RESULT_FILE,
                    "requirements": [{**r, "exact": bool(r.get("exact")), "key": requirement_key(r)}
                                     for r in requirements],
                    "coverage": [{**c, "unit_ids": [units.get(i) for i in c["hit_ids"]]}
                                 for c in coverage],
                    "searches": {"performed": reuse["searches_fresh"],
                                 "skipped": reuse["searches_skipped"]},
                    "answer_unit": answer_unit,
                    "reused_requirements": reuse["requirements"]})
            run.log("memory_persist", **written)
    except Exception as e:  # noqa: BLE001 - memory is optional
        memory_failed(run, "persist", e)
    try:  # the run-history record exact reuse serves: written whole or not at all
        write_json_atomic(run.dir / RESULT_FILE, result)
    except (OSError, TypeError, ValueError) as e:
        memory_failed(run, "result", e)


# ---- research-need ledger and shadow reuse gate -----------------------------------------
# The planner's case frame gives a deterministic need signature (research_need.py). Each run is
# classified against the ledger (research_memory table research_needs) and, for SAME_NEED, the
# selection-reuse gate is evaluated and recorded in the trace. Shadow mode: the selector always
# runs and nothing recorded reaches any stage. Errors are logged and never fail a request.

def selector_policy():
    """Hash of what decides a selection: the selector prompt and schema, the configured selector
    models, the limits that shape its input and output, and PIPELINE_POLICY_VERSION."""
    gemini = {"model": None, "enabled": None}
    try:
        from server import connections
        gemini = {"model": connections.worker.model(),
                  "enabled": not connections.CONNECTIONS["gemini"].disabled}
    except Exception:  # noqa: BLE001 - Gemini is optional; its absence is part of the policy
        pass
    return sha256({"version": PIPELINE_POLICY_VERSION, "prompt": SELECTOR_SYSTEM,
                   "schema": SELECTOR_SCHEMA,
                   "models": {"selector": gemini, "selector_fallback": SELECTOR},
                   "limits": {"selector_output": SELECTOR_MAX_OUTPUT_TOKENS,
                              "search_limit": SEARCH_LIMIT,
                              "memory": [MEMORY_PRIMARY_MAX, MEMORY_SECONDARY_MAX]}})


def record_memory_lookup(run, question, remembered, secondary):
    """The memory lookup keys (question + planner searches + case frame; FTS terms of each), the
    hit count per key before capping, and the capped hit counts."""
    try:
        import research_memory as rm
        key = " ".join(rm.query_terms(question))
    except Exception:  # noqa: BLE001 - trace only
        key = None
    run.trace["reuse_decision"]["memory_lookup"] = {
        "key_source": "question+planner", "key": key,
        "keys": run.trace["memory"].get("lookup_keys") or [],
        "primary_hits": len(remembered), "secondary_hits": len(secondary)}


def selection_units(units, selected, context_ids, coverage, decisions):
    """The round-1 selection as evidence structuring consumed it (kept hits in priority order,
    context hits, coverage, the selector's decisions) with every hit id rewritten as its primary
    unit id; None when any id has no unit."""
    ids = ([s["hit_id"] for s in selected] + list(context_ids) + [d["hit_id"] for d in decisions]
           + [i for c in coverage for i in c["hit_ids"] + c.get("leads", [])])
    if any(i not in units for i in ids):
        return None
    out = {}
    for d in decisions:
        item = {k: v for k, v in d.items() if k not in ("round", "hit_id")}
        if item.get("relations"):
            item["relations"] = [{**x, "hit_id": units.get(x.get("hit_id"))}
                                 for x in item["relations"]]
        out[units[d["hit_id"]]] = item
    return {"units": [units[s["hit_id"]] for s in selected],
            "context": [units[i] for i in context_ids],
            "coverage": [{**c, "hit_ids": [units[i] for i in c["hit_ids"]],
                          **({"leads": [units[i] for i in c["leads"]]} if "leads" in c else {})}
                         for c in coverage],
            "decisions": out}


def need_shadow(run, requirements, candidates, titles, selected, context_ids, coverage,
                selector_primary):
    """Classify this run's research need against the ledger and evaluate the reuse gate (shadow:
    recorded in the trace, never acted on); keep the need record for write_need_ledger. A
    bypassed round-1 selector made no selection to classify or reuse: recorded and skipped."""
    block = run.trace["reuse_decision"]
    if bypassed(run, 1):
        block["selector"] = "bypassed"
        block["stages"] = {"selector": "bypassed"}
        run.log("reuse_decision", **block)
        return
    try:
        import research_memory as rm
        import research_need as rn
        frame = rn.normalize_frame(run.case_frame_raw)
        signature = rn.need_signature(frame)
        policy, corpus = selector_policy(), corpus_identity()
        units, pool = {}, {}  # hit id -> unit id; selector-input unit id -> origin
        for c in candidates:
            uid = rm.unit_id_for(rm.PRIMARY, rm.primary_key(c["source_id"],
                                                             titles.get(c["source_id"])),
                                 rm.fingerprint(c["text"]))
            units[c["hit_id"]] = uid
            pool.setdefault(uid, "memory" if c.get("origin") == "memory" else "retrieval")
        decisions = [d for d in run.trace["selector"] if d["round"] == 1]
        facets = sorted({r["kind"] for r in requirements})
        run.need = {"frame": frame, "signature": signature, "policy": policy, "corpus": corpus,
                    "requirements": requirements, "facets": facets, "input_units": list(pool),
                    "selection": selection_units(units, selected, context_ids, coverage,
                                                 decisions),
                    "selection_args": (units, selected, context_ids, decisions),
                    "selector_primary": selector_primary}
        situation = match = diff = None
        resolvable, ledger = set(), "no_frame"
        if frame:
            store = open_memory(create=False)
            ledger = "ok" if store else ("disabled" if memory_disabled() else "no_database")
            if store is not None:
                with store:
                    situation, match, diff = rn.classify(frame,
                                                         store.needs_for_subject(frame["subject"]))
                    if situation == "SAME_NEED" and match.get("selection_units"):
                        resolvable = store.primary_units_present(match["selection_units"])
        current = {"signature": signature, "selector_policy": policy, "corpus": corpus,
                   "pool": pool, "facets": facets, "reuse_disabled": fresh_requested(run.fresh)}
        block["need"] = {"version": rn.NEED_SIGNATURE_VERSION, "signature": signature,
                         "frame": frame, "situation": situation, "ledger": ledger,
                         "matched_need_id": match["need_id"] if match else None,
                         "frame_diff": diff}
        block["gate"] = rn.gate(situation, current, match, resolvable, time.time(),
                                REUSE_MAX_AGE_HOURS_WEAK_CORPUS)
        retrieval = run.trace["retrieval"]
        block["stages"] = {"planner": "computed",
                           "retrieval": (f"cache_hit {retrieval['from_cache']}/"
                                         f"{retrieval['queries']}" if retrieval["from_cache"]
                                         else "computed"),
                           "selector": "computed", "reasoner": "computed"}
        block["versions"] = {"selector_policy": policy, "corpus": f"{NOTEBOOK}/{CORPUS_EPOCH}"}
    except Exception as e:  # noqa: BLE001 - shadow only; never fails a request
        block["error"] = repr(e)[:300]
    run.log("reuse_decision", **block)


def write_need_ledger(run, coverage=None):
    """After a completed run, upsert its research need's ledger entry, keyed on (signature,
    selector policy, corpus identity), when the round-1 selection came from the configured
    primary selector. `coverage` (the run's final coverage) replaces the selector's round-1
    coverage in the stored selection."""
    block, need = run.trace["reuse_decision"], run.need
    if bypassed(run, 1):
        block["ledger_write"] = "skipped: selector bypassed"
        return
    if need is not None and coverage is not None and need.get("selection_args"):
        units, selected, context_ids, decisions = need["selection_args"]
        need["selection"] = selection_units(units, selected, context_ids, coverage, decisions)
    if need is None or not need["frame"]:
        block["ledger_write"] = "skipped: no case frame"
        return
    if not need["selector_primary"]:
        block["ledger_write"] = "skipped: fallback selector"
        return
    try:
        store = open_memory(create=True)
        if store is None:
            block["ledger_write"] = "skipped: memory disabled"
            return
        import research_need as rn
        corpus = {k: need["corpus"].get(k) for k in ("notebooks", "epoch", "manifest", "strength")}
        need_id = "N-" + sha256({"signature": need["signature"], "policy": need["policy"],
                                 "corpus": [corpus["notebooks"], corpus["epoch"],
                                            corpus["manifest"]]})[:16]
        with store:
            store.upsert_need(need_id, need["signature"], rn.NEED_SIGNATURE_VERSION, need["frame"],
                              need["policy"], corpus, need["requirements"], need["facets"],
                              need["input_units"], need["selection"], True, run.dir.name,
                              time.time())
        block["ledger_write"] = f"upserted {need_id}"
    except Exception as e:  # noqa: BLE001 - memory is optional
        memory_failed(run, "need_ledger", e)
        block["ledger_write"] = "error"


PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent\b)", re.I)


def ungrounded_percentages(answer, evidence):
    """Percentages the answer states that appear nowhere in the evidence (100% excepted): a
    diagnostic for invented ratios, shown in Research Details. The answer is never changed."""
    known = {m.group(1) for m in PERCENT.finditer(evidence)} | {"100"}
    return list(dict.fromkeys(m.group(0) for m in PERCENT.finditer(answer)
                              if m.group(1) not in known))


# ---- evidence map: how the kept statements relate, per requirement ---------------------------

DATE_NUMERIC = re.compile(r"(?<!\d)(19[4-9]\d|20[0-4]\d)(?:[-_./](0?[1-9]|1[0-2])"
                          r"(?:[-_./](0?[1-9]|[12]\d|3[01]))?)?(?!\d)")
DATE_WORDS = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+"
                        r"(?:(\d{1,2})(?:st|nd|rd|th)?,?\s+)?(19[4-9]\d|20[0-4]\d)\b", re.I)
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def source_date(title):
    """The date a source title carries, as a sortable "YYYY", "YYYY-MM" or "YYYY-MM-DD", else
    None. Only the title is used: a date is never guessed from the passage."""
    title = title or ""
    m = DATE_WORDS.search(title)
    if m:
        month = f"{MONTHS.index(m.group(1).lower()[:3]) + 1:02d}"
        return m.group(3) + f"-{month}" + (f"-{int(m.group(2)):02d}" if m.group(2) else "")
    m = DATE_NUMERIC.search(title)
    if not m:
        return None
    return m.group(1) + (f"-{int(m.group(2)):02d}" if m.group(2) else "") \
        + (f"-{int(m.group(3)):02d}" if m.group(3) else "")


# Readable source names (citations, the sources panel, the evidence SOURCE line).
# Compilations gather material from many dates; a heading or date inside the passage is added.
COMPILATIONS = {"Misc": "Miscellaneous writings", "Questions And Answers": "Q&A compilation",
                "Newsletters": "Newsletters", "Interviews": "Interviews",
                "Videos": "Video transcripts"}
# Corpus-specific: single-topic compilation titles of the sample corpus.
TOPICAL_COMPILATIONS = {"Travel Notes", "Vacation Notes"}
TITLE_QA = re.compile(r"^Q&A Of (.+)$")
TITLE_WORKSHOP = re.compile(r"^(?:.+? )?Workshop(?: \+ Q&A)? Of (.+?)(?: \(([^)]+)\))?$")
TITLE_COURSE = re.compile(r"^(?:.+? )?Crash Course Of (.+)$")
PASSAGE_HEADING = re.compile(r"^\s*(?:#{1,6}\s*(.+?)\s*#*|\*\*([^*]{3,80})\*\*)\s*$", re.M)
DAY_FIRST_DATE = re.compile(r"\b(\d{1,2}) (jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.? "
                            r"((?:19|20)?\d\d)\b", re.I)
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August",
               "September", "October", "November", "December"]


def passage_label(text):
    """A heading (markdown heading or a bold line) or else a date found inside a compilation
    passage (pure), else None."""
    text = text or ""
    m = PASSAGE_HEADING.search(text)
    if m:
        heading = " ".join((m.group(1) or m.group(2)).replace("*", "").split()).strip(" :")
        if 3 <= len(heading) <= 80:
            return heading
    m = DATE_WORDS.search(text)
    if m:
        month = MONTH_NAMES[MONTHS.index(m.group(1).lower()[:3])]
        return f"{month} {int(m.group(2))}, {m.group(3)}" if m.group(2) else f"{month} {m.group(3)}"
    m = DAY_FIRST_DATE.search(text)
    if m:
        year = m.group(3) if len(m.group(3)) == 4 else f"20{m.group(3)}"
        return f"{MONTH_NAMES[MONTHS.index(m.group(2).lower()[:3])]} {int(m.group(1))}, {year}"
    return None


def readable_source(title, text=None):
    """A source title as people read it (pure): no ".md", "Q&A Of April 12, 2008" -> "Q&A, April 12,
    2008", "Benefits Workshop + Q&A Of March 9, 2021" -> "Workshop, March 9, 2021"; a
    compilation gets a readable name plus a heading or date found in `text` (its passage)."""
    if not title:
        return "(title unavailable)"
    base = re.sub(r"\.(md|txt)$", "", title.strip(), flags=re.I)
    m = TITLE_QA.match(base)
    if m:
        return f"Q&A, {m.group(1)}"
    m = TITLE_WORKSHOP.match(base)
    if m:
        kind = f" ({m.group(2).replace(' Of ', ' of ')})" if m.group(2) else ""
        return f"Workshop{kind}, {m.group(1)}"
    m = TITLE_COURSE.match(base)
    if m:
        return f"Crash course, {m.group(1)}"
    if base in COMPILATIONS or base in TOPICAL_COMPILATIONS:
        label = passage_label(text) if text else None
        name = COMPILATIONS.get(base, base)
        return f"{name} — {label}" if label else name
    return base


def date_order(a, b):
    """-1 when date a is before b, 1 when after, 0 when equal or not comparable at the shared
    precision, None when either is unknown."""
    if not a or not b:
        return None
    n = min(len(a), len(b))
    return (a[:n] > b[:n]) - (a[:n] < b[:n])


def statement_label(st):
    where = st["source"] + (f" ({st['date']})" if st["date"] and st["date"] not in st["source"] else "")
    scope = st["scope"] + (f": {st['applies_to']}" if st["applies_to"] else "")
    pred = f'"{st["predicate"]}"' if st["predicate"] else "(no predicate given)"
    if st["predicate"] and not st["predicate_verbatim"]:
        pred += " (not verbatim in the passage)"
    return f"{st['hit_id']} | {where} | {scope} | {pred}" + (" | context only" if st["use"] == "context" else "")


def evidence_map(requirements, coverage, selected, decisions, titles):
    """How the kept statements relate, per requirement (pure). Returns (map entries, text for the
    reasoner).

    Built only from the selector's claim structure and the source titles' dates; it adds no
    content. Relationships are checked where Python can check them: an "updates" link whose
    source is older than its target is not supported by the dates, and a "narrows" link from a
    general statement contradicts its own scope. Several general statements with different dates
    and no stated relationship are listed as such, because a later date alone does not make a
    statement an update. Kept hits of a follow-up search count for the requirement their premise
    serves."""
    kept = {h["hit_id"]: h for h in selected}
    parent = {c["requirement_id"]: c.get("for") for c in coverage if c.get("for")}
    cov = {c["requirement_id"]: c for c in coverage if not c.get("for")}
    premise_cov = [c for c in coverage if c.get("for")]

    def statement(i):
        d, h = decisions.get(i) or {}, kept[i]
        title = titles.get(h["source_id"]) or "(title unavailable)"
        return {"hit_id": i, "source": readable_source(title, h["text"]), "date": source_date(title),
                "role": d.get("role"),
                "scope": d.get("scope") or "unclear", "applies_to": d.get("applies_to") or "",
                "predicate": d.get("predicate") or "",
                "predicate_verbatim": bool(d.get("predicate_verbatim")),
                "use": d.get("use") or "answer", "relations": d.get("relations") or []}

    entries, assigned = [], set()
    for r in requirements:
        ids = [i for i in kept if any(c == r["id"] or parent.get(c) == r["id"]
                                      for c in (decisions.get(i) or {}).get("covers", []))]
        for c in [cov.get(r["id"])] + [p for p in premise_cov if p["for"] == r["id"]]:
            ids += [i for i in (c or {}).get("hit_ids", []) if i in kept and i not in ids]
        assigned.update(ids)
        statements = [statement(i) for i in ids]
        by_id = {st["hit_id"]: st for st in statements}
        relations = []
        for st in statements:
            for rel in st["relations"]:
                target = statement(rel["hit_id"]) if rel["hit_id"] in kept else None
                if target is None:
                    continue
                check = "ok"
                if rel["type"] == "updates":
                    order = date_order(st["date"], target["date"])
                    check = ("older_than_target" if order == -1 else "ok" if order == 1
                             else "not_dated")
                elif rel["type"] == "narrows" and st["scope"] == "general":
                    check = "general_scope"
                relations.append({"from": st["hit_id"], "type": rel["type"], "to": rel["hit_id"],
                                  "check": check, "to_predicate": target["predicate"]})
        linked = {frozenset((x["from"], x["to"])) for x in relations}
        general = [st for st in statements if st["scope"] == "general" and st["use"] == "answer"]
        dated = sorted((st for st in general if st["date"]), key=lambda st: st["date"])
        unlinked = [(a["hit_id"], b["hit_id"]) for n, a in enumerate(dated) for b in dated[n + 1:]
                    if date_order(a["date"], b["date"]) and frozenset((a["hit_id"], b["hit_id"])) not in linked]
        predicates = {}
        for st in statements:
            if st["use"] == "answer" and st["predicate"]:
                predicates.setdefault(squash(st["predicate"]), (st["predicate"], []))[1].append(st["hit_id"])
        c = cov.get(r["id"]) or {}
        entries.append({
            "requirement_id": r["id"], "statements": statements, "relations": relations,
            "changed": sorted({x["to"] for x in relations if x["type"] == "updates" and x["check"] == "ok"}),
            "unlinked_dated": unlinked,
            "predicates": [{"predicate": p, "hit_ids": hs} for p, hs in predicates.values()],
            "gap": ({"kind": c["gap"], "missing": c.get("missing", ""), "leads": c.get("leads", [])}
                    if c.get("gap") else None)})
    other = [i for i in kept if i not in assigned]
    return entries, render_map(requirements, entries, [statement(i) for i in other])


def render_map(requirements, entries, other):
    """The evidence map as text for the reasoner (see evidence_map)."""
    reqs = {r["id"]: r for r in requirements}
    out = []
    for e in entries:
        lines = [requirement_line(reqs[e["requirement_id"]]).lstrip("- ")]
        sts = {st["hit_id"]: st for st in e["statements"]}
        if sts:
            lines.append("  Statements:")
            lines += [f"  - {statement_label(st)}" for st in sts.values()]
        else:
            lines.append("  Statements: none kept")
        rel_lines = []
        for x in e["relations"]:
            text = f"{x['from']} {x['type'].replace('_', ' ')} {x['to']}"
            if x["type"] == "qualifies":
                text += (f" only (the statement \"{x['to_predicate']}\"); it does not qualify any "
                         "other statement") if x["to_predicate"] else " only; it does not qualify any other statement"
            if x["check"] == "older_than_target":
                text += f": NOT supported by the dates ({x['from']} is older than {x['to']}); treat the relationship as unresolved"
            elif x["check"] == "not_dated":
                text += ": the dates do not confirm which is later; treat it as an update only if the passage says so"
            elif x["check"] == "general_scope":
                text += f": but {x['from']} is labeled general, so check its scope"
            rel_lines.append(f"  - {text}")
        if rel_lines:
            lines.append("  Relationships:")
            lines += rel_lines
        notes = []
        for old in e["changed"]:
            by = [x["from"] for x in e["relations"] if x["type"] == "updates" and x["to"] == old and x["check"] == "ok"]
            notes.append(f"{old} is changed by the later {', '.join(by)}: the change replaces only what it "
                         f"addresses; what it does not change still comes from {old}.")
        if e["unlinked_dated"]:
            pairs = ", ".join(f"{a} ({sts[a]['date']}) and {b} ({sts[b]['date']})" for a, b in e["unlinked_dated"])
            notes.append(f"General statements from different dates with no stated relationship: {pairs}. "
                         "A later date alone does not make one an update of the other.")
        for scope, label in (("condition", "Condition-specific"), ("stage", "Stage-specific")):
            hits = [f"{st['hit_id']} ({st['applies_to'] or 'unspecified'})" for st in sts.values() if st["scope"] == scope]
            if hits:
                notes.append(f"{label}: {', '.join(hits)}; applies only where the user's stated situation matches.")
        cases = [f"{st['hit_id']} ({st['applies_to'] or 'one case'})" for st in sts.values() if st["scope"] == "individual"]
        if cases:
            notes.append(f"Individual cases: {', '.join(cases)}; examples for that person or case, not general rules.")
        if len(e["predicates"]) > 1:
            notes.append("Different predicates, not interchangeable: "
                         + "; ".join(f"\"{p['predicate']}\" ({', '.join(p['hit_ids'])})" for p in e["predicates"]) + ".")
        conflicts = [f"{x['from']} and {x['to']}" for x in e["relations"] if x["type"] == "contradicts"]
        if conflicts:
            notes.append(f"Possible contradiction to resolve: {'; '.join(conflicts)}.")
        context = [st["hit_id"] for st in sts.values() if st["use"] == "context"]
        if context:
            notes.append(f"Context only (informs interpretation; need not appear in the answer): {', '.join(context)}.")
        if e["gap"]:
            g = e["gap"]
            notes.append(f"Gap ({g['kind']}): {g['missing'] or 'not specified'}"
                         + (f"; leads: {', '.join(g['leads'])}" if g["leads"] else "")
                         + ". Resolve it from the evidence or a targeted follow-up search; never supply a value.")
        if notes:
            lines.append("  Notes:")
            lines += [f"  - {n}" for n in notes]
        out.append("\n".join(lines))
    if other:
        out.append("Other kept passages (no requirement named):\n"
                   + "\n".join(f"  - {statement_label(st)}" for st in other))
    return "\n\n".join(out)


def repair_leads(requirements, coverage):
    """Requirements with a named gap that targeted repair could resolve: a missing quantity or
    component with a lead passage, or on a requirement that needs exact details."""
    reqs = {r["id"]: r for r in requirements}
    leads = []
    for c in coverage:
        r = reqs.get(c["requirement_id"])
        if r is None or not c.get("gap") or c["gap"] not in ("quantity", "component", "referent", "chronology"):
            continue
        if c.get("leads") or r.get("exact") or r["kind"] == "quantity":
            leads.append({"requirement_id": r["id"], "gap": c["gap"], "missing": c.get("missing", ""),
                          "leads": c.get("leads", [])})
    return leads


def check_synthesis(out, entries):
    """Normalize the reasoner's synthesis (how it combined the evidence) and check it against the
    evidence map. Returns (items, issues). Issues are diagnostics for Research Details; the answer
    is never changed."""
    statements = {st["hit_id"]: st for e in entries for st in e["statements"]}
    qualifies = {(x["from"], x["to"]) for e in entries for x in e["relations"] if x["type"] == "qualifies"}
    changed = {old: [x["from"] for x in e["relations"] if x["type"] == "updates" and x["to"] == old and x["check"] == "ok"]
               for e in entries for old in e["changed"]}
    items, issues = [], []
    for item in out.get("synthesis") or []:
        if not isinstance(item, dict) or item.get("treatment") not in TREATMENTS:
            continue
        ids = [i for i in dict.fromkeys(item.get("hit_ids") or []) if i in statements]
        if not ids:
            continue
        target = clean(item.get("qualifies"))
        items.append({"hit_ids": ids, "treatment": item["treatment"],
                      "applies_to_user": item.get("applies_to_user") if item.get("applies_to_user") in APPLIES else "unknown",
                      "qualifies": target if target in statements else ""})
    treated = {i: it["treatment"] for it in items for i in it["hit_ids"]}
    for it in items:
        for i in it["hit_ids"]:
            st = statements[i]
            if it["treatment"] == "exception":
                if not it["qualifies"]:
                    issues.append({"issue": "exception_without_target", "hit_id": i})
                elif qualifies and any(src == i for src, _ in qualifies) and (i, it["qualifies"]) not in qualifies:
                    issues.append({"issue": "exception_target_differs", "hit_id": i,
                                   "applied_to": it["qualifies"],
                                   "labeled": sorted(t for src, t in qualifies if src == i)})
            if it["treatment"] == "base" and st["scope"] == "individual":
                issues.append({"issue": "individual_case_used_as_base", "hit_id": i})
            if it["treatment"] == "base" and st["scope"] in ("condition", "stage") and it["applies_to_user"] != "yes":
                issues.append({"issue": "specific_guidance_used_as_general_base", "hit_id": i})
            if it["treatment"] == "base" and i in changed and all(
                    treated.get(u) in (None, "not_used", "context_only") for u in changed[i]):
                issues.append({"issue": "earlier_version_used_without_its_update", "hit_id": i,
                               "updated_by": changed[i]})
    return items, issues


THRESHOLD_PHRASES = re.compile(
    r"\b(?:only (?:in|when \w+ in) excess|in excess|excessive amounts?|excess amounts?|normal amounts?|"
    r"moderate amounts?|in moderation|safe (?:amounts?|levels?|quantities)|reasonable amounts?)\b", re.I)
NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
                "seven": "7", "eight": "8", "nine": "9", "ten": "10", "twelve": "12", "half": "1/2",
                "a half": "1/2", "a quarter": "1/4", "a third": "1/3"}
QUANTITY = re.compile(
    r"\b(\d+(?:[.,/]\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|twelve|a half|half|"
    r"a quarter|a third)\s*(?:-\s*|of an?\s+|an?\s+)?"
    r"(cups?|tablespoons?|tbsps?|teaspoons?|tsps?|ounces?|oz|grams?|g|ml|milliliters?|millilitres?|"
    r"liters?|litres?|quarts?|pints?|pounds?|lbs?|glass(?:es)?|handfuls?|spoonfuls?|parts?)\b", re.I)
MECHANISM_VERBS = {"strip": r"strip(?:s|ped|ping)?\b", "leach": r"leach(?:es|ed|ing)?\b",
                   "deplete": r"deplet(?:e|es|ed|ing)\b", "drain": r"drain(?:s|ed|ing)?\b",
                   "rob": r"rob(?:s|bed|bing)?\b", "extract": r"extract(?:s|ed|ing)?\b",
                   "remove": r"remov(?:e|es|ed|ing)\b", "pull": r"pull(?:s|ed|ing)?\b",
                   "bind": r"(?:bind(?:s|ing)?|bound)\b", "draw": r"(?:draw(?:s|ing|n)?|drew) out\b"}


def threshold_phrases(answer, evidence):
    """Threshold wording ("only in excess", "normal amounts", ...) in the answer that the evidence
    never uses: a diagnostic for a manufactured safe/unsafe boundary."""
    source = squash(evidence)
    return list(dict.fromkeys(m.group(0).lower() for m in THRESHOLD_PHRASES.finditer(answer)
                              if squash(m.group(0)) not in source))


# Temperatures and durations are checked like amounts. °F, "degrees" and "degrees
# Fahrenheit" are one unit; a bare "degrees" counts as Fahrenheit (the sources' unit).
TEMPERATURE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:°|º|˚|degrees?\b|deg\b)\s*"
                         r"(f\b|c\b|fahrenheit|celsius|centigrade)?"
                         r"|(?<![\w.])(\d+(?:\.\d+)?)\s?([FC])\b", re.I)
# Recipe shorthand in the sources: "1 T of sugar to 1 t of salt" (case matters).
SPOON_ABBREVIATION = re.compile(r"(?<![\w.])(\d+(?:[./]\d+)?)\s?(T|t)\b(?=\.?\s+(?:of\b|[a-z]))")
DURATION = re.compile(
    r"\b(\d+(?:[.,/]\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|twelve|a half|half)"
    r"\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)\b", re.I)
# An answer number labeled as a conversion ("93°F (about 34°C)", "converted", "equivalent")
# restates a sourced number in another unit; it is not an unverified number.
CONVERSION_LABEL = re.compile(r"conver\w*|equivalent|equals|≈|~|\bin (?:metric|celsius|"
                              r"fahrenheit|grams|ml)\b|\bi\.e\.", re.I)


def number_key(n):
    n = NUMBER_WORDS.get(n.lower(), n.replace(",", "."))
    try:
        return n if "/" in n else f"{float(n):g}"
    except ValueError:
        return n


def quantity_keys(text):
    """{(number, unit): (matched text, start)} for the amounts, temperatures and durations."""
    keys = {}
    for m in QUANTITY.finditer(text):
        unit = m.group(2).lower().rstrip("s").replace("glasse", "glass")
        unit = {"tbsp": "tablespoon", "tsp": "teaspoon", "oz": "ounce", "lb": "pound", "g": "gram",
                "millilitre": "ml", "milliliter": "ml", "litre": "liter"}.get(unit, unit)
        keys.setdefault((number_key(m.group(1)), unit), (m.group(0), m.start()))
    for m in SPOON_ABBREVIATION.finditer(text):
        unit = "tablespoon" if m.group(2) == "T" else "teaspoon"
        keys.setdefault((number_key(m.group(1)), unit), (m.group(0), m.start()))
    for m in TEMPERATURE.finditer(text):
        n, scale = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        unit = "degC" if (scale or "").lower()[:1] == "c" else "degF"
        keys.setdefault((number_key(n), unit), (m.group(0).strip(), m.start()))
    for m in DURATION.finditer(text):
        unit = m.group(2).lower().rstrip("s")
        unit = {"sec": "second", "min": "minute", "hr": "hour"}.get(unit, unit)
        keys.setdefault((number_key(m.group(1)), unit), (m.group(0), m.start()))
    return keys


def labeled_conversion(answer, start):
    """Whether the number at `start` is labeled as a conversion: a conversion word just before
    it, or it opens a parenthesis right after another number ("93°F (about 34°C)")."""
    before = answer[max(0, start - 30):start]
    return bool(CONVERSION_LABEL.search(before) or re.search(
        r"\d[^\d()]{0,12}\(\s*(?:about|approx\w*|roughly|around)?\s*$", before, re.I))


def ungrounded_quantities(answer, evidence):
    """Numbers with a unit in the answer (amounts such as "2 cups" or "half a glass",
    temperatures, durations) that the evidence never states, with number words and unit
    spellings normalized: a diagnostic for invented numbers ("unverified numbers" in Research
    Details). A number the answer labels as a conversion is exempt."""
    known = quantity_keys(check_text(evidence))
    return [text for key, (text, start) in quantity_keys(answer).items()
            if key not in known and not labeled_conversion(answer, start)]


QUOTED = re.compile(r'"([^"\n]*)"|“([^”\n]*)”')
# Pipeline labels and page markers that the evidence text interleaves with source text.
EVIDENCE_MARKERS = re.compile(r"^\s*(?:PAGE \d+|CONTINUATION:|PRECEDING:|CONTEXT:|={3,})\s*$",
                              re.M)
# A degree sign that the book scans turned into "0" ("above 105 0 Fahrenheit").
OCR_DEGREE = re.compile(r"(\d) [0o] (?=fahrenheit\b|f\b)", re.I)


def check_text(evidence):
    """Evidence as the grounding checks read it: pipeline markers removed, OCR degrees fixed."""
    return OCR_DEGREE.sub(r"\1° ", EVIDENCE_MARKERS.sub(" ", evidence))
QUOTE_MIN_WORDS = 4
QUOTE_CHARS = str.maketrans({"'": None, "‘": None, "’": None, "‛": None})


def grounding_norm(text):
    """Text for verbatim quote checks: lowercase, apostrophes dropped, quote marks,
    punctuation, markdown and runs of whitespace collapsed to single spaces."""
    text = unicodedata.normalize("NFKC", str(text or ""))  # "½" -> "1⁄2", like "1/2"
    return " ".join(re.sub(r"[^\w%]+", " ", text.lower().translate(QUOTE_CHARS)).split())


def quote_segments(quote):
    """The checkable pieces of one quotation: split at ellipses and [editorial insertions],
    pieces of at least QUOTE_MIN_WORDS words."""
    parts = re.split(r"\.\.\.|…|\[[^\]]*\]", quote)
    return [p.strip() for p in parts if len(grounding_norm(p).split()) >= QUOTE_MIN_WORDS]


def ungrounded_quotes(answer, evidence):
    """Grounding check: quoted strings in the answer (in double quotes, or a blockquote
    line without quote marks, its trailing attribution removed) of at least QUOTE_MIN_WORDS
    words that do not appear verbatim in the evidence after normalizing whitespace, quote marks
    and punctuation. Each miss is a grounding failure. It never changes or blocks the answer."""
    text = check_text(answer.replace("*", "").replace("_", " "))  # an OCR "105 0 F" quoted as is
    source = f" {grounding_norm(check_text(evidence))} "
    quotes = [m.group(1) or m.group(2) for m in QUOTED.finditer(text)]
    for line in text.splitlines():
        body = line.strip()
        if body.startswith(">") and '"' not in body and "“" not in body:
            body = re.sub(r"\s*(?:\([^)]*\)|[—–]\s*[^—–]*)\s*$", "", body.lstrip("> "))
            if body:
                quotes.append(body)
    misses = []
    for q in quotes:
        for seg in quote_segments(q):
            words = grounding_norm(seg).split()
            # The first or last word may be adapted to the sentence ("[It'll →] will just stink").
            tries = [words] + ([words[1:], words[:-1], words[1:-1]]
                               if len(words) > QUOTE_MIN_WORDS + 1 else [])
            if (not any(f" {' '.join(w)} " in source for w in tries if len(w) >= QUOTE_MIN_WORDS)
                    and seg not in misses):
                misses.append(seg)
    return misses


# Citations: the answer cites evidence ids in brackets, "[h15]", "[s2]", "[h3][h7]" or
# "[h3, h7]"; the UI numbers them in order of first appearance.
CITATION = re.compile(r"\[((?:[hs]\d+)(?:\s*[,;]\s*[hs]\d+)*)\]")
BLOCK_LABEL = re.compile(r"^(?:PASSAGE|PRECEDING \(exact source text immediately before|SECONDARY) "
                         r"\[([hs]\d+)\]", re.M)


def cited_ids(text):
    """Every id the text cites, in order of first appearance (pure)."""
    ids = [i.strip() for m in CITATION.finditer(text or "") for i in re.split(r"[,;]", m.group(1))]
    return list(dict.fromkeys(ids))


def evidence_blocks(*texts):
    """{id: the text of the block that labels it} for evidence or secondary text (pure): a
    CONTINUATION, PRECEDING or CONTEXT belongs to the passages of its block."""
    blocks = {}
    for text in texts:
        for block in (text or "").split("\n\n=====\n\n"):
            for i in BLOCK_LABEL.findall(block):
                blocks.setdefault(i, block)
    return blocks


def answer_body(answer):
    """The answer after its first section (the Short answer), where citations are required."""
    lines = (answer or "").strip().splitlines()
    for n, line in enumerate(lines[1:], 1):
        if HEADING.match(line):
            return "\n".join(lines[n:])
    return "" if lines and HEADING.match(lines[0]) else answer or ""


def citation_checks(answer, evidence, secondary=None):
    """Citation grounding (pure; never changes the answer): cited ids that label no
    evidence block ("unknown_ids"); quotations whose words are in the evidence but not in any
    block they cite ("misattributed_quotes", with the ids whose blocks do contain them); body
    quotations with no citation in their sentence
    ("uncited_quotes"); and whether the Short answer carries citations."""
    blocks = evidence_blocks(evidence, secondary)
    norm = {i: f" {grounding_norm(check_text(b))} " for i, b in blocks.items()}
    text = check_text((answer or "").replace("*", "").replace("_", " "))
    body_start = len(text) - len(answer_body(text))
    unknown = [i for i in cited_ids(answer) if i not in blocks]
    misattributed, uncited = [], []
    quotes = list(QUOTED.finditer(text))
    for n, m in enumerate(quotes):
        quote = m.group(1) or m.group(2)
        segments = quote_segments(quote)
        if not segments:
            continue
        # A quotation's citations follow it within its sentence (or right after the sentence
        # ends): those before the next quotation, else the sentence's ("A" and "B" [h3]).
        line_end = text.find("\n", m.end())
        tail = text[m.end():line_end if line_end != -1 else len(text)]
        stop = re.search(r"[.!?](?=\s|$)", tail)
        if stop:
            after = re.match(r"\s*(?:\[[^\]\n]*\]\s*)*", tail[stop.end():]).group(0)
            tail = tail[:stop.end()] + after
        nxt = quotes[n + 1].start() - m.end() if n + 1 < len(quotes) else len(tail)
        cited = cited_ids(tail[:nxt]) or cited_ids(tail)
        words = [f" {grounding_norm(s)} " for s in segments]

        def contains(i):
            return all(w in norm.get(i, "") for w in words)
        if not cited:
            if m.start() >= body_start:
                uncited.append(quote)
            continue
        known = [i for i in cited if i in blocks]
        if known and not any(contains(i) for i in known):
            found = [i for i in blocks if contains(i)]
            if found:  # a quote found nowhere is already a grounding failure
                misattributed.append({"quote": quote, "cited": cited, "found_in": found})
    return {"unknown_ids": unknown, "misattributed_quotes": misattributed,
            "uncited_quotes": uncited,
            "short_answer_cited": bool(CITATION.search(text[:body_start]))}


def build_citations(answer, sources, secondary=None):
    """The answer's citations for display (pure), numbered in order of first appearance: each
    with its id, kind ("passage" or "community"), readable name, date and excerpt. Ids that
    label no evidence are left out (citation_checks flags them)."""
    by_hit = {}
    for src in sources or []:
        for e in src.get("excerpts") or []:
            for p in e.get("passages") or []:
                by_hit[p["hit_id"]] = {
                    "kind": "passage", "source_id": src["source_id"], "title": src.get("title"),
                    "name": e.get("label") or readable_source(src.get("title")),
                    "date": source_date(src.get("title")), "text": p["text"],
                    "preceding": p.get("preceding"), "continuation": e.get("continuation"),
                    "context": e.get("context")}
    for i, block in evidence_blocks(secondary).items():
        origin = re.search(r"^SECONDARY SOURCE \([^)]*\): (.*)$", block, re.M)
        claim = re.search(r"^SECONDARY CLAIM:\n(.*?)(?=^(?:RECORD|PLATFORMS|EARLIER VERIFICATION|VERIFICATION|"
                          r"CITED PRIMARY|SECONDARY WORDING)\b|\Z)", block, re.M | re.S)
        wording = re.search(r"^SECONDARY WORDING \(excerpt\):\n(.*)", block, re.M | re.S)
        by_hit[i] = {"kind": "community", "source_id": None, "title": None,
                     "name": "Community — " + (origin.group(1).strip() if origin else "record"),
                     "date": None, "text": (claim.group(1).strip() if claim else block),
                     "preceding": None, "continuation": None,
                     "context": wording.group(1).strip() if wording else None}
    out = []
    for i in cited_ids(answer):
        if i in by_hit:
            out.append({"n": len(out) + 1, "id": i, **by_hit[i]})
    return out


def unsourced_mechanism_verbs(answer, evidence):
    """Mechanism verbs the answer uses that no evidence passage uses (for example "strips" where
    the source says "pulls"): a diagnostic for normalized or strengthened predicates."""
    return [verb for verb, pattern in MECHANISM_VERBS.items()
            if re.search(r"\b" + pattern, answer, re.I) and not re.search(r"\b" + pattern, evidence, re.I)]


REPAIR_QUERY_WORDS = 12  # a follow-up NotebookLM query is a short search string
QUERY_STOPWORDS = frozenset("""
a an the any some or and but with without of for to in on at by from into about as is are was were
be been being it its this that these those there their them they he his she her you your i me my we
our what which who whom how when where why whether if does do did done should would could can may
might must will shall each every all both either neither much many more most other such only own
same so than too very just also well give show tell list full complete exact exactly please need
needed need's said says say saying author author's instructor created made make recommended
recommend recommends according per not no one ones missing given stated specified unknown
unclear evidence detail details
""".split())
GAP_TERMS = {"quantity": "amounts", "component": "ingredients"}


def search_terms(text, extra=(), limit=REPAIR_QUERY_WORDS):
    """The key terms of a premise or requirement as a short search string (pure): content words
    in order, stopwords and repeats dropped, plus `extra` terms, at most `limit` words."""
    words = []
    for w in re.findall(r"[\w'’-]+", str(text or "").lower()):
        w = w.strip("'’-")
        if w and w not in QUERY_STOPWORDS and not w.isdigit() and w not in words:
            words.append(w)
    for w in extra:
        if w not in words:
            words.append(w)
    return " ".join(words[:limit])


def cap_query(query, limit=REPAIR_QUERY_WORDS):
    return " ".join(str(query or "").split()[:limit])


def user_text(question, standalone=None, previous=None, related=None):
    """What the user wrote (pure): the question and its standalone rewrite, plus, for a turn
    that builds on an earlier one, the thread's earlier questions. Quoted strings and numbers
    found here are the user's own, not claims to ground in the evidence."""
    texts = [question, standalone]
    if previous and related is not None:
        texts += [t.get("question") for t in previous.get("turns") or []]
    return "\n\n".join(dict.fromkeys(t for t in texts if t))


def forced_repair(requirements, coverage, answer, evidence, searches):
    """A targeted repair round Python requests when the reasoner answered anyway although a
    material exact detail is missing but retrievable: a quantity or component gap on a
    requirement that needs it, with a lead passage pointing at it or with amounts in the answer
    that the evidence does not state. [] otherwise. Each item's query is a short search string:
    the requirement's key terms plus the kind of detail missing (amounts, ingredients). The
    selector's gap note is the premise, never part of the query."""
    invented = ungrounded_quantities(answer, evidence)
    reqs = {r["id"]: r for r in requirements}
    request, seen = [], {q.lower() for q in searches}
    for lead in repair_leads(requirements, coverage):
        r = reqs[lead["requirement_id"]]
        if lead["gap"] not in ("quantity", "component"):
            continue
        material = r.get("exact") or r["kind"] in ("quantity", "procedure", "composition")
        if not material or not (lead["leads"] or invented):
            continue
        query = search_terms(r["text"], [GAP_TERMS[lead["gap"]]])
        if not query or query.lower() in seen or repeats_query(query, list(searches) + [
                x["search"] for x in request]):
            continue
        seen.add(query.lower())
        request.append({"requirement_id": r["id"],
                        "premise": clean(f"exact detail not in the evidence: {lead['missing'] or lead['gap']}"),
                        "search": query})
    return request[:REPAIR_MAX_SEARCHES]


COVERAGE_RANK = {"covered": 3, "partial": 2, "unassessed": 1, "missing": 0}


def final_coverage(coverage, repaired):
    """The run's final coverage per requirement (pure): the selector's round-1 coverage, downgraded
    where the repair round judged a premise serving the requirement partial or missing (the worst
    premise counts). Repair never upgrades, and a failed follow-up search judges nothing. A
    downgraded requirement cites no hits; the hits it cited become leads. Returns (coverage,
    downgrades [{requirement_id, from, to, source: "repair", premise}])."""
    if not repaired or repaired.get("error"):
        return [dict(c) for c in coverage], []
    verdict = {}
    for p in repaired.get("coverage") or []:
        if p.get("for") and p.get("status") in ("partial", "missing"):
            cur = verdict.get(p["for"])
            if cur is None or COVERAGE_RANK[p["status"]] < COVERAGE_RANK[cur["status"]]:
                verdict[p["for"]] = p
    final, downgrades = [], []
    for c in coverage:
        p = verdict.get(c["requirement_id"])
        if p is None or COVERAGE_RANK[p["status"]] >= COVERAGE_RANK.get(c["status"], 1):
            final.append(dict(c))
            continue
        final.append({"requirement_id": c["requirement_id"], "status": p["status"], "hit_ids": [],
                      "missing": p.get("missing") or "judged not covered by the follow-up search",
                      "gap": p.get("gap") or "other",
                      "leads": list(dict.fromkeys(c.get("leads", []) + c["hit_ids"])),
                      "downgraded_from": c["status"]})
        downgrades.append({"requirement_id": c["requirement_id"], "from": c["status"],
                           "to": p["status"], "source": "repair", "premise": p["requirement_id"]})
    return final, downgrades


def coverage_note(entry):
    if not entry:
        return "unassessed"
    return entry["status"] + (f"; missing: {entry['missing']}" if entry.get("missing") else "")


def requested_part_line(x):
    """One requested part as the reasoner reads it (pure)."""
    if x.get("per_component_of"):
        return f"each component of {x['per_component_of']}: {x['attribute']}"
    return f"{x['part']}: {x['attribute']}"


def reasoner_prompt(question, depth, evidence, requirements, coverage, repaired=None,
                    evidence_map=None, secondary=None, trimmed=False, requested_parts=(),
                    told=None, referenced=None, findings=None, earlier=None,
                    from_turn=False):
    """The reasoner's input (pure): the question, its answer requirements with the selector's
    coverage, the repair state, the evidence map (how the kept statements relate, see
    evidence_map), then the evidence. The first call (repaired=None) may request a repair round
    and is told which gaps targeted repair could resolve; the final call after it lists each
    premise searched and what the follow-up evidence covered, and must answer.

    ask path (findings is NotebookLM's reply): no requirements, the reply as a lead before
    the evidence, and the repair round is one follow-up NotebookLM ask with fact questions.

    from_turn (ANSWER_FROM_TURN): the evidence is what the related turn retrieved; the
    repair round is one fact question, only for a missing premise that could change the
    conclusion, and none once the planner already asked one."""
    cov = {c["requirement_id"]: c for c in coverage}
    needs = "\n".join(f"{requirement_line(r)} (coverage: {coverage_note(cov.get(r['id']))})"
                      for r in requirements)
    if from_turn and repaired is None:
        repair = ("EVIDENCE REPAIR: answer from the evidence below. It is what the earlier turn "
                  "retrieved (cited and uncited passages); no new search was run, because the "
                  "question is an inference from it. Request one follow-up NotebookLM ask only "
                  "when one specific premise the evidence does not settle could change the "
                  "conclusion; then give exactly one query, the fact question for that premise. "
                  "Never ask merely to be more certain."
                  + (" A premise was already asked (see the FACT QUESTION heading), so answer "
                     "now." if findings else ""))
    elif findings is not None and repaired is None:
        repair = ("EVIDENCE REPAIR: available. You may request one follow-up NotebookLM ask "
                  "instead of answering: its queries are specific fact questions, not already "
                  "among the FACT QUESTION headings.")
    elif findings is not None:
        asked = "\n".join(f"- {q}" for q in repaired.get("questions") or [])
        outcome = (f"The follow-up ask failed: {repaired['error']}" if repaired.get("error")
                   else f"{repaired['selected']} new passage(s) from it were added to the evidence.")
        repair = ("EVIDENCE REPAIR: done. A follow-up NotebookLM ask was sent"
                  + (f" with these fact questions:\n{asked}\n" if asked else ".\n")
                  + f"{outcome} No more asks are possible; answer now. A material premise that "
                    "is still unresolved gets one short neutral line, without describing the "
                    "research.")
    elif repaired is None:
        repair = ("EVIDENCE REPAIR: available. You may request one round of follow-up searches "
                  "instead of answering.")
        leads = repair_leads(requirements, coverage)
        if leads:
            repair += ("\nTargeted repair candidates (exact details the selector found missing; "
                       "search for them rather than supplying a value):\n" + "\n".join(
                           f"- {x['requirement_id']}: {x['gap']} missing"
                           + (f" ({x['missing']})" if x["missing"] else "")
                           + (f"; leads: {', '.join(x['leads'])}" if x["leads"] else "")
                           for x in leads))
    else:
        found = {c["requirement_id"]: c for c in repaired.get("coverage") or []}
        searched = "\n".join(
            f"{requirement_line(p)} (search: {p['search']}; coverage: "
            f"{coverage_note(found.get(p['id']))})" for p in repaired["premises"])
        outcome = (f"The follow-up search failed: {repaired['error']}" if repaired.get("error")
                   else f"{repaired['selected']} new passage(s) from it were added to the evidence.")
        repair = ("EVIDENCE REPAIR: done. Follow-up searches were run for these missing premises:\n"
                  f"{searched}\n{outcome} No more searches are possible; answer now, and state "
                  "any material requirement that is still unresolved.")
    emap = (f"EVIDENCE MAP (the selector's labels for each kept passage, checked against source "
            f"dates; verify them against the evidence):\n\n{evidence_map}\n\n" if evidence_map else "")
    if coverage and all(c["status"] == "unassessed" for c in coverage) and not evidence_map:
        emap = ("EVIDENCE SELECTION: none. The retrieved passages fit the evidence budget, so all "
                "of them are included unjudged, in retrieval order grouped by source. Judge "
                "relevance yourself and silently ignore what does not bear on the question.\n\n")
        if trimmed:
            emap = ("EVIDENCE SELECTION: none. The retrieved passages exceeded the evidence "
                    "budget, so the lowest-ranked ones were dropped; the rest are included "
                    "unjudged, in retrieval order grouped by source. Judge relevance yourself and "
                    "silently ignore what does not bear on the question.\n\n")
    parts = "\n".join(f"- {requested_part_line(x)}" for x in requested_parts or ())
    parts = (f"REQUESTED PARTS (answer every one):\n{parts}\n\n"
             if parts else "")
    told = (f"ALREADY TOLD THE USER (the earlier turn this question builds on; do not repeat "
            f"it):\n{told}\n\n" if told else "")
    if earlier:
        told += ("EARLIER ANSWER (a previous run's answer to this same question: a draft to "
                 "re-check against the evidence below, not evidence; your answer replaces it "
                 f"entirely):\n{earlier}\n\n")
    if referenced:
        told += ("REFERENCED RECIPES (sub-recipes that retrieved recipes name as ingredients, "
                 f"searched separately; the passages found for each):\n{referenced}\n\n")
    if findings is not None:
        emap = ("EVIDENCE SELECTION: exact source text grouped by fact question: each FACT "
                "QUESTION heading (a precise question asked of the sources for this user "
                "question) is followed by the passages cited for it; passages from research "
                "memory and the earlier turn follow under OTHER PASSAGES, and community material "
                "last. Judge relevance yourself and silently ignore what does not bear on the "
                "question.\n\n")
        if from_turn:
            emap = ("EVIDENCE SELECTION: the passages the earlier turn retrieved (the ones its "
                    "answer cited first, then the uncited ones), plus, under its FACT QUESTION "
                    "heading, any passage asked for one missing premise; community material "
                    "last. Judge relevance yourself and silently ignore what "
                    "does not bear on the question.\n\n")
        if findings or not from_turn:
            told += ("NotebookLM findings (lead only; verify against the passages; never cite "
                     "it as a source). Its short reply to each fact question, with its citation "
                     f"marks replaced by the ids of the passages they cite:\n"
                     f"{findings or '(none)'}\n\n")
    needs = ("ANSWER REQUIREMENTS (from the question, before retrieval; coverage is the evidence "
             f"selector's assessment, which you must verify):\n{needs}\n\n" if requirements else "")
    return (f"USER QUESTION:\n{question}\n\nRESEARCH DEPTH: {depth}\n\n"
            f"{needs}{parts}{repair}\n\n{emap}"
            f"{told}"
            f"SOURCE EVIDENCE:\n\n{evidence}"
            + (f"\n\nSECONDARY EVIDENCE (community material from research memory; not primary "
               f"evidence):\n\n{secondary}" if secondary else ""))


def reason(run, stage, question, depth, evidence, requirements, coverage, repaired=None,
           evidence_map=None, secondary=None, stop_when=None, requested_parts=(), told=None,
           referenced=None, findings=None):
    """One reasoner call (see reasoner_prompt) in the plain-text protocol (see
    ReasonerText): no schema, so no forced structured-output turn; only the answer text streams.

    On the first call (repaired=None), stop_when(parsed control lines) returning True ends the
    call before any answer text (DECISION: search, or a coverage follow-up). The second call
    must answer: its DECISION line is ignored. Returns the fields the pipeline reads, in the
    shape the structured reasoner had: decision, answer, research_request (from the DECISION
    queries), synthesis (from META; [] when META is missing or invalid), plus coverage (parsed
    COVERAGE entries), coverage_line, queries, meta_status, turns and stopped."""
    prompt = reasoner_prompt(question, depth, evidence, requirements, coverage, repaired,
                             evidence_map, secondary, trimmed=trimmed(run),
                             requested_parts=requested_parts, told=told, referenced=referenced,
                             findings=findings,
                             earlier=None if run.related else prior_block(run.prior),
                             from_turn=run.reuse == ANSWER_FROM_TURN)
    run.save(f"{stage}.input.txt", prompt)

    def on_answer(text, reset):
        if run.first_answer_seconds is None:
            run.first_answer_seconds = round(time.monotonic() - run.started, 1)
            run.log("answer_first_text", stage=stage, seconds=run.first_answer_seconds)
        run.reading(False)
        run.writing(True)
        run.emit("answer_delta", stage=stage, text=text, reset=reset)

    def on_control(reader):
        if repaired is not None or stop_when is None:
            return False
        return bool(stop_when(reader.result()))
    run.reading(True)
    try:
        parsed = claude(run, stage, REASONER[depth], REASONER_SYSTEM, prompt,
                        text_reader=ReasonerText(on_answer, on_control))
    finally:
        run.reading(False)  # a call that stopped to search, or failed, wrote no text
    if repaired is not None:
        parsed["decision"] = "answer"  # the second call must answer
    meta = parsed["meta"] or {}
    out = {"decision": parsed["decision"], "answer": parsed["answer"],
           "research_request": [{"requirement_id": "", "premise": q, "search": q}
                                for q in parsed["queries"]] if parsed["decision"] == "search" else [],
           "synthesis": meta.get("synthesis") if isinstance(meta.get("synthesis"), list) else [],
           **{k: parsed[k] for k in ("coverage", "coverage_line", "decision_line", "queries",
                                     "meta_status", "turns", "stopped")}}
    run.log("reasoner_protocol", stage=stage, coverage_line=parsed["coverage_line"],
            decision_line=parsed["decision_line"], meta_status=parsed["meta_status"],
            turns=parsed["turns"], stopped=parsed["stopped"])
    run.save(f"{stage}.json", json.dumps(out, indent=2, ensure_ascii=False))
    return out


def coverage_check(run, question, evidence, requested_parts, secondary=None):
    """Coverage pre-check, before the answer pass: COVERAGE_MODEL reads the evidence and
    writes only a COVERAGE line (one entry per requested part, or per component of a
    per-component part; "yes" only when a passage states that attribute) and a DECISION line
    with queries for the "no" entries. Returns {"coverage", "queries", "coverage_line",
    "decision_line"} (the shape coverage_followup reads), or None when the check failed or its
    reply could not be parsed; the pipeline then goes straight to the answer pass."""
    parts = "\n".join(f"- {requested_part_line(x)}" for x in requested_parts)
    prompt = (f"USER QUESTION:\n{question}\n\nREQUESTED PARTS:\n{parts}\n\n"
              f"SOURCE EVIDENCE:\n\n{evidence}"
              + (f"\n\nSECONDARY EVIDENCE (community material):\n\n{secondary}" if secondary else ""))
    run.save("coverage-1.input.txt", prompt)
    # Gemini Flash first; Haiku only when Gemini fails.
    def parse(t):
        line = next((x.strip() for x in str(t).splitlines() if x.strip().startswith("COVERAGE:")),
                    None)
        return (t, None) if parse_coverage(line) else (None, "unusable_output")
    text = gemini_task(run, "coverage-1", "coverage", f"{COVERAGE_SYSTEM}\n\n{prompt}", parse)
    if text is None:
        try:
            text = claude(run, "coverage-1", COVERAGE, COVERAGE_SYSTEM, prompt)
        except ResearchError as e:
            run.log("coverage_check_failed", error=str(e))
            return None
    lines = [x.strip() for x in str(text).splitlines()]
    cov_line = next((x for x in lines if x.startswith("COVERAGE:")), None)
    dec_line = next((x for x in lines if x.startswith("DECISION:")), None)
    coverage = parse_coverage(cov_line)
    if not coverage:
        run.log("coverage_check_failed", error="unparsable reply", reply=str(text)[:500])
        return None
    _, queries = parse_decision(dec_line)
    return {"coverage": coverage, "queries": queries, "coverage_line": cov_line,
            "decision_line": dec_line}


def second_call_needed(decision, request):
    """Whether the repair round and a final answer call run (pure): there are follow-up
    searches, or the first call (None after a coverage follow-up) stopped or asked to search
    without writing an answer. The latter also when every query it proposed was rejected
    as a repeat, so the run still ends with an answer."""
    if request:
        return True
    return bool(decision) and (decision["stopped"] or (
        decision["decision"] == "search" and not (decision.get("answer") or "").strip()))


def repair_request(run, decision, searches, requirements):
    """The follow-up searches a first reasoner call asked for: at most REPAIR_MAX_SEARCHES, each
    with a premise, a query not already searched, and the requirement it serves ("" when none
    or unknown). [] means the call answered: a "decision" of "answer" with an answer ignores any
    request it also filled."""
    if decision.get("decision") == "answer" and (decision.get("answer") or "").strip():
        if decision.get("research_request"):
            run.log("repair_request_dropped", reason="decision was answer",
                    count=len(decision["research_request"]))
        return []
    valid = {r["id"] for r in requirements}
    request, seen, earlier = [], {q.lower() for q in searches}, list(searches)
    for item in decision.get("research_request") or []:
        if not isinstance(item, dict):
            continue
        # The reasoner's proposed search, else the premise's key terms; a short search string.
        premise = clean(item.get("premise"))
        query = cap_query(clean(item.get("search"))) or search_terms(premise)
        rid = clean(item.get("requirement_id"))
        repeat = repeats_query(query, earlier) if query else None
        if repeat:  # a reworded repeat of an earlier query: the next proposed one instead
            run.log("repair_query_rejected", query=query, repeats=repeat)
            continue
        if premise and query and query.lower() not in seen:
            seen.add(query.lower())
            earlier.append(query)
            request.append({"requirement_id": rid if rid in valid else "", "premise": premise,
                            "search": query})
    dropped = len(decision.get("research_request") or []) - len(request)
    if dropped:
        run.log("repair_request_dropped", reason="empty or already searched", count=dropped)
    return request[:REPAIR_MAX_SEARCHES]


def repair(run, question, depth, request, candidates, selected, context_ids, titles):
    """The single repair round: search the requested premises, drop hits already known, complete
    and select only the new hits against the premises, and rebuild the evidence with any newly
    selected ones.

    Returns (evidence text or None when nothing new was selected, stats, sources, info). info
    has the premises as requirement-like entries (p1, p2, each "for" the requirement it serves)
    and their coverage by the follow-up evidence; picked is the list of newly selected hits. A
    NotebookLM failure here is not fatal: info["error"] is set and the evidence is unchanged."""
    premises = [{"id": f"p{i}", "kind": "premise", "for": r["requirement_id"],
                 "text": r["premise"], "search": r["search"]} for i, r in enumerate(request, 1)]
    info = {"premises": premises, "searches": [r["search"] for r in request], "raw_hits": 0,
            "candidates": 0, "selected": 0, "context_hits": 0, "coverage": [], "error": None}

    def uncovered(why):
        return [{"requirement_id": p["id"], "for": p["for"], "status": "missing", "hit_ids": [],
                 "missing": why} for p in premises]

    try:
        raw, fresh = search(run, info["searches"], round_n=2, known=candidates)
    except ResearchError as e:
        info.update(error=str(e), coverage=uncovered("follow-up search failed"))
        return None, None, None, info, []
    collapse_duplicates(run, fresh, titles, round_n=2, known=candidates)
    info.update(raw_hits=len(raw), candidates=len(fresh))
    prior = sum(hit_chars(h) for h in selected)  # round-1 evidence the new hits join
    if not fresh:
        info["coverage"] = uncovered("follow-up search found no new passages")
        return None, None, None, info, []
    if any(c["needs_continuation"] or fragment_start(c["text"]) for c in fresh):
        hydrate(run, fresh, titles, round_n=2, budget=EVIDENCE_BUDGET[depth],
                prior_chars=prior)
    capture_primary(run, raw, fresh, titles, 2)
    picked, picked_context, coverage = select(run, question, depth, fresh, titles, premises,
                                              round_n=2, follow_up=True, prior_chars=prior)
    info.update(selected=len(picked), context_hits=len(picked_context), coverage=coverage)
    if not picked:
        return None, None, None, info, []
    evidence, ev, sources = build_evidence(run, selected + picked, context_ids + picked_context,
                                           depth, titles, round_n=2)
    return evidence, ev, sources, info, picked


def stage_usage(usage):
    """Per Claude stage (pure): configured model, input (fresh + cache read + cache write) and
    output tokens (output includes thinking), wall seconds. Plan usage per run."""
    rows = []
    for u in usage:
        if u.get("provider") != "claude":
            continue
        t = u.get("usage") or {}
        rows.append({"stage": u["stage"], "model": u.get("model") or ",".join(u.get("models") or []),
                     "input": sum(t.get(k) or 0 for k in ("input_tokens", "cache_read_input_tokens",
                                                          "cache_creation_input_tokens")),
                     "output": t.get("output_tokens") or 0, "seconds": u.get("wall_seconds"),
                     **({"stopped_early": True} if u.get("stopped_early") else {}),
                     **({"timed_out": True} if u.get("timed_out") else {})})
    return rows


def tokens_by_model(usage):
    """{model: {"calls", "input", "output"}} summed over the run's Claude stages (pure)."""
    out = {}
    for row in stage_usage(usage):
        m = out.setdefault(row["model"], {"calls": 0, "input": 0, "output": 0})
        m["calls"] += 1
        m["input"] += row["input"]
        m["output"] += row["output"]
    return out


def research(question, on_event=None, cancel=None, fresh=False, follow=None):
    """Research one question and return the result (see _research).

    on_event(dict) receives progress (see Run). Setting the `cancel` threading.Event kills the
    running children and stops the research. Raises ResearchError on failure and
    ResearchCancelled after a cancel or Ctrl+C. fresh (or CRA_FRESH=1) bypasses exact reuse
    and the retrieval cache for this run; its results still replace the stored entries.
    follow: the run folder of the answer this question follows up; the planner rewrites
    the question to stand alone from that exchange, and that answer's passages are offered as
    remembered candidates.
    """
    previous = previous_exchange(follow) if follow else None
    run = Run(question.strip(), on_event, cancel, fresh_requested(fresh), previous)
    try:
        return _research(run)
    except KeyboardInterrupt:
        # Ctrl+C anywhere (between stages, or racing a child's exit) cancels the whole question.
        run.cancel.set()
        run.cancelled()


# ---- NotebookLM chat as the primary retrieval (fact questions) --------------------------
# NotebookLM asks replace the source searches and the coverage check on the normal path. The
# planner's precise fact questions are asked in numbered batches (see ask_facts). Every citation
# becomes an exact evidence passage; the replies reach the answer pass only as a lead.
# search_research is the fallback.
# Seconds: a guard against a stuck call only. A slow ask is never a reason to fall back.
ASK_TIMEOUT = 300
ASK_PARSE_VERSION = "1"  # bump when the ask path reads different reply fields (ask cache)
ASK_QUERY = "(NotebookLM ask)"  # the "query" of an ask citation in raw hits and memory
ASK_DEPTH = "normal"  # evidence budget and answer effort of the ask path
ASK_FOLLOW_UP_MAX = 3  # fact questions in the one follow-up round
CITE_MARK = re.compile(r"(\s?)\[(\d+(?:\s*[-–,]\s*\d+)*)\]")


def ask_notebooklm(run, n):
    """One `notebooklm ask --json` with the prompt saved as ask-<n>.prompt.txt; returns (reply,
    error). Every ask starts a fresh conversation (--new; the client can only start one by
    deleting the notebook's current one, --yes skips its prompt). Continuing one conversation
    across runs let NotebookLM answer a topic it had answered before from the conversation's
    history: the earlier reply restated with source titles inline and `references: []`, so no
    passage could be recovered. Each fact question is self-contained; no ask needs a prior turn."""
    path = run.dir / f"ask-{n}.prompt.txt"
    args = ["ask", "-n", NOTEBOOK, "--new", "--yes", "--json", "--prompt-file", str(path)]
    returncode, stdout, stderr, status = run_child(run, NOTEBOOKLM + args, timeout=ASK_TIMEOUT)
    if status == CANCELLED:
        run.cancelled()
    if status == "timeout":
        return None, f"NotebookLM ask stuck: no reply after {ASK_TIMEOUT}s"
    if returncode != 0:
        return None, f"NotebookLM ask failed (exit {returncode}): {(stderr or stdout)[-300:]}"
    try:
        reply = json.loads(stdout)
    except ValueError:
        return None, f"NotebookLM ask failed: output is not JSON: {stderr[-300:]}"
    if not isinstance(reply, dict) or not str(reply.get("answer") or "").strip():
        return None, "NotebookLM ask returned an empty reply"
    return reply, None


def ask_key(prompt):
    """Identity of a NotebookLM ask: the notebook, the prompt as sent (NFKC, whitespace
    collapsed) and the parse version of its reply."""
    prompt = " ".join(unicodedata.normalize("NFKC", prompt).split())
    return sha256({"request": ["ask", "-n", NOTEBOOK, prompt], "parse": ASK_PARSE_VERSION})


def ask_cache_lookup(run, prompt):
    """The stored reply of an identical ask on the same corpus identity: ({"reply", "run_id",
    "retrieved_at"}, None, key), or (None, miss reason, key)."""
    key = ask_key(prompt)
    if run.fresh:
        return None, "bypassed (fresh run)", key
    try:
        entry, reason = read_entry(REUSE_DIR / "ask" / f"{key}.json")
        if entry is None:
            return None, reason, key
        corpus = corpus_identity()
        reason = (("parse version changed" if entry.get("parse") != ASK_PARSE_VERSION else None)
                  or corpus_mismatch(entry.get("corpus"), corpus)
                  or too_old(corpus, entry.get("retrieved_ts"))
                  or (None if isinstance(entry.get("reply"), dict) else "no reply stored")
                  # A reply without citations is asked again, never reused
                  or (None if entry["reply"].get("references") else "stored reply has no citations"))
        if reason:
            return None, reason, key
        return {"reply": entry["reply"], "run_id": entry.get("run_id"),
                "retrieved_at": entry.get("retrieved_at")}, None, key
    except Exception as e:  # noqa: BLE001 - a failed lookup is a miss
        return None, f"lookup error ({e!r})"[:200], key


def ask_cache_store(run, key, prompt, reply):
    """Store one successful ask's reply and citations for an identical later ask (only
    a reply that cites something)."""
    if not reply.get("references"):
        return
    try:
        now = time.time()
        write_json_atomic(REUSE_DIR / "ask" / f"{key}.json", {
            "version": 1, "prompt": prompt, "notebook": NOTEBOOK, "parse": ASK_PARSE_VERSION,
            "corpus": corpus_identity(), "run_id": run.dir.name if run.dir else None,
            "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
            "retrieved_ts": now, "reply": reply})
    except Exception as e:  # noqa: BLE001 - a failed cache write only costs a later miss
        run.log("ask_cache_write", key=key[:12], written=False, reason=f"{e!r}"[:200])


def record_ask_citations(run, candidates, questions):
    """cited_passages rows for the passages an ask cited: the run's question (and standalone
    rewrite) with the fact questions each passage was cited for (`questions` maps fact
    ids to their text) as the requirement."""
    written = run.trace["memory"]["written"]
    try:
        store = open_memory(create=True)
        if store is None:
            return
        import research_memory as rm
        with store:
            for c in candidates:
                if c.get("origin") != "notebooklm_ask":
                    continue
                answered = "; ".join(questions[q] for q in c.get("facts") or [] if q in questions)
                for uid in c.get("memory_units") or []:
                    if store.layer(uid) == rm.PRIMARY:
                        written["ask_citations"] = written.get("ask_citations", 0) + store.add_citation(
                            uid, run.dir.name, run.question, run.standalone, answered,
                            origin="notebooklm_ask")
    except Exception as e:  # noqa: BLE001 - memory is optional
        memory_failed(run, "ask-citations", e)


def ask(run, prompt, n):
    """Ask NotebookLM (ask n of this run), or reuse the stored reply of an identical ask (see
    ask_cache_lookup); returns (reply, error). reply is the `ask --json` output: answer, conversation_id and references (source_id, citation_number, cited_text,
    start_char, end_char, ...)."""
    record = {"n": n, "seconds": 0.0, "cache": False}
    run.trace["ask"]["asks"].append(record)
    start = time.monotonic()
    run.save(f"ask-{n}.prompt.txt", prompt)
    hit, miss, key = ask_cache_lookup(run, prompt)
    record["key"] = key[:12]
    if hit is not None:
        record.update(cache=True, cached_from=hit["run_id"], retrieved_at=hit["retrieved_at"])
        reply, err = hit["reply"], None
    else:
        record["miss"] = miss
        reply, err = ask_notebooklm(run, n)
        if reply is not None:
            ask_cache_store(run, key, prompt, reply)
    record["seconds"] = round(time.monotonic() - start, 1)
    record["error"] = err
    if reply is not None:
        record["citations"] = len(reply.get("references") or [])
        record["conversation_id"] = reply.get("conversation_id")
        run.save(f"ask-{n}.json", json.dumps(reply, indent=1, ensure_ascii=False))
    run.log("ask", **record)
    return reply, err


def utf16_index(text, units):
    """The index in `text` after `units` UTF-16 code units (NotebookLM citation offsets count
    them; a character outside the BMP counts two)."""
    count = 0
    for i, ch in enumerate(text):
        if count >= units:
            return i
        count += 2 if ord(ch) > 0xFFFF else 1
    return len(text)


def cited_raw(run, reply, titles, n):
    """The reply's citations as raw hits of exact source text (see search() for the shape).

    The cited text is located in the source's fulltext (exactly, then ignoring whitespace or
    markup) and the fulltext span becomes the passage; a cited text the fulltext cannot confirm
    (no fulltext, or no match) is kept as NotebookLM returned it, which is verbatim source text.
    A citation with offsets only is sliced from the fulltext at its (UTF-16) range. A citation
    with neither is not evidence."""
    refs = [r for r in (reply or {}).get("references") or []
            if isinstance(r, dict) and r.get("source_id")]
    sids = [r["source_id"] for r in refs]
    load_sources(run, sids, titles, sids)
    texts, raw, methods = run.sources["texts"], [], {}
    for i, ref in enumerate(refs, 1):
        sid, cited = ref["source_id"], str(ref.get("cited_text") or "").strip()
        content, text, start, end, method = texts.get(sid), None, None, None, None
        if cited:
            span = content and locate(content, {"text": cited, "start": None, "end": None},
                                      run.sources["aux"].setdefault(sid, {}))
            if span:
                start, end, how = span
                text, method = content[start:end].strip(), f"fulltext_{how}"
            if not text:
                text, start, end, method = cited, None, None, "cited_text"
        elif (content and isinstance(ref.get("start_char"), int)
              and isinstance(ref.get("end_char"), int)):
            start, end = utf16_index(content, ref["start_char"]), utf16_index(content, ref["end_char"])
            text, method = content[start:end].strip(), "fulltext_offsets"
        if not text:
            methods["unrecovered"] = methods.get("unrecovered", 0) + 1
            continue
        methods[method] = methods.get(method, 0) + 1
        raw.append({"raw_id": f"a{n}.{i}", "query": ASK_QUERY, "source_id": sid, "text": text,
                    "rank": ref.get("citation_number"), "start": start, "end": end,
                    "origin": "fresh", "citation": ref.get("citation_number"), "recovery": method})
    info = run.trace["ask"]
    info["citations"] += len(refs)
    info["passages_recovered"] += len(raw)
    for k, v in methods.items():
        info["recovery"][k] = info["recovery"].get(k, 0) + v
    return raw


def ask_candidates(run, raw, known=(), round_n=1):
    """Candidates from ask citations (the shape search() gives): identical or contained passages
    of one source are one candidate (the fuller text kept), and a citation repeating a `known`
    candidate adds nothing. New hit ids continue after the known ones."""
    base = max((int(c["hit_id"][1:]) for c in known if re.fullmatch(r"h\d+", c["hit_id"])),
               default=0)
    candidates = []
    for r in raw:
        found = {"query": r["query"], "rank": r["rank"], "start": r["start"], "end": r["end"]}
        facts = r.get("facts") or []  # The fact questions the citation answers
        prior = next((c for c in known if c["source_id"] == r["source_id"]
                      and duplicate_reason(c, r)), None)
        if prior:
            prior["facts"] = list(dict.fromkeys((prior.get("facts") or []) + facts))
            r.update(candidate=prior["hit_id"], merge="already found")
            continue
        for c in candidates:
            why = c["source_id"] == r["source_id"] and duplicate_reason(c, r)
            if why:
                if len(r["text"]) > len(c["text"]):
                    c.update(text=r["text"], start=r["start"], end=r["end"])
                c["found_by"].append(found)
                c["raw_ids"].append(r["raw_id"])
                c["citations"].append(r["citation"])
                c["facts"] = list(dict.fromkeys(c["facts"] + facts))
                r.update(candidate=c["hit_id"], merge=why)
                break
        else:
            hit_id = f"h{base + len(candidates) + 1}"
            candidates.append({"hit_id": hit_id, "source_id": r["source_id"], "text": r["text"],
                               "start": r["start"], "end": r["end"], "found_by": [found],
                               "raw_ids": [r["raw_id"]], "rank": r["rank"],
                               "origin": "notebooklm_ask", "citations": [r["citation"]],
                               "recovery": r["recovery"], "facts": list(facts)})
            r.update(candidate=hit_id, merge=None)
    for c in candidates:
        reason = continuation_reason(c["text"])
        c["needs_continuation"] = reason is not None
        c["continuation_reason"] = reason
    run.trace["raw"] += [dict(r, round=round_n) for r in raw]
    run.trace["candidates"] += candidates
    run.save(f"ask-citations-{round_n}.json", json.dumps({"raw_hits": raw, "candidates": candidates},
                                                         indent=1, ensure_ascii=False))
    return candidates


def mark_numbers(inside):
    """The citation numbers inside one [..] mark ("3", "1, 4", "2-5") (pure)."""
    numbers = []
    for part in re.split(r"\s*,\s*", inside):
        ends = [int(x) for x in re.split(r"\s*[-–]\s*", part) if x.strip().isdigit()]
        if len(ends) == 2 and 0 < ends[1] - ends[0] < 50:
            numbers += range(ends[0], ends[1] + 1)
        else:
            numbers += ends[:1]
    return numbers


def marks_to_ids(text, by_number, have):
    """Reply text with its [n] citation marks replaced by the evidence ids of the passages they
    cite (pure; by_number: citation number -> hit id); a mark whose passage is not in the
    evidence (`have`) is dropped."""
    def ids(m):
        hits = "".join(f"[{h}]" for h in dict.fromkeys(by_number.get(x)
                                                        for x in mark_numbers(m.group(2)))
                       if h in have)
        return m.group(1) + hits if hits else ""
    return CITE_MARK.sub(ids, text or "").strip()


def strip_marks(text):
    """Reply text without its [n] citation marks (pure)."""
    return CITE_MARK.sub("", text or "").strip()


# ---- Fact questions -----------------------------------------------------------------------
# The planner's asks go to NotebookLM as numbered batches; each citation is mapped back to the
# question whose answer carries it. A question an earlier run already asked (the ask ledger in
# research memory, matched by FTS candidates and one COVERAGE_MODEL call) reuses that run's cited
# passages instead of asking again.
# Notebooklm-py keeps one server-side conversation per notebook (a new one only by
# deleting the current one) and serializes asks in a conversation behind a lock, so separate
# concurrent asks are not possible; batches go one after another.
# A long batch can come back with inline "[Title.md]" text and an empty references list where
# single-question asks return citations, so a batch holds at most 3 questions and a question
# answered without a citation is asked again alone, once.
ASK_BATCH = 3  # numbered questions per NotebookLM ask
ASK_BATCH_PREFIX = ("Answer each numbered question briefly (at most 60 words each) from the "
                    "sources only. For each, cite every relevant source, including ones giving "
                    "timings, amounts, frequencies or conditions. No advice.")
# A section of a batch reply starts at a line beginning with its question's number.
NUMBERED_LINE = re.compile(r"^[ \t>#*_]*(?:question[ \t]*)?(\d{1,2})[ \t]*[.):]", re.I | re.M)
LEDGER_MATCH = (COVERAGE_MODEL, None)  # the Haiku fallback of the Gemini match
LEDGER_REPLY = re.compile(r"^\s*(\d+)\s*[:.)-]\s*L?(\d+|none)\b", re.I | re.M)
LEDGER_CANDIDATES = 3  # ledger rows offered per planned question
VOCAB_FILE = settings.CACHE_DIR / "corpus-vocabulary.json"
VOCAB_MIN_COUNT = 3  # a corpus word seen fewer times is not a spelling target (OCR noise)
_VOCAB = [None, None, None]  # signature, {word: count}, {first letter: [words]}
_VOCAB_LOCK = threading.Lock()


def batch_prompt(questions):
    """One NotebookLM ask for a batch of fact questions (pure)."""
    return ASK_BATCH_PREFIX + "\n\n" + "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))


def split_numbered(answer, count):
    """A batch reply split per question (pure): one section per question ("" when it has none),
    or None when the reply is not numbered. Sections start at lines beginning with the next
    expected number (1, 2, ...), so a numbered list inside a section starts no new one."""
    if count == 1:
        return [answer.strip()]
    starts, want = [], 1
    for m in NUMBERED_LINE.finditer(answer):
        if int(m.group(1)) == want:
            starts.append(m.start())
            want += 1
            if want > count:
                break
    if len(starts) < 2:
        return None
    sections = [answer[s:e].strip() for s, e in zip(starts, starts[1:] + [len(answer)])]
    return sections + [""] * (count - len(sections))


def fact_citations(reply, count):
    """(sections, {citation number: [question positions]}) for one batch reply (pure). A reply
    that is not numbered gives every question the whole reply and every citation."""
    answer = str((reply or {}).get("answer") or "")
    sections = split_numbered(answer, count)
    if sections is None:
        sections = [answer.strip()] * count
    cites = {}
    for k, section in enumerate(sections):
        for m in CITE_MARK.finditer(section):
            for x in mark_numbers(m.group(2)):
                if k not in cites.setdefault(x, []):
                    cites[x].append(k)
    return sections, cites


def corpus_vocabulary():
    """({word: count}, {first letter: [words]}) over the cached source fulltexts: lowercase words
    of 4+ letters seen at least VOCAB_MIN_COUNT times. Cached in VOCAB_FILE and in the process,
    both keyed by the fulltext cache's signature (file count and modification times)."""
    try:
        files = sorted(CACHE_DIR.glob("*.txt"))
        sig = f"{len(files)}:{sum(int(f.stat().st_mtime) for f in files)}"
    except OSError:
        files, sig = [], "0:0"
    with _VOCAB_LOCK:
        if _VOCAB[0] == sig:
            return _VOCAB[1], _VOCAB[2]
        words = None
        try:
            stored = json.loads(VOCAB_FILE.read_text(encoding="utf-8"))
            if stored.get("signature") == sig and isinstance(stored.get("words"), dict):
                words = stored["words"]
        except (OSError, ValueError, AttributeError):
            pass
        if words is None:
            counts = {}
            for f in files:
                try:
                    text = f.read_text(encoding="utf-8", errors="ignore").casefold()
                except OSError:
                    continue
                for w in re.findall(r"[a-z]{4,}", text):
                    counts[w] = counts.get(w, 0) + 1
            words = {w: c for w, c in counts.items() if c >= VOCAB_MIN_COUNT}
            try:
                write_json_atomic(VOCAB_FILE, {"signature": sig, "words": words})
            except OSError:
                pass
        index = {}
        for w in words:
            index.setdefault(w[0], []).append(w)
        _VOCAB[:] = [sig, words, index]
        return words, index


def edit_distance(a, b, limit):
    """Levenshtein distance of a and b, or limit + 1 once it exceeds `limit` (pure)."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > limit:
            return limit + 1
        prev = cur
    return prev[-1]


def ledger_key(question, vocab=None, index=None):
    """A fact question's ask-ledger key (pure given the vocabulary): NFKC, case-folded words,
    each word of 4+ letters that the corpus vocabulary lacks replaced by its nearest corpus word
    within edit distance 2 (1 for words of up to 5 letters; ties go to the more frequent word)
    when there is one. Only the key is normalized; the question sent to NotebookLM is not."""
    words = re.findall(r"\w+", unicodedata.normalize("NFKC", question or "").casefold())
    out = []
    for w in words:
        if vocab and len(w) >= 4 and w.isalpha() and w not in vocab:
            limit = 1 if len(w) <= 5 else 2
            best = None
            for cand in (index or {}).get(w[0], ()):
                if abs(len(cand) - len(w)) > limit:
                    continue
                d = edit_distance(w, cand, limit)
                if d <= limit and (best is None or (d, -vocab[cand]) < (best[0], -vocab[best[1]])):
                    best = (d, cand)
            w = best[1] if best else w
        out.append(w)
    return " ".join(out)


def new_facts(questions, round_n, start=0):
    """Fact-question records for one round (ids q<start+1>.. continue across rounds)."""
    return [{"id": f"q{start + i}", "question": q, "round": round_n, "origin": "asked",
             "batch": None, "citations": 0, "passages": 0, "ledger_id": None, "error": None}
            for i, q in enumerate(questions, 1)]


def ledger_match(run, facts):
    """Match this round's fact questions against the ask ledger: FTS candidates (top
    LEDGER_CANDIDATES per question by its ledger key), then one COVERAGE_MODEL call for all of
    them. Returns {fact id: the matched ledger row}; empty on "Research again" (a fresh run), with
    no memory, no candidates, or any failure (every question is then asked)."""
    info = run.trace["ask"]["ledger"]
    try:
        vocab, index = corpus_vocabulary()
    except Exception as e:  # noqa: BLE001 - an unnormalized key only matches less often
        vocab, index = {}, {}
        run.log("ledger_vocabulary_failed", error=repr(e)[:200])
    for f in facts:
        f["key"] = ledger_key(f["question"], vocab, index)
    if run.fresh:
        info["status"] = "bypassed (research again)"
        return {}
    offered = {}
    try:
        store = open_memory(create=False)
        if store is None:
            info["status"] = "no memory"
            return {}
        with store:
            for f in facts:
                rows = [r for r in store.ask_fact_candidates(f["key"], LEDGER_CANDIDATES)
                        if r["unit_ids"]]
                if rows:
                    offered[f["id"]] = rows
    except Exception as e:  # noqa: BLE001 - memory is optional
        memory_failed(run, "ledger-lookup", e)
        info["status"] = "error"
        return {}
    info["offered"] += len(offered)
    if not offered:
        info["status"] = "no candidates"
        return {}
    lines = []
    for n, f in enumerate(facts, 1):
        lines.append(f"{n}. {f['question']}")
        for r in offered.get(f["id"], []):
            lines.append(f"   L{r['ledger_id']}: {r['question']}")
        if f["id"] not in offered:
            lines.append("   (no candidates)")
    prompt = "PLANNED QUESTIONS:\n" + "\n".join(lines)
    stage = f"ledger-{facts[0]['round']}"
    run.save(f"{stage}.input.txt", prompt)
    # A mechanical match, so Gemini Flash does it; Haiku only when Gemini fails.
    text = gemini_task(run, stage, "ledger", f"{LEDGER_SYSTEM}\n\n{prompt}",
                       lambda t: (t, None) if LEDGER_REPLY.search(str(t))
                       else (None, "unusable_output"))
    if text is None:
        info["fallback"] = run.gemini_fallback
        try:
            text = claude(run, stage, LEDGER_MATCH, LEDGER_SYSTEM, prompt)
        except ResearchError as e:
            run.log("ledger_match_failed", error=str(e))
            info["status"] = "match failed"
            return {}
    matched = {}
    for m in LEDGER_REPLY.finditer(str(text)):
        n, wanted = int(m.group(1)), m.group(2)
        if not 1 <= n <= len(facts) or not wanted.isdigit():
            continue
        f = facts[n - 1]
        row = next((r for r in offered.get(f["id"], []) if r["ledger_id"] == int(wanted)), None)
        if row:
            matched[f["id"]] = row
    info["status"] = "ok"
    info["matched"] += len(matched)
    run.log("ledger_match", offered={k: [r["ledger_id"] for r in v] for k, v in offered.items()},
            matched={k: r["ledger_id"] for k, r in matched.items()}, reply=str(text)[:500])
    return matched


def ledger_passages(run, facts, matched):
    """The stored primary passages of each matched fact question ([{unit_id, text, source_id,
    source_title, fact}]), found under the current corpus; a match none of whose passages is
    usable is dropped, so that question is asked. Marks the reused facts."""
    if not matched:
        return []
    out = []
    try:
        store = open_memory(create=False)
        import research_memory as rm
        corpus = corpus_identity()
        with store:
            for f in facts:
                row = matched.get(f["id"])
                if row is None:
                    continue
                got = []
                for uid in row["unit_ids"]:
                    u = store.db.execute("SELECT * FROM units WHERE unit_id = ?", (uid,)).fetchone()
                    if (u is None or u["layer"] != rm.PRIMARY or u["source_id"] is None
                            or not discovered_in_corpus(store, uid, corpus)):
                        continue
                    got.append({"unit_id": uid, "text": u["text"], "source_id": u["source_id"],
                                "source_title": u["source_title"], "fact": f["id"]})
                if got:
                    f.update(origin="reused", ledger_id=row["ledger_id"], reply=row["reply"] or "",
                             reused_from=row["run_id"], recorded_at=row["recorded_at"])
                    out += got
    except Exception as e:  # noqa: BLE001 - memory is optional; the questions are asked instead
        memory_failed(run, "ledger-passages", e)
        for f in facts:
            if f["origin"] == "reused":
                f.update(origin="asked", ledger_id=None)
        return []
    return out


def merge_ledger(run, candidates, reused, titles):
    """Add reused ledger passages to `candidates` (in place): one a candidate already holds (same
    source; identical or contained text) only adds its fact question to it; others become
    candidates with origin "ask_memory", like fresh citations."""
    for p in reused:
        probe = {"source_id": p["source_id"], "text": p["text"], "start": None, "end": None}
        same = next((c for c in candidates
                     if c["source_id"] == p["source_id"] and duplicate_reason(c, probe)), None)
        if same is not None:
            same["facts"] = list(dict.fromkeys((same.get("facts") or []) + [p["fact"]]))
            same.setdefault("memory_units", [])
            if p["unit_id"] not in same["memory_units"]:
                same["memory_units"].append(p["unit_id"])
            continue
        base = max((int(c["hit_id"][1:]) for c in candidates
                    if re.fullmatch(r"h\d+", c["hit_id"])), default=0)
        reason = continuation_reason(p["text"])
        c = {"hit_id": f"h{base + 1}", "source_id": p["source_id"], "text": p["text"],
             "start": None, "end": None,
             "found_by": [{"query": "(fact memory)", "rank": None, "start": None, "end": None}],
             "raw_ids": [], "rank": None, "needs_continuation": reason is not None,
             "continuation_reason": reason, "origin": "ask_memory",
             "memory_units": [p["unit_id"]], "facts": [p["fact"]]}
        candidates.append(c)
        run.trace["candidates"].append(c)
        if p["source_title"] and not titles.get(p["source_id"]):
            titles[p["source_id"]] = p["source_title"]


def ask_facts(run, facts, titles, known, round_n):
    """One round of fact questions: the ask-ledger match first, then the unmatched questions
    asked of NotebookLM in numbered asks of at most ASK_BATCH, one after another in the app's
    conversation (the client cannot ask concurrently). Each citation is mapped to the
    question(s) whose section carries its mark; a question whose reply section cites nothing is
    asked again alone, once. Returns (raw hits, new candidates, errors); `known`
    (earlier rounds' candidates) only gains fact ids."""
    info = run.trace["ask"]
    reused = ledger_passages(run, facts, ledger_match(run, facts))
    to_ask = [f for f in facts if f["origin"] != "reused"]
    # the web app's "Asking NotebookLM (n asked, m reused)"
    run.emit("ask_plan", round=round_n, asked=len(to_ask), reused=len(facts) - len(to_ask))
    raw, errors = [], []

    def one_ask(batch, retry=False):
        n = len(info["asks"]) + 1
        reply, err = ask(run, batch_prompt([f["question"] for f in batch]), n)
        info["asks"][-1]["questions"] = [f["id"] for f in batch]
        if retry:
            info["asks"][-1]["retry"] = True
        if reply is None:
            for f in batch:
                if retry:  # the first reply stands
                    f["retry"] = {"batch": n, "error": err}
                else:
                    f.update(origin="failed", error=err, batch=n)
            if not retry:  # a failed retry leaves the first reply in place
                errors.append(err)
            return
        sections, cites = fact_citations(reply, len(batch))
        got = cited_raw(run, reply, titles, n)
        for k, f in enumerate(batch):
            count = sum(1 for qs in cites.values() if k in qs)
            if retry:
                f["retry"] = {"batch": n, "citations": count}
                if not count:
                    continue  # nothing better than the first reply
            f.update(batch=n, reply=sections[k], citations=count)
        for r in got:
            r["batch"] = n
            number = r["citation"]
            number = int(number) if str(number).strip().isdigit() else number
            r["facts"] = [batch[k]["id"] for k in cites.get(number, [])]
            if r["facts"]:
                r["query"] = next(f["question"] for f in batch if f["id"] == r["facts"][0])
        raw.extend(got)

    for s in range(0, len(to_ask), ASK_BATCH):
        one_ask(to_ask[s:s + ASK_BATCH])
    for f in to_ask:  # Answered without a citation -> asked again alone, once
        if f["origin"] == "asked" and not f.get("citations"):
            one_ask([f], retry=True)
    new = ask_candidates(run, raw, known=known, round_n=round_n)
    merged = list(known) + new
    merge_ledger(run, merged, reused, titles)
    new = merged[len(known):]
    for f in facts:
        f["passages"] = sum(1 for c in list(known) + new if f["id"] in (c.get("facts") or []))
    return raw, new, errors


def fact_groups(facts, selected):
    """The evidence groups (pure): per fact question that was answered or reused, its heading
    and the selected passages cited for it, in question order."""
    return [(f"FACT QUESTION {f['id']}: {f['question']}",
             [h["hit_id"] for h in selected if f["id"] in (h.get("facts") or [])])
            for f in facts if f["origin"] != "failed"]


def fact_findings(facts, raw, candidates):
    """The NotebookLM findings for the answer pass (pure): per fact question, its short reply
    with citation marks replaced by the evidence ids of the passages they cite (per batch); a
    reused question shows its stored reply. A reply repeated for several questions (an
    unnumbered batch reply) is shown once."""
    have = {c["hit_id"] for c in candidates}

    def number(r):  # the citation number as the reply's marks count it
        n = r.get("citation")
        return int(n) if str(n).strip().isdigit() else n
    cite_of = {r["raw_id"]: (r.get("batch"), number(r)) for r in raw}
    by_batch = {}
    for r in raw:
        if r.get("citation") is not None and r.get("candidate") in have:
            by_batch.setdefault(r.get("batch"), {})[number(r)] = r["candidate"]
    for c in candidates:  # a collapsed duplicate's citation points to the passage kept for it
        for rid in c.get("raw_ids") or []:
            if rid in cite_of and cite_of[rid][1] is not None:
                by_batch.setdefault(cite_of[rid][0], {})[cite_of[rid][1]] = c["hit_id"]
    out, shown = [], set()
    for f in facts:
        if f["origin"] == "failed":
            continue
        text = (marks_to_ids(f.get("reply") or "", by_batch.get(f["batch"], {}), have)
                if f["origin"] == "asked" else strip_marks(f.get("reply") or ""))
        if text and text in shown:
            text = "(answered in the reply above)"
        shown.add(text)
        out.append(f"{f['id']}. {f['question']}\n{text or '(no reply)'}")
    return "\n\n".join(out)


def ledger_store(run, facts, candidates):
    """Record every freshly asked fact question in the ask ledger with the primary passages cited
    for it and its short reply (citation marks removed). A question with no stored passage is not
    recorded, so a later run asks it again."""
    units = {}
    for c in candidates:
        for q in c.get("facts") or []:
            units.setdefault(q, []).extend(c.get("memory_units") or [])
    info = run.trace["ask"]["ledger"]
    try:
        store = open_memory(create=True)
        if store is None:
            return
        import research_memory as rm
        with store:
            for f in facts:
                if f["origin"] != "asked" or not f.get("key"):
                    continue
                kept = [u for u in units.get(f["id"], []) if store.layer(u) == rm.PRIMARY]
                if kept:
                    f["ledger_id"] = store.add_ask_fact(f["key"], f["question"], kept,
                                                        strip_marks(f.get("reply")), run.dir.name,
                                                        PIPELINE_POLICY_VERSION)
                    info["written"] += 1
    except Exception as e:  # noqa: BLE001 - memory is optional
        memory_failed(run, "ledger-store", e)


def planning(run, pending):
    """The planner stage on the main path: plan() timed as Planning while the preflight
    (`pending`) runs; a planner failure while the preflight failed reports the preflight's
    message. Returns (depth, requirements, searches), which the fallback reuses."""
    model, effort = PLANNER
    t = run.begin("plan", "Planning", cli="planning")
    try:
        depth, requirements, search_plan = plan(run, run.question, run.community)
    except ResearchError:
        checked = pending.result()
        if not checked["ok"]:
            raise ResearchError(checked["message"], provider=checked["provider"],
                                kind=checked["kind"])
        raise
    secs = run.timed("Planner", t)
    n = len(run.asks)
    run.done("plan", secs, detail=(
                 "Answering from earlier research" + (" + 1 premise question" if n else "")
                 if run.reuse == ANSWER_FROM_TURN
                 else f"{n} fact " + ("question" if n == 1 else "questions")),
             summary=f"Planner: {model} / {effort}, {secs}, {len(requirements)} requirements, "
                     f"{n} fact questions, {len(search_plan)} fallback searches")
    run.emit("plan", depth=depth, searches=[s["query"] for s in search_plan],
             requirements=requirements, asks=run.asks,
             standalone=run.standalone if run.standalone != run.question else None)
    run.log("planner", model=model, effort=effort, seconds=round(run.times[-1][1], 1), depth=depth,
            requirements=len(requirements), searches=len(search_plan), asks=run.asks,
            verifications=run.verifications)
    return depth, requirements, search_plan


def ask_research(run):
    """Primary research path: the planner (see planning) writes precise fact questions;
    each is matched against the ask ledger first and the rest go to NotebookLM in one numbered
    ask (see ask_facts). Every citation becomes an exact evidence passage under the fact
    question it answers; the memory lookup (community records, remembered passages) runs beside
    the asks; the answer pass reads NotebookLM's short replies as a lead. An answer pass that asks
    for more gets one more round of fact questions (ledger first, then NotebookLM), then the final
    answer pass.

    Returns the STATE_KEYS state, or None when the fallback must run: every ask failed (error,
    auth, empty reply, stuck call) or no passage was recovered from the asks or the ledger; the
    reason is in the trace."""
    info = run.trace["ask"] = {"seconds": 0.0, "asks": [], "citations": 0,
                               "passages_recovered": 0, "recovery": {},
                               "passages_from_memory": 0, "passages_from_ledger": 0,
                               "follow_up": None, "cache_hit": False, "fallback": None,
                               "facts": [], "reuse": None,
                               "ledger": {"status": "not_used", "offered": 0, "matched": 0,
                                          "written": 0}}
    side = ThreadPoolExecutor(max_workers=2)
    pending = side.submit(preflight)
    # The community lookup (local, fast) before the planner, keyed on the question
    # (spelling-normalized, alias-expanded), so the planner sees the records and writes the
    # verification question for those that attribute a statement to the corpus author
    run.community = community_lookup(run, community_keys([("question", run.question),
                                                          ("community_search", run.question)]),
                                     run.question, PLANNER_COMMUNITY_POOL) or []
    # The planner runs first again (beside the preflight): the standalone rewrite, the
    # related turn, the fact questions and, for the fallback, the searches.
    run.planned = planning(run, pending)
    question = run.standalone
    side.shutdown(wait=False)
    # A follow-up the planner answers from the related turn's evidence skips the memory
    # lookup and asks NotebookLM at most one missing premise (or nothing)
    from_turn = run.reuse == ANSWER_FROM_TURN
    remembered = [] if from_turn else memory_lookup(
        run, question, [("question", question), ("community_search", question)],
        with_secondary=False)[0]
    secondary = run.community
    facts = info["facts"] = new_facts(run.asks + [v["question"] for v in run.verifications], 1)
    add_verification_facts(facts, run.verifications, secondary)
    titles = load_titles()
    raw, candidates, errors = [], [], []
    if from_turn:
        info["reuse"] = {"mode": ANSWER_FROM_TURN,
                         "related_turn": run.related.get("turn_id"),
                         "passages": len(previous_candidates(run.previous, run.related, True)),
                         "premise_asks": list(run.asks), "follow_up_asks": []}
        run.emit("reuse", mode=ANSWER_FROM_TURN, passages=info["reuse"]["passages"],
                 premise_asks=run.asks)
    if facts:
        t = run.begin("ask", "Asking NotebookLM" if not from_turn
                      else "Asking NotebookLM (one missing premise)", cli="asking NotebookLM")
        raw, candidates, errors = ask_facts(run, facts, titles, [], 1)
        secs = run.timed("NotebookLM asks", t, sum(len(c["text"]) for c in candidates))
        info["seconds"] = round(run.times[-1][1], 1)
    reused = sum(f["origin"] == "reused" for f in facts)
    asked = len(facts) - reused
    fallback = None
    if from_turn:
        if facts:
            run.done("ask", secs, detail=f"{len(candidates)} new "
                     + ("passage" if len(candidates) == 1 else "passages")
                     + (f"; failed: {errors[0]}" if errors else ""),
                     summary=f"NotebookLM premise ask: {secs}; {len(candidates)} new passages"
                             + (f"; failed: {errors[0]}" if errors else ""))
    elif not candidates:
        fallback = (f"every ask failed: {errors[0]}"
                    if errors and not any(f["origin"] == "asked" for f in facts)
                    else "no cited passage could be recovered as exact text")
    if fallback:
        info["fallback"] = fallback
        run.done("ask", secs, detail=f"Failed: {fallback}",
                 summary=f"NotebookLM asks: {secs}; falling back to the search pipeline ({fallback})")
        run.log("ask_fallback", reason=fallback)
        run.emit("fallback", reason=fallback)
        return None
    if not from_turn:
        run.done("ask", secs, detail=f"{asked} asked, {reused} reused"
                 + (f", {len(errors)} of {len(info['asks'])} asks failed" if errors else ""),
                 summary=f"NotebookLM asks: {secs}; {len(facts)} fact questions ({asked} asked "
                         f"in {len(info['asks'])} asks, {reused} reused from fact memory); "
                         f"{info['citations']} citations, {info['passages_recovered']} "
                         f"recovered as exact text, {len(candidates)} unique passages")
    # A verification whose ask brought no passage is searched in the sources now
    passages = {f["question"]: f.get("passages") for f in facts}
    if any(not v.get("searched") and not passages.get(v["question"]) for v in run.verifications):
        t = run.begin("search", "Searching verification passages")
        vraw, vnew = verification_searches(run, facts, candidates, titles, 1, True)
        raw, candidates = raw + vraw, candidates + vnew
        secs = run.timed("Verification searches", t, sum(len(c["text"]) for c in vnew))
        run.done("search", secs, detail=f"{len(vnew)} new "
                 + ("passage" if len(vnew) == 1 else "passages"),
                 summary=f"Verification searches: {secs}; {len(vnew)} new candidates")

    settle_checks(secondary, facts)
    # When every fact question brought passages, remembered passages found only by word
    # overlap (never cited by an earlier answer) add breadth, not answers: leave them out.
    if all(passages.get(f["question"]) or f.get("passages") for f in facts):
        loose = [m for m in remembered if not m.get("cited_before")]
        if loose:
            remembered = [m for m in remembered if m.get("cited_before")]
            run.trace["memory"]["loose_dropped"] = len(loose)
    record_memory_lookup(run, question, remembered, secondary)
    if run.related or run.prior:  # the related turn's (or earlier answer's) cited passages
        # Answered from the related turn, every passage it retrieved
        prior = previous_candidates(run.previous, run.related or run.prior, from_turn)
        have = {p["unit_id"] for p in prior}
        remembered = prior + [m for m in remembered if m["unit_id"] not in have]
        run.trace["memory"]["previous_passages"] = len(prior)
    if remembered:
        merge_memory_primary(run, candidates, remembered, titles)
    collapse_duplicates(run, candidates, titles)

    run.reading(True)  # Reading sources: from here to the answer's first text
    clipped = []
    # The related turn's passages already carry their recovered continuation
    fresh = [c for c in candidates
             if not (from_turn and any(u.startswith("prev:") for u in c.get("memory_units") or []))]
    if any(c["needs_continuation"] or fragment_start(c["text"]) for c in fresh):
        clipped = hydrate(run, fresh, titles, budget=EVIDENCE_BUDGET[ASK_DEPTH])
    reuse = run.trace["memory"]["reuse"]
    if not from_turn:  # No sub-recipe searches when answering from the related turn
        raw = raw + referenced_recipes(run, candidates, titles, ASK_DEPTH, [])
    capture_primary(run, raw, candidates, titles, 1, ASK_DEPTH, run.planned[1],
                    [f["question"] for f in facts])
    record_ask_citations(run, candidates, {f["id"]: f["question"] for f in facts})
    ledger_store(run, facts, candidates)
    selected, context_ids, coverage = select(run, question, ASK_DEPTH, candidates, titles, [])
    evidence, ev, sources = build_evidence(run, selected, context_ids, ASK_DEPTH, titles,
                                           groups=fact_groups(facts, selected))
    secondary_text = secondary_evidence(run, secondary)
    findings = fact_findings(facts, raw, candidates)
    checked = pending.result()
    if not checked["ok"] and checked["provider"] == "claude":  # NotebookLM already answered
        run.reading(False)
        raise ResearchError(checked["message"], run.dir, checked["provider"], checked["kind"])

    t = time.monotonic()
    decision = reason(run, "reasoner-1", question, ASK_DEPTH, evidence, [], coverage,
                      secondary=secondary_text, stop_when=lambda p: p["decision"] == "search",
                      requested_parts=run.requested_parts,
                      told=already_told(run.related, candidates, evidence),
                      referenced=referenced_text(run, evidence), findings=findings)
    run.timed("Reasoner", t)
    final_selected = selected
    if decision["decision"] == "search" or decision["stopped"]:
        # Answered from the related turn, one ask in all for one missing premise
        most = (ASK_FOLLOW_UP_MAX if not from_turn
                else max(0, ANSWER_FROM_TURN_ASKS - len(run.asks)))
        more = normalize_asks(decision["queries"][:most], "", most, True)
        if from_turn:
            info["reuse"]["follow_up_asks"] = more
        if more:
            # The community lookup again, keyed on the follow-up fact questions too
            # (a newly found record gets no verification, since only the planner writes
            # verification questions). The planner's pick stays; only records the
            # planner was not shown are added, up to MEMORY_SECONDARY_MAX in all.
            again = community_lookup(run, lookup_keys_for_community(
                run, [f["question"] for f in facts] + more, run.planned[2]), question)
            if again is not None:
                seen = set(run.community_pool) | {x["unit_id"] for x in secondary}
                new = [x for x in again if x["unit_id"] not in seen]
                secondary = secondary + new[:max(0, MEMORY_SECONDARY_MAX - len(secondary))]
        facts2 = new_facts(more, 2, len(facts))
        info["follow_up"] = {"questions": more, "citations": 0, "new_passages": 0,
                             "reused": 0, "error": None}
        run.writing(False)  # a first answer that streamed before this round is interrupted
        if facts2:
            t = run.begin("ask", "Asking again", cli="asking NotebookLM again")
            before = info["citations"]
            raw2, new, errors2 = ask_facts(run, facts2, titles, candidates, 2)
            collapse_duplicates(run, new, titles, 2, known=candidates)
            # Every verification not searched yet runs as a source search too
            vraw, vnew = verification_searches(run, facts, candidates + new, titles, 2, False)
            raw2, new = raw2 + vraw, new + vnew
            facts += facts2
            reused2 = sum(f["origin"] == "reused" for f in facts2)
            err2 = errors2[0] if errors2 and len(errors2) == len(
                {f["batch"] for f in facts2 if f["origin"] != "reused"}) else None
            secs = run.timed("NotebookLM asks (follow-up)", t, sum(len(c["text"]) for c in new))
            info["seconds"] = round(info["seconds"] + run.times[-1][1], 1)
            info["follow_up"].update(citations=info["citations"] - before,
                                     new_passages=len(new), reused=reused2, error=err2)
            run.done("ask", secs,
                     detail=(f"Failed: {err2}" if err2 else
                             f"{len(facts2) - reused2} asked, {reused2} reused, {len(new)} new "
                             + ("passage" if len(new) == 1 else "passages")),
                     summary=f"NotebookLM follow-up asks: {secs}; {len(new)} new passages"
                             + (f"; failed: {err2}" if err2 else ""))
            if new:
                if any(c["needs_continuation"] or fragment_start(c["text"]) for c in new):
                    clipped += hydrate(run, new, titles, 2, EVIDENCE_BUDGET[ASK_DEPTH], ev["chars"])
                capture_primary(run, raw2, new, titles, 2)
                record_ask_citations(run, new, {f["id"]: f["question"] for f in facts2})
                ledger_store(run, facts2, candidates + new)
                picked, context2, _ = select(run, question, ASK_DEPTH, new, titles, [], 2,
                                             prior_chars=ev["chars"])
                candidates = candidates + new
                final_selected = selected + picked
                evidence, ev, sources = build_evidence(run, final_selected, context_ids + context2,
                                                       ASK_DEPTH, titles, round_n=2,
                                                       groups=fact_groups(facts, final_selected))
            raw = raw + raw2
            findings = fact_findings(facts, raw, candidates)
            settle_checks(secondary, facts)
            secondary_text = secondary_evidence(run, secondary)
        repaired = {"ask": True, "questions": more,
                    "selected": len(final_selected) - len(selected),
                    "error": (info["follow_up"]["error"] if facts2
                              else "the answer pass asked for more without a fact question")}
        t = time.monotonic()
        decision = reason(run, "reasoner-2", question, ASK_DEPTH, evidence, [], coverage,
                          repaired, secondary=secondary_text, requested_parts=run.requested_parts,
                          told=already_told(run.related, candidates, evidence),
                          referenced=referenced_text(run, evidence), findings=findings)
        run.timed("Reasoner (final)", t)
    info["cache_hit"] = any(a["cache"] for a in info["asks"])
    info["passages_from_memory"] = sum(c.get("origin") == "memory" for c in candidates)
    info["passages_from_ledger"] = sum(c.get("origin") == "ask_memory" for c in candidates)
    info["facts"] = [{k: f.get(k) for k in ("id", "question", "round", "origin", "batch",
                                            "citations", "passages", "ledger_id", "reused_from",
                                            "error", "verifies", "retry")} for f in facts]
    run.log("ask_summary", **{k: v for k, v in info.items() if k != "asks"})
    return {"question": question, "depth": ASK_DEPTH, "requirements": [], "search_plan": [],
            "reuse": reuse, "raw": raw, "candidates": candidates,
            "candidate_chars": sum(len(c["text"]) for c in candidates), "clipped": clipped,
            "titles": titles, "selected": selected, "context_ids": context_ids,
            "coverage": coverage, "final": [dict(c) for c in coverage], "downgrades": [],
            "map_entries": [], "decision": decision, "evidence": evidence, "ev": ev,
            "sources": sources, "secondary": secondary, "secondary_text": secondary_text,
            "final_selected": final_selected, "repaired": None}


# The research state a path hands _research to finish the answer from.
STATE_KEYS = ("question", "depth", "requirements", "search_plan", "reuse", "raw", "candidates",
              "candidate_chars", "clipped", "titles", "selected", "context_ids", "coverage",
              "final", "downgrades", "map_entries", "decision", "evidence", "ev", "sources",
              "secondary", "secondary_text", "final_selected", "repaired")


def search_research(run, planned=None):
    """The fallback research path (the pipeline before it): planner, source searches,
    continuation, trim, coverage check, answer pass and one repair round. Returns the state
    _research finishes the answer from (see STATE_KEYS). `planned` is the main path's
    plan (see planning); the planner does not run again."""
    question = run.question
    # Preflight: NotebookLM token fetch and a callable claude CLI; a success is cached
    # for PREFLIGHT_CACHE_SECONDS. It runs while the planner (which needs no NotebookLM)
    # works, and is joined before any NotebookLM call; a planner failure while the preflight
    # failed reports the preflight's message.
    t_auth = run.begin("auth", "Checking logins")
    checks = ThreadPoolExecutor(max_workers=1)
    pending = checks.submit(preflight)
    checks.shutdown(wait=False)

    def auth_finished(future):
        # Connecting ends when the preflight itself does, not when the planner does. A
        # failed preflight leaves the stage open; the run then fails after the planner.
        if not future.cancelled() and future.exception() is None and future.result()["ok"]:
            secs = time.monotonic() - t_auth
            run.emit("stage_end", stage="auth", seconds=round(secs, 1), timed=f"{secs:.1f}s")
    pending.add_done_callback(auth_finished)

    def preflight_failed():
        result = pending.result()
        if not result["ok"]:
            raise ResearchError(result["message"], provider=result["provider"],
                                kind=result["kind"])

    if planned is None:
        model, effort = PLANNER
        t = run.begin("plan", "Planning", cli="planning searches")
        try:
            depth, requirements, search_plan = plan(run, question)
        except ResearchError:
            preflight_failed()
            raise
        searches = [s["query"] for s in search_plan]
        secs = run.timed("Planner", t)
        # Before the Auth row below: done() reports the last timed row, and Planning must not
        # show the Auth wait.
        run.done("plan", secs, detail=f"{depth} depth, {len(searches)} "
                 + ("search" if len(searches) == 1 else "searches"),
                 summary=f"Planner: {model} / {effort}, {secs}, depth {depth}, "
                         f"{len(requirements)} requirements, {len(searches)} searches")
    else:  # The main path's plan (its searches were written for this fallback)
        depth, requirements, search_plan = planned
        searches = [s["query"] for s in search_plan]
    # Every verification question's search string runs with the planned searches, so a
    # failed ask does not hide the passage where the corpus author said it
    have = {q.lower() for q in searches}
    search_plan = search_plan + [{"query": v["search"], "covers": [], "type": "verification"}
                                 for v in run.verifications if v["search"].lower() not in have]
    searches = [s["query"] for s in search_plan]
    for v in run.verifications:
        v["searched"] = True
    question = run.standalone  # a follow-up is researched as its standalone rewrite
    t_wait = time.monotonic()
    preflight_failed()
    run.times.append(("Auth", time.monotonic() - t_wait, None))  # the wait beyond the planner
    if run.cancel.is_set():
        run.cancelled()
    if planned is None:
        run.emit("plan", depth=depth, searches=searches, requirements=requirements,
                 standalone=run.standalone if run.standalone != run.question else None)
        run.log("planner", model=model, effort=effort, seconds=round(run.times[-1][1], 1),
                depth=depth, requirements=len(requirements), searches=len(searches))

    community = [s["query"] for s in search_plan if s.get("type") == "community"]
    # After the ask path, its community records (with their verifications) are kept
    remembered, secondary = memory_lookup(run, question,
                                          memory_keys(question, searches, run.case_frame_raw,
                                                      community),
                                          lookup_keys_for_community(run, run.asks, search_plan),
                                          subject=question,
                                          with_secondary=run.community is None)
    if run.community is not None:
        secondary = run.community
    record_memory_lookup(run, question, remembered, secondary)
    if run.related or run.prior:  # The related turn's cited passages first, as free
        # recall (or those of the earlier answer to this same question)
        prior = previous_candidates(run.previous, run.related or run.prior)
        have = {p["unit_id"] for p in prior}
        remembered = prior + [m for m in remembered if m["unit_id"] not in have]
        run.trace["memory"]["previous_passages"] = len(prior)
    # No subject-word filter. The top MEMORY_SECONDARY_MAX lookup records plus every
    # community-facet hit reach the answer pass, each labeled SECONDARY; the answer model
    # ignores the irrelevant ones.
    # Related-question reuse (off): a planned search was skipped when every requirement it covers was
    # "established" by an earlier run, whose selector judgments were then reused. With it off,
    # nothing is established: every planned search runs and the selector judges every candidate.
    established = (established_requirements(run, requirements)
                   if LEGACY_RELATED_REUSE_ENABLED else {})
    reuse = run.trace["memory"]["reuse"]
    to_run = [s for s in search_plan if not (s["covers"] and all(c in established for c in s["covers"]))]
    reuse["searches_fresh"] = [s["query"] for s in to_run]
    reuse["searches_skipped"] = [s["query"] for s in search_plan if s not in to_run]
    unresolved = [r["id"] for r in requirements if r["id"] not in established]

    if to_run:
        t = run.begin("search", "Searching")
        raw, candidates = search(run, reuse["searches_fresh"])
        candidate_chars = sum(len(c["text"]) for c in candidates)
        secs = run.timed("Retrieval", t, candidate_chars)
        cached = run.trace["retrieval"]["from_cache"]
        run.done("search", secs, detail=f"{len(raw)} passages, {len(candidates)} unique"
                 + (f", {cached} of {len(to_run)} searches from cache" if cached else "")
                 + (f", {len(reuse['searches_skipped'])} searches covered by memory"
                    if reuse["searches_skipped"] else ""),
                 summary=f"Retrieval: {secs}; {len(raw)} raw hits, "
                         f"{len(candidates)} deduplicated candidates"
                         + (f"; {cached}/{len(to_run)} searches from the retrieval cache"
                            if cached else "")
                         + (f"; {len(reuse['searches_skipped'])} searches skipped (established)"
                            if reuse["searches_skipped"] else ""))
    else:
        raw, candidates, candidate_chars = [], [], 0
        run.emit("stage_skip", stage="search", detail="Covered by established research")
    run.log("retrieval", seconds=round(run.times[-1][1], 1) if to_run else 0, raw_hits=len(raw),
            candidates=len(candidates), candidate_chars=candidate_chars,
            skipped_searches=reuse["searches_skipped"])

    titles = load_titles()
    reused = add_established(run, candidates, established, titles) if established else {}
    if remembered and (to_run or unresolved):
        taken = {p["unit_id"] for p in reuse["primary"]}
        remembered = [m for m in remembered if m["unit_id"] not in taken]
        if remembered:
            merge_memory_primary(run, candidates, remembered, titles)
    if not candidates:
        run.fail("source search returned no passages")
    collapse_duplicates(run, candidates, titles)  # One passage reprinted in two sources
    clipped = []
    if any(c["needs_continuation"] or fragment_start(c["text"]) for c in candidates):
        t = run.begin("continuation", "Completing source text")
        clipped = hydrate(run, candidates, titles, budget=EVIDENCE_BUDGET[depth])
        completed = [c["hit_id"] for c in clipped if c["continuation_status"] in CONTINUATION_DONE]
        headless = [c for c in candidates if "backward_status" in c]
        headed = [c["hit_id"] for c in headless if c["backward_status"] in BACKWARD_DONE]
        secs = run.timed("Continuation", t,
                         sum(len(c.get("continuation_text") or "")
                             + len(c.get("backward_text") or "") for c in candidates))
        run.done("continuation", secs,
                 detail=f"{len(completed)} of {len(clipped)} cut-off passages completed"
                        + (f", {len(headed)} of {len(headless)} list starts recovered"
                           if headless else ""),
                 summary=f"Continuation: {secs}; {len(completed)}/{len(clipped)} clipped hits "
                         "completed"
                         + (f"; {len(headed)}/{len(headless)} mid-list starts recovered"
                            if headless else ""))
        run.log("continuation", seconds=round(run.times[-1][1], 1),
                clipped=[c["hit_id"] for c in clipped], completed=completed,
                starts_mid_list=[c["hit_id"] for c in headless], start_recovered=headed)
    else:
        run.emit("stage_skip", stage="continuation", detail="No cut-off passages")
    # Sub-recipes an ingredient line names in capitals, searched before selection
    raw = raw + referenced_recipes(run, candidates, titles, depth, reuse["searches_fresh"])
    capture_primary(run, raw, candidates, titles, 1, depth, requirements, reuse["searches_fresh"])

    t = run.begin("select", "Selecting evidence", cli="selecting hits")
    selected, context_ids, coverage = select_with_reuse(run, question, depth, candidates, titles,
                                                        requirements, established, reused)
    secs = run.timed("Selector", t)
    need_shadow(run, requirements, candidates, titles, selected, context_ids, coverage,
                selector_primary=run.selector_fallback is None)
    # Unassessed (selector bypassed) is not an open need: without a selector nothing was judged
    # missing, and only reasoner-1's missing premises trigger repair then.
    open_needs = [c["requirement_id"] for c in coverage if c["status"] not in ("covered", "unassessed")]
    mode = run.trace["selector_mode"][-1] if run.trace["selector_mode"] else {}
    if trimmed(run, 1):
        run.done("select", secs, detail=f"Over the evidence budget: {len(selected)} passages "
                                         f"kept, {len(mode['dropped'])} lowest-ranked dropped",
                 summary=f"Selector: trimmed ({mode['candidate_chars']:,} chars over the "
                         f"{mode['budget']:,} budget); {len(selected)} unjudged hits kept, "
                         f"{len(mode['dropped'])} dropped")
    elif bypassed(run, 1):
        run.done("select", secs, detail=f"Selector bypassed: all {len(selected)} passages fit "
                                         "the evidence budget",
                 summary=f"Selector: bypassed ({mode['candidate_chars']:,} of "
                         f"{mode['budget']:,} budget chars); {len(selected)} unjudged hits passed")
    else:
        run.done("select", secs, detail=f"{len(selected)} " + ("passage" if len(selected) == 1 else "passages")
                 + f" kept, {len(context_ids)} need context",
                 summary=f"Selector: {secs}, {len(selected)} selected hits, "
                         f"{len(context_ids)} need context"
                         + (f"; not fully covered: {', '.join(open_needs)}" if open_needs else ""))
    run.log("selector", seconds=round(run.times[-1][1], 1), selected=len(selected),
            context_hits=len(context_ids), not_fully_covered=open_needs,
            mode=mode.get("mode"))

    t = run.begin("context", "Loading context")
    evidence, ev, sources = build_evidence(run, selected, context_ids, depth, titles)
    secs = run.timed("Context", t, ev["chars"])
    run.done("context", secs,
             detail=f"{ev['chars']:,} chars of evidence from {len(sources)} "
                    + ("source" if len(sources) == 1 else "sources"),
             summary=f"Context: {secs}; {len(ev['fulltext_fetched'])} fulltexts fetched, "
                     f"{len(ev['fulltext_from_cache'])} from cache; {ev['chars']:,} evidence chars "
                     f"(budget {ev['budget_chars']:,})")

    map_entries, map_text = evidence_map(requirements, coverage, selected, run.claims, titles)
    run.save("evidence-map-1.json", json.dumps(map_entries, indent=1, ensure_ascii=False))
    if bypassed(run, 1):
        map_text = None  # the map is built from selector labels; unjudged hits have none
    t = run.begin("reason", "Reasoning", cli="reasoning")
    secondary_text = secondary_evidence(run, secondary)
    parts = run.requested_parts

    # Coverage pre-check: with requested
    # parts, outside quick depth, COVERAGE_MODEL checks them against the evidence first. On any
    # "no" the follow-up round runs before the one answer pass, which must answer; a failed or
    # unparsable check goes straight to the answer pass.
    cov_check = None
    if parts and depth != "direct":
        t_cov = time.monotonic()
        run.checking(True)
        cov_check = coverage_check(run, question, evidence, parts, secondary_text)
        run.timed("Coverage check", t_cov)
        t = time.monotonic()  # the Reasoner row below times the answer pass alone
    fire, cov_queries, why = coverage_followup(cov_check or {}, parts, depth, 1, False,
                                               reuse["searches_fresh"])
    if parts and depth != "direct" and cov_check is None:
        why = "coverage check failed or unparsable"
    run.trace["coverage_followup"] = {
        "requested_parts": parts, "model": COVERAGE_MODEL if cov_check else None,
        "coverage_line": (cov_check or {}).get("coverage_line"),
        "decision_line": (cov_check or {}).get("decision_line"),
        "fired": fire, "trigger": why, "queries": cov_queries}
    run.log("coverage_followup", **run.trace["coverage_followup"])
    if not fire:
        run.checking(False)
    decision, forced = None, False
    if fire:
        missing = [part for part, mark in cov_check["coverage"] if mark == "no"]
        request = [{"requirement_id": "", "search": q,
                    "premise": f"not stated in the evidence: {', '.join(missing)}"}
                   for q in cov_queries]
    else:
        decision = reason(run, "reasoner-1", question, depth, evidence, requirements, coverage,
                          evidence_map=map_text, secondary=secondary_text,
                          stop_when=lambda p: p["decision"] == "search", requested_parts=parts,
                          told=already_told(run.related, candidates, evidence),
                          referenced=referenced_text(run, evidence))
        request = repair_request(run, decision, reuse["searches_fresh"], requirements)
        if not request and not decision["stopped"] and not bypassed(run, 1):
            # The reasoner answered although an exact detail it needs is missing but
            # retrievable: search for that detail instead of accepting an answer that may
            # supply it.
            request = forced_repair(requirements, coverage, decision.get("answer") or "",
                                    evidence + "\n\n" + user_text(run.question, run.standalone,
                                                                  run.previous, run.related),
                                    reuse["searches_fresh"])
            forced = bool(request)
            if forced:
                run.log("repair_forced", request=request)
    secs = run.timed("Reasoner", t)
    run.done("reason", secs, summary=f"Reasoner: {secs}"
             + (f"; coverage check: {len(cov_queries)} parts not found, follow-up first"
                if fire else "")
             + (f", requested {len(request)} follow-up searches"
                if request and not forced and not fire else "")
             + (f"; exact detail missing, {len(request)} targeted searches" if forced else ""))
    run.log("reasoner", seconds=round(run.times[-1][1], 1), research_request=request)

    repaired, final_selected = None, selected
    final, downgrades = [dict(c) for c in coverage], []
    if second_call_needed(decision, request):
        # One repair round, then a final reasoner call that cannot request another. A first
        # call stopped before its answer always gets the second call, even when every proposed
        # query was rejected as a repeat.
        t = run.begin("repair", "Repairing evidence", cli="follow-up search")
        run.writing(False)  # a first answer that streamed before this round is interrupted
        run.checking(True)
        if request:
            new_evidence, new_ev, new_sources, info, picked = repair(
                run, question, depth, request, candidates, selected, context_ids, titles)
        else:
            new_evidence, new_ev, new_sources, picked = None, None, None, []
            info = {"premises": [], "searches": [], "raw_hits": 0, "candidates": 0,
                    "selected": 0, "context_hits": 0, "coverage": [],
                    "error": "every proposed follow-up query repeated an earlier search"}
        if new_evidence is not None:
            evidence, ev, sources = new_evidence, new_ev, new_sources
        final_selected = selected + picked
        repaired = {"request": request, "forced": forced, **info}
        if fire:
            # Diagnostic, no model call: "no" entries that no follow-up passage even names.
            found = {stem(w) for h in picked for w in re.findall(
                r"[\w'’-]+", f"{h['text']} {h.get('continuation_text') or ''}".lower())}
            run.trace["coverage_followup"]["still_no"] = [
                part for part, mark in cov_check["coverage"]
                if mark == "no" and not subject_stems(part) <= found]
        final, downgrades = final_coverage(coverage, repaired)
        if downgrades:
            run.log("coverage_downgrade", downgrades=downgrades)
        map_entries, map_text = evidence_map(requirements, final + info["coverage"],
                                             final_selected, run.claims, titles)
        run.save("evidence-map-2.json", json.dumps(map_entries, indent=1, ensure_ascii=False))
        if bypassed(run, 1):
            map_text = None
        run.timed("Repair retrieval", t, ev["chars"] if new_evidence is not None else None)
        run.checking(False)
        t_final = time.monotonic()
        # The answer pass after a coverage follow-up is the run's only one (reasoner-1).
        decision = reason(run, "reasoner-2" if decision else "reasoner-1", question, depth,
                          evidence, requirements, final,
                          repaired, evidence_map=map_text, secondary=secondary_text,
                          requested_parts=parts, told=already_told(run.related, candidates, evidence),
                          referenced=referenced_text(run, evidence))
        run.timed("Reasoner (final)", t_final)
        repair_secs = time.monotonic() - t
        run.emit("stage_end", stage="repair", seconds=round(repair_secs, 1),
                 timed=f"{repair_secs:.1f}s",
                 detail=f"{info['selected']} new " + ("passage" if info["selected"] == 1 else "passages"),
                 summary=f"Repair: {repair_secs:.1f}s; {len(request)} follow-up searches, "
                         f"{info['candidates']} new candidates, {info['selected']} selected"
                         + (f"; search failed: {info['error']}" if info["error"] else ""))
        run.log("repair", seconds=round(repair_secs, 1), **info)
    else:
        run.emit("stage_skip", stage="repair", detail="Evidence sufficient")
    return {k: v for k, v in locals().items() if k in STATE_KEYS}


def _research(run):
    question = run.question
    if not question:
        raise ResearchError("empty question")
    run_start = run.started = time.monotonic()
    # The same identity as a completed run: return its result before auth or any provider or
    # model client is constructed. A follow-up is never reused or indexed: its words
    # mean something else in another conversation.
    # Serving is off (EXACT_REUSE_SERVE); the earlier answer is re-checked instead.
    lookup = (exact_lookup(question, run.fresh) if EXACT_REUSE_SERVE and not run.previous
              else {"hit": False, "key": result_identity(question)[0][:12],
                    "reason": "follow-up" if run.previous else "serving off (re-checked)"})
    if lookup["hit"]:
        return exact_reuse_result(run, lookup, run_start)
    start_log(run)
    run.log("exact_reuse", key=lookup["key"], outcome="miss", reason=lookup["reason"])
    if not run.previous:
        run.prior, why = prior_answer(question, run.fresh)
        run.trace["prior_answer"] = (
            {k: run.prior[k] for k in ("key", "source_run", "run_dir", "completed_at",
                                       "age_hours", "policy_version")}
            | {"found": True, "passages": len(run.prior.get("cited") or [])}
            if run.prior else {"found": False, "reason": why})
        run.log("prior_answer", **run.trace["prior_answer"])

    state = ask_research(run)
    if state is None:
        state = search_research(run, run.planned)
    (question, depth, requirements, search_plan, reuse, raw, candidates, candidate_chars, clipped,
     titles, selected, context_ids, coverage, final, downgrades, map_entries, decision,
     evidence, ev, sources, secondary, secondary_text, final_selected, repaired) = (
        state[k] for k in STATE_KEYS)

    answer = (decision.get("answer") or "").strip()
    if not answer:
        run.fail("reasoner returned an empty answer")
    run.save("answer.md", answer)
    run.writing(False)
    run.emit("answer", answer=answer)
    checked = map_entries
    if bypassed(run):
        # Unjudged hits carry no selector labels, so the map lists none of them; the reasoner's
        # synthesis may still cite any evidence hit (it decides what the answer used).
        checked = map_entries + [{"requirement_id": None, "relations": [], "changed": [],
                                  "statements": [{"hit_id": h["hit_id"], "scope": "unclear"}
                                                 for h in final_selected]}]
    synthesis, synthesis_issues = check_synthesis(decision, checked)
    # Grounding: quotes and numbers are checked against everything the answer pass read
    # (primary evidence and the community records); a quote found in neither is a failure.
    grounded_in = evidence + ("\n\n" + secondary_text if secondary_text else "")
    # Quotes and numbers the user wrote are theirs, not grounding failures
    grounded_in += "\n\n" + user_text(run.question, run.standalone, run.previous, run.related)
    checks = {"grounding_failures": ungrounded_quotes(answer, grounded_in),
              "unverified_numbers": ungrounded_quantities(answer, grounded_in),
              "ungrounded_percentages": ungrounded_percentages(answer, evidence),
              "threshold_phrases": threshold_phrases(answer, evidence),
              "unsourced_mechanism_verbs": unsourced_mechanism_verbs(answer, evidence),
              "synthesis_issues": synthesis_issues}
    # Each quotation must be in the passage it cites; unknown ids are flagged
    cite = citation_checks(answer, evidence, secondary_text)
    checks.update(citation_unknown_ids=cite["unknown_ids"],
                  citation_misattributed=cite["misattributed_quotes"],
                  citation_uncited_quotes=cite["uncited_quotes"],
                  **({"citation_in_short_answer": True} if cite["short_answer_cited"] else {}))
    citations = build_citations(answer, sources, secondary_text)
    for name, values in checks.items():
        if values:
            run.log(f"answer_{name}", values=values)
    total = time.monotonic() - run_start
    # The non-stage intervals, so the listed rows sum to Total.
    run.times.append(("Progress output", run.output_wait, None))
    run.times.append(("Other", total - sum(secs for _, secs, _ in run.times), None))
    stages = [{"stage": label, "seconds": round(secs, 1), "chars": chars}
              for label, secs, chars in run.times]
    run.log("timing", stages=stages, total_seconds=round(total, 1))
    costs = [u["cost_usd"] for u in run.usage if isinstance(u["cost_usd"], (int, float))]
    summary = dict(
        depth=depth, requirements=requirements, search_plan=search_plan, coverage=final,
        selector_coverage=coverage, coverage_downgrades=downgrades, evidence_map=map_entries,
        synthesis=synthesis, checks=checks,
        memory=memory_summary(run, final_selected, secondary, bool(secondary_text)), reuse=reuse,
        searches=len(reuse["searches_fresh"]), raw_hits=len(raw),
        candidates=len(candidates), candidate_chars=candidate_chars, selected=len(selected),
        context_hits=len(context_ids), repair=repaired, evidence_chars=ev["chars"],
        evidence_budget=ev["budget_chars"], context_reduced=len(ev["context_reduced"]),
        fulltexts_fetched=ev["fulltext_fetched"], fulltexts_from_cache=ev["fulltext_from_cache"],
        stage_seconds=stages, total_seconds=round(total, 1),
        first_answer_seconds=run.first_answer_seconds, claude_usage=run.usage,
        claude_cost_usd=round(sum(costs), 4) if costs else None,
        tokens_by_model=tokens_by_model(run.usage),
        exact_reuse={k: lookup[k] for k in ("hit", "key", "reason")},
        prior_answer=run.trace.get("prior_answer"),
        version={"app": APP_VERSION, "commit": CODE_COMMIT},
        follow_up=({"question": run.question, "standalone": run.standalone,
                    "previous_question": run.previous["question"],
                    "previous_run": run.previous["run_dir"],
                    "related_turn": (run.related or {}).get("turn_id"),
                    "related_question": (run.related or {}).get("question"),
                    "turns_offered": len(run.previous.get("turns") or []),
                    "previous_passages": len(previous_candidates(
                        run.previous, run.related, run.reuse == ANSWER_FROM_TURN)),
                    # "answer_from_turn" = answered from earlier research (no new search)
                    "reuse": run.reuse}
                   if run.previous else None),
        retrieval=run.trace["retrieval"],
        # The NotebookLM ask (wall time, citations, passages recovered and from memory,
        # follow-up ask, cache hit) and, when it could not be used, the fallback's reason
        ask=run.trace["ask"])
    run.log("summary", **summary)
    trace = evidence_trace(run, titles)
    trace["coverage_downgrades"] = downgrades
    trace["stage_usage"] = stage_usage(run.usage)
    result = {
        "question": question, "answer": answer, "depth": depth,
        "search_queries": reuse["searches_fresh"], "sources": sources, "citations": citations,
        "run_dir": str(run.dir),
        "details": {**summary, "clipped": len(clipped),
                    "continuation_incomplete": len(ev["continuation_incomplete"]),
                    "evidence_blocks": ev["blocks"], "sources": len(sources), "trace": trace},
    }
    # The compact turn record a later question in this thread is planned and answered with
    result["turn"] = turn_record(run.question, run.standalone, answer, citations, run.dir,
                                 run.previous["run_dir"] if run.previous else None, sources)
    run.save(TURN_FILE, json.dumps(result["turn"], indent=1, ensure_ascii=False))
    # Memory counters in the result and trace are the same objects, so they include this write.
    persist_research(run, depth, requirements, run.trace["candidates"], final, final_selected,
                     synthesis, answer, result, titles)
    write_need_ledger(run, final)
    run.save("evidence-trace.json", json.dumps(trace, indent=1, ensure_ascii=False))
    run.log("done")
    if not run.previous:
        index_result(run, lookup)
    return result
