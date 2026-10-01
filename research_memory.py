"""Persistent research memory: evidence and research relationships the pipeline has ALREADY found.

This is not a retriever. NotebookLM stays the primary semantic search over the primary sources;
nothing here reads, parses or indexes that corpus, and nothing here calls a model, summarizes, or
draws conclusions. Memory only remembers what earlier research runs discovered, so it need not be
rediscovered from scratch, and keeps every item's provenance:

  primary_retrieved  exact primary-source text that NotebookLM returned (never rewritten)
  secondary          community or research material (add_secondary, or structured community
                     claims through add_secondary_claim / import-secondary: claim type, context,
                     provenance, verification status and typed links to primary units)
  derived            an explicit inference or synthesis, always linked to its supporting units

Primary material stays authoritative for what the corpus author taught. A secondary claim is never
stored, searched or rendered as primary: it keeps its own layer, and "verified_primary" is only accepted
when the claim links to a primary unit that supports it.

Units are deduplicated only when they are the same text (whitespace-normalized) from the same
provenance; overlapping or contained passages stay separate units. Discoveries, selector
decisions and relationships are many-to-one histories on a unit. Frequency is metadata, not a
truth or authority score.

The database is SQLite (stdlib sqlite3, FTS5 over saved memory only) at data/research-memory.db
by default (override with --db or CRA_MEMORY_DB); it is git-ignored runtime data that can be
deleted and rebuilt by re-importing run logs. See docs/research-memory.md.

CLI:
  python research_memory.py init
  python research_memory.py import-trace logs/<run>/evidence-trace.json   (or the run folder,
                                         or a saved research result JSON)
  python research_memory.py search "query words" [--limit 10] [--layer primary_retrieved]
  python research_memory.py show <unit-id or unique prefix>
  python research_memory.py history [--min 2]
  python research_memory.py stats
  python research_memory.py import-secondary <records.jsonl>   (processed community claims)
  python research_memory.py secondary-schema                   (the record format, as JSON Schema)
"""
import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import time
import unicodedata
from pathlib import Path

import settings

ROOT = Path(__file__).resolve().parent
DEFAULT_DB = settings.DATA_DIR / "research-memory.db"
TITLES_CACHE = settings.CACHE_DIR / "source-titles.json"  # the pipeline's source id -> title cache
SCHEMA_VERSION = 7  # 2: secondary_claims; 3: research_needs; 4: units_fts indexes claim context;
                    # 5: units_fts indexes message text only (no headers, authors, topic labels)
                    # 6: cited_passages (+ cited_fts): what each cited passage was cited for
                    # 7: ask_ledger (+ ask_fts): fact questions asked and the passages cited

PRIMARY, SECONDARY, DERIVED = "primary_retrieved", "secondary", "derived"
LAYERS = (PRIMARY, SECONDARY, DERIVED)
ID_PREFIX = {PRIMARY: "P", SECONDARY: "S", DERIVED: "D"}
# Relationship types are free snake_case strings; these are the ones the design anticipates.
KNOWN_RELATIONSHIPS = ("supports", "derived_from", "inspired_by", "contrasts", "contradicts",
                       "clarifies", "later_addition_to", "replaces", "same_proposition_as",
                       "secondary_source_for")  # SECONDARY_LINKS are added below

# ---- secondary (community) claims -----------------------------------------------------------
# What a community claim is, relative to the primary teaching:
CLAIM_TYPES = (
    "primary_quote",               # a direct quotation of, or exact reference to, primary material
    "primary_paraphrase",          # primary material restated in the author's words
    "combined_primary_inference",  # a conclusion drawn by combining several primary statements
    "practical_synthesis",         # a practical protocol or recipe assembled from the teaching
    "correction_or_source_lead",   # corrects a claim, or points to where a primary source says it
    "anecdote_or_experiment",      # personal experience, trial, or observation
    "external_fact",               # a fact from outside the corpus (science, product, history)
    "novel_community_claim",       # a claim with no stated primary basis
)
# How far a claim has been checked against primary material (never a truth score for claims that
# are not about the primary teaching):
VERIFICATION = (
    "unverified",               # not checked
    "verified_primary",         # a linked primary unit supports it (requires such a link)
    "partially_verified",       # a linked primary unit supports part of it
    "contradicted_by_primary",  # a linked primary unit contradicts it (requires such a link)
    "not_in_primary",           # checked; the primary material retrieved so far does not address it
    "not_applicable",           # not a claim about the primary teaching (anecdote, external fact)
)
# Links from a secondary claim to primary (or derived) units. The first three need a primary unit.
SECONDARY_LINKS = ("quotes_primary", "paraphrases_primary", "infers_from_primary",
                   "supported_by", "contradicted_by", "cites_primary", "corrects")
PRIMARY_ONLY_LINKS = ("quotes_primary", "paraphrases_primary", "infers_from_primary")
KNOWN_RELATIONSHIPS = KNOWN_RELATIONSHIPS + SECONDARY_LINKS
NEEDS_LINK = {"verified_primary": ("quotes_primary", "paraphrases_primary", "supported_by"),
              "partially_verified": ("quotes_primary", "paraphrases_primary", "infers_from_primary",
                                     "supported_by"),
              "contradicted_by_primary": ("contradicted_by",)}
NAME = re.compile(r"^[a-z][a-z0-9_]*$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  question TEXT,
  depth TEXT,
  started_at TEXT,
  imported_from TEXT,
  imported_at TEXT,
  metadata TEXT                      -- JSON: e.g. coverage, answer path, artifact versions
);

CREATE TABLE IF NOT EXISTS run_requirements (
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  requirement_id TEXT NOT NULL,
  kind TEXT,
  text TEXT,
  PRIMARY KEY (run_id, requirement_id)
);

CREATE TABLE IF NOT EXISTS run_searches (
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  round INTEGER NOT NULL,            -- 1 = planner searches, 2 = repair follow-ups
  position INTEGER NOT NULL,
  query TEXT NOT NULL,
  covers TEXT,                       -- JSON list of requirement ids, NULL when unknown
  PRIMARY KEY (run_id, round, position)
);

CREATE TABLE IF NOT EXISTS units (
  unit_id TEXT PRIMARY KEY,
  layer TEXT NOT NULL CHECK (layer IN ('primary_retrieved', 'secondary', 'derived')),
  text TEXT NOT NULL,                -- exact text as received; never rewritten
  fingerprint TEXT NOT NULL,         -- sha256 of the whitespace-normalized text
  provenance_key TEXT NOT NULL,      -- what makes the same text a different occurrence
  source_id TEXT,
  source_title TEXT,
  source_date TEXT,
  platform TEXT,                     -- secondary: e.g. forum, chat, article; primary: NULL
  author TEXT,
  url TEXT,
  kind TEXT,                         -- derived: e.g. combined_inference, source_lead
  first_seen_at TEXT,
  created_at TEXT NOT NULL,
  metadata TEXT,
  UNIQUE (layer, provenance_key, fingerprint)
);

CREATE TABLE IF NOT EXISTS discoveries (
  id INTEGER PRIMARY KEY,
  unit_id TEXT NOT NULL REFERENCES units(unit_id),
  run_id TEXT REFERENCES runs(run_id),
  round INTEGER,
  query TEXT,
  raw_rank INTEGER,
  provider TEXT NOT NULL,            -- e.g. notebooklm
  raw_ref TEXT,                      -- the run's own id for the raw hit, when known
  observed_at TEXT,
  metadata TEXT                      -- JSON: offsets, merge fate, candidate id, continuation
);
CREATE UNIQUE INDEX IF NOT EXISTS discoveries_once ON discoveries
  (unit_id, ifnull(run_id, ''), ifnull(round, 0), ifnull(query, ''), ifnull(raw_ref, ''));

CREATE TABLE IF NOT EXISTS selections (
  id INTEGER PRIMARY KEY,
  unit_id TEXT NOT NULL REFERENCES units(unit_id),
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  round INTEGER NOT NULL,
  hit_id TEXT NOT NULL,              -- the run's candidate id (h1, h2, ...)
  role TEXT,                         -- CORE / CONTRAST / SUPPORT / DROP, NULL when not recorded
  covers TEXT,                       -- JSON list of requirement ids, NULL when not recorded
  reason TEXT,
  kept INTEGER,                      -- 1/0, NULL when not recorded
  context_requested INTEGER,
  final_evidence INTEGER,            -- reached the reasoner's evidence: 1/0, NULL when unknown
  metadata TEXT,
  UNIQUE (unit_id, run_id, round, hit_id)
);

CREATE TABLE IF NOT EXISTS relationships (
  id INTEGER PRIMARY KEY,
  from_unit TEXT NOT NULL REFERENCES units(unit_id),
  rel_type TEXT NOT NULL,
  to_unit TEXT NOT NULL REFERENCES units(unit_id),
  note TEXT,
  created_at TEXT NOT NULL,
  metadata TEXT,
  UNIQUE (from_unit, rel_type, to_unit)
);

-- One row per structured secondary claim (a unit of layer 'secondary'). The unit's text is the
-- claim text exactly as extracted; the record identity says where it came from.
CREATE TABLE IF NOT EXISTS secondary_claims (
  unit_id TEXT PRIMARY KEY REFERENCES units(unit_id),
  record_id TEXT NOT NULL,           -- stable identity in the source, e.g. chat:<group>/<msg id>
  community TEXT,                    -- channel, group, board, site section
  thread_ref TEXT,                   -- thread / parent message reference, when known
  claim_type TEXT NOT NULL,
  context TEXT,                      -- surrounding message or thread text, for interpretation
  verification TEXT NOT NULL DEFAULT 'unverified',
  verification_note TEXT,
  verified_at TEXT,
  primary_refs TEXT,                 -- JSON: cited primary material not (yet) in memory
  processed_by TEXT,                 -- the processing step/version that produced the record
  processed_at TEXT
);
CREATE INDEX IF NOT EXISTS secondary_claims_record ON secondary_claims (record_id);

-- Research-need ledger: per research need (the planner's case frame, see
-- research_need.py), selector policy and corpus identity, the latest completed primary-selector
-- selection. A coverage record for the shadow reuse gate, never proof of exhaustive search.
-- Units are primary unit ids (source identity + text fingerprint).
CREATE TABLE IF NOT EXISTS research_needs (
  need_id TEXT PRIMARY KEY,          -- hash of (signature, selector_policy, corpus)
  signature TEXT NOT NULL,
  signature_version INTEGER NOT NULL,
  subject TEXT NOT NULL,             -- normalized frame subject, for related-need lookup
  frame TEXT NOT NULL,               -- JSON: normalized case frame
  selector_policy TEXT NOT NULL,
  corpus TEXT NOT NULL,              -- JSON: {notebooks, epoch, manifest, strength}
  requirements TEXT,                 -- JSON: the plan's requirements
  facets TEXT,                       -- JSON: requirement kinds
  input_units TEXT NOT NULL,         -- JSON: selector-input pool as unit ids
  selection TEXT,                    -- JSON: round-1 selection with hit ids as unit ids; NULL = unmappable
  selector_primary INTEGER NOT NULL, -- 1: the configured primary selector (not the fallback)
  source_run TEXT,
  recorded_at REAL NOT NULL          -- epoch seconds
);
CREATE INDEX IF NOT EXISTS research_needs_subject ON research_needs (subject);

-- Cited passages: a primary passage an answer cited (or, for older runs, one its
-- synthesis/evidence map used), with the question, its standalone rewrite and the requirement
-- it was cited for. A later question on the same topic finds the passage through this text
-- even when its own searches miss it. One row per (unit, run, requirement); '' = none known.
CREATE TABLE IF NOT EXISTS cited_passages (
  unit_id TEXT NOT NULL REFERENCES units(unit_id),
  run_id TEXT NOT NULL,
  question TEXT,
  standalone TEXT,
  requirement TEXT NOT NULL DEFAULT '',
  origin TEXT,                       -- citation / backfill_citation / backfill_used
  recorded_at TEXT,
  UNIQUE (unit_id, run_id, requirement)
);
CREATE VIRTUAL TABLE IF NOT EXISTS cited_fts USING fts5(
  unit_id UNINDEXED, question, standalone, requirement, tokenize = 'porter unicode61'
);

-- Ask ledger (fact memory): one row per fact question asked of NotebookLM, keyed by its
-- spelling-normalized form, with the primary passages NotebookLM cited for it and its short
-- reply. A later run's matching question reuses the passages instead of asking again.
CREATE TABLE IF NOT EXISTS ask_ledger (
  ledger_id INTEGER PRIMARY KEY,
  normalized TEXT NOT NULL UNIQUE,   -- the memory key (research.ledger_key)
  question TEXT NOT NULL,            -- as last sent to NotebookLM
  unit_ids TEXT NOT NULL,            -- JSON list of primary unit ids cited for it
  reply TEXT,                        -- NotebookLM's short reply, citation marks removed
  run_id TEXT,
  recorded_at TEXT,
  policy_version TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS ask_fts USING fts5(
  ledger_id UNINDEXED, normalized, tokenize = 'porter unicode61'
);

"""
# The full-text index is derived data (rebuilt from units + secondary_claims, never a source of
# truth). v4 adds `context`: a secondary claim's surrounding message/thread text, so a claim whose
# one-line text lacks the question's words is still reachable through its context. v5 indexes only
# the message text: not a secondary unit's source_title (a chat/topic label) nor the context's
# header lines, "[reply context]" markers and "[msg ids] Author:" prefixes (see message_body).
FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS units_fts USING fts5(
  unit_id UNINDEXED, layer UNINDEXED, source_title, text, context, tokenize = 'porter unicode61'
);
"""
FTS_COLUMNS_TEXT = "{source_title text}"  # FTS5 column filter: a match outside the context
CONTEXT_WEIGHT = 0.5  # a query term found only in a secondary claim's context counts half
FTS_INDEX_VERSION = 5  # an index built before this schema version is rebuilt from stored rows on init
CONTEXT_HEADER = re.compile(r"^\[[^\]\n]* · [^\]\n]*\]$")  # [Chat · member · <date> · msg 61]
CONTEXT_MARKER = re.compile(r"^\[reply context\]$")
CONTEXT_SPEAKER = re.compile(r"^\[\d+(?:,\d+)*\] [^:\n]{1,120}: ?")  # "[5742] member: " before a message


def message_body(context):
    """The message text of a secondary claim's stored context, for indexing: header lines,
    "[reply context]" markers and "[msg ids] Author:" prefixes removed. The stored context is
    unchanged."""
    lines = []
    for line in (context or "").split("\n"):
        if CONTEXT_HEADER.match(line.strip()) or CONTEXT_MARKER.match(line.strip()):
            continue
        lines.append(CONTEXT_SPEAKER.sub("", line))
    return "\n".join(lines).strip()


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ---- deterministic relevance (no model, no embeddings) ----------------------------------------
# Words that carry no subject in a research question. Generic English and question words only;
# nothing here is tuned to particular claims.
STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been before being
below between both but by can could did do does doing done down during each else ever few for
from further get gets got had has have having he her here hers him his how i if in into is it its
itself just me more most my no nor not now of off on once only or other our out over own same she
should so some such than that the their them then there these they this those through to too
under until up very was we were what when where which while who whom why will with would you your
yours author instructor he's said say says saying tell told think thought know want need
like way ways thing things good bad best better much many lot lots really actually please
make makes made making use uses used using work works worked happen happens mean means give gives
go goes come comes put take takes
""".split())
MIN_TERM_LENGTH = 3


def query_terms(text):
    """The distinct content words of a query, in order (lowercase, stopwords removed)."""
    words = re.findall(r"\w+", (text or "").lower(), re.UNICODE)
    return list(dict.fromkeys(w for w in words if len(w) >= MIN_TERM_LENGTH and w not in STOPWORDS
                              and not w.isdigit()))


def normalize(text):
    """The comparison form of a text: Unicode NFC with whitespace collapsed. Only used for the
    fingerprint; the stored text is always the exact original."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def fingerprint(text):
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def primary_key(source_id, source_title):
    """What distinguishes occurrences of primary text: the source id, else its title."""
    if source_id:
        return f"source:{source_id}"
    return f"title:{source_title}" if source_title else "unknown-source"


def unit_id_for(layer, provenance_key, fp):
    """Stable id: the same layer, provenance and text always get the same id, in any database."""
    digest = hashlib.sha256(f"{layer}\x00{provenance_key}\x00{fp}".encode("utf-8")).hexdigest()
    return f"{ID_PREFIX[layer]}-{digest[:16]}"


def dumps(value):
    return None if value is None else json.dumps(value, ensure_ascii=False, sort_keys=True)


def loads(value):
    return None if value is None else json.loads(value)


def flag(value):
    return None if value is None else int(bool(value))


class MemoryStore:
    """The research memory database. Every write is idempotent: re-recording the same unit,
    discovery, selection or relationship adds nothing new (missing fields may be filled in)."""

    def __init__(self, path=None):
        self.path = Path(path or os.environ.get("CRA_MEMORY_DB") or DEFAULT_DB)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.create_function("message_body", 1, message_body, deterministic=True)
        self.init()

    def init(self):
        """Create any missing tables; never drops or rewrites existing data."""
        with self.db:
            self.db.executescript(SCHEMA)  # new tables only: an older database gains them
            fts = [r[1] for r in self.db.execute("PRAGMA table_info(units_fts)")]
            version = self.db.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            # an index from before FTS_INDEX_VERSION: rebuild from stored rows
            stale = bool(fts) and ("context" not in fts or version is None
                                   or int(version[0]) < FTS_INDEX_VERSION)
            if stale:
                self.db.execute("DROP TABLE units_fts")
            self.db.executescript(FTS_SCHEMA)
            if stale:
                self._fill_search_index()
            self.db.execute("INSERT OR IGNORE INTO meta VALUES ('schema_version', ?)",
                            (str(SCHEMA_VERSION),))
            self.db.execute("UPDATE meta SET value = ? WHERE key = 'schema_version' "
                            "AND CAST(value AS INTEGER) < ?", (str(SCHEMA_VERSION), SCHEMA_VERSION))

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *exc):
        # Commit what the block wrote before closing: close() rolls back writes made outside a
        # `with self.db`.
        try:
            if exc_type is None:
                self.db.commit()
        finally:
            self.close()

    # ---- runs --------------------------------------------------------------------------------

    def add_run(self, run_id, question=None, depth=None, started_at=None, imported_from=None,
                metadata=None, requirements=(), searches=()):
        """Record a research run (fields already stored are kept; missing ones are filled in).
        requirements: [{"id", "kind", "text"}]. searches: [{"query", "covers"?, "round"?}]."""
        with self.db:
            self.db.execute(
                """INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (run_id) DO UPDATE SET
                     question = coalesce(runs.question, excluded.question),
                     depth = coalesce(runs.depth, excluded.depth),
                     started_at = coalesce(runs.started_at, excluded.started_at),
                     imported_from = coalesce(excluded.imported_from, runs.imported_from),
                     imported_at = excluded.imported_at,
                     metadata = coalesce(excluded.metadata, runs.metadata)""",
                (run_id, question, depth, started_at, imported_from, now(), dumps(metadata)))
            for r in requirements:
                self.db.execute("INSERT OR IGNORE INTO run_requirements VALUES (?, ?, ?, ?)",
                                (run_id, r["id"], r.get("kind"), r.get("text")))
            positions = {}
            for s in searches:
                rnd = s.get("round", 1)
                positions[rnd] = positions.get(rnd, 0) + 1
                covers = s.get("covers")
                self.db.execute("INSERT OR IGNORE INTO run_searches VALUES (?, ?, ?, ?, ?)",
                                (run_id, rnd, positions[rnd], s["query"],
                                 dumps(covers) if covers is not None else None))
        return run_id

    # ---- units -------------------------------------------------------------------------------

    def _add_unit(self, layer, text, provenance_key, seen_at=None, **fields):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("a unit needs non-empty text")
        fp = fingerprint(text)
        uid = unit_id_for(layer, provenance_key, fp)
        cols = ("source_id", "source_title", "source_date", "platform", "author", "url", "kind")
        values = [fields.get(c) for c in cols]
        with self.db:
            new = self.db.execute(
                f"""INSERT OR IGNORE INTO units (unit_id, layer, text, fingerprint, provenance_key,
                      {', '.join(cols)}, first_seen_at, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, {', '.join('?' * len(cols))}, ?, ?, ?)""",
                (uid, layer, text, fp, provenance_key, *values, seen_at, now(),
                 dumps(fields.get("metadata")))).rowcount
            if new:
                self.db.execute("INSERT INTO units_fts VALUES (?, ?, ?, ?, '')",
                                (uid, layer, (layer != SECONDARY and fields.get("source_title")) or "",
                                 text))
            else:  # fill in what an earlier record lacked; never change the text
                row = self.db.execute("SELECT * FROM units WHERE unit_id = ?", (uid,)).fetchone()
                updates = {c: v for c, v in zip(cols, values) if v is not None and row[c] is None}
                if seen_at and (row["first_seen_at"] is None or seen_at < row["first_seen_at"]):
                    updates["first_seen_at"] = seen_at
                if updates:
                    self.db.execute(f"UPDATE units SET {', '.join(f'{c} = ?' for c in updates)} "
                                    "WHERE unit_id = ?", (*updates.values(), uid))
                if "source_title" in updates and layer != SECONDARY:
                    self.db.execute("UPDATE units_fts SET source_title = ? WHERE unit_id = ?",
                                    (updates["source_title"], uid))
        return uid, bool(new)

    def add_primary(self, text, source_id=None, source_title=None, source_date=None,
                    seen_at=None, metadata=None):
        """Store an exact primary passage retrieved from NotebookLM. The same text from the same
        source is one unit; the same text from a different source stays a separate unit."""
        return self._add_unit(PRIMARY, text, primary_key(source_id, source_title), seen_at,
                              source_id=source_id,
                              source_title=source_title, source_date=source_date,
                              metadata=metadata)[0]

    def record_retrieval(self, text, run_id=None, query=None, raw_rank=None, round=None,
                         provider="notebooklm", raw_ref=None, source_id=None, source_title=None,
                         source_date=None, observed_at=None, metadata=None):
        """Store a retrieved primary passage (once) and record this discovery of it."""
        uid = self.add_primary(text, source_id, source_title, source_date, observed_at)
        self.add_discovery(uid, run_id, query, raw_rank, round, provider, raw_ref, observed_at,
                           metadata)
        return uid

    def add_discovery(self, unit_id, run_id=None, query=None, raw_rank=None, round=None,
                      provider="notebooklm", raw_ref=None, observed_at=None, metadata=None):
        with self.db:
            return bool(self.db.execute(
                """INSERT OR IGNORE INTO discoveries (unit_id, run_id, round, query, raw_rank,
                     provider, raw_ref, observed_at, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (unit_id, run_id, round, query, raw_rank, provider, raw_ref, observed_at,
                 dumps(metadata))).rowcount)

    def add_selection(self, unit_id, run_id, round, hit_id, role=None, covers=None, reason=None,
                      kept=None, context_requested=None, final_evidence=None, metadata=None):
        """Record one selector judgment of a unit in one run. Unknown fields stay NULL; a later
        record for the same run/round/hit fills in fields that were NULL."""
        with self.db:
            self.db.execute(
                """INSERT INTO selections (unit_id, run_id, round, hit_id, role, covers, reason,
                     kept, context_requested, final_evidence, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (unit_id, run_id, round, hit_id) DO UPDATE SET
                     role = coalesce(selections.role, excluded.role),
                     covers = coalesce(selections.covers, excluded.covers),
                     reason = coalesce(selections.reason, excluded.reason),
                     kept = coalesce(selections.kept, excluded.kept),
                     context_requested = coalesce(selections.context_requested,
                                                  excluded.context_requested),
                     final_evidence = coalesce(selections.final_evidence, excluded.final_evidence),
                     metadata = coalesce(selections.metadata, excluded.metadata)""",
                (unit_id, run_id, round, hit_id, role, dumps(covers), reason, flag(kept),
                 flag(context_requested), flag(final_evidence), dumps(metadata)))

    def add_secondary(self, text, platform, source_title=None, author=None, url=None,
                      source_date=None, source_id=None, metadata=None):
        """Store community or research material. It is always layer 'secondary' and is never
        presented as primary evidence."""
        if not platform:
            raise ValueError("secondary material needs a platform (where it came from)")
        where = source_id or url or source_title or author or "unspecified"
        return self._add_unit(SECONDARY, text, f"{platform}:{where}", source_id=source_id,
                              source_title=source_title, source_date=source_date,
                              platform=platform, author=author, url=url, metadata=metadata)[0]

    def add_secondary_claim(self, claim_text, platform, record_id, claim_type, context=None,
                            community=None, thread_ref=None, author=None, url=None,
                            source_date=None, verification="unverified", verification_note=None,
                            links=(), primary_refs=(), processed_by=None, metadata=None):
        """Store one structured community claim (layer 'secondary').

        claim_text is the claim exactly as extracted; record_id identifies the message or post it
        came from (one record may hold several claims). links: [{"relation", "unit_id"}] to
        existing units (SECONDARY_LINKS; quotes/paraphrases/infers-from need a primary unit).
        primary_refs: cited primary material that is not in memory yet, e.g. {"source_title",
        "source_date", "quote"}. The verification status is checked against the links (see
        set_verification). Re-importing the same record and claim changes nothing new. Returns the
        unit id."""
        if claim_type not in CLAIM_TYPES:
            raise ValueError(f"claim_type must be one of {', '.join(CLAIM_TYPES)}")
        if verification not in VERIFICATION:
            raise ValueError(f"verification must be one of {', '.join(VERIFICATION)}")
        if not record_id:
            raise ValueError("a secondary claim needs the record_id it came from")
        uid = self._add_unit(SECONDARY, claim_text, f"{platform}:{record_id}", source_date=source_date,
                             source_id=record_id, source_title=community, platform=platform,
                             author=author, url=url, kind=claim_type, metadata=metadata)[0]
        with self.db:
            self.db.execute(
                """INSERT INTO secondary_claims (unit_id, record_id, community, thread_ref,
                     claim_type, context, primary_refs, processed_by, processed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (unit_id) DO UPDATE SET
                     community = coalesce(secondary_claims.community, excluded.community),
                     thread_ref = coalesce(secondary_claims.thread_ref, excluded.thread_ref),
                     context = coalesce(secondary_claims.context, excluded.context),
                     primary_refs = coalesce(excluded.primary_refs, secondary_claims.primary_refs),
                     processed_by = coalesce(excluded.processed_by, secondary_claims.processed_by),
                     processed_at = excluded.processed_at""",
                (uid, record_id, community, thread_ref, claim_type, context,
                 dumps(list(primary_refs)) if primary_refs else None, processed_by, now()))
            self.db.execute("UPDATE units_fts SET context = (SELECT message_body(context) FROM "
                            "secondary_claims WHERE unit_id = ?) WHERE unit_id = ?", (uid, uid))
        for link in links:
            self.link_secondary(uid, link["relation"], link["unit_id"], link.get("note"))
        if verification != "unverified":
            self.set_verification(uid, verification, verification_note)
        return uid

    def link_secondary(self, secondary_uid, relation, target_uid, note=None):
        """Link a secondary claim to the unit it quotes, paraphrases, infers from, is supported
        or contradicted by, cites, or corrects."""
        if self.layer(secondary_uid) != SECONDARY:
            raise ValueError(f"{secondary_uid} is not a secondary unit")
        if relation not in SECONDARY_LINKS:
            raise ValueError(f"relation must be one of {', '.join(SECONDARY_LINKS)}")
        target = self.layer(target_uid)
        if target is None:
            raise ValueError(f"unknown unit {target_uid}")
        if relation in PRIMARY_ONLY_LINKS and target != PRIMARY:
            raise ValueError(f"{relation} must point to a primary unit")
        self.add_relationship(secondary_uid, relation, target_uid, note)

    def set_verification(self, secondary_uid, status, note=None):
        """Record how far a secondary claim has been checked against primary material. A status
        that asserts primary support or contradiction needs a matching link to a primary unit,
        so a community claim cannot become 'verified' without the evidence that verifies it."""
        if status not in VERIFICATION:
            raise ValueError(f"verification must be one of {', '.join(VERIFICATION)}")
        if self.db.execute("SELECT 1 FROM secondary_claims WHERE unit_id = ?",
                           (secondary_uid,)).fetchone() is None:
            raise ValueError(f"{secondary_uid} is not a structured secondary claim")
        needed = NEEDS_LINK.get(status)
        if needed and not self.db.execute(
                f"""SELECT 1 FROM relationships r JOIN units u ON u.unit_id = r.to_unit
                    WHERE r.from_unit = ? AND u.layer = 'primary_retrieved'
                      AND r.rel_type IN ({', '.join('?' * len(needed))})""",
                (secondary_uid, *needed)).fetchone():
            raise ValueError(f"{status} needs a {' or '.join(needed)} link to a primary unit")
        with self.db:
            self.db.execute("UPDATE secondary_claims SET verification = ?, verification_note = ?, "
                            "verified_at = ? WHERE unit_id = ?", (status, note, now(), secondary_uid))

    def find_quoted_primary(self, secondary_uid):
        """Primary units that contain the claim's text verbatim (whitespace-normalized): candidates
        for a quotes_primary link. Nothing is linked or verified automatically."""
        row = self.db.execute("SELECT text FROM units WHERE unit_id = ?", (secondary_uid,)).fetchone()
        if row is None:
            return []
        needle = normalize(row["text"]).strip(' "“”\'').lower()
        if len(needle) < 12:
            return []
        return [r["unit_id"] for r in self.db.execute(
            "SELECT unit_id, text FROM units WHERE layer = 'primary_retrieved' ORDER BY unit_id")
            if needle in normalize(r["text"]).lower()]

    def secondary_claim(self, unit_id):
        """A structured secondary claim with its record identity, verification and links."""
        row = self.db.execute(
            """SELECT u.unit_id, u.text AS claim_text, u.platform, u.author, u.url, u.source_date,
                      c.record_id, c.community, c.thread_ref, c.claim_type, c.context,
                      c.verification, c.verification_note, c.verified_at, c.primary_refs,
                      c.processed_by, c.processed_at
               FROM secondary_claims c JOIN units u USING (unit_id) WHERE unit_id = ?""",
            (unit_id,)).fetchone()
        if row is None:
            return None
        claim = dict(row, primary_refs=loads(row["primary_refs"]) or [])
        claim["links"] = [dict(r) for r in self.db.execute(
            """SELECT r.rel_type AS relation, r.to_unit AS unit_id, u.layer, r.note
               FROM relationships r JOIN units u ON u.unit_id = r.to_unit
               WHERE r.from_unit = ? ORDER BY r.rel_type, r.to_unit""", (unit_id,))]
        return claim

    def add_derived(self, text, kind, supports, inspired_by=(), producer="manual", metadata=None):
        """Store an explicit inference or synthesis. It must name the units that support it
        (primary or derived, linked as derived_from) and may name secondary material that
        inspired it (linked as inspired_by). Nothing here generates or checks the claim itself."""
        if not NAME.match(kind or ""):
            raise ValueError("derived kind must be snake_case, e.g. combined_inference")
        supports, inspired_by = list(dict.fromkeys(supports)), list(dict.fromkeys(inspired_by))
        if not supports:
            raise ValueError("a derived unit needs at least one supporting unit")
        for uid in supports:
            if self.layer(uid) not in (PRIMARY, DERIVED):
                raise ValueError(f"supporting unit {uid} must be an existing primary or derived unit")
        for uid in inspired_by:
            if self.layer(uid) != SECONDARY:
                raise ValueError(f"inspiration {uid} must be an existing secondary unit")
        uid = self._add_unit(DERIVED, text, f"producer:{producer}", kind=kind,
                             metadata=metadata)[0]
        for target in supports:
            self.add_relationship(uid, "derived_from", target)
        for target in inspired_by:
            self.add_relationship(uid, "inspired_by", target)
        return uid

    def add_relationship(self, from_unit, rel_type, to_unit, note=None, metadata=None):
        if not NAME.match(rel_type or ""):
            raise ValueError("relationship type must be snake_case, e.g. later_addition_to")
        for uid in (from_unit, to_unit):
            if self.layer(uid) is None:
                raise ValueError(f"unknown unit {uid}")
        if from_unit == to_unit:
            raise ValueError("a unit cannot relate to itself")
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO relationships (from_unit, rel_type, to_unit, note, "
                "created_at, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                (from_unit, rel_type, to_unit, note, now(), dumps(metadata)))

    # ---- reading -----------------------------------------------------------------------------

    def layer(self, unit_id):
        row = self.db.execute("SELECT layer FROM units WHERE unit_id = ?", (unit_id,)).fetchone()
        return row["layer"] if row else None

    def resolve(self, prefix):
        """A full unit id from an id or unique prefix."""
        rows = self.db.execute("SELECT unit_id FROM units WHERE unit_id LIKE ? LIMIT 2",
                               (prefix.replace("%", "") + "%",)).fetchall()
        if len(rows) != 1:
            raise KeyError(f"{'no' if not rows else 'more than one'} unit matches {prefix!r}")
        return rows[0]["unit_id"]

    def unit(self, unit_id):
        """Everything memory holds about one unit: the unit, its discoveries, selector history
        (with the requirement texts it covered) and relationships in both directions."""
        row = self.db.execute("SELECT * FROM units WHERE unit_id = ?", (unit_id,)).fetchone()
        if row is None:
            return None
        unit = dict(row, metadata=loads(row["metadata"]))
        unit["discoveries"] = [dict(r, metadata=loads(r["metadata"])) for r in self.db.execute(
            """SELECT d.run_id, d.round, d.query, d.raw_rank, d.provider, d.raw_ref, d.observed_at,
                      d.metadata, r.question
               FROM discoveries d LEFT JOIN runs r USING (run_id)
               WHERE d.unit_id = ? ORDER BY d.observed_at, d.id""", (unit_id,))]
        unit["selections"] = []
        for r in self.db.execute(
                """SELECT s.*, r.question FROM selections s LEFT JOIN runs r USING (run_id)
                   WHERE s.unit_id = ? ORDER BY r.started_at, s.id""", (unit_id,)):
            covers = loads(r["covers"])
            texts = {q["requirement_id"]: q["text"] for q in self.db.execute(
                "SELECT requirement_id, text FROM run_requirements WHERE run_id = ?", (r["run_id"],))}
            unit["selections"].append({
                **{k: r[k] for k in ("run_id", "round", "hit_id", "role", "reason", "kept",
                                     "context_requested", "final_evidence", "question")},
                "covers": covers, "metadata": loads(r["metadata"]),
                "covered_requirements": [{"id": c, "text": texts.get(c)} for c in covers or []]})
        unit["relationships"] = [dict(r) for r in self.db.execute(
            """SELECT 'out' AS direction, rel_type, to_unit AS other, note FROM relationships
               WHERE from_unit = ?
               UNION ALL
               SELECT 'in', rel_type, from_unit, note FROM relationships WHERE to_unit = ?
               ORDER BY 1, 2, 3""", (unit_id, unit_id))]
        return unit

    def search(self, query, limit=10, layer=None):
        """Full-text search over SAVED memory only (retrieved passages, secondary material,
        derived claims): all words first, any word if nothing matches all of them. Each result
        carries its layer and provenance."""
        words = re.findall(r"\w+", query, re.UNICODE)
        if not words:
            return []
        results = []
        for joiner in (" AND ", " OR "):
            match = joiner.join(f'"{w}"' for w in words)
            sql = ("""SELECT f.unit_id, f.layer, bm25(units_fts) AS score,
                        snippet(units_fts, 3, '[', ']', '…', 14) AS snippet,
                        u.source_title, u.source_id, u.source_date, u.platform, u.author, u.kind,
                        (SELECT count(*) FROM discoveries d WHERE d.unit_id = f.unit_id) AS discoveries,
                        (SELECT count(DISTINCT run_id) FROM discoveries d
                           WHERE d.unit_id = f.unit_id) AS runs
                      FROM units_fts f JOIN units u USING (unit_id)
                      WHERE units_fts MATCH ?""" + (" AND f.layer = ?" if layer else "")
                   + " ORDER BY score LIMIT ?")
            params = [match] + ([layer] if layer else []) + [limit]
            results = [dict(r) for r in self.db.execute(sql, params)]
            if results or len(words) == 1:
                break
        return results

    def relevant(self, query, layer, limit=5, min_score=0.5):
        """Units of one layer relevant to `query`, best first (deterministic).

        FTS (the same porter-stemmed index as `search`) finds every unit containing any query
        term, in its text or (secondary claims) its context. Each unit is scored by the share of
        the query's informative term weight it covers, a term's weight being its inverse document
        frequency within the layer, so a unit matching only a common word scores low and one
        matching the distinctive words scores high. A term that no unit contains keeps the highest
        weight, so a query mostly about something memory lacks does not turn a partial match into
        a strong one. A term found only in a claim's context counts CONTEXT_WEIGHT of its weight.
        Kept: score >= min_score and at least two matched terms (one for a one-term query).
        Returns [{"unit_id", "score", "matched", "context_only"}]."""
        terms = query_terms(query)
        if not terms:
            return []
        total = self.db.execute("SELECT count(*) FROM units WHERE layer = ?", (layer,)).fetchone()[0]
        if not total:
            return []
        sql = "SELECT unit_id FROM units_fts WHERE units_fts MATCH ? AND layer = ?"
        hits, weight = {}, {}
        for term in terms:
            ids = {r[0] for r in self.db.execute(sql, (f'"{term}"', layer))}
            weight[term] = math.log((total + 1) / (len(ids) + 0.5))
            if not ids:
                continue
            in_text = {r[0] for r in self.db.execute(sql, (f'{FTS_COLUMNS_TEXT} : "{term}"', layer))}
            for uid in ids:
                hits.setdefault(uid, {})[term] = 1.0 if uid in in_text else CONTEXT_WEIGHT
        full = sum(weight.values()) or 1.0
        need = min(2, len(terms))
        scored = [{"unit_id": uid, "score": round(sum(weight[t] * f for t, f in m.items()) / full, 4),
                   "matched": list(m), "context_only": [t for t, f in m.items() if f < 1.0]}
                  for uid, m in hits.items() if len(m) >= need]
        scored = [x for x in scored if x["score"] >= min_score]
        scored.sort(key=lambda x: (-x["score"], -len(x["matched"]), x["unit_id"]))
        return scored[:limit]

    def add_citation(self, unit_id, run_id, question, standalone, requirement="", origin="citation"):
        """Record that `unit_id` was cited in run `run_id` for `requirement` (idempotent)."""
        requirement = requirement or ""
        with self.db:
            cur = self.db.execute("INSERT OR IGNORE INTO cited_passages VALUES (?, ?, ?, ?, ?, ?, ?)",
                                  (unit_id, run_id, question, standalone, requirement, origin, now()))
            if cur.rowcount:
                self.db.execute("INSERT INTO cited_fts VALUES (?, ?, ?, ?)",
                                (unit_id, question or "", standalone or "", requirement))
        return bool(cur.rowcount)

    def add_ask_fact(self, normalized, question, unit_ids, reply, run_id, policy_version):
        """Record one fact question's answer (ask ledger): a question already in the
        ledger under the same normalized key takes the newer passages and reply. Returns its id."""
        row = self.db.execute("SELECT ledger_id FROM ask_ledger WHERE normalized = ?",
                              (normalized,)).fetchone()
        values = (question, dumps(list(dict.fromkeys(unit_ids))), reply, run_id, now(),
                  policy_version)
        with self.db:
            if row:
                self.db.execute("UPDATE ask_ledger SET question = ?, unit_ids = ?, reply = ?, "
                                "run_id = ?, recorded_at = ?, policy_version = ? "
                                "WHERE ledger_id = ?", (*values, row[0]))
                return row[0]
            cur = self.db.execute("INSERT INTO ask_ledger (normalized, question, unit_ids, reply, "
                                  "run_id, recorded_at, policy_version) "
                                  "VALUES (?, ?, ?, ?, ?, ?, ?)", (normalized, *values))
            self.db.execute("INSERT INTO ask_fts VALUES (?, ?)", (cur.lastrowid, normalized))
        return cur.lastrowid

    def ask_fact_candidates(self, normalized, limit=3):
        """The ledger rows whose normalized question best matches `normalized` (FTS over the
        question's content words, any word, ranked by bm25), best first."""
        terms = query_terms(normalized)
        if not terms:
            return []
        rows = self.db.execute(
            "SELECT l.*, bm25(ask_fts) AS score FROM ask_fts f "
            "JOIN ask_ledger l ON l.ledger_id = f.ledger_id "
            "WHERE ask_fts MATCH ? ORDER BY score LIMIT ?",
            (" OR ".join(f'"{t}"' for t in terms), limit)).fetchall()
        return [{**dict(r), "unit_ids": loads(r["unit_ids"]) or []} for r in rows]

    def cited_relevant(self, query, limit=5, min_score=0.5):
        """Primary units earlier answers cited for a question, rewrite or requirement relevant to
        `query`, best first (deterministic; scored like `relevant`, over the citation rows'
        question/rewrite/requirement text, each unit at its best row).
        Returns [{"unit_id", "score", "matched"}]."""
        terms = query_terms(query)
        total = self.db.execute("SELECT count(*) FROM cited_fts").fetchone()[0]
        if not terms or not total:
            return []
        hits, weight = {}, {}
        for term in terms:
            rows = self.db.execute("SELECT rowid, unit_id FROM cited_fts WHERE cited_fts MATCH ?",
                                   (f'"{term}"',)).fetchall()
            weight[term] = math.log((total + 1) / (len(rows) + 0.5))
            for rowid, uid in rows:
                hits.setdefault((uid, rowid), set()).add(term)
        full = sum(weight.values()) or 1.0
        need = min(2, len(terms))
        best = {}
        for (uid, _), matched in hits.items():
            if len(matched) < need:
                continue
            score = round(sum(weight[t] for t in matched) / full, 4)
            if score >= min_score and (uid not in best or score > best[uid]["score"]):
                best[uid] = {"unit_id": uid, "score": score, "matched": sorted(matched)}
        ranked = sorted(best.values(), key=lambda x: (-x["score"], -len(x["matched"]), x["unit_id"]))
        return ranked[:limit]

    def stats(self):
        one = lambda sql: self.db.execute(sql).fetchone()[0]  # noqa: E731
        by = lambda sql: {k if k is not None else "unrecorded": v  # noqa: E731
                          for k, v in self.db.execute(sql).fetchall()}
        return {
            "runs": one("SELECT count(*) FROM runs"),
            "searches": one("SELECT count(*) FROM run_searches"),
            "units": one("SELECT count(*) FROM units"),
            "units_by_layer": {layer: one(f"SELECT count(*) FROM units WHERE layer = '{layer}'")
                               for layer in LAYERS},
            "discoveries": one("SELECT count(*) FROM discoveries"),
            "selections": one("SELECT count(*) FROM selections"),
            "selector_roles": by("SELECT role, count(*) FROM selections GROUP BY role ORDER BY role"),
            "relationships": by("SELECT rel_type, count(*) FROM relationships GROUP BY rel_type "
                                "ORDER BY rel_type"),
            "secondary_claims": by("SELECT claim_type, count(*) FROM secondary_claims "
                                   "GROUP BY claim_type ORDER BY claim_type"),
            "secondary_verification": by("SELECT verification, count(*) FROM secondary_claims "
                                         "GROUP BY verification ORDER BY verification"),
        }

    # ---- research-history signals (metadata about past research, not truth scores) -----------

    def repeatedly_retrieved(self, min_runs=2, limit=50):
        """Units discovered in at least `min_runs` different runs."""
        return [dict(r) for r in self.db.execute(
            """SELECT unit_id, count(DISTINCT run_id) AS runs, count(*) AS discoveries
               FROM discoveries WHERE run_id IS NOT NULL GROUP BY unit_id
               HAVING runs >= ? ORDER BY runs DESC, discoveries DESC, unit_id LIMIT ?""",
            (min_runs, limit))]

    def repeatedly_selected(self, role="CORE", min_runs=2, limit=50):
        """Units the selector gave `role` in at least `min_runs` different runs."""
        return [dict(r) for r in self.db.execute(
            """SELECT unit_id, count(DISTINCT run_id) AS runs FROM selections WHERE role = ?
               GROUP BY unit_id HAVING runs >= ? ORDER BY runs DESC, unit_id LIMIT ?""",
            (role, min_runs, limit))]

    def mixed_selection(self, limit=50):
        """Units kept in some runs and dropped in others."""
        return [dict(r) for r in self.db.execute(
            """SELECT unit_id, sum(kept = 1) AS kept_runs, sum(kept = 0) AS dropped_runs
               FROM selections GROUP BY unit_id HAVING kept_runs > 0 AND dropped_runs > 0
               ORDER BY unit_id LIMIT ?""", (limit,))]

    def queries_for(self, unit_id):
        """Every search query that found the unit, with how often and in how many runs."""
        return [dict(r) for r in self.db.execute(
            """SELECT query, count(*) AS times, count(DISTINCT run_id) AS runs FROM discoveries
               WHERE unit_id = ? AND query IS NOT NULL GROUP BY query ORDER BY times DESC, query""",
            (unit_id,))]

    def requirements_covered(self, unit_id):
        """The run requirements the selector said the unit covered."""
        rows = self.db.execute(
            "SELECT run_id, covers FROM selections WHERE unit_id = ? AND covers IS NOT NULL",
            (unit_id,)).fetchall()
        out = []
        for r in rows:
            for rid in loads(r["covers"]):
                req = self.db.execute("SELECT kind, text FROM run_requirements WHERE run_id = ? "
                                      "AND requirement_id = ?", (r["run_id"], rid)).fetchone()
                out.append({"run_id": r["run_id"], "requirement_id": rid,
                            "kind": req["kind"] if req else None, "text": req["text"] if req else None})
        return out

    def dependents(self, unit_id):
        """Derived units that were derived from this unit."""
        return [r["from_unit"] for r in self.db.execute(
            """SELECT r.from_unit FROM relationships r JOIN units u ON u.unit_id = r.from_unit
               WHERE r.to_unit = ? AND r.rel_type = 'derived_from' AND u.layer = 'derived'
               ORDER BY r.from_unit""", (unit_id,))]

    # ---- reuse of completed research -------------------------------------------------

    def completed_runs(self, corpus_version):
        """Runs the live pipeline recorded as successfully completed research against
        `corpus_version` (metadata.completed_research), newest first, with parsed metadata. Runs
        from another corpus version, imports and unfinished runs never qualify."""
        out = []
        for r in self.db.execute("SELECT run_id, question, started_at, imported_from, metadata "
                                 "FROM runs WHERE metadata IS NOT NULL "
                                 "ORDER BY started_at DESC, run_id DESC"):
            try:
                md = loads(r["metadata"])
            except ValueError:
                continue
            if (isinstance(md, dict) and md.get("completed_research") is True
                    and md.get("corpus_version") == corpus_version):
                out.append({**dict(r), "metadata": md})
        return out

    def run_selections(self, run_id):
        """Every selector judgment recorded for one run, keyed by (round, hit_id), with the
        judged unit's exact text and source."""
        rows = self.db.execute(
            """SELECT s.*, u.layer, u.text, u.source_id, u.source_title FROM selections s
               JOIN units u ON u.unit_id = s.unit_id WHERE s.run_id = ?""", (run_id,))
        out = {}
        for r in rows:
            item = dict(r)
            item["covers"] = loads(item["covers"]) or []
            item["metadata"] = loads(item["metadata"]) or {}
            out[(item["round"], item["hit_id"])] = item
        return out

    # ---- research-need ledger --------------------------------------------------------

    def upsert_need(self, need_id, signature, signature_version, frame, selector_policy, corpus,
                    requirements, facets, input_units, selection, selector_primary, source_run,
                    recorded_at):
        """Record a need's latest completed selection (replaces the entry with the same
        need_id; no other row is touched)."""
        with self.db:
            self.db.execute(
                """INSERT INTO research_needs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(need_id) DO UPDATE SET frame = excluded.frame,
                     requirements = excluded.requirements, facets = excluded.facets,
                     input_units = excluded.input_units, selection = excluded.selection,
                     selector_primary = excluded.selector_primary,
                     source_run = excluded.source_run, recorded_at = excluded.recorded_at""",
                (need_id, signature, signature_version, frame["subject"], dumps(frame),
                 selector_policy, dumps(corpus), dumps(list(requirements)), dumps(facets),
                 dumps(list(input_units)), dumps(selection), flag(selector_primary), source_run,
                 recorded_at))

    def needs_for_subject(self, subject):
        """Ledger entries whose normalized frame subject is `subject`, parsed."""
        out = []
        for r in self.db.execute("SELECT * FROM research_needs WHERE subject = ?", (subject,)):
            item = dict(r)
            for k in ("frame", "corpus", "requirements", "facets", "input_units", "selection"):
                item[k] = loads(item[k])
            item["selector_primary"] = bool(item["selector_primary"])
            item["selection_units"] = (None if item["selection"] is None
                                       else item["selection"].get("units", []))
            out.append(item)
        return out

    def primary_units_present(self, unit_ids):
        """The subset of `unit_ids` stored as primary passages (with their exact text)."""
        ids = list(dict.fromkeys(unit_ids))
        found = set()
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            found.update(r[0] for r in self.db.execute(
                f"SELECT unit_id FROM units WHERE layer = ? AND unit_id IN "
                f"({','.join('?' * len(chunk))})", (PRIMARY, *chunk)))
        return found

    def rebuild_search_index(self):
        """Recreate the full-text index from the units table (e.g. after manual edits)."""
        with self.db:
            self.db.execute("DELETE FROM units_fts")
            self._fill_search_index()

    def _fill_search_index(self):
        self.db.execute("""INSERT INTO units_fts SELECT u.unit_id, u.layer,
                             CASE WHEN u.layer = ? THEN '' ELSE ifnull(u.source_title, '') END,
                             u.text, message_body(c.context)
                           FROM units u LEFT JOIN secondary_claims c USING (unit_id)
                           ORDER BY u.unit_id""", (SECONDARY,))


# ---- importing completed research runs ----------------------------------------------------------

def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def round_files(folder, pattern):
    """{round: path} for files like search-results-2.json."""
    out = {}
    for path in folder.glob(pattern.replace("{n}", "*")):
        m = re.fullmatch(pattern.replace(".", r"\.").replace("{n}", r"(\d+)"), path.name)
        if m:
            out[int(m.group(1))] = path
    return dict(sorted(out.items()))


def run_log_facts(folder):
    """Start time, depth and question from run.jsonl, when present."""
    facts = {}
    try:
        lines = (folder / "run.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError:
        return facts
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get("event") == "start":
            facts.setdefault("started_at", entry.get("time"))
            facts.setdefault("question", entry.get("question"))
        if entry.get("event") in ("planner", "summary") and entry.get("depth"):
            facts["depth"] = entry["depth"]
    return facts


class ImportReport(dict):
    def __init__(self, run_id):
        super().__init__(run_id=run_id, units_new=0, units_seen=0, discoveries_new=0,
                         selections=0, skipped_preview_only=0, notes=[])


def import_run_folder(store, folder, titles=None):
    """Import a completed run's log folder (logs/<run>/). Exact passage text comes from
    search-results-N.json; selector decisions from evidence-trace.json or, for older runs,
    selector-N.json; final evidence from evidence-N.diagnostics.json. Anything a run did not
    record is left NULL; previews are never stored as passage text."""
    folder = Path(folder)
    run_id = folder.name
    report = ImportReport(run_id)
    facts = run_log_facts(folder)
    question = None
    try:
        question = (folder / "question.txt").read_text(encoding="utf-8").strip() or None
    except OSError:
        question = facts.get("question")
    plan = read_json(folder / "plan-1.json") or read_json(folder / "search-queries-1.json") or {}
    requirements = [r for r in plan.get("requirements") or [] if isinstance(r, dict) and r.get("id")]
    searches = [{"query": s["query"], "covers": s.get("covers")} if isinstance(s, dict)
                else {"query": s} for s in plan.get("searches") or []
                if (isinstance(s, dict) and s.get("query")) or (isinstance(s, str) and s)]
    results = round_files(folder, "search-results-{n}.json")
    if not results:
        report["notes"].append("no search-results files: no exact passage text to import")
    for rnd, path in results.items():
        if rnd > 1 or not searches:  # round 1 comes from the plan when the run saved one
            searches += [{"query": q, "round": rnd} for q in (read_json(path) or {}).get("queries") or []]
    trace = read_json(folder / "evidence-trace.json") or {}
    if not trace:
        report["notes"].append("no evidence-trace.json: selection taken from selector output, if any")
    if not requirements:
        report["notes"].append("no answer requirements recorded for this run")
    diagnostics = round_files(folder, "evidence-{n}.diagnostics.json")
    final_ids = None
    if diagnostics:
        blocks = (read_json(list(diagnostics.values())[-1]) or {}).get("blocks") or []
        final_ids = {h["hit_id"] for b in blocks for h in b.get("hits") or [] if h.get("hit_id")}
    titles = dict(titles or {})
    for path in diagnostics.values():
        for b in (read_json(path) or {}).get("blocks") or []:
            if b.get("source_id") and b.get("source_title"):
                titles.setdefault(b["source_id"], b["source_title"])
    selector_out = {rnd: read_json(p) or {} for rnd, p in round_files(folder, "selector-{n}.json").items()}
    coverage = {rnd: out["coverage"] for rnd, out in selector_out.items() if out.get("coverage")}
    metadata = {k: v for k, v in (("coverage_by_round", coverage or None),
                                   ("answer_file", "answer.md" if (folder / "answer.md").exists() else None))
                if v}
    store.add_run(run_id, question, plan.get("depth") or facts.get("depth"), facts.get("started_at"),
                  str(folder), metadata or None, requirements, searches)
    seen = facts.get("started_at")

    trace_raw = {r.get("id"): r for r in trace.get("raw") or [] if r.get("id")}
    trace_cands = {c.get("id"): c for c in trace.get("candidates") or [] if c.get("id")}
    continuations = {}
    for path in round_files(folder, "continuations-{n}.json").values():
        for c in read_json(path) or []:
            if isinstance(c, dict) and c.get("hit_id"):
                continuations[c["hit_id"]] = {k: c.get(k) for k in (
                    "continuation_reason", "continuation_status", "continuation_text") if c.get(k)}

    def title_of(source_id, *trace_items):
        if source_id in titles:
            return titles[source_id]
        for item in trace_items:  # the trace falls back to the source id when it had no title
            if item and item.get("source") and item["source"] != source_id:
                return item["source"]
        return None

    for rnd, path in results.items():
        data = read_json(path) or {}
        text_units = {}
        for i, raw in enumerate(data.get("raw_hits") or [], 1):
            text = raw.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            ref = raw.get("raw_id") or f"r{rnd}.{i}"
            sid, title = raw.get("source_id"), title_of(raw.get("source_id"), trace_raw.get(ref))
            uid, new = store._add_unit(PRIMARY, text, primary_key(sid, title), seen,
                                       source_id=sid, source_title=title)
            report["units_new" if new else "units_seen"] += 1
            text_units[(raw.get("source_id"), fingerprint(text))] = uid
            meta = {k: raw.get(k) for k in ("start", "end", "candidate", "merge") if raw.get(k) is not None}
            report["discoveries_new"] += store.add_discovery(
                uid, run_id, raw.get("query"), raw.get("rank"), rnd, "notebooklm", ref, seen, meta or None)
        sel = selector_out.get(rnd) or {}
        decisions = {d.get("hit_id"): d for d in sel.get("decisions") or [] if isinstance(d, dict)}
        selected, context = set(sel.get("selected_hit_ids") or []), set(sel.get("context_hit_ids") or [])
        for cand in data.get("candidates") or []:
            text, hit_id, sid = cand.get("text"), cand.get("hit_id"), cand.get("source_id")
            if not isinstance(text, str) or not text.strip() or not hit_id:
                continue
            uid = text_units.get((sid, fingerprint(text)))
            if uid is None:  # older runs: a candidate's text may not appear among raw hits
                title = title_of(sid, trace_cands.get(hit_id))
                uid, new = store._add_unit(PRIMARY, text, primary_key(sid, title), seen,
                                           source_id=sid, source_title=title)
                report["units_new" if new else "units_seen"] += 1
                for found in cand.get("found_by") or []:
                    report["discoveries_new"] += store.add_discovery(
                        uid, run_id, found.get("query"), found.get("rank"), rnd, "notebooklm",
                        f"{hit_id}:{found.get('query')}", seen)
            t, d = trace_cands.get(hit_id), decisions.get(hit_id)
            if t is None and not sel:
                continue  # no selector record for this round: nothing to say about selection
            meta = continuations.get(hit_id) or {}
            if t is not None:
                role, covers, reason, kept, ctx = (t.get("role"), t.get("covers") if "covers" in t else None,
                                                   t.get("reason"), t.get("kept"), t.get("context"))
                meta["recorded_by"] = "evidence-trace"
                claim = {k: t[k] for k in ("predicate", "scope", "applies_to", "use", "relations") if k in t}
                if claim:  # the selector's claim structure
                    meta["claim"] = claim
            else:
                role, reason = (d or {}).get("role"), (d or {}).get("reason")
                covers = (d or {}).get("covers") if d and "covers" in d else None
                kept, ctx = hit_id in selected, hit_id in context
                meta["recorded_by"] = "selector output"
            final = None if final_ids is None else hit_id in final_ids
            store.add_selection(uid, run_id, rnd, hit_id, role, covers, reason, kept, ctx, final,
                                meta or None)
            report["selections"] += 1
    report["skipped_preview_only"] = 0 if results else len(trace_cands)
    return report


def import_result(store, data, origin, titles=None):
    """Import a saved research result (the app's result JSON, or a stored record holding one).
    When its run folder exists locally, the folder is imported instead (full fidelity). Otherwise
    only the final evidence passages carry exact text; other candidates are previews and skipped."""
    result = data.get("result") if isinstance(data.get("result"), dict) else data
    run_dir = result.get("run_dir")
    if run_dir and Path(run_dir).is_dir():
        return import_run_folder(store, run_dir, titles)
    details = result.get("details") or {}
    run_id = Path(run_dir).name if run_dir else "result-" + hashlib.sha256(
        json.dumps([result.get("question"), data.get("created_at")]).encode()).hexdigest()[:12]
    report = ImportReport(run_id)
    started = (time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(data["created_at"]))
               if isinstance(data.get("created_at"), (int, float)) else None)
    plan = details.get("search_plan") or [{"query": q} for q in result.get("search_queries") or []]
    metadata = {k: v for k, v in (("coverage", details.get("coverage")), ("repair", details.get("repair"))) if v}
    store.add_run(run_id, result.get("question"), result.get("depth") or details.get("depth"), started,
                  str(origin), metadata or None, details.get("requirements") or [], plan)
    trace = {c.get("id"): c for c in (details.get("trace") or {}).get("candidates") or []}
    placed = set()
    for source in result.get("sources") or []:
        sid, title = source.get("source_id"), source.get("title")
        for excerpt in source.get("excerpts") or []:
            for p in excerpt.get("passages") or []:
                if not isinstance(p.get("text"), str) or not p["text"].strip():
                    continue
                uid, new = store._add_unit(PRIMARY, p["text"], primary_key(sid, title), started,
                                           source_id=sid, source_title=title)
                report["units_new" if new else "units_seen"] += 1
                for q in p.get("queries") or []:
                    report["discoveries_new"] += store.add_discovery(
                        uid, run_id, q, None, None, "notebooklm", p.get("hit_id"), started)
                if p.get("hit_id"):
                    placed.add(p["hit_id"])
                    t = trace.get(p["hit_id"]) or {}
                    store.add_selection(uid, run_id, 1, p["hit_id"], t.get("role"),
                                        t.get("covers") if "covers" in t else None, t.get("reason"),
                                        t.get("kept", True), t.get("context"), True,
                                        {"recorded_by": "result sources"})
                    report["selections"] += 1
    report["skipped_preview_only"] = len([h for h in trace if h not in placed])
    if report["skipped_preview_only"]:
        report["notes"].append("candidates known only by preview were not stored as passages")
    return report


# ---- importing processed secondary (community) records ------------------------------------------

SECONDARY_RECORD_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Cited Research Assistant secondary claim record (one JSON object per line)",
    "type": "object",
    "required": ["platform", "record_id", "claim_type", "claim_text"],
    "properties": {
        "platform": {"type": "string", "description": "where it came from: chat, forum, article, ..."},
        "record_id": {"type": "string", "description": "stable id of the message/post, e.g. chat:<group>/<id>"},
        "community": {"type": "string", "description": "channel, group or board"},
        "thread_ref": {"type": "string"},
        "author": {"type": "string"},
        "url": {"type": "string"},
        "date": {"type": "string", "description": "ISO date of the message/post"},
        "claim_type": {"enum": list(CLAIM_TYPES)},
        "claim_text": {"type": "string", "description": "the claim exactly as extracted"},
        "context": {"type": "string", "description": "surrounding text needed to interpret it"},
        "verification": {"enum": list(VERIFICATION), "default": "unverified"},
        "verification_note": {"type": "string"},
        "primary_links": {"type": "array", "items": {
            "type": "object", "required": ["relation", "unit_id"],
            "properties": {"relation": {"enum": list(SECONDARY_LINKS)}, "unit_id": {"type": "string"},
                           "note": {"type": "string"}}}},
        "primary_refs": {"type": "array", "items": {
            "type": "object", "description": "cited primary material not in memory yet",
            "properties": {"source_title": {"type": "string"}, "source_date": {"type": "string"},
                           "quote": {"type": "string"}, "relation": {"enum": list(SECONDARY_LINKS)}}}},
        "processed_by": {"type": "string", "description": "processing step and version"},
        "metadata": {"type": "object"},
    },
}


def import_secondary(store, path):
    """Import processed community claims from a JSONL file (see SECONDARY_RECORD_SCHEMA). Each
    valid line becomes a secondary claim; a line with an error is reported and skipped, and a
    link to an unknown unit is dropped (the claim keeps it as a primary_ref) with a note. A
    verification status that its links cannot back is stored as "unverified", with a note."""
    report = {"file": str(path), "claims_new": 0, "claims_seen": 0, "errors": [], "notes": []}
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError as e:
            report["errors"].append({"line": n, "error": f"not JSON: {e}"})
            continue
        missing = [k for k in SECONDARY_RECORD_SCHEMA["required"] if not rec.get(k)]
        if missing:
            report["errors"].append({"line": n, "error": f"missing {', '.join(missing)}"})
            continue
        links, refs = [], list(rec.get("primary_refs") or [])
        for link in rec.get("primary_links") or []:
            if (isinstance(link, dict) and link.get("relation") in SECONDARY_LINKS
                    and store.layer(link.get("unit_id") or "") is not None
                    and (link["relation"] not in PRIMARY_ONLY_LINKS or store.layer(link["unit_id"]) == PRIMARY)):
                links.append(link)
            else:
                refs.append({"unresolved_link": link})
                report["notes"].append({"line": n, "note": f"link not stored: {link}"})
        status = rec.get("verification") or "unverified"
        existed = store.db.execute(
            "SELECT 1 FROM units WHERE unit_id = ?",
            (unit_id_for(SECONDARY, f"{rec['platform']}:{rec['record_id']}",
                         fingerprint(rec["claim_text"])),)).fetchone() is not None
        try:
            uid = store.add_secondary_claim(
                rec["claim_text"], rec["platform"], rec["record_id"], rec["claim_type"],
                context=rec.get("context"), community=rec.get("community"),
                thread_ref=rec.get("thread_ref"), author=rec.get("author"), url=rec.get("url"),
                source_date=rec.get("date"), links=links, primary_refs=refs,
                processed_by=rec.get("processed_by"), metadata=rec.get("metadata"))
        except ValueError as e:
            report["errors"].append({"line": n, "error": str(e)})
            continue
        if status != "unverified":
            try:
                store.set_verification(uid, status, rec.get("verification_note"))
            except ValueError as e:
                report["notes"].append({"line": n, "note": f"verification kept unverified: {e}"})
        report["claims_seen" if existed else "claims_new"] += 1
    return report


# A statement the community text attributes to the corpus author ("the author said", "Author: ...",
# "according to the author", "asked the author"), under the names in settings.AUTHOR_NAMES
# (CRA_AUTHOR_NAMES). Deterministic and deliberately broad: it only adds a caution label.
AUTHOR_ATTRIBUTION = re.compile(
    rf"\b(?:{settings.AUTHOR_NAMES})\b(?:\s*:|\W+(?:\w+\W+){{0,6}}?(?:said|says|say|told|tells|stated|"
    r"states|recommended|recommends|wrote|writes|answered|replied|claimed|claims|mentioned|"
    rf"explained|described|taught|advised|suggested|called))|according to (?:{settings.AUTHOR_NAMES})\b"
    rf"|\basked (?:{settings.AUTHOR_NAMES})\b", re.I)
MATCHED_PRIMARY = ("quotes_primary", "paraphrases_primary")
UNMATCHED_ATTRIBUTION = "community-reported, not matched to a primary passage"


def secondary_label(claim):
    """The deterministic label a secondary claim carries at the reasoner boundary: community
    source identity, category, and "unverified" until a primary link backs a checked status;
    plus the attribution caution for text attributed to the corpus author with no matching primary
    link."""
    identity = ", ".join(x for x in (claim.get("author"), claim.get("platform"),
                                     claim.get("community")) if x) or "unknown"
    links = [l for l in claim.get("links") or [] if l.get("layer") == PRIMARY]
    status = (claim.get("verification") if links and claim.get("verification") in
              ("verified_primary", "partially_verified", "contradicted_by_primary") else "unverified")
    attributed = (claim.get("claim_type") in ("primary_quote", "primary_paraphrase")
                  or AUTHOR_ATTRIBUTION.search(claim.get("claim_text") or "")
                  or AUTHOR_ATTRIBUTION.search(claim.get("context") or ""))
    matched = any(l.get("relation") in MATCHED_PRIMARY for l in links)
    return {"identity": identity, "category": claim.get("claim_type") or "unclassified",
            "status": status,
            "attribution": UNMATCHED_ATTRIBUTION if attributed and not matched else None}


def secondary_evidence_block(claim):
    """A secondary claim rendered for a reasoning step, visibly apart from primary evidence
    (never as SOURCE/PASSAGE): its origin, label, claim type, verification and primary links."""
    origin = " · ".join(x for x in (claim.get("platform"), claim.get("community"), claim.get("author"),
                                    claim.get("source_date")) if x)
    links = ", ".join(f"{l['relation']} {l['unit_id']}" for l in claim.get("links") or []) or "none"
    label = secondary_label(claim)
    return (f"SECONDARY SOURCE (community material, not the corpus author's own words): {origin}\n"
            f"LABEL: community source {label['identity']}; category {label['category']}; "
            f"{label['status']}\n"
            + (f"ATTRIBUTION TO THE CORPUS AUTHOR: {label['attribution']}\n" if label["attribution"] else "")
            + f"CLAIM TYPE: {claim['claim_type']}\nVERIFICATION: {claim['verification']}\n"
            f"PRIMARY LINKS: {links}\nSECONDARY CLAIM:\n{claim['claim_text']}")


def load_titles(path=TITLES_CACHE):
    data = read_json(path)
    return data if isinstance(data, dict) else {}


def citation_rows(result):
    """The passages a completed result cited, each with what it was cited for (pure). Results with
    citations: their passage citations; older ones: the hits their synthesis (else evidence map)
    marks as used. A passage's requirements are those the final coverage or evidence map lists
    it under, else the covers of the planner searches that found it; one row per requirement
    ('' when none is known). Returns [{hit_id, source_id, title, text, question, standalone,
    requirement, origin}]."""
    details = result.get("details") or {}
    turn = result.get("turn") or {}
    follow = details.get("follow_up") or {}
    question = turn.get("question") or follow.get("question") or result.get("question") or ""
    standalone = turn.get("standalone") or result.get("question") or question
    reqs = {r["id"]: r.get("text") or "" for r in details.get("requirements") or []
            if isinstance(r, dict) and r.get("id")}
    covered = {}
    for c in details.get("coverage") or []:
        for h in c.get("hit_ids") or []:
            covered.setdefault(h, []).append(c.get("for") or c.get("requirement_id"))
    for e in details.get("evidence_map") or []:
        for st in e.get("statements") or []:
            covered.setdefault(st.get("hit_id"), []).append(e.get("requirement_id"))
    query_covers = {x.get("query"): x.get("covers") or [] for x in details.get("search_plan") or []
                    if isinstance(x, dict)}
    passages = {}
    for src in result.get("sources") or []:
        for e in src.get("excerpts") or []:
            for p in e.get("passages") or []:
                passages.setdefault(p.get("hit_id"), {
                    "source_id": src.get("source_id"), "title": src.get("title"),
                    "text": p.get("text"), "queries": p.get("queries") or []})
    citations = result.get("citations")
    if citations is not None:
        origin = "citation"
        used = [c.get("id") for c in citations if c.get("kind") == "passage"]
        for c in citations:
            if c.get("kind") == "passage" and c.get("id") not in passages:
                passages[c.get("id")] = {"source_id": c.get("source_id"), "title": c.get("title"),
                                         "text": c.get("text"), "queries": []}
    else:
        origin = "used"
        used = [h for it in details.get("synthesis") or [] if it.get("treatment") != "not_used"
                for h in it.get("hit_ids") or []]
        used = used or [st.get("hit_id") for e in details.get("evidence_map") or []
                        for st in e.get("statements") or []]
    rows = []
    for hit in dict.fromkeys(used):
        p = passages.get(hit)
        if not p or not p.get("source_id") or not p.get("text"):
            continue
        rids = covered.get(hit) or [r for q in p["queries"]
                                    for r in (query_covers.get(q if isinstance(q, str) else
                                                               (q or {}).get("query")) or [])]
        texts = list(dict.fromkeys(reqs[r] for r in rids if r in reqs)) or [""]
        for text in texts:
            rows.append({"hit_id": hit, "source_id": p["source_id"], "title": p["title"],
                         "text": p["text"], "question": question, "standalone": standalone,
                         "requirement": text, "origin": origin})
    return rows


def backfill_citations(store, logs_dir=None):
    """Fill cited_passages once from the run folders' result.json files (deterministic, no
    model calls; idempotent). A row is recorded only when its passage is a stored primary unit."""
    logs_dir = Path(logs_dir or settings.LOGS_DIR)
    counts = {"runs": 0, "rows": 0, "units": set(), "not_in_memory": 0}
    with store.db:
        for f in sorted(logs_dir.glob("*/result.json")):
            try:
                result = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            rows = citation_rows(result)
            counts["runs"] += bool(rows)
            for r in rows:
                uid = unit_id_for(PRIMARY, primary_key(r["source_id"], r["title"]),
                                  fingerprint(r["text"]))
                if store.layer(uid) != PRIMARY:
                    counts["not_in_memory"] += 1
                    continue
                counts["units"].add(uid)
                counts["rows"] += store.add_citation(uid, f.parent.name, r["question"],
                                                     r["standalone"], r["requirement"],
                                                     "backfill_" + r["origin"])
        store.db.execute("INSERT OR REPLACE INTO meta VALUES ('citations_backfilled', ?)", (now(),))
    counts["units"] = len(counts["units"])
    return counts


def import_path(store, path, titles=None):
    """Import an evidence trace (reads its run folder), a run folder, or a saved result JSON."""
    path = Path(path)
    titles = load_titles() if titles is None else titles
    if path.is_dir():
        return import_run_folder(store, path, titles)
    data = read_json(path)
    if data is None:
        raise ValueError(f"{path} is not a readable JSON file or run folder")
    if isinstance(data, dict) and ("raw" in data or "candidates" in data) and "details" not in data:
        return import_run_folder(store, path.parent, titles)  # an evidence trace: use its run folder
    if isinstance(data, dict) and (data.get("result") or data.get("details") or data.get("sources")):
        return import_result(store, data, path, titles)
    raise ValueError(f"{path} is not an evidence trace, run folder or research result")


# ---- CLI ----------------------------------------------------------------------------------------

LABEL = {PRIMARY: "PRIMARY_RETRIEVED", SECONDARY: "SECONDARY", DERIVED: "DERIVED"}


def provenance(row):
    if row["layer"] == PRIMARY:
        return " · ".join(x for x in (row.get("source_title") or row.get("source_id") or "unknown source",
                                      row.get("source_date")) if x)
    if row["layer"] == SECONDARY:
        return " · ".join(x for x in (row.get("platform"), row.get("author"), row.get("source_title"),
                                      row.get("source_date"), row.get("kind")) if x)
    return f"{row.get('kind')} (derived)"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Persistent research memory (saved discoveries only).")
    parser.add_argument("--db", help="database path (default data/research-memory.db or $CRA_MEMORY_DB)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="create the database (safe to repeat)")
    sub.add_parser("stats", help="counts by layer, role and relationship")
    p = sub.add_parser("import-trace", help="import evidence-trace.json, a run folder, or a result JSON")
    p.add_argument("paths", nargs="+")
    p = sub.add_parser("search", help="full-text search over saved memory (not the source corpus)")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--layer", choices=LAYERS)
    p = sub.add_parser("show", help="one unit with its discoveries, selections and relationships")
    p.add_argument("unit")
    p = sub.add_parser("import-secondary", help="import processed community claims (JSONL)")
    p.add_argument("paths", nargs="+")
    sub.add_parser("secondary-schema", help="print the secondary record format as JSON Schema")
    p = sub.add_parser("history", help="repeatedly retrieved / selected units, mixed selections")
    p.add_argument("--min", type=int, default=2, help="minimum number of runs")
    sub.add_parser("reindex", help="rebuild the full-text index from stored units")
    p = sub.add_parser("backfill-citations", help="fill cited_passages from logs/*/result.json")
    p.add_argument("--logs", help="log folder (default logs/)")
    args = parser.parse_args(argv)

    with MemoryStore(args.db) as store:
        if args.command == "init":
            print(f"Research memory ready: {store.path}")
        elif args.command == "stats":
            print(json.dumps(store.stats(), indent=2))
        elif args.command == "import-trace":
            for path in args.paths:
                report = import_path(store, path)
                print(json.dumps(report, indent=2, ensure_ascii=False))
        elif args.command == "search":
            rows = store.search(args.query, args.limit, args.layer)
            if not rows:
                print("No saved memory matches. (This searches only what research has already found.)")
            for r in rows:
                seen = f" · found {r['discoveries']}x in {r['runs']} runs" if r["layer"] == PRIMARY else ""
                print(f"[{LABEL[r['layer']]}] {r['unit_id']} · {provenance(r)}{seen}\n    {r['snippet']}")
        elif args.command == "show":
            unit = store.unit(store.resolve(args.unit))
            unit["queries"] = store.queries_for(unit["unit_id"])
            if unit["layer"] == SECONDARY:
                unit["secondary_claim"] = store.secondary_claim(unit["unit_id"])
            unit["derived_dependents"] = store.dependents(unit["unit_id"])
            print(f"[{LABEL[unit['layer']]}] {unit['unit_id']} · {provenance(unit)}")
            print(json.dumps(unit, indent=2, ensure_ascii=False))
        elif args.command == "history":
            print(json.dumps({"repeatedly_retrieved": store.repeatedly_retrieved(args.min),
                              "repeatedly_core": store.repeatedly_selected("CORE", args.min),
                              "kept_and_dropped": store.mixed_selection()}, indent=2))
        elif args.command == "import-secondary":
            for path in args.paths:
                print(json.dumps(import_secondary(store, path), indent=2, ensure_ascii=False))
        elif args.command == "secondary-schema":
            print(json.dumps(SECONDARY_RECORD_SCHEMA, indent=2))
        elif args.command == "reindex":
            store.rebuild_search_index()
            print("Full-text index rebuilt.")
        elif args.command == "backfill-citations":
            print(json.dumps(backfill_citations(store, args.logs), indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, ValueError) as e:  # unknown or ambiguous unit id, unreadable import path
        print(f"error: {e.args[0] if e.args else e}", file=sys.stderr)
        sys.exit(2)
