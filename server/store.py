"""Local research history: one SQLite file, one row per question. Questions that follow up an
answer share its conversation thread (thread_id: the first question's id)."""
import json
import sqlite3
import time
from contextlib import contextmanager
import settings

DB_PATH = settings.DATA_DIR / "history.db"

# status: running | done | error | cancelled
SCHEMA = """
CREATE TABLE IF NOT EXISTS research (
    id          TEXT PRIMARY KEY,
    question    TEXT NOT NULL,
    status      TEXT NOT NULL,
    created_at  REAL NOT NULL,
    finished_at REAL,
    answer      TEXT,
    error       TEXT,
    result      TEXT,          -- JSON: the pipeline result (sources, details, ...)
    events      TEXT NOT NULL  -- JSON: progress events, replayed when the question is reopened
)
"""
JSON_FIELDS = ("result", "events", "turn")
# Columns added to an existing database when missing:
# - the conversation a question belongs to, and the question it follows up (NULL in older
#   rows: a thread of their own);
# - the compact turn record of an answered question (JSON, see research.turn_record);
# - the conversation's short sidebar title, kept on its first question's row
#   (server/titles.py); NULL until it is written.
ADDED_COLUMNS = {"thread_id": "TEXT", "parent_id": "TEXT", "turn": "TEXT", "title": "TEXT"}


@contextmanager
def connect():
    """A connection that commits (or rolls back) and closes when the block ends."""
    db = sqlite3.connect(DB_PATH, timeout=10)
    db.row_factory = sqlite3.Row
    try:
        with db:
            yield db
    finally:
        db.close()


def init():
    """Create the database, and mark questions left running by a previous server as failed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as db:
        db.execute(SCHEMA)
        have = {r["name"] for r in db.execute("PRAGMA table_info(research)")}
        for name, kind in ADDED_COLUMNS.items():
            if name not in have:
                db.execute(f"ALTER TABLE research ADD COLUMN {name} {kind}")
        db.execute("UPDATE research SET status = 'error', finished_at = ?, "
                   "error = 'The app was closed while this question was being researched.' "
                   "WHERE status = 'running'", (time.time(),))


def decode(row):
    item = dict(row)
    for field in JSON_FIELDS:
        if field in item:
            item[field] = json.loads(item[field]) if item[field] else None
    return item


def create(research_id, question, thread_id=None, parent_id=None):
    with connect() as db:
        db.execute("INSERT INTO research (id, question, status, created_at, events, thread_id, "
                   "parent_id) VALUES (?, ?, 'running', ?, '[]', ?, ?)",
                   (research_id, question, time.time(), thread_id or research_id, parent_id))
    return get(research_id)


def update(research_id, **fields):
    for field in JSON_FIELDS:
        if field in fields:
            fields[field] = json.dumps(fields[field], ensure_ascii=False)
    assignments = ", ".join(f"{name} = ?" for name in fields)
    with connect() as db:
        db.execute(f"UPDATE research SET {assignments} WHERE id = ?", (*fields.values(), research_id))


def get(research_id):
    with connect() as db:
        row = db.execute("SELECT *, COALESCE(thread_id, id) AS thread FROM research WHERE id = ?",
                         (research_id,)).fetchone()
    return with_thread(decode(row)) if row else None


def with_thread(item):
    """The item with thread_id always set (a row from before threads is its own thread)."""
    item["thread_id"] = item.pop("thread")
    return item


def history():
    """Every question, newest first; title is its conversation's title."""
    with connect() as db:
        rows = db.execute("SELECT r.id, r.question, r.status, r.created_at, r.finished_at, "
                          "r.parent_id, COALESCE(r.thread_id, r.id) AS thread, t.title "
                          "FROM research r LEFT JOIN research t "
                          "ON t.id = COALESCE(r.thread_id, r.id) "
                          "ORDER BY r.created_at DESC").fetchall()
    return [with_thread(dict(row)) for row in rows]


def untitled_threads():
    """(thread id, first question) of the conversations that have no title yet, newest first."""
    with connect() as db:
        rows = db.execute("SELECT id, question FROM research WHERE title IS NULL "
                          "AND COALESCE(thread_id, id) = id ORDER BY created_at DESC").fetchall()
    return [(row["id"], row["question"]) for row in rows]


def delete(research_id):
    with connect() as db:
        db.execute("DELETE FROM research WHERE id = ?", (research_id,))
