"""Short conversation titles for the sidebar ("Expense claim limits"), written once
per conversation from its first question, like a chat app names a session. Gemini Flash writes
it (the product's Gemini runtime), Haiku when Gemini is unavailable, and the question's first
words when neither answers. Titles are generated in the background and never block research."""
import logging
import re
import subprocess
import threading

from research import CLAUDE, ENV, FAST, STAGE_CWD
from server import connections, store

logger = logging.getLogger("cra")
TIMEOUT = 45
MAX_CHARS = 60
QUESTION_CHARS = 4000  # the start of a long question is enough to name it
PROMPT = """Name this conversation for a chat history sidebar. The user asked the question below
about the source library's subject. Write a short topic title of 2 to 6 words in
sentence case, like "Expense claim limits", "Vacation carry-over", "Remote work rules for
part-time staff". Name the topic; do not repeat the question word for word, answer it, or add
quotes, a final period or any other text. Correct obvious misspellings of subject terms.

Question:
{question}

Title:"""

_lock = threading.Lock()
_pending: set[str] = set()


def clean(text):
    """The first non-empty line, without quotes, labels or a final period; None when unusable."""
    line = next((l.strip() for l in (text or "").splitlines() if l.strip()), "")
    line = re.sub(r"^(title\s*:\s*)", "", line, flags=re.I).strip(" \"'`*#.")
    if not line or len(line) > MAX_CHARS * 2:
        return None
    return line[:MAX_CHARS].rstrip()


def fallback(question):
    """The question's first words, when no model names it."""
    words = question.split()
    title = " ".join(words[:7])
    return title + ("…" if len(words) > 7 else "")


def by_gemini(prompt):
    try:
        return clean(connections.run_gemini("title", prompt, {"stage": "title"}))
    except Exception as e:  # noqa: BLE001 - Gemini is optional
        logger.info("title: Gemini failed (%r)", e)
        return None


def by_haiku(prompt):
    cmd = [CLAUDE, "-p", "--model", FAST[0], "--tools", "", "--strict-mcp-config",
           "--setting-sources", "", "--no-session-persistence", "--output-format", "text"]
    try:
        done = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                              encoding="utf-8", cwd=STAGE_CWD, env=ENV, timeout=TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.info("title: Haiku failed (%r)", e)
        return None
    return clean(done.stdout) if done.returncode == 0 else None


def generate(thread_id, question):
    prompt = PROMPT.format(question=question.strip()[:QUESTION_CHARS])
    try:
        title = by_gemini(prompt) or by_haiku(prompt) or fallback(question)
        store.update(thread_id, title=title)
    except Exception:  # noqa: BLE001 - a title never breaks the app
        logger.exception("title generation failed")
    finally:
        with _lock:
            _pending.discard(thread_id)


def start(thread_id, question):
    """Name a conversation in the background (once; a repeat call while it runs is ignored)."""
    with _lock:
        if thread_id in _pending:
            return
        _pending.add(thread_id)
    threading.Thread(target=generate, args=(thread_id, question), name=f"title-{thread_id}",
                     daemon=True).start()


def backfill():
    """Name the conversations that have no title yet, one at a time, in the background."""
    def work():
        for thread_id, question in store.untitled_threads():
            with _lock:
                if thread_id in _pending:
                    continue
                _pending.add(thread_id)
            generate(thread_id, question)
    threading.Thread(target=work, name="title-backfill", daemon=True).start()
