"""Shared fixtures for the tests: a scripted stand-in for NotebookLM and Claude with a temporary
run folder, and builders for synthetic candidates, evidence and repair states.

All subject matter is invented (a fictional "morning rinse" for travel account and its ingredients) so the tests
exercise the reasoning contracts, not any real corpus answer.
"""
import copy
import json
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import research  # noqa: E402


def candidate(hit_id, source_id, text):
    """A deduplicated candidate as the selector receives it (no continuation needed)."""
    return {"hit_id": hit_id, "source_id": source_id, "text": text, "start": 0, "end": len(text),
            "needs_continuation": False, "found_by": [], "raw_ids": []}


def evidence(*blocks):
    """Reasoner evidence in the pipeline's own format: (source title, passage) blocks, labeled
    h1, h2, ... in order."""
    return "\n\n=====\n\n".join(f"SOURCE: {title}\nPASSAGE [h{n}]:\n{text}"
                                   for n, (title, text) in enumerate(blocks, 1))


def requirement(rid, kind, text):
    return {"id": rid, "kind": kind, "text": text}


def repair_done(request, why="follow-up search found no new passages"):
    """The state a final reasoner call sees after a repair round that found nothing new."""
    premises = [{"id": f"p{i}", "kind": "premise", "for": r["requirement_id"], "text": r["premise"],
                 "search": r["search"]} for i, r in enumerate(request, 1)]
    return {"request": request, "premises": premises, "searches": [r["search"] for r in request],
            "raw_hits": 0, "candidates": 0, "selected": 0, "context_hits": 0, "error": None,
            "coverage": [{"requirement_id": p["id"], "for": p["for"], "status": "missing",
                          "hit_ids": [], "missing": why} for p in premises]}


class Stub:
    """Scripted NotebookLM and Claude, so pipeline functions can be called without either.

    planner: the planner's structured output. selectors: one output (or callable taking the
    prompt) per selector call, in order. reasoner: callable (stage, schema, prompt) -> output.
    hits: search query -> NotebookLM results. titles: source id -> title (dates live in titles).
    Every Claude call is recorded in `calls` with its stage, schema and prompt.
    """

    def __init__(self, planner, selectors, reasoner, hits, titles):
        self.planner = planner
        self.selectors = list(selectors)
        self.reasoner = reasoner
        self.hits = hits
        self.titles = titles
        self.calls = []

    def claude(self, run, stage, model_effort, system, prompt, schema=None, **kw):
        self.calls.append({"stage": stage, "schema": schema, "prompt": prompt})
        if stage.startswith("planner"):
            return copy.deepcopy(self.planner)
        if stage.startswith("selector"):
            out = self.selectors.pop(0)
            return out(prompt) if callable(out) else copy.deepcopy(out)
        return self.reasoner(stage, schema, prompt)

    def notebooklm(self, run, label, args):
        if args[:2] == ["source", "search"]:
            return copy.deepcopy(self.hits.get(args[-1], [])), None
        return None, "fulltext is not available in tests"

    def call(self, stage):
        return next(c for c in self.calls if c["stage"] == stage)

    def stages(self):
        return [c["stage"] for c in self.calls]


@contextmanager
def offline(stub):
    """Point the pipeline at the stub, with logs and caches in a temporary folder."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        titles = root / "titles.json"
        titles.write_text(json.dumps(stub.titles), encoding="utf-8")
        with mock.patch.object(research, "ROOT", root), \
                mock.patch.object(research, "CACHE_DIR", root / "cache"), \
                mock.patch.object(research, "TITLES_FILE", titles), \
                mock.patch.object(research, "MEMORY_DB", root / "memory.db"), \
                mock.patch.object(research, "REUSE_DIR", root / "reuse"), \
                mock.patch.object(research, "VOCAB_FILE", root / "vocabulary.json"), \
                mock.patch.object(research.settings, "LOGS_DIR", root / "logs"), \
                mock.patch.object(research, "NOTEBOOK", "test-notebook"), \
                mock.patch.object(research, "preflight", lambda *a, **kw: {"ok": True, "cached": True}), \
                mock.patch.object(research, "run_child", lambda run, cmd, **kw: (0, "{}", "", None)), \
                mock.patch.object(research, "notebooklm", stub.notebooklm), \
                mock.patch.object(research, "claude", stub.claude):
            yield
