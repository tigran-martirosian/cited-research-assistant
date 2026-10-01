"""Tests for the persistent research memory (research_memory.py). Synthetic text only; no
NotebookLM, no model calls.

Run from the repository root:  python -m unittest tests.test_research_memory -v
"""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import research_memory as rm  # noqa: E402

A = "Alpha passage: the widget   needs\nwarm water to dissolve."  # odd spacing kept verbatim
B = "Beta passage: the gadget pulls heat out of the frame."
C = "Gamma passage: cold weather slows every widget down."


class MemoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.store = rm.MemoryStore(self.tmp / "memory.db")

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.tmp)

    def run_(self, run_id, question="q", **kw):
        return self.store.add_run(run_id, question, **kw)


class Storage(MemoryTest):
    def test_1_retrieved_passage_keeps_its_exact_text(self):
        self.run_("run-1")
        uid = self.store.record_retrieval(A, "run-1", "widget water", 1, 1, source_id="src-1",
                                          source_title="Talk one")
        unit = self.store.unit(uid)
        self.assertEqual(unit["text"], A)
        self.assertEqual(unit["layer"], rm.PRIMARY)
        self.assertEqual(unit["fingerprint"], rm.fingerprint(A))
        self.assertTrue(uid.startswith("P-"))

    def test_2_and_3_rediscovery_adds_history_not_a_second_unit(self):
        self.run_("run-1")
        self.run_("run-2")
        first = self.store.record_retrieval(A, "run-1", "widget water", 3, 1, source_id="src-1")
        again = self.store.record_retrieval(" ".join(A.split()), "run-2", "dissolving widgets", 1, 1,
                                            source_id="src-1")
        self.assertEqual(first, again)
        self.assertEqual(self.store.stats()["units"], 1)
        self.assertEqual(self.store.unit(first)["text"], A)  # the first exact text stays authoritative
        self.assertEqual({(q["query"], q["runs"]) for q in self.store.queries_for(first)},
                         {("widget water", 1), ("dissolving widgets", 1)})
        self.assertEqual(self.store.repeatedly_retrieved(2)[0]["unit_id"], first)
        # recording the very same discovery twice is idempotent
        self.store.record_retrieval(A, "run-1", "widget water", 3, 1, source_id="src-1")
        self.assertEqual(self.store.stats()["discoveries"], 2)

    def test_4_same_text_from_different_sources_stays_distinct(self):
        one = self.store.add_primary(A, source_id="src-1", source_title="Talk one")
        two = self.store.add_primary(A, source_id="src-2", source_title="Talk two")
        self.assertNotEqual(one, two)
        self.assertEqual(self.store.stats()["units_by_layer"][rm.PRIMARY], 2)

    def test_5_overlapping_or_contained_passages_are_not_merged(self):
        full = "One. Two. Three. Four."
        units = {self.store.add_primary(t, source_id="src-1")
                 for t in (full, "Three. Four. Five.", "Two. Three.")}
        self.assertEqual(len(units), 3)
        self.assertEqual({self.store.unit(u)["text"] for u in units},
                         {full, "Three. Four. Five.", "Two. Three."})


class SelectionHistory(MemoryTest):
    def test_6_and_7_roles_and_requirement_coverage_across_runs(self):
        self.run_("run-1", requirements=[{"id": "r1", "kind": "procedure", "text": "how to make it"}])
        self.run_("run-2", requirements=[{"id": "r1", "kind": "claim", "text": "whether it works"},
                                         {"id": "r2", "kind": "quantity", "text": "how much"}])
        self.run_("run-3")
        uid = self.store.add_primary(A, source_id="src-1")
        self.store.add_selection(uid, "run-1", 1, "h2", "CORE", ["r1"], "base recipe", True, False, True)
        self.store.add_selection(uid, "run-2", 1, "h5", "SUPPORT", ["r1", "r2"], "context", True)
        self.store.add_selection(uid, "run-3", 1, "h1", "DROP", [], "unrelated here", False)
        roles = [s["role"] for s in self.store.unit(uid)["selections"]]
        self.assertEqual(sorted(roles), ["CORE", "DROP", "SUPPORT"])
        self.assertEqual(self.store.mixed_selection()[0]["unit_id"], uid)
        covered = {(c["run_id"], c["requirement_id"], c["text"]) for c in self.store.requirements_covered(uid)}
        self.assertEqual(covered, {("run-1", "r1", "how to make it"), ("run-2", "r1", "whether it works"),
                                   ("run-2", "r2", "how much")})
        self.assertEqual(self.store.stats()["selector_roles"], {"CORE": 1, "DROP": 1, "SUPPORT": 1})

    def test_unrecorded_selection_fields_stay_null(self):
        self.run_("run-1")
        uid = self.store.add_primary(B, source_id="src-2")
        self.store.add_selection(uid, "run-1", 1, "h1", kept=True)
        s = self.store.unit(uid)["selections"][0]
        self.assertEqual((s["role"], s["covers"], s["reason"], s["final_evidence"]), (None, None, None, None))


class Search(MemoryTest):
    def setUp(self):
        super().setUp()
        self.p = self.store.add_primary(A, source_id="src-1", source_title="Talk one")
        self.s = self.store.add_secondary("A forum member says the widget needs warm water too.",
                                          platform="forum", author="member1")
        self.d = self.store.add_derived("Warm water is needed before the widget dissolves.",
                                        "combined_inference", supports=[self.p])

    def test_8_full_text_search_finds_saved_memory(self):
        hits = self.store.search("widget warm water")
        self.assertEqual({h["unit_id"] for h in hits}, {self.p, self.s, self.d})
        self.assertEqual(self.store.search("nonexistentword"), [])

    def test_9_results_carry_their_layer(self):
        layers = {h["unit_id"]: h["layer"] for h in self.store.search("widget")}
        self.assertEqual(layers, {self.p: rm.PRIMARY, self.s: rm.SECONDARY, self.d: rm.DERIVED})
        only = self.store.search("widget", layer=rm.SECONDARY)
        self.assertEqual([h["unit_id"] for h in only], [self.s])
        # any-word fallback when no unit contains every word
        self.assertTrue(self.store.search("widget zebra"))


class Layers(MemoryTest):
    def test_10_derived_unit_links_to_several_primary_units(self):
        a = self.store.add_primary(A, source_id="src-1")
        b = self.store.add_primary(B, source_id="src-2")
        d = self.store.add_derived("The widget and the gadget both depend on heat.", "combined_inference",
                                   supports=[a, b], producer="analysis-2026-01")
        links = {(r["rel_type"], r["other"]) for r in self.store.unit(d)["relationships"]}
        self.assertEqual(links, {("derived_from", a), ("derived_from", b)})
        self.assertEqual(self.store.dependents(a), [d])
        self.assertEqual(self.store.unit(d)["kind"], "combined_inference")
        with self.assertRaises(ValueError):
            self.store.add_derived("unsupported claim", "combined_inference", supports=[])

    def test_11_secondary_stays_secondary(self):
        s = self.store.add_secondary("Community note about the gadget.", platform="chat",
                                     source_title="Group thread", url="https://example.invalid/1")
        unit = self.store.unit(s)
        self.assertEqual((unit["layer"], unit["platform"]), (rm.SECONDARY, "chat"))
        self.assertTrue(s.startswith("S-"))
        with self.assertRaises(ValueError):  # secondary cannot stand in for primary support
            self.store.add_derived("claim", "combined_inference", supports=[s])
        p = self.store.add_primary(B, source_id="src-2")
        d = self.store.add_derived("claim", "source_lead", supports=[p], inspired_by=[s])
        self.assertIn(("inspired_by", s), {(r["rel_type"], r["other"]) for r in self.store.unit(d)["relationships"]})
        with self.assertRaises(ValueError):
            self.store.add_secondary("no platform given", platform="")

    def test_relationships_are_generic_and_validated(self):
        a = self.store.add_primary(A, source_id="src-1")
        b = self.store.add_primary(C, source_id="src-3")
        self.store.add_relationship(b, "later_addition_to", a, note="dated later")
        self.store.add_relationship(b, "later_addition_to", a)  # idempotent
        self.assertEqual(self.store.stats()["relationships"], {"later_addition_to": 1})
        for bad in ("Not Snake", ""):
            with self.assertRaises(ValueError):
                self.store.add_relationship(a, bad, b)
        with self.assertRaises(ValueError):
            self.store.add_relationship(a, "clarifies", "P-missing")


def write(folder, name, data):
    (folder / name).write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")


class Import(MemoryTest):
    def test_12_old_incomplete_run_imports_without_inventing_fields(self):
        """An older run: no plan file, no trace, no run.jsonl; candidates with found_by only; the
        selector output has ids but no roles, covers or coverage."""
        old = self.tmp / "20240101-000000-old-question"
        old.mkdir()
        write(old, "question.txt", "An old question?")
        write(old, "search-results-1.json", {"queries": ["old query"], "candidates": [
            {"hit_id": "h1", "source_id": "src-1", "text": A, "found_by": [{"query": "old query", "rank": 2}]},
            {"hit_id": "h2", "source_id": "src-2", "text": B, "found_by": [{"query": "old query", "rank": 5}]}]})
        write(old, "selector-1.json", {"selected_hit_ids": ["h1"], "context_hit_ids": []})
        report = rm.import_path(self.store, old, titles={})
        self.assertEqual(report["units_new"], 2)
        run = self.store.db.execute("SELECT * FROM runs").fetchone()
        self.assertEqual((run["question"], run["depth"], run["started_at"]), ("An old question?", None, None))
        self.assertEqual([dict(s)["query"] for s in self.store.db.execute("SELECT query FROM run_searches")],
                         ["old query"])
        sel = {s["hit_id"]: s for s in self.store.db.execute("SELECT * FROM selections")}
        self.assertEqual((sel["h1"]["kept"], sel["h1"]["role"], sel["h1"]["covers"]), (1, None, None))
        self.assertEqual((sel["h2"]["kept"], sel["h2"]["final_evidence"]), (0, None))
        unit = self.store.unit(sel["h1"]["unit_id"])
        self.assertEqual((unit["source_title"], unit["source_date"]), (None, None))

    def test_trace_without_run_files_stores_no_previews_as_passages(self):
        lone = self.tmp / "lone"
        lone.mkdir()
        write(lone, "evidence-trace.json", {"raw": [], "candidates": [
            {"id": "h1", "raw_ids": ["r1.1"], "source": "Talk one", "preview": "Alpha passage…",
             "role": "CORE", "kept": True}]})
        report = rm.import_path(self.store, lone / "evidence-trace.json", titles={})
        self.assertEqual((report["units_new"], report["skipped_preview_only"]), (0, 1))
        self.assertEqual(self.store.stats()["units"], 0)

    def test_13_init_and_reimport_are_idempotent_and_ids_deterministic(self):
        folder = self.tmp / "20240202-101010-q"
        folder.mkdir()
        write(folder, "question.txt", "Q?")
        write(folder, "run.jsonl", json.dumps({"time": "2024-02-02T10:10:10", "event": "start"}) + "\n")
        write(folder, "search-results-1.json", {"queries": ["q"], "raw_hits": [
            {"raw_id": "r1.1", "query": "q", "source_id": "src-1", "text": A, "rank": 1},
            {"raw_id": "r1.2", "query": "q", "source_id": "src-2", "text": C, "rank": 2}],
            "candidates": [{"hit_id": "h1", "source_id": "src-1", "text": A},
                           {"hit_id": "h2", "source_id": "src-2", "text": C}]})
        rm.import_path(self.store, folder, titles={})
        before = self.store.stats()
        self.store.init()
        rm.import_path(self.store, folder, titles={})
        self.assertEqual(self.store.stats(), before)
        ids = sorted(r["unit_id"] for r in self.store.db.execute("SELECT unit_id FROM units"))
        with rm.MemoryStore(self.tmp / "fresh.db") as fresh:
            rm.import_path(fresh, folder, titles={})
            self.assertEqual(sorted(r["unit_id"] for r in fresh.db.execute("SELECT unit_id FROM units")), ids)
        found = [h["unit_id"] for h in self.store.search("widget")]
        self.store.rebuild_search_index()
        self.assertEqual([h["unit_id"] for h in self.store.search("widget")], found)
        self.assertEqual(self.store.unit(ids[0])["first_seen_at"] or self.store.unit(ids[1])["first_seen_at"],
                         "2024-02-02T10:10:10")


QUESTION = "Does the morning rinse cause streaking?"
ANECDOTE = "My morning rinse caused streaking every day for a week, then it stopped."
COMMON_WORD_ONLY = ["The morning is the best time for a walk.", "Morning light helps you wake up.",
                    "Every morning I stretch before work."]


class Relevance(MemoryTest):
    def test_lookup_prefers_the_full_match_over_common_term_matches(self):
        self.store.add_primary("A: The morning rinse causes streaking for about an hour.", "s1",
                               "Workshop, 2004")
        self.store.add_secondary_claim(ANECDOTE, "chat", "chat:group/900", "anecdote_or_experiment",
                                       author="member", source_date="2026-08-20",
                                       context="[Chat · member] " + ANECDOTE)
        for n, text in enumerate(COMMON_WORD_ONLY):
            self.store.add_secondary_claim(text, "forum", f"forum:board/{n}", "practical_synthesis")
        self.store.add_secondary_claim("Dried receipts keeps for years in a sealed jar.", "forum",
                                       "forum:board/9", "external_fact")
        found = self.store.relevant(QUESTION, rm.SECONDARY, limit=10)
        texts = [self.store.db.execute("SELECT text FROM units WHERE unit_id = ?",
                                       (f["unit_id"],)).fetchone()[0] for f in found]
        self.assertEqual(texts, [ANECDOTE])
        self.assertEqual(rm.query_terms(QUESTION), ["morning", "rinse", "cause", "streaking"])


class Repository(unittest.TestCase):
    def test_14_database_files_are_git_ignored(self):
        if shutil.which("git") is None or not (REPO / ".git").exists():
            self.skipTest("needs git and a git checkout")
        default = rm.DEFAULT_DB.relative_to(REPO)
        for path in (default, default.with_name(default.name + "-journal")):
            check = subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=REPO)
            self.assertEqual(check.returncode, 0, f"{path} is not git-ignored")
        tracked = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True).stdout
        self.assertFalse([f for f in tracked.splitlines() if f.endswith((".db", ".sqlite", ".sqlite3"))])


if __name__ == "__main__":
    unittest.main()
