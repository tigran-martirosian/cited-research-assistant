"""Offline tests for the pipeline's pure normalization steps: the planner's output (answer
requirements, searches and their caps), the selector's output (roles and per-requirement
coverage), duplicate detection, repair requests, and what the prompts and schemas must contain.
No NotebookLM and no model call. Model behavior itself is covered by the opt-in live tests in
test_live_contracts.py.

Run from the repository root:  python -m unittest discover -s tests -t .
"""
import unittest

try:
    from .helpers import REPO, Stub, candidate, offline, requirement, research
except ImportError:  # run with tests/ as the top-level directory
    from helpers import REPO, Stub, candidate, offline, requirement, research

COMPOUND_PLAN = {
    "requirements": [
        {"id": "r1", "kind": "purpose", "text": "why the morning rinse is recommended"},
        {"id": "r2", "kind": "procedure", "text": "complete preparation of the morning rinse"},
        {"id": "r3", "kind": "applicability", "text": "changes for someone a few weeks in"},
    ],
    "depth": "normal",
    "searches": [
        {"query": "why the morning rinse is recommended", "covers": ["r1"]},
        {"query": "morning rinse preparation recipe additions", "covers": ["r2"]},
        {"query": "morning rinse first weeks of travel", "covers": ["r3"]},
    ],
}

class PlanNormalization(unittest.TestCase):
    def test_compound_question_keeps_one_requirement_per_obligation(self):
        depth, reqs, searches, notes = research.normalize_plan(COMPOUND_PLAN, "q")
        self.assertEqual(depth, "normal")
        self.assertEqual([(r["id"], r["kind"]) for r in reqs],
                         [("r1", "purpose"), ("r2", "procedure"), ("r3", "applicability")])
        self.assertEqual([s["covers"] for s in searches], [["r1"], ["r2"], ["r3"]])
        self.assertNotIn("requirements_uncovered", notes)

    def test_plan_without_requirements_falls_back_to_the_question(self):
        _, reqs, searches, notes = research.normalize_plan(
            {"depth": "direct", "searches": [{"query": "x y", "covers": []}]}, " Is X   Y? ")
        self.assertEqual(reqs, [requirement("r1", "claim", "Is X Y?")])
        self.assertEqual(searches[0]["covers"], ["r1"])  # a lone requirement is implied
        self.assertTrue(notes["requirements_fallback"])

    def test_ids_are_renumbered_and_unknown_kinds_or_covers_are_dropped(self):
        out = {"depth": "normal",
               "requirements": [{"id": "a", "kind": "purpose", "text": "why"},
                                {"id": "b", "kind": "", "text": ""},
                                {"id": "c", "kind": "bogus", "text": "how"}],
               "searches": [{"query": "q1", "covers": ["a", "zzz"]}, {"query": "q2", "covers": ["c"]}]}
        _, reqs, searches, _ = research.normalize_plan(out, "q")
        self.assertEqual(reqs, [requirement("r1", "purpose", "why"), requirement("r2", "claim", "how")])
        self.assertEqual([s["covers"] for s in searches], [["r1"], ["r2"]])

    def test_repeated_query_merges_its_covers(self):
        out = {"depth": "normal", "requirements": COMPOUND_PLAN["requirements"],
               "searches": [{"query": "Rinse recipe", "covers": ["r2"]},
                            {"query": "rinse  recipe", "covers": ["r3"]}]}
        _, _, searches, _ = research.normalize_plan(out, "q")
        self.assertEqual(searches, [{"query": "Rinse recipe", "covers": ["r2", "r3"], "type": "literal"}])

    def test_capping_keeps_searches_that_add_coverage(self):
        limit = research.DEPTH_MAX_SEARCHES["direct"]
        same = [{"query": f"a{i}", "covers": ["r1"]} for i in range(limit + 1)]
        out = {"depth": "direct", "requirements": COMPOUND_PLAN["requirements"][:2],
               "searches": same + [{"query": "c", "covers": ["r2"]}]}
        _, _, searches, notes = research.normalize_plan(out, "q")
        self.assertEqual([s["query"] for s in searches], [f"a{i}" for i in range(limit - 1)] + ["c"])
        self.assertEqual(notes["searches_capped"]["dropped"], [f"a{limit - 1}", f"a{limit}"])
        self.assertNotIn("requirements_uncovered", notes)

    def test_capping_reports_requirements_left_uncovered(self):
        limit = research.DEPTH_MAX_SEARCHES["direct"]
        ids = [f"r{i}" for i in range(1, limit + 2)]  # one requirement more than the cap
        out = {"depth": "direct",
               "requirements": [requirement(i, "claim", f"part {i}") for i in ids],
               "searches": [{"query": f"q {i}", "covers": [i]} for i in ids]}
        _, _, searches, notes = research.normalize_plan(out, "q")
        self.assertEqual(len(searches), limit)
        self.assertEqual(notes["requirements_uncovered"], [ids[-1]])


class SelectionNormalization(unittest.TestCase):
    REQS = [requirement("r1", "claim", "whether the rinse causes streaking")]
    CANDS = [candidate("h1", "s1", "clear statement"), candidate("h2", "s2", "same rule again"),
             candidate("h3", "s3", "unrelated")]

    def test_same_proposition_redundancy_keeps_only_non_drop_hits(self):
        """A redundant passage classed DROP never reaches the reasoner, even if the
        selector also lists it, and coverage cites only the kept passage."""
        out = {"decisions": [{"hit_id": "h1", "role": "CORE", "covers": ["r1"], "reason": "rule"},
                             {"hit_id": "h2", "role": "DROP", "covers": ["r1"], "reason": "same rule"},
                             {"hit_id": "h3", "role": "DROP", "covers": [], "reason": "unrelated"}],
               "coverage": [{"requirement_id": "r1", "status": "covered", "hit_ids": ["h1", "h2"],
                             "missing": ""}],
               "selected_hit_ids": ["h1", "h2"], "context_hit_ids": []}
        ids, _, decisions, coverage, notes = research.normalize_selection(out, self.CANDS, self.REQS)
        self.assertEqual(ids, ["h1"])
        self.assertEqual(decisions["h2"]["covers"], [])
        self.assertEqual(coverage, [{"requirement_id": "r1", "status": "covered",
                                     "hit_ids": ["h1"], "missing": ""}])
        self.assertIn(("selector_drop_conflict", {"hit_ids": ["h2"]}), notes)

    def test_coverage_claimed_only_by_dropped_hits_becomes_missing(self):
        """A requirement is never counted as covered by evidence that was dropped."""
        reqs = [requirement("r1", "procedure", "preparation"),
                requirement("r2", "applicability", "changes at six months")]
        out = {"decisions": [{"hit_id": "h1", "role": "CORE", "covers": ["r1"], "reason": "recipe"},
                             {"hit_id": "h3", "role": "DROP", "covers": [], "reason": "other stage"}],
               "coverage": [{"requirement_id": "r1", "status": "covered", "hit_ids": ["h1"], "missing": ""},
                            {"requirement_id": "r2", "status": "covered", "hit_ids": ["h3"], "missing": ""}],
               "selected_hit_ids": ["h1"], "context_hit_ids": []}
        _, _, _, coverage, notes = research.normalize_selection(out, self.CANDS, reqs)
        self.assertEqual(coverage[1]["status"], "missing")
        self.assertEqual(coverage[1]["hit_ids"], [])
        self.assertTrue(any(event == "selector_coverage_unsupported" for event, _ in notes))

    def test_unrated_requirement_is_unassessed_and_unknown_covers_are_ignored(self):
        reqs = [requirement("r1", "purpose", "why"), requirement("r2", "procedure", "how")]
        out = {"decisions": [{"hit_id": "h1", "role": "CORE", "covers": ["r2", "r9"], "reason": "x"}],
               "coverage": [{"requirement_id": "r1", "status": "missing", "hit_ids": [],
                             "missing": "no purpose stated"}],
               "selected_hit_ids": ["h1"], "context_hit_ids": []}
        _, _, decisions, coverage, _ = research.normalize_selection(out, self.CANDS, reqs)
        self.assertEqual(decisions["h1"]["covers"], ["r2"])
        self.assertEqual(coverage[0]["missing"], "no purpose stated")
        self.assertEqual(coverage[1], {"requirement_id": "r2", "status": "unassessed",
                                       "hit_ids": ["h1"], "missing": ""})

    def test_follow_up_round_may_select_nothing(self):
        premises = [{"id": "p1", "kind": "premise", "for": "r2", "text": "amount", "search": "q"}]
        out = {"decisions": [], "coverage": [], "selected_hit_ids": [], "context_hit_ids": []}
        ids, _, _, coverage, _ = research.normalize_selection(out, self.CANDS, premises, follow_up=True)
        self.assertEqual(ids, [])
        self.assertEqual(coverage[0]["for"], "r2")


class Dedupe(unittest.TestCase):
    def test_different_predicates_about_the_same_outcome_stay_separate(self):
        """Two passages of one source about the same subject and outcome but with
        different mechanisms are separate candidates; only identical or contained text merges."""
        needs = {"source_id": "s1", "text": "The rinse needs a lot of water to clear, so the account dries out.",
                 "start": 0, "end": 60}
        pulls = {"source_id": "s1", "text": "The reed shoots pull moisture out of the account directly.",
                 "start": 400, "end": 455}
        self.assertIsNone(research.duplicate_reason(needs, pulls))
        self.assertEqual(research.duplicate_reason(needs, dict(needs)), "identical text")


class RepairRequest(unittest.TestCase):
    def test_repair_request_with_unknown_requirement_is_kept_untied(self):
        reqs = [requirement("r1", "claim", "x")]
        stub = Stub({}, [], None, {}, {})
        with offline(stub):
            run = research.Run("q")
            request = research.repair_request(
                run, {"research_request": [{"requirement_id": "r7", "premise": "p", "search": "s"},
                                           {"requirement_id": "r1", "premise": "p", "search": "done before"}]},
                ["done before"], reqs)
        self.assertEqual(request, [{"requirement_id": "", "premise": "p", "search": "s"}])


class PromptContracts(unittest.TestCase):
    PROMPTS = {name: (REPO / "prompts" / f"{name}.txt").read_text(encoding="utf-8")
               for name in ("planner", "selector", "reasoner")}

    def test_prompts_state_the_reasoning_contracts(self):
        expected = {
            "planner": ["<requirements>", "Match the question's granularity",
                        "reconstructing an actionable recommendation", "its own applicability requirement"],
            "selector": ["SATURATION", "COVERAGE", "Do not over-compress dated evidence",
                         "Topical relevance is not CORE",
                         "A rule for one group, condition, or stage does not establish a rule for a different one"],
            "reasoner": ["SOURCE MODEL", "LAYERED RECOMMENDATIONS", "Direction is not quantity",
                         "QUESTION GRANULARITY", "Newer evidence does not automatically override older evidence",
                         "Evidence for one condition is not evidence for another",
                         "the user needs an exact amount and the evidence gives only a direction"],
        }
        for name, phrases in expected.items():
            for phrase in phrases:
                self.assertIn(phrase, self.PROMPTS[name], f"{name}: {phrase}")

    def test_schemas_carry_requirements_coverage_and_one_repair_round(self):
        self.assertIn("requirements", research.PLANNER_SCHEMA["required"])
        decision = research.SELECTOR_SCHEMA["properties"]["decisions"]["items"]
        self.assertIn("covers", decision["required"])
        self.assertIn("coverage", research.SELECTOR_SCHEMA["required"])
        item = research.REPAIRABLE_SCHEMA["properties"]["research_request"]["items"]
        self.assertIn("requirement_id", item["required"])
        self.assertEqual(research.REPAIRABLE_SCHEMA["properties"]["research_request"]["maxItems"],
                         research.REPAIR_MAX_SEARCHES)
        self.assertNotIn("research_request", research.REASONER_SCHEMA["properties"])


if __name__ == "__main__":
    unittest.main()
