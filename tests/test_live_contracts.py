"""Live contract tests: the real planner, selector and reasoner (Claude through the CLI login)
on synthetic fixtures. No NotebookLM is involved. Opt-in, because each case is a model call:

    CRA_LIVE_EVALS=1 python -m unittest tests.test_live_contracts -v

Assertions are on structured behavior (requirements, kept hits, coverage, repair requests,
invented percentages, presence of key terms), never on exact wording.
"""
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from .helpers import candidate, evidence, repair_done, requirement, research
    from . import test_synthesis as syn
except ImportError:  # run with tests/ as the top-level directory
    from helpers import candidate, evidence, repair_done, requirement, research
    import test_synthesis as syn

LIVE = bool(os.environ.get("CRA_LIVE_EVALS"))

BASE_1999 = ("Q: How do you make the morning rinse?\nA: Steep two handfuls of soapwort leaves and one "
             "handful of ivy leaves in a quart of warm rain water for an hour. Then strain it.")
ADDITION_2006 = ("A: I've changed how I make the morning rinse. After straining, I now stir a spoonful "
                 "of taxi into every batch, and I tell everyone to do the same.")
UNRESOLVED = re.compile(
    r"not (?:establish|address|specif|cover|say|give|state|resolve)|unresolved|"
    r"(?:does ?n[o']t|do ?n[o']t) (?:address|say|give|specify|cover|state|establish)|"
    r"no (?:passage|source|evidence|statement|retrieved)|is ?n[o']t (?:addressed|covered|specified|stated|established)",
    re.I)


def note(case, detail):
    print(f"\n[live] {case}: {detail}", file=sys.stderr)


@unittest.skipUnless(LIVE, "needs live model access: set CRA_LIVE_EVALS=1 with a signed-in claude CLI")
class LiveContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_ = research.Run("live contract eval")
        self.run_.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    # ---- planner -------------------------------------------------------------------------

    def plan(self, question):
        depth, reqs, searches = research.plan(self.run_, question)
        note(question, f"depth={depth} requirements={[(r['kind'], r['text']) for r in reqs]} "
                       f"searches={[(s['query'], s['covers']) for s in searches]}")
        return depth, reqs, searches

    def test_1_compound_procedural_question_plans_separate_requirements(self):
        _, reqs, searches = self.plan(
            "Why did the corpus author recommend the morning rinse, how should it be made, and how should that "
            "change for someone who travels fine yarn and is only a few weeks into travel?")
        kinds = [r["kind"] for r in reqs]
        self.assertTrue({"purpose", "procedure", "applicability"} <= set(kinds), kinds)
        self.assertGreaterEqual(kinds.count("applicability"), 2, "each stated condition is its own requirement")
        covered = {c for s in searches for c in s["covers"]}
        self.assertEqual([r["id"] for r in reqs if r["id"] not in covered], [], "every requirement searched")

    def test_9a_category_question_plans_composition_only(self):
        _, reqs, _ = self.plan("Which leaves go into the morning rinse?")
        self.assertEqual([r["kind"] for r in reqs], ["composition"])

    def test_9b_how_to_question_plans_complete_procedure(self):
        _, reqs, _ = self.plan("How do I make the morning rinse?")
        kinds = [r["kind"] for r in reqs]
        self.assertIn("procedure", kinds)
        self.assertNotIn("applicability", kinds, "no condition was stated")

    # ---- selector ------------------------------------------------------------------------

    def select(self, question, reqs, cands, titles):
        selected, _, coverage = research.select(self.run_, question, "normal", cands, titles, reqs)
        kept = [c["hit_id"] for c in selected]
        note(question, f"kept={kept} coverage={[(c['requirement_id'], c['status']) for c in coverage]}")
        return kept, coverage

    def test_2_later_additive_update_survives_with_the_base_recipe(self):
        kept, coverage = self.select(
            "How should the morning rinse be made?",
            [requirement("r1", "procedure", "complete preparation of the morning rinse")],
            [candidate("h1", "s1", BASE_1999), candidate("h2", "s2", ADDITION_2006),
             candidate("h3", "s3", "A: The morning rinse is pleasant work on hot days. I do mine on the porch."),
             candidate("h4", "s4", "A: Soapwort grows wild along the riverbanks in spring.")],
            {"s1": "Lecture, 1999", "s2": "Workshop, 2006", "s3": "Newsletter, 2003", "s4": "Walk, 2001"})
        self.assertTrue({"h1", "h2"} <= set(kept), kept)
        self.assertFalse({"h3", "h4"} & set(kept), kept)
        self.assertNotEqual(coverage[0]["status"], "missing")

    def test_6_same_proposition_compresses_to_the_clearer_passage(self):
        kept, _ = self.select(
            "Does the morning rinse cause streaking?",
            [requirement("r1", "claim", "whether the morning rinse causes streaking, and why")],
            [candidate("h1", "s1", "Q: Does the rinse leave streaks?\nA: Yes. The morning rinse "
                                   "causes streaking on most account for about an hour, because the "
                                   "soapwort foams unevenly."),
             candidate("h2", "s2", "A: Like I said before, the rinse causes streaking. It does."),
             candidate("h3", "s3", "A: Rain water is collected in March when the barrels fill.")],
            {"s1": "Lecture, 2002", "s2": "Q&A session, 2002", "s3": "Workshop, 2004"})
        self.assertIn("h1", kept)
        self.assertNotIn("h2", kept, "a less clear restatement of the same rule")
        self.assertNotIn("h3", kept)

    def test_7_different_predicates_for_the_same_outcome_both_survive(self):
        kept, _ = self.select(
            "Does the morning rinse dry out the account?",
            [requirement("r1", "claim", "whether the morning rinse dries out the account, and why")],
            [candidate("h1", "s1", "A: The rinse needs a lot of water to clear. If you don't "
                                   "add extra water with it, the account dries out."),
             candidate("h2", "s2", "A: The ivy in the rinse pulls moisture straight out of the account. "
                                   "That's a separate thing from clearing."),
             candidate("h3", "s3", "A: Dry account in winter mostly comes from cold wind.")],
            {"s1": "Workshop, 2005", "s2": "Lecture, 2009", "s3": "Newsletter, 2003"})
        self.assertTrue({"h1", "h2"} <= set(kept), kept)
        self.assertNotIn("h3", kept)

    # ---- reasoner ------------------------------------------------------------------------

    def reason(self, question, reqs, coverage, ev, repaired=None):
        out = research.reason(self.run_, "reasoner-1" if repaired is None else "reasoner-2",
                              question, "normal", ev, reqs, coverage, repaired)
        answer = (out.get("answer") or "").strip()
        note(question, f"request={out.get('research_request') or []} answer={answer[:600]!r}")
        return answer, out.get("research_request") or []

    @staticmethod
    def covered(*ids, status="covered", missing=""):
        return [{"requirement_id": i, "status": status, "hit_ids": ["h1"], "missing": missing} for i in ids]

    def test_3_condition_specific_change_does_not_replace_the_general_rule(self):
        answer, request = self.reason(
            "How much soapwort goes into the morning rinse?",
            [requirement("r1", "quantity", "amount of soapwort in the morning rinse")], self.covered("r1"),
            evidence(("Lecture, 1999", "A: For the morning rinse, use two handfuls of soapwort leaves for "
                                       "each quart of rain water."),
                     ("Workshop, 2008", "A: In the morning rinse for fine silk I now use "
                                        "only half a handful of soapwort per quart.")))
        self.assertEqual(request, [], "evidence is sufficient; no repair expected")
        self.assertRegex(answer, re.compile(r"two handfuls|2 handfuls", re.I))

    def test_4_directional_quantity_is_not_turned_into_a_percentage(self):
        ev = evidence(("Lecture, 1999", "A: The morning rinse is 60% rain water, 30% soapwort leaves and "
                                        "10% ivy leaves."),
                      ("Workshop, 2006", "A: Fine yarn should get a lot more ivy in the morning rinse."))
        request = [{"requirement_id": "r2", "premise": "exact amount of ivy for fine yarn",
                    "search": "morning rinse ivy amount fine yarn"}]
        answer, _ = self.reason(
            "What are the exact proportions of the morning rinse for fine yarn?",
            [requirement("r1", "quantity", "exact proportions of the morning rinse"),
             requirement("r2", "applicability", "changes for fine yarn")],
            self.covered("r1") + [{"requirement_id": "r2", "status": "partial", "hit_ids": ["h2"],
                                   "missing": "direction only, no amount"}],
            ev, repaired=repair_done(request))
        self.assertTrue(answer)
        self.assertEqual(research.ungrounded_percentages(answer, ev), [], f"invented percentage in: {answer}")

    def test_5_incomplete_preparation_requests_repair(self):
        answer, request = self.reason(
            "How do I make the morning rinse?",
            [requirement("r1", "procedure", "complete preparation of the morning rinse")],
            [{"requirement_id": "r1", "status": "partial", "hit_ids": ["h1", "h2"],
              "missing": "how the added powder is used"}],
            evidence(("Lecture, 1999", BASE_1999),
                     ("Workshop, 2007", "A: Everyone should now add the binding powder to the morning "
                                        "rinse. Without it the rinse does very little.")))
        self.assertGreaterEqual(len(request), 1, "missing component should trigger the repair round")
        self.assertEqual(answer, "")

    def test_8_unestablished_user_condition_is_marked_unresolved(self):
        request = [{"requirement_id": "r2", "premise": "amount at six months into travel",
                    "search": "morning rinse amount after six months"}]
        answer, _ = self.reason(
            "How much morning rinse should I use if I'm six months into travel?",
            [requirement("r1", "quantity", "amount of morning rinse per batch"),
             requirement("r2", "applicability", "six months into travel")],
            [{"requirement_id": "r1", "status": "partial", "hit_ids": ["h1", "h2"],
              "missing": "no general daily amount"},
             {"requirement_id": "r2", "status": "missing", "hit_ids": [],
              "missing": "no passage addresses six months"}],
            evidence(("Lecture, 2000", "A: In the first two weeks of travel, use one cup of the "
                                       "morning rinse per batch."),
                     ("Workshop, 2004", "A: Fine yarn can take up to three cups of the morning rinse per batch.")),
            repaired=repair_done(request))
        self.assertRegex(answer, UNRESOLVED)

    def test_9c_category_question_answer_stays_in_the_category(self):
        answer, _ = self.reason(
            "Which leaves go into the morning rinse?",
            [requirement("r1", "composition", "leaves that go into the morning rinse")], self.covered("r1"),
            evidence(("Lecture, 1999", BASE_1999), ("Workshop, 2006", ADDITION_2006)))
        self.assertRegex(answer, re.compile(r"soapwort", re.I))
        self.assertRegex(answer, re.compile(r"ivy", re.I))
        self.assertNotRegex(answer, re.compile(r"taxi", re.I))

    def test_9d_how_to_answer_includes_additions_outside_the_category(self):
        answer, _ = self.reason(
            "How do I make the morning rinse?",
            [requirement("r1", "procedure", "complete preparation of the morning rinse")], self.covered("r1"),
            evidence(("Lecture, 1999", BASE_1999), ("Workshop, 2006", ADDITION_2006)))
        self.assertRegex(answer, re.compile(r"taxi", re.I))


    # ---- layered synthesis (fixtures from test_synthesis) --------------------------------

    def test_10_selector_scopes_the_exception_to_its_own_mechanism(self):
        selected, _, _ = research.select(self.run_, "Does travel strip the allowance from account?", "normal",
                                         [dict(c) for c in syn.ALLOWANCE_CANDS], syn.ALLOWANCE_TITLES, syn.ALLOWANCE_REQS)
        kept = [c["hit_id"] for c in selected]
        claims = self.run_.claims
        note("allowance selector", {i: (claims[i].get("predicate"), claims[i].get("scope"),
                                  claims[i].get("relations")) for i in kept})
        self.assertTrue({"h1", "h2", "h3"} <= set(kept), kept)
        self.assertNotEqual(research.squash(claims["h1"]["predicate"]),
                            research.squash(claims["h2"]["predicate"]), "distinct predicates")
        targets = {r["hit_id"] for r in claims["h3"].get("relations", []) if r["type"] == "qualifies"}
        self.assertNotIn("h1", targets, "the exception does not qualify the demand mechanism")

    def reason_map(self, question, reqs, cands, decisions, coverage, titles, repaired=None):
        entries, text = research.evidence_map(reqs, coverage, cands, decisions, titles)
        ev = "\n\n=====\n\n".join(f"SOURCE: {titles[c['source_id']]}\nPASSAGE [{c['hit_id']}]:\n{c['text']}"
                                     for c in cands)
        out = research.reason(self.run_, "reasoner-1" if repaired is None else "reasoner-2", question,
                              "normal", ev, reqs, coverage, repaired, evidence_map=text)
        answer = (out.get("answer") or "").strip()
        items, issues = research.check_synthesis(out, entries)
        note(question, f"request={out.get('research_request') or []} synthesis={items} "
                       f"issues={issues} answer={answer[:600]!r}")
        return answer, out.get("research_request") or [], issues, ev

    def test_11_fat_answer_keeps_both_mechanisms_without_a_threshold(self):
        ids, _, decisions, coverage, _ = research.normalize_selection(syn.ALLOWANCE_SELECTION, syn.ALLOWANCE_CANDS,
                                                                      syn.ALLOWANCE_REQS)
        answer, _, issues, ev = self.reason_map(
            "Does travel strip the allowance from account?", syn.ALLOWANCE_REQS,
            [c for c in syn.ALLOWANCE_CANDS if c["hit_id"] in ids], decisions, coverage, syn.ALLOWANCE_TITLES,
            repaired=repair_done([]))
        self.assertEqual(research.threshold_phrases(answer, ev), [], answer)
        self.assertRegex(answer, re.compile(r"pull", re.I))
        self.assertRegex(answer, re.compile(r"demand|need|requir", re.I))
        self.assertEqual([i for i in issues if i["issue"].startswith("exception")], [], issues)

    def test_12_claim_answer_layers_and_does_not_invent_the_missing_amount(self):
        ids, _, decisions, coverage, _ = research.normalize_selection(syn.CLAIM_SELECTION, syn.CLAIM_CANDS,
                                                                      syn.CLAIM_REQS)
        cands = [c for c in syn.CLAIM_CANDS if c["hit_id"] in ids]
        answer, request, issues, ev = self.reason_map(
            "How do I make the expense claim? My account is fine.", syn.CLAIM_REQS, cands, decisions, coverage,
            syn.CLAIM_TITLES)
        if not request:
            self.assertEqual(research.ungrounded_quantities(answer, ev), [], answer)
            self.assertRegex(answer, re.compile(r"direct deposit", re.I))
        self.assertEqual([i for i in issues if i["issue"] in ("individual_case_used_as_base",
                                                              "earlier_version_used_without_its_update")],
                         [], issues)


if __name__ == "__main__":
    unittest.main()
