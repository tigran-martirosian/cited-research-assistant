"""Offline tests for layered synthesis: the selector's claim structure, the evidence map built
from it, the reasoner's synthesis bookkeeping, the answer diagnostics and targeted repair.

Two cases with synthetic passages (not source quotes):

ALLOWANCE DEMAND: heavy travel use creates a demand for allowance (and weak payout when allowance runs
short), while a separate passage says parking pulls allowance out; a taxi exception qualifies the
parking mechanism only. The answer must keep both predicates, scope the exception, and not invent a
threshold such as "only in excess".

EXPENSE CLAIM: an older general formula, a later general change (direct deposit), guidance
for fine account, stage guidance and a single client's example. Synthesis layers what applies
instead of latest-wins, never turns the client example or the condition guidance into the
general rule, never treats the older formula as the default merely because it was retrieved, and
searches for a missing direct deposit amount instead of inventing it.

Run from the repository root:  python -m unittest tests.test_synthesis -v
"""
import copy
import unittest

try:
    from .helpers import candidate, requirement, research
except ImportError:  # run with tests/ as the top-level directory
    from helpers import candidate, requirement, research


def decision(hit_id, covers, predicate="", scope="general", applies_to="", relations=(),
             use="answer", role="CORE", reason="test"):
    return {"hit_id": hit_id, "role": role, "covers": list(covers), "reason": reason,
            "predicate": predicate, "scope": scope, "applies_to": applies_to,
            "relations": [f"{t}:{h}" for t, h in relations], "use": use}


def dropped(hit_id):
    return decision(hit_id, [], role="DROP", scope="unclear", use="context")


CLAIM_FIELDS = ("predicate", "scope", "applies_to", "relations", "use")


def selection(decisions, coverage):
    """Selector output in the schema's shape: role decisions for every hit, claims for kept hits."""
    kept = [d["hit_id"] for d in decisions if d["role"] != "DROP"]
    return {"decisions": [{k: d[k] for k in ("hit_id", "role", "covers", "reason")} for d in decisions],
            "claims": [{"hit_id": d["hit_id"], **{k: d[k] for k in CLAIM_FIELDS}}
                       for d in decisions if d["role"] != "DROP"],
            "coverage": coverage, "selected_hit_ids": kept, "context_hit_ids": []}


# ---- ALLOWANCE DEMAND ---------------------------------------------------------------------------

ALLOWANCE_DEMAND = ("A: When you use a lot of travel, the account needs allowance to hold that travel. It creates "
              "a demand for allowance, and if you don't have enough of it the payout comes out weak.")
ALLOWANCE_SODA_ASH = "A: Parking pulls allowance out of the account. It binds with it and carries it out."
ALLOWANCE_TAXI = "A: Taxi doesn't pull allowance out the way parking does."
ALLOWANCE_OFFTOPIC = "A: Strong travel gives the deepest reds and browns."
ALLOWANCE_TITLES = {"s1": "Lecture, 2000", "s2": "Workshop, 2004", "s3": "Q&A, 2006", "s4": "Newsletter, 2001"}
ALLOWANCE_REQS = [requirement("r1", "claim", "whether and how travel strips allowance from the account")]
ALLOWANCE_CANDS = [candidate("h1", "s1", ALLOWANCE_DEMAND), candidate("h2", "s2", ALLOWANCE_SODA_ASH),
             candidate("h3", "s3", ALLOWANCE_TAXI), candidate("h4", "s4", ALLOWANCE_OFFTOPIC)]
ALLOWANCE_SELECTION = selection(
    [decision("h1", ["r1"], "creates a demand for allowance"),
     decision("h2", ["r1"], "pulls allowance out", relations=[("compatible", "h1")]),
     decision("h3", ["r1"], "doesn't pull allowance out", scope="condition", applies_to="taxi",
              relations=[("qualifies", "h2")]),
     dropped("h4")],
    [{"requirement_id": "r1", "status": "covered", "hit_ids": ["h1", "h2", "h3"], "missing": "",
      "gap": "none", "lead_hit_ids": []}])
ALLOWANCE_EVIDENCE = "\n".join((ALLOWANCE_DEMAND, ALLOWANCE_SODA_ASH, ALLOWANCE_TAXI))


def allowance_map(out=ALLOWANCE_SELECTION):
    ids, _, decisions, coverage, notes = research.normalize_selection(out, ALLOWANCE_CANDS, ALLOWANCE_REQS)
    kept = [c for c in ALLOWANCE_CANDS if c["hit_id"] in ids]
    entries, text = research.evidence_map(ALLOWANCE_REQS, coverage, kept, decisions, ALLOWANCE_TITLES)
    return entries, text, decisions, notes


class AllowanceDemand(unittest.TestCase):
    def test_distinct_predicates_survive_verbatim(self):
        entries, text, decisions, _ = allowance_map()
        self.assertTrue(decisions["h1"]["predicate_verbatim"])
        self.assertTrue(decisions["h2"]["predicate_verbatim"])
        self.assertEqual([p["predicate"] for p in entries[0]["predicates"]],
                         ["creates a demand for allowance", "pulls allowance out",
                          "doesn't pull allowance out"])
        self.assertIn('Different predicates, not interchangeable: "creates a demand for allowance" (h1); '
                      '"pulls allowance out" (h2)', text)

    def test_normalized_predicate_is_marked_not_verbatim(self):
        out = copy.deepcopy(ALLOWANCE_SELECTION)
        out["claims"][1]["predicate"] = "strips allowance from the account"
        entries, text, decisions, _ = allowance_map(out)
        self.assertFalse(decisions["h2"]["predicate_verbatim"])
        self.assertIn('"strips allowance from the account" (not verbatim in the passage)', text)

    def test_taxi_exception_qualifies_the_soda_ash_mechanism_only(self):
        entries, text, _, _ = allowance_map()
        self.assertEqual([(x["from"], x["type"], x["to"]) for x in entries[0]["relations"]],
                         [("h2", "compatible", "h1"), ("h3", "qualifies", "h2")])
        self.assertIn('h3 qualifies h2 only (the statement "pulls allowance out"); it does not '
                      "qualify any other statement", text)
        self.assertNotIn("h3 qualifies h1", text)
        self.assertIn("Condition-specific: h3 (taxi)", text)

    def test_relations_to_dropped_or_unknown_hits_are_rejected(self):
        out = copy.deepcopy(ALLOWANCE_SELECTION)
        out["claims"][2]["relations"] = [{"type": "qualifies", "hit_id": "h4"},
                                            {"type": "qualifies", "hit_id": "h3"},
                                            {"type": "bogus", "hit_id": "h2"},
                                            {"type": "qualifies", "hit_id": "h2"}]
        entries, _, _, notes = allowance_map(out)
        self.assertEqual([(x["from"], x["to"]) for x in entries[0]["relations"] if x["type"] == "qualifies"],
                         [("h3", "h2")])
        rejected = [data for event, data in notes if event == "selector_relations_rejected"]
        self.assertEqual(rejected[0]["hit_id"], "h3")
        self.assertEqual(len(rejected[0]["relations"]), 3)

    def test_synthesis_applying_the_exception_to_the_wrong_mechanism_is_flagged(self):
        entries, _, _, _ = allowance_map()
        wrong = {"synthesis": [{"hit_ids": ["h1", "h2"], "treatment": "mechanism", "applies_to_user": "yes",
                                "qualifies": ""},
                               {"hit_ids": ["h3"], "treatment": "exception", "applies_to_user": "yes",
                                "qualifies": "h1"}]}
        _, issues = research.check_synthesis(wrong, entries)
        self.assertEqual(issues, [{"issue": "exception_target_differs", "hit_id": "h3",
                                   "applied_to": "h1", "labeled": ["h2"]}])
        right = copy.deepcopy(wrong)
        right["synthesis"][1]["qualifies"] = "h2"
        items, issues = research.check_synthesis(right, entries)
        self.assertEqual(issues, [])
        self.assertEqual(items[1]["qualifies"], "h2")
        loose = copy.deepcopy(wrong)
        loose["synthesis"][1]["qualifies"] = ""
        self.assertEqual(research.check_synthesis(loose, entries)[1],
                         [{"issue": "exception_without_target", "hit_id": "h3"}])

    def test_manufactured_threshold_and_normalized_verbs_are_flagged(self):
        bad = ("Travel only strips allowance in excess; normal amounts are safe. Parking strips allowance "
               "and leaches hotel.")
        self.assertEqual(research.threshold_phrases(bad, ALLOWANCE_EVIDENCE), ["in excess", "normal amounts"])
        self.assertEqual(research.unsourced_mechanism_verbs(bad, ALLOWANCE_EVIDENCE), ["strip", "leach"])
        good = ("The corpus author describes two mechanisms: travel creates a demand for allowance, so the payout comes "
                "out weak when you don't have enough, and parking pulls allowance out. Taxi doesn't "
                "pull allowance out the way parking does.")
        self.assertEqual(research.threshold_phrases(good, ALLOWANCE_EVIDENCE), [])
        self.assertEqual(research.unsourced_mechanism_verbs(good, ALLOWANCE_EVIDENCE), [])
        # A threshold the source itself states is reported, not flagged.
        self.assertEqual(research.threshold_phrases("Using travel in excess...", "travel in excess causes"), [])

# ---- EXPENSE CLAIM -----------------------------------------------------------------------------

CLAIM_BASE = ("A: For the expense claim, dissolve hotel, meals, taxi and mileage together. "
              "Mostly hotel, with a little meals.")
CLAIM_MILEAGE_WATER = ("A: I've changed the expense claim. Don't dissolve the mileage any more: steep it and "
               "add the direct deposit to the claim, because the slow steep is what helps.")
CLAIM_FINE_ACCOUNT = "A: Fine account should get less hotel and more taxi in its expense claim."
CLAIM_STAGE = "A: In the first year of travel, use the expense claim twice per batch."
CLAIM_CLIENT = ("A: For her, because of her hard well water, I made the expense claim with a cup of meals "
                "liquor in every quart.")
CLAIM_OLDER = "A: The expense claim is hotel and meals."
CLAIM_TITLES = {"s1": "Workshop, 1998", "s2": "Lecture, 2006", "s3": "Q&A, 2007",
                "s4": "Lecture, 2009", "s5": "Consultation notes, 2008", "s6": "Newsletter, 1997"}
CLAIM_REQS = [requirement("r1", "procedure", "complete preparation of the expense claim"),
              requirement("r2", "applicability", "changes for fine account")]
CLAIM_REQS[0]["exact"] = True
CLAIM_CANDS = [candidate("h1", "s1", CLAIM_BASE), candidate("h2", "s2", CLAIM_MILEAGE_WATER),
               candidate("h3", "s3", CLAIM_FINE_ACCOUNT), candidate("h4", "s4", CLAIM_STAGE),
               candidate("h5", "s5", CLAIM_CLIENT), candidate("h6", "s6", CLAIM_OLDER)]
CLAIM_COVERAGE = [
    {"requirement_id": "r1", "status": "partial", "hit_ids": ["h1", "h2"],
     "missing": "amount of direct deposit not given", "gap": "quantity", "lead_hit_ids": ["h2"]},
    {"requirement_id": "r2", "status": "covered", "hit_ids": ["h3"], "missing": "", "gap": "none",
     "lead_hit_ids": []}]
CLAIM_SELECTION = selection(
    [decision("h1", ["r1"], "dissolve hotel, meals, taxi and mileage together"),
     decision("h2", ["r1"], "steep it and add the direct deposit to the claim",
              relations=[("updates", "h1")]),
     decision("h3", ["r2"], "should get less hotel and more taxi", scope="condition",
              applies_to="fine account", relations=[("narrows", "h1")]),
     decision("h4", ["r1"], "use the expense claim twice per batch", scope="stage",
              applies_to="first year of travel"),
     decision("h5", ["r1"], "made the expense claim with a cup of meals liquor in every quart",
              scope="individual", applies_to="one client with hard well water", use="context"),
     decision("h6", ["r1"], "The expense claim is hotel and meals", relations=[("updates", "h2")])],
    CLAIM_COVERAGE)


def claim_map(out=CLAIM_SELECTION):
    ids, _, decisions, coverage, _ = research.normalize_selection(out, CLAIM_CANDS, CLAIM_REQS)
    kept = [c for c in CLAIM_CANDS if c["hit_id"] in ids]
    entries, text = research.evidence_map(CLAIM_REQS, coverage, kept, decisions, CLAIM_TITLES)
    return entries, text, coverage


class ExpenseClaim(unittest.TestCase):
    def test_later_change_layers_on_the_older_formula_instead_of_replacing_it(self):
        entries, text, _ = claim_map()
        r1 = entries[0]
        self.assertEqual(r1["changed"], ["h1"])
        self.assertIn("h1 is changed by the later h2: the change replaces only what it addresses; what "
                      "it does not change still comes from h1.", text)

    def test_an_update_pointing_backwards_in_time_is_not_accepted(self):
        entries, text, _ = claim_map()
        backwards = [x for x in entries[0]["relations"] if x["from"] == "h6"][0]
        self.assertEqual(backwards["check"], "older_than_target")
        self.assertNotIn("h2", entries[0]["changed"])
        self.assertIn("h6 updates h2: NOT supported by the dates (h6 is older than h2)", text)

    def test_undated_update_is_not_confirmed(self):
        titles = dict(CLAIM_TITLES, s2="Lecture")
        ids, _, decisions, coverage, _ = research.normalize_selection(CLAIM_SELECTION, CLAIM_CANDS, CLAIM_REQS)
        entries, text = research.evidence_map(CLAIM_REQS, coverage,
                                              [c for c in CLAIM_CANDS if c["hit_id"] in ids], decisions, titles)
        self.assertEqual(entries[0]["changed"], [])
        self.assertIn("h2 updates h1: the dates do not confirm which is later", text)

    def test_dated_general_statements_without_a_relationship_are_not_ordered(self):
        out = copy.deepcopy(CLAIM_SELECTION)
        out["claims"][1]["relations"] = []
        _, text, _ = claim_map(out)
        self.assertIn("General statements from different dates with no stated relationship", text)
        self.assertIn("A later date alone does not make one an update of the other.", text)

    def test_condition_stage_and_client_guidance_stay_scoped(self):
        entries, text, _ = claim_map()
        self.assertIn("Condition-specific: h3 (fine account); applies only where the user's stated "
                      "situation matches.", text)
        self.assertIn("Stage-specific: h4 (first year of travel)", text)
        self.assertIn("Individual cases: h5 (one client with hard well water); examples for that person or case, "
                      "not general rules.", text)
        self.assertIn("Context only (informs interpretation; need not appear in the answer): h5.", text)
        self.assertEqual([st["hit_id"] for st in entries[1]["statements"]], ["h3"])

    def test_synthesis_misuse_is_flagged(self):
        entries, _, _ = claim_map()
        misuse = {"synthesis": [
            {"hit_ids": ["h1"], "treatment": "base", "applies_to_user": "yes", "qualifies": ""},
            {"hit_ids": ["h2"], "treatment": "not_used", "applies_to_user": "unknown", "qualifies": ""},
            {"hit_ids": ["h5"], "treatment": "base", "applies_to_user": "yes", "qualifies": ""},
            {"hit_ids": ["h3"], "treatment": "base", "applies_to_user": "no", "qualifies": ""}]}
        _, issues = research.check_synthesis(misuse, entries)
        self.assertEqual({i["issue"] for i in issues},
                         {"earlier_version_used_without_its_update", "individual_case_used_as_base",
                          "specific_guidance_used_as_general_base"})
        layered = {"synthesis": [
            {"hit_ids": ["h1"], "treatment": "base", "applies_to_user": "yes", "qualifies": ""},
            {"hit_ids": ["h2"], "treatment": "update", "applies_to_user": "yes", "qualifies": ""},
            {"hit_ids": ["h3"], "treatment": "condition", "applies_to_user": "yes", "qualifies": ""},
            {"hit_ids": ["h4"], "treatment": "stage", "applies_to_user": "unknown", "qualifies": ""},
            {"hit_ids": ["h5"], "treatment": "individual_example", "applies_to_user": "no", "qualifies": ""},
            {"hit_ids": ["h6", "h99"], "treatment": "not_used", "applies_to_user": "no", "qualifies": ""}]}
        items, issues = research.check_synthesis(layered, entries)
        self.assertEqual(issues, [])
        self.assertEqual(items[-1]["hit_ids"], ["h6"], "unknown hit ids are dropped")

    def test_missing_exact_amount_becomes_a_repair_lead(self):
        _, text, coverage = claim_map()
        self.assertEqual(coverage[0]["gap"], "quantity")
        self.assertEqual(coverage[0]["leads"], ["h2"])
        self.assertIn("Gap (quantity): amount of direct deposit not given; leads: h2. Resolve it from "
                      "the evidence or a targeted follow-up search; never supply a value.", text)
        prompt = research.reasoner_prompt("How do I make it?", "normal", "E", CLAIM_REQS, coverage,
                                          evidence_map=text)
        self.assertIn("- r1 [procedure; exact details needed] complete preparation", prompt)
        self.assertIn("Targeted repair candidates", prompt)
        self.assertIn("- r1: quantity missing (amount of direct deposit not given); leads: h2", prompt)
        final = research.reasoner_prompt("q", "normal", "E", CLAIM_REQS, coverage,
                                         repaired={"premises": [], "coverage": [], "selected": 0},
                                         evidence_map=text)
        self.assertNotIn("Targeted repair candidates", final)

    def test_quantities_are_compared_with_units_and_number_words_normalized(self):
        evidence = "Use 2 cups of hotel water and half a glass of direct deposit."
        self.assertEqual(research.ungrounded_quantities(
            "Use two cups of hotel water, 1/2 glass of direct deposit and 3 tablespoons of meals.", evidence),
            ["3 tablespoons"])

    def test_forced_repair_needs_a_material_exact_gap(self):
        reqs = [requirement("r1", "claim", "why the claim is recommended")]
        coverage = [{"requirement_id": "r1", "status": "partial", "hit_ids": ["h1"],
                     "missing": "amount not given", "gap": "quantity", "leads": ["h1"]}]
        self.assertEqual(research.forced_repair(reqs, coverage, "Use 2 cups.", "E", []), [])
        reqs[0].update(kind="quantity")
        request = research.forced_repair(reqs, coverage, "No amount given.", "E", [])
        self.assertEqual(len(request), 1)
        self.assertEqual(research.forced_repair(reqs, coverage, "x", "E", [request[0]["search"]]), [],
                         "an already searched query is not repeated")


class Dates(unittest.TestCase):
    def test_source_dates_from_titles(self):
        self.assertEqual(research.source_date("Workshop, 2006"), "2006")
        self.assertEqual(research.source_date("2003-05-12 Q&A"), "2003-05-12")
        self.assertEqual(research.source_date("Interview May 2004"), "2004-05")
        self.assertEqual(research.source_date("Lecture, August 17, 2004"), "2004-08-17")
        self.assertIsNone(research.source_date("Recipe book"))
        self.assertEqual(research.date_order("2006", "2006-05"), 0)
        self.assertEqual(research.date_order("2005-12", "2006"), -1)
        self.assertIsNone(research.date_order(None, "2006"))


class Schemas(unittest.TestCase):
    def test_claim_structure_and_synthesis_are_in_the_schemas(self):
        claim_schema = research.SELECTOR_SCHEMA["properties"]["claims"]["items"]
        for field in ("hit_id",) + CLAIM_FIELDS:
            self.assertIn(field, claim_schema["required"])
        self.assertIn("claims", research.SELECTOR_SCHEMA["required"])
        coverage = research.SELECTOR_SCHEMA["properties"]["coverage"]["items"]
        self.assertIn("gap", coverage["required"])
        self.assertIn("lead_hit_ids", coverage["required"])
        self.assertIn("synthesis", research.REASONER_SCHEMA["required"])
        self.assertIn("exact", research.PLANNER_SCHEMA["properties"]["requirements"]["items"]["required"])


if __name__ == "__main__":
    unittest.main()
