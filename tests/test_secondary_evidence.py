"""Secondary-evidence readiness in research memory: structured community claims with claim type,
provenance, verification and links to primary units, kept apart from primary evidence.
Synthetic records only; no community corpus is read.

Run from the repository root:  python -m unittest tests.test_secondary_evidence -v
"""
import json
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import research_memory as rm  # noqa: E402

PRIMARY_TEXT = ("A: Parking pulls allowance out of the account. It binds with it and carries it "
                "out, so use it with enough allowance.")


class SecondaryClaims(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.store = rm.MemoryStore(self.tmp / "memory.db")
        self.primary = self.store.add_primary(PRIMARY_TEXT, "s1", "Workshop, 2004")

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.tmp)

    def claim(self, text, claim_type="primary_quote", record="chat:group/1", **kw):
        return self.store.add_secondary_claim(text, "chat", record, claim_type,
                                              community="travel chat", **kw)

    def test_every_claim_type_is_stored_as_secondary_with_provenance(self):
        for n, kind in enumerate(rm.CLAIM_TYPES):
            uid = self.claim(f"claim number {n} of type {kind}", kind, record=f"chat:group/{n}",
                             context="the message around it", source_date="2021-03-04",
                             author="member")
            self.assertEqual(self.store.layer(uid), rm.SECONDARY)
            c = self.store.secondary_claim(uid)
            self.assertEqual((c["claim_type"], c["record_id"], c["community"], c["platform"]),
                             (kind, f"chat:group/{n}", "travel chat", "chat"))
            self.assertEqual((c["context"], c["verification"], c["source_date"]),
                             ("the message around it", "unverified", "2021-03-04"))
        self.assertEqual(sum(self.store.stats()["secondary_claims"].values()), len(rm.CLAIM_TYPES))
        with self.assertRaises(ValueError):
            self.claim("x", "rumor")

    def test_same_record_and_claim_is_one_unit_other_claims_stay_separate(self):
        a = self.claim("First claim in the message.")
        self.assertEqual(self.claim("First  claim in the message."), a, "whitespace-normalized repeat")
        b = self.claim("Second claim in the same message.")
        self.assertNotEqual(a, b)
        c = self.claim("First claim in the message.", record="forum:x/abc")
        self.assertNotEqual(a, c, "the same words from another record are another occurrence")

    def test_verification_needs_the_primary_evidence_behind_it(self):
        uid = self.claim("Parking pulls allowance out of the account.")
        with self.assertRaisesRegex(ValueError, "needs a"):
            self.store.set_verification(uid, "verified_primary")
        self.assertEqual(self.store.find_quoted_primary(uid), [self.primary])
        self.store.link_secondary(uid, "quotes_primary", self.primary)
        self.store.set_verification(uid, "verified_primary", "quote found verbatim")
        c = self.store.secondary_claim(uid)
        self.assertEqual(c["verification"], "verified_primary")
        self.assertEqual(c["links"], [{"relation": "quotes_primary", "unit_id": self.primary,
                                       "layer": rm.PRIMARY, "note": None}])
        other = self.claim("Taxi pulls allowance too.", "novel_community_claim", record="chat:group/9")
        with self.assertRaises(ValueError):
            self.store.set_verification(other, "contradicted_by_primary")
        self.store.set_verification(other, "not_in_primary")
        anecdote = self.claim("My payouts held better after a month.", "anecdote_or_experiment", record="chat:group/10")
        self.store.set_verification(anecdote, "not_applicable")

    def test_primary_only_links_refuse_non_primary_targets(self):
        a = self.claim("One community claim.")
        b = self.claim("Another community claim.", record="chat:group/2")
        with self.assertRaisesRegex(ValueError, "primary unit"):
            self.store.link_secondary(a, "quotes_primary", b)
        self.store.link_secondary(a, "corrects", b)  # a correction may target another claim
        with self.assertRaises(ValueError):
            self.store.link_secondary(self.primary, "supported_by", a)  # only from secondary

    def test_secondary_never_supports_a_derived_claim_as_primary_evidence(self):
        uid = self.claim("Parking pulls allowance out of the account.")
        with self.assertRaises(ValueError):
            self.store.add_derived("inference", "combined_inference", supports=[uid])
        derived = self.store.add_derived("inference", "combined_inference", supports=[self.primary],
                                         inspired_by=[uid])
        self.assertEqual(self.store.layer(derived), rm.DERIVED)

    def test_search_and_rendering_keep_the_layers_apart(self):
        uid = self.claim("Parking pulls allowance out of the account, someone said.", "primary_paraphrase")
        results = {r["unit_id"]: r for r in self.store.search("parking allowance")}
        self.assertEqual(results[self.primary]["layer"], rm.PRIMARY)
        self.assertEqual(results[uid]["layer"], rm.SECONDARY)
        self.assertEqual(results[uid]["kind"], "primary_paraphrase")
        block = rm.secondary_evidence_block(self.store.secondary_claim(uid))
        self.assertTrue(block.startswith("SECONDARY SOURCE (community material, not the corpus author's own words)"))
        self.assertIn("CLAIM TYPE: primary_paraphrase", block)
        self.assertNotIn("PASSAGE", block)
        self.assertNotIn("\nSOURCE:", "\n" + block)

    def test_jsonl_import(self):
        records = [
            {"platform": "chat", "record_id": "chat:group/100", "community": "travel chat",
             "author": "member", "date": "2021-05-01", "claim_type": "primary_quote",
             "claim_text": "Parking pulls allowance out of the account.", "context": "asked about parking",
             "verification": "verified_primary",
             "primary_links": [{"relation": "quotes_primary", "unit_id": self.primary}],
             "processed_by": "community-extract v0"},
            {"platform": "forum", "record_id": "forum:travel/xyz", "claim_type": "practical_synthesis",
             "claim_text": "A daily routine assembled from several talks.",
             "primary_refs": [{"source_title": "Lecture", "source_date": "2006", "quote": "..."}]},
            {"platform": "chat", "record_id": "chat:group/101", "claim_type": "primary_paraphrase",
             "claim_text": "He said to use parking with allowance.", "verification": "verified_primary"},
            {"platform": "chat", "record_id": "chat:group/102", "claim_type": "primary_quote",
             "claim_text": "Unknown link target.",
             "primary_links": [{"relation": "quotes_primary", "unit_id": "P-doesnotexist"}]},
            {"platform": "chat", "claim_type": "anecdote_or_experiment", "claim_text": "no record id"},
        ]
        path = self.tmp / "claims.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in records) + "\nnot json\n", encoding="utf-8")
        report = rm.import_secondary(self.store, path)
        self.assertEqual((report["claims_new"], report["claims_seen"]), (4, 0))
        self.assertEqual([e["line"] for e in report["errors"]], [5, 6])
        notes = " ".join(str(n["note"]) for n in report["notes"])
        self.assertIn("verification kept unverified", notes, "record 3 claims verification without a link")
        self.assertIn("link not stored", notes)
        claims = {c["record_id"]: c for c in (self.store.secondary_claim(r["unit_id"]) for r in
                  self.store.db.execute("SELECT unit_id FROM secondary_claims"))}
        self.assertEqual(claims["chat:group/100"]["verification"], "verified_primary")
        self.assertEqual(claims["chat:group/100"]["processed_by"], "community-extract v0")
        self.assertEqual(claims["chat:group/101"]["verification"], "unverified")
        self.assertEqual(claims["forum:travel/xyz"]["primary_refs"][0]["source_title"], "Lecture")
        self.assertEqual(claims["chat:group/102"]["links"], [])
        self.assertIn("unresolved_link", claims["chat:group/102"]["primary_refs"][0])
        again = rm.import_secondary(self.store, path)
        self.assertEqual((again["claims_new"], again["claims_seen"]), (0, 4), "re-import adds nothing")
        self.assertEqual(self.store.stats()["units_by_layer"][rm.PRIMARY], 1, "no secondary became primary")

    def test_record_schema_matches_the_vocabularies(self):
        props = rm.SECONDARY_RECORD_SCHEMA["properties"]
        self.assertEqual(props["claim_type"]["enum"], list(rm.CLAIM_TYPES))
        self.assertEqual(props["verification"]["enum"], list(rm.VERIFICATION))
        self.assertEqual(set(rm.SECONDARY_RECORD_SCHEMA["required"]),
                         {"platform", "record_id", "claim_type", "claim_text"})

    def test_version_1_database_gains_the_secondary_table(self):
        path = self.tmp / "old.db"
        rm.MemoryStore(path).close()
        db = sqlite3.connect(path)
        db.execute("DROP TABLE secondary_claims")
        db.execute("UPDATE meta SET value = '1' WHERE key = 'schema_version'")
        db.commit()
        db.close()
        with rm.MemoryStore(path) as store:
            version = store.db.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0]
            self.assertEqual(version, str(rm.SCHEMA_VERSION))
            primary = store.add_primary("Some primary passage text here.", "s1")
            uid = store.add_secondary_claim("A paraphrase of the passage.", "forum", "forum:1",
                                            "primary_paraphrase", links=[{"relation": "paraphrases_primary",
                                                                          "unit_id": primary}])
            self.assertEqual(store.secondary_claim(uid)["links"][0]["unit_id"], primary)


if __name__ == "__main__":
    unittest.main()
