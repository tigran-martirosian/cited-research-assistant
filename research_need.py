"""Research-need signatures, situation classification and the selection-reuse gate.

Pure functions only: no database, no model, no clock unless passed in. research.py derives the
inputs (the planner's case frame, the selector-input pool as primary unit ids with their origin,
the selector policy and corpus identity) and records the decision in the run trace; the
research-need ledger itself lives in research_memory (table research_needs).

Shadow mode: the gate is evaluated and recorded, never acted on. A missed match (planner
nondeterminism, a paraphrase surfacing a different pool) is acceptable; a false match is not, so
normalization only removes what cannot change meaning (case, punctuation, articles, plural -s).
"""
import hashlib
import json
import re
import unicodedata

NEED_SIGNATURE_VERSION = 1
PURPOSES = ["source_statement", "practical_applicability", "procedure", "mechanism", "comparison",
            "other"]
FIELDS = ["subject", "qualifiers", "processing_action", "purpose", "context"]
SITUATIONS = ["EXACT", "SAME_NEED", "RELATED", "RELATED_NEW_QUALIFIER", "NEW"]
ARTICLES = frozenset(("a", "an", "the"))
# Words the plural rule must leave alone (they end in s but are not plurals).
KEEP_S = frozenset(("this", "was", "has", "is", "its", "yes", "less", "unless", "gas", "bus",
                    "plus", "thus", "always", "perhaps", "news", "series", "species"))


def singular(word):
    """Deterministic plural stripping, no dictionary: batches -> batch, boxes -> box,
    boxes -> box; words ending in ss/us/is and short words stay as they are. An irregular or
    wrongly stripped word only makes a match less likely, never a false one."""
    if len(word) <= 3 or word in KEEP_S or not word.endswith("s") or not word.isalpha():
        return word
    if word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("xes", "ches", "shes", "sses", "zes")):
        return word[:-2]
    return word[:-1]


def normalize_text(text):
    """NFKC, lowercase, "n't" spelled "not", punctuation removed, whitespace collapsed, articles
    dropped, each word singularized. Negations (no, not, without, never) and temporal or
    conditional words (before, after, during, if, unless, until) are ordinary words and kept."""
    text = unicodedata.normalize("NFKC", text if isinstance(text, str) else "").lower()
    text = re.sub(r"n['’]t\b", " not", text)
    text = re.sub(r"[^\w\s]|_", " ", text)
    words = [singular(w) for w in text.split() if w not in ARTICLES]
    return " ".join(words)


def normalize_frame(raw):
    """The canonical case frame from the planner's case_frame object, or None when it is missing
    or has no subject or no valid purpose (the signature is then "unavailable")."""
    if not isinstance(raw, dict):
        return None
    subject = normalize_text(raw.get("subject"))
    purpose = raw.get("purpose") if raw.get("purpose") in PURPOSES else None
    if not subject or not purpose:
        return None
    qualifiers = raw.get("qualifiers") if isinstance(raw.get("qualifiers"), list) else []
    return {"subject": subject,
            "qualifiers": sorted({q for q in (normalize_text(x) for x in qualifiers) if q}),
            "processing_action": normalize_text(raw.get("processing_action")),
            "purpose": purpose,
            "context": normalize_text(raw.get("context"))}


def need_signature(frame):
    """sha256 of the canonical frame JSON plus NEED_SIGNATURE_VERSION, or "unavailable"."""
    if not frame:
        return "unavailable"
    body = json.dumps({"version": NEED_SIGNATURE_VERSION, "frame": frame}, sort_keys=True,
                      ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def frame_diff(prior, current):
    """What differs between two canonical frames."""
    before, after = set(prior["qualifiers"]), set(current["qualifiers"])
    return {"qualifiers_added": sorted(after - before),
            "qualifiers_removed": sorted(before - after),
            "changed": [f for f in FIELDS if prior.get(f) != current.get(f)]}


def classify(frame, entries):
    """Situation of `frame` against ledger entries ({"need_id", "frame", "recorded_at", ...}).
    Returns (situation, matched entry or None, frame_diff or None). EXACT is decided earlier by
    the exact-result cache and never here.

    Among entries with the same subject, the match has the most equal fields, then the most
    shared qualifiers, then the latest recorded_at, then the smallest need_id (deterministic).
    SAME_NEED: every field equal. RELATED_NEW_QUALIFIER: the qualifier set differs in any way.
    RELATED: subject and qualifiers equal, purpose, action or context differ. NEW: no entry has
    the subject."""
    if not frame:
        return None, None, None
    same = [e for e in entries if (e.get("frame") or {}).get("subject") == frame["subject"]]
    if not same:
        return "NEW", None, None

    def rank(e):
        f = e["frame"]
        equal = sum(f.get(k) == frame[k] for k in FIELDS)
        shared = len(set(f.get("qualifiers") or []) & set(frame["qualifiers"]))
        return (-equal, -shared, -(e.get("recorded_at") or 0), e.get("need_id") or "")

    match = sorted(same, key=rank)[0]
    diff = frame_diff(match["frame"], frame)
    if not diff["changed"]:
        return "SAME_NEED", match, diff
    if diff["qualifiers_added"] or diff["qualifiers_removed"]:
        return "RELATED_NEW_QUALIFIER", match, diff
    return "RELATED", match, diff


def corpus_changed(stored, current):
    """Whether a stored corpus identity ({"notebooks", "epoch", "manifest"}) differs."""
    if not isinstance(stored, dict):
        return True
    return any(stored.get(k) != current.get(k) for k in ("notebooks", "epoch", "manifest"))


def gate(situation, current, entry, resolvable, now, max_age_hours):
    """The selection-reuse gate, evaluated without short-circuit so every failing check is
    recorded. Shadow mode: the caller never acts on it.

    current: {"signature", "selector_policy", "corpus", "pool": {unit id: "retrieval"|"memory"},
              "facets": [requirement kinds], "reuse_disabled": bool}
    entry:   the matched ledger entry: {"signature", "selector_policy", "corpus", "recorded_at",
              "input_units", "selection_units" (None when ids were unmappable), "selector_primary",
              "facets"}
    resolvable: the recorded selection's unit ids that resolve to primary text in memory.

    Returns {"result": "would_reuse"|"fresh"|"n/a", "failed", "unseen_passages",
    "passes_ignoring_memory_unseen", "facet_gate"}."""
    if situation != "SAME_NEED" or entry is None:
        return {"result": "n/a", "failed": [], "unseen_passages": None,
                "passes_ignoring_memory_unseen": None, "facet_gate": None}
    failed = []
    if current["signature"] != entry.get("signature"):  # (a)
        failed.append("signature_mismatch")
    if current["selector_policy"] != entry.get("selector_policy"):  # (b)
        failed.append("selector_policy_changed")
    if corpus_changed(entry.get("corpus"), current["corpus"]):  # (c)
        failed.append("corpus_changed")
    recorded = entry.get("recorded_at")
    if current["corpus"].get("strength") == "weak" and (
            not isinstance(recorded, (int, float)) or (now - recorded) / 3600 > max_age_hours):
        failed.append("entry_too_old")
    known = set(entry.get("input_units") or [])  # (d)
    unseen = {"retrieval": 0, "memory": 0}
    for uid, origin in current["pool"].items():
        if uid not in known:
            unseen["memory" if origin == "memory" else "retrieval"] += 1
    if unseen["retrieval"] or unseen["memory"]:
        failed.append("unseen_passages")
    selection = entry.get("selection_units")  # (e)
    if selection is None:
        failed.append("ids_unmappable")
    elif set(selection) - set(resolvable):
        failed.append("selection_unresolvable")
    if not entry.get("selector_primary"):  # (f)
        failed.append("selector_fallback")
    if entry.get("facets") is None:  # (g) requirement kinds are the plan's enumerable facets
        facet_gate = "frame_only"
    elif set(current["facets"]) <= set(entry["facets"]):
        facet_gate = "kinds_subset"
    else:
        facet_gate = "kinds_not_subset"
        failed.append("facets_not_subset")
    if current.get("reuse_disabled"):  # (h)
        failed.append("reuse_disabled")
    others = [f for f in failed if f != "unseen_passages"]
    return {"result": "fresh" if failed else "would_reuse", "failed": failed,
            "unseen_passages": unseen,
            "passes_ignoring_memory_unseen": not others and unseen["retrieval"] == 0,
            "facet_gate": facet_gate}
