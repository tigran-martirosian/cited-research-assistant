"""Score answers against required / forbidden regex patterns (evals/questions.json) and report
each run's automatic signals.

Usage: python tools/eval.py <run log dir | answer file> [more ...] [--id b]

A run directory is matched to its question by question.txt (unless --id is given); an answer
file (e.g. logs/baseline/b.md) needs --id. Patterns are case-insensitive. Prints each fact
hit/miss, forbidden matches, and, for a run directory, its signals: first-text time, total
time, output tokens by model (plan usage), grounding failures, unverified numbers, requested
parts still "no" after the coverage follow-up, and community records used vs dropped. With
several targets it ends with a table and the average output tokens per run by model. Exit code
1 when a required fact is missing or a forbidden pattern matches.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "evals" / "questions.json"
FAMILIES = ("opus", "sonnet", "haiku", "fable", "gemini")


def norm(text):
    return " ".join(text.lower().split())


def score(answer, spec):
    """(hits, misses, forbidden matches) of one answer against one question's patterns. Markdown
    emphasis is removed first ("**2 days** of leave" matches like "2 days of leave")."""
    answer = answer.replace("*", "")
    hits, misses = [], []
    for name, pattern in spec.get("required", {}).items():
        (hits if re.search(pattern, answer, re.I) else misses).append(name)
    bad = [name for name, pattern in spec.get("forbidden", {}).items()
           if re.search(pattern, answer, re.I)]
    return hits, misses, bad


def family(u):
    """The model family of one usage record ("opus", "sonnet", ...), from the configured model
    or, in older runs, the CLI's model ids."""
    names = " ".join([u.get("model") or ""] + list(u.get("models") or [])).lower()
    return next((f for f in FAMILIES if f in names), names or "?")


def tokens_by_model(details):
    """{family: output tokens} over the run's model calls (output includes thinking)."""
    out = {}
    for u in details.get("claude_usage", []):
        if not u.get("usage"):
            continue
        fam = family(u)
        out[fam] = out.get(fam, 0) + ((u.get("usage") or {}).get("output_tokens") or 0)
    return out


def run_facts(folder):
    """The run's automatic signals, when recorded in its result.json."""
    path = folder / "result.json"
    if not path.exists():
        return {}
    details = json.loads(path.read_text(encoding="utf-8"))["details"]
    checks = details.get("checks") or {}
    trace = details.get("trace") or {}
    memory = details.get("memory") or {}
    filt = memory.get("secondary_filter") or {}
    follow = trace.get("coverage_followup") or {}
    return {"first_text_s": details.get("first_answer_seconds"),
            "total_s": details.get("total_seconds"), "depth": details.get("depth"),
            "out_tokens": tokens_by_model(details),
            "grounding_failures": checks.get("grounding_failures"),
            "unverified_numbers": checks.get("unverified_numbers",
                                             checks.get("ungrounded_quantities")),
            "follow_up": bool(details.get("repair")),
            "coverage_fired": follow.get("fired"), "still_no": follow.get("still_no"),
            "community_used": filt.get("used", len(memory.get("secondary") or [])),
            "community_dropped": filt.get("dropped")}


def evaluate(target, specs, qid=None):
    """Score one target; returns (question id, facts line, run facts, failed)."""
    folder = target if target.is_dir() else None
    answer_path = folder / "answer.md" if folder else target
    if qid is None and folder and (folder / "question.txt").exists():
        question = norm((folder / "question.txt").read_text(encoding="utf-8"))
        qid = next((k for k, s in specs.items() if norm(s["question"]) == question), None)
    if qid not in specs:
        sys.exit(f"no eval question for {target} (use --id)")
    hits, misses, bad = score(answer_path.read_text(encoding="utf-8"), specs[qid])
    print(f"{qid}: {len(hits)}/{len(hits) + len(misses)} facts  {target}")
    print(f"  hit:    {', '.join(hits) or '-'}")
    print(f"  missed: {', '.join(misses) or '-'}")
    if bad:
        print(f"  FORBIDDEN: {', '.join(bad)}")
    facts = run_facts(folder) if folder else {}
    if facts:
        print(f"  run:    {facts}")
    return qid, f"{len(hits)}/{len(hits) + len(misses)}", facts, bool(misses or bad)


def main(argv):
    if not argv:
        sys.exit(__doc__)
    qid = argv[argv.index("--id") + 1] if "--id" in argv else None
    targets = [Path(a) for i, a in enumerate(argv)
               if a != "--id" and (i == 0 or argv[i - 1] != "--id")]
    specs = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    rows = [evaluate(t, specs, qid) for t in targets]
    runs = [r for r in rows if r[2]]
    if len(runs) > 1:
        print(f"\n{'id':5} {'facts':6} {'first':>6} {'total':>6}  {'out tokens':30} "
              f"{'ground':>6} {'nums':>4} {'still_no':>8} {'comm':>7}")
        for q, facts_line, f, _ in runs:
            toks = " ".join(f"{k}:{v}" for k, v in sorted(f["out_tokens"].items()))
            print(f"{q:5} {facts_line:6} {f['first_text_s'] or '-':>6} {f['total_s'] or '-':>6}  "
                  f"{toks:30} {len(f['grounding_failures'] or []):>6} "
                  f"{len(f['unverified_numbers'] or []):>4} {len(f['still_no'] or []):>8} "
                  f"{f['community_used']}/{f['community_dropped'] if f['community_dropped'] is not None else '-':>5}")
        totals = {}
        for _, _, f, _ in runs:
            for k, v in f["out_tokens"].items():
                totals[k] = totals.get(k, 0) + v
        print("average output tokens per run: " + ", ".join(
            f"{k} {v / len(runs):,.0f}" for k, v in sorted(totals.items()))
            + f"; all {sum(totals.values()) / len(runs):,.0f} ({len(runs)} runs)")
    return 1 if any(r[3] for r in rows) else 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    sys.exit(main(sys.argv[1:]))
