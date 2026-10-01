"""Offline validation of the evidence triage (one Haiku call: KEEP + COVERAGE/DECISION).

For up to --max saved runs of the eval questions (evals/questions.json) at normal or deep depth
(quick runs would not be triaged), newest first and spread across questions, the run's last
evidence file (what its answer call read) goes to the triage call with the run's requirements
and requested parts. Only the passages it keeps are replayed: every required eval fact present
in the evidence before must still be present (tools/trim_sim.py's matching), and the kept
evidence's size is compared with the whole. A reply without a usable KEEP line keeps everything.

Adopt only with 100% fact retention and an average evidence reduction of at least 25%.

Usage: python tools/triage_sim.py [--max 25] [--out triage-sim.json]
"""
import argparse
import json
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import research as R  # noqa: E402
import trim_sim as T  # noqa: E402

TRIAGE_SYSTEM = (ROOT / "prompts" / "triage.txt").read_text(encoding="utf-8")
KEEP_LINE = re.compile(r"^\s*KEEP:\s*(.*)$", re.M)


def parse_keep(text, ids):
    """The kept passage ids (pure): the KEEP line's known ids, or every id when the reply has no
    KEEP line or it names none of them."""
    m = KEEP_LINE.search(str(text or ""))
    named = set(re.findall(r"\bh\d+\b", m.group(1))) if m else set()
    kept = [i for i in ids if i in named]
    return kept if kept else list(ids)


def triage_prompt(question, requirements, parts, evidence):
    reqs = "\n".join(f"- {r['id']}: {r['text']}" for r in requirements)
    parts_text = "\n".join(f"- {R.requested_part_line(x)}" for x in parts) or "none"
    return (f"USER QUESTION:\n{question}\n\nANSWER REQUIREMENTS:\n{reqs}\n\n"
            f"REQUESTED PARTS:\n{parts_text}\n\nSOURCE EVIDENCE:\n\n{evidence}")


def candidates(specs, most):
    """Saved normal/deep eval runs, newest first, taken round-robin across questions."""
    by_q = {T.norm(s["question"]): k for k, s in specs.items()}
    per_q = {}
    for folder in sorted(R.settings.LOGS_DIR.iterdir(), reverse=True):
        q = folder / "question.txt"
        diags = sorted(folder.glob("evidence-*.diagnostics.json"))
        if not q.exists() or not diags or not (folder / "plan-1.json").exists():
            continue
        qid = by_q.get(T.norm(q.read_text(encoding="utf-8")))
        depth = json.loads(diags[-1].read_text(encoding="utf-8"))["stats"].get("depth")
        if qid and depth in ("normal", "deep"):
            per_q.setdefault(qid, []).append(folder)
    out, i = [], 0
    while len(out) < most and any(i < len(v) for v in per_q.values()):
        for qid in sorted(per_q):
            if i < len(per_q[qid]) and len(out) < most:
                out.append((qid, per_q[qid][i]))
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=25)
    ap.add_argument("--out", default=None)
    ap.add_argument("--timeout", type=float, default=90,
                    help="seconds per call (the pipeline's stage timeout is 30)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    specs = json.loads((ROOT / "evals" / "questions.json").read_text(encoding="utf-8"))
    run = R.Run("triage validation")
    run.dir = Path(tempfile.mkdtemp())
    rows = []
    for qid, folder in candidates(specs, a.max):
        n = sorted(folder.glob("evidence-*.txt"))[-1]
        evidence = n.read_text(encoding="utf-8")
        texts, heads = T.hits_of(evidence)
        hits = [{"hit_id": i, "text": t, "source_line": heads[i]} for i, t in texts.items()]
        plan = json.loads((folder / "plan-1.json").read_text(encoding="utf-8"))
        planner = folder / "planner-1.json"
        parts = R.normalize_requested_parts(
            json.loads(planner.read_text(encoding="utf-8")).get("requested_parts")
            if planner.exists() else [])
        prompt = triage_prompt(folder.joinpath("question.txt").read_text(encoding="utf-8"),
                               plan["requirements"], parts, evidence)
        start = time.monotonic()
        try:
            reply = R.claude(run, "triage", R.COVERAGE, TRIAGE_SYSTEM, prompt, timeout=a.timeout)
            failed = False
        except R.ResearchError as e:
            reply, failed = f"(failed: {e})", True
        secs = time.monotonic() - start
        kept_ids = parse_keep(reply, [h["hit_id"] for h in hits])
        kept = [h for h in hits if h["hit_id"] in kept_ids]
        before, after = T.facts_in(T.joined(hits), specs[qid]), T.facts_in(T.joined(kept), specs[qid])
        chars_b = sum(R.hit_chars(h) for h in hits)
        chars_a = sum(R.hit_chars(h) for h in kept)
        cov = next((x for x in str(reply).splitlines() if x.startswith("COVERAGE:")), None)
        row = {"qid": qid, "run": folder.name, "parts": len(parts), "failed": failed, "hits": len(hits),
               "kept": len(kept), "chars_before": chars_b, "chars_after": chars_a,
               "facts_before": sorted(before), "lost": sorted(before - after),
               "seconds": round(secs, 1), "coverage_line": cov, "reply": str(reply)[:600]}
        rows.append(row)
        print(f"{qid:4} {folder.name[:15]} parts {len(parts)} kept {len(kept):>2}/{len(hits):<2} "
              f"chars {chars_b:>6,} -> {chars_a:>6,} ({1 - chars_a / chars_b:5.0%})  "
              f"{secs:4.1f}s" + (f"  LOST {row['lost']}" if row["lost"] else "")
              + ("  CALL FAILED" if failed else ""), flush=True)
    failures = [r for r in rows if r["failed"]]
    rows = [r for r in rows if not r["failed"]]
    print(f"\n{len(failures)} calls failed (excluded): "
          + "; ".join(f"{r['qid']} {r['seconds']}s {r['reply'][:60]}" for r in failures))
    facts = sum(len(r["facts_before"]) for r in rows)
    lost = sum(len(r["lost"]) for r in rows)
    reduction = statistics.mean(1 - r["chars_after"] / r["chars_before"] for r in rows)
    print(f"\n{len(rows)} runs: facts kept {facts - lost}/{facts}; average evidence reduction "
          f"{reduction:.0%}; triage seconds median {statistics.median(r['seconds'] for r in rows):.1f}"
          f" (min {min(r['seconds'] for r in rows):.1f}, max {max(r['seconds'] for r in rows):.1f})")
    print("ADOPT" if lost == 0 and reduction >= 0.25 else "DO NOT ADOPT")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
