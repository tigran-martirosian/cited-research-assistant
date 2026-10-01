"""Offline evidence-trim tuning: replay the saved eval runs' evidence under candidate
PER_QUERY_KEEP / evidence-budget settings and check that every required eval fact still in the
evidence before is still in it after.

For every run folder in logs/ whose question.txt is an eval question (evals/questions.json) and
that has an evidence-N.txt with its evidence-N.diagnostics.json, the last round's evidence (what
the answer model saw) is split into its hits, and the search candidates its own trim dropped
(search-results-N.json) are added back. Each hit's per-query rank comes from the
diagnostics (None for memory passages). A hit is kept when its rank is within K or it has none,
and the rest go through research.trim_candidates at the depth's budget. Round-2 hits (coverage
follow-up) are trimmed within what round 1 leaves, as in the pipeline. Continuations are part
of the hit text, so they stay attached to their hit. Community records are not in these files
and are not trimmed.

Usage: python tools/trim_sim.py [--k 5] [--quick 20000] [--normal 30000] [--deep 50000] [--grid]
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import research as R  # noqa: E402

HEAD = re.compile(r"^(PASSAGE \[(h\d+)\].*|PASSAGE:|CONTEXT:|SOURCE: .*|=====)$")
# chars per evidence token, measured on one saved run (--calibrate <run folder>): its reasoner
# input against the answer call's cache-write tokens; 3.0 without one. Set in main().
CHARS_PER_TOKEN = None


def norm(text):
    return " ".join(text.lower().split())


def hits_of(evidence_text):
    """({hit_id: text}, {hit_id: SOURCE line}) from an evidence file: each PASSAGE [hN] with its
    text (continuation included), and a CONTEXT window counted with the hit before it in the same
    source. The SOURCE line (title and date) is what the reasoner sees above the hit."""
    out, heads, current, source = {}, {}, None, ""
    for line in evidence_text.splitlines():
        m = HEAD.match(line)
        if m and m.group(2):
            current = m.group(2)
            out.setdefault(current, "")
            heads.setdefault(current, source)
            continue
        if m and (line.startswith("SOURCE") or line == "====="):
            current = None if line == "=====" else current
            source = line if line.startswith("SOURCE") else source
            continue
        if m:  # PASSAGE: / CONTEXT: without an id: belongs to the current hit
            continue
        if current:
            out[current] += line + "\n"
    return out, heads


def dropped_candidates(folder, seen):
    """Search candidates the run's own trim kept out of the evidence (search-results-N.json, with
    their continuation text from continuations-N.json), so a fact that only a dropped hit carried
    still counts as present before. They have no SOURCE line."""
    out = []
    for path in sorted(folder.glob("search-results-*.json")):
        rnd = int(re.search(r"search-results-(\d+)", path.name).group(1))
        cont_path = folder / f"continuations-{rnd}.json"
        cont = {c["hit_id"]: c.get("continuation_text") or ""
                for c in (json.loads(cont_path.read_text(encoding="utf-8"))
                          if cont_path.exists() else [])}
        for c in json.loads(path.read_text(encoding="utf-8")).get("candidates", []):
            if c["hit_id"] in seen:
                continue
            seen.add(c["hit_id"])
            text = c["text"].strip() + ("\n" + cont[c["hit_id"]] if cont.get(c["hit_id"]) else "")
            out.append({"hit_id": c["hit_id"], "rank": c.get("rank"), "text": text,
                        "source_line": "", "round": min(rnd, 2), "origin": "fresh"})
    return out


def runs(specs):
    """(qid, folder, depth, hits [{hit_id, rank, round, text}], evidence text) per eval run."""
    by_q = {norm(s["question"]): k for k, s in specs.items()}
    for folder in sorted(R.settings.LOGS_DIR.iterdir()):
        qfile = folder / "question.txt"
        if not qfile.exists():
            continue
        qid = by_q.get(norm(qfile.read_text(encoding="utf-8")))
        files = sorted(folder.glob("evidence-*.diagnostics.json"))
        if not qid or not files:
            continue
        last = files[-1]
        n = re.search(r"evidence-(\d+)", last.name).group(1)
        text = (folder / f"evidence-{n}.txt").read_text(encoding="utf-8")
        diag = json.loads(last.read_text(encoding="utf-8"))
        ranks, depth = {}, diag.get("stats", {}).get("depth", "normal")
        for b in diag.get("blocks", []):
            for h in b.get("hits", []):
                ranks[h["hit_id"]] = h.get("rank")
        round1 = set()
        if n != "1" and (folder / "evidence-1.diagnostics.json").exists():
            d1 = json.loads((folder / "evidence-1.diagnostics.json").read_text(encoding="utf-8"))
            round1 = {h["hit_id"] for b in d1.get("blocks", []) for h in b.get("hits", [])}
        texts, heads = hits_of(text)
        hits = [{"hit_id": i, "rank": ranks.get(i), "text": t, "source_line": heads[i],
                 "round": 1 if n == "1" or i in round1 else 2,
                 "origin": "memory" if ranks.get(i) is None else "fresh"}
                for i, t in texts.items()]
        hits += dropped_candidates(folder, set(texts))
        hits.sort(key=lambda h: int(h["hit_id"][1:]))
        yield qid, folder, depth, hits, text


def simulate(hits, depth, k, budgets):
    """The hits kept at per-query keep `k` and the depth budgets (research.trim_candidates)."""
    budget = budgets[depth]
    kept = []
    for rnd in (1, 2):
        cands = [dict(h, continuation_text=None, backward_text=None) for h in hits
                 if h["round"] == rnd and (h["rank"] is None or h["rank"] <= k)]
        prior = sum(R.hit_chars(h) for h in kept)
        if cands and prior + sum(R.hit_chars(c) for c in cands) > budget:
            cands, _ = R.trim_candidates(cands, max(budget - prior, 0))
        kept += cands
    return kept


def joined(hits):
    """The hits as the fact patterns see them: each with its SOURCE line, in hit order."""
    return "\n".join(h["source_line"] + "\n" + h["text"]
                     for h in sorted(hits, key=lambda h: int(h["hit_id"][1:])))


def facts_in(text, spec):
    t = text.replace("*", "")
    return {name for name, p in spec.get("required", {}).items() if re.search(p, t, re.I)}


def evaluate(all_runs, specs, k, budgets, show=False):
    """(ok, rows): ok when every fact present before is present after in every run."""
    ok, rows = True, []
    for qid, folder, depth, hits, text in all_runs:
        before = facts_in(joined(hits), specs[qid])
        kept = simulate(hits, depth, k, budgets)
        after = facts_in(joined(kept), specs[qid])
        lost = before - after
        ok &= not lost
        rows.append({"qid": qid, "run": folder.name[:15], "depth": depth,
                     "facts": f"{len(after & before)}/{len(before)}", "lost": sorted(lost),
                     "hits": f"{len(kept)}/{len(hits)}",
                     "chars_before": sum(R.hit_chars(h) for h in hits),
                     "chars_after": sum(R.hit_chars(h) for h in kept)})
    return ok, rows


def calibrate(folder=None):
    if not folder:
        return 3.0
    folder = Path(folder)
    usage = json.loads((folder / "reasoner-1.claude.json").read_text(encoding="utf-8"))
    text = (folder / "reasoner-1.input.txt").read_text(encoding="utf-8")
    u = usage.get("usage") or usage
    write = u.get("cache_creation_input_tokens") or 0
    return len(text) / write if write else 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--quick", type=int, default=20_000)
    ap.add_argument("--normal", type=int, default=30_000)
    ap.add_argument("--deep", type=int, default=50_000)
    ap.add_argument("--grid", action="store_true", help="search the lowest passing settings")
    ap.add_argument("--calibrate", help="a saved run folder to measure chars per token on")
    a = ap.parse_args()
    specs = json.loads((ROOT / "evals" / "questions.json").read_text(encoding="utf-8"))
    all_runs = list(runs(specs))
    cpt = calibrate(a.calibrate)
    print(f"{len(all_runs)} eval runs; {cpt:.2f} chars per evidence token")
    if a.grid:  # lowest K that loses nothing with unlimited budgets, then each depth's budget
        start = {"direct": a.quick, "normal": a.normal, "deep": a.deep}
        k = next(k for k in range(a.k, R.SEARCH_LIMIT + 1)
                 if evaluate(all_runs, specs, k, dict.fromkeys(start, 10**9))[0])
        best = {}
        for depth, low in start.items():
            some = [r for r in all_runs if r[2] == depth]
            budget = low
            while some and not evaluate(some, specs, k, {**start, depth: budget})[0]:
                budget += 1000
            best[depth] = budget
            print(f"K={k} {depth}: {len(some)} runs, lowest budget {budget:,}"
                  + ("" if some else " (no runs: start value kept)"))
        print(f"chosen: --k {k} --quick {best['direct']} --normal {best['normal']} "
              f"--deep {best['deep']}")
        return 0
    b = {"direct": a.quick, "normal": a.normal, "deep": a.deep}
    ok, rows = evaluate(all_runs, specs, a.k, b)
    for r in rows:
        print(f"{r['qid']:4} {r['run']} {r['depth']:6} facts {r['facts']:5} hits {r['hits']:6} "
              f"chars {r['chars_before']:>6,} -> {r['chars_after']:>6,}"
              + (f"  LOST {r['lost']}" if r["lost"] else ""))
    before = sum(r["chars_before"] for r in rows) / len(rows)
    after = sum(r["chars_after"] for r in rows) / len(rows)
    print(f"\nK={a.k} budgets={b}: {'every fact kept' if ok else 'FACTS LOST'}")
    print(f"average evidence per run: {before:,.0f} -> {after:,.0f} chars, "
          f"~{before / cpt:,.0f} -> ~{after / cpt:,.0f} tokens")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
