"""Rewrite community (secondary) records into plain-English statements.

Each record's summary line and raw chat/forum wording go to Gemini Flash (the product's Gemini
runtime, server/connections.run_gemini) with prompts/community_statement.txt; the statement is
stored in the unit's metadata ("statement", "statement_version"). The reasoner reads the
statement instead of the raw wording (research.secondary_evidence). Records that already have a
statement of the current version are skipped, so the tool can be re-run after an import.

    python tools/community_statements.py [--limit N] [--batch N] [--redo] [--dry-run]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import research  # noqa: E402
import research_memory as rm  # noqa: E402

PROMPT = (ROOT / "prompts" / "community_statement.txt").read_text(encoding="utf-8")
STATEMENT_VERSION = 1
CONTEXT_CHARS = 3000  # raw wording sent per record


def pending(store, redo):
    rows = store.db.execute(
        "SELECT u.unit_id, u.text, u.metadata, c.context FROM secondary_claims c "
        "JOIN units u USING (unit_id) ORDER BY u.unit_id").fetchall()
    out = []
    for r in rows:
        md = rm.loads(r["metadata"]) or {}
        if redo or md.get("statement_version") != STATEMENT_VERSION or not md.get("statement"):
            out.append({"unit_id": r["unit_id"], "summary": r["text"],
                        "context": (r["context"] or "")[:CONTEXT_CHARS], "metadata": md})
    return out


def rewrite(batch):
    from server import connections
    body = "\n\n".join(f"RECORD r{n}\nSUMMARY: {x['summary']}\nWORDING:\n{x['context'] or '(none)'}"
                       for n, x in enumerate(batch, 1))
    text = connections.run_gemini("community-statement", f"{PROMPT}\n\n{body}",
                                  {"stage": "community-statement"})
    out = research.parse_json_object(text) or {}
    return {batch[int(k[1:]) - 1]["unit_id"]: v.strip() for k, v in out.items()
            if isinstance(k, str) and k[1:].isdigit() and 1 <= int(k[1:]) <= len(batch)
            and isinstance(v, str) and v.strip()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=0, help="rewrite at most N records")
    ap.add_argument("--batch", type=int, default=8, help="records per Gemini call")
    ap.add_argument("--redo", action="store_true", help="rewrite records that already have one")
    ap.add_argument("--dry-run", action="store_true", help="print, do not store")
    args = ap.parse_args()
    store = research.open_memory(create=False)
    if store is None:
        sys.exit("No research memory database.")
    with store:
        todo = pending(store, args.redo)
        if args.limit:
            todo = todo[:args.limit]
        print(f"{len(todo)} records to rewrite")
        done = failed = 0
        for i in range(0, len(todo), args.batch):
            batch = todo[i:i + args.batch]
            try:
                got = rewrite(batch)
            except Exception as e:  # noqa: BLE001 - report and continue with the next batch
                print(f"batch {i // args.batch + 1}: failed ({e})")
                failed += len(batch)
                continue
            for x in batch:
                statement = got.get(x["unit_id"])
                if not statement:
                    failed += 1
                    continue
                done += 1
                if args.dry_run:
                    print(f"\n{x['unit_id']}\n  {x['summary']}\n  -> {statement}")
                    continue
                md = dict(x["metadata"], statement=statement, statement_version=STATEMENT_VERSION)
                with store.db:
                    store.db.execute("UPDATE units SET metadata = ? WHERE unit_id = ?",
                                     (json.dumps(md, ensure_ascii=False), x["unit_id"]))
            print(f"{min(i + args.batch, len(todo))}/{len(todo)} (ok {done}, failed {failed})")
    print(f"done: {done} rewritten, {failed} failed" + (" (dry run, nothing stored)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
