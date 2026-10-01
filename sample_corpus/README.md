# Sample corpus (sample data)

This folder is **sample data written for this repository**. It stands in for my private
document collection, which is not included.

- The company ("Kestrel Hollow"), its handbook and the HR lead who answers questions are
  **fictional**. I invented the amounts, limits and rules to exercise the pipeline: dated
  statements that change over time, a procedure that names a sub-procedure, advice given to one
  person, and community notes. It is not real HR or expense policy.
- The documents are short on purpose. The real collection has far more sources, so how well
  search works on this sample says little about a large library.

## Files

| File | What it is |
|---|---|
| `Benefits Workshop + Q&A Of March 9, 2021.md` | Workshop transcript with `Q:` / `A:` lines |
| `Benefits Workshop + Q&A Of October 21, 2023.md` | Later workshop; changes some earlier rules (remote days, expense claim) |
| `The Kestrel Hollow Employee Handbook.md` | Handbook; limits in bold, one sub-procedure (the travel request) in capitals |
| `Newsletters.md` | Compilation of dated newsletter items |
| `community_records.jsonl` | Three community notes (secondary material, not the HR lead's words) |

## How to use it

1. Create a notebook in NotebookLM and upload the four `.md` files as sources.
2. Put the notebook id in `.env` as `CRA_NOTEBOOK_ID` (see `.env.example` at the repo root).
3. Optional: load the community notes into research memory:

   ```
   python research_memory.py init
   python research_memory.py import-secondary sample_corpus/community_records.jsonl
   ```

The eval questions in `evals/questions.json`, the alias groups in `research.py`
(`COMMUNITY_ALIASES`) and the alias list in `prompts/planner.txt` are written for this sample.
