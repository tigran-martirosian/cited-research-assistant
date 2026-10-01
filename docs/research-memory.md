# Research memory

`research_memory.py` remembers what earlier research runs found: the exact passages NotebookLM returned, which searches and runs found them, which answers cited them, and the fact questions already asked.

It is not a search engine over the documents. NotebookLM stays the only search over the library. Memory does not read or index the source documents, uses no embeddings and calls no model.

## Layers

| Layer | What it holds | Rule |
| --- | --- | --- |
| `primary_retrieved` | Exact source text that NotebookLM returned | Stored word for word, never rewritten |
| `secondary` | Community notes (chats, forums) | Always labelled as secondary |
| `derived` | A stated inference or summary | Must link to the passages that support it |

A passage is stored once per source. Finding the same text again from the same source adds a discovery, not a new record. Overlapping passages are never merged. How often a passage was found is history, not a score of how true it is.

## In a research run

1. Before the searches, memory is asked for remembered passages and community notes that match the question (`MemoryStore.relevant`).
2. Remembered passages join the new NotebookLM passages as evidence. A passage found both ways appears once.
3. Every passage NotebookLM returns in the run is saved, with its source, title, date, query and run.
4. Community notes go to the answer model in their own `SECONDARY EVIDENCE` section. They never count toward what a requirement needs, and the model is told they are not the author's words.

Matching uses SQLite full-text search (FTS5) and word weights. A record is kept when it covers at least half of the question's weighted words and matches at least two of them.

Memory is optional. If the database is missing or empty, or a memory step fails, the run goes on with NotebookLM only. `CRA_MEMORY=off` turns memory off and `CRA_MEMORY_DB` moves the database.

## Command line

```
python research_memory.py init                              # create the database (safe to repeat)
python research_memory.py import-trace logs\<run>           # load one saved run
python research_memory.py search "later recommendation" --limit 10
python research_memory.py show <unit-id or unique prefix>
python research_memory.py history --min 2                   # passages found more than once
python research_memory.py stats
python research_memory.py secondary-schema                  # format of a community record
python research_memory.py import-secondary claims.jsonl     # one JSON record per line
```

Importing the same run or file twice changes nothing.

## Community notes

Each record becomes one `secondary` entry: the claim as it was written, where it came from (platform, thread, author, date), its type (for example a quote, a paraphrase, an anecdote) and how far it has been checked against the sources.

The sources stay the authority on what the author said:

- A community note keeps its own layer in storage, search and display.
- "Verified" is accepted only with a link to a source passage that supports it. Without one, it is stored as `unverified`.
- A community note can inspire a derived claim but can never support one.
- `find_quoted_primary()` lists source passages that contain a note's text word for word. It only suggests; nothing is linked automatically.

## Storage

The database is `data/research-memory.db`. It is local data that can be deleted and rebuilt by importing the run folders in `logs/` again. Record ids are made from the layer, the source and the text, so a rebuild gives the same ids.
