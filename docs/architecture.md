# How a question moves through the pipeline

This follows one question from the web page to the saved answer. Function names are in `research.py` unless another file is named.

## 1. From the browser to the pipeline

- `web/src/api.ts` posts the question to `/api/research`, then listens on `/api/research/{id}/events` for live progress (server-sent events).
- `ask()` in `server/app.py` checks that Claude and NotebookLM are signed in, saves a history row (`server/store.py`, SQLite) and starts the job on a background thread. It returns the row right away.
- The job calls `research.research(...)`. Progress events are kept so a reloaded page can replay them; the streamed answer text is live only. A run ends as done, cancelled or error.
- `ask.py` calls the same `research()` from the terminal.

## 2. Main path (`ask_research`)

1. **Community notes.** Memory is searched for community notes that match the question, so the planner can see them.
2. **Planner** (`plan`, `prompts/planner.txt`). It rewrites a follow-up so it stands alone and lists what the answer has to cover (the requirements), up to eight fact questions, up to two questions that check a community claim, and backup search queries.
3. **Memory lookup.** Passages that earlier runs found and that match the question are fetched from memory.
4. **Fact questions** (`ask_facts`). A fact that was asked before comes from memory (`ledger_match`). The rest go to NotebookLM as one numbered batch, always in a new conversation.
5. **Citations to passages.** NotebookLM's citations are matched to the exact source text. If no passage comes back as exact text, the run switches to the fallback path.
6. **Evidence.** A passage that is cut off gets the source text that continues it (`hydrate`), and the remembered passages are added. New passages and fact questions are saved to memory. If everything fits the evidence budget it all goes to the answer model; otherwise it is trimmed.
7. **Answer model** (`reason`, `prompts/reasoner.txt`). It writes the cited answer, which streams to the browser. Instead of answering it can ask for more, once: a second round of fact questions runs, and the second pass has to answer.

## 3. Fallback path (`search_research`)

Used when the fact questions bring back no usable passage. It reuses the plan from the main path.

1. The planner's search queries run as NotebookLM source searches in parallel. These return raw passages, with no chat.
2. The hits are merged with remembered passages and get their continuations, the same way as in the main path.
3. An evidence map is built in Python (`evidence_map`): for each requirement, the statements found, their dates taken from the source titles, and how they relate.
4. A coverage check on a cheaper model (`coverage_check`) looks for requirements the evidence leaves open.
5. If something is open, or the answer model asks for a missing detail, one repair round of extra searches runs before the final answer.

## 4. Checking and saving the answer (end of `_research`)

| Check | Compared with |
|---|---|
| `ungrounded_quotes`, `ungrounded_quantities` | the evidence, the community notes and the user's own words |
| `ungrounded_percentages`, `threshold_phrases`, `unsourced_mechanism_verbs` | the evidence only |
| `citation_checks` | unknown passage ids, a quote that is not in the passage it cites, quotes with no citation |
| `check_synthesis` | the answer's stated reasoning against the evidence map |

Each result is logged and shown in the run details. The checks never block or change the answer.

Everything for a run is saved in `logs/<timestamp>-<slug>/`: the answer, each model call, the evidence, and `evidence-trace.json`, which records what was retrieved and used. The run is also written to memory (see [research-memory.md](research-memory.md)).

## 5. Follow-ups and repeated questions

A follow-up carries the id of the earlier run. The planner picks one of two modes. `research` is the normal path. `answer_from_turn` is the fast path: the question can be worked out from the passages the earlier turn already retrieved, so those go straight to the answer model, with at most one new fact question.

When the same question is asked again (not as a follow-up), the stored answer is not served. Its text and cited passages go into the new run as a draft to re-check (`prior_answer`).

## 6. Outside commands

Every model and search call is a child process.

| What | Command |
|---|---|
| Model steps | `claude -p --model ... --system-prompt-file ...` |
| NotebookLM question | `notebooklm ask -n <notebook> --new --yes --json --prompt-file ...` |
| NotebookLM search | `notebooklm source search -n <notebook> --limit ... --json` |
| Source text and titles | `notebooklm source fulltext` / `source list` (full text is cached on disk) |
| Gemini (optional) | `agy` in stream-json mode, through `tools/gemini_worker/worker.py` |

Gemini does the coverage check and the fact matching when it is available; otherwise Claude does them. A network failure or timeout pauses Gemini for ten minutes.

## 7. Sign-ins

The Connections page under Settings in the UI shows Claude, NotebookLM and Gemini. Research needs the first two. The app never reads credentials; it runs each tool's own commands (`server/connections.py`).

| Provider | Status check | Sign-in |
|---|---|---|
| Claude | `claude auth status --json` | `claude auth login --claudeai` |
| NotebookLM | `notebooklm auth check --test --json` | `notebooklm login --browser chrome` |
| Gemini | `agy -p /quota --output-format json` | `agy` |

- A sign-in that has expired or was rejected shows as "sign in again". Network trouble, rate limits, quota and timeouts are not treated as sign-in problems (`classify_failure`).
- Disconnect only tells this app to stop using a provider (a flag in `data/connections.json`). The sign-in saved on the computer is left alone, because other tools may use it.

## 8. Switched off but still in the code

- **Selector** (`RUNTIME_SELECTOR_ENABLED = False`): a model step that judged every passage against the requirements. Passages are passed through or trimmed instead.
- **Serving a stored result** for a repeated question (`EXACT_REUSE_SERVE = False`).
- **Reuse across related questions** (`LEGACY_RELATED_REUSE_ENABLED = False`).
- **Reuse gate** (`need_shadow`, `research_need.py`): it compares the question with earlier ones and records in the run trace whether reuse would have been safe. It never changes what is retrieved.

## Eval run

On 2026-10-01 I ran the six questions in `evals/questions.json` live against the sample corpus and scored the answers with `tools/eval.py`. The scorer checks with regular expressions that the key amounts, dates and phrases appear in the answer. It does not judge whether an answer is correct.

**6 of 6 passed.** The sample documents, questions and patterns were written before the run and stayed unchanged after it, and each question was run once. The scorer also reported one number in the first answer that it could not find in the passages ("four parts", my wording for the list, not a figure from a source). The community-notes question passes only after the sample community notes are loaded (step 3 in `sample_corpus/README.md`). An earlier run on a different sample topic gave 5 of 6, so the score can change from run to run.

The scorer's summary table for that run (`facts` is required patterns found; `ground` is grounding failures; `nums` is numbers it could not find in the passages):

```
id    facts   first  total  ground nums
s1    6/6      48.3   59.3       0    1
s2    3/3      40.5   43.2       0    0
s3    3/3      37.7   40.8       0    0
s4    4/4      65.5   71.9       0    0
s5    3/3      45.9   52.2       0    0
s6    3/3      39.2   42.1       0    0
```

## Known limits

- **The scores come from the sample corpus only.** I tuned the pipeline on my private library, so the results mentioned in code comments cannot be reproduced here.
- **The web UI is covered by the type check and the build** (`npm run build`).

- **Shaped by its first library.** The pipeline assumes one main author, dated talks with `Q:` / `A:` lines and recipe-like passages with amounts. The number checks know kitchen-style units, Fahrenheit and durations (seconds to years). They do not know dollar amounts, so a figure like "180 dollars per night" is not checked against the passages; days, weeks and years are. Percentages have their own check (section 4).
- **Library-specific settings.** The alias groups and generic subject words in `research.py`, the source-title patterns, the alias list in `prompts/planner.txt` and the example chips in the UI are set for the sample corpus.
- **Unused code is kept** (see section 8).
- **Developed on Windows.** The offline tests also run on Linux in GitHub Actions.
