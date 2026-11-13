# Notes

## Status

Answers rate / fee questions from extracted facts and everything else from passage search over the
selected snapshot. `conflicts()` is not implemented yet (returns `[]`).

Local results (public eval, no judge): final score 0.871 with `OPENAI_API_KEY`, 0.859 without.
p95 latency about 0.4 s; ingest about 40 s with embeddings, about 15–25 s without.

## Approach

`build()` returns a `KnowledgeSystem`. `ingest(snapshot)` runs, per snapshot:

1. **Page store** (`store.py`): every version of every page; URLs normalised (query variants merged);
   pages split into blocks with their heading path; blocks repeated on 10+ pages flagged as boilerplate.
   A view of snapshot N shows each page's latest version up to N, marked present or carried forward.
2. **Page classifier** (`classify.py`, rules): page type from URL structure, publish / "rates effective" /
   end dates with quotes, "offer has ended" notices.
3. **Fact extractor** (`extract.py`, rules): rate tables, comparison tables, labelled lines and rates in
   prose become facts `product + attribute + qualifiers = value` (term, balance / loan tier, model year,
   variant, business), each with its quote, table effective date and block end date. Every value is
   checked against its quote.
4. **Fact history** (`facts.py`): every sighting with page, snapshot, page type and dates.
5. **Vocabulary** (`vocab.py`): product names, section words, page titles and the abbreviations the site
   defines itself ("certificate (CD)") — learned from the snapshots, no hand-written synonyms.
6. **Passage index** (`passages.py`): short chunks of every block (a label with its short lines stays one
   chunk), BM25 postings, and — with an OpenAI key — `text-embedding-3-small` vectors.

`answer(question, as_of)`:

1. **Parse** (`question.py`, rules): intent (list, highest / lowest, yes-no, URL, past), attribute,
   qualifiers (term, amount, model year, variant, business), dates named in the question.
2. **View**: latest snapshot on or before `as_of` (or a date in the question); before the first snapshot
   → refusal.
3. **Fact path** (`answer.py`): products matched against the vocabulary (with a per-view quality prior),
   fact keys whose qualifiers fit, each resolved by the **resolver** (`resolve.py`) in a fixed order:
   end date passed / offer ended → out; not yet effective → reported as upcoming; source authority per
   attribute class; content date; present over carried forward; newer snapshot; rate sheet first; value
   stated by more pages; extractor. Values compare as numbers. Tier rules: menus list every row, tiered
   products lead with the base tier, a named tier / "highest" picks that row. Rejected values go to
   `conflict_note`, never into the text. A rate question with no matching fact is refused (no numbers).
4. **Passage path** (`passages.py`): hybrid BM25 + embedding search inside the view, weighted by page
   type, presence and recency; the evidence a question needs must be in the passage (a time for hours, a
   street address, a URL, a year for "when", a count for "how many", every specific name in the
   question). Extractive answer from the best passages; refusal when nothing qualifies.
5. Answers are cached per (question, as_of).

**Fact timeline** (`timeline.py`) gives each fact's value per snapshot as periods (for "what was it in
May" and for conflicts).

## API keys

- `OPENAI_API_KEY` (optional). Read from the environment, else from a `.env` file next to or above the
  submission. Used for `text-embedding-3-small` (product and passage retrieval). `gpt-4o-mini` answer
  composition is implemented but off by default (`COMPOSE_MODE` in `passages.py`): it did not beat the
  extractive answers on keyword scoring and added about 1 s per question.
- Without the key, or without the `openai` / `numpy` packages, everything runs rules-only.

## Running

```bash
python harness/run_eval.py --submission solution --eval eval/public_eval.jsonl
```

Dev tools (not used by the solution): `dev/ask.py` (answers with parse and score per question),
`dev/questions.jsonl` (our own 40 questions), `dev/compare_modes.py`, `dev/inspect_*.py` reports.
Issues found while testing are logged in `docs/ISSUES.md`.
