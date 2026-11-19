# Notes

## Status

Official command from a fresh clone, rules only (no key): final **0.895**, 0 errors, 0 stale-value hits,
p95 0.02 s, ingest 23 s. With `OPENAI_API_KEY`: public mean item score 0.921, p95 ~1.7–2.1 s, ingest
~50 s. Selftest (`harness/selftest`, a different site) 1.0. `conflicts()` returns 18 entries on the
public snapshots. Answers are deterministic: identical across runs and Python hash seeds.

Nothing is keyed to the practice questions or to this website. Page types come from general URL
conventions, the organisation's name is learned from the snapshots' domain and titles, product names
and abbreviations from the pages; word lists are general English and banking grammar.

## How it works

`build()` returns a `KnowledgeSystem` (`solution/solution.py`) with exactly the three contract methods;
each hands the call to `knowledge.pipeline.Pipeline`. Parts live in `solution/knowledge/`:

| Folder | Role |
|---|---|
| `ingest/` | page store (every version of every page, URL variants merged, pages split into blocks, boilerplate flagged), page classifier (page type, publish / "rates effective" / end dates, "offer has ended"), fact extractor (rate tables, labelled lines, rates in prose → `product + attribute + qualifiers = value`, each with its quote) |
| `facts/` | fact history (every sighting: page, snapshot, type, dates), resolver (which sighting is current), timeline (value per snapshot), standard statements (repeated count sentences, ordinal sentences) |
| `query/` | question parser (intent, attribute, term, amount, model year, variant, business, dates) and the vocabulary learned from the site |
| `answering/` | fact answers (rates, fees), "how many" answers, passage answers (search over the site's own text) |
| `llm/` | optional gpt-4o-mini question reading and embeddings |
| `conflicts.py`, `site.py`, `text.py` | conflict register, learned site name, shared text helpers |

**`ingest(snapshot)`** stores and classifies the new pages, extracts facts, indexes passages (plus
embeddings with a key), collects standard statements and records the contradictions of that snapshot.

**`answer(question, as_of)`** picks the snapshot view (the latest snapshot on or before `as_of`, or a
date named in the question; `as_of` may be a date object, `yyyy-mm`, `yyyy` or a written date), then:
1. **Rates and fees** come only from extracted facts, resolved by the rules below. A rate the site
   doesn't publish is refused with no number; one listed earlier but gone now is "not listed anymore".
2. **"How many" questions** use the newest dated version of the site's standard sentence
   ("It has over 600 employees and 22 branch office locations") or, for "how many times / years", the
   newest ordinal sentence ("the fifth time Forbes recognized…"), if it covers the question's words.
3. **Everything else** (hours, addresses, programs, events) is answered from the site's own passages
   via BM25 (+ embeddings), with the evidence the question needs (a time, an address, a count, every
   name asked). Hours asked without a branch say they vary and list each distinct schedule once.
4. **Hybrid:** if the rules are not confident, gpt-4o-mini reads the question (route, product,
   qualifiers, which sentences answer it). The answer is still built by code from the site's values and
   sentences; the LLM never writes answer text, and it overrides the rules only when it adds something.

**`conflicts()`** lists two pages online in the same snapshot that state different values for the same
fact (same product, attribute, term, variants, account type, overlapping tier). Winner and reason come
from the same resolver as `answer()`. Standard count sentences whose pages disagree are listed too,
newest dated page winning. A value that changes between snapshots is an update, not a contradiction.

## What "current" means

Applied in this order when sightings of one fact disagree:

1. **Qualifiers first.** Different term, tier, model year, variant (jumbo, special) or personal vs
   business are different facts, never conflicts. Rate vs APY are different attributes.
2. **Ended is out.** A passed end date ("offer ends June 30") or an "offer has ended" notice removes the
   sighting; a not-yet-effective rate is reported as upcoming.
3. **Authority by page type.** For rates: rate sheet and product pages > info > promo > calculator >
   newsroom > article > document. For contact details: info pages first. A calculator, PDF or old blog
   post never beats the rates page.
4. **Newer content date.** The table's "rates effective" date, else the page's publish date.
5. **Present beats carried forward.** A page missing from the latest pull is still visible, but loses to
   one that is present.
6. **Newer snapshot**, then **rate sheet first**, then **the value more pages state**.
7. **Counts and ordinals in press releases:** the newest dated statement wins; older releases keep old
   numbers.
8. **Not on the site → "I don't know."** No matching fact or passage means `confident=False` and a
   "please contact" answer, never a guess.

Values that lost are named in `conflict_note`, never in the answer text.

## API keys

- `OPENAI_API_KEY` (optional, **paid API**). Read from the environment, else from a `.env` file next to
  or above the submission. Used for `text-embedding-3-small` (retrieval) and `gpt-4o-mini` (reading
  questions the rules are unsure about). Every call has a deadline; on failure the rules answer stands.
- Without the key, or without the `openai` / `numpy` packages, everything runs rules-only.

## Running

```bash
pip install -r requirements.txt
python harness/run_eval.py --submission solution --eval eval/public_eval.jsonl
```

Dev tools (not used by the solution): `tools/score_all.py` (all question sets with latency, `--no-llm`
for rules only), `tools/ask.py` (answers with parse and score per question), and our own question sets
`tools/questions.jsonl`, `tools/paraphrases.jsonl` (rewordings of the public questions) and
`tools/probes.jsonl` (new wordings, as-of forms, counts). Issues found while testing are in
`docs/ISSUES.md`.
