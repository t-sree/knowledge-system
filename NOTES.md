# Notes

## Status

Official command from a fresh clone, rules only (no key): final **0.895**, 0 errors, 0 stale-value hits,
p95 0.02 s, ingest 9–23 s. With `OPENAI_API_KEY`: public mean item score 0.921, p95 ~1.7–2.1 s, ingest
~50 s. Selftest (`harness/selftest`, a different site) 1.0. `conflicts()` returns 21 entries on the
public snapshots. Rules-only answers are deterministic (identical across runs and Python hash seeds);
with the key, gpt-4o-mini's reading can vary slightly between runs.

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
date named in the question; `as_of` may be a date object, `yyyy-mm`, `yyyy` or a written date; a date
before the first snapshot gets an honest "too early" answer), then:
1. **Rates and fees** come only from extracted facts, resolved by the rules below. A rate the site
   doesn't publish is refused with no number; one listed earlier but gone now is "not listed anymore".
   A question naming only an attribute ("your phone number?") takes the value most pages state.
2. **Change questions** ("How has the 10-month certificate rate changed since May?") are answered from
   each fact's value per snapshot (`facts/timeline.py`): old → new value with dates, or "listed since".
3. **"How many" questions** use the newest dated version of the site's standard sentence
   ("It has over 600 employees and 22 branch office locations") or, for "how many times / years", the
   newest ordinal sentence ("the fifth time Forbes recognized…"), if it covers the question's words.
4. **Everything else** (hours, addresses, programs, events) is answered from the site's own passages
   via BM25 (+ embeddings), with the evidence the question needs (a time, an address, a count, every
   name asked, an announcement for "what was added"). Hours asked without a branch say they vary and
   list each distinct schedule once.
5. **Hybrid:** if the rules are not confident, gpt-4o-mini reads the question (route, product,
   qualifiers, which sentences answer it). The answer is still built by code from the site's values and
   sentences; the LLM never writes answer text, and it overrides the rules only when it adds something.

**`conflicts()`** lists two pages online in the same snapshot that state different values for the same
fact (same product, attribute, term, variants, account type, overlapping tier); the values checked are
rates, APYs, APRs, dividend rates and fees. Winner and reason come
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

## Key decisions

| Decision | Why |
|---|---|
| Rates and fees are answered only from extracted facts; no matching fact means a refusal with no number. | A rate guessed from an article is worse than "please contact us"; a wrong or old value scores zero. |
| Rules answer first; gpt-4o-mini reads only the questions the rules are unsure about, and never writes answer text. | Fast and repeatable for the common cases; the LLM helps with wording, while values and sentences always come from the site. |
| The LLM overrides the rules only when it adds something (not a refusal over real evidence, not a shorter product list). | Early runs showed the LLM replacing correct rules answers. |
| No hand-written synonyms: abbreviations come from the site ("certificate (CD)"), meaning from embeddings. | Must work on the hidden rewordings, not only on words we thought of. |
| Nothing specific to this website or the practice questions: page types from general URL conventions, the organisation's name learned from the data. | The hidden set must be answered by the same general rules. |
| A rate that takes effect after the reference date cannot win; it is reported as upcoming. | "As of" answers must show the site as it stood then. |
| A conflict is two pages online in the same snapshot that disagree; a value changing between snapshots is an update. | That is a contradiction a member could actually see. |
| Counts in press releases: the newest dated standard sentence wins; a sentence counts as standard only when its values repeat across pages. | Old releases keep old numbers; yearly report sentences ("8th consecutive year") are not standing facts. |
| Ordinal sentences ("the fifth time") answer "how many times" questions but are not listed as conflicts. | An older release saying "fourth time" was true when published; listing it would add false conflicts. |
| Hours asked without a branch: say they vary and list each distinct schedule; sources are the pages that state them. | The site has no current hours page; schedules exist only in branch-opening releases. |
| `as_of` accepts date objects, `yyyy-mm`, `yyyy` and written dates; a partial date means the end of that period; unreadable means the latest snapshot. | The contract says "ISO date", but a robust reader shouldn't fail on another form. |
| Answers must not depend on Python's hash order (sorted iteration, exact float sums). | Asking twice must give the same answer (consistency pillar). |
| A question that names only the attribute ("What is your phone number?") takes the value most pages state. | That is the organisation's own number, not a department's or a card line's. |
| "What was added / launched recently?" is answered only from a passage that announces something; otherwise "I don't know". | The site has no "what's new" list; a passage that merely contains "new" is not an answer. |
| Left as is: the light word stemming ("ranked" vs "ranks"), conflicts that are not numbers (changed links, wording). | Changing the stemmer moves every answer; text conflicts risk many false positives. |

## API keys

- `OPENAI_API_KEY` (optional, **paid API**). Read from the environment, else from a `.env` file next to
  or above the submission. Used for `text-embedding-3-small` (retrieval) and `gpt-4o-mini` (reading
  questions the rules are unsure about). Every call has a deadline; on failure the rules answer stands.
- Without the key, or without the `openai` / `numpy` packages, everything runs rules-only.
- To grade the full (hybrid) system, set the key before running the scorer, e.g.
  `export OPENAI_API_KEY=...` (PowerShell: `$env:OPENAI_API_KEY="..."`). On the public set this raises
  the mean item score from ~0.895 (rules only) to ~0.92; answers stay within the 2 s speed target.
  No key is stored in the repository.

## Running

The website snapshots are not in this repository (they were shared for the event only). Copy the kit's
`snapshots_zip/` folder (the five zips) into the repository root first; the scorer unpacks them on the
first run.

```bash
pip install -r requirements.txt
python harness/run_eval.py --submission solution --eval eval/public_eval.jsonl
```

Dev tools (not used by the solution): `tools/score_all.py` (all question sets with latency, `--no-llm`
for rules only), `tools/ask.py` (answers with parse and score per question), and our own question sets
`tools/questions.jsonl`, `tools/paraphrases.jsonl` (rewordings of the public questions) and
`tools/probes.jsonl` (new wordings, as-of forms, counts). Issues found while testing are in
`docs/ISSUES.md`.
