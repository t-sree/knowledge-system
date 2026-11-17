# Notes

## Status

Hybrid: rules answer every question first (rate / fee questions from extracted facts, everything else
from passage search over the selected snapshot); confident rules answers are returned as is, the rest
are read by gpt-4o-mini. `conflicts()` lists contradictions found within each snapshot (selftest
conflicts F1 1.0, resolution 1.0; 19 conflicts on the public snapshots).

Local results (no judge), mean item score: public 0.884, our dev set 1.000, our rewordings 0.972, our
probe questions 0.933; selftest final 1.0. Public official final 0.871 with the key (p50 0.4 s, p95
~2.0–2.2 s), 0.859 mean without the key (p95 0.1 s). Ingest 40–90 s with embeddings.

## Approach

`build()` returns a `KnowledgeSystem` (`solution.py`) that defines exactly the three contract methods:
`ingest(snapshot)`, `answer(question, as_of=None)` and `conflicts()`. Each hands the call to
`pipeline.Pipeline`; the other files in `solution/` are its parts.

`ingest(snapshot)` runs, per snapshot:

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

`answer(question, as_of)` — hybrid:

0. The rules path below runs first. If it is confident (an exact site-wide fact, a rate with a strong
   product match covering every term asked, or a passage covering most of the question with the needed
   evidence) its answer is returned. Otherwise **gpt-4o-mini** (`understand.py`) reads the question in
   one compact JSON call: route (rate fact / passage / not on the site), products chosen only from a
   candidate list of real product keys, attribute, terms, model year, amount, variants, intent, and the
   ids of the numbered passage sentences that answer it (short reply = fast call, ~1.2 s). The answer
   is still built by code from facts and the site's own sentences, with the same evidence checks. The
   LLM may only override the rules when it adds something: an LLM refusal does not replace a rules
   answer found on real evidence (a rate fact, or a passage containing the question's own words), only
   one that was merely close in meaning; a narrower product list than the rules gave does not replace
   theirs; a rules refusal with a concrete reason (variant not offered, product discontinued) is final;
   an LLM product pick is rejected if the question names a product word it doesn't carry. Every OpenAI
   call has an enforced deadline (chat 2.5 s); on failure the rules answer stands.
1. **Parse** (`question.py`, rules): intent (list, highest / lowest, yes-no, URL, past, change, still
   available), attribute (incl. hours / address wording), qualifiers (term, amount, model year, variant,
   business), dates named in the question.
2. **View**: latest snapshot on or before `as_of` (or a date in the question); before the first snapshot
   → refusal.
3. **Fact path** (`answer.py`): products matched against the vocabulary (with a per-view quality prior),
   fact keys whose qualifiers fit, each resolved by the **resolver** (`resolve.py`) in a fixed order:
   end date passed / offer ended → out; not yet effective → reported as upcoming; source authority per
   attribute class; content date; present over carried forward; newer snapshot; rate sheet first; value
   stated by more pages; extractor. Values compare as numbers. Tier rules: menus list every row, tiered
   products lead with the base tier, a named tier / "highest" picks that row. Rejected values go to
   `conflict_note`, never into the text. A rate question with no matching fact is refused (no numbers);
   if the fact was listed in an earlier snapshot the answer says it isn't listed anymore. "What changed"
   questions are answered from the fact timeline.
4. **Passage path** (`passages.py`): hybrid BM25 + embedding search inside the view, weighted by page
   type, presence and recency; the evidence a question needs must be in the passage (a time for hours, a
   street address, a URL, a year for "when", a count for "how many", every specific name in the
   question). Extractive answer from the best passages; refusal when nothing qualifies.
5. Answers are cached per (question, as_of).

**Fact timeline** (`timeline.py`) gives each fact's value per snapshot as periods (for "what was it in
May").

**Conflicts** (`conflicts.py`), computed per snapshot at ingest. A contradiction is two pages online at
the same time (same snapshot) stating different values for the same fact; a value that simply changes
from one snapshot to the next is an update and is not listed.
- Rates and fees: same attribute, term, variants and account type; product names where one contains
  the other ("auto loan" / "auto & motorcycle loan"); tier or model-year bands equal, or the most
  overlapping band when a page uses shifted bands (a calculator's "2023–26" vs the rate sheet's
  "2024–27"). The winner and reason come from the same resolver as `answer()`. Pages saying an offer
  has ended count as out of date. One entry per topic (product + attribute + term + variants); tiers and
  snapshots fold into it.
- Counts in standard paragraphs repeated across press releases ("over 600 employees and 23 branch office
  locations"): versions of the same paragraph with different numbers; the newest page wins.

## API keys

- `OPENAI_API_KEY` (optional, paid API). Read from the environment, else from a `.env` file next to or
  above the submission. Used for `text-embedding-3-small` (product and passage retrieval, at ingest and
  one call per question) and `gpt-4o-mini` (question understanding for questions the rules are not sure
  about). The LLM never writes answer text; answers are always the site's own values and sentences.
- Without the key, or without the `openai` / `numpy` packages, everything runs rules-only.

## Running

```bash
python harness/run_eval.py --submission solution --eval eval/public_eval.jsonl
```

Dev tools (not used by the solution): `dev/ask.py` (answers with parse and score per question),
`dev/questions.jsonl` (our own 47 questions), `dev/paraphrases.jsonl` (our rewordings of the public
questions), `dev/probes.jsonl` (hard new wordings), `dev/score_all.py` (all sets with latency).
Issues found while testing are logged in `docs/ISSUES.md`.
