# Issues log

Things found while building and testing that need attention later. Newest first within each section.
Status: **open** (needs a decision or work), **watch** (handled for now, may come back), **resolved**.

## Open

| # | Found in | Issue | Impact | Suggested handling |
|---|---|---|---|---|
| 42 | H–K step 2 (2026-11-13) | "How many branches?" — the site states 23 (older articles, older footer) and 22 (current newsroom footer, after the Hy-Vee closure). Passage ranking flips between them; "newest page first" and "most pages" rules both proved unstable, so neither is used. | dup-073 fails (conflict_resolution). | Resolve in G (count statements as conflicting values, newest wins) or with the step 3 LLM over dated passages. |
| 43 | H–K step 2 (2026-11-13) | Questions worded differently from the site: "Who is eligible to become a member?" (site: "You can find us in Iowa, Illinois and Wisconsin"), "How can I update my contact information?", "student … loan options with flexibility". Keyword search finds related but wrong passages. | dup-081, dup-060, dup-101 fail. | Step 3: embeddings for retrieval + gpt-4o-mini composition. |
| 44 | H–K step 2 (2026-11-13) | Hours vary by branch; extractive answers list two or three branch schedules verbatim (some repeated). "Do all branches keep the same hours?" needs a synthesised "No, at least two schedules …". | Readable but raw; dup-104 gets 0.5. | Step 3 LLM phrasing over the same passages. |
| 45 | H–K step 2 (2026-11-13) | Expected sources for hours questions are the home page `/` and `/contact`; we cite branch newsroom posts. | Source credit on dup-064 / dup-066 (0.85 each). | Check whether the home page states hours in a form we index (see #6); otherwise accept. |
| 46 | H–K step 2 (2026-11-13) | Passage search uses hand-set values (minimum coverage 0.45, page-type weights, recency up to −15% over 3 years, repeated-statement boost, 0.75 / 0.4 thresholds for extra pages). Checked on the public set and 10 new dev passage questions (all pass), but they are heuristics. | Hidden questions may rank differently. | Re-check with embeddings in step 3. |
| 47 | H–K step 2 (2026-11-13) | Ingest time rose from ~5 s to ~13 s with the passage index (19k unique chunks). | Far below the 30-minute limit. | None. |
| 32 | H–K step 1 (2026-11-13) | Product matching uses hand-set weights (section/page words 0.6, specificity 0.7–1.0, breadth 0.7–1.0, list cut-off 80% of best). They were checked on the public set **and** our own 30-question dev set (`dev/questions.jsonl`), but they are heuristics. | Hidden questions with unusual wording can pick the wrong product. | Step 3 adds embeddings + gpt-4o-mini fallback for weak/ambiguous matches; re-check weights then. |
| 33 | H–K step 1 (2026-11-13) | Credit-card list answers include rows of a junk product `card` (deposit tiers from `/newcard-balancetransfer-lp`, named after a page heading). | Noise in the answer text; a judge may mark it down. Keywords still pass. | Revisit when tightening product names (C) or filter list answers to products whose name matches the question's product noun. |
| 34 | H–K step 1 (2026-11-13) | Row names can repeat words: "Home Loans - Purchase Purchase: 30-Year Fixed Conventional …" (section name + row label). Mortgage answers list every tier for purchase and refinance (long). | Readability only. | Step 3 can phrase long lists; or trim repeated words in templates. |
| 35 | H–K step 1 (2026-11-13) | Expected sources often point at product pages (`/personal/loans/home-loans`, `/personal/loans/auto-motorcycle-loans`, `/business/cash-management`) while the winner is `/rates`. | Source credit (15%) on 8 public items. | Done: sources = winner + pages stating the same value, product pages first (up to 5). |
| 18 | E/F (2026-11-13) | Stale pages still present in the latest view carry old rates: `/homebuying-guidance` and `/mortgage-refi` show the May mortgage table (e.g. 30-yr 6.000%) next to `/rates` at 7.25%. | Classic stale trap. E picks `/rates` on content date (2026-09-25 vs 2026-05-11). | G must report these as conflicts with the stale page named; K must never cite them for "current" questions. |
| 19 | E/F (2026-11-13) | `/switch-and-earn` (promo, rates "effective 10/22/2024" for personal loans) shows the same personal-loan APRs as `/rates` today (36 mo 10.49%), but its auto/boat tables differ from `/rates`. | Promo pages mix old and current tables. | Authority already ranks promo below rate sheet; G should flag the differing auto/boat values. |
| 20 | E/F (2026-11-13) | Scholarship "Awards announced" says April 2026 on `/about/what-is-dupaco/our-impact/` but April 2027 on `/community` and `/foundation/`. Both pages undated, so E decides on URL order. | Wrong winner possible (likely 2027 is the current cycle). | G: report as conflict; consider "majority of present pages" as a tie-break before URL for non-rate facts. Needs your call. |
| 21 | E/F (2026-11-13) | Prose rates without tier on `/earn-more` ("0.00% APY" vs "3.66% APY") collide under one key. | Low: the tiered table facts are separate and rank above prose. | K should prefer table facts; ignore prose-only collisions. |
| 22 | E/F (2026-11-13) | Newsroom posts give branch hours per branch ("Lobby", "Drive-up" sub-headings); product names for these come from the post title (e.g. "credit union celebrate mineral point branch ribbon cutting"). | Hours questions need branch names; structured keys are clumsy here. | J passage retrieval for hours/branch questions; structured facts only as support. |
| 1 | C/D (2026-11-13) | The same value is written differently across pages and snapshots: `$25,000` vs `$25,000.00`, `20.5%` vs `20.50%`, points `0.00%` vs `0`, `—` vs `None`. | False conflicts if compared as text. | Handled in E/F: values are compared as numbers + units; `—`/None/N/A/Not applicable count as the same "no amount". G must use the same comparison. |
| 2 | C/D (2026-11-13) | One page can state the same fact more than once in one snapshot, e.g. a hero line "as low as 6.25% APR" plus the rate table, or a payment example "If you borrow $10,000 at 6.74% APR". | Two sightings of one key on one page. | Handled in E: tie-break rate sheet first, then table > key-value > prose. |
| 3 | C/D (2026-11-13) | Some product names are noise from calculator and article headings (`'submit'`, `'monthly breakdown'`, `'move that right money'`, 29 sightings with an empty product). | Low: they rank low and rarely match a question. Could pollute H's vocabulary. | H should only use vocabulary from rate-sheet / product / info pages for rules-based matching; revisit if wrong matches show up. |
| 4 | C/D (2026-11-13) | Article lists such as garage-sale prices ("Shoes: $3-$10") become key-value facts. | Low, article authority is lowest. | Watch in J; drop key-value facts from articles if they cause wrong answers. |
| 5 | C/D (2026-11-13) | `hubfs/Visa-Comparison-Chart-202502.pdf` has scrambled table columns after PDF conversion, so its table is skipped. | Its facts (annual fee, minimum line of credit, cash advance fee) are only reachable through passage retrieval. | J passage retrieval; check list/credit-card questions in testing. |
| 6 | C/D (2026-11-13) | Branch hours, branch addresses, closures, counts and events are mostly free text, not structured facts. Some branch hours appear as key-value lines in newsroom posts ("Saturday: 8:30 a.m.-12:30 p.m."). | These questions (lobby/drive-up hours, Galena address, Hy-Vee closing, branch count) depend on J + K. | J passage retrieval ranked by authority and date; possibly the deferred article-event LLM. |
| 7 | Planning (2026-11-13) | The "targeted LLM for article event facts" in component C was deferred until H–K work. | Event questions may need it. | Decide after H–K testing shows what passage retrieval misses. |
| 8 | Eval (earlier) | dup-005 has an empty `must_include`, so it scores 0 without the judge; local max is ~0.97. | Local score ceiling only. | None. |
| 9 | Eval (earlier) | Unknown whether graders run `--judge` and whether they provide an OpenAI key. | Shapes answer wording and the LLM fallback. | Ask the organizers; designed as if the judge is on and the rules-only path works without a key. |

## Watch

| # | Found in | Issue | Note |
|---|---|---|---|
| 10 | C/D (2026-11-13) | Rate floors/caps in card terms ("never be less than 12.00%", "never be greater than 21.00%") are tagged as `variant=floor/cap`. | Detected from the words just before the number; other phrasings may slip through as plain rates. |
| 11 | C/D (2026-11-13) | The architecture doc could not be re-read through the docs connector (server error); its C/D text was taken from the earlier conversation copy. | Re-check the doc when the connector works. |
| 12 | B (earlier) | End dates in articles are noisy (IRA deadlines, grant windows). | End dates apply only to facts from the same block, never to a whole page. |

## Decisions

| Date | Decision |
|---|---|
| 2026-11-13 | A sighting whose effective date is after the reference date cannot win; it is reported as "upcoming" so K can mention it (option a). |
| 2026-11-13 | #20: for any fact, after authority / date / recency / rate-sheet-first, the value stated by more present pages wins before URL order. |
| 2026-11-13 | No hand-written synonym list. Aliases come from the site ("certificate (CD)", "(HELOC)"); step 3 adds `text-embedding-3-small` for meaning-based matching and `gpt-4o-mini` only for ambiguous cases. |
| 2026-11-13 | Rate questions are answered only from extracted facts; no matching fact = "not published" refusal (no numbers in the text). |

## Resolved

| # | Found in | Issue | Fix |
|---|---|---|---|
| 48 | H–K step 2 (2026-11-13) | Starter's naive search (all pages of all snapshots, raw word overlap, never refuses) answered every non-rate question with junk (PDF forms, spreadsheets, impact report). | Replaced by BM25 passage search inside the selected view, weighted by page type / presence / recency, with extractive answers and evidence-based refusals. |
| 49 | H–K step 2 (2026-11-13) | Names written with a non-breaking hyphen ("Hy‑Vee") and the `$` in "Home$tart" broke matching; "October" in a question was treated as a required name. | Names compared as tokens; `$` kept inside words; month / weekday names excluded. |
| 50 | H–K step 2 (2026-11-13) | Address / hours lines never contain the words "address" / "hours"; FAQ questions have their answer in the next block; schedules are written one short line per block under a label ("**Lobby**" / "Monday-Thursday: …"). | Pattern evidence (time, street address, phone, URL, year for "when", number or number word for "how many") stands in for those words; FAQ answers appended; a label and its short lines are indexed as one chunk. |
| 51 | H–K step 2 (2026-11-13) | Word forms differ between question and site ("flexibility" / "flexible", "recognition" / "recognized"). | Light prefix stemming (6 characters) in passage search. |
| 36 | H–K step 1 (2026-11-13) | "auto loan" picked the stale switch-and-earn promo table: the product prior came from the whole history (older snapshots had it on a product page). | Prior is computed per snapshot view from the pages present in it. |
| 37 | H–K step 1 (2026-11-13) | Question filler broke matching: "does" was singularised to "doe"; "between", "as", "provide", "their", "tiers" counted as unexplained words; "APRs" was not an attribute; adding "saving" to the verb list removed the product word "savings". | Stop words checked before singularising; general filler list; "saving" kept. |
| 38 | H–K step 1 (2026-11-13) | A balance tier matched one dollar above its top ($10,000 fell into $2,500–$9,999.99). | Tier bounds use a one-cent tolerance. |
| 39 | H–K step 1 (2026-11-13) | APY was hidden when equal to the dividend rate (0.15% / 0.15%). | Only an "interest rate" that repeats the APR text is deduplicated. |
| 40 | H–K step 1 (2026-11-13) | "60-month share certificate" stopped at the best word match (Share Savings, no terms) and refused. | Falls through to the next products above the threshold before refusing. |
| 41 | H–K step 1 (2026-11-13) | A word the site never uses ("share" on the selftest site) counted fully against the match and caused a refusal. | Unknown words weigh 30%. |
| 23 | E/F (2026-11-13) | Certificate specials in the May–Aug snapshots put the product name in the "Term" column ("10 Month Fixed Jumbo Certificate Special") with no label column, so they had no product. The 10-month jumbo timeline (3.90% → 4.00%, needed for dup-006) was missing. | A qualifier cell that names a product is used as the row label when there is no label column. |
| 24 | E/F (2026-11-13) | "Model Year 2026-23" (label first, descending years) was not parsed, so ATV rows of different model years collided. | Both orders and directions parsed. |
| 25 | E/F (2026-11-13) | Column headers with qualifiers ("Dividend Rate for 1st 6 Months" vs "... After Promotional Period") collided. | Header-level `introductory` / `after promotional` variants. |
| 26 | E/F (2026-11-13) | `$10,000-24,999` (second `$` missing) was not read as a loan amount. | Money ranges accept a missing second `$`. |
| 27 | E/F (2026-11-13) | Plural product names ("Personal Loans", "Auto Loans") were not recognised as products, so products fell back to generic section names ("Loan Rates"). "personal" was also stripped as an account type, breaking "personal loan". | Plural-aware product check; only "business" is an account-type qualifier. |
| 28 | E/F (2026-11-13) | Routing number was split across junk product keys ("dubuque ia", "sign up"). | Routing number is a site-wide fact (no product); resolves to `/contact` with 5 agreeing pages. |
| 29 | E/F (2026-11-13) | Bold-only lines ("**Drive-up**") were not used as sub-headings, so lobby and drive-up hours collided. | Bold-only lines count as sub-headings; labelled values keep the nearest sub-heading as a variant. |
| 30 | E/F (2026-11-13) | Equal values on `/rates` and a product page were cited from whichever URL sorted first. | Tie-break cites the rate sheet first. |
| 31 | E/F (2026-11-13) | Floor/cap wording with words in between ("floor rate as low as 4.00% APR") was missed. | Floor/cap words may be up to four words before the number. |
| 13 | C/D (2026-11-13) | Earlier notes said the carried-forward Visa comparison PDF is "the likely source of the stale 9.9%". No snapshot contains 9.9% / 9.90%. | Corrected here; no such stale value exists in the public snapshots. |
| 14 | C/D (2026-11-13) | Ingest went from ~7 s to 54 s after adding the extractor (phrase matching rebuilt regexes millions of times). | Word-boundary string checks + caches; back to ~7.5 s. |
| 15 | C/D (2026-11-13) | Marketing headings ("Put your equity to work") were used as product names. | Headings addressed to the reader (you/your/we/our) are skipped as product context. |
| 16 | C/D (2026-11-13) | Calculator example outputs and PDF form tables (`Col2`, `Col3` headers) produced junk facts. | Calculator pages keep only rate facts; tables with placeholder headers are skipped. |
| 17 | C/D (2026-11-13) | Labels starting with a digit ("6-month certificate APY: 2.95%") were missed. | Key-value labels may start with a digit; footnote lines ("1Term Share ...") are still excluded. |
