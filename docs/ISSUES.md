# Issues log

Things found while building and testing that need attention later. Newest first within each section.
Status: **open** (needs a decision or work), **watch** (handled for now, may come back), **resolved**.

## Open

| # | Found in | Issue | Impact | Suggested handling |
|---|---|---|---|---|
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

## Resolved

| # | Found in | Issue | Fix |
|---|---|---|---|
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
