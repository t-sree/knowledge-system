""""How many" answers from standard statements (facts/statements.py).

A question such as "How many branches are there?" is answered from the newest dated version of the
standard sentence that counts it ("It has over 600 employees and 22 branch office locations"), not from
whichever page a search ranks first, since older pages keep old numbers. Used only when the question asks
for the counted noun and the sentence's wording covers at least MIN_COVERAGE of the question's words
(idf-weighted): "How many employees work here?" is answered, "How many employees used volunteer time off?"
is not answered from the boilerplate employee count. An as-of question uses the snapshot view of that date.
"""
from __future__ import annotations

import re
from typing import Optional

from knowledge.answering.facts import _fmt_date
from knowledge.answering.passages import tokens
from knowledge.facts.statements import StatementIndex
from knowledge.site import SITE

HOW_MANY = re.compile(r"\bhow many\b|\bnumber of\b", re.I)
MIN_COVERAGE = 0.5


class CountAnswerer:
    def __init__(self, ks) -> None:
        self.ks = ks

    def answer(self, q, index: int, ref_date: str, past: bool, snapshot_date: str):
        from contract import Answer, Source
        if not HOW_MANY.search(q.text or "") or "change" in q.intents:
            return None
        query, _ = self.ks.passage_answerer.query_tokens(q)
        query = {t for t in query if not t.isdigit()}
        if not query:
            return None
        stmts = [s for s in self.ks.statements.by_view.get(index, ()) if s.date <= ref_date]
        best: Optional[tuple] = None
        for noun in sorted({s.noun for s in stmts}):
            noun_words = set(tokens(noun))
            if not noun_words & query:
                continue                         # the question doesn't ask for this count
            ranked = StatementIndex.ranked_values(StatementIndex.family(stmts, noun))
            if not ranked:
                continue
            win = ranked[0]
            known = win.words | noun_words
            idf = self.ks.passages.idf
            coverage = sum(idf(t) for t in query & known) / sum(idf(t) for t in query)
            if coverage < MIN_COVERAGE:
                continue                         # the question asks about more than this sentence says
            score = (len(noun_words & query), coverage, win.date)
            if best is None or score > best[0]:
                best = (score, win)
        if best is None:
            return None
        win = best[1]
        when = f" ({_fmt_date(win.date)})" if win.date else ""
        prefix = f"As of {_fmt_date(snapshot_date)}, " if past else ""
        text = f'{prefix}{SITE.name}\'s most recent statement{when} says: "{win.sentence}"'
        return Answer(text=text, sources=[Source(url=win.url, snapshot_id=win.snapshot_id, quote=win.sentence[:200])],
                      confident=True)
