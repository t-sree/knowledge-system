""""How many" answers from standard statements (facts/statements.py).

A question such as "How many branches are there?" is answered from the newest dated version of the
standard sentence that counts it ("It has over 600 employees and 22 branch office locations"), not from
whichever page a search ranks first, since older pages keep old numbers. Used only when the question asks
for the counted noun and the sentence's wording covers at least MIN_COVERAGE of the question's words
(idf-weighted): "How many employees work here?" is answered, "How many employees used volunteer time off?"
is not answered from the boilerplate employee count. An as-of question uses the snapshot view of that date.

"How many times / years ..." is answered from the newest ordinal sentence on its subject ("This was the
fifth time Forbes recognized Dupaco"): the question's rarest name must appear in the sentence or its page
title, which must cover MIN_COVERAGE of the question's words.
"""
from __future__ import annotations

import re
from typing import Optional

from knowledge.answering.facts import _fmt_date
from knowledge.answering.passages import tokens
from knowledge.facts.statements import StatementIndex
from knowledge.site import SITE

HOW_MANY = re.compile(r"\bhow many\b|\bnumber of\b", re.I)
HOW_MANY_TIMES = re.compile(r"\bhow many (times|years)\b", re.I)
MIN_COVERAGE = 0.5


class CountAnswerer:
    def __init__(self, ks) -> None:
        self.ks = ks

    def answer(self, q, index: int, ref_date: str, past: bool, snapshot_date: str):
        if not HOW_MANY.search(q.text or "") or "change" in q.intents:
            return None
        query, _ = self.ks.passage_answerer.query_tokens(q)
        query = {t for t in query if not t.isdigit()}
        unit = HOW_MANY_TIMES.search(q.text or "")
        if unit:
            # "How many times / years ...": the newest ordinal sentence on its subject ("the fifth time").
            unit = unit.group(1).lower().rstrip("s")
            query -= {unit}
            names = self.ks.passage_answerer._names(q)
            stmts = [s for s in self.ks.statements.ordinals_by_view.get(index, ())
                     if s.noun == unit and s.date <= ref_date]
            if names:
                rarest = max(names, key=self.ks.passages.idf)
                stmts = [s for s in stmts if rarest in s.words]
            ranked = sorted((s for s in stmts if self._coverage(query, s.words) >= MIN_COVERAGE),
                            key=lambda s: (s.date, self._coverage(query, s.words)), reverse=True)
            return self._answer(ranked[0], past, snapshot_date) if ranked and query else None
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
            coverage = self._coverage(query, win.words | noun_words)
            if coverage < MIN_COVERAGE:
                continue                         # the question asks about more than this sentence says
            score = (len(noun_words & query), coverage, win.date)
            if best is None or score > best[0]:
                best = (score, win)
        if best is None:
            return None
        return self._answer(best[1], past, snapshot_date)

    def _coverage(self, query: set, words) -> float:
        """Share of the question's words (idf-weighted) that `words` contains."""
        idf = self.ks.passages.idf
        total = sum(idf(t) for t in query)
        return sum(idf(t) for t in query & words) / total if total else 0.0

    @staticmethod
    def _answer(win, past: bool, snapshot_date: str):
        from contract import Answer, Source
        when = f" ({_fmt_date(win.date)})" if win.date else ""
        prefix = f"As of {_fmt_date(snapshot_date)}, " if past else ""
        text = f'{prefix}{SITE.name}\'s most recent statement{when} says: "{win.sentence}"'
        return Answer(text=text, sources=[Source(url=win.url, snapshot_id=win.snapshot_id, quote=win.sentence[:200])],
                      confident=True)
