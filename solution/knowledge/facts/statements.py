"""Standard statements (counts in repeated paragraphs), shared by the conflict register (G3) and the
"how many" answers.

Organisations repeat a standard sentence on many pages and update its numbers over time ("It has over
600 employees and 22 branch office locations" on every press release). Such a sentence is found by its
wording with the numbers masked ("it has over # employees and # branch office locations"): a wording on
at least REPEATED_MIN_PAGES present pages of one snapshot, repeating its values (PAGES_PER_VALUE), is a
standard statement. Every page keeps its
own statement and its own publish date, so the newest dated page says what is current; older pages still
carrying old numbers are the contradiction. Nothing here names a topic: the counted nouns come from the
sentences ("<number> <noun phrase>", "<noun> has grown to <number>").
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

REPEATED_MIN_PAGES = 3
STATEMENT_PAGE_TYPES = {"newsroom", "info"}      # press-release boilerplate and about pages
FAMILY_SIMILARITY = 0.5                          # word overlap (Jaccard) of wordings that are one statement
PAGES_PER_VALUE = 2      # a standard sentence repeats each value on several pages; a yearly report
                         # sentence ("the 8th consecutive year ... 161,000 member-owners") states each once

_PHRASE_STOP = {"with", "and", "of", "in", "the", "by", "for", "to", "from", "at", "on", "a", "an", "as"}
_YEARISH = re.compile(r"^(?:19|20)\d{2}$")
_UNIT_WORDS = {"days", "weeks", "months", "years", "hours", "minutes", "seconds", "miles", "percent", "points"}
_COUNT_AFTER = re.compile(r"(?<![\d$.,])(\d{1,3}(?:,\d{3})+|\d+)\s+((?:[A-Za-z-]+\s+){0,2}[A-Za-z-]+s)\b")
_COUNT_BEFORE = re.compile(
    r"\b([A-Za-z]+)\s+(?:has grown to|totals|totaling|exceeding|exceeds|reached)\s+"
    r"(?:more than|over|nearly|about|approximately|almost)?\s*(\$?\d[\d,.]*\s*(?:billion|million)?)", re.I)
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])|\n+")
_NUM = re.compile(r"\$?\d[\d,]*(?:\.\d+)?")


def counts_in(text: str) -> list[tuple[str, str]]:
    """(counted noun, value) pairs in plain text: "22 branch office locations", "assets exceeding $3.6 billion"."""
    out = []
    for m in _COUNT_AFTER.finditer(text):
        noun = " ".join(m.group(2).lower().split())
        if noun.split()[0] in _PHRASE_STOP or noun.split()[-1] in _UNIT_WORDS:
            continue
        if _YEARISH.match(m.group(1)):
            continue
        out.append((noun, m.group(1)))
    for m in _COUNT_BEFORE.finditer(text):
        if m.group(1).lower() in _PHRASE_STOP:
            continue
        out.append((m.group(1).lower(), " ".join(m.group(2).split())))
    return out


def masked(sentence: str) -> str:
    """The wording of a sentence with its numbers masked."""
    return " ".join(re.findall(r"[a-z#]+", _NUM.sub("#", sentence.lower())))


@dataclass(frozen=True)
class Statement:
    noun: str             # counted noun phrase ("branch office locations")
    value: str            # as written ("22", "179,000", "$3.6 billion")
    sentence: str         # the sentence it was read from
    wording: str          # masked wording (the family key)
    words: frozenset      # stemmed words of the wording (for similarity and question matching)
    date: str             # the page's publish / effective date ("" when undated)
    url: str
    snapshot_id: str
    page_key: str


class StatementIndex:
    def __init__(self, store) -> None:
        self.store = store
        self._by_version: dict[tuple[str, int], list[Statement]] = {}
        self.by_view: dict[int, list[Statement]] = {}       # standard statements on present pages

    def _version_statements(self, version) -> list[Statement]:
        ck = (version.key, version.snapshot_index)
        hit = self._by_version.get(ck)
        if hit is not None:
            return hit
        from knowledge.answering.passages import display_text, tokens
        date = (version.info.content_date if version.info else None) or ""
        out, seen = [], set()
        for b in version.blocks:
            if not any(ch.isdigit() for ch in b.text):
                continue
            for sent in _SENT.split(display_text(b.text)):
                sent = " ".join(sent.split())
                found = counts_in(sent)
                if not found:
                    continue
                wording = masked(sent)
                if wording in seen:
                    continue                       # the same sentence twice on one page counts once
                seen.add(wording)
                words = frozenset(t for t in tokens(wording) if t != "#")
                for noun, value in found:
                    out.append(Statement(noun, value, sent, wording, words, date, version.url,
                                         version.snapshot_id, version.key))
        self._by_version[ck] = out
        return out

    def add_view(self, index: int) -> None:
        """Collect the standard statements on the present pages of snapshot view `index`."""
        pages_of: dict[str, set] = defaultdict(set)
        stmts = []
        for pk, e in sorted(self.store.view(index).items()):
            info = e.version.info
            if not e.present or info is None or info.page_type not in STATEMENT_PAGE_TYPES:
                continue
            for s in self._version_statements(e.version):
                pages_of[s.wording].add(pk)
                stmts.append(s)
        self.by_view[index] = [s for s in stmts if len(pages_of[s.wording]) >= REPEATED_MIN_PAGES]

    @staticmethod
    def family(stmts: list[Statement], noun: str) -> list[Statement]:
        """The statements of one standard sentence about `noun` across its versions (similar wording,
        numbers differ). The largest such family is used; an unrelated sentence counting the same noun
        ("ten packing company employees founded it") is left out."""
        cands = [s for s in stmts if s.noun == noun]
        wordings = sorted({s.words for s in cands}, key=lambda w: sorted(w))
        best = []
        for w in wordings:
            fam = [s for s in cands if len(w & s.words) / max(1, len(w | s.words)) >= FAMILY_SIMILARITY]
            if len(fam) > len(best):
                best = fam
        if len({s.page_key for s in best}) < PAGES_PER_VALUE * len({s.value for s in best}):
            return []
        return best

    @staticmethod
    def ranked_values(family: list[Statement]) -> list[Statement]:
        """One statement per distinct value, the newest dated first (ties: the earlier URL)."""
        best: dict[str, Statement] = {}
        for s in sorted(family, key=lambda s: s.url):
            prev = best.get(s.value)
            if prev is None or s.date > prev.date:
                best[s.value] = s
        by_url = sorted(best.values(), key=lambda s: s.url)
        return sorted(by_url, key=lambda s: s.date, reverse=True)       # stable: equal dates keep URL order

    def nouns(self, index: int) -> list[str]:
        return sorted({s.noun for s in self.by_view.get(index, ())})
