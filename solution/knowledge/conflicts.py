"""Conflict Register (component G).

Lists places where the website contradicts itself: two pages online at the same time (in the same
snapshot) stating different values for the same fact. A value simply changing from one snapshot to
the next on the same page is an update, not a contradiction, and is not listed.

G1  rates and fees: sightings of the same fact on different present pages of one snapshot view.
    "Same fact" = same attribute, same term / variants / account type, product names where one
    contains the other, and tier or model-year ranges that are equal or (when a page uses shifted
    bands) overlap the most. The winner and reason come from the resolver (E), so conflicts() and
    answer() always agree. An offer page that says it has ended still counts as a contradiction.
G2  conflicts are reported per topic (product + attribute + term + variants); tiers fold into one
    entry and the same topic seen in several snapshots is one entry.
G3  counts in standard paragraphs repeated across many pages ("It has over 600 employees and 23 branch
    office locations"): different numbers on present pages are a contradiction; the newest page wins.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from knowledge.facts.history import ViewSighting
from knowledge.facts.resolve import resolve, same_value
from knowledge.site import SITE

HEADLINE_ATTRS = ("apy", "apr", "rate", "fee")
_RANGE_QUALS = ("balance_tier", "loan_amount", "model_year")
_ATTR_LABEL = {"apy": "APY", "apr": "APR", "rate": "interest rate", "fee": "fee"}
_LOW_AUTHORITY = {"article", "newsroom"}
MAX_VALUES = 8

# G3: counts in repeated standard paragraphs
REPEATED_MIN_PAGES = 3
_COUNT_PAGE_TYPES = {"newsroom", "info"}           # press-release boilerplate and about pages
_PHRASE_STOP = {"with", "and", "of", "in", "the", "by", "for", "to", "from", "at", "on", "a", "an", "as"}
_YEARISH = re.compile(r"^(?:19|20)\d{2}$")
_UNIT_WORDS = {"days", "weeks", "months", "years", "hours", "minutes", "seconds", "miles", "percent", "points"}
_AUTH_RANK = {"rate_sheet": 0, "product": 0, "info": 1, "promo": 2, "calculator": 3, "newsroom": 4,
              "article": 5, "document": 6, "legal": 7}
_COUNT_AFTER = re.compile(r"(?<![\d$.,])(\d{1,3}(?:,\d{3})+|\d+)\s+((?:[A-Za-z-]+\s+){0,2}[A-Za-z-]+s)\b")
_COUNT_BEFORE = re.compile(
    r"\b([A-Za-z]+)\s+(?:has grown to|totals|totaling|exceeding|exceeds|reached)\s+"
    r"(?:more than|over|nearly|about|approximately|almost)?\s*(\$?\d[\d,.]*\s*(?:billion|million)?)", re.I)


@dataclass
class _Pair:
    winner: ViewSighting
    loser: ViewSighting
    reason: str
    view: int


@dataclass
class _Topic:
    title: str
    pairs: list = field(default_factory=list)
    latest_view: int = -1


def _term_text(months: int) -> str:
    if months >= 120 and months % 12 == 0:
        return f"{months // 12}-year"
    return f"{months}-month"


def _overlap(a, b) -> float:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if hi < lo:
        return 0.0
    span = max(a[1], b[1]) - min(a[0], b[0])
    return 1.0 if span == 0 else (hi - lo) / span


class ConflictRegister:
    def __init__(self, ks) -> None:
        self.ks = ks
        self.topics: dict[tuple, _Topic] = {}
        self.counts: dict[str, dict] = {}         # noun -> {value text -> (newest date, sighting info)}
        self._done_views: set[int] = set()
        self.errors = 0

    # ================================================================== ingest
    def add_view(self, index: int) -> None:
        if index in self._done_views:
            return
        self._done_views.add(index)
        try:
            self._rate_conflicts(index)
        except Exception:
            self.errors += 1
        try:
            self._count_statements(index)
        except Exception:
            self.errors += 1

    # ------------------------------------------------------------------ G1
    def _product_groups(self, products: set[str]) -> dict[str, str]:
        """Product key -> group id: names where one name's words contain the other's (2+ words) merge."""
        keys = sorted(products, key=lambda p: (len(p.split()), p))
        group = {}
        for p in keys:
            words = set(p.split())
            parent = None
            for q in keys:
                if q == p or q not in group:
                    continue
                qw = set(q.split())
                small, big = (qw, words) if len(qw) <= len(words) else (words, qw)
                if len(small) >= 2 and small <= big:
                    parent = group[q]
                    break
            group[p] = parent or p
        return group

    def _rate_conflicts(self, index: int) -> None:
        R = self.ks.resolver
        vidx = R.view_index(index)
        ref = self.ks.store.snapshots[index].taken_on
        products_with_apr = {k[0] for k in vidx if k[1] == "apr"}
        by_attr: dict[str, list] = defaultdict(list)
        for key, vss in vidx.items():
            attr = key[1]
            if attr not in HEADLINE_ATTRS or not key[0]:
                continue
            if attr == "rate" and key[0] in products_with_apr:
                continue                       # loans: APR is the headline value
            for vs in vss:
                if vs.present and self._usable(vs):
                    by_attr[attr].append(vs)
        for attr, sightings in by_attr.items():
            groups = self._product_groups({vs.sighting.fact.product_key for vs in sightings})
            by_group: dict[str, list] = defaultdict(list)
            for vs in sightings:
                by_group[groups[vs.sighting.fact.product_key]].append(vs)
            for gid, vss in by_group.items():
                self._conflicts_in_group(index, ref, attr, gid, vss)

    @staticmethod
    def _usable(vs) -> bool:
        """Rate floors / caps are not the rate. A rate written in a sentence counts only on a promo page
        or when its own sentence names the product (not one borrowed from a heading)."""
        from knowledge.answering.passages import tokens
        f = vs.sighting.fact
        if {"floor", "cap"} & set(dict(f.qualifiers).get("variant", ())):
            return False
        if f.extractor != "prose" or vs.sighting.page_type == "promo":
            return True
        return set(tokens(f.product_key)) <= set(tokens(f.quote))

    @staticmethod
    def _base_quals(fact) -> tuple:
        q = dict(fact.qualifiers)
        return (q.get("account_type", ""), q.get("term"), tuple(q.get("variant", ())),
                tuple(sorted(k for k in q if k in _RANGE_QUALS)))

    def _conflicts_in_group(self, index, ref, attr, gid, vss) -> None:
        by_page: dict[str, list] = defaultdict(list)
        for vs in vss:
            by_page[vs.sighting.page_key].append(vs)
        if len(by_page) < 2:
            return
        seen_sets = set()
        for page in sorted(by_page):
            for a in by_page[page]:
                # The same fact on every other page: equal ranges, else the most-overlapping band.
                members = [a] + [m for p in sorted(by_page) if p != page
                                 for m in [self._partner(a, by_page[p])] if m is not None]
                if len(members) < 2:
                    continue
                ident = frozenset(id(m) for m in members)
                if ident in seen_sets:
                    continue
                seen_sets.add(ident)
                facts = [m.sighting.fact for m in members]
                if all(same_value(facts[0], f) for f in facts[1:]):
                    continue
                if all(m.sighting.page_type in _LOW_AUTHORITY for m in members):
                    continue
                res = resolve(a.sighting.key, members, ref)
                if res.winner is None:
                    continue
                wf = res.winner.sighting.fact
                for m in members:
                    if m is res.winner or same_value(m.sighting.fact, wf):
                        continue
                    self._add(index, attr, gid, res.winner, m, self._reason(res, m))

    def _partner(self, a, candidates):
        """The sighting on the other page that states the same fact: equal ranges, else most overlap."""
        fa = a.sighting.fact
        base = self._base_quals(fa)
        qa = dict(fa.qualifiers)
        best, best_score = None, 0.0
        for b in candidates:
            fb = b.sighting.fact
            if self._base_quals(fb) != base:
                continue
            qb = dict(fb.qualifiers)
            score = 1.0
            for r in _RANGE_QUALS:
                if r in qa:
                    score *= _overlap(qa[r], qb[r])
            if score > best_score:
                best, best_score = b, score
        return best if best_score > 0 else None

    @staticmethod
    def _reason(res, loser) -> str:
        w, l = res.winner.sighting, loser.sighting
        for vs, why in res.disqualified:
            if vs is loser:
                return f"{l.url} is out of date: {why}; {w.page_type.replace('_', ' ')} {w.url} is current"
        wt, lt = w.page_type.replace("_", " "), l.page_type.replace("_", " ")
        wd = f" (dated {w.content_date})" if w.content_date else ""
        ld = f" (dated {l.content_date})" if l.content_date else ""
        if lt == "calculator":
            return f"{wt} page {w.url}{wd} is authoritative; calculator page {l.url}{ld} is stale"
        if res.winner.sighting.page_type != l.page_type and _AUTH_RANK.get(w.page_type, 9) < _AUTH_RANK.get(l.page_type, 9):
            return f"{wt} page {w.url}{wd} outranks {lt} page {l.url}{ld}"
        if w.content_date and l.content_date and w.content_date > l.content_date:
            return f"{w.url} has the newer effective date ({w.content_date} vs {l.content_date})"
        return f"{wt} page {w.url}{wd} preferred over {lt} page {l.url}{ld}"

    def _add(self, index, attr, gid, winner, loser, reason) -> None:
        f = winner.sighting.fact
        q = dict(f.qualifiers)
        variants = tuple(v for v in q.get("variant", ()) if v != "after promotional")
        key = (gid, attr, q.get("term"), variants, q.get("account_type", ""))
        t = self.topics.get(key)
        if t is None:
            bits = []
            if q.get("account_type") == "business":
                bits.append("business")
            if q.get("term"):
                bits.append(_term_text(q["term"]))
            bits.extend(variants)
            bits.append(f.product_key)
            bits.append(_ATTR_LABEL.get(attr, attr))
            t = self.topics[key] = _Topic(" ".join(bits))
        t.pairs.append(_Pair(winner, loser, reason, index))
        t.latest_view = max(t.latest_view, index)

    # ------------------------------------------------------------------ G3
    def _count_statements(self, index: int) -> None:
        """Counts in standard paragraphs (repeated on several press-release / info pages)."""
        view = self.ks.store.view(index)
        pages_of: dict[str, set] = defaultdict(set)
        text_of: dict[str, str] = {}
        for pk, e in view.items():
            info = e.version.info
            if not e.present or info is None or info.page_type not in _COUNT_PAGE_TYPES:
                continue
            for b in e.version.blocks:
                if any(ch.isdigit() for ch in b.text):
                    pages_of[b.key].add(pk)
                    text_of.setdefault(b.key, b.text)
        for bkey, pages in pages_of.items():
            if len(pages) < REPEATED_MIN_PAGES:
                continue
            dated = [(view[p].version.info.content_date or "", p) for p in pages]
            newest, page = max(dated)
            v = view[page].version
            from knowledge.answering.passages import tokens
            words = frozenset(t for t in tokens(text_of[bkey]) if not t.isdigit())
            for noun, value in self._counts_in(text_of[bkey]):
                self.counts.setdefault(noun, []).append(
                    {"value": value, "words": words, "date": newest, "url": v.url, "sid": v.snapshot_id,
                     "pages": len(pages), "view": index})

    @staticmethod
    def _counts_in(text: str) -> list[tuple[str, str]]:
        from knowledge.answering.passages import display_text
        plain = display_text(text)
        out = []
        for m in _COUNT_AFTER.finditer(plain):
            noun = " ".join(m.group(2).lower().split())
            if noun.split()[0] in _PHRASE_STOP or noun.split()[-1] in _UNIT_WORDS:
                continue
            if _YEARISH.match(m.group(1)):
                continue
            out.append((noun, m.group(1)))
        for m in _COUNT_BEFORE.finditer(plain):
            if m.group(1).lower() in _PHRASE_STOP:
                continue
            out.append((m.group(1).lower(), " ".join(m.group(2).split())))
        return out

    def _count_family(self, noun: str) -> list[dict]:
        """Statements of one standard paragraph across its versions (similar wording, numbers differ).
        The largest such family is used; unrelated paragraphs that happen to share the noun are ignored."""
        stmts = self.counts.get(noun, [])
        best = []
        for s in stmts:
            fam = [o for o in stmts if len(s["words"] & o["words"]) / max(1, len(s["words"] | o["words"])) >= 0.5]
            if len(fam) > len(best):
                best = fam
        return best

    # ================================================================== output
    def conflicts(self) -> list:
        from contract import Conflict, Source
        out = []
        for key, t in sorted(self.topics.items(), key=lambda kv: kv[1].title):
            latest = [p for p in t.pairs if p.view == t.latest_view]
            winners, losers = [], []
            for p in latest:
                if not any(same_value(p.winner.sighting.fact, w.sighting.fact) for w in winners):
                    winners.append(p.winner)
            for p in t.pairs:
                lf = p.loser.sighting.fact
                if any(same_value(lf, w.sighting.fact) for w in winners + losers):
                    continue
                losers.append(p.loser)
            values = [w.sighting.fact.value for w in winners] + [l.sighting.fact.value for l in losers]
            values = values[:MAX_VALUES]
            srcs, seen = [], set()
            for vs in winners + losers:
                s = vs.sighting
                if s.url not in seen:
                    seen.add(s.url)
                    srcs.append(Source(url=s.url, snapshot_id=s.snapshot_id, quote=s.fact.quote[:200]))
            reason = latest[0].reason if latest else t.pairs[-1].reason
            out.append(Conflict(topic=t.title, values=values, sources=srcs[:MAX_VALUES],
                                resolution="; ".join(w.sighting.fact.value for w in winners[:4]),
                                reason=reason))
        for noun in sorted(self.counts):
            ranked = self._ranked_values(noun)
            if len(ranked) < 2:
                continue
            win = ranked[0]
            values = [s["value"] for s in ranked][:MAX_VALUES]
            srcs = [Source(url=s["url"], snapshot_id=s["sid"]) for s in ranked][:MAX_VALUES]
            older = ", ".join(f"{s['value']} ({s['date'] or 'undated'})" for s in ranked[1:4])
            out.append(Conflict(topic=f"{SITE.name} {noun}", values=values, sources=srcs, resolution=win["value"],
                                reason=f"the newest page with this standard paragraph ({win['date'] or 'undated'}, "
                                       f"{win['url']}) says {win['value']}; older pages still say {older}"))
        return out

    def _ranked_values(self, noun: str) -> list[dict]:
        """One statement per distinct value of the paragraph family, newest first."""
        best: dict[str, dict] = {}
        for s in self._count_family(noun):
            prev = best.get(s["value"])
            if prev is None or (s["date"], s["pages"]) > (prev["date"], prev["pages"]):
                best[s["value"]] = s
        return sorted(best.values(), key=lambda s: (s["date"], s["pages"]), reverse=True)
