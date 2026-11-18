"""Resolver (component E).

Given every sighting of one fact visible in one snapshot view, pick the winner in a fixed order,
so the same evidence always gives the same answer:

  1-2. same fact identity and qualifiers     guaranteed by the fact key
  3.   available in the selected view         guaranteed by the view
  4.   disqualified                           an end date stated in the same block that lies before
                                              the reference date, or a page saying the offer ended
       not yet in effect                      an effective date after the reference date; such a
                                              sighting cannot win but is reported as "upcoming"
  5.   source authority, per attribute class
  6.   content date                           newer effective / publish date wins
  7.   recency                                present page beats carried forward; newer snapshot wins
  tie  extractor (table > key-value > prose), then URL

Values are compared as parsed numbers + units, never as raw text ("$25,000" == "$25,000.00").
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

RATE_ATTRIBUTES = {"apy", "apr", "rate", "dividend_rate", "fee", "points", "min_open", "min_earn",
                   "min_balance", "max_balance", "payment_example", "monthly_payment"}
CONTACT_ATTRIBUTES = {"phone", "address", "routing_number"}
_DAY_WORDS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "hours")

# Lower rank = more authoritative. Page types not listed rank last.
AUTHORITY = {
    "rates": {"rate_sheet": 0, "product": 0, "info": 1, "promo": 2, "calculator": 3, "newsroom": 4,
              "article": 5, "document": 6, "legal": 7},
    "contact": {"info": 0, "product": 1, "rate_sheet": 1, "legal": 2, "promo": 3, "newsroom": 3,
                "article": 4, "document": 5, "calculator": 6},
    "other": {"product": 0, "info": 0, "rate_sheet": 0, "promo": 1, "newsroom": 2, "article": 3,
              "document": 4, "legal": 4, "calculator": 5},
}
_LAST = 99
EXTRACTOR_RANK = {"table": 0, "kv": 1, "prose": 2}
CRITERIA = ("source authority", "content date", "present in the view", "newer snapshot", "rate sheet first",
            "more pages agree", "extractor", "url")


def attribute_class(attribute: str) -> str:
    if attribute in RATE_ATTRIBUTES:
        return "rates"
    if attribute in CONTACT_ATTRIBUTES or any(w in attribute for w in _DAY_WORDS):
        return "contact"
    return "other"


_NONE_VALUE = re.compile(r"^\W*(?:none|n/?a|not applicable|no charge|no fee)?\W*$", re.I)


def _is_none(text: str) -> bool:
    """'—', 'None', 'N/A', 'Not applicable', 'No charge' all mean "no amount"."""
    return bool(_NONE_VALUE.match(text or ""))


def same_value(a, b) -> bool:
    """Two facts carry the same value: equal numbers, units equal or one side unit-less."""
    if _is_none(a.value) and _is_none(b.value):
        return True
    na, nb = a.numbers, b.numbers
    if na or nb:
        if len(na) != len(nb):
            return False
        return all(abs(x - y) < 1e-9 and (ua == ub or not ua or not ub) for (x, ua), (y, ub) in zip(na, nb))
    return " ".join(a.value.lower().split()) == " ".join(b.value.lower().split())


def _ordinal(iso: Optional[str]) -> int:
    if not iso:
        return 0
    try:
        return date.fromisoformat(iso).toordinal()
    except ValueError:
        return 0


@dataclass
class Resolution:
    key: tuple
    winner: Optional[object] = None                 # facts.ViewSighting
    agreeing: list = field(default_factory=list)    # other eligible sightings with the winner's value
    disagreeing: list = field(default_factory=list) # eligible sightings with another value, best first
    disqualified: list = field(default_factory=list)  # (ViewSighting, reason)
    upcoming: list = field(default_factory=list)    # not yet in effect at the reference date
    reason: str = ""
    reference_date: str = ""

    @property
    def value(self) -> Optional[str]:
        return self.winner.sighting.fact.value if self.winner else None

    @property
    def fact(self):
        return self.winner.sighting.fact if self.winner else None


def rank_tuple(vs, support: int = 0) -> tuple:
    """Sort key, lower is better. `support` = number of present pages stating the same value."""
    s = vs.sighting
    cls = attribute_class(s.fact.attribute)
    return (AUTHORITY[cls].get(s.page_type, _LAST),
            -_ordinal(s.content_date),
            0 if vs.present else 1,
            -s.snapshot_index,
            0 if s.page_type == "rate_sheet" else 1,     # equal evidence: cite the rate sheet
            -support,                                     # then the value more present pages agree on
            EXTRACTOR_RANK.get(s.fact.extractor, 9),
            s.url)


def _describe(vs) -> str:
    s = vs.sighting
    bits = [s.page_type.replace("_", " ") + " page", s.url]
    if s.content_date:
        bits.append(f"dated {s.content_date}")
    if not vs.present:
        bits.append(f"carried forward from {s.snapshot_id}")
    return ", ".join(bits)


def resolve(key: tuple, sightings: list, reference_date: str) -> Resolution:
    res = Resolution(key=key, reference_date=reference_date)
    eligible = []
    for vs in sightings:
        s = vs.sighting
        if s.expires and s.expires < reference_date:
            res.disqualified.append((vs, f"offer ended {s.expires}"))
        elif s.page_expired:
            res.disqualified.append((vs, "page says the offer has ended"))
        elif s.content_date and s.content_date > reference_date:
            res.upcoming.append(vs)
        else:
            eligible.append(vs)
    res.upcoming.sort(key=rank_tuple)
    if not eligible:
        res.reason = "no eligible sighting" + (f" ({len(res.disqualified)} disqualified)" if res.disqualified else "")
        return res
    support = {}
    for vs in eligible:
        pages = {o.sighting.page_key for o in eligible if o.present and same_value(o.sighting.fact, vs.sighting.fact)}
        support[id(vs)] = len(pages)
    eligible.sort(key=lambda v: rank_tuple(v, support[id(v)]))
    res.winner = eligible[0]
    wf = res.winner.sighting.fact
    for vs in eligible[1:]:
        (res.agreeing if same_value(vs.sighting.fact, wf) else res.disagreeing).append(vs)
    res.reason = "winner: " + _describe(res.winner)
    if res.disagreeing:
        loser = res.disagreeing[0]
        wt, lt = rank_tuple(res.winner, support[id(res.winner)]), rank_tuple(loser, support[id(loser)])
        crit = next((CRITERIA[i] for i in range(len(wt)) if wt[i] != lt[i]), "tie")
        res.reason += f"; beats {loser.sighting.fact.value!r} on {crit} ({_describe(loser)})"
    if res.disqualified:
        res.reason += f"; {len(res.disqualified)} disqualified ({res.disqualified[0][1]})"
    return res


class Resolver:
    """Resolves facts per snapshot view, with per-view indexes and a result cache."""

    def __init__(self, history) -> None:
        self.history = history
        self._index: dict[int, dict[tuple, list]] = {}
        self._cache: dict[tuple, Resolution] = {}
        self._generation = -1

    def _check_generation(self) -> None:
        n = len(self.history.store.snapshots)
        if n != self._generation:
            self._index.clear()
            self._cache.clear()
            self._generation = n

    def view_index(self, index: int) -> dict[tuple, list]:
        """Fact key -> visible sightings in snapshot view `index`."""
        self._check_generation()
        idx = self._index.get(index)
        if idx is None:
            idx = {}
            for vs in self.history.sightings_in_view(index):
                idx.setdefault(vs.sighting.key, []).append(vs)
            self._index[index] = idx
        return idx

    def resolve(self, key: tuple, index: int, reference_date: Optional[str] = None) -> Resolution:
        self._check_generation()
        ref = reference_date or self.history.store.snapshots[index].taken_on
        ck = (key, index, ref)
        hit = self._cache.get(ck)
        if hit is None:
            hit = resolve(key, self.view_index(index).get(key, []), ref)
            self._cache[ck] = hit
        return hit
