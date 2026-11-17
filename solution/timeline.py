"""Fact Timeline (component F).

For one fact, the resolved value in every snapshot view, merged into periods:

    value V  first seen in snapshot A, last seen in snapshot B  (+ valid_from / valid_to when the
    winning source states them)

Used for "what was X in May", "when did X change" and the temporal side of the conflict register.
Computed on first use per fact and cached until the next snapshot is ingested.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from resolve import Resolver, same_value


@dataclass
class Period:
    value: str
    fact: object                     # the winning Fact of the first view in the period
    first_index: int
    last_index: int
    first_snapshot: str
    last_snapshot: str
    first_date: str                  # snapshot dates
    last_date: str
    valid_from: Optional[str] = None # effective date stated by the source
    valid_to: Optional[str] = None   # end date stated by the source
    sources: list = field(default_factory=list)   # winning ViewSighting per view, oldest first


class Timeline:
    def __init__(self, resolver: Resolver) -> None:
        self.resolver = resolver
        self._cache: dict[tuple, list[Period]] = {}
        self._generation = -1

    def periods(self, key: tuple) -> list[Period]:
        snaps = self.resolver.history.store.snapshots
        if len(snaps) != self._generation:
            self._cache.clear()
            self._generation = len(snaps)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        out: list[Period] = []
        for rec in snaps:
            res = self.resolver.resolve(key, rec.index)
            if res.winner is None:
                continue                     # not visible or nothing eligible in this view: a gap
            w = res.winner.sighting
            cur = out[-1] if out else None
            if cur and cur.last_index == rec.index - 1 and same_value(cur.fact, w.fact):
                cur.last_index, cur.last_snapshot, cur.last_date = rec.index, rec.snapshot_id, rec.taken_on
                cur.valid_to = w.expires or cur.valid_to
                cur.sources.append(res.winner)
            else:
                out.append(Period(value=w.fact.value, fact=w.fact, first_index=rec.index, last_index=rec.index,
                                  first_snapshot=rec.snapshot_id, last_snapshot=rec.snapshot_id,
                                  first_date=rec.taken_on, last_date=rec.taken_on,
                                  valid_from=w.fact.effective, valid_to=w.expires, sources=[res.winner]))
        self._cache[key] = out
        return out
