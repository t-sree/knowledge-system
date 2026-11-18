"""Fact History (component D).

Records every sighting of every fact: which page version said it, in which snapshot, with the
page's type and dates. Nothing is resolved here; the resolver (E) decides between sightings.

A fact is identified by (product key, attribute, qualifiers). Sightings with different qualifiers
are different facts and never compete.

The closed vocabulary (product keys, attributes, qualifier values) that the question parser
matches against is built here from the snapshots, at ingest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from knowledge.ingest.extract import Fact, FactExtractor


@dataclass(frozen=True)
class Sighting:
    fact: Fact
    version: object            # store.PageVersion the fact was read from

    # -- convenience accessors used by the resolver and the answer composer --
    @property
    def key(self) -> tuple:
        return self.fact.key

    @property
    def url(self) -> str:
        return self.version.url

    @property
    def page_key(self) -> str:
        return self.version.key

    @property
    def snapshot_id(self) -> str:
        return self.version.snapshot_id

    @property
    def snapshot_index(self) -> int:
        return self.version.snapshot_index

    @property
    def snapshot_date(self) -> str:
        return self.version.taken_on

    @property
    def page_type(self) -> str:
        return self.version.info.page_type if self.version.info else "info"

    @property
    def content_date(self) -> Optional[str]:
        """The table's own effective date when it has one, else the page's content date."""
        if self.fact.effective:
            return self.fact.effective
        return self.version.info.content_date if self.version.info else None

    @property
    def expires(self) -> Optional[str]:
        return self.fact.expires

    @property
    def page_expired(self) -> bool:
        return bool(self.version.info and self.version.info.expired_notice)


@dataclass(frozen=True)
class ViewSighting:
    """A sighting as visible in one snapshot view."""
    sighting: Sighting
    present: bool              # False = the page was carried forward from an earlier snapshot


class FactHistory:
    def __init__(self, store) -> None:
        self.store = store
        self.extractor = FactExtractor()
        self.by_version: dict[tuple[str, int], list[Sighting]] = {}   # (page key, snapshot index) -> sightings
        self.by_key: dict[tuple, list[Sighting]] = {}
        self.products: dict[str, set[str]] = {}        # product key -> display names seen
        self.attributes: dict[str, int] = {}           # attribute -> number of sightings
        self.qualifier_values: dict[str, set] = {}     # qualifier name -> values seen
        self.errors = 0
        self._views: dict[int, list[ViewSighting]] = {}

    def add_snapshot(self, rec) -> None:
        """Extract facts for every page present in the snapshot record (after classification)."""
        for page_key in sorted(rec.members):          # sorted: sighting order must not depend on set order
            version = self.store.versions[page_key][-1]
            if version.info is not None and version.info.page_type == "listing":
                continue
            try:
                skip = frozenset(b.index for b in version.blocks if b.key in rec.boilerplate)
                facts = self.extractor.extract_version(version, skip)
            except Exception:
                self.errors += 1
                continue
            sightings = [Sighting(f, version) for f in facts]
            self.by_version[(page_key, rec.index)] = sightings
            for s in sightings:
                self.by_key.setdefault(s.key, []).append(s)
                f = s.fact
                if f.product_key:
                    self.products.setdefault(f.product_key, set()).add(f.product)
                self.attributes[f.attribute] = self.attributes.get(f.attribute, 0) + 1
                for q, v in f.qualifiers:
                    vals = v if q == "variant" else (v,)
                    self.qualifier_values.setdefault(q, set()).update(vals)
        self._views.clear()

    def sightings_in_view(self, index: int) -> list[ViewSighting]:
        """All sightings visible as of snapshot `index`, each flagged present or carried forward."""
        cached = self._views.get(index)
        if cached is not None:
            return cached
        out = []
        for page_key, entry in self.store.view(index).items():
            for s in self.by_version.get((page_key, entry.version.snapshot_index), ()):
                out.append(ViewSighting(s, entry.present))
        self._views[index] = out
        return out
