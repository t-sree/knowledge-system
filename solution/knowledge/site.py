"""The organisation behind the website, learned from the snapshots instead of written into the code.

The name is the site's own domain label (the most common host among the pages, without "www." and the
top-level domain); its display spelling is the one the page titles use most. The name and its plural /
possessive form are added to the stop-word lists that register here, since they appear on every page and
say nothing about what a question asks. Answer wording uses `SITE.name`.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Callable

_FALLBACK = "the organization"
_registered: list[tuple[set, Callable[[str], str], tuple]] = []


class Site:
    def __init__(self) -> None:
        self.name = _FALLBACK          # display name used in answers ("Acme" for acme.com)
        self.words: frozenset[str] = frozenset()   # lowercase forms treated as stop words

    def learn(self, versions) -> None:
        """Learn the name from page versions (url, title); only once, from the first snapshot."""
        if self.words:
            return
        versions = list(versions)
        hosts = Counter()
        for v in versions:
            m = re.match(r"(?:[a-z]+://)?([^/:?#]+)", (v.url or "").lower())
            if m:
                parts = [p for p in m.group(1).split(".") if p and p != "www"]
                if len(parts) >= 2:
                    hosts[parts[-2]] += 1
        if not hosts:
            return
        label = hosts.most_common(1)[0][0]
        parts = [p for p in re.split(r"[-_]", label) if p]
        # "example-cu" may be written "Example CU"; titles first, page text if no title names it.
        pat = re.compile(r"\b" + r"[\s-]+".join(map(re.escape, parts)) + r"\b", re.I)
        spellings = Counter()
        for field in ("title", "markdown"):
            for v in versions:
                spellings.update(m.group(0) for m in pat.finditer(getattr(v, field, "") or ""))
            if spellings:
                break
        self.name = spellings.most_common(1)[0][0] if spellings else " ".join(p.capitalize() for p in parts)
        self.words = frozenset({label, label + "s"} | set(parts) | {p + "s" for p in parts})
        for target, transform, cached in _registered:
            target.update(transform(w) for w in self.words)
            for fn in cached:
                fn.cache_clear()             # results computed with the old stop words


SITE = Site()


def register_stopwords(target: set, transform: Callable[[str], str] = lambda w: w, cached=()) -> None:
    """Have `target` receive the site's name words (passed through `transform`) once they are learned;
    `cached` are lru_cached functions that read `target` and must forget earlier results."""
    _registered.append((target, transform, tuple(cached)))
    target.update(transform(w) for w in SITE.words)
