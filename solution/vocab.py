"""Closed vocabulary built from the snapshots (used by the question parser, component H).

Everything here is learned from the site itself, nothing is listed by hand:

  products    product keys that carry facts, with the words of their key and of the sections
              they appear under ("Credit Cards" > "Platinum Visa"), and their attributes
  aliases     abbreviations the site defines itself: "Home Equity Line of Credit (HELOC)",
              "certificate (CD)", "Individual Retirement Accounts (IRAs)"
  variants    variant words seen in fact qualifiers (jumbo, special, purchase, ...)
  idf         how informative a word is across product names and sections
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from extract import _norm, _singular, is_product_phrase, product_key

_ABBR_PAREN = re.compile(r"\(([A-Z]{2,6})s?\)")          # find "(HELOC)" first, then look back
_ABBR_BEFORE = re.compile(r"\b([A-Z]{2,6})s?\s*\(([A-Za-z][^()]{3,60})\)")
_SKIP_INITIAL = {"of", "and", "the", "for", "to", "a", "an"}
# Pages whose sections describe products; articles and listings are too noisy for the vocabulary.
_VOCAB_PAGE_TYPES = {"rate_sheet", "product", "info", "promo"}
_RATE_ATTRS = {"apy", "apr", "rate", "dividend_rate"}


def words(text: str) -> list[str]:
    return [_singular(w) for w in _norm(text).split()]


def _initials_match(cand: list[str], abbr: str) -> bool:
    full = "".join(w[0] for w in cand)
    short = "".join(w[0] for w in cand if w.lower() not in _SKIP_INITIAL)
    return abbr in (full, short)


def _expansion(preceding: str, abbr: str) -> list[str] | None:
    toks = re.findall(r"[A-Za-z][\w'’&-]*", preceding)
    abbr = abbr.lower()
    for k in range(1, min(len(toks), 8) + 1):
        cand = [t.lower() for t in toks[-k:]]
        if _initials_match(cand, abbr):
            return cand
    # Single-word abbreviations the site defines itself, e.g. "certificate (CD)".
    if toks and toks[-1].lower()[0] == abbr[0] and len(toks[-1]) >= 5:
        return [toks[-1].lower()]
    return None


@dataclass
class ProductInfo:
    key: str
    key_words: set[str]
    section_words: set[str] = field(default_factory=set)
    page_words: set[str] = field(default_factory=set)      # titles / URL words of product pages carrying it
    attributes: Counter = field(default_factory=Counter)
    qualifier_kinds: set[str] = field(default_factory=set)
    sightings: int = 0
    # How product-like this key is: stated in tables of rate/product pages (1.0), as labelled lines
    # there (0.9), or only in prose / articles (0.6).
    prior: float = 0.6


class Vocabulary:
    def __init__(self, history) -> None:
        self.history = history
        self.products: dict[str, ProductInfo] = {}
        self.aliases: dict[str, set[tuple[str, ...]]] = defaultdict(set)
        self.variants: set[str] = set()
        self.rate_variants: set[str] = set()     # variants seen on rate facts of product / rate pages
        self.idf: dict[str, float] = {}
        self._generation = -1
        self._alias_cache: dict[str, list[tuple[str, tuple[str, ...]]]] = {}   # content hash -> aliases

    def ensure(self) -> "Vocabulary":
        n = len(self.history.store.snapshots)
        if n != self._generation:
            self._build()
            self._generation = n
        return self

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        self.products.clear()
        self.aliases.clear()
        self.variants = set()
        self.rate_variants = set()
        for key, sightings in self.history.by_key.items():
            pkey = key[0]
            info = self.products.get(pkey)
            if info is None:
                info = self.products[pkey] = ProductInfo(pkey, set(pkey.split()))
            info.attributes[key[1]] += len(sightings)
            info.sightings += len(sightings)
            for q, v in key[2]:
                info.qualifier_kinds.add(q)
                if q == "variant":
                    self.variants.update(v)
            rate_key = key[1] in _RATE_ATTRS
            for s in sightings[:20]:
                if s.page_type in _VOCAB_PAGE_TYPES:
                    for c in s.fact.context:
                        if is_product_phrase(c) or len(c.split()) <= 4:
                            info.section_words.update(words(c))
                    if s.page_type in ("product", "promo") and s.fact.extractor == "table":
                        info.page_words.update(words(s.version.title.split("|")[0]))
                        info.page_words.update(words(s.version.key.split("/", 1)[-1].replace("-", " ").replace("/", " ")))
                    if s.page_type in ("rate_sheet", "product"):
                        info.prior = max(info.prior, 1.0 if s.fact.extractor == "table" else
                                         0.9 if s.fact.extractor == "kv" else 0.6)
                    if rate_key:
                        for q, v in key[2]:
                            if q == "variant":
                                self.rate_variants.update(x for x in v if len(x) >= 3)
        self._mine_aliases()
        # idf over product key words + section words (document = product)
        df: Counter = Counter()
        for info in self.products.values():
            for w in info.key_words | info.section_words | info.page_words:
                df[w] += 1
        n = max(1, len(self.products))
        self.idf = {w: math.log(1 + n / c) for w, c in df.items()}

    def _mine_aliases(self) -> None:
        store = self.history.store
        for versions in store.versions.values():
            v = versions[-1]
            if v.info is not None and v.info.page_type == "listing":
                continue
            found = self._alias_cache.get(v.content_hash)
            if found is None:
                found = self._aliases_in(v.markdown or "")
                self._alias_cache[v.content_hash] = found
            for abbr, exp in found:
                self.aliases[abbr].add(exp)

    @staticmethod
    def _aliases_in(text: str) -> list[tuple[str, tuple[str, ...]]]:
        out = []
        if "(" not in text:
            return out
        for m in _ABBR_PAREN.finditer(text):
            exp = _expansion(text[max(0, m.start() - 90): m.start()], m.group(1))
            if exp:
                out.append((m.group(1).lower(), tuple(_singular(w) for w in _norm(" ".join(exp)).split())))
        for m in _ABBR_BEFORE.finditer(text):
            inner = m.group(2)
            toks = [t.lower() for t in re.findall(r"[A-Za-z][\w'’&-]*", inner)]
            if toks and _initials_match(toks, m.group(1).lower()):
                out.append((m.group(1).lower(), tuple(_singular(w) for w in _norm(inner).split())))
        return out

    # ------------------------------------------------------------------ use
    def expand(self, tokens: list[str]) -> list[str]:
        """Question tokens plus the site-defined expansions of any abbreviation among them."""
        out = list(tokens)
        for t in tokens:
            for key in (t, t.rstrip("s")):
                for exp in self.aliases.get(key, ()):
                    out.extend(exp)
        return out

    def variant_words(self) -> set[str]:
        """Words used in product variants anywhere on the site ("adjustable", "jumbo"), 4+ letters so
        fragments of table headings ("up", "or") don't count."""
        return {w for v in self.variants for w in v.split() if len(w) >= 4 and w.isalpha()}

    def product_display(self, pkey: str) -> str:
        return pkey
