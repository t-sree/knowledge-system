"""Page Classifier (component B). Rules only.

For every page version it decides:
  * page_type   what kind of page this is, from the URL structure of the site, refined by content
  * dates       publish date (article header), "rates effective" dates and explicit end dates,
                each with the quote it came from and the block it sits in
  * expired     whether the page itself says the offer has ended

Page types, roughly from most to least authoritative for rates and fees (the resolver decides
the actual ranking per attribute):
  rate_sheet, product, info, promo, calculator, newsroom, article, document, legal, listing
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from knowledge.ingest.store import Block, normalize_url

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}
_MONTH_RE = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_DATE_RE = re.compile(
    r"(?P<mdy>\b\d{1,2}/\d{1,2}/\d{2}(?:\d{2})?\b)"
    r"|(?P<iso>\b\d{4}-\d{2}-\d{2}\b)"
    r"|(?P<long>\b" + _MONTH_RE + r"\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b)",
    re.I,
)
_HEADER_DATE = re.compile(r"^\s*" + _MONTH_RE + r"\.?\s+\d{1,2},\s+\d{4}\s*$", re.I)
# Byline variants: "Written by [Name](...) | Mar 3, 2025 7:00:02 AM", "Posted on ...", "Published ..."
_BYLINE = re.compile(r"^\s*(?:written by|posted|published)\b", re.I)
_HEADER_LINES = 25

_EFFECTIVE_KW = re.compile(r"\beffective\b", re.I)
_EXPIRY_KW = re.compile(
    r"\b(?:offer ends|offer ended|ends on|ending on|expires on|expires|expired on|valid through|"
    r"valid until|available through|available until|good through|through|until|ends)\b",
    re.I,
)
_EXPIRED_NOTICE = re.compile(
    r"\b(?:(?:offer|promotion|special) has (?:ended|expired)|(?:offer|promotion) (?:is )?(?:now )?(?:closed|over))\b",
    re.I,
)
_TABLE_RATE = re.compile(r"^\s*\|.*\d\s*%", re.M)

# How far after a keyword a date may start to still belong to it.
_EFFECTIVE_WINDOW = 40
_EXPIRY_WINDOW = 25


def parse_date(text: str) -> Optional[str]:
    """First date in `text` as ISO yyyy-mm-dd, or None."""
    m = _DATE_RE.search(text or "")
    return _to_iso(m) if m else None


def _to_iso(m: re.Match) -> Optional[str]:
    try:
        if m.group("iso"):
            y, mo, d = (int(x) for x in m.group("iso").split("-"))
        elif m.group("mdy"):
            mo, d, y = (int(x) for x in m.group("mdy").split("/"))
            if y < 100:
                y += 2000
        else:
            parts = re.findall(r"[a-z]+|\d+", m.group("long").lower())
            name, d, y = parts[0], int(parts[1]), int(parts[-1])
            mo = next(v for k, v in _MONTHS.items() if k.startswith(name[:3]))
        if not (1 <= mo <= 12 and 1 <= d <= 31 and 1990 <= y <= 2100):
            return None
        return f"{y:04d}-{mo:02d}-{d:02d}"
    except (ValueError, StopIteration):
        return None


@dataclass(frozen=True)
class DateMark:
    date: str            # ISO date
    kind: str            # "published" | "effective" | "expires"
    quote: str           # the sentence fragment it came from
    block_index: int     # block of the page it sits in (-1 = page header)


@dataclass
class PageInfo:
    page_type: str
    published: Optional[DateMark] = None
    effective: list[DateMark] = field(default_factory=list)
    expires: list[DateMark] = field(default_factory=list)
    expired_notice: Optional[str] = None     # quote when the page says the offer has ended
    has_rate_table: bool = False

    @property
    def content_date(self) -> Optional[str]:
        """Newest stated effective or publish date, used as the page's content date."""
        dates = [d.date for d in self.effective]
        if self.published:
            dates.append(self.published.date)
        return max(dates) if dates else None

    @property
    def undated(self) -> bool:
        return self.content_date is None


# ---------------- page type from URL structure ----------------
# Only general web and banking conventions; no section or product names of one particular site.
# A page the rules don't recognise becomes "info", and an info page with a rate table becomes
# "product" (see classify()), so unknown layouts still degrade sensibly.
_DOC_EXT = (".pdf", ".xls", ".xlsx", ".csv", ".doc", ".docx")
_DOC_DIRS = {"hubfs", "uploads", "files", "documents", "downloads", "downloadables", "wp-content"}  # CMS file stores
_NEWS_WORDS = {"news", "newsroom", "press", "press-releases", "media-center"}
_ARTICLE_WORDS = {"learn", "blog", "blogs", "articles", "resources", "insights", "education", "guides"}
_LEGAL_WORDS = {"privacy", "disclosure", "disclosures", "terms", "legal"}
_PROMO_WORDS = ("offer", "promo", "challenge", "giveaway", "sweepstakes", "switch-and-earn")
# Banking sections and products, matched against the first or last word of a top-level section name
# ("personal-loans", "online-and-mobile-banking"); a word inside it ("kccc-membership-grant") doesn't count.
_PRODUCT_WORDS = {"personal", "business", "banking", "membership", "protection", "insurance", "retirement",
                  "mobile", "savings", "checking", "loan", "loans", "mortgage", "mortgages", "account",
                  "accounts", "card", "cards", "certificate", "certificates", "investments", "lending"}
_SECTION_WORDS = _NEWS_WORDS | _ARTICLE_WORDS
_DATED_PATH = re.compile(r"/(?:19|20)\d{2}/\d{2}/")


def _tokens(segment: str) -> list[str]:
    return [t for t in re.split(r"[-_.]", segment) if t]


def type_from_url(url: str) -> str:
    key = normalize_url(url)
    path = key[key.find("/"):] if "/" in key else ""
    segs = [s for s in path.split("/") if s]
    last = segs[-1] if segs else ""
    inner = segs[:-1]                        # the folders above the page itself
    if path.endswith(_DOC_EXT) or any(s in _DOC_DIRS for s in inner):
        return "document"
    # Listings: tag/author/category/paging pages and section front pages ("/learn/blog", "/newsroom",
    # "/resources/personal-resources", "/business-calculators").
    if re.search(r"/(tag|author|category)/|/page/\d+$", path):
        return "listing"
    last_toks = _tokens(last)
    # A section front page has a short name ("personal-resources"); a story title ending in "insights" doesn't.
    if last in _SECTION_WORDS or (len(segs) <= 2 and 0 < len(last_toks) <= 3 and last_toks[-1] in _SECTION_WORDS):
        return "listing"
    if last and _tokens(last)[-1] == "calculators":
        return "listing"
    if any("calculator" in s for s in inner):
        return "calculator"
    if any(s in _NEWS_WORDS for s in inner):
        return "newsroom"
    if any(s in _ARTICLE_WORDS for s in inner) or _DATED_PATH.search(path):
        return "article"
    if any(t in _LEGAL_WORDS for s in segs for t in _tokens(s)):
        return "legal"
    if last.endswith("-lp") or any(w in last for w in _PROMO_WORDS):
        return "promo"
    if any(seg == "rates" or seg.endswith("-rates") for seg in segs):
        return "rate_sheet"
    if segs:
        toks = _tokens(segs[0])
        if toks and (toks[0] in _PRODUCT_WORDS or toks[-1] in _PRODUCT_WORDS):
            return "product"
    return "info"


# ---------------- dates from content ----------------
def _marks_after_keyword(block: Block, kw: re.Pattern, window: int, kind: str) -> list[DateMark]:
    out = []
    text = block.text
    for m in kw.finditer(text):
        tail = text[m.end(): m.end() + window + 30]
        d = _DATE_RE.search(tail)
        if not d or d.start() > window:
            continue
        iso = _to_iso(d)
        if iso:
            start = max(0, m.start() - 60)
            quote = " ".join(text[start: m.end() + d.end()].split())
            out.append(DateMark(iso, kind, quote, block.index))
    return out


def _published(markdown: str) -> Optional[DateMark]:
    for line in (markdown or "").splitlines()[:_HEADER_LINES]:
        if _HEADER_DATE.match(line) or _BYLINE.match(line):
            iso = parse_date(line)
            if iso:
                return DateMark(iso, "published", " ".join(line.split())[:160], -1)
    return None


def classify(url: str, markdown: str, blocks: list[Block]) -> PageInfo:
    page_type = type_from_url(url)
    info = PageInfo(page_type=page_type)
    info.published = _published(markdown)
    for b in blocks:
        info.effective += _marks_after_keyword(b, _EFFECTIVE_KW, _EFFECTIVE_WINDOW, "effective")
        info.expires += _marks_after_keyword(b, _EXPIRY_KW, _EXPIRY_WINDOW, "expires")
        if not info.has_rate_table and _TABLE_RATE.search(b.text):
            info.has_rate_table = True
    # The page-level "this offer has ended" notice usually sits in a heading, so search the raw page.
    n = _EXPIRED_NOTICE.search(markdown or "")
    if n:
        info.expired_notice = " ".join(markdown[max(0, n.start() - 40): n.end() + 40].split())
    # Content refinement: an info page that carries a rate table behaves like a product page.
    if page_type == "info" and info.has_rate_table:
        info.page_type = "product"
    return info


class PageClassifier:
    """Classifies page versions, caching by (page key, content hash) so unchanged pages are free."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], PageInfo] = {}

    def classify_version(self, version) -> PageInfo:
        ck = (version.key, version.content_hash)
        info = self._cache.get(ck)
        if info is None:
            info = classify(version.url, version.markdown, version.blocks)
            self._cache[ck] = info
        version.info = info
        return info
