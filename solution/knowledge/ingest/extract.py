"""Fact Extractor (component C). Rules only.

Turns one page version into a list of facts. A fact is "product + attribute + qualifiers = value",
together with the exact text it came from. Three extractors, from most to least structured:

  table   markdown tables, in two orientations:
            * rate tables: one row per product/tier, one column per attribute
            * comparison tables: one row per attribute, one column per product/variant
  kv      labelled lines such as "**Routing Number:** 273974581" or "6-month certificate APY: 2.95%"
  prose   rates written in sentences or headings, such as "a competitive 2.25% APY"

Product names on this kind of site are often plain short lines above a table rather than
markdown headings, so every block also gets a "context": its heading path plus the nearest
title-like lines above it. A row label that does not name a product itself inherits the
product from that context.

Every value is checked against the source text before it is kept.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional
from urllib.parse import urlsplit

from knowledge.ingest.classify import DateMark
from knowledge.ingest.store import Block
from knowledge.site import register_stopwords
from knowledge.text import TERM_RE, normalize, singular

# ---------------------------------------------------------------- vocabulary (generic banking terms)
# Attribute phrases, longest first so "annual percentage rate" wins over "rate".
_ATTRIBUTE_PHRASES = [
    ("annual percentage yield", "apy"), ("apy", "apy"),
    ("annual percentage rate", "apr"), ("apr", "apr"),
    ("dividend rate", "dividend_rate"), ("interest rate", "rate"),
    ("minimum to open", "min_open"), ("min to open", "min_open"),
    ("minimum to earn", "min_earn"), ("min to earn", "min_earn"),
    ("minimum balance", "min_balance"), ("min balance", "min_balance"),
    ("maximum balance", "max_balance"), ("max balance", "max_balance"),
    ("payment example", "payment_example"), ("monthly payment", "monthly_payment"),
    ("routing number", "routing_number"), ("phone", "phone"), ("address", "address"),
    ("points", "points"), ("fees", "fee"), ("fee", "fee"), ("rates", "rate"), ("rate", "rate"),
]
# Column headers that carry qualifiers or the row label rather than an attribute.
_QUALIFIER_HEADERS = [
    ("balance tier", "balance_tier"), ("loan amount", "loan_amount"), ("model year", "model_year"),
    ("term length", "term"), ("term", "term"), ("balance", "balance_tier"),
]
_LABEL_HEADERS = ("account type", "account", "certificate", "product", "program", "loan program", "loan type",
                  "card type", "ira type", "type", "category", "plan", "description", "name")

# Words that describe a variant of a product. They become qualifiers, never part of the product key.
_VARIANT_PHRASES = [
    "balance transfer", "cash advance", "first time", "mini max", "cash out",
    "jumbo", "special", "fixed", "variable", "adjustable", "standard", "regular", "introductory", "intro",
    "promotional", "purchase", "refinance", "refi", "roth", "traditional", "conventional", "unsecured",
    "secured", "new", "used", "student", "youth", "senior",
]
_VARIANT_CANON = {"intro": "introductory", "refi": "refinance"}
# Only "business" splits facts: "personal" is part of product names ("personal loan") and is the default.
_ACCOUNT_TYPES = ("business",)

# Nouns that make a phrase name a product (used to decide label vs context).
_PRODUCT_NOUNS = ("certificate", "cd", "savings", "saving", "checking", "loan", "mortgage", "visa", "card",
                  "money market", "ira", "heloc", "hsa", "account", "club", "line of credit", "equity",
                  "share", "mastercard", "credit")

_STOP = {"the", "a", "an", "of", "for", "to", "our", "your", "with", "on", "in", "at", "is", "as",
         "low", "as low as", "and", "&"}

# ---------------------------------------------------------------- value patterns
_FOOTNOTE = re.compile(r"(%|\bAPR|\bAPY|\))(\d{1,2})(?=\s|$|\||\*|,|\))")
_TRAILING_FOOTNOTE = re.compile(r"(%|\bAPR|\bAPY)\s+\d$")    # a footnote digit after a space, at the very end
_MD_NOISE = re.compile(r"\\([*_#|])")
_NUM = re.compile(r"(?P<cur>\$)?\s?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?P<pct>\s?%)?")
_MODEL_YEAR = re.compile(r"\b((?:19|20)\d{2})(?:\s*[-–]\s*(\d{2,4}))?\s*model\s*years?(\s*\(and older\)|\s*and older)?", re.I)
# Label-first form: "Model Year 2026-23", "Model Years 2019 and older".
_MODEL_YEAR_FIRST = re.compile(r"\bmodel\s*years?\s*((?:19|20)\d{2})(?:\s*[-–]\s*(\d{2,4}))?(\s*\(and older\)|\s*and older)?", re.I)
# Column headers that split one attribute into an introductory and an after-promotion value.
_HEADER_AFTER_PROMO = re.compile(r"\b(?:after|following)\b.*\b(?:promo|promotional|intro|introductory)\b", re.I)
_HEADER_INTRO = re.compile(r"\b(?:promo|promotional|intro|introductory)\b|\b(?:1st|first)\s+\d+\s+months?\b", re.I)
_MONEY_RANGE = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*(?:[-–]|to)\s*\$?\s?([\d,]+(?:\.\d+)?)")   # "$10,000-24,999" too
_MONEY_UP = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*(?:\+|or more|and up|and above|or greater)", re.I)
_MONEY_DOWN = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*(?:or less|and under|or below)", re.I)
_TABLE_LINE = re.compile(r"^\s*\|")
_SEP_LINE = re.compile(r"^\s*\|?\s*:?-{2,}")
_KV = re.compile(r"^\s*(?:[-*+]\s+)?\**(?P<label>[A-Za-z0-9][^:|\[\]\n]{1,70}?)\**\s*:\s*\**\s*(?P<value>[^\n]{1,160})$")
_PCT_IN_TEXT = re.compile(r"(?<![\w.])(\d{1,2}(?:\.\d{1,3})?)\s?%\s*(APY|APR)?\d?", re.I)
_CLAUSE_BREAK = re.compile(r"[.!?;:(\n]|—|–\s")
_LINK = re.compile(r"\]\([^)]*\)|https?://\S+")

INF = float("inf")


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def clean_cell(text: str) -> str:
    """Cell text without footnote markers, markdown escapes, <br> and link targets."""
    t = (text or "").replace("<br>", " ").replace("&amp;", "&")
    t = _MD_NOISE.sub(r"\1", t)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = _FOOTNOTE.sub(r"\1", t)
    t = t.replace("**", "").replace("*", "")
    t = " ".join(t.split()).strip()
    return _TRAILING_FOOTNOTE.sub(r"\1", t)          # "15.50% 1" -> "15.50%"


def parse_numbers(text: str) -> tuple[tuple[float, str], ...]:
    """Numbers in a value with their unit: '%' or '$' or '' (bare)."""
    out = []
    for m in _NUM.finditer(text or ""):
        unit = "%" if m.group("pct") else "$" if m.group("cur") else ""
        try:
            out.append((_to_float(m.group("num")), unit))
        except ValueError:
            continue
    return tuple(out)


# Phrases are matched on whole words of normalised text, so plain padded string checks suffice.
def _has_phrase(norm: str, phrase: str) -> bool:
    return f" {phrase} " in f" {norm} "


def _remove_phrase(norm: str, phrase: str) -> str:
    padded, target = f" {norm} ", f" {phrase} "
    while target in padded:
        padded = padded.replace(target, " ")
    return " ".join(padded.split())


def product_key(name: str) -> str:
    """Normalised product identity: lower case, no stop words, simple singulars."""
    toks = [singular(t) for t in normalize(name).split() if t not in _STOP]
    if len(toks) > 1 and toks[-1] == "account":           # "1st Checking Account" == "1st Checking"
        toks = toks[:-1]
    return " ".join(toks)


@lru_cache(maxsize=65536)
def attribute_of(text: str) -> Optional[str]:
    n = normalize(text)
    for phrase, attr in _ATTRIBUTE_PHRASES:
        if _has_phrase(n, phrase):
            return attr
    return None


def slug(text: str) -> str:
    return "_".join(normalize(text).split()[:6])


@lru_cache(maxsize=65536)
def is_product_phrase(text: str) -> bool:
    n = " ".join(singular(t) for t in normalize(text).split())       # "Auto Loans" names a product too
    return any(_has_phrase(n, singular(p)) for p in _PRODUCT_NOUNS)


# ---------------------------------------------------------------- qualifiers
def parse_term(text: str) -> Optional[int]:
    """Term in months: '17 Month' -> 17, '30-Year' -> 360."""
    m = TERM_RE.search(text or "")
    if not m:
        return None
    n = int(m.group(1))
    return n * 12 if m.group(2).lower().startswith("y") else n


def parse_money_range(text: str) -> Optional[tuple[float, float]]:
    t = text or ""
    m = _MONEY_RANGE.search(t)
    if m:
        return (_to_float(m.group(1)), _to_float(m.group(2)))
    m = _MONEY_UP.search(t)
    if m:
        return (_to_float(m.group(1)), INF)
    m = _MONEY_DOWN.search(t)
    if m:
        return (0.0, _to_float(m.group(1)))
    return None


def parse_model_year(text: str) -> Optional[tuple[int, int]]:
    """Model year range: '2024-27 Model Year' -> (2024, 2027), 'Model Year 2026-23' -> (2023, 2026)."""
    m = _MODEL_YEAR.search(text or "") or _MODEL_YEAR_FIRST.search(text or "")
    if not m:
        return None
    lo = int(m.group(1))
    if m.group(3):                       # "(and older)"
        return (0, lo)
    if m.group(2):
        hi = m.group(2)
        hi = int(hi) if len(hi) == 4 else int(str(lo)[:2] + hi)
        return (min(lo, hi), max(lo, hi))
    return (lo, lo)


@dataclass(frozen=True)
class LabelParse:
    product: str                          # remaining product words (display form), may be ""
    qualifiers: tuple[tuple[str, object], ...]
    attribute: Optional[str]


@lru_cache(maxsize=65536)
def parse_label(text: str, strip_attribute: bool = True) -> LabelParse:
    """Split a label such as '17 Month Fixed Jumbo Certificate Special' into product + qualifiers."""
    raw = clean_cell(text)
    quals: dict[str, object] = {}
    work = raw
    my = parse_model_year(work)
    if my:
        quals["model_year"] = my
        work = _MODEL_YEAR_FIRST.sub(" ", _MODEL_YEAR.sub(" ", work))
    term = parse_term(work)
    if term:
        quals["term"] = term
        work = TERM_RE.sub(" ", work, count=1)
    amt = parse_money_range(work)
    if amt:
        quals["loan_amount"] = amt
        work = _MONEY_RANGE.sub(" ", _MONEY_UP.sub(" ", _MONEY_DOWN.sub(" ", work)))
    n = normalize(work)
    attr = None
    for phrase, a in _ATTRIBUTE_PHRASES:
        if _has_phrase(n, phrase):
            attr = attr or a
            if strip_attribute:
                n = _remove_phrase(n, phrase)
    variants = []
    for v in _VARIANT_PHRASES:
        if _has_phrase(n, v):
            variants.append(_VARIANT_CANON.get(v, v))
            n = _remove_phrase(n, v)
    for acct in _ACCOUNT_TYPES:
        if _has_phrase(n, acct):
            quals["account_type"] = acct
            n = _remove_phrase(n, acct)
    if variants:
        quals["variant"] = tuple(sorted(set(variants)))
    words = [w for w in n.split() if w not in _STOP and not w.isdigit()]
    return LabelParse(" ".join(words), tuple(sorted(quals.items())), attr)


def _merge_quals(*parts) -> tuple[tuple[str, object], ...]:
    out: dict[str, object] = {}
    for p in parts:
        for k, v in (p or ()):
            if k == "variant" and "variant" in out:
                out[k] = tuple(sorted(set(out[k]) | set(v)))
            elif k not in out:
                out[k] = v
    return tuple(sorted(out.items()))


# ---------------------------------------------------------------- the fact
@dataclass(frozen=True)
class Fact:
    product: str                          # display name, from the page text
    product_key: str                      # normalised identity
    attribute: str
    qualifiers: tuple[tuple[str, object], ...]
    value: str                            # cleaned value text as shown on the page
    numbers: tuple[tuple[float, str], ...]
    extractor: str                        # "table" | "kv" | "prose"
    quote: str                            # the source text the value was read from
    context: tuple[str, ...]              # headings / title lines above the block
    block_index: int
    effective: Optional[str] = None       # "rates effective" date that belongs to this table/block
    expires: Optional[str] = None         # end date stated in the same block
    label: str = ""                       # the row / line label as written on the page

    @property
    def key(self) -> tuple:
        return (self.product_key, self.attribute, self.qualifiers)


# ---------------------------------------------------------------- block context
def _is_title_line(block: Block) -> bool:
    t = block.text.strip()
    bold = re.fullmatch(r"\*\*([^*]+)\*\*:?", t) or re.fullmatch(r"__([^_]+)__:?", t)
    if bold:                                    # a bold-only line works as a sub-heading ("**Drive-up**")
        t = bold.group(1).strip()
    if "\n" in t or len(t) > 70 or t.startswith(("|", "!", "[", "-", "*", ">")) or "](" in t:
        return False
    if t.endswith((".", ",", ":", ";", "?", "!")) or "%" in t or "$" in t:
        return False
    words = t.split()
    return 1 <= len(words) <= 8 and t[0].isalpha()


def block_contexts(blocks: list[Block]) -> list[tuple[str, ...]]:
    """Heading path plus up to two nearest title-like lines above each block (outer -> inner)."""
    out = []
    for i, b in enumerate(blocks):
        found: list[str] = []
        j = i - 1
        while j >= 0 and i - j <= 8 and len(found) < 2:
            pb = blocks[j]
            if pb.heading_path != b.heading_path and found:
                break
            if _is_title_line(pb):
                if not found or clean_cell(pb.text) != found[-1]:
                    found.append(clean_cell(pb.text))
            elif _TABLE_LINE.match(pb.text) and found:
                break
            j -= 1
        ctx = tuple(h for h in b.heading_path) + tuple(reversed(found))
        out.append(ctx)
    return out


# Headings written to the reader ("Put your equity to work", "We've got you covered") are marketing
# lines, not product names.
_ADDRESSED = {"you", "your", "youre", "we", "weve", "our", "us", "my", "i", "lets"}


def _is_marketing(text: str) -> bool:
    return any(w in _ADDRESSED for w in normalize(text).split())


@lru_cache(maxsize=65536)
def context_product(context: tuple[str, ...], title: str) -> tuple[str, tuple]:
    """Innermost context entry that names a product, parsed into (product words, qualifiers)."""
    context = tuple(c for c in context if not _is_marketing(c))
    for c in reversed(context):
        if is_product_phrase(c):
            lp = parse_label(c)
            if lp.product:
                return lp.product, lp.qualifiers
    for c in reversed(context):
        lp = parse_label(c)
        if lp.product:
            return lp.product, lp.qualifiers
    if "://" in (title or "") or (title or "").lower().startswith("www."):
        return "", ()                           # documents are titled by their URL
    lp = parse_label(title or "")
    return lp.product, lp.qualifiers


# ---------------------------------------------------------------- tables
def _split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _classify_header(h: str) -> tuple[str, Optional[str]]:
    """('attr', name) | ('qual', name) | ('label', None) | ('other', None)."""
    n = normalize(clean_cell(h))
    if not n or re.fullmatch(r"col\d*", n):
        return "other", None
    for phrase, q in _QUALIFIER_HEADERS:
        if n == phrase or n.startswith(phrase + " ") and q != "balance_tier":
            return "qual", q
    for phrase, a in _ATTRIBUTE_PHRASES:
        if _has_phrase(n, phrase):
            return "attr", a
    if any(n == p or n.startswith(p + " ") for p in _LABEL_HEADERS):
        return "label", None
    return "other", None


def _qual_value(kind: str, cell: str):
    if kind == "term":
        return parse_term(cell)
    if kind in ("balance_tier", "loan_amount"):
        return parse_money_range(cell)
    if kind == "model_year":
        return parse_model_year(cell) or parse_model_year(cell + " model year")
    return None


def _table_rows(text: str) -> Optional[tuple[list[str], list[list[str]]]]:
    lines = [ln for ln in text.splitlines() if _TABLE_LINE.match(ln)]
    if len(lines) < 3 or not _SEP_LINE.match(lines[1]):
        return None
    header = _split_row(lines[0])
    rows = [_split_row(ln) for ln in lines[2:]]
    rows = [r + [""] * (len(header) - len(r)) for r in rows if any(c.strip() for c in r)]
    return header, rows


# Rows labelled only by a number ("1", "Point 3") are chart data, not attributes.
_NUMBERED_ROW = re.compile(r"^\s*[A-Za-z]*\s*\d+\s*$")
_RATE_ATTRIBUTES = ("apy", "apr", "rate", "dividend_rate")


def _table_facts(block: Block, context, title, acct_default) -> list[Fact]:
    parsed = _table_rows(block.text)
    if not parsed:
        return []
    header, rows = parsed
    # PDF forms converted to markdown carry placeholder headers ("Col2", "Col3"): no usable layout.
    if any(re.fullmatch(r"col\d+", clean_cell(h).lower()) for h in header):
        return []
    kinds = [_classify_header(h) for h in header]
    ctx_prod, ctx_quals = context_product(context, title)
    header_line = " | ".join(clean_cell(h) for h in header)
    facts: list[Fact] = []
    n_attr = sum(k == "attr" for k, _ in kinds)
    n_other = sum(k == "other" for k, _ in kinds[1:])

    if n_attr == 0 and n_other >= 2 and kinds[0][0] in ("other", "label"):
        # Comparison table: rows are attributes, columns are products/variants.
        for r in rows:
            row_label = clean_cell(r[0])
            if not row_label or _NUMBERED_ROW.match(row_label):
                continue
            attr = attribute_of(row_label) or slug(row_label)
            for ci in range(1, len(header)):
                col = clean_cell(header[ci])
                val = clean_cell(r[ci]) if ci < len(r) else ""
                if not col or not val or len(val) > 240 or re.fullmatch(r"col\d*", col.lower()):
                    continue
                lp = parse_label(col)
                if is_product_phrase(col) and lp.product:
                    prod, quals = lp.product, _merge_quals(lp.qualifiers, acct_default)
                else:
                    prod = ctx_prod
                    quals = _merge_quals(lp.qualifiers, ctx_quals, acct_default,
                                         (("variant", (lp.product,)),) if lp.product else ())
                quote = f"{header_line} || {row_label} | {col}: {val}"
                facts.append(_fact(prod, attr, quals, val, "table", quote, context, block.index,
                                   label=f"{row_label} ({col})"))
        return facts

    if n_attr == 0:
        return []
    # Qualifiers written in an attribute column's header ("Dividend Rate for 1st 6 Months").
    header_quals = []
    for h in header:
        hc = clean_cell(h)
        if _HEADER_AFTER_PROMO.search(hc):
            header_quals.append((("variant", ("after promotional",)),))
        elif _HEADER_INTRO.search(hc):
            header_quals.append((("variant", ("introductory",)),))
        else:
            header_quals.append(())
    label_col = next((i for i, (k, _) in enumerate(kinds) if k == "label"), None)
    if label_col is None and kinds[0][0] == "other":
        label_col = 0
    for r in rows:
        cells = [clean_cell(c) for c in r]
        row_quals: dict[str, object] = {}
        for i, (k, q) in enumerate(kinds):
            if k == "qual" and i < len(cells) and cells[i]:
                v = _qual_value(q, cells[i])
                if v is not None:
                    row_quals[q] = v
                # The same cell may carry more qualifiers, e.g. a term cell "30-Year Fixed".
                for qk, qv in parse_label(cells[i]).qualifiers:
                    if qk == "loan_amount" and q in ("balance_tier", "loan_amount"):
                        continue
                    if qk == "variant":
                        row_quals["variant"] = tuple(sorted(set(row_quals.get("variant", ())) | set(qv)))
                    else:
                        row_quals.setdefault(qk, qv)
        prod, quals = ctx_prod, ctx_quals
        label_text = cells[label_col] if label_col is not None and label_col < len(cells) else ""
        if not label_text:
            # No label column: a qualifier cell may carry the full product name
            # ("Term" = "10 Month Fixed Jumbo Certificate Special").
            label_text = next((cells[i] for i, (k, _) in enumerate(kinds)
                               if k == "qual" and i < len(cells) and is_product_phrase(cells[i])), "")
        if label_text:
            lp = parse_label(label_text)
            if lp.product and is_product_phrase(label_text):
                prod, quals = lp.product, _merge_quals(lp.qualifiers, _drop(ctx_quals, "variant"))
            else:
                extra = (("variant", tuple(lp.product.split())),) if lp.product else ()
                quals = _merge_quals(lp.qualifiers, extra, ctx_quals)
        quals = _merge_quals(tuple(row_quals.items()), quals, acct_default)
        row_line = " | ".join(cells)
        # Human-readable row label: the label cell plus any qualifier cells not already in it.
        qual_cells = [cells[i] for i, (k, _) in enumerate(kinds) if k == "qual" and i < len(cells)
                      and cells[i] and cells[i] != label_text]
        row_label_text = " · ".join([c for c in [label_text] + qual_cells if c])
        for i, (k, a) in enumerate(kinds):
            if k == "attr" and i < len(cells) and cells[i]:
                cell_attr = attribute_of(cells[i])            # a unit written in the cell beats the header
                attr = cell_attr if cell_attr in ("apr", "apy") else a
                quote = f"{header_line} || {row_line}"
                facts.append(_fact(prod, attr, _merge_quals(header_quals[i], quals), cells[i], "table",
                                   quote, context, block.index, label=row_label_text))
    return facts


def _drop(quals, name):
    return tuple((k, v) for k, v in quals if k != name)


def _fact(product, attr, quals, value, extractor, quote, context, block_index, label="") -> Fact:
    return Fact(product=product, product_key=product_key(product), attribute=attr,
                qualifiers=tuple(sorted(quals)), value=value, numbers=parse_numbers(value),
                extractor=extractor, quote=quote[:400], context=context, block_index=block_index,
                label=label)


# ---------------------------------------------------------------- key-value lines and prose
_SITE_ATTRIBUTES = {"routing_number"}
_FOOTNOTE_LABEL = re.compile(r"^\d+[A-Z]")       # "1Term Share Certificate Rates: ..." is a footnote, not a label


def _kv_facts(block: Block, context, title, acct_default) -> list[Fact]:
    facts = []
    for line in block.text.splitlines():
        if _TABLE_LINE.match(line):
            continue
        m = _KV.match(line)
        if not m:
            continue
        label = clean_cell(m.group("label"))
        value = clean_cell(_LINK.sub(")", m.group("value")).replace("[", "")).rstrip(")").strip()
        if (not value or len(value) > 80 or not re.search(r"\d", value) or len(label.split()) > 6
                or len(re.findall(r"[A-Za-z]", label)) < 2 or _FOOTNOTE_LABEL.match(label) or "http" in label.lower()):
            continue
        lp = parse_label(label)
        attr = lp.attribute
        ctx_prod, ctx_quals = context_product(context, title)
        if attr in _SITE_ATTRIBUTES and not (lp.product and is_product_phrase(label)):
            prod, quals = "", acct_default                  # identifiers of the institution itself
            facts.append(_fact(prod, attr, quals, value, "kv", " ".join(line.split()), context, block.index,
                               label=label))
            continue
        if attr is None:
            # The label itself names the attribute ("Phone", "Hours"). Values with a rate in them are
            # sentences and are left to the prose extractor.
            if "%" in value or len(value) > 40:
                continue
            attr = slug(label)
            prod, quals = ctx_prod, ctx_quals
            # The nearest sub-heading tells apart repeated labels ("Lobby" / "Drive-up" hours).
            inner = next((c for c in reversed(context) if not is_product_phrase(c) and not _is_marketing(c)), "")
            section = product_key(inner)
            if section and section != product_key(prod):
                quals = _merge_quals(quals, (("variant", (section,)),))
        elif lp.product and is_product_phrase(label):
            prod, quals = lp.product, lp.qualifiers
        elif attr == "fee" and lp.product:
            prod, quals = lp.product, lp.qualifiers          # "Lost debit card replacement fee"
        else:
            extra = (("variant", tuple(lp.product.split())),) if lp.product else ()
            prod, quals = ctx_prod, _merge_quals(lp.qualifiers, extra, ctx_quals)
        quals = _merge_quals(quals, acct_default)
        facts.append(_fact(prod, attr, quals, value, "kv", " ".join(line.split()), context, block.index,
                           label=label))
    return facts


# "... 12.50% to 19.50% APR": the second number is the upper end of the same value.
_PCT_RANGE_TAIL = re.compile(r"\s*(?:-|–|—|to)\s*(\d{1,2}(?:\.\d{1,3})?)\s?%\s*(APY|APR)?", re.I)
# Limit words just before the number, allowing a few words in between ("floor rate of 4.00%").
_FLOOR_WORDS = re.compile(r"\b(?:less than|minimum|floor|at least|lower than)\b(?:\W+\w+){0,4}\W*$", re.I)
_CAP_WORDS = re.compile(r"\b(?:exceed|maximum|cap|ceiling|more than|higher than|greater than|up to)\b(?:\W+\w+){0,4}\W*$", re.I)
_SENTENCE_END = re.compile(r"[.!?;](?:\s|$)")


def _prose_facts(block: Block, context, title, acct_default, skip_lines: set[str]) -> list[Fact]:
    facts = []
    for line in block.text.splitlines():
        if _TABLE_LINE.match(line) or line in skip_lines or "%" not in line:
            continue
        plain = clean_cell(_LINK.sub(" ", line))
        range_end = -1
        for m in _PCT_IN_TEXT.finditer(plain):
            if m.start() < range_end:
                continue                       # upper end of a range already taken with its lower end
            # Range forms: "12.50%–19.50% APR", "12.50% APR to 19.50% APR".
            hi = _PCT_RANGE_TAIL.match(plain, m.end()) or _PCT_RANGE_TAIL.match(plain, plain.index("%", m.start()) + 1)
            before = plain[: m.start()]
            cut = max((b.end() for b in _CLAUSE_BREAK.finditer(before)), default=0)
            label = " ".join(before[cut:].split()[-10:])
            rest = plain[m.end(): m.end() + 120]
            e = _SENTENCE_END.search(rest)
            rest = rest[: e.start()] if e else rest
            after = rest[:40].lower()
            attr = (m.group(2) or "").lower() or None
            if attr is None:
                if re.match(r"\s*(apy|annual percentage yield)", after):
                    attr = "apy"
                elif re.match(r"\s*(apr|annual percentage rate)", after):
                    attr = "apr"
                else:
                    attr = attribute_of(label)
            if attr not in ("apy", "apr", "rate", "dividend_rate"):
                continue
            lp = parse_label(label)
            ctx_prod, ctx_quals = context_product(context, title)
            if lp.product and is_product_phrase(label) and len(lp.product.split()) <= 5:
                prod, quals = lp.product, (lp.qualifiers or ctx_quals)
            else:
                prod, quals = ctx_prod, _merge_quals(lp.qualifiers, ctx_quals)
            # Qualifiers stated after the number in the same sentence: "... with a 60-month term".
            tail_quals = []
            if parse_term(rest):
                tail_quals.append(("term", parse_term(rest)))
            if parse_model_year(rest):
                tail_quals.append(("model_year", parse_model_year(rest)))
            # Stated limits of a variable rate are their own facts, not the rate itself.
            if _FLOOR_WORDS.search(label):
                tail_quals.append(("variant", ("floor",)))
            elif _CAP_WORDS.search(label):
                tail_quals.append(("variant", ("cap",)))
            quals = _merge_quals(quals, tuple(tail_quals), acct_default)
            value = f"{m.group(1)}%"
            if hi:
                value += f" – {hi.group(1)}%"
                range_end = hi.end()
            unit = m.group(2) or (hi.group(2) if hi else None)
            if unit:
                value += f" {unit.upper()}"
            quote = plain[max(0, m.start() - 120): max(m.end(), range_end) + 60]
            facts.append(_fact(prod, attr, quals, value, "prose", quote, context, block.index, label=label))
    return facts


# ---------------------------------------------------------------- page level
def _nearest_effective(block_index: int, effective: list[DateMark], table_blocks: set[int]) -> Optional[str]:
    """The 'rates effective' line that belongs to a table: just below it first, else just above."""
    by_block: dict[int, str] = {}
    for m in effective:
        by_block.setdefault(m.block_index, m.date)
    for d in range(0, 4):
        j = block_index + d
        if d and j in table_blocks:
            break
        if j in by_block:
            return by_block[j]
    for d in range(1, 3):
        j = block_index - d
        if j in table_blocks:
            break
        if j in by_block:
            return by_block[j]
    return None


def _check(f: Fact) -> bool:
    """Every number in the value must appear in the quoted source text."""
    src = f.quote.replace(",", "")
    for m in _NUM.finditer(f.value):
        if m.group("num").replace(",", "") not in src:
            return False
    return bool(f.value.strip())


def extract_page(url: str, title: str, blocks: list[Block], info, skip_blocks: set[int] = frozenset()):
    """All facts of one page version, plus the number of values rejected by the source check."""
    path = urlsplit(url if "//" in url else "https://" + url).path.lower()
    acct_default = (("account_type", "business"),) if path.startswith("/business") else ()
    page_type = getattr(info, "page_type", "info")
    contexts = block_contexts(blocks)
    table_blocks = {b.index for b in blocks if _TABLE_LINE.match(b.text)}
    effective = list(getattr(info, "effective", []) or [])
    expires_by_block: dict[int, str] = {}
    for m in getattr(info, "expires", []) or []:
        expires_by_block.setdefault(m.block_index, m.date)
    facts: list[Fact] = []
    rejected = 0
    for b, ctx in zip(blocks, contexts):
        if b.index in skip_blocks:
            continue
        found: list[Fact] = []
        if b.index in table_blocks:
            found += _table_facts(b, ctx, title, acct_default)
        kv = _kv_facts(b, ctx, title, acct_default)
        found += kv
        kv_lines = {ln for ln in b.text.splitlines() if _KV.match(ln) and any(" ".join(ln.split()) == f.quote for f in kv)}
        found += _prose_facts(b, ctx, title, acct_default, kv_lines)
        if not found:
            continue
        eff = _nearest_effective(b.index, effective, table_blocks)
        exp = expires_by_block.get(b.index)
        for f in found:
            # Calculator pages show example results; only the rates they quote are site facts.
            if page_type == "calculator" and f.attribute not in _RATE_ATTRIBUTES:
                continue
            # Converted spreadsheets and PDF forms: only values that carry a number are usable.
            if page_type == "document" and not re.search(r"\d", f.value):
                continue
            if not _check(f):
                rejected += 1
                continue
            if eff or exp:
                f = Fact(**{**f.__dict__, "effective": eff, "expires": exp})
            facts.append(f)
    return facts, rejected


class FactExtractor:
    """Extracts facts per page version, cached by (page key, content hash, boilerplate set)."""

    def __init__(self) -> None:
        self._cache: dict[tuple, tuple[list[Fact], int]] = {}
        self.rejected = 0

    def extract_version(self, version, skip_blocks: frozenset[int]) -> list[Fact]:
        ck = (version.key, version.content_hash, skip_blocks)
        hit = self._cache.get(ck)
        if hit is None:
            hit = extract_page(version.url, version.title, version.blocks, version.info, skip_blocks)
            self._cache[ck] = hit
            self.rejected += hit[1]
        return hit[0]


# The site's own name is a stop word in product names (site.py); cached parses must then be redone.
register_stopwords(_STOP, cached=(parse_label, context_product))
