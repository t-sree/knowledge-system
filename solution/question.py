"""Question Parser (component H), rules only.

Turns a member question into a structured query: intent, attribute, qualifiers, date words, and
the content words used to match products against the closed vocabulary (vocab.py). Nothing here
is keyed to particular questions: the word lists are general English / banking grammar, and the
product and variant vocabulary comes from the snapshots.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from typing import Optional

from extract import _TERM, _norm, _singular, attribute_of, parse_term

# ---------------------------------------------------------------- general word lists
_STOP = set("""
a an the of for to on in at by with from and or but is are was were be been being do does did can could
would should will shall may might must i me my we our us you your it its this that these those there here
what which who whom whose when where why how much many any some all every each
please tell give show get find know need want like let about tell say repeat again also just
dupaco dupacos credit union cu currently current now today right offer offered offering have has had
available option options type types kind kinds still yet as than then so if not no yes between among
their them they his her he she him its im ive id youre whats whats wheres hows thats theres
use using used see seeing look looking provide find check tell know get got getting go going want wanted
borrow borrowing borrowed take taking put putting open opening save saved how
would like love need needs help could please thanks thank also just only really very more most less
account accounts product products service services information info detail details
""".split())
_ATTRIBUTE_WORDS = set("""
rate rates interest apy apr yield yields annual percentage dividend dividends fee fees cost costs charge
charges price pay pays paying earn earns earning return minimum min maximum max number routing phone
address hours hour open opening balance link url website webpage page
""".split())
_QUALIFIER_WORDS = set("""
month months mo year years yr term terms model new used
highest lowest best top most least cheapest tier tiers level levels
over under each carry carries buy buying bought plan planning thinking looking would like
change changed changes since go went up down increase increased decrease decreased raise raised rise rose drop dropped before after
""".split())
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                 "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "eighteen": 18,
                 "twenty": 20, "thirty": 30, "forty": 40, "sixty": 60}

_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
_MONTH_YEAR = re.compile(r"\b(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?\s+(\d{1,2},?\s+)?((?:19|20)\d{2})\b", re.I)
_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")
_MONEY = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*(k|thousand)?", re.I)
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s?%")

_URL_INTENT = re.compile(r"\b(link|url|website|web ?page|which page|what page)\b", re.I)
_LIST_INTENT = re.compile(r"\b(what|which)\b[^?]*\b(offer|have|available|options|types|kinds)\b|\b(all|list|every)\b", re.I)
_HIGH = re.compile(r"\b(highest|best|top|maximum|max|most)\b", re.I)
_LOW = re.compile(r"\b(lowest|cheapest|least|smallest)\b", re.I)
_YESNO = re.compile(r"^\s*(do|does|is|are|can|will|did|has|have)\b", re.I)
_PAST = re.compile(r"\b(was|were|used to|previously|back in|as of)\b", re.I)
_CHANGE = re.compile(r"\b(change|changed|changes|increase|increased|decrease|decreased|go up|went up|go down|went down|"
                     r"raise|raised|lower|lowered|drop|dropped|rise|rose)\b", re.I)
# Hours: "hours", "what time ...", or opening / closing of lobbies / drive-ups / branches in general.
_HOURS = re.compile(r"\bhours?\b|\bwhat time\b|\b(?:lobby|lobbies|drive[- ]?(?:up|thru|through)s?)\b[^?]*\b(?:open|close)", re.I)
_WHERE = re.compile(r"\bwhere is\b|\blocated\b|\bstreet address\b", re.I)
# Is something still offered / available / open, or has it been discontinued?
_AVAILABLE = re.compile(r"\bstill\b[^?]*\b(?:offer|offered|available|open|exist|sold|valid|running)|\bno longer\b|"
                        r"\bdiscontinued\b|\bstill (?:have|has|offer)\b", re.I)
_VEHICLE = re.compile(r"\b(model|vehicle|car|truck|suv|motorcycle|boat|rv|atv)\b", re.I)
_BUSINESS = re.compile(r"\bbusiness\b", re.I)


@dataclass
class Query:
    text: str
    tokens: list[str]                          # normalised, singular
    content: list[str]                         # words left for product matching
    attribute: Optional[str] = None            # specific attribute, or "rate" = generic rate
    term: Optional[int] = None                 # months
    model_year: Optional[int] = None
    amounts: list[float] = field(default_factory=list)
    percents: list[float] = field(default_factory=list)
    variants: set[str] = field(default_factory=set)
    business: bool = False
    ref_date: Optional[str] = None             # date named in the question (end of that month)
    intents: set[str] = field(default_factory=set)   # url, list, high, low, yesno, past, change, available
    terms: list[int] = field(default_factory=list)   # several terms asked ("12 month or 24 month")
    products: Optional[list[str]] = None       # product keys chosen by the LLM step (skips name matching)


def _end_of_month(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"


def _numbers_to_digits(text: str) -> str:
    def sub(m):
        return str(_NUMBER_WORDS[m.group(1).lower()]) + m.group(2)
    return re.sub(r"\b(" + "|".join(_NUMBER_WORDS) + r")([\s-]+(?:month|year|yr|mo)s?\b)", sub, text, flags=re.I)


def parse(question: str, vocab) -> Query:
    text = question or ""
    q = Query(text=text, tokens=[], content=[])
    t = _numbers_to_digits(text)

    # dates named in the question
    date_spans = []
    for m in _MONTH_YEAR.finditer(t):
        month = _MONTHS[m.group(1).lower().rstrip(".")]
        year = int(m.group(3))
        if m.group(2):
            day = int(re.sub(r"\D", "", m.group(2)))
            q.ref_date = f"{year:04d}-{month:02d}-{day:02d}"
        else:
            q.ref_date = _end_of_month(year, month)
        date_spans.append(m.span())

    # qualifiers
    q.term = parse_term(t)
    # every term named ("12 month or 24 month?"), in order
    for m in _TERM.finditer(t):
        months = int(m.group(1)) * (12 if m.group(2).lower().startswith("y") else 1)
        if months not in q.terms:
            q.terms.append(months)
    for m in _YEAR.finditer(t):
        if any(a <= m.start() < b for a, b in date_spans):
            continue
        if _VEHICLE.search(t):
            q.model_year = int(m.group(1))
            break
    for m in _MONEY.finditer(t):
        v = float(m.group(1).replace(",", ""))
        if m.group(2):
            v *= 1000
        q.amounts.append(v)
    q.percents = [float(x) for x in _PERCENT.findall(t)]
    q.business = bool(_BUSINESS.search(t))
    norm = _norm(t)
    sing = " ".join(_singular(w) for w in norm.split())
    for v in vocab.rate_variants:
        if f" {v} " in f" {norm} " or f" {v} " in f" {sing} ":
            q.variants.add(v)
    if "intro" in norm.split():
        q.variants.add("introductory")

    # attribute
    attr = attribute_of(" ".join(_singular(w) for w in _norm(t).split()))   # "APRs" -> "apr"
    if attr is None:
        low = norm.split()
        if "dividend" in low:
            attr = "dividend_rate"
        elif "yield" in low:
            attr = "apy"
        # "pay"/"earn" ask for a rate, but a capitalised "Pay" is a name ("Apple Pay")
        elif re.search(r"\b(?:pay|pays|paying|earn|earns|interest)\b", t):
            attr = "rate"
        elif _HOURS.search(t):
            attr = "hours"
        elif _WHERE.search(t):
            attr = "address"
    q.attribute = attr

    # intents
    if _URL_INTENT.search(t):
        q.intents.add("url")
    if _LIST_INTENT.search(t) or re.search(r"\b(cards|loans|accounts|certificates|cds|rates|mortgages)\b", t, re.I):
        q.intents.add("list")
    if _HIGH.search(t):
        q.intents.add("high")
    if _LOW.search(t):
        q.intents.add("low")
    if _YESNO.search(t) and q.percents:
        q.intents.add("yesno")
    if _PAST.search(t):
        q.intents.add("past")
    if _CHANGE.search(t):
        q.intents.add("change")
    if _AVAILABLE.search(t):
        q.intents.add("available")

    # content words for product matching (abbreviations expanded with the site's own definitions)
    # words of dates named in the question are not content
    date_words = set()
    for a, b in date_spans:
        date_words.update(_norm(t[a:b]).split())
    raw = norm.split()
    q.tokens = [_singular(w) for w in raw]
    variant_words = {w for v in q.variants for w in v.split()} | ({"intro"} if "introductory" in q.variants else set())
    content = []
    for w in raw:                      # stop words are checked before singularising ("does" != "doe")
        if (w in _STOP or w in _ATTRIBUTE_WORDS or w in _QUALIFIER_WORDS or w in variant_words or w in date_words
                or w.isdigit() or len(w) < 2 or w == "business"):
            continue
        sw = _singular(w)
        if sw in _STOP or sw in _ATTRIBUTE_WORDS or sw in _QUALIFIER_WORDS:
            continue
        content.append(sw)
    expanded = vocab.expand(content)
    # an abbreviation the site defines is replaced by its expansion ("cd" -> "certificate")
    q.content = [w for w in expanded if not (w in content and vocab.aliases.get(w) and len(expanded) > len(content))
                 or w not in vocab.aliases]
    return q
