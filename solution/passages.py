"""Passage retrieval (component J, passage path).

Used when a question is not about a rate or fee (hours, addresses, programs, events, how-to).

* Every block of every page version is split into short chunks (a few sentences, a few list lines
  or a few table rows). Identical chunks are stored once and remember every (page, snapshot) they
  appear on, so unchanged pages cost nothing and repeated text is indexed once.
* Search is BM25 over the chunks visible in the selected snapshot view, multiplied by how much of
  the question the chunk covers and by the authority of the best page carrying it (page type,
  present vs carried forward).
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field

from extract import _singular

_TOKEN = re.compile(r"[a-z0-9]+(?:\$[a-z0-9]+)*")       # keeps "home$tart" in one piece
_LINK = re.compile(r"!\[[^\]]*\]\([^)]*\)")              # images
_MDLINK = re.compile(r"\[([^\]]*)\]\(([^)\s]*)[^)]*\)")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9“\"(])|\n+")
_COL_HEADER = re.compile(r"\|\s*col\d+\s*\|", re.I)

CHUNK_WORDS = 70
PAGE_WEIGHT = {"product": 1.0, "info": 1.0, "rate_sheet": 0.95, "newsroom": 0.9, "promo": 0.85,
               "article": 0.8, "legal": 0.55, "calculator": 0.55, "document": 0.6, "listing": 0.4}

STOP = set("""
a an the of for to on in at by with from and or but is are was were be been being do does did can could
would should will shall may might must i me my we our us you your it its this that these those there here
what which who whom whose when where why how much many any some all every each please tell give show
get know need want like let about say repeat again also just dupaco dupacos credit union cu as than then
so if not no yes their them they im ive whats wheres hows thats theres use see look find have has had
going go new
""".split())

STEM_LEN = 6


def stem(t: str) -> str:
    """Light stemming by prefix: flexibility / flexible -> 'flexib', branches / branch -> 'branch'."""
    if t.isdigit():
        return t
    t = _singular(t)
    return t[:STEM_LEN] if len(t) > STEM_LEN else t


def tokens(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN.findall((text or "").lower().replace("’", "").replace("'", ""))]


STOP = {stem(w) for w in STOP}


def display_text(text: str) -> str:
    """Markdown block as plain text; links keep their URL ("Join (https://join.dupaco.com/...)")."""
    t = _LINK.sub(" ", text or "")

    def link(m):
        label, url = m.group(1).strip(), m.group(2).strip()
        if not url or url.startswith("#") or url.startswith("javascript"):
            return label
        if not label or label == url:
            return url
        return f"{label} ({url})"
    t = _MDLINK.sub(link, t)
    t = t.replace("**", "").replace("__", "").replace("\\", "")
    t = re.sub(r"^\s*#+\s*", "", t, flags=re.M)
    return "\n".join(" ".join(ln.split()) for ln in t.splitlines() if ln.strip())


def _recency(content_date, ref: str) -> float:
    """Dated pages: newer is slightly better (x0.85 for three years or older, x1.0 for current).
    Undated pages (product / info pages) are treated as current."""
    if not content_date:
        return 1.0
    try:
        from datetime import date
        age = (date.fromisoformat(ref) - date.fromisoformat(content_date)).days
    except ValueError:
        return 1.0
    return 1.0 - 0.15 * min(max(age, 0), 3 * 365) / (3 * 365)


def _is_label(text: str) -> bool:
    """A short label line such as "Lobby" or "Drive-up hours": up to four words, no digits, no sentence end."""
    t = (text or "").strip().rstrip(":")
    return 0 < len(t) <= 40 and len(t.split()) <= 4 and not re.search(r"[\d.!?|(\[]", t)


_INLINE_LABEL = re.compile(r"^\s*(?:\*\*|__)[^*_\n]{1,40}(?:\*\*|__)")   # "**Drive-up** Monday-Friday: ..."
_RUN_LINE_MAX = 150


def _run_end(blocks, i: int) -> int:
    """End (exclusive) of a label run starting at block i, or i when block i does not start one.
    A run is a label (own line, or bold at the start of a short line) followed by short lines."""
    first = display_text(blocks[i].text)
    if not first or len(first) > _RUN_LINE_MAX:
        return i
    if not (_is_label(first) or _INLINE_LABEL.match(blocks[i].text)):
        return i
    j = i + 1
    while j < len(blocks) and j - i <= 12 and blocks[j].heading_path == blocks[i].heading_path:
        t = display_text(blocks[j].text)
        if not t or len(t) > _RUN_LINE_MAX or _is_label(t) or _INLINE_LABEL.match(blocks[j].text) \
                or t.startswith("|"):
            break
        j += 1
    return j


@dataclass
class Chunk:
    cid: int
    text: str                         # display text
    tf: dict[str, int]
    length: int
    occurrences: list[tuple[str, int, int]] = field(default_factory=list)   # (page key, snapshot idx, block idx)
    run: bool = False                 # a label with the short lines under it (a schedule, a short list)


def _split_block(block_text: str) -> list[str]:
    """Chunks of about CHUNK_WORDS words; table rows and list lines stay whole."""
    text = display_text(block_text)
    if not text:
        return []
    lines = text.splitlines()
    if lines and lines[0].startswith("|"):
        header = lines[0]
        rows = [ln for ln in lines[2:] if ln.startswith("|")] or lines[1:]
        return ["\n".join([header] + rows[i:i + 6]) for i in range(0, len(rows), 6)] or [header]
    units = [u.strip() for u in _SENT.split(text) if u and u.strip()]
    out, buf, n = [], [], 0
    for u in units:
        w = len(u.split())
        if buf and n + w > CHUNK_WORDS:
            out.append(" ".join(buf) if not any(b.startswith(("-", "*")) for b in buf) else "\n".join(buf))
            buf, n = [], 0
        buf.append(u)
        n += w
    if buf:
        out.append(" ".join(buf) if not any(b.startswith(("-", "*")) for b in buf) else "\n".join(buf))
    return out


class PassageIndex:
    def __init__(self, store) -> None:
        self.store = store
        self.chunks: list[Chunk] = []
        self._by_text: dict[str, int] = {}
        self.postings: dict[str, list[int]] = defaultdict(list)
        self.df: dict[str, int] = defaultdict(int)
        self._block_chunks: dict[str, list[int]] = {}        # block key -> chunk ids
        self.total_len = 0

    def add_snapshot(self, rec) -> None:
        for page_key in rec.members:
            v = self.store.versions[page_key][-1]
            ptype = v.info.page_type if v.info is not None else "info"
            if ptype == "listing":
                continue
            blocks = v.blocks
            i = 0
            while i < len(blocks):
                b = blocks[i]
                # A label and the run of short lines under it ("**Lobby**" / "Monday-Thursday: ..." /
                # "Friday: ..." / "Sunday: Closed") form one chunk, so a schedule stays together.
                end = _run_end(blocks, i)
                if end > i + 1:
                    ck = ("run",) + tuple(blocks[j].key for j in range(i, end))
                    cids = self._block_chunks.get(ck)
                    if cids is None:
                        head = display_text(blocks[i].text).rstrip(":")
                        lines = [display_text(blocks[j].text) for j in range(i + 1, end)]
                        cid = self._chunk_id(head + " · " + "\n".join(lines), run=True)
                        cids = [cid] if cid is not None else []
                        self._block_chunks[ck] = cids
                    for c in cids:
                        self.chunks[c].occurrences.append((page_key, rec.index, i))
                    i = end
                    continue
                cids = self._block_chunks.get(b.key)
                if cids is None:
                    cids = [] if _COL_HEADER.search(b.text) else [self._chunk_id(t) for t in _split_block(b.text)]
                    cids = [c for c in cids if c is not None]
                    self._block_chunks[b.key] = cids
                for c in cids:
                    self.chunks[c].occurrences.append((page_key, rec.index, b.index))
                i += 1

    def _chunk_id(self, text: str, run: bool = False):
        toks = [t for t in tokens(text) if t not in STOP]
        if len(toks) < 2:
            return None
        cid = self._by_text.get(text)
        if cid is not None:
            return cid
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        cid = len(self.chunks)
        self.chunks.append(Chunk(cid, text, tf, len(toks), run=run))
        self._by_text[text] = cid
        for t in tf:
            self.postings[t].append(cid)
            self.df[t] += 1
        self.total_len += len(toks)
        return cid

    # ------------------------------------------------------------------ search
    def present_pages(self, chunk, index: int) -> int:
        view = self.store.view(index)
        return len({pk for pk, si, _ in chunk.occurrences if (e := view.get(pk)) is not None
                    and e.present and e.version.snapshot_index == si})

    def idf(self, t: str) -> float:
        n = max(1, len(self.chunks))
        return math.log(1 + (n - self.df.get(t, 0) + 0.5) / (self.df.get(t, 0) + 0.5))

    def search(self, query_tokens: list[str], index: int, k: int = 8, accept=None):
        """Top chunks visible in view `index`: (score, coverage, chunk, best ViewEntry, block idx).
        `accept(chunk_text)` filters candidates before ranking (pattern evidence such as a count)."""
        q = [t for t in dict.fromkeys(query_tokens) if t not in STOP]
        if not q or not self.chunks:
            return []
        view = self.store.view(index)
        avg = self.total_len / max(1, len(self.chunks))
        idfs = {t: self.idf(t) for t in q}
        total_idf = sum(idfs.values()) or 1.0
        scores: dict[int, float] = defaultdict(float)
        matched: dict[int, float] = defaultdict(float)
        for t in q:
            plist = self.postings.get(t)
            if not plist:
                continue
            w = idfs[t]
            for cid in plist:
                c = self.chunks[cid]
                f = c.tf[t]
                scores[cid] += w * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * c.length / avg))     # BM25, b = 0.75
                matched[cid] += w
        ranked = sorted(scores.items(), key=lambda x: -x[1])[: k * 40]
        ref = self.store.snapshots[index].taken_on
        out = []
        if accept is not None:
            ranked = [(cid, s) for cid, s in sorted(scores.items(), key=lambda x: -x[1])
                      if accept(self.chunks[cid].text)][: k * 40]
        for cid, s in ranked:
            c = self.chunks[cid]
            best = None
            for page_key, snap_idx, bidx in c.occurrences:
                e = view.get(page_key)
                if e is None or e.version.snapshot_index != snap_idx:
                    continue
                ptype = e.version.info.page_type if e.version.info is not None else "info"
                w = PAGE_WEIGHT.get(ptype, 0.7) * (1.0 if e.present else 0.85)
                w *= _recency(e.version.info.content_date if e.version.info else None, ref)
                cand = (w, e.present, (e.version.info.content_date if e.version.info else None) or "", page_key)
                if best is None or cand > best[0]:
                    best = (cand, e, bidx)
            if best is None:
                continue
            coverage = matched[cid] / total_idf
            final = s * best[0][0] * (0.3 + 0.7 * coverage)
            # A statement repeated on many present pages is the site's standing statement (small boost).
            pages = {pk for pk, si, _ in c.occurrences if (e := view.get(pk)) is not None
                     and e.present and e.version.snapshot_index == si}
            final *= 1.0 + 0.08 * math.log1p(len(pages))
            out.append((final, coverage, c, best[1], best[2]))
        out.sort(key=lambda x: -x[0])
        return out[:k]


# ====================================================================== extractive answers
_TIME = re.compile(r"\b\d{1,2}(?::\d{2})?\s?(?:a\.?m\.?|p\.?m\.?)|\bnoon\b", re.I)
_ADDRESS = re.compile(r"\b\d{2,6}\s+[A-Z0-9][\w.]*(?:\s+[A-Z0-9][\w.]*){0,4}\s+(?:St|Street|Ave|Avenue|Rd|Road|Dr|"
                      r"Drive|Ln|Lane|Blvd|Boulevard|Way|Ct|Court|Pkwy|Parkway|Hwy|Highway|Pl|Place|Cir|Circle)\b\.?"
                      r"|\b\d{5}\b")
_PHONE = re.compile(r"\b\d{3}[-.)\s]\s?\d{3}[-.]\d{4}\b")
EVIDENCE = {"hours": _TIME, "address": _ADDRESS, "phone": _PHONE}
_QUESTION_NOISE = {"today", "still", "now", "currently", "right", "current", "anymore", "general",
                   "same", "keep", "different", "differ", "vary", "varies"}
_CAP_WORD = re.compile(r"(?<![\w$])([A-Z][\w$’'-]*[A-Za-z0-9])")
_MONTH_NAMES = {"january", "february", "march", "april", "may", "june", "july", "august", "september",
                "october", "november", "december"}
MIN_COVERAGE = 0.45
_DATE_NAMES = _MONTH_NAMES | {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "today"}
_ATTR_WORDS = {"hours": ("hours", "hour", "open", "time", "times"), "address": ("address", "located", "location"),
               "phone": ("phone", "number", "call")}
# A block opening with a bold label starts a new section ("**Drive-up** Monday-Friday: 8 a.m.-6 p.m.").
_SECTION_START = re.compile(r"^\s*(?:\*\*|__)[^*_\n]{1,40}(?:\*\*|__)")
_URL = re.compile(r"https?://|\bwww\.|\b[a-z0-9-]+\.(?:com|org|net)\b", re.I)
_COMPARE = re.compile(r"\b(same|different|differ|vary|varies|all)\b", re.I)
_HOW_MANY = re.compile(r"\bhow many\b|\bnumber of\b", re.I)
_NUMBER_TOKEN = re.compile(r"^\d[\d,]*$")


# A count: a number that is not part of a phone number, date or code, followed by up to three words.
_COUNT = re.compile(r"(?<![\d\-().$/])\b(\d{1,3}(?:,\d{3})*|\d+)\b(?![\d\-)/.:%])((?:\s+[A-Za-z’'-]+){1,3})")


_NUMBER_WORDS = re.compile(
    r"\b(once|twice|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|first|second|third|fourth|"
    r"fifth|sixth|seventh|eighth|ninth|tenth)\b((?:\s+[A-Za-z’'-]+){1,3})", re.I)
_WHEN = re.compile(r"\b(?:19|20)\d{2}\b")


def _has_count(text: str, query: set[str]) -> bool:
    """A number (digits or words) followed within three words by one of the asked words
    ("22 branch office locations", "the fifth time Forbes recognized")."""
    for m in list(_COUNT.finditer(text or "")) + list(_NUMBER_WORDS.finditer(text or "")):
        if any(t in query for t in tokens(m.group(2))):
            return True
    return False


def named_terms(question: str, common: set[str]) -> list[str]:
    """Capitalised words that are not the first word or generic ("Galena", "Mullan", "Hy-Vee")."""
    out = []
    for m in _CAP_WORD.finditer(question):
        w = m.group(1)
        if m.start() == 0 or w.lower() in common or (w.isupper() and len(w) <= 4) or w.lower() in _DATE_NAMES:
            continue
        out.append(w.lower().replace("’", "").replace("'", ""))
    return out


def _sentences(text: str) -> list[str]:
    return [u.strip() for u in _SENT.split(text) if u and u.strip()]


class PassageAnswerer:
    def __init__(self, ks) -> None:
        self.ks = ks

    def query_tokens(self, q) -> tuple[list[str], set[str]]:
        """Stemmed query words (site-defined abbreviations expanded) and the words a pattern stands for."""
        raw = tokens(q.text)
        date_words = {w for w in raw if w.isdigit() or w in _MONTH_NAMES} if q.ref_date else set()
        words = [t for t in raw if t not in STOP and t not in _QUESTION_NOISE and t not in date_words]
        for w in list(words):
            for exp in self.ks.vocab.aliases.get(w, ()):
                words.extend(stem(x) for x in exp)
        # Words whose evidence is a pattern, not the word: "address" -> a street address, "hours" -> times.
        pattern_words = set()
        attr = q.attribute or ""
        if attr in EVIDENCE:
            pattern_words |= {stem(w) for w in _ATTR_WORDS.get(attr, ())}
        if "url" in q.intents:
            pattern_words |= {stem(w) for w in ("url", "link", "website", "page", "webpage")}
        query = [t for t in dict.fromkeys(words) if t not in pattern_words]
        return query, pattern_words

    def answer(self, q, index: int, past: bool, snapshot_date: str):
        from contract import Answer, Source
        idx: PassageIndex = self.ks.passages
        query, _ = self.query_tokens(q)
        counting = bool(_HOW_MANY.search(q.text))
        # Evidence a chunk must contain for this kind of question: a time, an address, a phone number,
        # a URL, a year for "when" questions, a count for "how many" questions.
        evidence = []
        if (q.attribute or "") in EVIDENCE:
            evidence.append(EVIDENCE[q.attribute])
        if "url" in q.intents:
            evidence.append(_URL)
        if re.match(r"\s*when\b", q.text, re.I) and (q.attribute or "") not in EVIDENCE:
            evidence.append(_WHEN)
        qset = set(query)
        strict = counting or bool(evidence)

        def accept(text):
            if any(not pt.search(text) for pt in evidence):
                return False
            return not counting or _has_count(text, qset)
        # Questions that need such evidence look at a wider pool, filtered before ranking.
        results = idx.search(query, index, k=40 if strict else 15, accept=accept if strict else None)
        if not results:
            return self._refusal()
        common = {stem(w) for w in STOP | {"dupaco", "branch", "branches", "visa", "credit", "union"}}
        names = [t for n in named_terms(q.text, common) for t in tokens(n) if t not in common]

        def evident(r):
            _, _, c, entry, _ = r
            ctoks = set(tokens(c.text))
            title_toks = set(tokens(entry.version.title))
            if any(n not in ctoks and n not in title_toks for n in names):
                return False
            if any(not e.search(c.text) for e in evidence):
                return False
            if counting and not _has_count(c.text, qset):
                return False
            return True

        def ok(r):
            if not evident(r):
                return False
            _, cov, c, entry, _ = r
            ctoks = set(tokens(c.text))
            title_toks = set(tokens(entry.version.title))
            # words found only in the page title count half
            extra = sum(idx.idf(t) for t in qset if t in title_toks and t not in ctoks)
            total = sum(idx.idf(t) for t in qset) or 1.0
            return cov + 0.5 * extra / total >= MIN_COVERAGE

        good = [r for r in results if ok(r)]
        if not good:
            return self._refusal()
        best = good[0]
        picks = [best]
        # A second chunk of the same page (the definition next to the best-matching sentence) ...
        # The earliest such chunk wins: articles state the key facts in their lead paragraph.
        same_page = [r for r in results[1:] if r[3].version.key == best[3].version.key and r[2].text != best[2].text
                     and r[0] >= 0.5 * best[0] and evident(r)]
        same = min(same_page, key=lambda r: r[4]) if same_page else None
        if same is not None:
            picks.append(same)
        # ... and the best chunk of another page when it is nearly as good. Comparison questions
        # ("do all branches keep the same hours?") gather up to three pages.
        # Hours / address / phone without a named branch vary by branch: show several, like a comparison.
        comparing = bool(_COMPARE.search(q.text)) or ((q.attribute or "") in EVIDENCE and not names)
        others, used = (2, 0.4) if comparing else (1, 0.75)
        seen_pages = {best[3].version.key}
        for r in good[1:]:
            if others <= 0:
                break
            if r[0] >= used * best[0] and r[2].text not in {p[2].text for p in picks} \
                    and r[3].version.key not in seen_pages:
                picks.append(r)
                seen_pages.add(r[3].version.key)
                others -= 1
        # same-page chunks in page order
        picks = sorted(picks[:1 + (same is not None)], key=lambda r: r[4]) + picks[1 + (same is not None):]
        parts = []
        for r in picks:
            part = self._extract(r[2].text, query)
            blocks = r[3].version.blocks
            # A short line ("Lobby Monday-Thursday: 8:30 a.m.-5 p.m."): add the short lines after it.
            if len(part) < 150 and not r[2].run:
                extra = []
                for b in blocks[r[4] + 1: r[4] + 8]:
                    t = display_text(b.text)
                    if (not t or len(t) > 150 or b.heading_path != blocks[r[4]].heading_path
                            or _SECTION_START.match(b.text) or t.isupper() or t.endswith(":")):
                        break
                    extra.append(t)
                if extra:
                    part = "\n".join([part] + extra)
            # A FAQ question: its answer is the next block of the page.
            if part.rstrip().endswith("?") or part.rstrip().rstrip(")").rstrip("#").endswith("?"):
                blocks = r[3].version.blocks
                if r[4] + 1 < len(blocks):
                    nxt = display_text(blocks[r[4] + 1].text)
                    if nxt:
                        part = f"{part}\n{nxt[:600]}"
            parts.append(part)
        text = "\n\n".join(p for p in parts if p)
        if past:
            text = f"As of {snapshot_date}: " + text
        sources = []
        for r in picks:
            v = r[3].version
            sources.append(Source(url=v.url, snapshot_id=v.snapshot_id, quote=r[2].text[:200]))
        return Answer(text=text, sources=sources, confident=True)

    @staticmethod
    def _extract(text: str, query: list[str]) -> str:
        """The best sentences of a chunk (whole chunk when it is short or a list)."""
        if len(text) <= 420 or text.startswith(("|", "-", "*")) or "\n" in text:
            return text[:900]
        sents = _sentences(text)
        qs = set(query)
        scored = [(sum(1 for t in set(tokens(s)) if t in qs), i) for i, s in enumerate(sents)]
        scored.sort(key=lambda x: (-x[0], x[1]))
        top = scored[0][1]
        keep = [top]
        if top + 1 < len(sents) and len(sents[top]) < 250:
            keep.append(top + 1)
        return " ".join(sents[i] for i in sorted(keep))

    @staticmethod
    def _refusal():
        from contract import Answer
        return Answer(text="I couldn't find that on Dupaco's website, so I can't confirm it. "
                           "Please contact Dupaco directly.", confident=False)
