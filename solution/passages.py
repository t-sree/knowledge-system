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
        self.vecs = None                                      # chunk embeddings (rows = chunk ids)
        self._embedded = 0

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
    def idf(self, t: str) -> float:
        n = max(1, len(self.chunks))
        return math.log(1 + (n - self.df.get(t, 0) + 0.5) / (self.df.get(t, 0) + 0.5))

    # ------------------------------------------------------------------ embeddings
    def embed_pending(self, llm) -> None:
        """Embed chunks added since the last call (page title + chunk text). Skipped without an LLM."""
        if llm is None or not llm.enabled:
            return
        start = self._embedded
        if start >= len(self.chunks):
            return
        texts = []
        for c in self.chunks[start:]:
            title = ""
            if c.occurrences:
                pk, si, _ = c.occurrences[0]
                for v in self.store.versions.get(pk, ()):
                    if v.snapshot_index == si:
                        title = v.title.split("|")[0].strip()
                        break
            texts.append(f"{title}\n{c.text}" if title else c.text)
        from llm import EMBED_BUDGET, EMBED_RETRIES
        m = llm.embed(texts, retries=EMBED_RETRIES, budget=EMBED_BUDGET)
        if m is None:
            return                                      # dense search stays off for these chunks
        np = llm.np
        self.vecs = m if self.vecs is None else np.vstack([self.vecs, m])
        self._embedded = len(self.chunks)

    def dense(self, qvec, n: int = 80) -> dict[int, float]:
        """Cosine similarity of the n nearest embedded chunks."""
        if qvec is None or self.vecs is None:
            return {}
        sims = self.vecs @ qvec
        n = min(n, len(sims))
        top = sims.argpartition(-n)[-n:]
        return {int(i): float(sims[i]) for i in top}

    def coverage(self, toks: set, groups) -> float:
        """Share of the question's words (idf-weighted) found in `toks`. A word counts as found when it
        or one of its alternatives is there. Words that occur nowhere on the site count at 30% weight:
        they can't be matched ("drive-thru" vs "drive-up"), but a question about something the site never
        mentions ("timeshare loan") must not be reduced to its common words."""
        total = got = 0.0
        max_idf = self.idf("\0")                      # idf of a word with no occurrences
        for primary, alts in groups:
            options = [primary, *alts]
            known = [o for o in options if self.df.get(o, 0) > 0]
            if not known:
                total += 0.3 * max_idf
                continue
            w = max(self.idf(o) for o in known)
            total += w
            if any(o in toks for o in options):
                got += w
        return got / total if total else 0.0

    def search(self, query_tokens: list[str], index: int, k: int = 8, accept=None, qvec=None, groups=None):
        """Top chunks visible in view `index`: (score, coverage, chunk, best ViewEntry, block idx, cosine).
        `accept(chunk_text)` filters candidates before ranking (pattern evidence such as a count).
        With `qvec` (question embedding) the nearest chunks by meaning join the keyword candidates."""
        q = [t for t in dict.fromkeys(query_tokens) if t not in STOP]
        if (not q and qvec is None) or not self.chunks:
            return []
        view = self.store.view(index)
        avg = self.total_len / max(1, len(self.chunks))
        idfs = {t: self.idf(t) for t in q}
        groups = groups if groups is not None else [(t, set()) for t in q]
        scores: dict[int, float] = defaultdict(float)
        for t in q:
            plist = self.postings.get(t)
            if not plist:
                continue
            w = idfs[t]
            for cid in plist:
                c = self.chunks[cid]
                f = c.tf[t]
                scores[cid] += w * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * c.length / avg))     # BM25, b = 0.75
        ranked = sorted(scores.items(), key=lambda x: -x[1])[: k * 40]
        ref = self.store.snapshots[index].taken_on
        out = []
        if accept is not None:
            ranked = [(cid, s) for cid, s in sorted(scores.items(), key=lambda x: -x[1])
                      if accept(self.chunks[cid].text)][: k * 40]
        near = self.dense(qvec)
        if near:
            have = {cid for cid, _ in ranked}
            ranked += [(cid, scores.get(cid, 0.0)) for cid in near
                       if cid not in have and (accept is None or accept(self.chunks[cid].text))]
        max_s = max((s for _, s in ranked), default=0.0) or 1.0
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
            coverage = self.coverage(set(c.tf), groups)
            cos = near.get(cid)
            if cos is None and near and self.vecs is not None and cid < len(self.vecs):
                cos = float(self.vecs[cid] @ qvec)
            if near:
                # Hybrid: keyword relevance (scaled to 0..1) and meaning similarity.
                lexical = (s / max_s) * (0.3 + 0.7 * coverage)
                meaning = min(1.0, max(0.0, ((cos or 0.0) - 0.25) / 0.45))
                final = best[0][0] * (0.6 * lexical + 0.4 * meaning)
            else:
                final = s * best[0][0] * (0.3 + 0.7 * coverage)
            # A statement repeated on many present pages is the site's standing statement (small boost).
            pages = {pk for pk, si, _ in c.occurrences if (e := view.get(pk)) is not None
                     and e.present and e.version.snapshot_index == si}
            final *= 1.0 + 0.08 * math.log1p(len(pages))
            out.append((final, coverage, c, best[1], best[2], cos or 0.0))
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
DENSE_OK_WITH_EVIDENCE = 0.45
DENSE_OK = 0.55             # cosine above which a passage counts as on-topic even with few shared words
_DATE_NAMES = _MONTH_NAMES | {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "today"}
_ATTR_WORDS = {"hours": ("hours", "hour", "open", "opening", "close", "closing", "time", "times"), "address": ("address", "located", "location"),
               "phone": ("phone", "number", "call")}
# A block opening with a bold label starts a new section ("**Drive-up** Monday-Friday: 8 a.m.-6 p.m.").
_SECTION_START = re.compile(r"^\s*(?:\*\*|__)[^*_\n]{1,40}(?:\*\*|__)")
_URL = re.compile(r"https?://|\bwww\.|\b[a-z0-9-]+\.(?:com|org|net)\b", re.I)
_COMPARE = re.compile(r"\b(same|different|differ|vary|varies|all)\b", re.I)
_HOW_MANY = re.compile(r"\bhow many\b|\bnumber of\b", re.I)


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


def _with_title(part: str, title: str) -> str:
    """Prefix a passage with its page's title when the passage doesn't name the page's subject
    ("Student Loans: Whether you need to fill the funding gap ...")."""
    head = (title or "").split("|")[0].strip()
    if not head or len(head) > 80:
        return part
    words = {t for t in tokens(head) if t not in STOP and len(t) > 2}
    if not words or words & set(tokens(part)):
        return part
    return f"{head}: {part}"


def _loose(text: str) -> str:
    """Comparison form: lower case, unified dashes and quotes, single spaces."""
    t = (text or "").lower().replace("–", "-").replace("—", "-").replace("’", "'").replace("“", '"').replace("”", '"')
    return " ".join(t.split())


def _sentences(text: str) -> list[str]:
    return [u.strip() for u in _SENT.split(text) if u and u.strip()]


class PassageAnswerer:
    def __init__(self, ks) -> None:
        self.ks = ks
        self.last_coverage = 0.0
        self.last_meaning_only = False

    def query_tokens(self, q) -> tuple[list[str], set[str]]:
        """Stemmed query words (site-defined abbreviations expanded) and the words a pattern stands for."""
        raw = tokens(q.text)
        date_words = {w for w in raw if w.isdigit() or w in _MONTH_NAMES} if q.ref_date else set()
        words = [t for t in raw if t not in STOP and t not in _QUESTION_NOISE and t not in date_words]
        # Abbreviations the site defines are alternatives: "atm" or "automated teller machine".
        self._alts = {}
        for w in list(words):
            for exp in self.ks.vocab.aliases.get(w, ()):
                alt = [stem(x) for x in exp if stem(x) not in STOP]
                self._alts.setdefault(w, set()).update(alt)
                words.extend(alt)
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
        self.last_coverage = 0.0
        self.last_meaning_only = False
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
        qvec = self.ks._question_vector(q.text)
        # Questions that need such evidence look at a wider pool, filtered before ranking.
        expansions = {a for alts in self._alts.values() for a in alts} - set(self._alts)
        groups = [(t, self._alts.get(t, set())) for t in query if t not in expansions or t in self._alts]
        results = idx.search(query, index, k=40 if strict else 15, accept=accept if strict else None, qvec=qvec,
                             groups=groups)
        if not results:
            return self._refusal()
        common = {stem(w) for w in STOP | {"dupaco", "branch", "branches", "visa", "credit", "union"}}
        names = [t for n in named_terms(q.text, common) for t in tokens(n) if t not in common]

        rarest = max(names, key=idx.idf) if names else None

        def evident(r):
            c, entry = r[2], r[3]
            ctoks = set(tokens(c.text))
            title_toks = set(tokens(entry.version.title))
            # Names from the question: the rarest must appear, and at least half of them
            # ("Galena, Illinois" vs a page writing "Galena, IL").
            if names:
                present = [n for n in names if n in ctoks or n in title_toks]
                if rarest not in present or 2 * len(present) < len(names):
                    return False
            if any(not e.search(c.text) for e in evidence):
                return False
            if counting and not _has_count(c.text, qset):
                return False
            return True

        def ok(r):
            if not evident(r):
                return False
            cov, c, entry, cos = r[1], r[2], r[3], r[5]
            # Close in meaning even with few shared words; looser when a pattern (time, address,
            # URL, count) is already required of the passage.
            if cos >= (DENSE_OK_WITH_EVIDENCE if evidence or counting else DENSE_OK):
                meaning_only.add(id(r))          # close in meaning even with few shared words
                return True
            ctoks = set(tokens(c.text))
            title_toks = set(tokens(entry.version.title))
            # words found only in the page title count half
            with_title = idx.coverage(ctoks | title_toks, groups)
            return cov + 0.5 * (with_title - cov) >= MIN_COVERAGE

        meaning_only: set[int] = set()
        good = [r for r in results if ok(r)]
        if not good:
            return self._refusal()
        self.last_coverage = good[0][1]          # how much of the question the best passage covers
        # Accepted only because it is close in meaning, with few of the question's own words
        self.last_meaning_only = id(good[0]) in meaning_only and good[0][1] < MIN_COVERAGE
        picks = self._picks(q, good, results, names, evident)
        parts = [self._part(r, query) for r in picks]
        text = "\n\n".join(p for p in parts if p)
        sources = [Source(url=r[3].version.url, snapshot_id=r[3].version.snapshot_id, quote=r[2].text[:200])
                   for r in picks]
        if past:
            text = f"As of {snapshot_date}: " + text
        return Answer(text=text, sources=sources, confident=True)

    # ------------------------------------------------------------------ LLM path
    def candidates(self, q, index: int, n: int = 8):
        """Passages to show the LLM step: the best hybrid matches plus, when the question needs a
        pattern (time, address, URL, year, count), the best matches that contain it."""
        idx: PassageIndex = self.ks.passages
        query, _ = self.query_tokens(q)
        expansions = {a for alts in self._alts.values() for a in alts} - set(self._alts)
        groups = [(t, self._alts.get(t, set())) for t in query if t not in expansions or t in self._alts]
        qvec = self.ks._question_vector(q.text)
        out = idx.search(query, index, k=n, qvec=qvec, groups=groups)
        evidence = []
        if (q.attribute or "") in EVIDENCE:
            evidence.append(EVIDENCE[q.attribute])
        if "url" in q.intents:
            evidence.append(_URL)
        if re.match(r"\s*when\b", q.text, re.I) and (q.attribute or "") not in EVIDENCE:
            evidence.append(_WHEN)
        counting = bool(_HOW_MANY.search(q.text))
        if evidence or counting:
            qset = set(query)
            strict = idx.search(query, index, k=4, qvec=qvec, groups=groups,
                                accept=lambda t: all(e.search(t) for e in evidence)
                                and (not counting or _has_count(t, qset)))
            seen = {r[2].text for r in out}
            out += [r for r in strict if r[2].text not in seen]
        return out, query

    def evidence_check(self, q, query):
        """The checks every passage answer must pass, whoever picked the passage: names from the
        question (the rarest one and at least half), and the pattern the question type needs (a time
        for hours, a street address, a phone number, a URL, a year for "when", a count for "how many")."""
        idx: PassageIndex = self.ks.passages
        common = {stem(w) for w in STOP | {"dupaco", "branch", "branches", "visa", "credit", "union"}}
        names = [t for n in named_terms(q.text, common) for t in tokens(n) if t not in common]
        rarest = max(names, key=idx.idf) if names else None
        evidence = []
        if (q.attribute or "") in EVIDENCE:
            evidence.append(EVIDENCE[q.attribute])
        if "url" in q.intents:
            evidence.append(_URL)
        if re.match(r"\s*when\b", q.text, re.I) and (q.attribute or "") not in EVIDENCE:
            evidence.append(_WHEN)
        counting = bool(_HOW_MANY.search(q.text))
        qset = set(query)

        def check(r) -> bool:
            c, entry = r[2], r[3]
            if names:
                toks = set(tokens(c.text)) | set(tokens(entry.version.title))
                present = [n for n in names if n in toks]
                if rarest not in present or 2 * len(present) < len(names):
                    return False
            if any(not e.search(c.text) for e in evidence):
                return False
            return not counting or _has_count(c.text, qset)
        return check

    def answer_from(self, q, picks, query, past: bool, snapshot_date: str, quotes=()):
        """Extractive answer from passages the LLM step judged to answer the question. The same evidence
        checks as the rules path apply; no passage left means a refusal. Verified word-for-word quotes
        (the exact answering sentences) are used as the answer when they pass those checks."""
        from contract import Answer, Source
        check = self.evidence_check(q, query)
        good_quotes = [(r, t) for r, t in quotes if check(r)]
        if good_quotes:
            # The quote finds the right passage; the passage's text keeps the facts next to the quote
            # ("…close March 31, 2026" and "…go to Key West and Sycamore Street").
            parts, used = [], set()
            for r, t in good_quotes:
                if r[2].text in used:
                    continue
                used.add(r[2].text)
                part = self._part(r, query) if _loose(t) in _loose(self._part(r, query)) else r[2].text
                parts.append(_with_title(part, r[3].version.title))
            text = "\n\n".join(parts)
            if past:
                text = f"As of {snapshot_date}: " + text
            seen, sources = set(), []
            for r, t in good_quotes:
                v = r[3].version
                if v.url not in seen:
                    seen.add(v.url)
                    sources.append(Source(url=v.url, snapshot_id=v.snapshot_id, quote=t[:200]))
            return Answer(text=text, sources=sources, confident=True)
        picks = [r for r in picks if check(r)]
        if not picks:
            return self._refusal()
        parts = [self._part(r, query) for r in picks]
        text = "\n\n".join(p for p in parts if p)
        if past:
            text = f"As of {snapshot_date}: " + text
        sources = [Source(url=r[3].version.url, snapshot_id=r[3].version.snapshot_id, quote=r[2].text[:200])
                   for r in picks]
        return Answer(text=text, sources=sources, confident=True)

    # ------------------------------------------------------------------ picking passages
    def _picks(self, q, good, results, names, evident):
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
        # ("do all branches keep the same hours?") and hours / address / phone without a named branch
        # (they vary by branch) gather up to three pages.
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
        n_same = 1 + (same is not None)
        return sorted(picks[:n_same], key=lambda r: r[4]) + picks[n_same:]

    def _part(self, r, query) -> str:
        """Text of one picked chunk: best sentences, plus the short lines after a short line and the
        answer block after a FAQ question."""
        part = self._extract(r[2].text, query)
        blocks = r[3].version.blocks
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
        if part.rstrip().rstrip(")").rstrip("#").endswith("?") and r[4] + 1 < len(blocks):
            nxt = display_text(blocks[r[4] + 1].text)
            if nxt:
                part = f"{part}\n{nxt[:600]}"
        return part

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
