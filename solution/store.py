"""Page Store (component A).

Keeps every version of every page, per snapshot, and answers one question for the rest of the
system: "what did the website look like as of snapshot N?"

* URLs are normalised so that www/no-www, trailing slashes, fragments and query-string variants
  of the same page collapse into one page key.
* A page that is missing from a later snapshot stays visible in later views, but it is marked
  as carried forward (with the snapshot it was last seen in), never as a fresh observation.
* Pages are split into blocks (paragraphs, tables, lists) that remember their heading path.
  Identical blocks are shared between versions, so later stages can cache work per block.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlsplit

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_WS = re.compile(r"\s+")

# A block repeated on at least this many pages of one snapshot is treated as site boilerplate
# (footers, navigation, "about us" paragraphs). It is flagged, never dropped.
BOILERPLATE_MIN_PAGES = 10


def normalize_url(url: str) -> str:
    """Page key: host without www + lower-cased path without trailing slash; query and fragment dropped."""
    url = (url or "").strip()
    parts = urlsplit(url if "//" in url else "https://" + url)
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = parts.path.rstrip("/").lower()
    return host + path


def _norm_text(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Block:
    """One paragraph, list or table of a page, with the headings it sits under."""
    index: int                       # position within the page
    heading_path: tuple[str, ...]    # e.g. ("Auto Loans", "Rates")
    text: str                        # original markdown of the block
    key: str                         # hash of the normalised text (cache key, boilerplate key)


def split_blocks(markdown: str) -> list[Block]:
    """Split markdown into blank-line separated blocks; heading lines update the heading path."""
    blocks: list[Block] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []

    def flush() -> None:
        text = "\n".join(buf).strip()
        buf.clear()
        if text:
            path = tuple(h for _, h in stack)
            blocks.append(Block(len(blocks), path, text, _hash(_norm_text(text))))

    for line in (markdown or "").splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            if title:
                stack.append((level, title))
        elif not line.strip():
            flush()
        else:
            buf.append(line)
    flush()
    return blocks


@dataclass
class PageVersion:
    """A page as it was downloaded in one snapshot."""
    key: str
    url: str
    title: str
    markdown: str
    fetched_at: str
    snapshot_id: str
    snapshot_index: int
    taken_on: str
    content_hash: str
    blocks: list[Block]
    aliases: tuple[str, ...] = ()    # other raw URLs that collapsed into this page key
    info: Optional[object] = None    # filled by the page classifier (component B)


@dataclass(frozen=True)
class ViewEntry:
    """A page as visible in a snapshot view."""
    version: PageVersion
    present: bool                    # False = carried forward from an earlier snapshot
    last_seen_index: int             # snapshot index the version comes from


@dataclass
class SnapshotRecord:
    index: int
    snapshot_id: str
    taken_on: str
    members: set[str] = field(default_factory=set)      # page keys present in this pull
    boilerplate: set[str] = field(default_factory=set)  # block keys repeated across many pages
    skipped: int = 0                                     # pages that failed to load


class PageStore:
    def __init__(self) -> None:
        self.snapshots: list[SnapshotRecord] = []
        self.versions: dict[str, list[PageVersion]] = {}   # page key -> versions, oldest first
        self._block_cache: dict[str, list[Block]] = {}     # content hash -> blocks
        self._views: dict[int, dict[str, ViewEntry]] = {}

    # ---------- ingest ----------
    def add_snapshot(self, snapshot) -> SnapshotRecord:
        """Store one snapshot. Snapshots must arrive oldest first."""
        if self.snapshots and snapshot.taken_on < self.snapshots[-1].taken_on:
            raise ValueError("snapshots must be ingested oldest first")
        rec = SnapshotRecord(len(self.snapshots), snapshot.snapshot_id, snapshot.taken_on)

        # Group raw pages by key; the variant without a query string (else the shortest URL) wins.
        grouped: dict[str, list] = {}
        for page in snapshot.pages:
            try:
                grouped.setdefault(normalize_url(page.url), []).append(page)
            except Exception:
                rec.skipped += 1
        for key, pages in grouped.items():
            try:
                pages.sort(key=lambda p: ("?" in p.url, len(p.url), p.url))
                page = pages[0]
                content_hash = _hash(page.markdown or "")
                blocks = self._block_cache.get(content_hash)
                if blocks is None:
                    blocks = split_blocks(page.markdown or "")
                    self._block_cache[content_hash] = blocks
                version = PageVersion(
                    key=key, url=page.url, title=page.title or "", markdown=page.markdown or "",
                    fetched_at=page.fetched_at or "", snapshot_id=snapshot.snapshot_id,
                    snapshot_index=rec.index, taken_on=snapshot.taken_on, content_hash=content_hash,
                    blocks=blocks, aliases=tuple(p.url for p in pages[1:]),
                )
                self.versions.setdefault(key, []).append(version)
                rec.members.add(key)
            except Exception:
                rec.skipped += len(pages)

        # Boilerplate: block keys that appear on many distinct pages of this snapshot.
        counts: dict[str, int] = {}
        for key in rec.members:
            for bkey in {b.key for b in self.versions[key][-1].blocks}:
                counts[bkey] = counts.get(bkey, 0) + 1
        rec.boilerplate = {k for k, n in counts.items() if n >= BOILERPLATE_MIN_PAGES}

        self.snapshots.append(rec)
        self._views.clear()
        return rec

    # ---------- reading ----------
    def view(self, index: int) -> dict[str, ViewEntry]:
        """Every page known as of snapshot `index`: present pages first-class, dropped ones carried forward."""
        if index < 0 or index >= len(self.snapshots):
            raise IndexError(index)
        cached = self._views.get(index)
        if cached is not None:
            return cached
        members = self.snapshots[index].members
        out: dict[str, ViewEntry] = {}
        for key, versions in self.versions.items():
            latest = None
            for v in versions:                       # versions are oldest first
                if v.snapshot_index <= index:
                    latest = v
                else:
                    break
            if latest is not None:
                out[key] = ViewEntry(latest, key in members, latest.snapshot_index)
        self._views[index] = out
        return out

    def is_boilerplate(self, block: Block, index: int) -> bool:
        return block.key in self.snapshots[index].boilerplate

    def history(self, key: str) -> list[PageVersion]:
        return self.versions.get(key, [])
