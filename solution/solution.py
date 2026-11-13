"""
Knowledge system for the credit-union snapshot hackathon.

Build status: page store (A), page classifier (B), fact extractor (C), fact history (D),
resolver (E) and fact timeline (F) are in place. `answer()` is still the
starter's naive paragraph search and `conflicts()` returns nothing; both are replaced as the
remaining components land.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(next(d / "harness" for d in Path(__file__).resolve().parents if (d / "harness" / "contract.py").exists())))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from contract import Answer, Conflict, Snapshot, Source  # noqa: E402

from answer import RATE_ATTRS, FactAnswerer  # noqa: E402
from classify import PageClassifier  # noqa: E402
from facts import FactHistory  # noqa: E402
from resolve import Resolver  # noqa: E402
from store import PageStore  # noqa: E402
from question import parse  # noqa: E402
from timeline import Timeline  # noqa: E402
from vocab import Vocabulary  # noqa: E402

WORD = re.compile(r"[a-z0-9$%.]+")


class KnowledgeSystem:
    def __init__(self):
        self.store = PageStore()
        self.classifier = PageClassifier()
        self.classify_errors = 0
        self.facts = FactHistory(self.store)
        self.resolver = Resolver(self.facts)
        self.timeline = Timeline(self.resolver)
        self.vocab = Vocabulary(self.facts)
        self.fact_answerer = FactAnswerer(self)
        self._cache: dict = {}
        self.answer_errors = 0
        self.pages = {}  # url -> (snapshot_id, markdown); used by the interim naive answer()

    def ingest(self, snapshot: Snapshot) -> None:
        rec = self.store.add_snapshot(snapshot)
        for key in rec.members:
            try:
                self.classifier.classify_version(self.store.versions[key][-1])
            except Exception:
                self.classify_errors += 1
        self.facts.add_snapshot(rec)
        try:
            self.vocab.ensure()          # built at ingest so questions don't pay for it
        except Exception:
            pass
        for p in snapshot.pages:
            self.pages[p.url] = (snapshot.snapshot_id, p.markdown)

    # ------------------------------------------------------------------ I: view selector
    def select_view(self, ref_date):
        """Latest snapshot taken on or before `ref_date` (None = latest). None if before the first."""
        snaps = self.store.snapshots
        if not snaps:
            return None
        if ref_date is None:
            return len(snaps) - 1
        idx = None
        for rec in snaps:
            if rec.taken_on <= ref_date:
                idx = rec.index
        return idx

    def answer(self, question, as_of=None):
        ck = (" ".join((question or "").lower().split()), as_of, len(self.store.snapshots))
        hit = self._cache.get(ck)
        if hit is not None:
            return hit
        ans = self._answer(question, as_of)
        self._cache[ck] = ans
        return ans

    def _answer(self, question, as_of=None):
        self.vocab.ensure()
        q = parse(question, self.vocab)
        ref = as_of or q.ref_date
        index = self.select_view(ref)
        if index is None:
            return Answer(text="I don't have information from that early; Dupaco's website records here start "
                               "later than that date, so I can't confirm what applied then.", confident=False)
        rec = self.store.snapshots[index]
        ref_date = ref if ref and ref >= rec.taken_on else rec.taken_on
        past = bool(as_of) or (q.ref_date is not None and index < len(self.store.snapshots) - 1)
        try:
            ans = self.fact_answerer.answer(q, index, ref_date, rec.taken_on, past)
        except Exception:
            self.answer_errors += 1
            ans = None
        if ans is not None:
            return ans
        if q.attribute in RATE_ATTRS:
            # Rates are only ever answered from extracted facts; no matching fact = not published.
            return FactAnswerer._refusal(q)
        return self._naive(question)

    def _naive(self, question):
        """Interim fallback until passage retrieval (step 2) lands: the starter's paragraph search."""
        q = set(WORD.findall(question.lower()))
        best, best_score = None, 0
        for url, (sid, md) in self.pages.items():
            for para in md.split("\n\n"):
                s = len(q & set(WORD.findall(para.lower())))
                if s > best_score:
                    best, best_score = (url, sid, para.strip()), s
        if not best:
            return Answer(text="I don't know.", confident=False)
        url, sid, para = best
        return Answer(text=para[:600], sources=[Source(url=url, snapshot_id=sid)])

    def conflicts(self):
        return []


def build():
    return KnowledgeSystem()
