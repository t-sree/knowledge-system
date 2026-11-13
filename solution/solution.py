"""
Knowledge system for the credit-union snapshot hackathon.

Build status: components A-F, the rules-only question parser (H), view selector (I), fact and
passage matching (J) and template / extractive answers (K) are in place. `conflicts()` (G) and the
OpenAI fallback are still to come.

answer():  parse (H) -> select the snapshot view (I) -> fact path for rates and fees (J/K);
           otherwise passage search (J) with an extractive answer (K); refusal when neither has evidence.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(next(d / "harness" for d in Path(__file__).resolve().parents if (d / "harness" / "contract.py").exists())))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from contract import Answer, Conflict, Snapshot  # noqa: E402

from answer import RATE_ATTRS, FactAnswerer  # noqa: E402
from classify import PageClassifier  # noqa: E402
from facts import FactHistory  # noqa: E402
from llm import LLM  # noqa: E402
from resolve import Resolver  # noqa: E402
from store import PageStore  # noqa: E402
from passages import PassageAnswerer, PassageIndex  # noqa: E402
from question import parse  # noqa: E402
from timeline import Timeline  # noqa: E402
from vocab import Vocabulary  # noqa: E402


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
        self.passages = PassageIndex(self.store)
        self.passage_answerer = PassageAnswerer(self)
        # OpenAI is optional: checked once here; without a key everything runs on rules.
        self.llm = LLM()
        self._qvecs: dict = {}
        self._cache: dict = {}
        self.answer_errors = 0
        self.passage_errors = 0

    def ingest(self, snapshot: Snapshot) -> None:
        rec = self.store.add_snapshot(snapshot)
        for key in rec.members:
            try:
                self.classifier.classify_version(self.store.versions[key][-1])
            except Exception:
                self.classify_errors += 1
        self.facts.add_snapshot(rec)
        try:
            self.passages.add_snapshot(rec)
        except Exception:
            self.passage_errors += 1
        try:
            self.vocab.ensure()          # built at ingest so questions don't pay for it
        except Exception:
            pass
        try:
            self.passages.embed_pending(self.llm)
        except Exception:
            self.passage_errors += 1

    def question_vector(self, text: str):
        """Embedding of a question (cached), or None without an LLM."""
        if not self.llm.enabled:
            return None
        key = " ".join((text or "").lower().split())
        if key not in self._qvecs:
            m = self.llm.embed([text], timeout=3.0)
            self._qvecs[key] = m[0] if m is not None else None
        return self._qvecs[key]

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
        try:
            return self.passage_answerer.answer(q, index, past, rec.taken_on)
        except Exception:
            self.answer_errors += 1
            return FactAnswerer._refusal(q)

    def conflicts(self):
        return []


def build():
    return KnowledgeSystem()
