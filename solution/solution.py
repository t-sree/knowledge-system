"""
Knowledge system for the credit-union snapshot hackathon.

Build status: page store (A), page classifier (B), fact extractor (C) and fact history (D) are in place. `answer()` is still the
starter's naive paragraph search and `conflicts()` returns nothing; both are replaced as the
remaining components land.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(next(d / "harness" for d in Path(__file__).resolve().parents if (d / "harness" / "contract.py").exists())))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from contract import Answer, Conflict, Snapshot, Source  # noqa: E402

from classify import PageClassifier  # noqa: E402
from facts import FactHistory  # noqa: E402
from store import PageStore  # noqa: E402

WORD = re.compile(r"[a-z0-9$%.]+")


class KnowledgeSystem:
    def __init__(self):
        self.store = PageStore()
        self.classifier = PageClassifier()
        self.classify_errors = 0
        self.facts = FactHistory(self.store)
        self.pages = {}  # url -> (snapshot_id, markdown); used by the interim naive answer()

    def ingest(self, snapshot: Snapshot) -> None:
        rec = self.store.add_snapshot(snapshot)
        for key in rec.members:
            try:
                self.classifier.classify_version(self.store.versions[key][-1])
            except Exception:
                self.classify_errors += 1
        self.facts.add_snapshot(rec)
        for p in snapshot.pages:
            self.pages[p.url] = (snapshot.snapshot_id, p.markdown)

    def answer(self, question, as_of=None):
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
