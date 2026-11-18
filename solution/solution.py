"""
Knowledge system for the credit-union snapshot hackathon.

`build()` returns a `KnowledgeSystem` with exactly the three contract methods:

    ingest(snapshot)              learn one snapshot (called oldest first)
    answer(question, as_of=None)  answer from the newest snapshot, or as the site stood on `as_of`
    conflicts()                   contradictions on the site, with the value believed right and why

Each method hands the work to `knowledge.pipeline.Pipeline`; the `knowledge` package holds its parts.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(next(d / "harness" for d in Path(__file__).resolve().parents if (d / "harness" / "contract.py").exists())))
from contract import Answer, Conflict, Snapshot  # noqa: E402

from knowledge.pipeline import Pipeline  # noqa: E402


class KnowledgeSystem:
    def __init__(self):
        self.pipeline = Pipeline()

    def ingest(self, snapshot: Snapshot) -> None:
        self.pipeline.ingest(snapshot)

    def answer(self, question: str, as_of: str = None) -> Answer:
        return self.pipeline.answer(question, as_of)

    def conflicts(self) -> list[Conflict]:
        return self.pipeline.conflicts()


def build():
    return KnowledgeSystem()
