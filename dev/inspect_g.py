"""Dev report for the conflict register (G). Not used by the solution.

    python dev/inspect_g.py [--snapshots snapshots] [--no-llm]
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "solution"))
sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", type=Path, default=ROOT / "snapshots")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()
    if args.no_llm:
        import llm
        llm.LLM.__init__ = lambda self: [setattr(self, k, v) for k, v in
                                         dict(enabled=False, client=None, np=None, errors=0, calls=0).items()] and None
    from run_eval import load_snapshots
    from solution import build
    ks = build()
    t = time.time()
    for s in load_snapshots(args.snapshots):
        ks.ingest(s)
    print(f"ingest {time.time() - t:.1f}s, register errors {ks.register.errors}")
    t = time.time()
    cs = ks.conflicts()
    print(f"{len(cs)} conflicts in {time.time() - t:.2f}s\n")
    for c in cs:
        print(f"- {c.topic}\n    values: {c.values}\n    resolution: {c.resolution}\n    reason: {c.reason[:200]}")


if __name__ == "__main__":
    main()
