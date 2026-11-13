"""Dev helper: answer eval questions and show text, parse and keyword hits. Not used by the solution.

    python dev/ask.py [--eval eval/public_eval.jsonl] [--snapshots snapshots] [--ids dup-001,dup-009] [--full]
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "solution"))
sys.stdout.reconfigure(encoding="utf-8")

from run_eval import load_snapshots  # noqa: E402
from scoring import score_item  # noqa: E402
from solution import build  # noqa: E402
from question import parse  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=ROOT / "eval" / "public_eval.jsonl")
    ap.add_argument("--snapshots", type=Path, default=ROOT / "snapshots")
    ap.add_argument("--ids", default="")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--failing", action="store_true", help="only show items scoring below 1")
    args = ap.parse_args()
    ks = build()
    for s in load_snapshots(args.snapshots):
        ks.ingest(s)
    ids = set(filter(None, args.ids.split(",")))
    total = 0.0
    n = 0
    for line in open(args.eval, encoding="utf-8"):
        d = json.loads(line)
        if ids and d["id"] not in ids:
            continue
        t = time.time()
        a = ks.answer(d["question"], d.get("as_of"))
        dt = time.time() - t
        sc = score_item(d, a)
        total += sc["score"]
        n += 1
        if args.failing and sc["score"] >= 0.999:
            continue
        q = parse(d["question"], ks.vocab)
        text = a.text if args.full else a.text[:260].replace("\n", " | ")
        print(f"== {d['id']} [{d.get('category')}] score={sc['score']:.2f} {dt * 1000:.0f}ms conf={a.confident}")
        print(f"   Q: {d['question']}")
        print(f"   parse: attr={q.attribute} content={q.content} term={q.term} my={q.model_year} "
              f"amt={q.amounts} var={sorted(q.variants)} intents={sorted(q.intents)}")
        print(f"   A: {text}")
        print(f"   src: {[s.url.split('.com', 1)[-1] for s in a.sources]}  must={d.get('must_include')} "
              f"expected_src={[u.split('.com', 1)[-1] for u in d.get('expected_sources', [])]}")
    print(f"\nmean item score (unweighted, no judge): {total / max(n, 1):.3f} over {n}; errors {ks.answer_errors}")


if __name__ == "__main__":
    main()
