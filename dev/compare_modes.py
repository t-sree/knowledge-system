"""Dev: score public + dev questions under each passage composition mode. Not used by the solution.

    python dev/compare_modes.py [--modes never,weak,always]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="never,weak,always")
    args = ap.parse_args()
    ks = build()
    for s in load_snapshots(ROOT / "snapshots"):
        ks.ingest(s)
    sets = {name: [json.loads(l) for l in open(p, encoding="utf-8")]
            for name, p in (("public", ROOT / "eval" / "public_eval.jsonl"), ("dev", ROOT / "dev" / "questions.jsonl"))}
    for mode in args.modes.split(","):
        ks.passage_answerer.compose_mode = mode
        ks._cache.clear()
        for name, items in sets.items():
            scores, lat, fails = [], [], []
            for d in items:
                t = time.time()
                a = ks.answer(d["question"], d.get("as_of"))
                lat.append(time.time() - t)
                sc = score_item(d, a)["score"]
                scores.append(sc)
                if sc < 0.999:
                    fails.append(d["id"])
            lat.sort()
            p95 = lat[int(0.95 * (len(lat) - 1))]
            print(f"{mode:7s} {name:6s} mean={sum(scores) / len(scores):.3f} p95={p95:.2f}s  below-1: {' '.join(fails)}")
    print("llm calls", ks.llm.calls, "errors", ks.llm.errors)


if __name__ == "__main__":
    main()
