"""Dev: score public, dev and rewording sets with latency stats. Not used by the solution.

    python dev/score_all.py [--no-llm] [--show-fails]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--show-fails", action="store_true")
    args = ap.parse_args()
    if args.no_llm:
        import llm
        llm.LLM.__init__ = lambda self: [setattr(self, k, v) for k, v in dict(
            enabled=False, client=None, np=None, errors=0, calls=0, last_error="").items()] and None
    from run_eval import load_snapshots
    from scoring import score_item
    from solution import build
    ks = build()
    for s in load_snapshots(ROOT / "snapshots"):
        ks.ingest(s)
    all_lat = []
    for name in ("eval/public_eval.jsonl", "dev/questions.jsonl", "dev/paraphrases.jsonl", "dev/probes.jsonl"):
        items = [json.loads(l) for l in open(ROOT / name, encoding="utf-8")]
        scores, lat, fails = [], [], []
        for d in items:
            t = time.time()
            a = ks.answer(d["question"], d.get("as_of"))
            lat.append(time.time() - t)
            sc = score_item(d, a)["score"]
            scores.append(sc)
            if sc < 0.999:
                fails.append((d["id"], sc, a.text[:150].replace("\n", " | ")))
        all_lat += lat
        lat.sort()
        print(f"{name:26s} mean={sum(scores) / len(scores):.3f}  p50={lat[len(lat) // 2]:.2f}s "
              f"p95={lat[int(0.95 * (len(lat) - 1))]:.2f}s max={lat[-1]:.2f}s  fails: {' '.join(f[0] for f in fails)}")
        if args.show_fails:
            for i, sc, t in fails:
                print(f"    {i} {sc:.2f} {t}")
    all_lat.sort()
    print(f"all: p95={all_lat[int(0.95 * (len(all_lat) - 1))]:.2f}s  llm calls={ks.pipeline.llm.calls} errors={ks.pipeline.llm.errors} "
          f"answer errors={ks.pipeline.answer_errors}")


if __name__ == "__main__":
    main()
