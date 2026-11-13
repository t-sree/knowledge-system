"""Dev report for the resolver (E) and fact timeline (F). Not used by the solution.

    python dev/inspect_ef.py [--snapshots snapshots] [--product certificate] [--samples 15]

Prints:
  * every fact with more than one distinct value in the latest view, with winner and reason
  * disqualified and upcoming sightings in the latest view
  * timelines with more than one period for the --product filters
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "solution"))
sys.stdout.reconfigure(encoding="utf-8")

from run_eval import load_snapshots  # noqa: E402
from solution import build  # noqa: E402


def qstr(quals):
    return ", ".join(f"{k}={v}" for k, v in quals) or "-"


def short(url):
    return url.split(".com", 1)[-1] or url


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", type=Path, default=ROOT / "snapshots")
    ap.add_argument("--product", action="append", default=[])
    ap.add_argument("--samples", type=int, default=15)
    ap.add_argument("--all-types", action="store_true", help="include conflicts won by article/newsroom pages")
    args = ap.parse_args()

    snaps = load_snapshots(args.snapshots)
    ks = build()
    t0 = time.time()
    for s in snaps:
        ks.ingest(s)
    t_ingest = time.time() - t0
    last = len(ks.store.snapshots) - 1
    idx = ks.resolver.view_index(last)
    t0 = time.time()
    results = [ks.resolver.resolve(k, last) for k in idx]
    t_resolve = time.time() - t0
    print(f"ingest {t_ingest:.1f}s; resolved {len(results)} facts in latest view in {t_resolve * 1000:.0f} ms")

    contested = [r for r in results if r.disagreeing]
    print(f"facts with disagreeing values in latest view: {len(contested)}")
    print("  won on:", dict(Counter(r.reason.split(" on ", 1)[1].split(" (")[0] for r in contested)))
    print("  winner page types:", dict(Counter(r.winner.sighting.page_type for r in contested)))
    shown = 0
    for r in sorted(contested, key=lambda r: (r.winner.sighting.page_type not in ("rate_sheet", "product"), r.key[0])):
        if not args.all_types and r.winner.sighting.page_type in ("article", "newsroom"):
            continue
        k = r.key
        losers = "; ".join(f"{d.sighting.fact.value!r}@{short(d.sighting.url)}[{d.sighting.page_type}]"
                           for d in r.disagreeing[:3])
        print(f"  {k[0]!r} {k[1]} ({qstr(k[2])}) = {r.value!r} @ {short(r.winner.sighting.url)}"
              f"[{r.winner.sighting.page_type}, {r.winner.sighting.content_date}]  vs  {losers}")
        shown += 1
        if shown >= args.samples:
            break

    dq = [(r, vs, why) for r in results for vs, why in r.disqualified]
    print(f"\ndisqualified sightings in latest view: {len(dq)}")
    for r, vs, why in dq[: args.samples]:
        print(f"  {r.key[0]!r} {r.key[1]} = {vs.sighting.fact.value!r} @ {short(vs.sighting.url)}: {why}")
    up = [(r, vs) for r in results for vs in r.upcoming]
    print(f"\nnot-yet-effective sightings in latest view: {len(up)}")
    for r, vs in up[: args.samples]:
        print(f"  {r.key[0]!r} {r.key[1]} = {vs.sighting.fact.value!r} effective {vs.sighting.content_date} "
              f"@ {short(vs.sighting.url)}")
    nowin = [r for r in results if r.winner is None]
    print(f"\nfacts with no eligible winner in latest view: {len(nowin)}")
    for r in nowin[: args.samples]:
        print(f"  {r.key[0]!r} {r.key[1]} ({qstr(r.key[2])}): {r.reason}")

    for want in args.product:
        print(f"\n--- timelines with changes for products containing {want!r}")
        n = 0
        for k in sorted(ks.facts.by_key, key=str):
            if want not in k[0]:
                continue
            ps = ks.timeline.periods(k)
            if len(ps) < 2:
                continue
            print(f"  {k[0]!r} {k[1]} ({qstr(k[2])}): " + " -> ".join(
                f"{p.value} [{p.first_snapshot[:2]}-{p.last_snapshot[:2]}"
                + (f", eff {p.valid_from}" if p.valid_from else "") + "]" for p in ps))
            n += 1
            if n >= args.samples:
                break


if __name__ == "__main__":
    main()
