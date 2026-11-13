"""Dev report for the fact extractor (C) and fact history (D). Not used by the solution.

    python dev/inspect_cd.py [--snapshots snapshots] [--product certificate] [--samples 12]
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", type=Path, default=ROOT / "snapshots")
    ap.add_argument("--product", action="append", default=[], help="product key substring to dump")
    ap.add_argument("--samples", type=int, default=12)
    args = ap.parse_args()

    snaps = load_snapshots(args.snapshots)
    ks = build()
    t0 = time.time()
    for s in snaps:
        ks.ingest(s)
    h = ks.facts
    print(f"ingest: {time.time() - t0:.1f}s  sightings: {h.total}  fact keys: {len(h.by_key)}  "
          f"products: {len(h.products)}  rejected by source check: {h.extractor.rejected}  errors: {h.errors}\n")

    last = len(ks.store.snapshots) - 1
    view = h.sightings_in_view(last)
    print(f"latest view: {len(view)} sightings ({sum(not v.present for v in view)} from carried-forward pages)")
    print("  by extractor:", dict(Counter(v.sighting.fact.extractor for v in view)))
    print("  by page type:", dict(Counter(v.sighting.page_type for v in view).most_common()))
    print("  top attributes:", dict(Counter(v.sighting.fact.attribute for v in view).most_common(15)))
    print("  qualifier values:", {q: len(vs) for q, vs in h.qualifier_values.items()})

    prods = Counter(v.sighting.fact.product_key for v in view)
    print(f"\ntop products in latest view:")
    for p, n in prods.most_common(25):
        print(f"  {n:5d}  {p!r}")

    for want in args.product:
        print(f"\n--- latest-view facts for products containing {want!r} (rate sheet / product / promo pages)")
        rows = [v for v in view if want in v.sighting.fact.product_key
                and v.sighting.page_type in ("rate_sheet", "product", "promo", "info")]
        for v in rows[: args.samples * 4]:
            s = v.sighting
            f = s.fact
            print(f"  [{f.extractor:5s}] {f.product_key!r} {f.attribute} ({qstr(f.qualifiers)}) = {f.value!r}"
                  f"  eff={s.content_date} exp={s.expires} {s.url.split('.com', 1)[-1]}")

    print("\nfacts whose value changes across snapshots (same page, same key):")
    shown = 0
    for key, ss in h.by_key.items():
        per_page: dict = {}
        for s in ss:
            per_page.setdefault(s.page_key, []).append(s)
        for page, lst in per_page.items():
            vals = []
            for s in lst:
                if not vals or vals[-1][1] != s.fact.value:
                    vals.append((s.snapshot_id[:13], s.fact.value))
            if len(vals) > 1 and lst[0].page_type in ("rate_sheet", "product"):
                print(f"  {key[0]!r} {key[1]} ({qstr(key[2])})  {page.split('.com', 1)[-1]}: "
                      + " -> ".join(f"{d}:{v}" for d, v in vals))
                shown += 1
                break
        if shown >= args.samples:
            break


if __name__ == "__main__":
    main()
