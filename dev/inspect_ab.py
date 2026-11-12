"""Dev report for the page store (A) and page classifier (B). Not used by the solution.

    python dev/inspect_ab.py [--snapshots snapshots] [--samples 8]
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", type=Path, default=ROOT / "snapshots")
    ap.add_argument("--samples", type=int, default=8)
    args = ap.parse_args()

    snaps = load_snapshots(args.snapshots)
    ks = build()
    t0 = time.time()
    for s in snaps:
        ks.ingest(s)
    print(f"ingest: {time.time() - t0:.1f}s, classify errors: {ks.classify_errors}\n")

    store = ks.store
    print("snapshot                    raw  keys  skipped  boilerplate  view(present/carried)")
    for s, rec in zip(snaps, store.snapshots):
        view = store.view(rec.index)
        present = sum(e.present for e in view.values())
        print(f"{rec.snapshot_id:26s} {len(s.pages):4d}  {len(rec.members):4d}  {rec.skipped:7d}  "
              f"{len(rec.boilerplate):11d}  {present}/{len(view) - present}")

    last = store.view(len(store.snapshots) - 1)
    entries = list(last.values())
    print("\npage types in latest view (pages, dated, with rate table):")
    by_type = Counter(e.version.info.page_type for e in entries)
    for t, n in by_type.most_common():
        es = [e for e in entries if e.version.info.page_type == t]
        print(f"  {t:12s} {n:4d}  dated {sum(not e.version.info.undated for e in es):4d}  "
              f"rate-table {sum(e.version.info.has_rate_table for e in es):3d}")

    def paths(es):
        return sorted(e.version.url.split(".com", 1)[-1] or "/" for e in es)

    print("\n'info' pages (no structural rule matched):")
    print("  " + ", ".join(paths([e for e in entries if e.version.info.page_type == "info"])))
    undated = [e for e in entries if e.version.info.undated and e.version.info.page_type in ("newsroom", "article")]
    print(f"\nundated newsroom/article pages ({len(undated)}):")
    print("  " + ", ".join(paths(undated)))
    carried = [e for e in entries if not e.present]
    print(f"\ncarried forward into latest view ({len(carried)}):")
    for e in carried:
        print(f"  {e.version.url}  (last seen {store.snapshots[e.last_seen_index].snapshot_id}, "
              f"{e.version.info.page_type})")

    for kind in ("effective", "expires"):
        marks = [(e.version, m) for e in entries for m in getattr(e.version.info, kind)]
        print(f"\n{kind} marks in latest view: {len(marks)} on {len({v.key for v, _ in marks})} pages")
        seen = set()
        for v, m in marks:
            if (v.key, m.date) in seen or len(seen) >= args.samples:
                continue
            seen.add((v.key, m.date))
            print(f"  {m.date}  {v.url.split('.com', 1)[-1]}  | {m.quote[:110]}")
    notices = [e for e in entries if e.version.info.expired_notice]
    print(f"\nexpired notices in latest view ({len(notices)}):")
    for e in notices:
        print(f"  {e.version.url.split('.com', 1)[-1]}  | {e.version.info.expired_notice[:100]}")


if __name__ == "__main__":
    main()
