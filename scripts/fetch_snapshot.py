"""Download one option chain snapshot from Yahoo Finance and save it as CSV.

Usage:
    uv run --extra data python scripts/fetch_snapshot.py [--ticker ^SPX]

Writes data/raw/<name>_<asof>.csv, named after the snapshot time.
"""

import argparse
from pathlib import Path

from volkit.surface.yahoo import fetch_yahoo_chain, save_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ticker", default="^SPX")
    parser.add_argument("--max-days", type=float, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    snap = fetch_yahoo_chain(args.ticker, max_days=args.max_days)
    name = args.ticker.lstrip("^").lower()
    path = args.out_dir / f"{name}_{snap.asof:%Y%m%d_%H%M}.csv"
    save_snapshot(snap, path)
    q = snap.quotes
    print(f"asof {snap.asof}, spot {snap.spot:.2f}")
    print(f"{len(q)} quotes, {q['expiry'].nunique()} expiries -> {path}")


if __name__ == "__main__":
    main()
