"""Offline tests for the Yahoo chain parser and snapshot CSV storage."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from volkit.surface import Snapshot, load_snapshot, save_snapshot
from volkit.surface.yahoo import TZ, parse_symbols, yahoo_to_quotes


def _yahoo_frame(symbols: list[str]) -> pd.DataFrame:
    """A frame shaped like yfinance's option_chain output."""
    n = len(symbols)
    return pd.DataFrame(
        {
            "contractSymbol": symbols,
            "lastTradeDate": pd.to_datetime([1795550400] * n, unit="s", utc=True),
            "strike": [0.0] * n,  # not used: the strike comes from the symbol
            "bid": np.linspace(1.0, 2.0, n),
            "ask": np.linspace(1.1, 2.1, n),
            "volume": [5.0] + [np.nan] * (n - 1),
            "openInterest": [np.nan] + [7.0] * (n - 1),
        }
    )


def test_parse_symbols_roots_and_settlement() -> None:
    out = parse_symbols(
        pd.Series(["SPXW261127C05000000", "SPX261218P04512500", "SPY261218C00600000"])
    )
    assert out["root"].tolist() == ["SPXW", "SPX", "SPY"]
    assert out["call"].tolist() == [True, False, True]
    assert out["strike"].tolist() == [5000.0, 4512.5, 600.0]
    # SPXW and other roots settle at 16:00 ET; standard SPX at 09:30 ET.
    assert out["expiry"].tolist() == [
        pd.Timestamp("2026-11-27 16:00", tz=TZ),
        pd.Timestamp("2026-12-18 09:30", tz=TZ),
        pd.Timestamp("2026-12-18 16:00", tz=TZ),
    ]


def test_same_date_am_and_pm_are_different_expiries() -> None:
    out = parse_symbols(pd.Series(["SPX261218C05000000", "SPXW261218C05000000"]))
    assert out["expiry"].nunique() == 2
    gap = out["expiry"].iloc[1] - out["expiry"].iloc[0]
    assert gap == pd.Timedelta(hours=6, minutes=30)


def test_parse_symbols_rejects_bad_format() -> None:
    with pytest.raises(ValueError, match="unrecognised"):
        parse_symbols(pd.Series(["SPXW261127C05000000", "garbage"]))


def test_yahoo_to_quotes_columns_and_fill() -> None:
    calls = _yahoo_frame(["SPXW261127C05000000", "SPXW261127C05100000"])
    puts = _yahoo_frame(["SPXW261127P04900000"])
    q = yahoo_to_quotes(calls, puts)
    assert list(q.columns) == [
        "expiry",
        "strike",
        "call",
        "bid",
        "ask",
        "volume",
        "open_interest",
        "last_trade",
        "root",
        "symbol",
    ]
    assert q["call"].tolist() == [True, True, False]
    assert q["strike"].tolist() == [5000.0, 5100.0, 4900.0]
    assert q["volume"].tolist() == [5, 0, 5]
    assert q["open_interest"].tolist() == [0, 7, 0]
    assert str(q["last_trade"].dt.tz) == TZ


def test_snapshot_csv_round_trip(tmp_path: Path) -> None:
    quotes = yahoo_to_quotes(
        _yahoo_frame(["SPXW261127C05000000", "SPX261218C05000000"]),
        _yahoo_frame(["SPXW261127P04900000"]),
    )
    snap = Snapshot(quotes, pd.Timestamp("2026-11-24 16:00", tz=TZ), 5012.34)
    path = tmp_path / "sub" / "snap.csv"
    save_snapshot(snap, path)
    back = load_snapshot(path)
    assert back.asof == snap.asof
    assert back.spot == snap.spot
    pd.testing.assert_frame_equal(back.quotes, snap.quotes, check_dtype=False)
