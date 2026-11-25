"""Option chain snapshots from Yahoo Finance (via yfinance) and CSV storage.

Network access lives only in :func:`fetch_yahoo_chain`, which needs the
optional dependency (``uv sync --extra data``). Everything else works offline
on saved snapshots, so results are reproducible and tests need no network.

Expiry timestamps include the settlement time, which matters for short
expiries. Yahoo lists SPX contracts under two roots:

- ``SPX``: standard (third-Friday) options, AM-settled at the open, 09:30 ET.
- ``SPXW``: weeklies and end-of-month options, PM-settled at the close,
  16:00 ET.

Both trade on the same dates sometimes, so the same date can carry two
distinct expiries. Roots not listed in ``AM_SETTLED_ROOTS`` are taken as
PM-settled.

The snapshot time ``asof`` is Yahoo's ``regularMarketTime`` of the
underlying: the time of its last regular-session price. Outside trading hours
this is the previous close, so time to expiry is not understated.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TZ = "America/New_York"
AM_SETTLED_ROOTS = frozenset({"SPX"})
# Roots to keep for an SPX surface; others (e.g. SPXO) have unknown settlement.
SPX_ROOTS = frozenset({"SPX", "SPXW"})
_AM_SETTLE = pd.Timedelta(hours=9, minutes=30)
_PM_SETTLE = pd.Timedelta(hours=16)

# OCC-style symbol: root, yymmdd, C or P, strike x 1000 in 8 digits.
_SYMBOL = r"^(?P<root>[A-Z]+)(?P<date>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$"


@dataclass(frozen=True)
class Snapshot:
    """One option chain snapshot.

    Attributes:
        quotes: one row per option; columns ``expiry``, ``strike``, ``call``,
            ``bid``, ``ask``, ``volume``, ``open_interest``, ``last_trade``,
            ``root``, ``symbol``.
        asof: time the snapshot refers to (timezone-aware).
        spot: underlying price at ``asof``.
    """

    quotes: pd.DataFrame
    asof: pd.Timestamp
    spot: float


def parse_symbols(symbols: pd.Series) -> pd.DataFrame:
    """Split OCC-style option symbols into root, expiry, type and strike.

    E.g. ``SPXW261127C05000000`` -> root SPXW, 2026-11-27 16:00 ET, call,
    strike 5000.0. The expiry carries the settlement time of its root.

    Raises:
        ValueError: if a symbol does not match the format.
    """
    parts = symbols.str.extract(_SYMBOL)
    if parts.isna().any(axis=None):
        bad = symbols[parts.isna().any(axis=1)].tolist()
        raise ValueError(f"unrecognised option symbols: {bad[:5]}")
    date = pd.to_datetime(parts["date"], format="%y%m%d").dt.tz_localize(TZ)
    settle = (
        parts["root"].isin(AM_SETTLED_ROOTS).map({True: _AM_SETTLE, False: _PM_SETTLE})
    )
    return pd.DataFrame(
        {
            "root": parts["root"],
            "expiry": date + settle,
            "call": parts["cp"] == "C",
            "strike": parts["strike"].astype(int) / 1000.0,
        },
        index=symbols.index,
    )


def yahoo_to_quotes(calls: pd.DataFrame, puts: pd.DataFrame) -> pd.DataFrame:
    """Convert yfinance ``option_chain`` frames to volkit's quote table.

    Uses the yfinance columns ``contractSymbol``, ``bid``, ``ask``,
    ``volume``, ``openInterest`` and ``lastTradeDate``. Root, expiry, type and
    strike come from the contract symbol. Missing volume and open interest
    (Yahoo omits zeros) become 0.
    """
    raw = pd.concat([calls, puts], ignore_index=True)
    parsed = parse_symbols(raw["contractSymbol"])
    return pd.DataFrame(
        {
            "expiry": parsed["expiry"],
            "strike": parsed["strike"],
            "call": parsed["call"],
            "bid": raw["bid"].astype(float),
            "ask": raw["ask"].astype(float),
            "volume": raw["volume"].fillna(0).astype(int),
            "open_interest": raw["openInterest"].fillna(0).astype(int),
            "last_trade": pd.to_datetime(raw["lastTradeDate"], utc=True).dt.tz_convert(
                TZ
            ),
            "root": parsed["root"],
            "symbol": raw["contractSymbol"],
        }
    )


def fetch_yahoo_chain(
    ticker: str = "^SPX", *, max_days: float | None = None, pause: float = 0.5
) -> Snapshot:
    """Download every listed expiry of an option chain from Yahoo Finance.

    One request per expiry date, ``pause`` seconds apart. Needs network
    access and the optional ``yfinance`` dependency.

    Args:
        ticker: Yahoo ticker of the underlying, e.g. ``^SPX``.
        max_days: skip expiry dates more than this many days ahead.
        pause: seconds to wait between requests.

    Raises:
        ImportError: if yfinance is not installed.
        RuntimeError: if Yahoo returns no expiries or no quotes.
    """
    import yfinance as yf

    tk = yf.Ticker(ticker)
    dates = list(tk.options)
    if not dates:
        raise RuntimeError(f"Yahoo returned no option expiries for {ticker}")

    frames = []
    underlying: dict[str, object] = {}
    today = pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)
    for i, date in enumerate(dates):
        if max_days is not None and (pd.Timestamp(date) - today).days > max_days:
            continue
        if i > 0:
            time.sleep(pause)
        chain = tk.option_chain(date)
        if chain.calls is None or chain.puts is None:
            continue
        frames.append(yahoo_to_quotes(chain.calls, chain.puts))
        underlying = chain.underlying or underlying
    if not frames:
        raise RuntimeError(f"Yahoo returned no option quotes for {ticker}")

    asof = pd.Timestamp(int(str(underlying["regularMarketTime"])), unit="s", tz="UTC")
    quotes = pd.concat(frames, ignore_index=True)
    return Snapshot(
        quotes=quotes,
        asof=asof.tz_convert(TZ),
        spot=float(str(underlying["regularMarketPrice"])),
    )


def save_snapshot(snapshot: Snapshot, path: str | Path) -> None:
    """Write a snapshot to CSV; ``asof`` and ``spot`` are stored as columns."""
    out = snapshot.quotes.copy()
    out["asof"] = snapshot.asof.isoformat()
    out["spot"] = snapshot.spot
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def load_snapshot(path: str | Path) -> Snapshot:
    """Read a snapshot written by :func:`save_snapshot`."""
    df = pd.read_csv(path)
    for col in ("expiry", "last_trade"):
        df[col] = pd.to_datetime(df[col], utc=True, format="ISO8601").dt.tz_convert(TZ)
    df["call"] = df["call"].astype(bool)
    asof = pd.Timestamp(df["asof"].iloc[0]).tz_convert(TZ)
    spot = float(df["spot"].iloc[0])
    return Snapshot(quotes=df.drop(columns=["asof", "spot"]), asof=asof, spot=spot)
