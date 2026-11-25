"""Raw implied volatility surface from a table of option quotes.

Pipeline, per quote snapshot taken at time ``asof``:

1. Time to expiry T = (expiry - asof) / 365 days (ACT/365); drop expiries
   closer than ``min_days``.
2. Forward F and discount factor D per expiry from put-call parity
   (:func:`volkit.surface.forward.fit_forwards`), with a pooled rate for
   very short expiries.
3. Clean the quotes (:func:`clean_quotes`): positive bid, ask >= bid,
   relative spread (ask - bid) / mid at most ``max_rel_spread``, and only
   out-of-the-money options (puts with K < F, calls with K >= F).
4. Implied vol of the mid price. With F and D known, Black-76 prices are

       C = D [F N(d1) - K N(d2)],   d1 = [ln(F/K) + sigma^2 T / 2] / (sigma sqrt T)

   which is Black-Scholes-Merton with S = F, r = q = -ln(D) / T, since then
   S e^{-qT} = F D and K e^{-rT} = K D. So the existing solver is reused.
5. SVI coordinates: log-moneyness k = ln(K / F) and total implied variance
   w = sigma^2 T.

Only information in the snapshot itself is used, so there is no look-ahead.
"""

from collections.abc import Collection

import numpy as np
import pandas as pd

from volkit.iv.solver import implied_vol
from volkit.surface.forward import fit_forwards

_SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0

QUOTE_COLUMNS = ("expiry", "strike", "call", "bid", "ask")


def time_to_expiry(expiry: pd.Series, asof: pd.Timestamp) -> pd.Series:
    """Year fraction from ``asof`` to each expiry, ACT/365.

    T = (expiry - asof) in seconds / (365 * 24 * 3600)

    Both must be timezone-aware (or both naive) timestamps. Settlement time
    matters for short expiries: SPX monthlies settle at the open (AM),
    weeklies at the close (PM), so ``expiry`` should include the time of day.
    """
    seconds = (expiry - asof).dt.total_seconds()
    out: pd.Series = seconds / _SECONDS_PER_YEAR
    return out


def clean_quotes(
    quotes: pd.DataFrame,
    forward: pd.Series,
    *,
    max_rel_spread: float = 0.5,
) -> pd.DataFrame:
    """Filter quotes to liquid out-of-the-money options and add mid prices.

    Keeps a row if
        bid > 0,  ask >= bid,  (ask - bid) / mid <= max_rel_spread,
    and the option is out of the money: a put with K < F or a call with
    K >= F, where mid = (bid + ask) / 2 and F is the forward of its expiry.
    OTM options are used because ITM ones carry the same information (by
    put-call parity) with wider spreads.

    Args:
        quotes: one row per option with the columns in ``QUOTE_COLUMNS``.
        forward: forward price per expiry (index = expiry). Expiries missing
            from it are dropped.
        max_rel_spread: largest allowed bid-ask spread relative to the mid.

    Returns:
        The kept rows with extra columns ``forward`` and ``mid``.
    """
    out = quotes.copy()
    out["forward"] = out["expiry"].map(forward)
    out["mid"] = 0.5 * (out["bid"] + out["ask"])
    rel_spread = (out["ask"] - out["bid"]) / out["mid"]
    otm = np.where(
        out["call"], out["strike"] >= out["forward"], out["strike"] < out["forward"]
    )
    keep = (
        out["forward"].notna()
        & (out["bid"] > 0)
        & (out["ask"] >= out["bid"])
        & (rel_spread <= max_rel_spread)
        & otm
    )
    return out.loc[keep].reset_index(drop=True)


def raw_surface(
    quotes: pd.DataFrame,
    asof: pd.Timestamp,
    spot: float,
    *,
    roots: Collection[str] | None = None,
    min_days: float = 7.0,
    max_rel_spread: float = 0.5,
    min_quotes: int = 10,
    n_strikes: int = 20,
    pool_below: float = 0.1,
    rate_bounds: tuple[float, float] = (-0.02, 0.15),
) -> pd.DataFrame:
    """Raw implied volatility surface from one snapshot of option quotes.

    See the module docstring for the steps. Quotes whose mid lies outside the
    no-arbitrage bounds (no implied vol exists) are dropped, and then any
    expiry left with fewer than ``min_quotes`` quotes.

    Args:
        quotes: one row per option with columns ``expiry`` (timestamp
            including settlement time), ``strike``, ``call`` (bool), ``bid``
            and ``ask``; extra columns are kept.
        asof: time the quotes were taken.
        spot: underlying price at ``asof``; only used to locate the money
            for the forward fit, so it need only be roughly right.
        roots: if given, keep only quotes whose ``root`` column is in it
            (e.g. ``{"SPX", "SPXW"}``).
        min_days: drop expiries less than this many calendar days away.
        max_rel_spread: passed to :func:`clean_quotes`.
        min_quotes: drop expiries with fewer kept quotes than this.
        n_strikes, pool_below, rate_bounds: passed to
            :func:`volkit.surface.forward.fit_forwards`.

    Returns:
        One row per kept quote, sorted by expiry and strike, with the input
        columns plus ``T``, ``forward``, ``discount``, ``mid``, ``iv``,
        ``k`` = ln(K/F) and ``w`` = iv^2 T.

    Raises:
        ValueError: if a column in ``QUOTE_COLUMNS`` (or ``root``, when
            ``roots`` is given) is missing.
    """
    required = [*QUOTE_COLUMNS, *(["root"] if roots is not None else [])]
    missing = [c for c in required if c not in quotes.columns]
    if missing:
        raise ValueError(f"quotes is missing columns {missing}")

    q = quotes.copy()
    if roots is not None:
        q = q.loc[q["root"].isin(roots)]
    q["T"] = time_to_expiry(q["expiry"], asof)
    q = q.loc[q["T"] * 365.0 >= min_days]

    fwd = fit_forwards(
        q, spot, n_strikes=n_strikes, pool_below=pool_below, rate_bounds=rate_bounds
    )
    q = clean_quotes(q, fwd["forward"], max_rel_spread=max_rel_spread)
    q["discount"] = q["expiry"].map(fwd["discount"])

    T = q["T"].to_numpy()
    F = q["forward"].to_numpy()
    rate = -np.log(q["discount"].to_numpy()) / T
    # Black-76 through the BSM solver: S = F and r = q = -ln(D) / T.
    q["iv"] = implied_vol(
        q["mid"].to_numpy(),
        F,
        q["strike"].to_numpy(),
        T,
        rate,
        rate,
        q["call"].to_numpy(dtype=bool),
    )
    q = q.loc[q["iv"].notna()]
    q = q.loc[q.groupby("expiry")["iv"].transform("size") >= min_quotes].copy()
    q["k"] = np.log(q["strike"] / q["forward"])
    q["w"] = q["iv"] ** 2 * q["T"]
    return q.sort_values(["expiry", "strike"]).reset_index(drop=True)
