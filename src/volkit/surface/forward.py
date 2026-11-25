"""Implied forward and discount factor from put-call parity.

For European options on the same underlying, strike K and expiry T,
put-call parity holds without any model:

    C(K) - P(K) = D (F - K)

where D = e^{-rT} is the discount factor and F the forward price of the
underlying for delivery at T. So across strikes, C - P is a straight line in K
with slope -D and intercept D F. A least-squares fit of that line gives the
market-implied D and F without assuming a rate or a dividend yield:

    C - P = a + b K,   D = -b,   F = a / D = -a / b

Only the strikes closest to an anchor price (the spot) are used: there both
quotes are liquid and tight, while far from the money quotes can be stale.
The anchor only has to be roughly right.

For very short expiries the slope is poorly determined, and the implied rate
r = -ln(D) / T magnifies its error by 1 / T. There D is taken from a rate
pooled over longer expiries, and F is refitted with D fixed:

    F = mean over strikes of [K + (C - P) / D]
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ForwardFit:
    """Result of a put-call parity fit for one expiry.

    Attributes:
        forward: implied forward price F.
        discount: implied discount factor D = e^{-rT}.
        n_strikes: number of strikes used in the fit.
    """

    forward: float
    discount: float
    n_strikes: int


def _near_pairs(
    K: ArrayLike,
    call_price: ArrayLike,
    put_price: ArrayLike,
    anchor: float,
    n_strikes: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Strikes and C - P for the ``n_strikes`` finite pairs nearest ``anchor``."""
    K_ = np.asarray(K, dtype=np.float64).ravel()
    diff = (
        np.asarray(call_price, dtype=np.float64).ravel()
        - np.asarray(put_price, dtype=np.float64).ravel()
    )
    ok = np.isfinite(K_) & np.isfinite(diff)
    K_, diff = K_[ok], diff[ok]
    near = np.argsort(np.abs(K_ - anchor), kind="stable")[:n_strikes]
    return K_[near], diff[near]


def implied_forward(
    K: ArrayLike,
    call_price: ArrayLike,
    put_price: ArrayLike,
    *,
    anchor: float,
    n_strikes: int = 20,
) -> ForwardFit:
    """Implied forward F and discount factor D for one expiry.

    Fits C - P = D (F - K) by ordinary least squares on the ``n_strikes``
    strikes closest to ``anchor``. With slope b and intercept a of the
    fitted line,

        D = -b,   F = -a / b.

    Args:
        K: strikes (one per call/put pair, same expiry).
        call_price: call prices (e.g. mids) at those strikes.
        put_price: put prices at those strikes.
        anchor: rough location of the money, e.g. the spot price.
        n_strikes: number of strikes to use; all are used if fewer exist.

    Returns:
        The fitted forward, discount factor and number of strikes used.

    Raises:
        ValueError: if fewer than 2 strikes have finite call and put prices,
            if the strikes used are all equal, or if the fitted discount
            factor is not positive.
    """
    K_, diff = _near_pairs(K, call_price, put_price, anchor, n_strikes)
    if K_.size < 2:
        raise ValueError("need at least 2 strikes with call and put prices")
    if np.ptp(K_) == 0:
        raise ValueError("strikes used in the fit are all equal")

    b, a = np.polyfit(K_, diff, deg=1)  # diff = a + b K
    discount = -b
    if discount <= 0:
        raise ValueError(f"fitted discount factor {discount:.6g} is not positive")
    return ForwardFit(forward=-a / b, discount=discount, n_strikes=int(K_.size))


def implied_forward_given_discount(
    K: ArrayLike,
    call_price: ArrayLike,
    put_price: ArrayLike,
    discount: float,
    *,
    anchor: float,
    n_strikes: int = 20,
) -> ForwardFit:
    """Implied forward for one expiry when the discount factor D is known.

    Each pair gives F = K + (C - P) / D by put-call parity; the estimate is
    the mean over the ``n_strikes`` strikes closest to ``anchor``.

    Raises:
        ValueError: if no strike has finite call and put prices, or if
            ``discount`` is not positive.
    """
    if discount <= 0:
        raise ValueError("discount must be positive")
    K_, diff = _near_pairs(K, call_price, put_price, anchor, n_strikes)
    if K_.size == 0:
        raise ValueError("need at least 1 strike with call and put prices")
    forward = float(np.mean(K_ + diff / discount))
    return ForwardFit(forward=forward, discount=discount, n_strikes=int(K_.size))


class _Row(NamedTuple):
    """One row of the :func:`fit_forwards` output."""

    T: float
    forward: float
    discount: float
    rate: float
    pooled: bool
    n_strikes: int


def fit_forwards(
    quotes: pd.DataFrame,
    spot: float,
    *,
    n_strikes: int = 20,
    pool_below: float = 0.1,
    rate_bounds: tuple[float, float] = (-0.02, 0.15),
) -> pd.DataFrame:
    """Implied forward and discount factor for every expiry in a quote table.

    1. Pair calls and puts by (expiry, strike) on mid prices (bid + ask) / 2,
       using only pairs where both bids are positive.
    2. For each expiry with T >= ``pool_below``, fit F and D freely with
       :func:`implied_forward` and compute r = -ln(D) / T. Drop the expiry if
       r is outside ``rate_bounds`` (a sign of a broken fit).
    3. Pooled rate r_pool = median of the rates kept in step 2. For each
       expiry with T < ``pool_below``, set D = e^{-r_pool T} and refit F with
       :func:`implied_forward_given_discount`. If no expiry is long enough,
       the short expiries are fitted freely, filtered by ``rate_bounds``, and
       pooled among themselves instead.

    Args:
        quotes: one row per option with columns ``expiry``, ``T`` (years),
            ``strike``, ``call`` (bool), ``bid`` and ``ask``.
        spot: spot price, used only to pick strikes near the money.
        n_strikes: strikes per expiry used in each fit.
        pool_below: expiries shorter than this (years) use the pooled rate.
        rate_bounds: allowed range of free-fit implied rates.

    Returns:
        DataFrame indexed by expiry with columns ``T``, ``forward``,
        ``discount``, ``rate``, ``pooled`` (bool) and ``n_strikes``.
        Expiries without a valid fit are left out.
    """
    q = quotes.loc[quotes["bid"] > 0].copy()
    q["mid"] = 0.5 * (q["bid"] + q["ask"])
    pairs = q.pivot_table(
        index=["expiry", "strike"], columns="call", values="mid", aggfunc="first"
    ).dropna()
    T_of = quotes.groupby("expiry")["T"].first()

    groups: dict[object, tuple[float, NDArray[np.float64], pd.DataFrame]] = {}
    if True in pairs.columns and False in pairs.columns:
        for expiry in pairs.index.unique(level="expiry"):
            group = pairs.loc[expiry]  # rows indexed by strike, columns call
            strikes = group.index.to_numpy(dtype=np.float64)
            groups[expiry] = (float(T_of.loc[expiry]), strikes, group)

    long_ = [e for e in groups if groups[e][0] >= pool_below]
    short = [e for e in groups if groups[e][0] < pool_below]
    if not long_:  # nothing to pool from: let the short expiries pool themselves
        long_, short = short, []

    rows: dict[object, _Row] = {}
    for e in long_:
        T, strikes, group = groups[e]
        try:
            fit = implied_forward(
                strikes, group[True], group[False], anchor=spot, n_strikes=n_strikes
            )
        except ValueError:
            continue
        rate = float(-np.log(fit.discount) / T)
        if rate_bounds[0] <= rate <= rate_bounds[1]:
            rows[e] = _Row(T, fit.forward, fit.discount, rate, False, fit.n_strikes)

    if rows and short:
        r_pool = float(np.median([row.rate for row in rows.values()]))
        for e in short:
            T, strikes, group = groups[e]
            fit = implied_forward_given_discount(
                strikes,
                group[True],
                group[False],
                float(np.exp(-r_pool * T)),
                anchor=spot,
                n_strikes=n_strikes,
            )
            rows[e] = _Row(T, fit.forward, fit.discount, r_pool, True, fit.n_strikes)

    out = pd.DataFrame(
        list(rows.values()),
        columns=list(_Row._fields),
        index=pd.Index(list(rows), name="expiry"),
    )
    return out.astype({"pooled": bool, "n_strikes": int}).sort_index()
