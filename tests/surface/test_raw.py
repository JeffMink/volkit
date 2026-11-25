"""Tests for quote cleaning and the raw implied volatility surface."""

import numpy as np
import pandas as pd
import pytest

from volkit.pricing import bs_price
from volkit.surface import clean_quotes, raw_surface, time_to_expiry

S0, R, Q = 100.0, 0.03, 0.01
TZ = "America/New_York"
ASOF = pd.Timestamp("2026-11-20 15:00", tz=TZ)
DAYS = [3, 14, 45, 120, 365]
STRIKES = np.arange(60.0, 155.0, 2.5)
HALF_SPREAD = 0.005


def _smile(k: np.ndarray, T: float) -> np.ndarray:
    """A skewed smile in log-moneyness, flattening with maturity."""
    return 0.2 + (-0.1 * k + 0.3 * k**2) / np.sqrt(1.0 + T)


def _chain() -> pd.DataFrame:
    """Synthetic chain of exact BSM prices with a symmetric bid-ask spread.

    The mid equals the model price exactly, so every step of the pipeline has
    a known answer: F = S e^{(r-q)T}, D = e^{-rT}, iv = _smile(k, T).
    """
    rows = []
    for d in DAYS:
        expiry = (ASOF.normalize() + pd.Timedelta(days=d)).replace(hour=16)
        T = (expiry - ASOF).total_seconds() / (365 * 86400)
        F = S0 * np.exp((R - Q) * T)
        sigma = _smile(np.log(STRIKES / F), T)
        for call in (True, False):
            price = bs_price(S0, STRIKES, T, R, sigma, Q, call)
            rows.append(
                pd.DataFrame(
                    {
                        "expiry": expiry,
                        "strike": STRIKES,
                        "call": call,
                        "bid": price - HALF_SPREAD,
                        "ask": price + HALF_SPREAD,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def test_time_to_expiry_act_365() -> None:
    expiry = pd.Series([ASOF + pd.Timedelta(days=365), ASOF + pd.Timedelta(hours=12)])
    T = time_to_expiry(expiry, ASOF)
    np.testing.assert_allclose(T, [1.0, 0.5 / 365], rtol=1e-15)


def test_clean_quotes_filters() -> None:
    quotes = pd.DataFrame(
        {
            "expiry": "e",
            "strike": [90.0, 110.0, 90.0, 110.0, 95.0, 105.0, 104.0, 96.0],
            "call": [False, True, True, False, False, True, True, False],
            "bid": [1.0, 1.0, 11.0, 11.0, 0.0, 1.0, 2.0, 1.5],
            "ask": [1.1, 1.1, 11.2, 11.2, 0.1, 2.0, 1.0, 1.6],
        }
    )
    out = clean_quotes(quotes, pd.Series({"e": 100.0}), max_rel_spread=0.5)
    # Kept: OTM put 90 and call 110 (tight), OTM put 96.
    # Dropped: ITM call 90 and put 110, zero bid, spread 2/3 of mid, ask < bid.
    assert sorted(out["strike"]) == [90.0, 96.0, 110.0]
    assert out.loc[out["strike"] == 90.0, "mid"].item() == pytest.approx(1.05)
    assert (out["forward"] == 100.0).all()


def test_clean_quotes_call_at_the_forward_is_otm() -> None:
    quotes = pd.DataFrame(
        {
            "expiry": "e",
            "strike": [100.0, 100.0],
            "call": [True, False],
            "bid": [4.0, 4.0],
            "ask": [4.2, 4.2],
        }
    )
    out = clean_quotes(quotes, pd.Series({"e": 100.0}))
    assert out["call"].tolist() == [True]


def test_clean_quotes_drops_unknown_expiry() -> None:
    quotes = pd.DataFrame(
        {"expiry": ["e", "x"], "strike": 110.0, "call": True, "bid": 1.0, "ask": 1.1}
    )
    out = clean_quotes(quotes, pd.Series({"e": 100.0}))
    assert out["expiry"].tolist() == ["e"]


def test_raw_surface_recovers_known_smile() -> None:
    surf = raw_surface(_chain(), ASOF, S0, min_days=7, min_quotes=1)

    # The 3-day expiry is dropped; the rest remain.
    assert surf["expiry"].nunique() == len(DAYS) - 1
    assert (surf["T"] * 365 >= 7).all()

    # Forward and discount factor per expiry from parity; the 14-day expiry
    # uses the pooled rate, which is exact here because every rate is R.
    np.testing.assert_allclose(
        surf["forward"], S0 * np.exp((R - Q) * surf["T"]), rtol=1e-12
    )
    np.testing.assert_allclose(surf["discount"], np.exp(-R * surf["T"]), rtol=1e-12)

    # Implied vols equal the smile that generated the prices. Deep OTM quotes
    # with tiny vega are excluded from the tight check (sigma error is about
    # price error / vega).
    true_iv = _smile(surf["k"].to_numpy(), surf["T"].to_numpy())
    near = surf["mid"] > 0.05
    assert near.sum() >= 0.9 * len(surf)
    np.testing.assert_allclose(surf.loc[near, "iv"], true_iv[near], atol=1e-8)
    np.testing.assert_allclose(surf["iv"], true_iv, atol=1e-5)

    # SVI coordinates.
    np.testing.assert_allclose(surf["k"], np.log(surf["strike"] / surf["forward"]))
    np.testing.assert_allclose(surf["w"], surf["iv"] ** 2 * surf["T"])


def test_raw_surface_keeps_only_otm_one_per_strike() -> None:
    surf = raw_surface(_chain(), ASOF, S0, min_quotes=1)
    assert not surf.duplicated(["expiry", "strike"]).any()
    assert (surf.loc[surf["call"], "strike"] >= surf.loc[surf["call"], "forward"]).all()
    puts = surf.loc[~surf["call"]]
    assert (puts["strike"] < puts["forward"]).all()
    assert surf.equals(surf.sort_values(["expiry", "strike"]).reset_index(drop=True))


def test_raw_surface_drops_prices_without_implied_vol() -> None:
    # A call whose mid is above the forward value D F has no implied vol.
    quotes = _chain()
    row = quotes.index[(quotes["call"]) & (quotes["strike"] == 150.0)][-1]
    quotes.loc[row, ["bid", "ask"]] = [150.0, 150.2]
    surf = raw_surface(quotes, ASOF, S0)
    assert surf["iv"].notna().all()
    expiry = quotes.loc[row, "expiry"]
    assert 150.0 not in surf.loc[surf["expiry"] == expiry, "strike"].to_numpy()


def test_raw_surface_requires_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        raw_surface(_chain().drop(columns="ask"), ASOF, S0)


def test_raw_surface_filters_roots() -> None:
    quotes = _chain()
    quotes["root"] = np.where(quotes.index % 2 == 0, "SPXW", "SPXO")
    surf = raw_surface(quotes, ASOF, S0, roots={"SPXW"}, min_quotes=1)
    assert (surf["root"] == "SPXW").all()
    with pytest.raises(ValueError, match="root"):
        raw_surface(_chain(), ASOF, S0, roots={"SPXW"})


def test_raw_surface_drops_thin_expiries() -> None:
    # Keep 9 strikes in the 45-day expiry; it must be dropped at min_quotes=10.
    quotes = _chain()
    days = (quotes["expiry"] - ASOF).dt.days
    thin = (days == 45) & ~quotes["strike"].between(90.0, 110.0)
    surf = raw_surface(quotes.loc[~thin], ASOF, S0, min_quotes=10)
    kept_days = sorted(set((surf["expiry"] - ASOF).dt.days))
    assert 45 not in kept_days
    assert {120, 365} <= set(kept_days)
    assert (surf.groupby("expiry").size() >= 10).all()
