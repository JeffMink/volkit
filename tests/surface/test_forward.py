"""Tests for the put-call parity forward fit."""

import numpy as np
import pandas as pd
import pytest

from volkit.pricing import bs_price
from volkit.surface import fit_forwards, implied_forward, implied_forward_given_discount

S0, R, Q, T = 100.0, 0.03, 0.01, 0.5
F_TRUE = S0 * np.exp((R - Q) * T)
D_TRUE = np.exp(-R * T)


def _prices(K: np.ndarray, sigma: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    return bs_price(S0, K, T, R, sigma, Q, True), bs_price(S0, K, T, R, sigma, Q, False)


def test_recovers_forward_and_discount_exactly() -> None:
    # Parity C - P = D (F - K) holds exactly for BSM prices, so the fit must
    # return F = S e^{(r-q)T} and D = e^{-rT} to rounding error.
    K = np.arange(40.0, 165.0, 5.0)
    C, P = _prices(K)
    fit = implied_forward(K, C, P, anchor=S0)
    assert fit.forward == pytest.approx(F_TRUE, abs=1e-10)
    assert fit.discount == pytest.approx(D_TRUE, abs=1e-12)
    assert fit.n_strikes == 20


def test_parity_holds_for_any_smile() -> None:
    # Parity is model-free: a different vol at each strike must not matter.
    K = np.arange(70.0, 135.0, 5.0)
    sigma = np.linspace(0.35, 0.15, K.size)
    C = bs_price(S0, K, T, R, sigma, Q, True)
    P = bs_price(S0, K, T, R, sigma, Q, False)
    fit = implied_forward(K, C, P, anchor=S0)
    assert fit.n_strikes == K.size
    assert fit.forward == pytest.approx(F_TRUE, abs=1e-10)
    assert fit.discount == pytest.approx(D_TRUE, abs=1e-12)


def test_ignores_bad_quotes_far_from_anchor() -> None:
    # Corrupt the far wings; only strikes near the anchor enter the fit.
    K = np.arange(20.0, 185.0, 5.0)
    C, P = _prices(K)
    wings = np.abs(K - S0) > 50
    C = np.where(wings, C + 3.0, C)
    fit = implied_forward(K, C, P, anchor=S0, n_strikes=20)
    assert fit.forward == pytest.approx(F_TRUE, abs=1e-10)


def test_stale_wing_quote_with_small_c_minus_p_is_not_used() -> None:
    # Real SPX data had a deep ITM call quoted near zero, giving |C - P| close
    # to 0 far from the money. Choosing strikes by distance to the anchor
    # keeps it out of the fit.
    K = np.arange(5.0, 165.0, 5.0)
    C, P = _prices(K)
    C[0], P[0] = 0.02, 0.4  # stale pair at K = 5
    fit = implied_forward(K, C, P, anchor=S0, n_strikes=20)
    assert fit.forward == pytest.approx(F_TRUE, abs=1e-10)
    assert fit.discount == pytest.approx(D_TRUE, abs=1e-12)


def test_anchor_only_needs_to_be_rough() -> None:
    K = np.arange(40.0, 165.0, 5.0)
    C, P = _prices(K)
    fit = implied_forward(K, C, P, anchor=0.9 * S0)
    assert fit.forward == pytest.approx(F_TRUE, abs=1e-10)


def test_skips_non_finite_prices() -> None:
    K = np.arange(80.0, 125.0, 5.0)
    C, P = _prices(K)
    C[2] = np.nan
    fit = implied_forward(K, C, P, anchor=S0)
    assert fit.n_strikes == K.size - 1
    assert fit.forward == pytest.approx(F_TRUE, abs=1e-10)


def test_raises_on_too_few_strikes() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        implied_forward([100.0], [5.0], [4.0], anchor=S0)


def test_raises_on_non_positive_discount() -> None:
    # C - P increasing in K is impossible under no arbitrage.
    with pytest.raises(ValueError, match="not positive"):
        implied_forward([90.0, 110.0], [1.0, 3.0], [1.0, 1.0], anchor=S0)


def test_given_discount_recovers_forward() -> None:
    # F = K + (C - P) / D holds strike by strike.
    K = np.arange(70.0, 135.0, 5.0)
    C, P = _prices(K)
    fit = implied_forward_given_discount(K, C, P, D_TRUE, anchor=S0)
    assert fit.forward == pytest.approx(F_TRUE, abs=1e-10)
    assert fit.discount == D_TRUE


def test_given_discount_validates() -> None:
    with pytest.raises(ValueError, match="positive"):
        implied_forward_given_discount([100.0], [5.0], [4.0], 0.0, anchor=S0)
    with pytest.raises(ValueError, match="at least 1"):
        implied_forward_given_discount([100.0], [np.nan], [4.0], 0.99, anchor=S0)


def _quotes(expiries: list[float], r: float = R) -> pd.DataFrame:
    """Exact BSM quotes with a 0.01 half-spread, expiry labelled by T."""
    K = np.arange(60.0, 145.0, 5.0)
    rows = []
    for t in expiries:
        for call in (True, False):
            price = bs_price(S0, K, t, r, 0.2, Q, call)
            rows.append(
                pd.DataFrame(
                    {
                        "expiry": t,
                        "T": t,
                        "strike": K,
                        "call": call,
                        "bid": price - 0.01,
                        "ask": price + 0.01,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def test_fit_forwards_free_and_pooled() -> None:
    expiries = [0.02, 0.05, 0.25, 1.0]
    fwd = fit_forwards(_quotes(expiries), S0, pool_below=0.1)
    assert list(fwd.index) == expiries
    assert fwd["pooled"].tolist() == [True, True, False, False]
    np.testing.assert_allclose(fwd["rate"], R, atol=1e-10)
    for t in expiries:
        assert fwd.loc[t, "forward"] == pytest.approx(
            S0 * np.exp((R - Q) * t), abs=1e-9
        )
        assert fwd.loc[t, "discount"] == pytest.approx(np.exp(-R * t), abs=1e-12)


def test_fit_forwards_pools_short_expiries_by_themselves() -> None:
    fwd = fit_forwards(_quotes([0.02, 0.05]), S0, pool_below=0.1)
    assert not fwd["pooled"].any()
    np.testing.assert_allclose(fwd["rate"], R, atol=1e-9)


def test_fit_forwards_drops_rate_outside_bounds() -> None:
    quotes = pd.concat([_quotes([0.5]), _quotes([1.0], r=0.4)], ignore_index=True)
    fwd = fit_forwards(quotes, S0, rate_bounds=(-0.02, 0.15))
    assert list(fwd.index) == [0.5]


def test_fit_forwards_skips_expiry_without_puts() -> None:
    quotes = pd.DataFrame(
        {
            "expiry": 1.0,
            "T": 1.0,
            "strike": [90.0, 100.0],
            "call": True,
            "bid": [11.0, 5.0],
            "ask": [11.2, 5.2],
        }
    )
    fwd = fit_forwards(quotes, S0)
    assert fwd.empty
    assert list(fwd.columns) == [
        "T", "forward", "discount", "rate", "pooled", "n_strikes"
    ]
