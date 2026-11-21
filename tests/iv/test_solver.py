"""Tests for the implied volatility solver."""

import numpy as np
import pytest

from volkit.iv import implied_vol
from volkit.pricing import bs_price, vega

S0, R, Q = 100.0, 0.03, 0.01


def _grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """sigma 1%-300%, K/S 0.5-2, T one week to five years."""
    sigma = np.geomspace(0.01, 3.0, 40)
    moneyness = np.geomspace(0.5, 2.0, 41)
    T = np.array([7 / 365, 30 / 365, 0.25, 1.0, 5.0])
    SIG, M, TT = np.meshgrid(sigma, moneyness, T, indexing="ij")
    return SIG, S0 * M, TT


@pytest.mark.parametrize("call", [True, False])
def test_round_trip_recovers_sigma(call: bool) -> None:
    # sigma -> price -> sigma. Float rounding in the price is about 1e-14, so
    # sigma can only be recovered to roughly 1e-14 / vega. We check sigma to
    # 1e-7 wherever vega > 1e-6 (here about 5,000 of the 8,200 points), and
    # the repriced value everywhere a solution was returned.
    SIG, K, T = _grid()
    price = bs_price(S0, K, T, R, SIG, Q, call)
    iv = implied_vol(price, S0, K, T, R, Q, call)

    informative = vega(S0, K, T, R, SIG, Q) > 1e-6
    assert informative.sum() > 4000
    np.testing.assert_allclose(iv[informative], SIG[informative], rtol=0, atol=1e-7)

    solved = ~np.isnan(iv)
    repriced = bs_price(S0, K[solved], T[solved], R, iv[solved], Q, call)
    np.testing.assert_allclose(repriced, price[solved], rtol=0, atol=1e-12)


@pytest.mark.parametrize("call", [True, False])
def test_nan_only_outside_no_arbitrage_bounds(call: bool) -> None:
    # Every price strictly inside the bounds must get a solution. Prices whose
    # time value rounds to zero sit on the lower bound and must give NaN.
    SIG, K, T = _grid()
    price = bs_price(S0, K, T, R, SIG, Q, call)
    disc_s, disc_k = S0 * np.exp(-Q * T), K * np.exp(-R * T)
    lower = np.maximum(disc_s - disc_k, 0) if call else np.maximum(disc_k - disc_s, 0)
    upper = disc_s if call else disc_k
    inside = (price > lower) & (price < upper)

    iv = implied_vol(price, S0, K, T, R, Q, call)
    assert not np.isnan(iv[inside]).any()
    assert np.isnan(iv[~inside]).all()


def test_hull_example_round_trip() -> None:
    # The Hull example price (sigma = 20%) must give back 20%.
    price = bs_price(42.0, 40.0, 0.5, 0.10, 0.20)
    assert float(implied_vol(price, 42.0, 40.0, 0.5, 0.10)) == pytest.approx(
        0.20, abs=1e-12
    )


def test_call_and_put_from_parity_give_same_vol() -> None:
    # A put priced from a call via put-call parity has the same implied vol.
    K = np.linspace(70, 140, 15)
    T = 0.75
    c = bs_price(S0, K, T, R, 0.3, Q, call=True)
    p = c - S0 * np.exp(-Q * T) + K * np.exp(-R * T)
    iv_c = implied_vol(c, S0, K, T, R, Q, call=True)
    iv_p = implied_vol(p, S0, K, T, R, Q, call=False)
    np.testing.assert_allclose(iv_c, iv_p, rtol=0, atol=1e-10)
    np.testing.assert_allclose(iv_c, 0.3, rtol=0, atol=1e-10)


@pytest.mark.parametrize(
    ("price", "call"),
    [
        (25.0, True),  # above call upper bound S e^{-qT} (S = 20)
        (0.5, True),  # below intrinsic value
        (-1.0, False),  # negative price
        (np.nan, True),  # missing quote
    ],
)
def test_out_of_bounds_gives_nan(price: float, call: bool) -> None:
    # S=20, K=15, T=1, r=q=0: call bounds (5, 20), put bounds (0, 15).
    assert np.isnan(implied_vol(price, 20.0, 15.0, 1.0, 0.0, 0.0, call))


def test_shape_and_mixed_call_put() -> None:
    K = np.array([[80.0, 90.0, 100.0], [100.0, 110.0, 120.0]])
    call = K >= 100.0  # OTM calls above spot, OTM puts below
    price = bs_price(S0, K, 0.5, R, 0.25, Q, call)
    iv = implied_vol(price, S0, K, 0.5, R, Q, call)
    assert iv.shape == (2, 3)
    np.testing.assert_allclose(iv, 0.25, rtol=0, atol=1e-10)


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        implied_vol(5.0, 100.0, 100.0, 0.0, 0.0)
