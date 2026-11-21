"""Tests for the Black-Scholes-Merton pricer."""

import numpy as np
import pytest

from volkit.pricing.black_scholes import bs_price, d1_d2


def test_hull_textbook_example() -> None:
    # Hull, "Options, Futures, and Other Derivatives": S=42, K=40, r=10%,
    # sigma=20%, T=0.5 gives c = 4.76 and p = 0.81 (quoted to 2 decimals).
    call = float(bs_price(42.0, 40.0, 0.5, 0.10, 0.20, call=True))
    put = float(bs_price(42.0, 40.0, 0.5, 0.10, 0.20, call=False))
    assert call == pytest.approx(4.76, abs=5e-3)
    assert put == pytest.approx(0.81, abs=5e-3)


def test_put_call_parity_random_grid() -> None:
    # C - P = S e^{-qT} - K e^{-rT}, a model-free identity.
    rng = np.random.default_rng(0)
    n = 10_000
    S = rng.uniform(10, 200, n)
    K = S * rng.uniform(0.5, 2.0, n)
    T = rng.uniform(0.01, 5.0, n)
    r = rng.uniform(-0.01, 0.10, n)
    q = rng.uniform(0.0, 0.05, n)
    sigma = rng.uniform(0.01, 2.0, n)

    c = bs_price(S, K, T, r, sigma, q, call=True)
    p = bs_price(S, K, T, r, sigma, q, call=False)
    np.testing.assert_allclose(
        c - p, S * np.exp(-q * T) - K * np.exp(-r * T), rtol=0, atol=1e-10
    )


def test_large_vol_call_tends_to_discounted_spot() -> None:
    # As sigma -> infinity, d1 -> +inf and d2 -> -inf, so C -> S e^{-qT}.
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.02
    c = float(bs_price(S, K, T, r, 50.0, q))
    assert c == pytest.approx(S * np.exp(-q * T), rel=1e-10)


@pytest.mark.parametrize("call", [True, False])
def test_small_vol_tends_to_discounted_intrinsic(call: bool) -> None:
    # As sigma -> 0 the price tends to max(+-(S e^{-qT} - K e^{-rT}), 0).
    S, K, T, r, q = 100.0, 95.0, 1.0, 0.05, 0.01
    fwd_diff = S * np.exp(-q * T) - K * np.exp(-r * T)
    expected = max(fwd_diff, 0.0) if call else max(-fwd_diff, 0.0)
    price = float(bs_price(S, K, T, r, 1e-6, q, call=call))
    assert price == pytest.approx(expected, abs=1e-10)


def test_price_is_increasing_in_vol() -> None:
    sigma = np.linspace(0.05, 1.0, 50)
    prices = bs_price(100.0, 110.0, 0.5, 0.02, sigma)
    assert np.all(np.diff(prices) > 0)


def test_broadcasting_and_mixed_call_put() -> None:
    K = np.array([[90.0], [100.0], [110.0]])  # shape (3, 1)
    T = np.array([0.25, 0.5, 1.0, 2.0])  # shape (4,)
    call = K < 100.0  # puts for K >= 100, calls below
    prices = bs_price(100.0, K, T, 0.01, 0.2, call=call)
    assert prices.shape == (3, 4)
    expected_put = bs_price(100.0, 110.0, T, 0.01, 0.2, call=False)
    np.testing.assert_allclose(prices[2], expected_put)


def test_d2_equals_d1_minus_vol_sqrt_t() -> None:
    d1, d2 = d1_d2(100.0, 90.0, 0.75, 0.03, 0.25, 0.01)
    assert float(d1 - d2) == pytest.approx(0.25 * np.sqrt(0.75))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"S": 0.0},
        {"K": -1.0},
        {"T": 0.0},
        {"sigma": 0.0},
    ],
)
def test_invalid_inputs_raise(kwargs: dict[str, float]) -> None:
    args: dict[str, float] = {
        "S": 100.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.0,
        "sigma": 0.2,
    }
    args.update(kwargs)
    with pytest.raises(ValueError):
        bs_price(**args)
