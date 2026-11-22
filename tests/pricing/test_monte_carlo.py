"""Tests for the Monte Carlo pricer, against the Black-Scholes closed form."""

import itertools

import numpy as np
import pytest

from volkit.pricing.black_scholes import bs_price
from volkit.pricing.monte_carlo import mc_price

# (S, K, T, r, sigma, q): ATM, ITM, OTM, short and long expiry, high vol, q > 0.
CASES = [
    (100.0, 100.0, 1.0, 0.05, 0.20, 0.00),
    (100.0, 80.0, 0.5, 0.03, 0.25, 0.01),
    (100.0, 130.0, 2.0, 0.02, 0.30, 0.02),
    (50.0, 55.0, 0.1, 0.01, 0.60, 0.00),
    (42.0, 40.0, 0.5, 0.10, 0.20, 0.00),
]
FLAGS = list(itertools.product([False, True], repeat=2))  # antithetic, cv


@pytest.mark.parametrize("call", [True, False])
@pytest.mark.parametrize(("antithetic", "control_variate"), FLAGS)
@pytest.mark.parametrize(("S", "K", "T", "r", "sigma", "q"), CASES)
def test_matches_black_scholes_within_4_stderr(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    call: bool,
    antithetic: bool,
    control_variate: bool,
) -> None:
    res = mc_price(
        S,
        K,
        T,
        r,
        sigma,
        q,
        call=call,
        n_paths=100_000,
        antithetic=antithetic,
        control_variate=control_variate,
        seed=1,
    )
    exact = float(bs_price(S, K, T, r, sigma, q, call=call))
    assert abs(res.price - exact) < 4 * res.stderr


def test_stderr_shrinks_like_one_over_sqrt_n() -> None:
    # Quadrupling n should halve the standard error.
    args = (100.0, 100.0, 1.0, 0.05, 0.20)
    small = mc_price(
        *args, n_paths=40_000, antithetic=False, control_variate=False, seed=2
    )
    large = mc_price(
        *args, n_paths=160_000, antithetic=False, control_variate=False, seed=3
    )
    assert small.stderr / large.stderr == pytest.approx(2.0, rel=0.05)


@pytest.mark.parametrize("call", [True, False])
def test_variance_reduction_lowers_stderr(call: bool) -> None:
    args = (100.0, 100.0, 1.0, 0.05, 0.20)
    kw = {"call": call, "n_paths": 100_000, "seed": 4}
    plain = mc_price(*args, antithetic=False, control_variate=False, **kw)
    anti = mc_price(*args, antithetic=True, control_variate=False, **kw)
    cv = mc_price(*args, antithetic=False, control_variate=True, **kw)
    both = mc_price(*args, antithetic=True, control_variate=True, **kw)
    assert anti.stderr < plain.stderr
    assert cv.stderr < plain.stderr
    assert both.stderr < min(anti.stderr, cv.stderr)


def test_seed_reproducible() -> None:
    a = mc_price(100.0, 100.0, 1.0, 0.05, 0.2, seed=7)
    b = mc_price(100.0, 100.0, 1.0, 0.05, 0.2, seed=np.random.default_rng(7))
    assert a == b


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        mc_price(100.0, 100.0, 1.0, 0.05, 0.0)
    with pytest.raises(ValueError):
        mc_price(100.0, 100.0, 1.0, 0.05, 0.2, n_paths=1001, antithetic=True)
    with pytest.raises(ValueError):
        mc_price(100.0, 100.0, 1.0, 0.05, 0.2, n_paths=1, antithetic=False)
