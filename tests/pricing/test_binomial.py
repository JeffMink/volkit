"""Tests for the CRR binomial tree (European and American)."""

import math

import numpy as np
import pytest

from volkit.pricing.binomial import binomial_price
from volkit.pricing.black_scholes import bs_price


def _naive_tree(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    call: bool,
    n: int,
    american: bool,
) -> float:
    """Independent, loop-based CRR tree storing every node (for small n)."""
    dt = T / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    disc = math.exp(-r * dt)

    def payoff(s: float) -> float:
        return max(s - K, 0.0) if call else max(K - s, 0.0)

    stock = [[S * u**j * d ** (i - j) for j in range(i + 1)] for i in range(n + 1)]
    value = [payoff(s) for s in stock[n]]
    for i in range(n - 1, -1, -1):
        value = [disc * (p * value[j + 1] + (1 - p) * value[j]) for j in range(i + 1)]
        if american:
            value = [max(v, payoff(s)) for v, s in zip(value, stock[i], strict=True)]
    return value[0]


@pytest.mark.parametrize("american", [False, True])
@pytest.mark.parametrize("call", [True, False])
@pytest.mark.parametrize("n", [1, 2, 5, 20])
def test_matches_naive_tree(n: int, call: bool, american: bool) -> None:
    args = (50.0, 52.0, 5 / 12, 0.10, 0.40, 0.03)
    fast = binomial_price(*args, call=call, n_steps=n, american=american)
    slow = _naive_tree(*args, call=call, n=n, american=american)
    assert fast == pytest.approx(slow, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("call", [True, False])
@pytest.mark.parametrize("K", [80.0, 95.0, 100.0, 110.0, 130.0])
def test_european_converges_to_black_scholes(K: float, call: bool) -> None:
    # CRR error is O(1/n) but oscillates with n for non-ATM strikes, so bound
    # n * |error| rather than assume it falls monotonically. Observed n*|error|
    # on this grid is at most about 2; the bound 3 leaves margin.
    S, T, r, sigma, q = 100.0, 1.0, 0.05, 0.20, 0.01
    exact = float(bs_price(S, K, T, r, sigma, q, call=call))
    for n in (100, 200, 400, 800, 1600):
        err = binomial_price(S, K, T, r, sigma, q, call=call, n_steps=n) - exact
        assert n * abs(err) < 3.0


def test_atm_error_halves_when_steps_double() -> None:
    # For an ATM option with even n the CRR error is smooth: err ~ c / n.
    args = (100.0, 100.0, 1.0, 0.05, 0.20)
    exact = float(bs_price(*args))
    errs = [binomial_price(*args, n_steps=n) - exact for n in (200, 400, 800)]
    assert errs[0] / errs[1] == pytest.approx(2.0, rel=0.01)
    assert errs[1] / errs[2] == pytest.approx(2.0, rel=0.01)


def test_european_put_call_parity_exact() -> None:
    # The tree is linear in the payoff and p makes the discounted stock a
    # martingale on the tree, so parity holds exactly for every n.
    rng = np.random.default_rng(0)
    for _ in range(200):
        S, K = rng.uniform(20, 200), rng.uniform(20, 200)
        T, r = rng.uniform(0.05, 3.0), rng.uniform(-0.01, 0.08)
        sigma, q = rng.uniform(0.1, 0.8), rng.uniform(0.0, 0.05)
        n = int(rng.integers(1, 300))
        c = binomial_price(S, K, T, r, sigma, q, call=True, n_steps=n)
        p = binomial_price(S, K, T, r, sigma, q, call=False, n_steps=n)
        assert c - p == pytest.approx(
            S * math.exp(-q * T) - K * math.exp(-r * T), abs=1e-9
        )


@pytest.mark.parametrize("K", [70.0, 100.0, 130.0])
def test_american_call_no_dividend_equals_european(K: float) -> None:
    # With q = 0 and r >= 0, C >= S - K e^{-rT} >= S - K, so early exercise
    # is never optimal: American call = European call node by node.
    args = (100.0, K, 1.0, 0.05, 0.30)
    am = binomial_price(*args, n_steps=500, american=True)
    eu = binomial_price(*args, n_steps=500, american=False)
    assert am == pytest.approx(eu, rel=1e-13, abs=1e-13)


@pytest.mark.parametrize("K", [70.0, 100.0, 130.0])
def test_american_put_at_least_european_and_intrinsic(K: float) -> None:
    args = (100.0, K, 1.0, 0.05, 0.30)
    am = binomial_price(*args, call=False, n_steps=500, american=True)
    eu = binomial_price(*args, call=False, n_steps=500, american=False)
    assert am >= eu
    assert am >= max(K - 100.0, 0.0)


def test_american_put_early_exercise_premium_positive() -> None:
    # With r > 0 an ITM American put is worth strictly more than European.
    args = (90.0, 100.0, 1.0, 0.08, 0.25)
    am = binomial_price(*args, call=False, n_steps=500, american=True)
    eu = binomial_price(*args, call=False, n_steps=500, american=False)
    assert am - eu > 0.1


def test_deep_itm_american_put_is_intrinsic() -> None:
    # Far enough in the money, immediate exercise is optimal: V = K - S.
    assert binomial_price(
        10.0, 100.0, 1.0, 0.10, 0.20, call=False, n_steps=500, american=True
    ) == pytest.approx(90.0, abs=1e-12)


def test_american_call_with_dividend_exceeds_european() -> None:
    # With a dividend yield, early exercise of a deep ITM call can be optimal.
    args = (150.0, 100.0, 1.0, 0.02, 0.20, 0.08)
    am = binomial_price(*args, n_steps=500, american=True)
    eu = binomial_price(*args, n_steps=500, american=False)
    assert am > eu


def test_american_put_converges_in_n() -> None:
    args = (50.0, 50.0, 5 / 12, 0.10, 0.40)
    a = binomial_price(*args, call=False, n_steps=2000, american=True)
    b = binomial_price(*args, call=False, n_steps=4000, american=True)
    assert abs(a - b) < 1e-3


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        binomial_price(100.0, 100.0, 1.0, 0.05, 0.2, n_steps=0)
    with pytest.raises(ValueError):
        binomial_price(100.0, -1.0, 1.0, 0.05, 0.2)
    with pytest.raises(ValueError, match="probability"):
        # e^{(r-q)dt} = e^{0.5} > u = e^{0.01}, so p > 1.
        binomial_price(100.0, 100.0, 1.0, 0.50, 0.01, n_steps=1)


@pytest.mark.parametrize(
    ("n", "expected", "tol"),
    [(5, 4.49, 5e-3), (30, 4.263, 5e-4), (50, 4.272, 5e-4), (100, 4.278, 5e-4)],
)
def test_hull_american_put(n: int, expected: float, tol: float) -> None:
    # Hull, "Risk Management and Financial Institutions", 4th ed., Appendix F,
    # pp. 605-606 and Figure F.1: five-month American put, S = K = 50,
    # r = 10%, sigma = 40%, no dividends. Tolerance = half the last printed
    # digit.
    price = binomial_price(
        50.0, 50.0, 5 / 12, 0.10, 0.40, call=False, n_steps=n, american=True
    )
    assert price == pytest.approx(expected, abs=tol)
