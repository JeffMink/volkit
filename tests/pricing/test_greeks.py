"""Tests for the analytic Greeks: finite differences and identities."""

import numpy as np
import pytest

from volkit.pricing.black_scholes import bs_price
from volkit.pricing.greeks import delta, gamma, rho, theta, vega

# A grid of (S, K, T, r, sigma, q) covering ITM/ATM/OTM, short and long dated.
rng = np.random.default_rng(1)
N = 500
S = rng.uniform(50, 150, N)
K = rng.uniform(50, 150, N)
T = rng.uniform(0.05, 3.0, N)
R = rng.uniform(0.0, 0.08, N)
SIGMA = rng.uniform(0.05, 0.8, N)
Q = rng.uniform(0.0, 0.04, N)

# Central differences have truncation error O(h^2) plus rounding error
# O(eps / h). With these step sizes the largest observed gap between analytic
# and finite-difference Greeks on this grid is about 1e-6, ten times below ATOL.
ATOL = 1e-5


@pytest.mark.parametrize("call", [True, False])
def test_delta_matches_finite_difference(call: bool) -> None:
    h = 1e-4 * S
    fd = (
        bs_price(S + h, K, T, R, SIGMA, Q, call)
        - bs_price(S - h, K, T, R, SIGMA, Q, call)
    ) / (2 * h)
    np.testing.assert_allclose(delta(S, K, T, R, SIGMA, Q, call), fd, atol=ATOL)


def test_gamma_matches_finite_difference() -> None:
    h = 1e-3 * S
    fd = (
        bs_price(S + h, K, T, R, SIGMA, Q)
        - 2 * bs_price(S, K, T, R, SIGMA, Q)
        + bs_price(S - h, K, T, R, SIGMA, Q)
    ) / h**2
    np.testing.assert_allclose(gamma(S, K, T, R, SIGMA, Q), fd, atol=ATOL)


def test_vega_matches_finite_difference() -> None:
    h = 1e-5
    fd = (bs_price(S, K, T, R, SIGMA + h, Q) - bs_price(S, K, T, R, SIGMA - h, Q)) / (
        2 * h
    )
    np.testing.assert_allclose(vega(S, K, T, R, SIGMA, Q), fd, atol=ATOL)


@pytest.mark.parametrize("call", [True, False])
def test_theta_matches_finite_difference(call: bool) -> None:
    # theta = dV/dt = -dV/dT
    h = 1e-5
    fd = -(
        bs_price(S, K, T + h, R, SIGMA, Q, call)
        - bs_price(S, K, T - h, R, SIGMA, Q, call)
    ) / (2 * h)
    np.testing.assert_allclose(theta(S, K, T, R, SIGMA, Q, call), fd, atol=ATOL)


@pytest.mark.parametrize("call", [True, False])
def test_rho_matches_finite_difference(call: bool) -> None:
    h = 1e-5
    fd = (
        bs_price(S, K, T, R + h, SIGMA, Q, call)
        - bs_price(S, K, T, R - h, SIGMA, Q, call)
    ) / (2 * h)
    np.testing.assert_allclose(rho(S, K, T, R, SIGMA, Q, call), fd, atol=ATOL)


def test_delta_call_minus_put_is_discount_factor() -> None:
    # Differentiate put-call parity C - P = S e^{-qT} - K e^{-rT} in S.
    diff = delta(S, K, T, R, SIGMA, Q, True) - delta(S, K, T, R, SIGMA, Q, False)
    np.testing.assert_allclose(diff, np.exp(-Q * T), atol=1e-12)


def test_rho_call_minus_put() -> None:
    # Differentiate put-call parity in r: rho_C - rho_P = K T e^{-rT}.
    diff = rho(S, K, T, R, SIGMA, Q, True) - rho(S, K, T, R, SIGMA, Q, False)
    np.testing.assert_allclose(diff, K * T * np.exp(-R * T), atol=1e-10)


def test_black_scholes_pde() -> None:
    # Any BSM price satisfies
    # theta + (r - q) S delta + 0.5 sigma^2 S^2 gamma - r V = 0.
    V = bs_price(S, K, T, R, SIGMA, Q)
    lhs = (
        theta(S, K, T, R, SIGMA, Q)
        + (R - Q) * S * delta(S, K, T, R, SIGMA, Q)
        + 0.5 * SIGMA**2 * S**2 * gamma(S, K, T, R, SIGMA, Q)
        - R * V
    )
    np.testing.assert_allclose(lhs, 0.0, atol=1e-10)


def test_vega_gamma_relation() -> None:
    # vega = sigma S^2 T gamma follows from the two closed forms.
    np.testing.assert_allclose(
        vega(S, K, T, R, SIGMA, Q),
        SIGMA * S**2 * T * gamma(S, K, T, R, SIGMA, Q),
        rtol=1e-12,
    )
