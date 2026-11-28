"""Tests for the raw SVI fit and the static-arbitrage checks."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from volkit.surface import (
    SVIParams,
    butterfly_g,
    check_arbitrage,
    fit_svi,
    fit_svi_surface,
    svi_derivatives,
    svi_total_variance,
)
from volkit.surface.svi import PARAM_NAMES

# A typical equity-index slice: negative skew, positive minimum variance.
EQUITY = SVIParams(a=0.02, b=0.1, rho=-0.6, m=0.05, sigma=0.15)
# Gatheral and Jacquier (2014), Arbitrage-free SVI volatility surfaces, Sec. 3:
# a slice (due to Axel Vogt) with butterfly arbitrage.
VOGT = SVIParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)

FIT_CASES = [
    EQUITY,
    SVIParams(a=-0.01, b=0.2, rho=-0.3, m=0.1, sigma=0.3),  # a < 0, min w > 0
    SVIParams(a=0.005, b=0.05, rho=0.2, m=-0.05, sigma=0.05),  # sharp, short-dated
    SVIParams(a=0.1, b=0.3, rho=-0.8, m=0.2, sigma=0.5),  # long-dated, steep skew
]


def _black_call(K: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Undiscounted Black call with F = 1: C = N(d1) - K N(d2)."""
    sw = np.sqrt(w)
    d1 = -np.log(K) / sw + 0.5 * sw
    return norm.cdf(d1) - K * norm.cdf(d1 - sw)


def test_wings_are_linear_with_slopes_b_one_minus_plus_rho() -> None:
    p = EQUITY
    k = np.array([-1e4, -1e4 + 1.0, 1e4, 1e4 + 1.0])
    w = svi_total_variance(k, p)
    np.testing.assert_allclose(w[1] - w[0], -p.b * (1 - p.rho), rtol=1e-6)
    np.testing.assert_allclose(w[3] - w[2], p.b * (1 + p.rho), rtol=1e-6)


@pytest.mark.parametrize("p", [EQUITY, VOGT])
def test_derivatives_match_finite_differences(p: SVIParams) -> None:
    k = np.linspace(-1.0, 1.0, 41)
    h = 1e-4
    w, w1, w2 = svi_derivatives(k, p)
    wp, wm = svi_total_variance(k + h, p), svi_total_variance(k - h, p)
    np.testing.assert_allclose(w, svi_total_variance(k, p), rtol=0, atol=0)
    np.testing.assert_allclose(w1, (wp - wm) / (2 * h), atol=1e-8)
    np.testing.assert_allclose(w2, (wp - 2 * w + wm) / h**2, atol=1e-6)


def test_g_is_one_for_flat_smile() -> None:
    p = SVIParams(a=0.04, b=0.0, rho=0.0, m=0.0, sigma=0.1)
    np.testing.assert_allclose(butterfly_g(np.linspace(-2, 2, 11), p), 1.0)


@pytest.mark.parametrize("p", [EQUITY, VOGT])
def test_g_matches_density_from_call_prices(p: SVIParams) -> None:
    """Breeden-Litzenberger: d^2C/dK^2 = density of S_T at K (F = 1, D = 1).

    The density of k = ln(S_T / F) is g(k) / sqrt(2 pi w) exp(-d2^2 / 2) with
    d2 = -k / sqrt(w) - sqrt(w) / 2, so the density in K is that divided by K.
    """
    k = np.linspace(-0.8, 0.8, 33)
    K = np.exp(k)
    h = 1e-4

    def call(K: np.ndarray) -> np.ndarray:
        return _black_call(K, svi_total_variance(np.log(K), p))

    fd_density = (call(K + h) - 2 * call(K) + call(K - h)) / h**2
    w = svi_total_variance(k, p)
    d2 = -k / np.sqrt(w) - 0.5 * np.sqrt(w)
    density = butterfly_g(k, p) * np.exp(-0.5 * d2**2) / np.sqrt(2 * np.pi * w) / K
    np.testing.assert_allclose(fd_density, density, atol=1e-5)


def test_vogt_slice_has_butterfly_arbitrage() -> None:
    k = np.linspace(-1.5, 1.5, 1001)
    assert butterfly_g(k, VOGT).min() < 0
    assert butterfly_g(k, EQUITY).min() > 0


@pytest.mark.parametrize("p", FIT_CASES)
def test_fit_recovers_exact_svi(p: SVIParams) -> None:
    k = np.linspace(-0.6, 0.4, 40)
    fit = fit_svi(k, svi_total_variance(k, p))
    assert fit.success
    assert fit.rmse < 1e-9
    got = np.array([getattr(fit.params, n) for n in PARAM_NAMES])
    want = np.array([getattr(p, n) for n in PARAM_NAMES])
    np.testing.assert_allclose(got, want, atol=1e-6)


@pytest.mark.parametrize("seed", range(5))
def test_fit_respects_constraints_on_noisy_data(seed: int) -> None:
    """Noisy and badly shaped data: bounds must still hold."""
    rng = np.random.default_rng(seed)
    k = np.sort(rng.uniform(-0.5, 0.3, 30))
    # Steep wings (slope 3 > Lee's 2) and a dip close to zero.
    w = 0.002 + 3.0 * np.abs(k) * rng.uniform(0.5, 1.5, k.size)
    p = fit_svi(k, w).params
    assert p.b >= 0
    assert abs(p.rho) <= 1
    assert p.b * (1 + abs(p.rho)) <= 2 + 1e-9  # Lee
    assert p.a + p.b * p.sigma * np.sqrt(1 - p.rho**2) >= -1e-12  # min w >= 0
    assert 1e-3 <= p.sigma <= 1.0


def test_fit_noise_level() -> None:
    rng = np.random.default_rng(0)
    k = np.linspace(-0.6, 0.4, 60)
    noise = 1e-4
    w = svi_total_variance(k, EQUITY) + rng.normal(0, noise, k.size)
    fit = fit_svi(k, w)
    # 5 parameters fitted to 60 points: rmse slightly below the noise level.
    assert 0.8 * noise < fit.rmse < noise


def test_fit_input_validation() -> None:
    with pytest.raises(ValueError):
        fit_svi([0.0, 0.1], [0.01])
    with pytest.raises(ValueError):
        fit_svi([0.0, 0.1, 0.2, 0.3], [0.01] * 4)
    with pytest.raises(ValueError):
        fit_svi(np.linspace(-0.1, 0.1, 6), [0.01, 0.01, 0.0, 0.01, 0.01, 0.01])


def _raw_from_svi(slices: list[tuple[float, SVIParams]]) -> pd.DataFrame:
    k = np.linspace(-0.5, 0.3, 25)
    rows = []
    for i, (T, p) in enumerate(slices):
        w = svi_total_variance(k, p)
        rows.append(
            pd.DataFrame(
                {"expiry": f"e{i}", "T": T, "k": k, "w": w, "iv": np.sqrt(w / T)}
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_fit_surface_and_clean_arbitrage_check() -> None:
    slices = [
        (0.25, SVIParams(0.01, 0.06, -0.5, 0.02, 0.1)),
        (0.5, SVIParams(0.02, 0.08, -0.5, 0.03, 0.15)),
        (1.0, SVIParams(0.04, 0.1, -0.5, 0.05, 0.2)),
    ]
    raw = _raw_from_svi(slices[::-1])  # unsorted input
    fits = fit_svi_surface(raw)
    np.testing.assert_allclose(fits["T"], [0.25, 0.5, 1.0])
    for (_, p), (_, row) in zip(slices, fits.iterrows(), strict=True):
        for n in PARAM_NAMES:
            assert row[n] == pytest.approx(getattr(p, n), abs=1e-6)
    assert (fits["rmse_iv"] < 1e-8).all()

    checked = check_arbitrage(fits)
    assert checked["butterfly_ok"].all()
    assert checked["calendar_ok"].all()
    assert np.isnan(checked["calendar_gap"].iloc[0])
    assert (checked["calendar_gap"].iloc[1:] > 0).all()


def test_check_arbitrage_flags_violations() -> None:
    flat = {"b": 0.0, "rho": 0.0, "m": 0.0, "sigma": 0.1}
    vogt = {n: getattr(VOGT, n) for n in PARAM_NAMES}
    fits = pd.DataFrame(
        [
            {"expiry": "e0", "T": 0.5, "a": 0.04, **flat},
            {"expiry": "e1", "T": 1.0, "a": 0.03, **flat},  # below e0 everywhere
            {"expiry": "e2", "T": 2.0, **vogt},
        ]
    )
    out = check_arbitrage(fits)
    assert "min_g_data" not in out.columns  # no k_min / k_max given
    assert out["butterfly_ok"].tolist() == [True, True, False]
    # Vogt's minimum variance, a + b sigma sqrt(1 - rho^2) = 0.0116, is below
    # the flat 0.03 of e1, so e2 has calendar arbitrage too.
    assert out["calendar_ok"].tolist() == [True, False, False]
    assert out["calendar_gap"].iloc[1] == pytest.approx(-0.01, abs=1e-15)
    assert out["min_g"].iloc[0] == pytest.approx(1.0)
    assert out["min_g"].iloc[2] < 0


def test_check_arbitrage_quoted_range() -> None:
    """Vogt's density turns negative only on the right; a quoted range on the
    left is clean, and calendar is checked only where the ranges overlap."""
    vogt = {n: getattr(VOGT, n) for n in PARAM_NAMES}
    k = np.linspace(-1.5, 1.5, 1001)
    g = butterfly_g(k, VOGT)
    k_bad = k[g < 0]
    assert k_bad.min() > 0  # all negative g is at k > 0

    flat = {"b": 0.0, "rho": 0.0, "m": 0.0, "sigma": 0.1}
    fits = pd.DataFrame(
        [
            {"expiry": "e0", "T": 0.5, "a": 0.04, **flat, "k_min": 0.5, "k_max": 1.0},
            {"expiry": "e1", "T": 1.0, "a": 0.03, **flat, "k_min": -1.0, "k_max": 0.0},
            {"expiry": "e2", "T": 2.0, **vogt, "k_min": -1.0, "k_max": 0.0},
        ]
    )
    out = check_arbitrage(fits)
    assert out["butterfly_ok"].tolist() == [True, True, False]
    assert out["butterfly_ok_data"].tolist() == [True, True, True]
    # e0 and e1 ranges do not overlap: no calendar check in range.
    assert np.isnan(out["calendar_gap_data"].iloc[1])
    assert out["calendar_ok_data"].iloc[1]
    assert not out["calendar_ok"].iloc[1]
    # e2 vs e1 on [-1, 0]: Vogt's w there, minus 0.03.
    ko = np.linspace(-1.0, 0.0, 501)
    want = (svi_total_variance(ko, VOGT) - 0.03).min()
    assert out["calendar_gap_data"].iloc[2] == pytest.approx(want, abs=1e-15)


def test_fit_keeps_m_inside_quoted_range() -> None:
    """Data on the left branch only: m must stay at or below k_max."""
    k = np.linspace(-0.6, 0.0, 30)
    p = SVIParams(a=0.01, b=0.1, rho=0.3, m=0.4, sigma=0.3)  # vertex at k = 0.4
    fit = fit_svi(k, svi_total_variance(k, p))
    assert k.min() <= fit.params.m <= k.max()


def _check_inner(y: np.ndarray, w: np.ndarray, sigma_s: float, seed: int) -> None:
    """The active-set QP solution is feasible and at least as good as
    multi-start SLSQP on the same convex problem."""
    from scipy.optimize import minimize

    from volkit.surface.svi import _solve_inner

    x, obj = _solve_inner(y[None, :], w, np.array([sigma_s]))
    a, d, c = x[0]
    assert c - abs(d) >= -1e-12
    assert 2 * sigma_s - c - abs(d) >= -1e-12
    assert a + np.sqrt(max(c * c - d * d, 0.0)) >= -1e-12

    A = np.column_stack([np.ones_like(y), y, np.sqrt(y * y + 1)])
    cons = [
        {"type": "ineq", "fun": lambda x: x[2] - abs(x[1])},
        {"type": "ineq", "fun": lambda x: 2 * sigma_s - x[2] - abs(x[1])},
        {
            "type": "ineq",
            "fun": lambda x: x[0] + np.sqrt(max(x[2] ** 2 - x[1] ** 2, 0)),
        },
    ]
    rng = np.random.default_rng(seed)
    best = np.inf
    for x0 in rng.uniform([-1, -sigma_s, 0], [2, sigma_s, 2 * sigma_s], (10, 3)):
        res = minimize(
            lambda x: np.sum((A @ x - w) ** 2), x0, method="SLSQP", constraints=cons
        )
        a2, d2, c2 = res.x
        if (
            c2 - abs(d2) >= -1e-8
            and 2 * sigma_s - c2 - abs(d2) >= -1e-8
            and a2 + np.sqrt(max(c2 * c2 - d2 * d2, 0)) >= -1e-8
        ):
            best = min(best, res.fun)
    assert obj[0] <= best * (1 + 1e-6) + 1e-12


@pytest.mark.parametrize("seed", range(20))
def test_inner_solver_matches_generic_optimiser(seed: int) -> None:
    rng = np.random.default_rng(seed)
    y = np.sort(rng.uniform(-4, 4, 25))
    w = np.abs(rng.normal(1.0, 0.5, y.size)) + 0.1 * np.abs(y) * rng.uniform(0, 3)
    _check_inner(y, w, rng.uniform(0.05, 2.0), seed)


@pytest.mark.parametrize("sigma_s", [0.5, 2.0])
def test_inner_solver_min_variance_binds(sigma_s: float) -> None:
    """Hinge-shaped data: the polyhedral solution has negative minimum
    variance, so the SLSQP branch runs; its result has min w = 0."""
    from volkit.surface.svi import _solve_inner

    y = np.linspace(-4, 4, 41)
    w = np.maximum(np.abs(y) - 1, 0) + 1e-3
    _check_inner(y, w, sigma_s, 0)
    x, _ = _solve_inner(y[None, :], w, np.array([sigma_s]))
    a, d, c = x[0]
    assert a + np.sqrt(c * c - d * d) == pytest.approx(0.0, abs=1e-12)


def test_inner_solver_batch_matches_rows() -> None:
    from volkit.surface.svi import _solve_inner

    rng = np.random.default_rng(1)
    Y = np.sort(rng.uniform(-3, 3, (6, 20)), axis=1)
    w = rng.uniform(0.5, 1.5, 20)
    s = rng.uniform(0.1, 2.0, 6)
    X, F = _solve_inner(Y, w, s)
    for i in range(6):
        x, f = _solve_inner(Y[i : i + 1], w, s[i : i + 1])
        np.testing.assert_allclose(X[i], x[0], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(F[i], f[0], rtol=1e-12)
