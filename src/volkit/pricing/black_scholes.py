"""Black-Scholes-Merton closed-form prices for European options.

All functions are vectorised: every argument may be a scalar or an array, and
arguments are broadcast together with the usual numpy rules. Scalar inputs give
a 0-d array; use ``float(...)`` to get a Python float.

Notation used throughout volkit:

- ``S``: spot price of the underlying
- ``K``: strike
- ``T``: time to expiry in years
- ``r``: continuously compounded risk-free rate
- ``sigma``: volatility (annualised, e.g. 0.2 for 20%)
- ``q``: continuous dividend yield
- ``call``: True for a call, False for a put (may be a boolean array)
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def norm_cdf(x: ArrayLike) -> FloatArray:
    """Standard normal cumulative distribution function N(x)."""
    out: FloatArray = ndtr(np.asarray(x, dtype=np.float64))
    return out


def norm_pdf(x: ArrayLike) -> FloatArray:
    """Standard normal density phi(x) = exp(-x^2 / 2) / sqrt(2 pi)."""
    x = np.asarray(x, dtype=np.float64)
    out: FloatArray = _INV_SQRT_2PI * np.exp(-0.5 * x * x)
    return out


def _as_positive(name: str, x: ArrayLike) -> FloatArray:
    """Convert to a float array and check that every element is > 0."""
    arr = np.asarray(x, dtype=np.float64)
    if np.any(arr <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return arr


def d1_d2(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Return the Black-Scholes-Merton quantities d1 and d2.

    d1 = [ln(S/K) + (r - q + sigma^2 / 2) T] / (sigma sqrt(T))
    d2 = d1 - sigma sqrt(T)

    Raises:
        ValueError: if S, K, T or sigma is not strictly positive.
    """
    S_ = _as_positive("S", S)
    K_ = _as_positive("K", K)
    T_ = _as_positive("T", T)
    sigma_ = _as_positive("sigma", sigma)
    r_ = np.asarray(r, dtype=np.float64)
    q_ = np.asarray(q, dtype=np.float64)

    vol_sqrt_t = sigma_ * np.sqrt(T_)
    d1 = (np.log(S_ / K_) + (r_ - q_ + 0.5 * sigma_**2) * T_) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def bs_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    call: ArrayLike = True,
) -> FloatArray:
    """Black-Scholes-Merton price of a European call or put.

    C = S e^{-qT} N(d1) - K e^{-rT} N(d2)
    P = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)

    with d1, d2 as in :func:`d1_d2` and N the standard normal CDF.

    Raises:
        ValueError: if S, K, T or sigma is not strictly positive.
    """
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    S_ = np.asarray(S, dtype=np.float64)
    K_ = np.asarray(K, dtype=np.float64)
    T_ = np.asarray(T, dtype=np.float64)
    disc_s = S_ * np.exp(-np.asarray(q, dtype=np.float64) * T_)
    disc_k = K_ * np.exp(-np.asarray(r, dtype=np.float64) * T_)
    is_call = np.asarray(call, dtype=np.bool_)

    call_price = disc_s * norm_cdf(d1) - disc_k * norm_cdf(d2)
    put_price = disc_k * norm_cdf(-d2) - disc_s * norm_cdf(-d1)
    out: FloatArray = np.where(is_call, call_price, put_price)
    return out
