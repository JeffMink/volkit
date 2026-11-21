"""Closed-form Black-Scholes-Merton Greeks for European options.

Greeks are raw partial derivatives of the price V:

- delta = dV/dS
- gamma = d^2V/dS^2
- vega  = dV/dsigma   (per 1.00 change in vol, not per 1%)
- theta = dV/dt = -dV/dT   (per year, not per day)
- rho   = dV/dr   (per 1.00 change in rate, not per 1%)

Arguments and broadcasting follow :mod:`volkit.pricing.black_scholes`.
phi is the standard normal density and N the standard normal CDF.
"""

import numpy as np
from numpy.typing import ArrayLike

from volkit.pricing.black_scholes import FloatArray, d1_d2, norm_cdf, norm_pdf


def delta(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    call: ArrayLike = True,
) -> FloatArray:
    """Delta, dV/dS.

    call:  e^{-qT} N(d1)
    put:  -e^{-qT} N(-d1)
    """
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-np.asarray(q, dtype=np.float64) * np.asarray(T, dtype=np.float64))
    out: FloatArray = np.where(
        np.asarray(call, dtype=np.bool_),
        disc_q * norm_cdf(d1),
        -disc_q * norm_cdf(-d1),
    )
    return out


def gamma(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
) -> FloatArray:
    """Gamma, d^2V/dS^2 (identical for calls and puts).

    gamma = e^{-qT} phi(d1) / (S sigma sqrt(T))
    """
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    S_ = np.asarray(S, dtype=np.float64)
    T_ = np.asarray(T, dtype=np.float64)
    sigma_ = np.asarray(sigma, dtype=np.float64)
    disc_q = np.exp(-np.asarray(q, dtype=np.float64) * T_)
    out: FloatArray = disc_q * norm_pdf(d1) / (S_ * sigma_ * np.sqrt(T_))
    return out


def vega(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
) -> FloatArray:
    """Vega, dV/dsigma (identical for calls and puts).

    vega = S e^{-qT} phi(d1) sqrt(T)
    """
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    S_ = np.asarray(S, dtype=np.float64)
    T_ = np.asarray(T, dtype=np.float64)
    disc_q = np.exp(-np.asarray(q, dtype=np.float64) * T_)
    out: FloatArray = S_ * disc_q * norm_pdf(d1) * np.sqrt(T_)
    return out


def theta(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    call: ArrayLike = True,
) -> FloatArray:
    """Theta, dV/dt = -dV/dT, per year.

    call: -S e^{-qT} phi(d1) sigma / (2 sqrt(T)) - r K e^{-rT} N(d2)
          + q S e^{-qT} N(d1)
    put:  -S e^{-qT} phi(d1) sigma / (2 sqrt(T)) + r K e^{-rT} N(-d2)
          - q S e^{-qT} N(-d1)
    """
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    S_ = np.asarray(S, dtype=np.float64)
    K_ = np.asarray(K, dtype=np.float64)
    T_ = np.asarray(T, dtype=np.float64)
    r_ = np.asarray(r, dtype=np.float64)
    q_ = np.asarray(q, dtype=np.float64)
    sigma_ = np.asarray(sigma, dtype=np.float64)
    disc_s = S_ * np.exp(-q_ * T_)
    disc_k = K_ * np.exp(-r_ * T_)

    decay = -disc_s * norm_pdf(d1) * sigma_ / (2.0 * np.sqrt(T_))
    call_theta = decay - r_ * disc_k * norm_cdf(d2) + q_ * disc_s * norm_cdf(d1)
    put_theta = decay + r_ * disc_k * norm_cdf(-d2) - q_ * disc_s * norm_cdf(-d1)
    out: FloatArray = np.where(np.asarray(call, dtype=np.bool_), call_theta, put_theta)
    return out


def rho(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    call: ArrayLike = True,
) -> FloatArray:
    """Rho, dV/dr.

    call:  K T e^{-rT} N(d2)
    put:  -K T e^{-rT} N(-d2)
    """
    _, d2 = d1_d2(S, K, T, r, sigma, q)
    K_ = np.asarray(K, dtype=np.float64)
    T_ = np.asarray(T, dtype=np.float64)
    kt_disc = K_ * T_ * np.exp(-np.asarray(r, dtype=np.float64) * T_)
    out: FloatArray = np.where(
        np.asarray(call, dtype=np.bool_),
        kt_disc * norm_cdf(d2),
        -kt_disc * norm_cdf(-d2),
    )
    return out
