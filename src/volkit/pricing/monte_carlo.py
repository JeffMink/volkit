"""Monte Carlo pricing of European options under Black-Scholes dynamics.

Under the risk-neutral measure the terminal price is lognormal, so it is
sampled exactly in one step, with no time-discretisation error:

    S_T = S exp((r - q - sigma^2 / 2) T + sigma sqrt(T) Z),   Z ~ N(0, 1).

The price is the sample mean of the discounted payoff e^{-rT} (S_T - K)^+
(call) or e^{-rT} (K - S_T)^+ (put). Two variance-reduction techniques are
available and on by default:

- Antithetic variates: every draw Z is paired with -Z. The estimator averages
  each pair first; pairs are independent, so the standard error is computed
  from the pair averages.
- Control variate: Y = e^{-rT} S_T has known mean S e^{-qT}. The estimator
  uses X - beta (Y - S e^{-qT}) with beta = Cov(X, Y) / Var(Y) estimated from
  the same sample (this adds a bias of order 1/n, negligible next to the
  standard error).

Inputs are scalars: this is a reference pricer for validation, not a tool for
pricing whole surfaces (use :func:`volkit.pricing.bs_price` for that).
"""

from dataclasses import dataclass

import numpy as np

from volkit.pricing.black_scholes import _as_positive


@dataclass(frozen=True)
class MCResult:
    """Monte Carlo estimate with its standard error.

    Attributes:
        price: the Monte Carlo price estimate.
        stderr: estimated standard error of ``price``; an approximate 95%
            confidence interval is ``price +/- 1.96 * stderr``.
        n_paths: number of simulated terminal prices actually used.
    """

    price: float
    stderr: float
    n_paths: int


def mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    call: bool = True,
    n_paths: int = 100_000,
    antithetic: bool = True,
    control_variate: bool = True,
    seed: int | np.random.Generator | None = None,
) -> MCResult:
    """Monte Carlo price of a European call or put.

    S_T = S exp((r - q - sigma^2 / 2) T + sigma sqrt(T) Z)
    price = mean of e^{-rT} payoff(S_T), optionally with antithetic variates
    (Z, -Z) and the control variate Y = e^{-rT} S_T, E[Y] = S e^{-qT}.

    Args:
        S, K, T, r, sigma, q: scalar model inputs (see
            :mod:`volkit.pricing.black_scholes` for notation).
        call: True for a call, False for a put.
        n_paths: number of simulated terminal prices. With antithetic
            variates this must be even (n_paths / 2 pairs).
        antithetic: use antithetic variates.
        control_variate: use the discounted terminal price as control variate.
        seed: seed or ``numpy.random.Generator`` for reproducible results.

    Returns:
        :class:`MCResult` with price, standard error and number of paths.

    Raises:
        ValueError: if S, K, T or sigma is not strictly positive, or n_paths
            is too small (or odd with antithetic variates).
    """
    S_ = float(_as_positive("S", S))
    K_ = float(_as_positive("K", K))
    T_ = float(_as_positive("T", T))
    sigma_ = float(_as_positive("sigma", sigma))
    if antithetic and n_paths % 2 != 0:
        raise ValueError("n_paths must be even when antithetic=True")
    n_samples = n_paths // 2 if antithetic else n_paths
    if n_samples < 2:
        raise ValueError("n_paths too small to estimate a standard error")

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_samples)
    if antithetic:
        z = np.stack([z, -z])  # shape (2, n_samples): row 0 is Z, row 1 is -Z

    discount = np.exp(-r * T_)
    drift = (r - q - 0.5 * sigma_**2) * T_
    s_t = S_ * np.exp(drift + sigma_ * np.sqrt(T_) * z)
    payoff = np.maximum(s_t - K_, 0.0) if call else np.maximum(K_ - s_t, 0.0)

    x = discount * payoff  # discounted payoff
    y = discount * s_t  # control variate
    if antithetic:
        x = x.mean(axis=0)  # pair averages are i.i.d.
        y = y.mean(axis=0)

    if control_variate:
        y_mean = S_ * np.exp(-q * T_)  # E[e^{-rT} S_T] under Q
        y_c = y - y.mean()
        beta = float(np.dot(x - x.mean(), y_c) / np.dot(y_c, y_c))
        x = x - beta * (y - y_mean)

    price = float(x.mean())
    stderr = float(x.std(ddof=1) / np.sqrt(n_samples))
    return MCResult(price=price, stderr=stderr, n_paths=n_paths)
