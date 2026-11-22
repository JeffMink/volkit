"""Cox-Ross-Rubinstein (CRR) binomial tree for European and American options.

The time to expiry is split into n steps of length dt = T / n. Each step the
price moves up by u or down by d = 1 / u:

    u = exp(sigma sqrt(dt)),   d = 1 / u,
    p = (exp((r - q) dt) - d) / (u - d)      (risk-neutral up probability).

p makes the tree's expected growth over one step exactly exp((r - q) dt), and
u, d match the Black-Scholes variance sigma^2 dt to first order in dt, so the
tree price converges to Black-Scholes as n -> infinity (error of order 1/n).

The price is found by backward induction from the payoffs at expiry:

    V_{i,j} = e^{-r dt} [p V_{i+1,j+1} + (1 - p) V_{i+1,j}]          European
    V_{i,j} = max(payoff(S_{i,j}), e^{-r dt} [...])                  American

where node (i, j) is j up-moves after i steps, S_{i,j} = S u^j d^{i-j}.

Inputs are scalars: this is a reference pricer, not a surface tool.
"""

import numpy as np

from volkit.pricing.black_scholes import FloatArray, _as_positive


def _payoff(s: FloatArray, K: float, call: bool) -> FloatArray:
    out: FloatArray = np.maximum(s - K, 0.0) if call else np.maximum(K - s, 0.0)
    return out


def binomial_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    call: bool = True,
    n_steps: int = 500,
    american: bool = False,
) -> float:
    """CRR binomial tree price of a European or American call or put.

    u = e^{sigma sqrt(dt)}, d = 1/u, p = (e^{(r-q) dt} - d) / (u - d), dt = T/n
    V_i = e^{-r dt} [p V_{i+1}^{up} + (1-p) V_{i+1}^{down}]
    American: V_i = max(payoff(S_i), V_i) at every node.

    Args:
        S, K, T, r, sigma, q: scalar model inputs (see
            :mod:`volkit.pricing.black_scholes` for notation).
        call: True for a call, False for a put.
        n_steps: number of time steps n (at least 1).
        american: allow early exercise at every node.

    Returns:
        The option price as a float.

    Raises:
        ValueError: if S, K, T or sigma is not strictly positive, n_steps < 1,
            or the risk-neutral probability p is outside (0, 1) (dt too large
            for the given r - q and sigma).
    """
    S_ = float(_as_positive("S", S))
    K_ = float(_as_positive("K", K))
    T_ = float(_as_positive("T", T))
    sigma_ = float(_as_positive("sigma", sigma))
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")

    dt = T_ / n_steps
    u = np.exp(sigma_ * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError(
            f"risk-neutral probability p = {p:.4g} is outside (0, 1); increase n_steps"
        )
    disc = np.exp(-r * dt)
    disc_p_up = disc * p
    disc_p_down = disc * (1.0 - p)

    # Prices at expiry: S u^j d^(n-j) = S u^(2j - n), j = 0..n.
    j = np.arange(n_steps + 1)
    values = _payoff(S_ * u ** (2 * j - n_steps), K_, call)

    for i in range(n_steps - 1, -1, -1):
        # values[j] for j = 0..i at step i, from j and j + 1 at step i + 1.
        values = disc_p_up * values[1:] + disc_p_down * values[:-1]
        if american:
            s_i = S_ * u ** (2 * j[: i + 1] - i)
            values = np.maximum(values, _payoff(s_i, K_, call))

    return float(values[0])
