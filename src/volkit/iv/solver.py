"""Implied volatility: invert the Black-Scholes-Merton price for sigma.

Method: Newton-Raphson safeguarded by bisection, vectorised over arrays.

The BSM price is strictly increasing in sigma (vega > 0), going from the
discounted intrinsic value as sigma -> 0 to the upper bound as sigma -> inf.
So every price strictly inside the no-arbitrage bounds has exactly one implied
volatility. The solver keeps a bracket [lo, hi] that always contains it:

- Newton step: sigma_new = sigma - (BS(sigma) - price) / vega(sigma)
- If that step leaves the bracket, or vega is too small to trust, bisect.

Newton converges quadratically near the root; bisection guarantees the
bracket keeps shrinking, so the iteration cannot diverge.
"""

import numpy as np
from numpy.typing import ArrayLike

from volkit.pricing.black_scholes import FloatArray, bs_price
from volkit.pricing.greeks import vega

_MIN_VEGA = 1e-12  # below this a Newton step is numerically meaningless
_MAX_SIGMA = 100.0  # give up if the upper bracket grows past 10,000% vol


def implied_vol(
    price: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike = 0.0,
    call: ArrayLike = True,
    *,
    xtol: float = 1e-12,
    max_iter: int = 100,
) -> FloatArray:
    """Black-Scholes-Merton implied volatility of European option prices.

    Solves BS(S, K, T, r, sigma, q, call) = price for sigma. A solution exists
    only for prices strictly inside the no-arbitrage bounds

    call: max(S e^{-qT} - K e^{-rT}, 0) < price < S e^{-qT}
    put:  max(K e^{-rT} - S e^{-qT}, 0) < price < K e^{-rT}

    Args:
        price: observed option prices.
        S, K, T, r, q, call: as in :func:`volkit.pricing.bs_price`.
        xtol: stop when sigma is known to within xtol, i.e. when the Newton
            step |BS(sigma) - price| / vega or the bracket width is <= xtol.
            A tolerance on sigma rather than on price matters for options
            with tiny vega, where a small price error is a large sigma error.
        max_iter: maximum number of iterations.

    Returns:
        Implied volatilities, broadcast to the common shape of the inputs.
        NaN where the price is outside the bounds or the solver did not
        converge within ``max_iter`` iterations.

    Raises:
        ValueError: if S, K or T is not strictly positive.
    """
    arrays = np.broadcast_arrays(
        np.asarray(price, dtype=np.float64),
        np.asarray(S, dtype=np.float64),
        np.asarray(K, dtype=np.float64),
        np.asarray(T, dtype=np.float64),
        np.asarray(r, dtype=np.float64),
        np.asarray(q, dtype=np.float64),
        np.asarray(call, dtype=np.bool_),
    )
    shape = arrays[0].shape
    price_, S_, K_, T_, r_, q_, call_ = (a.ravel() for a in arrays)
    for name, arr in (("S", S_), ("K", K_), ("T", T_)):
        if np.any(arr <= 0):
            raise ValueError(f"{name} must be strictly positive")

    # No-arbitrage bounds; prices on or outside them have no implied vol.
    disc_s = S_ * np.exp(-q_ * T_)
    disc_k = K_ * np.exp(-r_ * T_)
    lower = np.where(
        call_, np.maximum(disc_s - disc_k, 0.0), np.maximum(disc_k - disc_s, 0.0)
    )
    upper = np.where(call_, disc_s, disc_k)
    valid = (price_ > lower) & (price_ < upper)  # also False for NaN prices

    # Bracket: BS(lo) < price < BS(hi). At sigma -> 0, BS -> lower < price, so
    # lo = 0 works. The upper end is unknown at first (inf) and found by
    # doubling sigma.
    lo = np.zeros_like(price_)
    hi = np.full_like(price_, np.inf)

    # Start at the inflection point of BS(sigma), sqrt(2 |ln(F/K)| / T)
    # (Manaster & Koehler, 1982), floored at 10% so ATM options do not start
    # at sigma = 0.
    log_fk = np.log(S_ / K_) + (r_ - q_) * T_
    sigma = np.maximum(np.sqrt(2.0 * np.abs(log_fk) / T_), 0.1)

    converged = np.zeros_like(valid)
    active = valid.copy()
    for _ in range(max_iter):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        s = sigma[idx]
        args = (S_[idx], K_[idx], T_[idx], r_[idx])
        diff = bs_price(*args, s, q_[idx], call_[idx]) - price_[idx]
        v = vega(*args, s, q_[idx])

        # Shrink the bracket using the sign of the pricing error.
        too_low = diff < 0
        lo[idx] = np.where(too_low, s, lo[idx])
        hi[idx] = np.where(too_low, hi[idx], s)

        with np.errstate(divide="ignore", invalid="ignore"):
            step = diff / v
        newton = s - step
        bisect = np.where(np.isfinite(hi[idx]), 0.5 * (lo[idx] + hi[idx]), 2.0 * s)
        use_newton = (v > _MIN_VEGA) & (newton > lo[idx]) & (newton < hi[idx])
        new_sigma = np.where(use_newton, newton, bisect)

        # Converged: sigma pinned down to xtol by the Newton step or bracket.
        done = (diff == 0) | (use_newton & (np.abs(step) <= xtol))
        done |= hi[idx] - lo[idx] <= xtol
        converged[idx[done]] = True
        sigma[idx] = np.where(done & ~use_newton, s, new_sigma)

        # Give up if the upper bracket has grown past any plausible vol.
        active[idx[done | (new_sigma > _MAX_SIGMA)]] = False

    out: FloatArray = np.where(converged, sigma, np.nan).reshape(shape)
    return out
