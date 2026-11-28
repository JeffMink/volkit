"""Raw SVI fit per expiry and static-arbitrage checks.

Raw SVI (Gatheral, 2004) writes the total implied variance w = sigma_BS^2 T of
one expiry as a function of log-moneyness k = ln(K / F):

    w(k) = a + b [rho (k - m) + sqrt((k - m)^2 + sigma^2)]

with a level a, wing slope b >= 0, skew |rho| < 1, shift m and curvature
sigma > 0. The wings are asymptotically linear with slopes b (1 - rho) on the
left and b (1 + rho) on the right.

Fit (quasi-explicit, Zeliade 2009 / De Marco and Martini). With
y = (k - m) / sigma,

    w = a + d y + c sqrt(y^2 + 1),    c = b sigma,  d = rho b sigma,

which is linear in (a, d, c) once (m, sigma) are fixed. So each slice is fit
in two levels:

* inner: least squares in (a, d, c) for fixed (m, sigma) under the constraints
  |d| <= c                      (b >= 0, |rho| <= 1),
  c + |d| <= 2 sigma            (Lee's moment bound b (1 + |rho|) <= 2),
  a + sqrt(c^2 - d^2) >= 0      (minimum variance a + b sigma sqrt(1 - rho^2)).
  The feasible set is convex, so the minimum is global. The two linear
  constraints form a diamond in the (d, c) plane; the QP over it is solved
  exactly by enumerating active sets. If that solution breaks the minimum-
  variance condition, SLSQP solves the full problem instead.
* outer: (m, sigma) over a 15 x 15 grid (all 225 inner problems solved in one
  numpy batch), then bounded Nelder-Mead from the best grid point. m is kept
  inside the quoted k range.

Static arbitrage (Gatheral and Jacquier, 2014):

* butterfly: the risk-neutral density is non-negative if and only if
      g(k) = (1 - k w' / (2 w))^2 - (w'^2 / 4) (1 / w + 1 / 4) + w'' / 2 >= 0;
* calendar: w(k, T) must be non-decreasing in T at every k.

The checks are reported, not enforced in the fit, both on a wide fixed
k-grid and on the quoted range of each slice.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

FloatArray = NDArray[np.float64]

PARAM_NAMES = ("a", "b", "rho", "m", "sigma")

SIGMA_BOUNDS = (1e-3, 1.0)
GRID_SIZE = 15
XATOL = 1e-8
FATOL_PER_POINT = 1e-14  # objective is in units of mean(w)^2

CHECK_GRID = np.linspace(-1.5, 1.5, 1001)
ARB_TOL = 1e-10

_FEAS_TOL = 1e-12


@dataclass(frozen=True)
class SVIParams:
    """Raw SVI parameters of one expiry.

    w(k) = a + b [rho (k - m) + sqrt((k - m)^2 + sigma^2)]
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float


@dataclass(frozen=True)
class SVIFit:
    """Result of :func:`fit_svi`.

    Attributes:
        params: fitted parameters.
        rmse: root-mean-square error in total variance, sqrt(mean((w_fit - w)^2)).
        n_points: number of points fitted.
        success: whether Nelder-Mead reported convergence.
    """

    params: SVIParams
    rmse: float
    n_points: int
    success: bool


def svi_total_variance(k: ArrayLike, p: SVIParams) -> FloatArray:
    """Total implied variance w(k) = a + b [rho (k - m) + sqrt((k - m)^2 + sigma^2)]."""
    x = np.asarray(k, dtype=float) - p.m
    out: FloatArray = p.a + p.b * (p.rho * x + np.sqrt(x * x + p.sigma**2))
    return out


def svi_derivatives(
    k: ArrayLike, p: SVIParams
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """w(k) and its first two derivatives in k.

    With x = k - m and s = sqrt(x^2 + sigma^2):

        w   = a + b (rho x + s)
        w'  = b (rho + x / s)
        w'' = b sigma^2 / s^3
    """
    x = np.asarray(k, dtype=float) - p.m
    s = np.sqrt(x * x + p.sigma**2)
    w = p.a + p.b * (p.rho * x + s)
    w1 = p.b * (p.rho + x / s)
    w2 = p.b * p.sigma**2 / s**3
    return w, w1, w2


def butterfly_g(k: ArrayLike, p: SVIParams) -> FloatArray:
    """Gatheral-Jacquier density function g(k); no butterfly arbitrage iff g >= 0.

        g(k) = (1 - k w' / (2 w))^2 - (w'^2 / 4) (1 / w + 1 / 4) + w'' / 2

    The risk-neutral density of k is g(k) / sqrt(2 pi w) exp(-d2^2 / 2) with
    d2 = -k / sqrt(w) - sqrt(w) / 2. Where w <= 0 there is no valid smile and
    g is returned as -inf.
    """
    kk = np.asarray(k, dtype=float)
    w, w1, w2 = svi_derivatives(kk, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (
            (1.0 - kk * w1 / (2.0 * w)) ** 2
            - 0.25 * w1**2 * (1.0 / w + 0.25)
            + 0.5 * w2
        )
    out: FloatArray = np.where(w > 0, g, -np.inf)
    return out


# Linear constraints G x <= h on x = (a, d, c): a diamond in the (d, c) plane,
#   d - c <= 0,  -d - c <= 0,  c + d <= 2 sigma,  c - d <= 2 sigma.
_G = np.array([[0.0, 1.0, -1.0], [0.0, -1.0, -1.0], [0.0, 1.0, 1.0], [0.0, -1.0, 1.0]])
# Every possible active set: none, one edge, or one of the four vertices
# (opposite edges are parallel, so they are never active together).
_ACTIVE_SETS = ((), (0,), (1,), (2,), (3,), (0, 1), (0, 2), (1, 3), (2, 3))


def _solve_inner(
    Y: FloatArray, w: FloatArray, sigma_s: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Constrained least squares in x = (a, d, c) for fixed (m, sigma), batched.

    Row g of ``Y`` holds y = (k - m_g) / sigma_g for one candidate (m, sigma).
    For each row, minimises ||A x - w||^2 with A = [1, y, sqrt(y^2 + 1)]
    subject to |d| <= c, c + |d| <= 2 sigma_s and c^2 - d^2 - min(a, 0)^2 >= 0.
    The last one, given |d| <= c, is the minimum-variance condition
    a + sqrt(c^2 - d^2) >= 0 written without the square root, so its gradient
    is smooth. ``w`` and ``sigma_s`` are in scaled units (see :func:`fit_svi`).

    The linear part is solved exactly: for each candidate active set E x = f
    the KKT system [[A'A, E'], [E, 0]] [x; lambda] = [A'w; f] gives the
    equality-constrained minimiser. The true optimum is the minimiser for its
    own active set, so the best feasible candidate is optimal. Only rows where
    it breaks the minimum-variance condition use SLSQP, followed by an exact
    projection onto the feasible set.

    Args:
        Y: shape (n_candidates, n_points).
        w: shape (n_points,).
        sigma_s: shape (n_candidates,).

    Returns:
        x of shape (n_candidates, 3) and the sums of squared residuals,
        shape (n_candidates,).
    """
    n_cand = Y.shape[0]
    A = np.stack([np.ones_like(Y), Y, np.sqrt(Y * Y + 1.0)], axis=-1)  # (G, n, 3)
    AtA = np.einsum("gni,gnj->gij", A, A)
    Atw = np.einsum("gni,n->gi", A, w)
    h = np.zeros((n_cand, 4))
    h[:, 2:] = 2.0 * sigma_s[:, None]
    tol = _FEAS_TOL * np.maximum(1.0, sigma_s)

    best_x = np.zeros((n_cand, 3))
    best_f = np.full(n_cand, np.inf)
    for active in _ACTIVE_SETS:
        idx = list(active)
        n_act = len(idx)
        E = _G[idx]
        kkt = np.zeros((n_cand, 3 + n_act, 3 + n_act))
        kkt[:, :3, :3] = AtA
        kkt[:, :3, 3:] = E.T
        kkt[:, 3:, :3] = E
        rhs = np.concatenate([Atw, h[:, idx]], axis=1)
        sol = np.einsum("gij,gj->gi", np.linalg.pinv(kkt), rhs)
        x = sol[:, :3]
        feasible = np.all(x @ _G.T - h <= tol[:, None], axis=1)
        r = np.einsum("gni,gi->gn", A, x) - w
        obj = np.einsum("gn,gn->g", r, r)
        better = feasible & (obj < best_f)
        best_x[better] = x[better]
        best_f[better] = obj[better]

    a, d, c = best_x[:, 0], best_x[:, 1], best_x[:, 2]
    broken = a + np.sqrt(np.maximum(c * c - d * d, 0.0)) < -tol
    for g in np.flatnonzero(broken):
        best_x[g] = _solve_min_variance(A[g], w, float(sigma_s[g]))
        r_g = A[g] @ best_x[g] - w
        best_f[g] = r_g @ r_g
    return best_x, best_f


def _solve_min_variance(A: FloatArray, w: FloatArray, sigma_s: float) -> FloatArray:
    """Inner problem with the minimum-variance constraint, by SLSQP.

    Used only when the polyhedral solution has negative minimum variance. The
    SLSQP result is projected exactly onto the feasible set: d into
    [-sigma_s, sigma_s], c into [|d|, 2 sigma_s - |d|], then
    a >= -sqrt(c^2 - d^2).
    """
    # Linear constraints -G x + h >= 0 in one block, with constant Jacobian -G,
    # and the smooth minimum-variance constraint with its analytic gradient.
    h = np.array([0.0, 0.0, 2.0 * sigma_s, 2.0 * sigma_s])
    cons = [
        {"type": "ineq", "fun": lambda x: h - _G @ x, "jac": lambda x: -_G},
        {
            "type": "ineq",
            "fun": lambda x: x[2] ** 2 - x[1] ** 2 - min(x[0], 0.0) ** 2,
            "jac": lambda x: np.array([-2.0 * min(x[0], 0.0), -2.0 * x[1], 2.0 * x[2]]),
        },
    ]
    x0 = np.array([max(float(np.mean(w)), 0.0), 0.0, 0.0])  # feasible: flat smile
    res = minimize(
        lambda x: float(np.sum((A @ x - w) ** 2)),
        x0,
        jac=lambda x: 2.0 * A.T @ (A @ x - w),
        method="SLSQP",
        constraints=cons,
        options={"ftol": 1e-15, "maxiter": 200},
    )
    a, d, c = (float(v) for v in res.x)
    d = float(np.clip(d, -sigma_s, sigma_s))
    c = float(np.clip(c, abs(d), 2.0 * sigma_s - abs(d)))
    a = max(a, -float(np.sqrt(c * c - d * d)))
    return np.array([a, d, c])


def fit_svi(k: ArrayLike, w: ArrayLike) -> SVIFit:
    """Fit raw SVI to one expiry's (k, w) points by the quasi-explicit method.

    Minimises sum_i (w_SVI(k_i) - w_i)^2 (equal weights). See the module
    docstring for the method and constraints. Internally w is divided by its
    mean, so the tolerances do not depend on the variance level.

    Search bounds: m in [k_min, k_max], the quoted range, so the vertex of the
    smile sits on observed data (with m outside it, rho is poorly identified
    and the fit extrapolates one wing); sigma in ``SIGMA_BOUNDS``. The
    15 x 15 grid is evaluated in one batch. Nelder-Mead stops at xatol = 1e-8
    and fatol = 1e-14 n in scaled units, i.e. 1e-14 n mean(w)^2.

    Args:
        k: log-moneyness ln(K / F), at least 5 points.
        w: total implied variance at each k, positive.

    Raises:
        ValueError: on mismatched shapes, fewer than 5 points or w <= 0.
    """
    kk = np.asarray(k, dtype=float).ravel()
    ww = np.asarray(w, dtype=float).ravel()
    if kk.shape != ww.shape:
        raise ValueError("k and w must have the same length")
    if kk.size < 5:
        raise ValueError("need at least 5 points to fit 5 SVI parameters")
    if np.any(ww <= 0):
        raise ValueError("w must be positive")

    scale = float(np.mean(ww))
    ws = ww / scale
    m_lo, m_hi = float(kk.min()), float(kk.max())
    s_lo, s_hi = SIGMA_BOUNDS

    def solve(m: FloatArray, sigma: FloatArray) -> tuple[FloatArray, FloatArray]:
        Y = (kk[None, :] - m[:, None]) / sigma[:, None]
        return _solve_inner(Y, ws, sigma / scale)

    mg, sg = np.meshgrid(
        np.linspace(m_lo, m_hi, GRID_SIZE), np.geomspace(s_lo, s_hi, GRID_SIZE)
    )
    _, f_grid = solve(mg.ravel(), sg.ravel())
    i = int(np.argmin(f_grid))
    best_z, best_f = np.array([mg.ravel()[i], sg.ravel()[i]]), float(f_grid[i])

    def objective(z: FloatArray) -> float:
        return float(solve(z[:1], z[1:])[1][0])

    res = minimize(
        objective,
        best_z,
        method="Nelder-Mead",
        bounds=[(m_lo, m_hi), (s_lo, s_hi)],
        options={
            "xatol": XATOL,
            "fatol": FATOL_PER_POINT * kk.size,
            "maxiter": 2000,
        },
    )
    z = np.asarray(res.x, dtype=float) if res.fun <= best_f else best_z
    x_s, _ = solve(z[:1], z[1:])
    m, sigma = float(z[0]), float(z[1])

    a, d, c = (float(v) * scale for v in x_s[0])
    b = c / sigma
    rho = float(np.clip(d / c, -1.0, 1.0)) if c > 0 else 0.0
    params = SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)
    resid = svi_total_variance(kk, params) - ww
    return SVIFit(
        params=params,
        rmse=float(np.sqrt(np.mean(resid**2))),
        n_points=int(kk.size),
        success=bool(res.success),
    )


def fit_svi_surface(raw: pd.DataFrame) -> pd.DataFrame:
    """Fit raw SVI to every expiry of a raw surface.

    Args:
        raw: output of :func:`volkit.surface.raw.raw_surface`, with columns
            ``expiry``, ``T``, ``k``, ``w`` and ``iv``.

    Returns:
        One row per expiry, sorted by T, with columns ``expiry``, ``T``,
        ``a``, ``b``, ``rho``, ``m``, ``sigma``, ``k_min`` and ``k_max`` (the
        quoted range), ``n_points``, ``rmse_w`` (RMSE in total variance),
        ``rmse_iv`` (RMSE in implied vol, sqrt(w_fit / T) - iv) and
        ``success``.
    """
    rows = []
    for expiry, g in raw.groupby("expiry", sort=False):
        T = float(g["T"].iloc[0])
        k = g["k"].to_numpy()
        fit = fit_svi(k, g["w"].to_numpy())
        iv_fit = np.sqrt(svi_total_variance(k, fit.params) / T)
        rows.append(
            {
                "expiry": expiry,
                "T": T,
                **{n: getattr(fit.params, n) for n in PARAM_NAMES},
                "k_min": float(k.min()),
                "k_max": float(k.max()),
                "n_points": fit.n_points,
                "rmse_w": fit.rmse,
                "rmse_iv": float(np.sqrt(np.mean((iv_fit - g["iv"].to_numpy()) ** 2))),
                "success": fit.success,
            }
        )
    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)


def params_from_row(row: pd.Series) -> SVIParams:
    """SVIParams from a row of :func:`fit_svi_surface` output."""
    return SVIParams(**{n: float(row[n]) for n in PARAM_NAMES})


def _argmin(values: FloatArray, grid: FloatArray) -> tuple[float, float]:
    """(min value, grid point where it occurs)."""
    i = int(np.argmin(values))
    return float(values[i]), float(grid[i])


def check_arbitrage(
    fits: pd.DataFrame,
    k: ArrayLike | None = None,
    *,
    tol: float = ARB_TOL,
    n_data: int = 501,
) -> pd.DataFrame:
    """Check fitted SVI slices for butterfly and calendar arbitrage.

    Butterfly: min of g(k) (see :func:`butterfly_g`); a violation if it is
    below -tol. Calendar: for each slice after the first (sorted by T), min of
    w_T(k) - w_prev(k); a violation if it is below -tol.

    Each check runs on two grids:

    * the wide grid ``k`` (default ``CHECK_GRID``, 1,001 points on
      [-1.5, 1.5]), the same for every slice; this includes extrapolated
      wings;
    * the quoted range, ``n_data`` points on [k_min, k_max] of the slice (for
      calendar: on the overlap of the two slices' ranges). Only if ``fits``
      has ``k_min`` and ``k_max`` columns, as :func:`fit_svi_surface` output
      does.

    Args:
        fits: output of :func:`fit_svi_surface`.
        k: wide grid of log-moneyness.
        tol: tolerance for both checks.
        n_data: number of points on each quoted-range grid.

    Returns:
        ``fits`` sorted by T with extra columns ``min_g``, ``k_min_g``,
        ``butterfly_ok``, ``calendar_gap``, ``k_calendar`` (NaN for the first
        slice) and ``calendar_ok`` for the wide grid, plus the same with
        suffix ``_data`` for the quoted range (calendar NaN, and ok, where the
        ranges do not overlap).
    """
    grid = CHECK_GRID if k is None else np.asarray(k, dtype=float)
    out = fits.sort_values("T").reset_index(drop=True).copy()
    has_range = {"k_min", "k_max"} <= set(out.columns)
    nan2 = (np.nan, np.nan)
    wide_g, wide_cal, data_g, data_cal = [], [], [], []
    prev: tuple[SVIParams, float, float] | None = None
    for _, row in out.iterrows():
        p = params_from_row(row)
        wide_g.append(_argmin(butterfly_g(grid, p), grid))
        if prev is None:
            wide_cal.append(nan2)
        else:
            diff = svi_total_variance(grid, p) - svi_total_variance(grid, prev[0])
            wide_cal.append(_argmin(diff, grid))
        lo = float(row["k_min"]) if has_range else np.nan
        hi = float(row["k_max"]) if has_range else np.nan
        if has_range:
            kd = np.linspace(lo, hi, n_data)
            data_g.append(_argmin(butterfly_g(kd, p), kd))
            if prev is None or max(lo, prev[1]) >= min(hi, prev[2]):
                data_cal.append(nan2)
            else:
                ko = np.linspace(max(lo, prev[1]), min(hi, prev[2]), n_data)
                diff = svi_total_variance(ko, p) - svi_total_variance(ko, prev[0])
                data_cal.append(_argmin(diff, ko))
        prev = (p, lo, hi)

    blocks = [("", wide_g, wide_cal)]
    if has_range:
        blocks.append(("_data", data_g, data_cal))
    for sfx, gs, cals in blocks:
        out["min_g" + sfx] = [v for v, _ in gs]
        out["k_min_g" + sfx] = [x for _, x in gs]
        out["butterfly_ok" + sfx] = out["min_g" + sfx] >= -tol
        out["calendar_gap" + sfx] = [v for v, _ in cals]
        out["k_calendar" + sfx] = [x for _, x in cals]
        gap = out["calendar_gap" + sfx]
        out["calendar_ok" + sfx] = gap.isna() | (gap >= -tol)
    return out
