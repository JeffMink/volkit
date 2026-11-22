# Progress

## Status
Session 2 (Monte Carlo, binomial tree, American exercise) complete: merged to
`main` in PR #3 with CI green (129 tests). Session 3 not started.

## Checklist
- [x] 0 Setup (repo, CI, CLAUDE.md)
- [x] 1 Black-Scholes, Greeks, IV solver
- [x] 2 Monte Carlo and binomial tree (+ American exercise)
- [ ] 3 Market data and raw IV surface
- [ ] 4 SVI fitting and arbitrage checks
- [ ] 5 Plotly visuals and Streamlit app
- [ ] 6 README, deploy, v1.0

## Done
- Session 0: GitHub repo `JeffMink/volkit`; uv library layout under
  `src/volkit/` with empty `pricing`, `iv`, `surface`, `viz` subpackages;
  ruff, mypy and pytest configured in `pyproject.toml`; GitHub Actions CI
  (`.github/workflows/ci.yml`); short README; MIT license; `docs/theory.md`
  stub. Merged in PR #1.
- Session 1:
  - `volkit.pricing.bs_price`: Black-Scholes-Merton with dividend yield q,
    vectorised with numpy broadcasting, mixed calls and puts in one call.
  - `volkit.pricing` Greeks: delta, gamma, vega, theta, rho (closed form).
  - `volkit.iv.implied_vol`: vectorised Newton-Raphson with bisection
    fallback, NaN outside the no-arbitrage bounds.
  - Tests: Hull textbook example (call 4.7594, put 0.8086 vs 4.76 / 0.81),
    put-call parity on 10,000 random points, vol limits, Greeks against
    central finite differences (largest gap about 1e-6), Black-Scholes PDE
    identity, IV round trip on an 8,200-point grid per option type.
  - `docs/theory.md`: Black-Scholes, Greeks and implied volatility sections.
- Session 2:
  - `volkit.pricing.mc_price`: European Monte Carlo with exact one-step
    lognormal sampling, antithetic variates and a control variate
    (Y = e^{-rT} S_T, known mean S e^{-qT}); returns `MCResult(price, stderr,
    n_paths)`.
  - `volkit.pricing.binomial_price`: CRR tree, European or American, O(n)
    memory backward induction.
  - Tests: MC within 4 standard errors of Black-Scholes on 5 cases x calls/puts
    x all variance-reduction settings; stderr ratio about 2 when n is
    quadrupled; each technique lowers stderr. ATM call, 100,000 paths,
    seed 4: stderr 0.0465 plain, 0.0327 antithetic, 0.0177 control variate,
    0.0086 both. Tree: matches an independent loop-based tree; n * |error vs
    BS| < 3 for n = 100..1600 (observed up to about 2); ATM error halves when
    n doubles; exact put-call parity; American call = European when q = 0;
    American put >= European and >= intrinsic; deep ITM American put = K - S;
    American call > European with q > 0; Hull reference American put
    (RMFI 4th ed., Appendix F, Figure F.1): 4.49 / 4.263 / 4.272 / 4.278
    for 5 / 30 / 50 / 100 steps, all matched to the printed digits.
  - `docs/theory.md`: Monte Carlo and binomial tree sections, with American
    exercise.

## Decisions
- Python 3.12 only (`requires-python = ">=3.12"`, CI on 3.12).
- mypy `strict = true` on `src/`.
- scipy has no type information: mypy skips type-checking scipy imports
  (override in `pyproject.toml`) instead of adding `scipy-stubs`.
- Runtime dependencies: numpy, scipy, pandas. plotly and streamlit deferred to
  session 5.
- Dev dependencies: pytest, ruff, mypy, pandas-stubs.
- MIT license.
- The user makes all commits; Claude suggests messages and files to stage.
- Commits use the GitHub no-reply email; no personal email in the repo.
- Pricing includes a continuous dividend yield `q` (default 0).
- Calls and puts are selected with `call: bool | bool array`, so one call can
  price or invert a mix of OTM calls and puts.
- Greeks are raw derivatives: vega per 1.00 vol, theta per year, rho per 1.00
  rate.
- Inputs S, K, T, sigma must be strictly positive (ValueError otherwise); no
  special-casing of T = 0.
- IV solver stops on a tolerance in sigma (`xtol = 1e-12`), not in price. The
  plan said a 1e-10 price tolerance; that left sigma errors up to 7e-5 for
  low-vega options, because sigma error is about price error / vega.
- European options stay the main focus of volkit. American exercise is a
  secondary feature, added only to the binomial tree (see Next steps).
- Market data for the surface (session 3) should be European-style index
  options (e.g. SPX), so Black-Scholes implied vols are exact. American
  single-stock options would bias implied vol upward through the
  early-exercise premium, mostly for ITM puts.
- Monte Carlo and the tree take scalar inputs only: they are reference
  pricers for validation; vectorised `bs_price` is the surface tool.
- MC samples S_T exactly in one step (no Euler scheme); path simulation only
  if path-dependent payoffs are added later.
- With antithetic variates the stderr is computed from pair averages, and
  n_paths must be even.
- No American Monte Carlo (Longstaff-Schwartz); American exercise is tree-only.
- American put reference: Hull, *Risk Management and Financial
  Institutions*, 4th ed., Appendix F (pp. 605-606), checked by the user from
  the book. volkit: 4.4885 (n = 5), 4.2634 (30), 4.2720 (50), 4.2781 (100).
  Test tolerance is half the last printed digit.

## Open issues
- `app/` is not created yet; it is added with its first file in session 5.
- CI uses `actions/checkout@v4` and `astral-sh/setup-uv@v6`; check for newer
  major versions.
- Python installs done from inside the Claude desktop app land in the app's
  private folder. Install Python versions from your own terminal.
- IV solver: low-vega options fall back to bisection and can take up to about
  100 iterations. Fine for thousands of quotes; a better initial guess or
  Jäckel's "Let's Be Rational" would cut this if speed matters later.
- IV accuracy is limited to about 1e-16 * price / vega by floating point; the
  tests check sigma to 1e-7 only where vega > 1e-6.

- CRR error oscillates in n; Richardson extrapolation or averaging n and n+1
  would speed convergence if needed.

## Next steps
1. Session 3: market data (European index options, e.g. SPX) and the raw
   implied volatility surface.
