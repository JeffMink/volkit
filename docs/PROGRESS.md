# Progress

## Status
Session 1 (Black-Scholes, Greeks, IV solver) done locally on branch
`feat/black-scholes`: ruff, ruff format, mypy (strict) and pytest (41 tests) all
pass. Item 1 gets ticked once CI is green on the PR.

## Checklist
- [x] 0 Setup (repo, CI, CLAUDE.md)
- [ ] 1 Black-Scholes, Greeks, IV solver (awaiting CI on the PR)
- [ ] 2 Monte Carlo and binomial tree
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

## Next steps
1. Open a PR for `feat/black-scholes`, merge when CI is green, tick item 1.
2. Session 2: Monte Carlo (with variance reduction) and binomial tree pricers,
   tested for convergence to the Black-Scholes price.
