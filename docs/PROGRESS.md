# Progress

## Status
Session 0 (setup) done locally on branch `chore/setup`: ruff, ruff format, mypy
(strict) and pytest (5 smoke tests) all pass. Waiting for the first CI run on
GitHub before ticking item 0.

## Checklist
- [ ] 0 Setup (repo, CI, CLAUDE.md) — awaiting first green CI run
- [ ] 1 Black-Scholes, Greeks, IV solver
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
  stub.

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

## Open issues
- `app/` and per-module test folders are not created yet; git does not track
  empty folders. They are added with their first files.
- CI uses `actions/checkout@v4` and `astral-sh/setup-uv@v6`; check for newer
  major versions.
- Python installs done from inside the Claude desktop app land in the app's
  private folder. Install Python versions from your own terminal.

## Next steps
1. Merge `chore/setup` once CI is green, then tick item 0.
2. Session 1: Black-Scholes prices, Greeks, and implied volatility solver,
   with tests against known values and put-call parity.
