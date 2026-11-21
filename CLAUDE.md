# Project: volkit — Option pricing and implied volatility surface

## Purpose
A Python library that (1) prices options with several methods and (2) builds and
fits implied volatility surfaces from real market data using SVI. Portfolio
project for a mathematical modelling student targeting quantitative finance roles.
Quality bar: correct maths, validated numerics, clean code, great README.

## Stack
Python 3.12, uv, numpy, scipy, pandas, plotly, streamlit, pytest, ruff, mypy.

## Structure
```
src/volkit/pricing/   closed-form, Monte Carlo, binomial tree
src/volkit/iv/        implied vol solver
src/volkit/surface/   data cleaning, SVI fitting, arbitrage checks
src/volkit/viz/       plotly figures
app/                  streamlit app
tests/                pytest, mirrors src/
docs/                 theory.md (derivations), PROGRESS.md (status)
```

## Commands
```
uv sync                                  install
uv run pytest                            tests
uv run ruff check . && uv run mypy src   lint and types
```

## Rules
- Every numerical method must be tested against a closed-form or known value.
- Type hints and docstrings with the formula used on all public functions.
- Never invent results or numbers; compute them and show the output.
- Avoid look-ahead bias and survivorship issues in any data work.
- Prefer clear, readable code over clever code. Vectorise with numpy.
- Ask before adding dependencies or making big design changes.
- Explain the maths briefly when implementing something new; the user must be
  able to explain it in an interview.

## Workflow
- Start of session: read this file and docs/PROGRESS.md, summarise where we are,
  propose the next step. Do not write code until I approve the plan.
- Work on a feature branch with small commits. I make all commits myself; never
  run `git commit` or `git push`. When a piece of work is ready, suggest a
  conventional commit message and tell me which files to stage.
- Run tests and linters before saying work is ready to commit, and show the output.
- End of session: update docs/PROGRESS.md (done, decisions, open issues, next steps).
