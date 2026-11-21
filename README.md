# volkit

Option pricing and implied volatility surfaces in Python.

volkit prices European options with closed-form Black-Scholes, Monte Carlo and
binomial trees, solves for implied volatility, and fits arbitrage-checked SVI
surfaces to market option data.

Work in progress: see [docs/PROGRESS.md](docs/PROGRESS.md) for status.

## Install

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Development

```bash
uv run pytest                                  # tests
uv run ruff check . && uv run mypy             # lint and types
uv run ruff format .                           # format
```

## License

MIT, see [LICENSE](LICENSE).
