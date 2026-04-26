# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-04-21

### Added
- OpenEnv-compliant project structure (`models.py`, `client.py`, `__init__.py` at root)
- `MiniSocEnv` typed HTTP client for agent interaction
- `create_app()` factory pattern in `server/app.py`
- Structured logging via `server/logging_config.py`
- Global unhandled exception handler
- Production tooling: `Makefile`, `ruff`, `mypy`, `pytest-cov`
- `docs/` directory with API reference and architecture diagram
- `.env.example` template for environment variables
- `conftest.py` with shared test fixtures
- `py.typed` PEP 561 marker
- `outputs/` directory for runtime logs and evaluations
- `CHANGELOG.md`, `LICENSE`

### Changed
- Restructured from `app/` package to OpenEnv standard layout
- `server/app.py` — rewritten with factory pattern and structured logging
- `server/mini_soc_environment.py` — dual-import pattern for Docker compatibility
- `Dockerfile` moved into `server/` with updated COPY paths
- `docker-compose.yml` updated for new Dockerfile location
- `pyproject.toml` — full production config (classifiers, dev deps, tool configs)
- Tests moved to `tests/` directory with shared fixtures

### Removed
- `app/` package (dissolved into root + server/)
- `full_test.py` (superseded by pytest)
- `full_test_output.txt`, `pytest_output.txt` (build artifacts)
- `uv.lock` (unnecessary 562KB file)
