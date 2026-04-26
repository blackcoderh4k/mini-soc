.PHONY: help dev server test lint format typecheck clean docker docker-run

# Default target
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Development
# ---------------------------------------------------------------------------

dev: ## Install dependencies (including dev tools)
	pip install -e ".[dev]"

server: ## Start the environment server (dev mode with auto-reload)
	uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

test: ## Run test suite
	python -m pytest tests/ -v

test-cov: ## Run tests with coverage report
	python -m pytest tests/ --cov --cov-report=term-missing --cov-report=html

lint: ## Run linter (ruff check)
	ruff check .

format: ## Auto-format code (ruff format)
	ruff format .
	ruff check --fix .

typecheck: ## Run type checker (mypy)
	mypy models.py client.py server/

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

agent: ## Run the deterministic expert agent
	python run_agent.py

inference: ## Run the LLM baseline inference
	python inference.py

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

docker: ## Build the Docker image
	docker build -t mini-soc -f server/Dockerfile .

docker-run: ## Run the environment in Docker
	docker run -p 8000:8000 mini-soc

docker-up: ## Start environment + agent via docker compose
	docker compose up --build

docker-test: ## Run tests in Docker
	docker compose --profile test run soc-tests

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ htmlcov/ .coverage
