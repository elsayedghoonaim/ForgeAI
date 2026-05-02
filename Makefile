.PHONY: install dev test lint format docker clean help bootstrap-wsl smoke-api

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	python -m pip install --upgrade pip setuptools wheel
	python -m pip install --no-build-isolation -e .

dev: ## Install with development dependencies
	python -m pip install --upgrade pip setuptools wheel
	python -m pip install --no-build-isolation -e ".[dev]"

test: ## Run test suite
	python -m pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage
	python -m pytest tests/ -v --cov=forgeai --cov-report=term-missing

lint: ## Run linter
	python -m ruff check src/ tests/

format: ## Auto-format code
	python -m ruff format src/ tests/

typecheck: ## Run type checker
	python -m mypy src/forgeai/

docker: ## Build Docker image
	docker build -t forgeai:latest .

docker-run: ## Run Docker container with GPU
	docker run --gpus all -p 8000:8000 forgeai:latest

bootstrap-wsl: ## Create a WSL virtualenv, install dependencies, and run tests
	bash scripts/bootstrap_wsl.sh

smoke-api: ## Run a simple API smoke test against a running server
	bash scripts/smoke_api.sh

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
