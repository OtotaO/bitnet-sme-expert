# dspy-sme-expert — Makefile (uv-based)
.PHONY: help install dev test test-eval lint format check build run optimize-math optimize-code optimize-general bitnet-setup bitnet-serve clean

help: ## Show this help message
	@echo "dspy-sme-expert — Development Commands"
	@echo "======================================"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install all dependencies (dev + extras) via uv
	uv sync --all-extras
	uv run pre-commit install || true

dev: ## Run the FastAPI server with hot reload
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

run: ## Run the server (production-style, no reload)
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

test: ## Run unit tests (no LM calls)
	uv run pytest tests --ignore=tests/eval -v

test-eval: ## Run DSPy evaluation tests (requires provider API keys)
	RUN_EVAL=1 uv run pytest tests/eval -v -s

lint: ## Ruff + mypy
	uv run ruff check app tests scripts
	uv run ruff format --check app tests scripts
	uv run mypy app

format: ## Format with ruff
	uv run ruff format app tests scripts
	uv run ruff check --fix app tests scripts

check: lint test ## All checks

build: ## Build the production Docker image
	docker build --target production -t dspy-sme-expert:latest .

# DSPy optimizers ----------------------------------------------------------
optimize-math: ## Compile the math program (MIPROv2 light)
	uv run python scripts/optimize.py --domain math --optimizer miprov2 --auto light

optimize-code: ## Compile the code program
	uv run python scripts/optimize.py --domain code --optimizer miprov2 --auto light

optimize-general: ## Compile the general program
	uv run python scripts/optimize.py --domain general --optimizer miprov2 --auto light

optimize-math-gepa: ## Compile math with GEPA (slower, often better)
	uv run python scripts/optimize.py --domain math --optimizer gepa

# bitnet.cpp ---------------------------------------------------------------
bitnet-setup: ## Build bitnet.cpp + download the b1.58 2B 4T model (~3GB)
	./scripts/bitnet_setup.sh

bitnet-serve: ## Run the local bitnet.cpp llama-server on :8080
	./scripts/bitnet_serve.sh

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov dist build *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
