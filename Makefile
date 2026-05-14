# dspy-sme-expert — Makefile (uv-based)
.PHONY: help install dev test test-eval lint format check build run optimize-math optimize-code optimize-general bitnet-setup bitnet-serve rag-install rag-index rag-up rag-clean modal-deploy modal-serve modal-finetune clean

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

# Agentic Hybrid RAG -------------------------------------------------------
rag-install: ## Install the [rag] extra (lancedb + BGE-M3 + reranker)
	uv sync --extra rag

rag-index: ## Build the LanceDB index from rag/corpus/ (default starter corpus)
	uv run python scripts/build_rag_index.py \
		--source-dir rag/corpus \
		--index-path rag/index

rag-up: rag-install rag-index ## Install [rag] + build the default index in one shot
	@echo ""
	@echo "RAG index ready at rag/index."
	@echo "Run with: RAG_ENABLED=1 RAG_INDEX_PATH=rag/index make dev"

rag-clean: ## Drop the local RAG index (forces a rebuild on next run)
	rm -rf rag/index

# Modal (hosted vLLM + fine-tuning) ----------------------------------------
modal-serve: ## Run an ephemeral vLLM endpoint on Modal (Ctrl-C to stop)
	uv run modal serve scripts/modal_serve.py

modal-deploy: ## Deploy a persistent vLLM endpoint on Modal
	uv run modal deploy scripts/modal_serve.py

modal-finetune: ## Fine-tune a HF base on Modal (override BASE/DATASET/OUTPUT)
	uv run modal run scripts/modal_finetune.py::train \
		--base-model "$(or $(BASE),unsloth/llama-3.1-8b-bnb-4bit)" \
		--dataset-repo "$(DATASET)" \
		--output-repo "$(OUTPUT)"

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov dist build *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
