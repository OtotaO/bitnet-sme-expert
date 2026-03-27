# BitNet SME Expert System - Development Makefile
.PHONY: help install dev test lint format clean build deploy docs

# Default target
help: ## Show this help message
	@echo "BitNet SME Expert System - Development Commands"
	@echo "=============================================="
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Environment setup
install: ## Install dependencies
	pip install -e .[dev,docs,monitoring]
	pre-commit install

install-dev: ## Install development dependencies only
	pip install -r requirements.txt
	pre-commit install

# Development
dev: ## Start development server with hot reload
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

dev-docker: ## Start development server using Docker Compose
	docker-compose up --build

# Testing
test: ## Run all tests
	pytest tests/ -v --cov=app --cov-report=term-missing --cov-report=html

test-fast: ## Run tests without coverage
	pytest tests/ -v

test-watch: ## Run tests in watch mode
	pytest-watch tests/ app/

# Code quality
lint: ## Run all linting tools
	ruff check app/ tests/
	mypy app/
	black --check app/ tests/
	isort --check-only app/ tests/

format: ## Format code
	black app/ tests/
	isort app/ tests/
	ruff check --fix app/ tests/

check: ## Run all quality checks
	make lint
	make test

# Security
security: ## Run security checks
	bandit -r app/
	safety check
	pip-audit

# Documentation
docs: ## Generate documentation
	mkdocs build

docs-serve: ## Serve documentation locally
	mkdocs serve

docs-deploy: ## Deploy documentation
	mkdocs gh-deploy

# Database
db-init: ## Initialize database
	alembic upgrade head

db-migrate: ## Create new migration
	@read -p "Enter migration message: " msg; \
	alembic revision --autogenerate -m "$$msg"

db-upgrade: ## Apply migrations
	alembic upgrade head

db-downgrade: ## Rollback last migration
	alembic downgrade -1

db-reset: ## Reset database (WARNING: destroys all data)
	@echo "WARNING: This will destroy all database data!"
	@read -p "Are you sure? [y/N]: " confirm && [ "$$confirm" = "y" ] || exit 1
	rm -f bitnet_sme.db
	alembic upgrade head

# Docker
build: ## Build Docker image
	docker build -t bitnet-sme-expert:latest .

build-prod: ## Build production Docker image
	docker build --target production -t bitnet-sme-expert:prod .

# Deployment
deploy-staging: ## Deploy to staging
	@echo "Deploying to staging..."
	# Add your staging deployment commands here

deploy-prod: ## Deploy to production
	@echo "Deploying to production..."
	# Add your production deployment commands here

# Monitoring
metrics: ## Start metrics collection
	docker-compose up prometheus grafana -d

logs: ## View application logs
	docker-compose logs -f api

# Utilities
clean: ## Clean up generated files
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

update: ## Update dependencies
	pip-compile requirements.in
	pip install -r requirements.txt

env: ## Create .env file from template
	cp .env.example .env
	@echo "Please edit .env file with your configuration"

# AI Model management
download-models: ## Download required AI models
	@echo "Downloading models..."
	# Add model download commands here

# Performance testing
perf-test: ## Run performance tests
	@echo "Running performance tests..."
	# Add performance testing commands here

load-test: ## Run load tests
	@echo "Running load tests..."
	# Add load testing commands here

# Release
release: ## Create a new release
	@read -p "Enter version (current: $$(python -c 'from app import __version__; print(__version__)')): " version; \
	git tag -a v$$version -m "Release v$$version"; \
	git push origin v$$version

# Development utilities
shell: ## Start interactive shell with app context
	python -c "from app.main import app; import IPython; IPython.embed()"

debug: ## Start development server in debug mode
	python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Backup and restore
backup: ## Backup database
	@mkdir -p backups
	@cp bitnet_sme.db backups/bitnet_sme_$$(date +%Y%m%d_%H%M%S).db
	@echo "Database backed up to backups/"

restore: ## Restore database from backup
	@ls -1 backups/
	@read -p "Enter backup filename: " backup; \
	cp backups/$$backup bitnet_sme.db