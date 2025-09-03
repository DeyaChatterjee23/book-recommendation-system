# Book Recommender System Makefile

.PHONY: help dev test lint format clean docker-build docker-run docker-test docs install

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	pip install -r requirements-dev.txt
	pre-commit install

dev: ## Start development environment
	docker-compose up -d postgres redis ollama
	python main.py

test: ## Run tests
	pytest tests/ -v --cov=src/book_recommender --cov-report=html

test-fast: ## Run fast tests only
	pytest tests/ -v -m "not slow"

lint: ## Run code linting
	flake8 src/ tests/
	mypy src/
	black --check src/ tests/
	isort --check-only src/ tests/

format: ## Format code
	black src/ tests/
	isort src/ tests/

clean: ## Clean up generated files
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

docker-build: ## Build Docker image
	docker build -t book-recommender:latest .

docker-run: ## Run with Docker Compose
	docker-compose up -d

docker-test: ## Run tests in Docker
	docker-compose run --rm app pytest tests/

docker-stop: ## Stop Docker services
	docker-compose down

docs: ## Generate documentation
	mkdocs build

docs-serve: ## Serve documentation locally
	mkdocs serve

data-pipeline: ## Run data pipeline
	python scripts/data_pipeline.py --download-sample --split-data

train-models: ## Train all models
	python scripts/train_models.py --model all

train-models-fast: ## Train models without Ollama
	python scripts/train_models.py --model all --no-ollama

setup-dev: ## Set up development environment
	python -m venv venv
	./venv/bin/pip install -r requirements-dev.txt
	cp .env.example .env
	docker-compose up -d postgres redis ollama
	python scripts/data_pipeline.py --download-sample
	python scripts/train_models.py --model all --no-ollama

security-scan: ## Run security scans
	safety check
	bandit -r src/

benchmark: ## Run performance benchmarks
	python -m pytest tests/performance/ -v

requirements: ## Update requirements files
	pip-compile requirements.in
	pip-compile requirements-dev.in

check: lint test ## Run all checks

all: clean install lint test docker-build ## Run complete build pipeline
