.PHONY: install install-dev test lint format clean help

# Default Python interpreter
PYTHON := python3

help:
	@echo "NYC Taxi Fare Prediction - Development Commands"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  test         Run test suite"
	@echo "  lint         Run linters (flake8, black --check, isort --check)"
	@echo "  format       Auto-format code with black and isort"
	@echo "  clean        Remove build artifacts and cache files"
	@echo ""

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	flake8 src tests --max-line-length=88 --extend-ignore=E203
	black --check src tests
	isort --check-only src tests

format:
	black src tests
	isort src tests

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
