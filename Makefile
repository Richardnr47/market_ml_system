.PHONY: install install-dev lint test train serve backtest

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

lint:
	ruff check .

test:
	pytest -q

train:
	python scripts/run_train.py

serve:
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000