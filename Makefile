.PHONY: setup lint test run pipeline compose-up compose-down

setup:
	python -m pip install --upgrade pip
	python -m pip install -e .[dev]

lint:
	ruff check src tests
	ruff format --check src tests
	mypy src

test:
	pytest

run:
	uvicorn mlprg.api.app:app --host 0.0.0.0 --port 8000 --reload

pipeline:
	python -m mlprg.pipelines.run

compose-up:
	docker compose up --build

compose-down:
	docker compose down
