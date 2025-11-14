.PHONY: install api ui dev fmt test compose-up compose-down

install:
	pip install -U pip
	pip install -e ".[all]"

api:
	uvicorn sacp_suite.api.main:app --reload --host ${SACP_BIND:-127.0.0.1} --port ${SACP_PORT:-8000}

ui:
	python -m sacp_suite.ui.app

dev: install api

fmt:
	python -m black src tests
	python -m isort src tests

test:
	pytest -q

compose-up:
	docker compose up --build

compose-down:
	docker compose down -v
