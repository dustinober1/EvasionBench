.PHONY: env install test lint format verify-structure ci-check run-api run-dashboard

env:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

install:
	pip install -r requirements.txt

test:
	pytest -q

lint:
	black --check .

format:
	black .

verify-structure:
	python scripts/verify_repo_structure.py

ci-check:
	bash scripts/ci_check.sh

run-api:
	uvicorn api.main:app --reload --port 8080

run-dashboard:
	streamlit run dashboard/app.py
