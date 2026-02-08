.PHONY: env install test lint run-api run-dashboard

env:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

install:
	pip install -r requirements.txt

test:
	pytest -q

lint:
	black .

run-api:
	uvicorn api.main:app --reload --port 8080

run-dashboard:
	streamlit run dashboard/app.py
