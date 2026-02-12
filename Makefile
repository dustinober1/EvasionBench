.PHONY: env install test lint format verify-structure ci-check run-api run-dashboard data-fetch data-validate data-prepare run-experiment analysis-phase3 analysis-phase4 model-phase5 label-diagnostics report-phase7 docker-build docker-up docker-down

env:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

install:
	pip install -r requirements.txt

test:
	pytest -q --cov=src --cov-report=term-missing --cov-report=html:artifacts/coverage

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

data-fetch:
	python scripts/download_data.py --output data/raw/evasionbench.parquet --revision main
	python scripts/write_data_manifest.py --data data/raw/evasionbench.parquet --output data/contracts/evasionbench_manifest.json --revision main

data-validate:
	python scripts/validate_dataset.py --data data/raw/evasionbench.parquet --contract data/contracts/evasionbench_manifest.json

data-prepare:
	python scripts/prepare_data.py --input data/raw/evasionbench.parquet --output data/processed/evasionbench_prepared.parquet

run-experiment:
	python scripts/run_experiment.py --tracking-uri file:./mlruns --experiment-name evasionbench-baselines

analysis-phase3:
	python scripts/run_phase3_analyses.py --input data/processed/evasionbench_prepared.parquet --output-root artifacts/analysis/phase3 --families all

analysis-phase4:
	python scripts/run_phase4_analyses.py --input data/processed/evasionbench_prepared.parquet --output-root artifacts/analysis/phase4 --families all

model-phase5:
	python scripts/run_classical_baselines.py --input data/processed/evasionbench_prepared.parquet --output-root artifacts/models/phase5 --families all --compare

model-phase6:
	python scripts/run_transformer_baselines.py --input data/processed/evasionbench_prepared.parquet --output-root artifacts/models/phase6/transformer --max-epochs 3 --learning-rate 2e-5

xai-classical:
	python scripts/run_explainability_analysis.py --data data/processed/evasionbench_prepared.parquet --models-root artifacts/models/phase5 --output-root artifacts/explainability/phase6 --families all

xai-transformer:
	python scripts/run_transformer_explainability.py --model-path artifacts/models/phase6/transformer/model --data-path data/processed/evasionbench_prepared.parquet --output-root artifacts/explainability/phase6/transformer --n-samples 20

xai-all: xai-classical xai-transformer

label-diagnostics:
	python scripts/run_label_diagnostics.py --input data/processed/evasionbench_prepared.parquet --output-root artifacts/diagnostics/phase6 --random-state 42

report-phase7:
	python3 scripts/run_research_pipeline.py --input data/processed/evasionbench_prepared.parquet --output-root artifacts/reports/phase7 --skip-existing

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down
