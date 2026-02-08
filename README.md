# EvasionBench — Financial NLP Portfolio

EvasionBench is a script-first research repository for earnings-call Q&A evasion detection using the Hugging Face dataset `FutureMa/EvasionBench`.

## Script-first quickstart

1. Create and activate a virtual environment.
2. Install dependencies.
3. Run the baseline checks used by CI.
4. Run project workflows from `scripts/` and `src/` modules.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make ci-check
python scripts/download_data.py
```

## Repository boundaries

- `src/`: reusable project logic and analysis code
- `scripts/`: executable entrypoints for data, analysis, and validation workflows
- `tests/`: unit/smoke/integration test coverage
- `docs/`: runbooks and policy documentation
- `notebooks/`: legacy/reference artifacts only during migration

New production logic should be added to `src/` and exposed through `scripts/`, not implemented directly in notebooks.

## Core developer commands

- `make ci-check`: run structure checks, formatting check, and tests (same path as CI)
- `make verify-structure`: validate required repository boundaries
- `make test`: run test suite
- `make lint`: run formatting check (`black --check .`)
- `make format`: auto-format code with `black`

See `docs/script_first_workflow.md` for end-to-end workflow and migration policy.

License: MIT
