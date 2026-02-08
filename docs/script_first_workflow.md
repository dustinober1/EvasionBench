# Script-First Workflow

This project uses a script-first contract so local development and CI run through the same command surface.

## Canonical command flow

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make ci-check
python scripts/download_data.py
```

## Command ownership

- `scripts/`: executable workflow entrypoints
- `src/`: implementation modules imported by scripts
- `tests/`: checks for script and module behavior
- `docs/`: workflow and policy documentation

## CI parity

CI executes:

```bash
bash scripts/ci_check.sh
```

This runs:

1. `python scripts/verify_repo_structure.py`
2. `black --check .`
3. `pytest -q`

Local parity command:

```bash
make ci-check
```

## Notebook migration policy

- Notebooks in `notebooks/` are legacy/reference during migration.
- Do not add new production logic to notebooks.
- When notebook logic is useful, move it into `src/` and expose it via `scripts/`.
- Keep notebook content reproducible by referencing script outputs rather than ad-hoc execution.

## When adding new work

- Add logic in `src/`.
- Add an executable command in `scripts/`.
- Add tests in `tests/`.
- Update docs if the command surface or boundaries change.
