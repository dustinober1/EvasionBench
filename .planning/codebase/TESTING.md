TESTING
=======

Test framework and CI
---------------------
- Tests run with `pytest` (CI uses `pytest -q` in `.github/workflows/ci.yml`).
- Only one test file found: `tests/test_smoke.py` — indicates lightweight smoke testing present.

Test layout
-----------
- `tests/` directory houses test files; add more unit and integration tests to cover `src/` modules.

Mocks & Dependencies
--------------------
- For model- and data-heavy tests, use fixtures to provide small in-memory datasets or temporary files (pytest fixtures).
- Consider using `pytest-mock` or `unittest.mock` to stub network/IO (e.g., huggingface hub, mlflow remote).

CI test steps
-------------
- CI installs `requirements.txt` then runs `pytest -q`.
- Lint step runs `black --check .` which will fail for unformatted code.

Coverage & metrics
------------------
- No coverage tooling found (no `coverage` config or badge). Add `coverage` or `pytest-cov` to capture coverage reports in CI.

Recommendations
---------------
1. Expand unit tests for `src/models.py`, `src/features.py`, and `src/data.py` with deterministic small fixtures.
2. Add integration tests for `api/main.py` using `httpx` or `starlette.testclient` to exercise endpoints without network.
3. Add `pytest-cov` to CI and upload coverage report as an artifact or badge.
4. Add tests that run `dvc repro` in a sandbox or mock DVC operations for pipeline-level validation.
