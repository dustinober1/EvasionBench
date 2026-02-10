# Debug Mode Rules

## Project-Specific Debugging Patterns

- **Test discovery**: Tests use `conftest.py` to add project root to `sys.path` - if imports fail in tests, check [`tests/conftest.py`](tests/conftest.py)
- **CI failures**: Run `bash scripts/ci_check.sh` locally to reproduce full CI (structure verify + black + pytest)
- **Artifact validation failures**: Check phase-specific artifact contracts in [`src/analysis/artifacts.py`](src/analysis/artifacts.py) - missing `artifact_index.json` is a common issue
- **Model evaluation errors**: Ensure all 4 required files exist in output directory (see [`src/evaluation.py:17`](src/evaluation.py:17))

## Common Issues

- **Import errors in scripts**: Scripts must set `ROOT = Path(__file__).resolve().parents[1]` and add to `sys.path` before importing `src.*`
- **Stratification failures**: [`src/models.py`](src/models.py) handles single-class edge cases by disabling stratify - check `split_metadata["stratify"]` in outputs
