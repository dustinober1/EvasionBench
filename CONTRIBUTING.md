# Contributing

## Workflow expectations

1. Create a branch for your change.
2. Implement production logic in `src/` and script entrypoints in `scripts/`.
3. Add or update tests in `tests/` for behavioral changes.
4. Run `make ci-check` before opening a PR.

## Ownership boundaries

- `src/`: business/research logic
- `scripts/`: orchestration commands and CLI-like entrypoints
- `docs/`: process/runbook documentation
- `notebooks/`: reference artifacts only; do not add new production logic

If logic currently exists only in notebooks, extract it into `src/` and call it from `scripts/`.

## Validation commands

Run the same baseline checks locally that CI runs:

```bash
make ci-check
```

Additional useful commands:

```bash
make verify-structure
make test
make lint
```

## Pull request checklist

- [ ] Tests added or updated for behavior changes
- [ ] `make ci-check` passes locally
- [ ] Documentation updated when command surfaces or boundaries change
- [ ] Changes keep script-first conventions intact

Open an issue before large architectural changes.
