# Repository Guidelines

## Project Structure & Module Organization

This repository currently contains research planning material for the Ariel Data Challenge 2025. Keep high-level model plans and experiment notes in `plans/`; the current main document is `plans/ariel_2025_model_plan.md`. Agent configuration and local workflow guidance live under `.agents/`, including `.agents/rules/` and `.agents/skills/`.

When implementation begins, use a conventional Python layout:

```text
src/              # reusable pipeline, feature, model, and evaluation code
tests/            # pytest tests mirroring src/ modules
notebooks/        # exploratory analysis only
data/             # local data, ignored by Git
outputs/          # generated models, metrics, plots, and submissions
```

## Build, Test, and Development Commands

There is no runnable package or test suite yet. For documentation-only changes, verify Markdown with your editor preview and keep links and paths valid.

For future Python work, prefer `uv`:

```powershell
uv venv
uv pip install -r requirements.txt
pytest
```

Use `pytest` for tests once `tests/` exists. If notebooks are added, keep production logic in `src/` and import it from notebooks rather than duplicating code.

## Coding Style & Naming Conventions

Use Python 3.11+ for new code. Follow PEP 8 with 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and clear module names such as `feature_extraction.py` or `sigma_calibration.py`. Prefer small, explicit functions for stages described in the plan: calibration, light-curve extraction, feature generation, PCA modeling, and uncertainty calibration.

## Testing Guidelines

Add focused `pytest` coverage for every non-trivial logic change. Name test files `test_<module>.py` and test functions `test_<behavior>()`. Prioritize deterministic tests for numerical shape checks, calibration formulas, feature extraction boundaries, and sigma calibration. Use small synthetic arrays instead of challenge-scale data in unit tests.

## Commit & Pull Request Guidelines

This directory currently has no Git history, so no repository-specific commit convention is available. Use short imperative commit subjects, for example `Add transit depth feature tests`. Pull requests should include a concise summary, validation commands run, affected paths, and screenshots or plots when visual outputs change.

## Security & Configuration Tips

Do not commit `.env`, raw challenge data, generated submissions, model checkpoints, or large output artifacts. Keep machine-specific settings local. Use `.env_example` for shared configuration keys when runtime code is introduced.

## Agent-Specific Instructions

Keep changes narrow and tied to the requested task. For non-trivial code changes, add or update tests first, then implement the smallest passing change.
