# GitHub Actions Workflows

This directory contains CI/CD workflows for automated quality assurance.

## Workflows

### `ci.yml` - Main CI/CD Pipeline

Comprehensive pipeline that runs on every push and pull request:

1. **Lint Job**: Code formatting and quality checks
   - Black (code formatting)
   - isort (import sorting)
   - Flake8 (code quality)
   - Pylint (static analysis)

2. **Type Check Job**: Static type checking
   - MyPy (type validation)

3. **Test Job**: Unit and integration tests
   - Runs on Python 3.10, 3.11, 3.12
   - Coverage reporting (minimum 70%)
   - Codecov integration

4. **Security Job**: Security scanning
   - Safety (dependency vulnerabilities)
   - Bandit (security linter)

5. **Build Job**: Final verification
   - Import verification
   - Build status confirmation

### `unittests.yml` - Dedicated Test Workflow

Focused test workflow that runs:
- Unit tests across multiple Python versions
- Integration tests
- Coverage reporting and threshold enforcement
- Artifact uploads

## Workflow Triggers

- **Push**: `main`, `develop`, `task-*` branches
- **Pull Request**: To `main` or `develop`
- **Manual**: Via `workflow_dispatch` (for `ci.yml`)

## Status Badges

Add these to your README:

```markdown
![CI/CD Pipeline](https://github.com/your-org/financial-fraud-detection/workflows/CI/CD%20Pipeline/badge.svg)
![Tests](https://github.com/your-org/financial-fraud-detection/workflows/Unit%20Tests/badge.svg)
```

## Local Testing

Run checks locally before pushing:

```bash
# Run all quality checks
./scripts/check_code_quality.sh

# Or individually
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
pytest tests/ --cov=src
```

## Configuration Files

- **`.flake8`**: Flake8 linting rules
- **`.pylintrc`**: Pylint configuration
- **`pyproject.toml`**: Black, isort, mypy, pytest, coverage settings
- **`.pre-commit-config.yaml`**: Pre-commit hooks (optional)

## Troubleshooting

### Workflow Fails on Linting

```bash
# Auto-fix formatting
black src/ tests/
isort src/ tests/
```

### Workflow Fails on Tests

```bash
# Run tests locally
pytest tests/ -v

# Check coverage
pytest tests/ --cov=src --cov-report=term-missing
```

### Workflow Fails on Type Checking

```bash
# Run type checker
mypy src/ --ignore-missing-imports
```
