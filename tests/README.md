# Tests Directory

This directory contains unit and integration tests for the project.

## Structure

```
tests/
├── unit/              # Unit tests for individual modules
├── integration/       # Integration tests for complete pipelines
└── __init__.py
```

## Running Tests

### Run All Tests

```bash
# From project root
pytest tests/ -v
```

### Run Unit Tests Only

```bash
pytest tests/unit/ -v
```

### Run Integration Tests Only

```bash
pytest tests/integration/ -v
```

### Run with Coverage

```bash
pytest tests/ --cov=src --cov-report=html --cov-report=term
```

Coverage report will be generated in `htmlcov/` directory.

## Test Organization

- **Unit Tests**: Test individual functions and classes in isolation
- **Integration Tests**: Test complete pipelines and workflows
- **Fixtures**: Shared test data and configurations

## Writing Tests

Follow pytest conventions:
- Test files: `test_*.py`
- Test functions: `test_*()`
- Use fixtures for shared setup/teardown
- Mock external dependencies

