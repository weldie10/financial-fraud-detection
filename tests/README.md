# Tests Directory

This directory contains comprehensive unit and integration tests for the financial fraud detection project, ensuring code correctness, reliability, and regulatory compliance.

## Structure

```
tests/
├── conftest.py              # Shared pytest fixtures and configuration
├── unit/                    # Unit tests for individual modules
│   ├── test_data_loader.py
│   ├── test_data_cleaner.py
│   ├── test_feature_engineer.py
│   ├── test_imbalance_handler.py
│   ├── test_data_transformer.py
│   ├── test_data_preparator.py
│   ├── test_model_trainer.py
│   ├── test_model_evaluator.py
│   ├── test_cross_validator.py
│   └── test_geolocation.py
├── integration/             # Integration tests for complete pipelines
│   ├── test_preprocessing_pipeline.py
│   └── test_model_pipeline.py
└── README.md
```

## Test Coverage

### Unit Tests

**Data Processing Modules:**
- `DataLoader`: CSV loading, validation, error handling
- `DataCleaner`: Missing value handling, duplicate removal, data type correction
- `FeatureEngineer`: Time features, transaction frequency/velocity, time since signup
- `ImbalanceHandler`: SMOTE, undersampling, combined methods, class distribution analysis
- `DataTransformer`: Scaling (Standard/MinMax/Robust), encoding (OneHot), save/load

**Model Training Modules:**
- `DataPreparator`: Stratified train-test split, feature-target separation
- `ModelTrainer`: Baseline (Logistic Regression), ensemble models (RF, XGBoost, LightGBM)
- `ModelEvaluator`: Metrics calculation (PR-AUC, F1-Score, ROC-AUC), plot generation
- `CrossValidator`: Stratified K-Fold cross-validation, results summary

**Geolocation Module:**
- `GeolocationMapper`: IP to integer conversion, country mapping, fraud pattern analysis

### Integration Tests

- **PreprocessingPipeline**: End-to-end data preprocessing for fraud and credit card datasets
- **ModelPipeline**: Complete training workflow (preparation → resampling → transformation → training → evaluation)

## Running Tests

### Prerequisites

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

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

### Run Specific Test File

```bash
pytest tests/unit/test_data_loader.py -v
```

### Run Specific Test Function

```bash
pytest tests/unit/test_data_loader.py::TestDataLoader::test_load_csv_success -v
```

### Run with Coverage Report

```bash
# Terminal report
pytest tests/ --cov=src --cov-report=term-missing

# HTML report (generated in htmlcov/)
pytest tests/ --cov=src --cov-report=html

# XML report (for CI/CD)
pytest tests/ --cov=src --cov-report=xml
```

### Run with Markers

```bash
# Run only unit tests
pytest tests/ -m unit

# Run only integration tests
pytest tests/ -m integration

# Skip slow tests
pytest tests/ -m "not slow"
```

## Test Configuration

Configuration is defined in `pytest.ini` at the project root:

- **Coverage threshold**: Minimum 70% code coverage required
- **Output format**: Verbose with short traceback
- **Warnings**: Deprecation and user warnings are ignored
- **Logging**: CLI logging enabled at INFO level

## Test Organization

### Unit Tests

Unit tests verify individual functions and classes in isolation:
- **Isolation**: Each test is independent
- **Mocking**: External dependencies are mocked where appropriate
- **Fixtures**: Shared test data via `conftest.py`
- **Edge Cases**: Tests cover error handling and boundary conditions

### Integration Tests

Integration tests verify complete workflows:
- **End-to-End**: Test full pipelines from input to output
- **Real Data**: Use realistic synthetic data
- **Persistence**: Verify model/transformer save/load functionality
- **Reproducibility**: Ensure consistent results with fixed random seeds

## Writing New Tests

### Test File Structure

```python
"""
Comprehensive unit tests for ModuleName.

Tests cover:
- Feature 1
- Feature 2
- Edge cases
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from module_name import ModuleClass


class TestModuleClass:
    """Test cases for ModuleClass."""
    
    def test_feature_basic(self, fixture_name):
        """Test basic feature functionality."""
        # Arrange
        obj = ModuleClass()
        
        # Act
        result = obj.method()
        
        # Assert
        assert result is not None
```

### Best Practices

1. **Naming**: Use descriptive test names (`test_feature_scenario_expected_result`)
2. **Arrange-Act-Assert**: Structure tests clearly
3. **Fixtures**: Use fixtures for shared setup (see `conftest.py`)
4. **Assertions**: Use specific assertions with helpful messages
5. **Edge Cases**: Test error conditions, empty inputs, boundary values
6. **Documentation**: Document what each test verifies

### Using Fixtures

Fixtures are defined in `conftest.py`:

```python
def test_example(sample_fraud_data, temp_data_dir):
    """Test using fixtures."""
    # sample_fraud_data: Sample fraud detection dataset
    # temp_data_dir: Temporary directory for test files
    pass
```

Available fixtures:
- `sample_fraud_data`: E-commerce fraud dataset
- `sample_credit_card_data`: Banking fraud dataset
- `sample_imbalanced_data`: Highly imbalanced dataset
- `sample_data_with_missing`: Data with missing values
- `sample_data_with_duplicates`: Data with duplicate rows
- `sample_model_data`: Data for model training
- `temp_data_dir`: Temporary directory (auto-cleaned)

## Continuous Integration

Tests are automatically run in CI/CD pipeline (see `.github/workflows/unittests.yml`):

- **Trigger**: On push to main/task branches and pull requests
- **Python Version**: 3.12.3
- **Coverage**: Minimum 70% required
- **Reports**: Coverage reports uploaded as artifacts

## Regulatory Compliance

For financial applications, comprehensive testing is critical:

- **Correctness**: All core functions verified (feature engineering, SMOTE handling, model scoring)
- **Reproducibility**: Fixed random seeds ensure consistent results
- **Documentation**: All tests documented for audit purposes
- **Coverage**: High test coverage demonstrates thoroughness

## Troubleshooting

### Import Errors

If you encounter import errors:
```bash
# Ensure src is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Test Failures

1. **Check dependencies**: Ensure all packages in `requirements.txt` are installed
2. **Check data**: Some tests require data files in `data/raw/`
3. **Check random seeds**: Some tests depend on fixed random seeds (42)

### Coverage Issues

If coverage is below threshold:
```bash
# Generate detailed coverage report
pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html to see uncovered lines
```

## Contributing

When adding new features:

1. **Write tests first** (TDD approach recommended)
2. **Achieve >70% coverage** for new code
3. **Update this README** if adding new test categories
4. **Run all tests** before committing: `pytest tests/ -v`
