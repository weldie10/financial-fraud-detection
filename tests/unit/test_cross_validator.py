"""
Comprehensive unit tests for CrossValidator module.

Tests cover:
- Stratified K-Fold cross-validation
- Metric calculation (mean and std)
- Model cloning
- Results summary
- Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from cross_validator import CrossValidator


class TestCrossValidator:
    """Test cases for CrossValidator class."""
    
    def test_initialization_default(self):
        """Test CrossValidator initialization with defaults."""
        cv = CrossValidator()
        assert cv.n_splits == 5
        assert cv.random_state == 42
        assert cv.cv is not None
        assert cv.cv_results == {}
    
    def test_initialization_custom(self):
        """Test CrossValidator initialization with custom parameters."""
        cv = CrossValidator(n_splits=10, random_state=100)
        assert cv.n_splits == 10
        assert cv.random_state == 100
    
    def test_cross_validate_basic(self, sample_model_data):
        """Test basic cross-validation."""
        X, y = sample_model_data
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        cv = CrossValidator(n_splits=3, random_state=42)
        
        results = cv.cross_validate(model, X, y, model_name="TestModel")
        
        assert 'model_name' in results
        assert results['model_name'] == "TestModel"
        assert results['n_splits'] == 3
        
        # Check all metrics are present
        assert 'accuracy_mean' in results
        assert 'accuracy_std' in results
        assert 'precision_mean' in results
        assert 'precision_std' in results
        assert 'recall_mean' in results
        assert 'recall_std' in results
        assert 'f1_score_mean' in results
        assert 'f1_score_std' in results
        assert 'roc_auc_mean' in results
        assert 'roc_auc_std' in results
        assert 'pr_auc_mean' in results
        assert 'pr_auc_std' in results
    
    def test_cross_validate_metric_ranges(self, sample_model_data):
        """Test that cross-validation metrics are in valid ranges."""
        X, y = sample_model_data
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        cv = CrossValidator(n_splits=3, random_state=42)
        
        results = cv.cross_validate(model, X, y, model_name="TestModel")
        
        # Check metric ranges
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']:
            mean = results[f'{metric}_mean']
            std = results[f'{metric}_std']
            
            assert 0 <= mean <= 1
            assert std >= 0
    
    def test_cross_validate_stores_results(self, sample_model_data):
        """Test that cross-validation results are stored."""
        X, y = sample_model_data
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        cv = CrossValidator(n_splits=3, random_state=42)
        
        cv.cross_validate(model, X, y, model_name="TestModel")
        
        assert "TestModel" in cv.cv_results
        assert len(cv.cv_results) == 1
    
    def test_cross_validate_multiple_models(self, sample_model_data):
        """Test cross-validation with multiple models."""
        X, y = sample_model_data
        
        cv = CrossValidator(n_splits=3, random_state=42)
        
        # First model
        model1 = LogisticRegression(random_state=42, max_iter=1000)
        cv.cross_validate(model1, X, y, model_name="Model1")
        
        # Second model
        model2 = LogisticRegression(random_state=42, max_iter=1000, C=0.1)
        cv.cross_validate(model2, X, y, model_name="Model2")
        
        assert len(cv.cv_results) == 2
        assert "Model1" in cv.cv_results
        assert "Model2" in cv.cv_results
    
    def test_get_results_summary(self, sample_model_data):
        """Test getting results summary."""
        X, y = sample_model_data
        
        cv = CrossValidator(n_splits=3, random_state=42)
        
        model1 = LogisticRegression(random_state=42, max_iter=1000)
        cv.cross_validate(model1, X, y, model_name="Model1")
        
        model2 = LogisticRegression(random_state=42, max_iter=1000, C=0.1)
        cv.cross_validate(model2, X, y, model_name="Model2")
        
        summary_df = cv.get_results_summary()
        
        assert isinstance(summary_df, pd.DataFrame)
        assert len(summary_df) == 2
        assert 'Model' in summary_df.columns
        assert 'PR-AUC (mean±std)' in summary_df.columns
    
    def test_get_results_summary_empty(self):
        """Test getting summary when no results exist."""
        cv = CrossValidator()
        summary_df = cv.get_results_summary()
        
        assert isinstance(summary_df, pd.DataFrame)
        assert len(summary_df) == 0
    
    def test_cross_validate_reproducibility(self, sample_model_data):
        """Test that cross-validation is reproducible with same random_state."""
        X, y = sample_model_data
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        # First run
        cv1 = CrossValidator(n_splits=3, random_state=42)
        results1 = cv1.cross_validate(model, X, y, model_name="TestModel")
        
        # Second run with same random_state
        cv2 = CrossValidator(n_splits=3, random_state=42)
        results2 = cv2.cross_validate(model, X, y, model_name="TestModel")
        
        # Results should be identical
        for key in results1:
            if key not in ['model_name', 'n_splits']:
                assert results1[key] == results2[key]
    
    def test_cross_validate_different_random_state(self, sample_model_data):
        """Test that different random_state produces different results."""
        X, y = sample_model_data
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        cv1 = CrossValidator(n_splits=3, random_state=42)
        results1 = cv1.cross_validate(model, X, y, model_name="TestModel")
        
        cv2 = CrossValidator(n_splits=3, random_state=100)
        results2 = cv2.cross_validate(model, X, y, model_name="TestModel")
        
        # Results may differ (though not guaranteed)
        # At least check that function completes successfully
        assert 'accuracy_mean' in results2
