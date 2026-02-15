"""
Comprehensive unit tests for ModelEvaluator module.

Tests cover:
- Model evaluation metrics (PR-AUC, F1-Score, ROC-AUC, etc.)
- Confusion matrix generation
- Plot generation (confusion matrix, ROC curve, PR curve)
- Model comparison
- Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from model_evaluator import ModelEvaluator


class TestModelEvaluator:
    """Test cases for ModelEvaluator class."""
    
    def test_initialization(self, temp_data_dir):
        """Test ModelEvaluator initialization."""
        evaluator = ModelEvaluator(output_dir=str(temp_data_dir))
        assert evaluator.logger is not None
        assert evaluator.output_dir.exists()
    
    def test_evaluate_model_basic(self, sample_model_data, temp_data_dir):
        """Test basic model evaluation."""
        X_train, y_train = sample_model_data
        X_test = X_train.iloc[:100]
        y_test = y_train.iloc[:100]
        
        # Train a simple model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        evaluator = ModelEvaluator(output_dir=str(temp_data_dir))
        metrics = evaluator.evaluate_model(
            model,
            X_test,
            y_test,
            model_name="TestModel",
            plot=False
        )
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
        assert 'pr_auc' in metrics
        assert 'confusion_matrix' in metrics
        
        # Check metric ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1
        assert 0 <= metrics['pr_auc'] <= 1
    
    def test_evaluate_model_with_plots(self, sample_model_data, temp_data_dir):
        """Test model evaluation with plot generation."""
        X_train, y_train = sample_model_data
        X_test = X_train.iloc[:100]
        y_test = y_train.iloc[:100]
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        evaluator = ModelEvaluator(output_dir=str(temp_data_dir))
        metrics = evaluator.evaluate_model(
            model,
            X_test,
            y_test,
            model_name="TestModel",
            plot=True
        )
        
        # Check that plots were generated
        assert (temp_data_dir / "testmodel_confusion_matrix.png").exists()
        assert (temp_data_dir / "testmodel_pr_curve.png").exists()
        assert (temp_data_dir / "testmodel_roc_curve.png").exists()
    
    def test_compare_models(self, temp_data_dir):
        """Test model comparison functionality."""
        evaluator = ModelEvaluator(output_dir=str(temp_data_dir))
        
        # Create mock results
        results = {
            'Model1': {
                'accuracy': 0.95,
                'precision': 0.90,
                'recall': 0.85,
                'f1_score': 0.87,
                'roc_auc': 0.92,
                'pr_auc': 0.88
            },
            'Model2': {
                'accuracy': 0.93,
                'precision': 0.88,
                'recall': 0.90,
                'f1_score': 0.89,
                'roc_auc': 0.91,
                'pr_auc': 0.90
            }
        }
        
        comparison_df = evaluator.compare_models(results)
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 2
        assert 'Model' in comparison_df.columns
        assert 'PR-AUC' in comparison_df.columns
        
        # Should be sorted by PR-AUC descending
        assert comparison_df.iloc[0]['PR-AUC'] >= comparison_df.iloc[1]['PR-AUC']
    
    def test_compare_models_saves_file(self, temp_data_dir):
        """Test that model comparison saves to file."""
        evaluator = ModelEvaluator(output_dir=str(temp_data_dir))
        
        results = {
            'Model1': {
                'accuracy': 0.95,
                'precision': 0.90,
                'recall': 0.85,
                'f1_score': 0.87,
                'roc_auc': 0.92,
                'pr_auc': 0.88
            }
        }
        
        output_file = "model_comparison.csv"
        comparison_df = evaluator.compare_models(results, output_file=output_file)
        
        assert (temp_data_dir / output_file).exists()
    
    def test_evaluate_model_confusion_matrix_format(self, sample_model_data, temp_data_dir):
        """Test that confusion matrix is in correct format."""
        X_train, y_train = sample_model_data
        X_test = X_train.iloc[:100]
        y_test = y_train.iloc[:100]
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        evaluator = ModelEvaluator(output_dir=str(temp_data_dir))
        metrics = evaluator.evaluate_model(
            model,
            X_test,
            y_test,
            model_name="TestModel",
            plot=False
        )
        
        cm = metrics['confusion_matrix']
        assert isinstance(cm, list)
        assert len(cm) == 2  # Binary classification
        assert len(cm[0]) == 2
        assert len(cm[1]) == 2
    
    def test_evaluate_model_handles_zero_division(self, temp_data_dir):
        """Test that evaluation handles zero division gracefully."""
        # Create data where one class might not be predicted
        X_test = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        y_test = pd.Series([0, 0, 0])  # All same class
        
        # Model that always predicts 0
        class DummyModel:
            def predict(self, X):
                return np.zeros(len(X))
            def predict_proba(self, X):
                return np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        
        model = DummyModel()
        evaluator = ModelEvaluator(output_dir=str(temp_data_dir))
        
        # Should not raise error, but metrics may be 0
        metrics = evaluator.evaluate_model(
            model,
            X_test,
            y_test,
            model_name="DummyModel",
            plot=False
        )
        
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
