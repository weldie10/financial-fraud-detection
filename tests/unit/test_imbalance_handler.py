"""
Comprehensive unit tests for ImbalanceHandler module.

Tests cover:
- SMOTE resampling
- Undersampling
- Combined methods (SMOTE-Tomek, SMOTE-ENN)
- Imbalance analysis
- Class distribution verification
- Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from imbalance_handler import ImbalanceHandler


class TestImbalanceHandler:
    """Test cases for ImbalanceHandler class."""
    
    def test_initialization_default(self):
        """Test ImbalanceHandler initialization with default method."""
        handler = ImbalanceHandler()
        assert handler.method == "smote"
        assert handler.random_state == 42
        assert handler.sampler is not None
    
    def test_initialization_custom_method(self):
        """Test initialization with custom method."""
        handler = ImbalanceHandler(method="undersample", random_state=100)
        assert handler.method == "undersample"
        assert handler.random_state == 100
    
    def test_initialization_unknown_method(self):
        """Test initialization with unknown method defaults to SMOTE."""
        handler = ImbalanceHandler(method="unknown_method")
        assert handler.method == "unknown_method"
        # Should still have a sampler (defaults to SMOTE)
        assert handler.sampler is not None
    
    def test_analyze_imbalance(self, sample_imbalanced_data):
        """Test imbalance analysis."""
        handler = ImbalanceHandler()
        stats = handler.analyze_imbalance(sample_imbalanced_data['target'])
        
        assert 'class_counts' in stats
        assert 'class_proportions' in stats
        assert 'total_samples' in stats
        assert 'imbalance_ratio' in stats
        assert stats['total_samples'] == len(sample_imbalanced_data)
        assert stats['imbalance_ratio'] > 1
    
    def test_resample_smote(self, sample_imbalanced_data):
        """Test SMOTE resampling."""
        handler = ImbalanceHandler(method="smote", random_state=42)
        
        X = sample_imbalanced_data[['feature1', 'feature2']]
        y = sample_imbalanced_data['target']
        
        X_resampled, y_resampled, stats = handler.resample(X, y)
        
        # Check that resampling occurred
        assert len(X_resampled) > len(X)
        assert len(y_resampled) == len(X_resampled)
        
        # Check class distribution improved
        assert 'before' in stats
        assert 'after' in stats
        after_ratio = max(stats['after']['class_counts'].values()) / min(stats['after']['class_counts'].values())
        before_ratio = stats['before']['imbalance_ratio']
        assert after_ratio <= before_ratio
    
    def test_resample_undersample(self, sample_imbalanced_data):
        """Test random undersampling."""
        handler = ImbalanceHandler(method="undersample", random_state=42)
        
        X = sample_imbalanced_data[['feature1', 'feature2']]
        y = sample_imbalanced_data['target']
        
        X_resampled, y_resampled, stats = handler.resample(X, y)
        
        # Undersampling should reduce samples
        assert len(X_resampled) <= len(X)
        assert len(y_resampled) == len(X_resampled)
        
        # Check that classes are more balanced
        class_counts = y_resampled.value_counts()
        ratio = max(class_counts) / min(class_counts)
        assert ratio < stats['before']['imbalance_ratio']
    
    def test_resample_smote_tomek(self, sample_imbalanced_data):
        """Test SMOTE-Tomek combined method."""
        handler = ImbalanceHandler(method="smote_tomek", random_state=42)
        
        X = sample_imbalanced_data[['feature1', 'feature2']]
        y = sample_imbalanced_data['target']
        
        X_resampled, y_resampled, stats = handler.resample(X, y)
        
        assert len(X_resampled) > 0
        assert len(y_resampled) == len(X_resampled)
        assert stats['method'] == 'smote_tomek'
    
    def test_resample_preserves_dataframe_structure(self, sample_imbalanced_data):
        """Test that resampling preserves DataFrame structure."""
        handler = ImbalanceHandler(method="smote", random_state=42)
        
        X = sample_imbalanced_data[['feature1', 'feature2']]
        y = sample_imbalanced_data['target']
        
        X_resampled, y_resampled, _ = handler.resample(X, y)
        
        # Should return DataFrame and Series
        assert isinstance(X_resampled, pd.DataFrame)
        assert isinstance(y_resampled, pd.Series)
        
        # Should preserve column names
        assert list(X_resampled.columns) == list(X.columns)
    
    def test_resample_with_numpy_arrays(self, sample_imbalanced_data):
        """Test resampling with numpy arrays instead of DataFrames."""
        handler = ImbalanceHandler(method="smote", random_state=42)
        
        X = sample_imbalanced_data[['feature1', 'feature2']].values
        y = sample_imbalanced_data['target'].values
        
        X_resampled, y_resampled, _ = handler.resample(X, y)
        
        # Should still work and return appropriate types
        assert len(X_resampled) > 0
        assert len(y_resampled) == len(X_resampled)
    
    def test_resample_statistics(self, sample_imbalanced_data):
        """Test that resampling statistics are accurate."""
        handler = ImbalanceHandler(method="smote", random_state=42)
        
        X = sample_imbalanced_data[['feature1', 'feature2']]
        y = sample_imbalanced_data['target']
        
        X_resampled, y_resampled, stats = handler.resample(X, y)
        
        # Check statistics structure
        assert 'before' in stats
        assert 'after' in stats
        assert 'method' in stats
        assert 'samples_added' in stats or 'samples_removed' in stats
        
        # Verify before stats match original data
        original_counts = y.value_counts().to_dict()
        assert stats['before']['class_counts'] == original_counts
    
    def test_justify_method_choice_smote(self):
        """Test method justification for SMOTE."""
        handler = ImbalanceHandler(method="smote")
        justification = handler.justify_method_choice(imbalance_ratio=10.0, dataset_size=10000)
        
        assert "SMOTE" in justification
        assert "synthetic" in justification.lower()
    
    def test_justify_method_choice_undersample(self):
        """Test method justification for undersampling."""
        handler = ImbalanceHandler(method="undersample")
        justification = handler.justify_method_choice(imbalance_ratio=5.0, dataset_size=50000)
        
        assert "undersampling" in justification.lower()
    
    def test_resample_extreme_imbalance(self):
        """Test resampling with extremely imbalanced data."""
        # Create extremely imbalanced dataset (99:1)
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(0, 1, 1000)
        })
        y = pd.Series([0] * 990 + [1] * 10)
        
        handler = ImbalanceHandler(method="smote", random_state=42)
        X_resampled, y_resampled, stats = handler.resample(X, y)
        
        # Should handle extreme imbalance
        assert len(X_resampled) > 0
        assert len(y_resampled) == len(X_resampled)
        
        # Check that minority class is represented
        assert 1 in y_resampled.values
    
    def test_resample_already_balanced(self):
        """Test resampling with already balanced data."""
        # Create balanced dataset
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 200),
            'feature2': np.random.normal(0, 1, 200)
        })
        y = pd.Series([0] * 100 + [1] * 100)
        
        handler = ImbalanceHandler(method="smote", random_state=42)
        X_resampled, y_resampled, stats = handler.resample(X, y)
        
        # Should still work (may not change much)
        assert len(X_resampled) > 0
        assert len(y_resampled) == len(X_resampled)
