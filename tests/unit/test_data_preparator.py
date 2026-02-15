"""
Comprehensive unit tests for DataPreparator module.

Tests cover:
- Stratified train-test split
- Feature-target separation
- Class distribution preservation
- Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data_preparator import DataPreparator


class TestDataPreparator:
    """Test cases for DataPreparator class."""
    
    def test_initialization(self):
        """Test DataPreparator initialization."""
        preparator = DataPreparator()
        assert preparator.logger is not None
    
    def test_prepare_data_basic(self, sample_imbalanced_data):
        """Test basic data preparation."""
        preparator = DataPreparator()
        X_train, X_test, y_train, y_test = preparator.prepare_data(
            sample_imbalanced_data,
            target_column='target',
            test_size=0.2,
            random_state=42
        )
        
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        
        # Check shapes
        total_samples = len(sample_imbalanced_data)
        assert len(X_train) + len(X_test) == total_samples
        assert len(y_train) + len(y_test) == total_samples
        
        # Check that target is not in features
        assert 'target' not in X_train.columns
        assert 'target' not in X_test.columns
    
    def test_prepare_data_stratification(self, sample_imbalanced_data):
        """Test that stratification preserves class distribution."""
        preparator = DataPreparator()
        X_train, X_test, y_train, y_test = preparator.prepare_data(
            sample_imbalanced_data,
            target_column='target',
            test_size=0.2,
            random_state=42
        )
        
        # Check that class distribution is similar in train and test
        train_dist = y_train.value_counts(normalize=True)
        test_dist = y_test.value_counts(normalize=True)
        
        # Proportions should be similar (within 5%)
        for cls in train_dist.index:
            assert abs(train_dist[cls] - test_dist[cls]) < 0.05
    
    def test_prepare_data_test_size(self, sample_imbalanced_data):
        """Test different test sizes."""
        preparator = DataPreparator()
        
        # Test with 30% test size
        X_train, X_test, y_train, y_test = preparator.prepare_data(
            sample_imbalanced_data,
            target_column='target',
            test_size=0.3,
            random_state=42
        )
        
        total = len(sample_imbalanced_data)
        assert abs(len(X_test) / total - 0.3) < 0.02  # Allow small tolerance
        assert abs(len(X_train) / total - 0.7) < 0.02
    
    def test_prepare_data_missing_target_column(self, sample_imbalanced_data):
        """Test error handling for missing target column."""
        preparator = DataPreparator()
        
        with pytest.raises(ValueError, match="not found"):
            preparator.prepare_data(
                sample_imbalanced_data,
                target_column='nonexistent',
                test_size=0.2
            )
    
    def test_prepare_data_reproducibility(self, sample_imbalanced_data):
        """Test that same random_state produces same split."""
        preparator = DataPreparator()
        
        # First split
        X_train1, X_test1, y_train1, y_test1 = preparator.prepare_data(
            sample_imbalanced_data,
            target_column='target',
            test_size=0.2,
            random_state=42
        )
        
        # Second split with same random_state
        X_train2, X_test2, y_train2, y_test2 = preparator.prepare_data(
            sample_imbalanced_data,
            target_column='target',
            test_size=0.2,
            random_state=42
        )
        
        # Should produce identical splits
        assert X_train1.index.equals(X_train2.index)
        assert X_test1.index.equals(X_test2.index)
    
    def test_prepare_data_different_random_state(self, sample_imbalanced_data):
        """Test that different random_state produces different splits."""
        preparator = DataPreparator()
        
        X_train1, X_test1, _, _ = preparator.prepare_data(
            sample_imbalanced_data,
            target_column='target',
            test_size=0.2,
            random_state=42
        )
        
        X_train2, X_test2, _, _ = preparator.prepare_data(
            sample_imbalanced_data,
            target_column='target',
            test_size=0.2,
            random_state=100
        )
        
        # Should produce different splits
        assert not X_train1.index.equals(X_train2.index)
    
    def test_prepare_data_all_classes_present(self, sample_imbalanced_data):
        """Test that all classes are present in both train and test sets."""
        preparator = DataPreparator()
        X_train, X_test, y_train, y_test = preparator.prepare_data(
            sample_imbalanced_data,
            target_column='target',
            test_size=0.2,
            random_state=42
        )
        
        # Both sets should have both classes
        assert set(y_train.unique()) == set(sample_imbalanced_data['target'].unique())
        assert set(y_test.unique()) == set(sample_imbalanced_data['target'].unique())
    
    def test_prepare_data_small_dataset(self):
        """Test data preparation with small dataset."""
        # Create small dataset
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'target': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        })
        
        preparator = DataPreparator()
        X_train, X_test, y_train, y_test = preparator.prepare_data(
            df,
            target_column='target',
            test_size=0.2,
            random_state=42
        )
        
        # Should still work with small dataset
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
