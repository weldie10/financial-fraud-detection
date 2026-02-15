"""
Comprehensive unit tests for DataTransformer module.

Tests cover:
- Feature scaling (StandardScaler, MinMaxScaler, RobustScaler)
- Categorical encoding (OneHotEncoder)
- Column type identification
- Fit and transform pipeline
- Save/load transformers
- Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data_transformer import DataTransformer


class TestDataTransformer:
    """Test cases for DataTransformer class."""
    
    def test_initialization_default(self):
        """Test DataTransformer initialization with default scaler."""
        transformer = DataTransformer()
        assert transformer.scaling_method == "standard"
        assert transformer.scaler is not None
        assert transformer.encoder is not None
        assert transformer.is_fitted is False
    
    def test_initialization_custom_scaler(self):
        """Test initialization with custom scaling method."""
        transformer = DataTransformer(scaling_method="minmax")
        assert transformer.scaling_method == "minmax"
    
    def test_identify_column_types_auto(self, sample_numeric_features, sample_categorical_features):
        """Test automatic column type identification."""
        df = pd.concat([sample_numeric_features, sample_categorical_features], axis=1)
        
        transformer = DataTransformer()
        column_types = transformer.identify_column_types(df)
        
        assert 'numeric' in column_types
        assert 'categorical' in column_types
        assert len(column_types['numeric']) == 3
        assert len(column_types['categorical']) == 3
    
    def test_identify_column_types_explicit(self):
        """Test column type identification with explicit lists."""
        df = pd.DataFrame({
            'num1': [1, 2, 3],
            'num2': [4, 5, 6],
            'cat1': ['A', 'B', 'C']
        })
        
        transformer = DataTransformer()
        column_types = transformer.identify_column_types(
            df,
            numeric_columns=['num1', 'num2'],
            categorical_columns=['cat1']
        )
        
        assert set(column_types['numeric']) == {'num1', 'num2'}
        assert set(column_types['categorical']) == {'cat1'}
    
    def test_identify_column_types_exclude(self):
        """Test column type identification with excluded columns."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 0]
        })
        
        transformer = DataTransformer()
        column_types = transformer.identify_column_types(
            df,
            exclude_columns=['target']
        )
        
        assert 'target' not in column_types['numeric']
        assert 'target' not in column_types['categorical']
    
    def test_fit_transform_numeric_only(self, sample_numeric_features):
        """Test fit_transform with numeric features only."""
        transformer = DataTransformer(scaling_method="standard")
        df_transformed = transformer.fit_transform(sample_numeric_features)
        
        assert transformer.is_fitted is True
        assert len(df_transformed.columns) == len(sample_numeric_features.columns)
        
        # Check that values are scaled (mean ~0, std ~1 for standard scaler)
        for col in sample_numeric_features.columns:
            assert col in df_transformed.columns
            # Mean should be close to 0, std close to 1
            assert abs(df_transformed[col].mean()) < 0.1
            assert abs(df_transformed[col].std() - 1.0) < 0.1
    
    def test_fit_transform_categorical_only(self, sample_categorical_features):
        """Test fit_transform with categorical features only."""
        transformer = DataTransformer()
        df_transformed = transformer.fit_transform(sample_categorical_features)
        
        assert transformer.is_fitted is True
        # One-hot encoding should create more columns
        assert len(df_transformed.columns) >= len(sample_categorical_features.columns)
    
    def test_fit_transform_mixed_features(self, sample_numeric_features, sample_categorical_features):
        """Test fit_transform with both numeric and categorical features."""
        df = pd.concat([sample_numeric_features, sample_categorical_features], axis=1)
        
        transformer = DataTransformer()
        df_transformed = transformer.fit_transform(df)
        
        assert transformer.is_fitted is True
        # Should have transformed both types
        assert len(df_transformed.columns) > len(df.columns)
    
    def test_fit_transform_excludes_target(self):
        """Test that target column is excluded from transformation."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'target': [0, 1, 0, 1, 0]
        })
        
        transformer = DataTransformer()
        df_transformed = transformer.fit_transform(df, target_column='target')
        
        # Target should still be present and unchanged
        assert 'target' in df_transformed.columns
        assert (df_transformed['target'] == df['target']).all()
    
    def test_transform_before_fit_raises_error(self, sample_numeric_features):
        """Test that transform raises error if not fitted."""
        transformer = DataTransformer()
        
        with pytest.raises(ValueError, match="not fitted"):
            transformer.transform(sample_numeric_features)
    
    def test_transform_after_fit(self, sample_numeric_features):
        """Test transform after fit_transform."""
        transformer = DataTransformer()
        
        # Fit on first half
        df_train = sample_numeric_features.iloc[:50]
        transformer.fit_transform(df_train)
        
        # Transform second half
        df_test = sample_numeric_features.iloc[50:]
        df_transformed = transformer.transform(df_test)
        
        assert len(df_transformed) == len(df_test)
        assert len(df_transformed.columns) == len(df_test.columns)
    
    def test_save_and_load_transformers(self, sample_numeric_features, temp_data_dir):
        """Test saving and loading transformers."""
        transformer1 = DataTransformer(scaling_method="standard")
        transformer1.fit_transform(sample_numeric_features)
        
        # Save
        save_path = temp_data_dir / "transformers.pkl"
        transformer1.save_transformers(str(save_path))
        assert save_path.exists()
        
        # Load
        transformer2 = DataTransformer()
        transformer2.load_transformers(str(save_path))
        
        assert transformer2.is_fitted is True
        assert transformer2.scaling_method == "standard"
        assert len(transformer2.numeric_columns) == len(transformer1.numeric_columns)
    
    def test_load_transformers_file_not_found(self):
        """Test loading non-existent transformer file."""
        transformer = DataTransformer()
        
        with pytest.raises(FileNotFoundError):
            transformer.load_transformers("nonexistent.pkl")
    
    def test_fit_transform_preserves_index(self, sample_numeric_features):
        """Test that fit_transform preserves DataFrame index."""
        custom_index = ['a', 'b', 'c', 'd', 'e'] * 20
        sample_numeric_features.index = custom_index[:len(sample_numeric_features)]
        
        transformer = DataTransformer()
        df_transformed = transformer.fit_transform(sample_numeric_features)
        
        assert list(df_transformed.index) == list(sample_numeric_features.index)
    
    def test_minmax_scaler(self, sample_numeric_features):
        """Test MinMaxScaler scaling."""
        transformer = DataTransformer(scaling_method="minmax")
        df_transformed = transformer.fit_transform(sample_numeric_features)
        
        # Values should be in [0, 1] range (approximately)
        for col in df_transformed.columns:
            assert df_transformed[col].min() >= -0.1  # Allow small tolerance
            assert df_transformed[col].max() <= 1.1
    
    def test_robust_scaler(self, sample_numeric_features):
        """Test RobustScaler scaling."""
        # Add outliers
        df_with_outliers = sample_numeric_features.copy()
        df_with_outliers.loc[0, 'feature1'] = 1000
        
        transformer = DataTransformer(scaling_method="robust")
        df_transformed = transformer.fit_transform(df_with_outliers)
        
        # Robust scaler should handle outliers better
        assert transformer.is_fitted is True
