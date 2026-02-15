"""
Comprehensive unit tests for DataCleaner module.

Tests cover:
- Missing value handling (various strategies)
- Duplicate removal
- Data type correction
- Complete cleaning pipeline
- Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data_cleaner import DataCleaner


class TestDataCleaner:
    """Test cases for DataCleaner class."""
    
    def test_initialization(self):
        """Test DataCleaner initialization."""
        cleaner = DataCleaner()
        assert cleaner.logger is not None
        assert cleaner.imputation_strategy == "mean"
    
    def test_initialization_custom_strategy(self):
        """Test initialization with custom strategy."""
        cleaner = DataCleaner(imputation_strategy="median")
        assert cleaner.imputation_strategy == "median"
    
    def test_handle_missing_values_mean(self, sample_data_with_missing):
        """Test missing value imputation with mean strategy."""
        cleaner = DataCleaner(imputation_strategy="mean")
        df_cleaned = cleaner.handle_missing_values(sample_data_with_missing)
        
        assert df_cleaned['feature1'].isnull().sum() == 0
        assert df_cleaned['feature2'].isnull().sum() == 0
    
    def test_handle_missing_values_median(self, sample_data_with_missing):
        """Test missing value imputation with median strategy."""
        cleaner = DataCleaner(imputation_strategy="median")
        df_cleaned = cleaner.handle_missing_values(sample_data_with_missing)
        
        assert df_cleaned['feature1'].isnull().sum() == 0
        assert df_cleaned['feature2'].isnull().sum() == 0
    
    def test_handle_missing_values_mode(self, sample_data_with_missing):
        """Test missing value imputation with mode strategy."""
        cleaner = DataCleaner(imputation_strategy="most_frequent")
        df_cleaned = cleaner.handle_missing_values(sample_data_with_missing)
        
        assert df_cleaned['feature3'].isnull().sum() == 0
    
    def test_handle_missing_values_drop(self, sample_data_with_missing):
        """Test missing value handling with drop strategy."""
        cleaner = DataCleaner(imputation_strategy="drop")
        initial_len = len(sample_data_with_missing)
        df_cleaned = cleaner.handle_missing_values(sample_data_with_missing)
        
        assert len(df_cleaned) < initial_len
        assert df_cleaned.isnull().sum().sum() == 0
    
    def test_handle_missing_values_specific_columns(self, sample_data_with_missing):
        """Test missing value handling for specific columns."""
        cleaner = DataCleaner(imputation_strategy="mean")
        df_cleaned = cleaner.handle_missing_values(
            sample_data_with_missing,
            columns=['feature1']
        )
        
        assert df_cleaned['feature1'].isnull().sum() == 0
        # feature2 should still have missing values if not processed
        # (depending on implementation)
    
    def test_handle_missing_values_threshold(self):
        """Test dropping columns with high missing percentage."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [np.nan] * 5,  # 100% missing
            'feature3': [10, 20, np.nan, 40, 50]
        })
        
        cleaner = DataCleaner()
        df_cleaned = cleaner.handle_missing_values(df, threshold=0.5)
        
        # feature2 should be dropped
        assert 'feature2' not in df_cleaned.columns
    
    def test_remove_duplicates_default(self, sample_data_with_duplicates):
        """Test duplicate removal with default settings."""
        cleaner = DataCleaner()
        initial_len = len(sample_data_with_duplicates)
        df_cleaned = cleaner.remove_duplicates(sample_data_with_duplicates)
        
        assert len(df_cleaned) < initial_len
        assert df_cleaned.duplicated().sum() == 0
    
    def test_remove_duplicates_subset(self):
        """Test duplicate removal with specific columns."""
        df = pd.DataFrame({
            'feature1': [1, 2, 2, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'feature3': [100, 200, 200, 400, 500]
        })
        
        cleaner = DataCleaner()
        df_cleaned = cleaner.remove_duplicates(df, subset=['feature1'])
        
        assert len(df_cleaned) == 4  # One duplicate removed
    
    def test_remove_duplicates_keep_last(self):
        """Test duplicate removal keeping last occurrence."""
        df = pd.DataFrame({
            'feature1': [1, 2, 2, 4],
            'feature2': [10, 20, 30, 40]
        })
        
        cleaner = DataCleaner()
        df_cleaned = cleaner.remove_duplicates(df, subset=['feature1'], keep='last')
        
        assert len(df_cleaned) == 3
        assert df_cleaned.iloc[1]['feature2'] == 30  # Last occurrence kept
    
    def test_remove_duplicates_no_duplicates(self, sample_fraud_data):
        """Test duplicate removal when no duplicates exist."""
        cleaner = DataCleaner()
        initial_len = len(sample_fraud_data)
        df_cleaned = cleaner.remove_duplicates(sample_fraud_data)
        
        assert len(df_cleaned) == initial_len
    
    def test_correct_data_types_datetime(self, sample_fraud_data):
        """Test datetime conversion."""
        cleaner = DataCleaner()
        df_cleaned = cleaner.correct_data_types(
            sample_fraud_data,
            date_columns=['signup_time', 'purchase_time']
        )
        
        assert pd.api.types.is_datetime64_any_dtype(df_cleaned['signup_time'])
        assert pd.api.types.is_datetime64_any_dtype(df_cleaned['purchase_time'])
    
    def test_correct_data_types_dtype_mapping(self):
        """Test data type conversion with mapping."""
        df = pd.DataFrame({
            'feature1': ['1', '2', '3'],
            'feature2': [1.5, 2.5, 3.5]
        })
        
        cleaner = DataCleaner()
        df_cleaned = cleaner.correct_data_types(
            df,
            dtype_mapping={'feature1': int}
        )
        
        assert df_cleaned['feature1'].dtype == int
    
    def test_clean_complete_pipeline(self, sample_data_with_missing, sample_data_with_duplicates):
        """Test complete cleaning pipeline."""
        # Combine missing and duplicates
        df = pd.concat([sample_data_with_missing, sample_data_with_duplicates], ignore_index=True)
        df = df.drop_duplicates().reset_index(drop=True)
        
        cleaner = DataCleaner(imputation_strategy="mean")
        df_cleaned = cleaner.clean(
            df,
            handle_missing=True,
            remove_dups=True,
            correct_types=True
        )
        
        assert df_cleaned.isnull().sum().sum() == 0
        assert df_cleaned.duplicated().sum() == 0
    
    def test_clean_partial_pipeline(self, sample_data_with_missing):
        """Test cleaning pipeline with some steps disabled."""
        cleaner = DataCleaner()
        df_cleaned = cleaner.clean(
            sample_data_with_missing,
            handle_missing=True,
            remove_dups=False,
            correct_types=False
        )
        
        # Missing values should be handled
        assert df_cleaned.isnull().sum().sum() == 0
    
    def test_handle_missing_values_knn(self):
        """Test KNN imputation."""
        df = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [10, 20, 30, np.nan, 50],
            'feature3': [100, 200, 300, 400, 500]
        })
        
        cleaner = DataCleaner(imputation_strategy="knn")
        df_cleaned = cleaner.handle_missing_values(df)
        
        assert df_cleaned.isnull().sum().sum() == 0
