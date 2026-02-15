"""
Comprehensive unit tests for FeatureEngineer module.

Tests cover:
- Time feature extraction
- Time since signup calculation
- Transaction frequency calculation
- Transaction velocity calculation
- Complete feature engineering pipeline
- Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from feature_engineer import FeatureEngineer


class TestFeatureEngineer:
    """Test cases for FeatureEngineer class."""
    
    def test_initialization(self):
        """Test FeatureEngineer initialization."""
        engineer = FeatureEngineer()
        assert engineer.logger is not None
    
    def test_extract_time_features(self, sample_fraud_data):
        """Test time feature extraction."""
        engineer = FeatureEngineer()
        
        # Convert to datetime first
        sample_fraud_data['purchase_time'] = pd.to_datetime(sample_fraud_data['purchase_time'])
        
        df_features = engineer.extract_time_features(
            sample_fraud_data,
            datetime_column='purchase_time',
            prefix='purchase'
        )
        
        assert 'purchase_hour_of_day' in df_features.columns
        assert 'purchase_day_of_week' in df_features.columns
        assert 'purchase_day_of_month' in df_features.columns
        assert 'purchase_month' in df_features.columns
        assert 'purchase_year' in df_features.columns
        assert 'purchase_is_weekend' in df_features.columns
        assert 'purchase_is_business_hours' in df_features.columns
    
    def test_extract_time_features_without_prefix(self, sample_fraud_data):
        """Test time feature extraction without prefix."""
        engineer = FeatureEngineer()
        sample_fraud_data['purchase_time'] = pd.to_datetime(sample_fraud_data['purchase_time'])
        
        df_features = engineer.extract_time_features(
            sample_fraud_data,
            datetime_column='purchase_time'
        )
        
        assert 'hour_of_day' in df_features.columns
        assert 'day_of_week' in df_features.columns
    
    def test_extract_time_features_missing_column(self, sample_fraud_data):
        """Test error handling for missing datetime column."""
        engineer = FeatureEngineer()
        
        with pytest.raises(ValueError, match="not found"):
            engineer.extract_time_features(
                sample_fraud_data,
                datetime_column='nonexistent_column'
            )
    
    def test_extract_time_features_converts_string_to_datetime(self):
        """Test automatic conversion of string to datetime."""
        df = pd.DataFrame({
            'timestamp': ['2024-01-01 10:00:00', '2024-01-02 15:30:00']
        })
        
        engineer = FeatureEngineer()
        df_features = engineer.extract_time_features(df, datetime_column='timestamp')
        
        assert 'hour_of_day' in df_features.columns
        assert df_features['hour_of_day'].iloc[0] == 10
    
    def test_calculate_time_since_signup(self, sample_fraud_data):
        """Test time since signup calculation."""
        engineer = FeatureEngineer()
        
        # Convert to datetime
        sample_fraud_data['signup_time'] = pd.to_datetime(sample_fraud_data['signup_time'])
        sample_fraud_data['purchase_time'] = pd.to_datetime(sample_fraud_data['purchase_time'])
        
        df_features = engineer.calculate_time_since_signup(
            sample_fraud_data,
            signup_column='signup_time',
            reference_column='purchase_time'
        )
        
        assert 'time_since_signup' in df_features.columns
        assert 'time_since_signup_days' in df_features.columns
        assert 'time_since_signup_minutes' in df_features.columns
        
        # Check that values are non-negative
        assert (df_features['time_since_signup'] >= 0).all()
    
    def test_calculate_time_since_signup_handles_negative(self):
        """Test handling of negative time differences (data quality issue)."""
        df = pd.DataFrame({
            'signup_time': pd.to_datetime(['2024-01-02', '2024-01-01']),
            'purchase_time': pd.to_datetime(['2024-01-01', '2024-01-02'])
        })
        
        engineer = FeatureEngineer()
        df_features = engineer.calculate_time_since_signup(
            df,
            signup_column='signup_time',
            reference_column='purchase_time'
        )
        
        # Negative values should be set to 0
        assert (df_features['time_since_signup'] >= 0).all()
    
    def test_calculate_time_since_signup_missing_columns(self, sample_fraud_data):
        """Test error handling for missing columns."""
        engineer = FeatureEngineer()
        
        with pytest.raises(ValueError, match="not found"):
            engineer.calculate_time_since_signup(
                sample_fraud_data,
                signup_column='nonexistent',
                reference_column='purchase_time'
            )
    
    def test_calculate_transaction_frequency(self):
        """Test transaction frequency calculation."""
        # Create sample data with multiple transactions per user
        dates = pd.date_range('2024-01-01', periods=20, freq='1H')
        df = pd.DataFrame({
            'user_id': ['user1'] * 10 + ['user2'] * 10,
            'purchase_time': dates,
            'amount': np.random.uniform(10, 100, 20)
        })
        
        engineer = FeatureEngineer()
        df_features = engineer.calculate_transaction_frequency(
            df,
            user_column='user_id',
            datetime_column='purchase_time',
            time_windows=['1H', '24H']
        )
        
        assert 'txn_freq_1H' in df_features.columns
        assert 'txn_freq_24H' in df_features.columns
        
        # Check that frequency values are non-negative
        assert (df_features['txn_freq_1H'] >= 0).all()
        assert (df_features['txn_freq_24H'] >= 0).all()
    
    def test_calculate_transaction_frequency_missing_columns(self):
        """Test error handling for missing columns."""
        df = pd.DataFrame({'user_id': ['user1', 'user2']})
        
        engineer = FeatureEngineer()
        
        with pytest.raises(ValueError, match="not found"):
            engineer.calculate_transaction_frequency(
                df,
                user_column='user_id',
                datetime_column='nonexistent'
            )
    
    def test_calculate_transaction_velocity(self):
        """Test transaction velocity calculation."""
        dates = pd.date_range('2024-01-01', periods=10, freq='1H')
        df = pd.DataFrame({
            'user_id': ['user1'] * 10,
            'purchase_time': dates,
            'amount': np.random.uniform(10, 100, 10)
        })
        
        engineer = FeatureEngineer()
        df_features = engineer.calculate_transaction_velocity(
            df,
            user_column='user_id',
            datetime_column='purchase_time',
            amount_column='amount'
        )
        
        assert 'time_since_last_txn' in df_features.columns
        assert 'txn_velocity' in df_features.columns
        assert 'amount_velocity' in df_features.columns
        
        # First transaction should have 0 velocity
        assert df_features['txn_velocity'].iloc[0] == 0
    
    def test_calculate_transaction_velocity_without_amount(self):
        """Test velocity calculation without amount column."""
        dates = pd.date_range('2024-01-01', periods=10, freq='1H')
        df = pd.DataFrame({
            'user_id': ['user1'] * 10,
            'purchase_time': dates
        })
        
        engineer = FeatureEngineer()
        df_features = engineer.calculate_transaction_velocity(
            df,
            user_column='user_id',
            datetime_column='purchase_time'
        )
        
        assert 'txn_velocity' in df_features.columns
        assert 'amount_velocity' not in df_features.columns
    
    def test_engineer_all_features(self, sample_fraud_data):
        """Test complete feature engineering pipeline."""
        engineer = FeatureEngineer()
        
        # Convert to datetime
        sample_fraud_data['signup_time'] = pd.to_datetime(sample_fraud_data['signup_time'])
        sample_fraud_data['purchase_time'] = pd.to_datetime(sample_fraud_data['purchase_time'])
        
        original_cols = len(sample_fraud_data.columns)
        
        df_features = engineer.engineer_all_features(
            sample_fraud_data,
            user_column='user_id',
            purchase_datetime='purchase_time',
            signup_datetime='signup_time',
            amount_column='purchase_value'
        )
        
        # Should have more columns after feature engineering
        assert len(df_features.columns) > original_cols
        
        # Check for key engineered features
        assert 'time_since_signup' in df_features.columns
        assert 'purchase_hour_of_day' in df_features.columns
    
    def test_engineer_all_features_without_signup(self, sample_fraud_data):
        """Test feature engineering without signup datetime."""
        engineer = FeatureEngineer()
        sample_fraud_data['purchase_time'] = pd.to_datetime(sample_fraud_data['purchase_time'])
        
        df_features = engineer.engineer_all_features(
            sample_fraud_data,
            user_column='user_id',
            purchase_datetime='purchase_time',
            amount_column='purchase_value'
        )
        
        # Should still create time and frequency features
        assert 'purchase_hour_of_day' in df_features.columns
        # time_since_signup should not be present
        assert 'time_since_signup' not in df_features.columns
