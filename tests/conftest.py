"""
Pytest configuration and fixtures for financial fraud detection tests.

This module provides shared test fixtures and utilities for all test modules.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timedelta


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_fraud_data():
    """Create sample fraud detection dataset."""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data
    data = {
        'user_id': [f'user_{i}' for i in range(n_samples)],
        'signup_time': [
            datetime.now() - timedelta(days=np.random.randint(0, 365))
            for _ in range(n_samples)
        ],
        'purchase_time': [
            datetime.now() - timedelta(hours=np.random.randint(0, 720))
            for _ in range(n_samples)
        ],
        'purchase_value': np.random.uniform(10, 200, n_samples),
        'device_id': [f'device_{np.random.randint(1, 50)}' for _ in range(n_samples)],
        'source': np.random.choice(['web', 'mobile', 'api'], n_samples),
        'browser': np.random.choice(['chrome', 'firefox', 'safari'], n_samples),
        'sex': np.random.choice(['M', 'F'], n_samples),
        'age': np.random.randint(18, 80, n_samples),
        'ip_address': [
            f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}."
            f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
            for _ in range(n_samples)
        ],
        'class': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])  # 10% fraud
    }
    
    df = pd.DataFrame(data)
    # Ensure some fraud cases have instant purchases
    fraud_indices = df[df['class'] == 1].index[:50]
    df.loc[fraud_indices, 'purchase_time'] = df.loc[fraud_indices, 'signup_time'] + timedelta(seconds=np.random.randint(0, 3600))
    
    return df


@pytest.fixture
def sample_credit_card_data():
    """Create sample credit card dataset."""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate PCA-like features (V1-V28)
    data = {}
    for i in range(1, 29):
        data[f'V{i}'] = np.random.normal(0, 1, n_samples)
    
    data['Time'] = np.random.uniform(0, 172792, n_samples)
    data['Amount'] = np.random.exponential(88, n_samples)
    data['Class'] = np.random.choice([0, 1], n_samples, p=[0.998, 0.002])  # 0.2% fraud
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_data_with_missing():
    """Create sample data with missing values."""
    df = pd.DataFrame({
        'feature1': [1, 2, np.nan, 4, 5],
        'feature2': [10, np.nan, 30, 40, 50],
        'feature3': ['A', 'B', 'C', np.nan, 'E'],
        'target': [0, 1, 0, 1, 0]
    })
    return df


@pytest.fixture
def sample_data_with_duplicates():
    """Create sample data with duplicate rows."""
    df = pd.DataFrame({
        'feature1': [1, 2, 2, 4, 5],
        'feature2': [10, 20, 20, 40, 50],
        'target': [0, 1, 1, 1, 0]
    })
    return df


@pytest.fixture
def sample_imbalanced_data():
    """Create highly imbalanced dataset."""
    np.random.seed(42)
    n_normal = 900
    n_fraud = 100
    
    data = {
        'feature1': np.concatenate([
            np.random.normal(0, 1, n_normal),
            np.random.normal(2, 1, n_fraud)
        ]),
        'feature2': np.concatenate([
            np.random.normal(0, 1, n_normal),
            np.random.normal(-2, 1, n_fraud)
        ]),
        'target': np.concatenate([
            np.zeros(n_normal),
            np.ones(n_fraud)
        ])
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_ip_country_mapping():
    """Create sample IP to country mapping."""
    return pd.DataFrame({
        'lower_bound_ip_address': ['1.0.0.0', '2.0.0.0', '3.0.0.0'],
        'upper_bound_ip_address': ['1.255.255.255', '2.255.255.255', '3.255.255.255'],
        'country': ['CountryA', 'CountryB', 'CountryC']
    })


@pytest.fixture
def sample_numeric_features():
    """Create sample numeric features for transformation."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.normal(100, 15, 100),
        'feature2': np.random.normal(50, 10, 100),
        'feature3': np.random.normal(200, 30, 100)
    })


@pytest.fixture
def sample_categorical_features():
    """Create sample categorical features for encoding."""
    return pd.DataFrame({
        'category1': np.random.choice(['A', 'B', 'C'], 100),
        'category2': np.random.choice(['X', 'Y'], 100),
        'category3': np.random.choice(['P', 'Q', 'R', 'S'], 100)
    })


@pytest.fixture
def sample_model_data():
    """Create sample data for model training."""
    np.random.seed(42)
    n_samples = 500
    
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples)
    })
    
    y = pd.Series(np.random.choice([0, 1], n_samples, p=[0.8, 0.2]))
    
    return X, y
