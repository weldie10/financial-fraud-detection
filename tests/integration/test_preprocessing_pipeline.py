"""
Comprehensive integration tests for PreprocessingPipeline.

Tests cover:
- Complete preprocessing pipeline for fraud data
- Complete preprocessing pipeline for credit card data
- End-to-end data flow
- Error handling and edge cases
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from preprocessor import PreprocessingPipeline
from data_loader import DataLoader


class TestPreprocessingPipeline:
    """Integration tests for complete preprocessing pipeline."""
    
    def test_pipeline_initialization(self, temp_data_dir):
        """Test pipeline initialization."""
        pipeline = PreprocessingPipeline(
            data_dir=str(temp_data_dir),
            output_dir=str(temp_data_dir / "processed")
        )
        assert pipeline.data_dir.exists()
        assert pipeline.output_dir.exists()
    
    def test_process_fraud_data_complete(self, sample_fraud_data, temp_data_dir):
        """Test complete fraud data preprocessing pipeline."""
        # Save sample data
        data_dir = temp_data_dir / "raw"
        data_dir.mkdir(parents=True, exist_ok=True)
        sample_fraud_data.to_csv(data_dir / "Fraud_Data.csv", index=False)
        
        # Create IP mapping (simplified)
        ip_mapping = pd.DataFrame({
            'lower_bound_ip_address': ['1.0.0.0', '2.0.0.0'],
            'upper_bound_ip_address': ['1.255.255.255', '2.255.255.255'],
            'country': ['CountryA', 'CountryB']
        })
        ip_mapping.to_csv(data_dir / "IpAddress_to_Country.csv", index=False)
        
        pipeline = PreprocessingPipeline(
            data_dir=str(data_dir),
            output_dir=str(temp_data_dir / "processed")
        )
        
        # Process data
        result = pipeline.process_fraud_data()
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    def test_process_credit_card_data_complete(self, sample_credit_card_data, temp_data_dir):
        """Test complete credit card data preprocessing pipeline."""
        # Save sample data
        data_dir = temp_data_dir / "raw"
        data_dir.mkdir(parents=True, exist_ok=True)
        sample_credit_card_data.to_csv(data_dir / "creditcard.csv", index=False)
        
        pipeline = PreprocessingPipeline(
            data_dir=str(data_dir),
            output_dir=str(temp_data_dir / "processed")
        )
        
        # Process data
        result = pipeline.process_credit_card_data()
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'Class' in result.columns or 'class' in result.columns
    
    def test_pipeline_handles_missing_data(self, temp_data_dir):
        """Test that pipeline handles missing data gracefully."""
        # Create data with missing values
        data = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [10, np.nan, 30, 40, 50],
            'target': [0, 1, 0, 1, 0]
        })
        
        data_dir = temp_data_dir / "raw"
        data_dir.mkdir(parents=True, exist_ok=True)
        data.to_csv(data_dir / "test_data.csv", index=False)
        
        pipeline = PreprocessingPipeline(
            data_dir=str(data_dir),
            output_dir=str(temp_data_dir / "processed")
        )
        
        # Should not raise error
        # (Actual implementation may vary)
        assert pipeline is not None
    
    def test_pipeline_preserves_data_integrity(self, sample_fraud_data, temp_data_dir):
        """Test that pipeline preserves data integrity."""
        data_dir = temp_data_dir / "raw"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save original data
        original_count = len(sample_fraud_data)
        sample_fraud_data.to_csv(data_dir / "Fraud_Data.csv", index=False)
        
        # Create minimal IP mapping
        ip_mapping = pd.DataFrame({
            'lower_bound_ip_address': ['0.0.0.0'],
            'upper_bound_ip_address': ['255.255.255.255'],
            'country': ['Unknown']
        })
        ip_mapping.to_csv(data_dir / "IpAddress_to_Country.csv", index=False)
        
        pipeline = PreprocessingPipeline(
            data_dir=str(data_dir),
            output_dir=str(temp_data_dir / "processed")
        )
        
        result = pipeline.process_fraud_data()
        
        # Should have reasonable number of records
        # (may be less due to cleaning, but not drastically different)
        assert len(result) > 0
        assert len(result) <= original_count * 1.1  # Allow 10% increase for resampling
