"""
Comprehensive unit tests for DataLoader module.

Tests cover:
- CSV loading with various scenarios
- Data validation
- Error handling
- Edge cases
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data_loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    def test_initialization(self, temp_data_dir):
        """Test DataLoader initialization."""
        loader = DataLoader(data_dir=str(temp_data_dir))
        assert loader.data_dir.exists()
        assert loader.logger is not None
    
    def test_initialization_creates_directory(self):
        """Test that DataLoader creates directory if it doesn't exist."""
        temp_dir = tempfile.mkdtemp()
        try:
            data_dir = Path(temp_dir) / "new_data_dir"
            loader = DataLoader(data_dir=str(data_dir))
            assert data_dir.exists()
        finally:
            shutil.rmtree(temp_dir)
    
    def test_load_csv_success(self, temp_data_dir, sample_fraud_data):
        """Test successful CSV loading."""
        # Save test data
        csv_path = temp_data_dir / "test_data.csv"
        sample_fraud_data.to_csv(csv_path, index=False)
        
        loader = DataLoader(data_dir=str(temp_data_dir))
        df = loader.load_csv("test_data.csv")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_fraud_data)
        assert list(df.columns) == list(sample_fraud_data.columns)
    
    def test_load_csv_file_not_found(self, temp_data_dir):
        """Test loading non-existent file raises FileNotFoundError."""
        loader = DataLoader(data_dir=str(temp_data_dir))
        
        with pytest.raises(FileNotFoundError):
            loader.load_csv("nonexistent.csv")
    
    def test_load_csv_empty_file(self, temp_data_dir):
        """Test loading empty CSV raises EmptyDataError."""
        # Create empty CSV
        csv_path = temp_data_dir / "empty.csv"
        csv_path.write_text("")
        
        loader = DataLoader(data_dir=str(temp_data_dir))
        
        with pytest.raises(pd.errors.EmptyDataError):
            loader.load_csv("empty.csv")
    
    def test_load_csv_with_kwargs(self, temp_data_dir, sample_fraud_data):
        """Test loading CSV with additional pandas arguments."""
        csv_path = temp_data_dir / "test_data.csv"
        sample_fraud_data.to_csv(csv_path, index=False)
        
        loader = DataLoader(data_dir=str(temp_data_dir))
        df = loader.load_csv("test_data.csv", nrows=10)
        
        assert len(df) == 10
    
    def test_load_multiple_csvs(self, temp_data_dir, sample_fraud_data, sample_credit_card_data):
        """Test loading multiple CSV files."""
        # Save multiple files
        sample_fraud_data.to_csv(temp_data_dir / "fraud_data.csv", index=False)
        sample_credit_card_data.to_csv(temp_data_dir / "credit_card.csv", index=False)
        
        loader = DataLoader(data_dir=str(temp_data_dir))
        dataframes = loader.load_multiple_csvs(["fraud_data.csv", "credit_card.csv"])
        
        assert len(dataframes) == 2
        assert "fraud_data" in dataframes
        assert "credit_card" in dataframes
        assert isinstance(dataframes["fraud_data"], pd.DataFrame)
        assert isinstance(dataframes["credit_card"], pd.DataFrame)
    
    def test_load_multiple_csvs_with_missing(self, temp_data_dir, sample_fraud_data):
        """Test loading multiple CSVs when one is missing."""
        sample_fraud_data.to_csv(temp_data_dir / "fraud_data.csv", index=False)
        
        loader = DataLoader(data_dir=str(temp_data_dir))
        dataframes = loader.load_multiple_csvs(["fraud_data.csv", "missing.csv"])
        
        # Should only load the existing file
        assert len(dataframes) == 1
        assert "fraud_data" in dataframes
    
    def test_validate_dataframe_success(self, sample_fraud_data):
        """Test successful dataframe validation."""
        loader = DataLoader()
        result = loader.validate_dataframe(sample_fraud_data)
        assert result is True
    
    def test_validate_dataframe_empty(self):
        """Test validation fails for empty dataframe."""
        loader = DataLoader()
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="DataFrame is empty"):
            loader.validate_dataframe(empty_df)
    
    def test_validate_dataframe_min_rows(self, sample_fraud_data):
        """Test validation with minimum rows requirement."""
        loader = DataLoader()
        
        # Should pass with sufficient rows
        loader.validate_dataframe(sample_fraud_data, min_rows=100)
        
        # Should fail with insufficient rows
        with pytest.raises(ValueError, match="fewer than"):
            loader.validate_dataframe(sample_fraud_data, min_rows=10000)
    
    def test_validate_dataframe_required_columns(self, sample_fraud_data):
        """Test validation with required columns."""
        loader = DataLoader()
        
        # Should pass with all required columns
        loader.validate_dataframe(
            sample_fraud_data,
            required_columns=['user_id', 'purchase_value', 'class']
        )
        
        # Should fail with missing required columns
        with pytest.raises(ValueError, match="Missing required columns"):
            loader.validate_dataframe(
                sample_fraud_data,
                required_columns=['user_id', 'nonexistent_column']
            )
    
    def test_validate_dataframe_combined_checks(self, sample_fraud_data):
        """Test validation with multiple checks."""
        loader = DataLoader()
        
        result = loader.validate_dataframe(
            sample_fraud_data,
            required_columns=['user_id', 'class'],
            min_rows=500
        )
        assert result is True
