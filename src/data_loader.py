"""
Data Loader Module

This module provides functionality to load and validate data files
with proper error handling and logging.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import os


class DataLoader:
    """
    A class to load and validate data files with error handling.
    
    Attributes:
        data_dir (Path): Path to the data directory
        logger (logging.Logger): Logger instance for tracking operations
    """
    
    def __init__(self, data_dir: str = "data/raw", logger: Optional[logging.Logger] = None):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir (str): Path to the raw data directory
            logger (logging.Logger, optional): Logger instance. If None, creates a new one.
        """
        self.data_dir = Path(data_dir)
        self.logger = logger or self._setup_logger()
        
        # Validate data directory exists
        if not self.data_dir.exists():
            self.logger.warning(f"Data directory {self.data_dir} does not exist. Creating it.")
            self.data_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Setup a logger for the DataLoader."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def load_csv(
        self, 
        filename: str, 
        **kwargs
    ) -> pd.DataFrame:
        """
        Load a CSV file with error handling.
        
        Args:
            filename (str): Name of the CSV file to load
            **kwargs: Additional arguments to pass to pd.read_csv
            
        Returns:
            pd.DataFrame: Loaded dataframe
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            pd.errors.EmptyDataError: If the file is empty
            ValueError: If the file cannot be parsed
        """
        filepath = self.data_dir / filename
        
        try:
            if not filepath.exists():
                raise FileNotFoundError(
                    f"File not found: {filepath}. "
                    f"Please ensure the file exists in {self.data_dir}"
                )
            
            self.logger.info(f"Loading CSV file: {filepath}")
            df = pd.read_csv(filepath, **kwargs)
            
            if df.empty:
                raise pd.errors.EmptyDataError(f"File {filepath} is empty")
            
            self.logger.info(
                f"Successfully loaded {filepath}. "
                f"Shape: {df.shape[0]} rows, {df.shape[1]} columns"
            )
            
            return df
            
        except FileNotFoundError as e:
            self.logger.error(str(e))
            raise
        except pd.errors.EmptyDataError as e:
            self.logger.error(str(e))
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error loading {filepath}: {str(e)}")
            raise ValueError(f"Failed to load {filepath}: {str(e)}")
    
    def load_multiple_csvs(
        self, 
        filenames: list[str], 
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Load multiple CSV files.
        
        Args:
            filenames (list[str]): List of CSV filenames to load
            **kwargs: Additional arguments to pass to pd.read_csv
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping filename to dataframe
        """
        dataframes = {}
        
        for filename in filenames:
            try:
                # Use filename without extension as key
                key = Path(filename).stem
                dataframes[key] = self.load_csv(filename, **kwargs)
            except Exception as e:
                self.logger.warning(f"Failed to load {filename}: {str(e)}")
                continue
        
        return dataframes
    
    def validate_dataframe(
        self, 
        df: pd.DataFrame, 
        required_columns: Optional[list[str]] = None,
        min_rows: int = 1
    ) -> bool:
        """
        Validate a dataframe structure.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            required_columns (list[str], optional): List of required column names
            min_rows (int): Minimum number of rows required
            
        Returns:
            bool: True if validation passes
            
        Raises:
            ValueError: If validation fails
        """
        try:
            if df.empty:
                raise ValueError("DataFrame is empty")
            
            if len(df) < min_rows:
                raise ValueError(f"DataFrame has fewer than {min_rows} rows")
            
            if required_columns:
                missing_cols = set(required_columns) - set(df.columns)
                if missing_cols:
                    raise ValueError(
                        f"Missing required columns: {missing_cols}. "
                        f"Available columns: {list(df.columns)}"
                    )
            
            self.logger.info("DataFrame validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"DataFrame validation failed: {str(e)}")
            raise

