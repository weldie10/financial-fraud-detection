"""
Data Cleaning Module

This module provides functionality to clean datasets by handling
missing values, removing duplicates, and correcting data types.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, List
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder


class DataCleaner:
    """
    A class to clean datasets with reusable methods.
    
    Attributes:
        logger (logging.Logger): Logger instance for tracking operations
        imputation_strategy (str): Strategy for handling missing values
        imputer (SimpleImputer or KNNImputer): Imputer instance
    """
    
    def __init__(
        self, 
        imputation_strategy: str = "mean",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the DataCleaner.
        
        Args:
            imputation_strategy (str): Strategy for imputation ('mean', 'median', 
                                      'mode', 'knn', 'drop')
            logger (logging.Logger, optional): Logger instance
        """
        self.logger = logger or self._setup_logger()
        self.imputation_strategy = imputation_strategy
        self.imputer = None
        self._setup_imputer()
    
    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Setup a logger for the DataCleaner."""
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
    
    def _setup_imputer(self):
        """Setup the imputer based on strategy."""
        if self.imputation_strategy == "knn":
            self.imputer = KNNImputer(n_neighbors=5)
        elif self.imputation_strategy in ["mean", "median", "most_frequent"]:
            self.imputer = SimpleImputer(strategy=self.imputation_strategy)
        else:
            self.imputer = None
    
    def handle_missing_values(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]] = None,
        strategy: Optional[str] = None,
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataframe.
        
        Args:
            df (pd.DataFrame): DataFrame to clean
            columns (list[str], optional): Specific columns to process. 
                                          If None, processes all columns
            strategy (str, optional): Override instance strategy
            threshold (float): Drop columns if missing percentage > threshold
            
        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        try:
            df_cleaned = df.copy()
            strategy = strategy or self.imputation_strategy
            
            # Report missing values
            missing_counts = df_cleaned.isnull().sum()
            missing_pct = (missing_counts / len(df_cleaned)) * 100
            
            if missing_counts.sum() > 0:
                self.logger.info("Missing values detected:")
                for col, count in missing_counts[missing_counts > 0].items():
                    self.logger.info(
                        f"  {col}: {count} ({missing_pct[col]:.2f}%)"
                    )
            
            # Drop columns with too many missing values
            cols_to_drop = missing_pct[missing_pct > threshold * 100].index.tolist()
            if cols_to_drop:
                self.logger.warning(
                    f"Dropping columns with >{threshold*100}% missing values: {cols_to_drop}"
                )
                df_cleaned = df_cleaned.drop(columns=cols_to_drop)
            
            # Process specified columns or all numeric columns
            columns_to_process = columns or df_cleaned.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            columns_to_process = [c for c in columns_to_process if c in df_cleaned.columns]
            
            if strategy == "drop":
                self.logger.info("Dropping rows with missing values")
                df_cleaned = df_cleaned.dropna(subset=columns_to_process)
            elif strategy in ["mean", "median", "most_frequent", "knn"]:
                self.logger.info(f"Imputing missing values using {strategy} strategy")
                
                if strategy == "knn":
                    numeric_cols = df_cleaned[columns_to_process].select_dtypes(
                        include=[np.number]
                    ).columns.tolist()
                    if numeric_cols:
                        df_cleaned[numeric_cols] = self.imputer.fit_transform(
                            df_cleaned[numeric_cols]
                        )
                else:
                    for col in columns_to_process:
                        if df_cleaned[col].dtype in [np.number]:
                            imputer = SimpleImputer(strategy=strategy)
                            df_cleaned[col] = imputer.fit_transform(
                                df_cleaned[[col]]
                            ).ravel()
                        elif strategy == "most_frequent":
                            # For categorical columns, use mode
                            mode_value = df_cleaned[col].mode()
                            if len(mode_value) > 0:
                                df_cleaned[col].fillna(mode_value[0], inplace=True)
            
            remaining_missing = df_cleaned.isnull().sum().sum()
            if remaining_missing > 0:
                self.logger.warning(
                    f"Still {remaining_missing} missing values after imputation"
                )
            else:
                self.logger.info("All missing values handled successfully")
            
            return df_cleaned
            
        except Exception as e:
            self.logger.error(f"Error handling missing values: {str(e)}")
            raise
    
    def remove_duplicates(
        self, 
        df: pd.DataFrame, 
        subset: Optional[List[str]] = None,
        keep: str = "first"
    ) -> pd.DataFrame:
        """
        Remove duplicate rows from the dataframe.
        
        Args:
            df (pd.DataFrame): DataFrame to clean
            subset (list[str], optional): Columns to consider for duplicates
            keep (str): Which duplicates to keep ('first', 'last', False)
            
        Returns:
            pd.DataFrame: DataFrame with duplicates removed
        """
        try:
            initial_count = len(df)
            df_cleaned = df.drop_duplicates(subset=subset, keep=keep)
            duplicates_removed = initial_count - len(df_cleaned)
            
            if duplicates_removed > 0:
                self.logger.info(
                    f"Removed {duplicates_removed} duplicate rows "
                    f"({duplicates_removed/initial_count*100:.2f}%)"
                )
            else:
                self.logger.info("No duplicates found")
            
            return df_cleaned
            
        except Exception as e:
            self.logger.error(f"Error removing duplicates: {str(e)}")
            raise
    
    def correct_data_types(
        self, 
        df: pd.DataFrame, 
        dtype_mapping: Optional[Dict[str, Any]] = None,
        date_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Correct data types in the dataframe.
        
        Args:
            df (pd.DataFrame): DataFrame to clean
            dtype_mapping (dict, optional): Dictionary mapping column names to dtypes
            date_columns (list[str], optional): List of columns to convert to datetime
            
        Returns:
            pd.DataFrame: DataFrame with corrected data types
        """
        try:
            df_cleaned = df.copy()
            
            # Convert date columns
            if date_columns:
                for col in date_columns:
                    if col in df_cleaned.columns:
                        try:
                            df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
                            self.logger.info(f"Converted {col} to datetime")
                        except Exception as e:
                            self.logger.warning(
                                f"Failed to convert {col} to datetime: {str(e)}"
                            )
            
            # Apply dtype mapping
            if dtype_mapping:
                for col, dtype in dtype_mapping.items():
                    if col in df_cleaned.columns:
                        try:
                            df_cleaned[col] = df_cleaned[col].astype(dtype)
                            self.logger.info(f"Converted {col} to {dtype}")
                        except Exception as e:
                            self.logger.warning(
                                f"Failed to convert {col} to {dtype}: {str(e)}"
                            )
            
            return df_cleaned
            
        except Exception as e:
            self.logger.error(f"Error correcting data types: {str(e)}")
            raise
    
    def clean(
        self, 
        df: pd.DataFrame,
        handle_missing: bool = True,
        remove_dups: bool = True,
        correct_types: bool = True,
        dtype_mapping: Optional[Dict[str, Any]] = None,
        date_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Complete cleaning pipeline.
        
        Args:
            df (pd.DataFrame): DataFrame to clean
            handle_missing (bool): Whether to handle missing values
            remove_dups (bool): Whether to remove duplicates
            correct_types (bool): Whether to correct data types
            dtype_mapping (dict, optional): Data type mapping
            date_columns (list[str], optional): Date columns to convert
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        try:
            self.logger.info("Starting data cleaning pipeline")
            df_cleaned = df.copy()
            
            initial_shape = df_cleaned.shape
            
            if handle_missing:
                df_cleaned = self.handle_missing_values(df_cleaned)
            
            if remove_dups:
                df_cleaned = self.remove_duplicates(df_cleaned)
            
            if correct_types:
                df_cleaned = self.correct_data_types(
                    df_cleaned, 
                    dtype_mapping=dtype_mapping,
                    date_columns=date_columns
                )
            
            final_shape = df_cleaned.shape
            self.logger.info(
                f"Cleaning complete. Shape: {initial_shape} -> {final_shape}"
            )
            
            return df_cleaned
            
        except Exception as e:
            self.logger.error(f"Error in cleaning pipeline: {str(e)}")
            raise

