"""
Data Transformation Module

This module provides functionality for scaling numerical features
and encoding categorical features.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import joblib
from pathlib import Path


class DataTransformer:
    """
    A class to transform data through scaling and encoding.
    
    Attributes:
        logger (logging.Logger): Logger instance
        scaler: Scaler instance (StandardScaler, MinMaxScaler, etc.)
        encoder: Encoder instance (OneHotEncoder)
        label_encoders: Dictionary of label encoders for categorical columns
        is_fitted: Whether transformers have been fitted
    """
    
    def __init__(
        self,
        scaling_method: str = "standard",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the DataTransformer.
        
        Args:
            scaling_method (str): Scaling method ('standard', 'minmax', 'robust')
            logger (logging.Logger, optional): Logger instance
        """
        self.logger = logger or self._setup_logger()
        self.scaling_method = scaling_method
        self.scaler = self._setup_scaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
        self.label_encoders = {}
        self.is_fitted = False
        self.numeric_columns = []
        self.categorical_columns = []
    
    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Setup a logger for the DataTransformer."""
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
    
    def _setup_scaler(self):
        """Setup the scaler based on method."""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        return scalers.get(self.scaling_method, StandardScaler())
    
    def identify_column_types(
        self, 
        df: pd.DataFrame,
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Identify numeric and categorical columns.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
            numeric_columns (list[str], optional): Explicit list of numeric columns
            categorical_columns (list[str], optional): Explicit list of categorical columns
            exclude_columns (list[str], optional): Columns to exclude from transformation
            
        Returns:
            dict: Dictionary with 'numeric' and 'categorical' keys
        """
        try:
            exclude_columns = exclude_columns or []
            
            if numeric_columns and categorical_columns:
                # Use provided lists
                self.numeric_columns = [c for c in numeric_columns if c in df.columns and 
                                      c not in exclude_columns]
                self.categorical_columns = [c for c in categorical_columns if c in df.columns and 
                                          c not in exclude_columns]
            else:
                # Auto-detect
                all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
                all_categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
                
                self.numeric_columns = [c for c in all_numeric if c not in exclude_columns]
                self.categorical_columns = [c for c in all_categorical if c not in exclude_columns]
            
            self.logger.info(
                f"Identified {len(self.numeric_columns)} numeric and "
                f"{len(self.categorical_columns)} categorical columns"
            )
            
            return {
                'numeric': self.numeric_columns,
                'categorical': self.categorical_columns
            }
            
        except Exception as e:
            self.logger.error(f"Error identifying column types: {str(e)}")
            raise
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fit transformers and transform the data.
        
        Args:
            df (pd.DataFrame): DataFrame to transform
            target_column (str, optional): Target column to exclude from transformation
            
        Returns:
            pd.DataFrame: Transformed dataframe
        """
        try:
            self.logger.info("Fitting and transforming data")
            
            exclude_cols = [target_column] if target_column else []
            self.identify_column_types(df, exclude_columns=exclude_cols)
            
            df_transformed = df.copy()
            
            # Scale numeric columns
            if self.numeric_columns:
                self.logger.info(f"Scaling {len(self.numeric_columns)} numeric columns")
                scaled_data = self.scaler.fit_transform(df_transformed[self.numeric_columns])
                scaled_df = pd.DataFrame(
                    scaled_data,
                    columns=[f"{col}_scaled" for col in self.numeric_columns],
                    index=df_transformed.index
                )
                df_transformed = pd.concat([df_transformed, scaled_df], axis=1)
                df_transformed = df_transformed.drop(columns=self.numeric_columns)
                # Rename scaled columns back to original names
                rename_dict = {f"{col}_scaled": col for col in self.numeric_columns}
                df_transformed = df_transformed.rename(columns=rename_dict)
            
            # Encode categorical columns
            if self.categorical_columns:
                self.logger.info(f"Encoding {len(self.categorical_columns)} categorical columns")
                encoded_data = self.encoder.fit_transform(
                    df_transformed[self.categorical_columns]
                )
                encoded_df = pd.DataFrame(
                    encoded_data,
                    columns=self.encoder.get_feature_names_out(self.categorical_columns),
                    index=df_transformed.index
                )
                df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
                df_transformed = df_transformed.drop(columns=self.categorical_columns)
            
            self.is_fitted = True
            self.logger.info("Transformation complete")
            
            return df_transformed
            
        except Exception as e:
            self.logger.error(f"Error in fit_transform: {str(e)}")
            raise
    
    def transform(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Transform data using fitted transformers.
        
        Args:
            df (pd.DataFrame): DataFrame to transform
            target_column (str, optional): Target column to exclude from transformation
            
        Returns:
            pd.DataFrame: Transformed dataframe
        """
        try:
            if not self.is_fitted:
                raise ValueError(
                    "Transformers not fitted. Call fit_transform first or load saved transformers."
                )
            
            self.logger.info("Transforming data")
            
            exclude_cols = [target_column] if target_column else []
            df_transformed = df.copy()
            
            # Scale numeric columns
            if self.numeric_columns:
                available_numeric = [c for c in self.numeric_columns if c in df_transformed.columns]
                if available_numeric:
                    scaled_data = self.scaler.transform(df_transformed[available_numeric])
                    scaled_df = pd.DataFrame(
                        scaled_data,
                        columns=[f"{col}_scaled" for col in available_numeric],
                        index=df_transformed.index
                    )
                    df_transformed = pd.concat([df_transformed, scaled_df], axis=1)
                    df_transformed = df_transformed.drop(columns=available_numeric)
                    # Rename scaled columns back to original names
                    rename_dict = {f"{col}_scaled": col for col in available_numeric}
                    df_transformed = df_transformed.rename(columns=rename_dict)
            
            # Encode categorical columns
            if self.categorical_columns:
                available_categorical = [
                    c for c in self.categorical_columns if c in df_transformed.columns
                ]
                if available_categorical:
                    encoded_data = self.encoder.transform(df_transformed[available_categorical])
                    encoded_df = pd.DataFrame(
                        encoded_data,
                        columns=self.encoder.get_feature_names_out(available_categorical),
                        index=df_transformed.index
                    )
                    df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
                    df_transformed = df_transformed.drop(columns=available_categorical)
            
            self.logger.info("Transformation complete")
            
            return df_transformed
            
        except Exception as e:
            self.logger.error(f"Error in transform: {str(e)}")
            raise
    
    def save_transformers(self, filepath: str):
        """
        Save fitted transformers to disk.
        
        Args:
            filepath (str): Path to save transformers
        """
        try:
            save_path = Path(filepath)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            transformers = {
                'scaler': self.scaler,
                'encoder': self.encoder,
                'numeric_columns': self.numeric_columns,
                'categorical_columns': self.categorical_columns,
                'scaling_method': self.scaling_method
            }
            
            joblib.dump(transformers, save_path)
            self.logger.info(f"Saved transformers to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving transformers: {str(e)}")
            raise
    
    def load_transformers(self, filepath: str):
        """
        Load fitted transformers from disk.
        
        Args:
            filepath (str): Path to load transformers from
        """
        try:
            load_path = Path(filepath)
            if not load_path.exists():
                raise FileNotFoundError(f"Transformer file not found: {load_path}")
            
            transformers = joblib.load(load_path)
            
            self.scaler = transformers['scaler']
            self.encoder = transformers['encoder']
            self.numeric_columns = transformers['numeric_columns']
            self.categorical_columns = transformers['categorical_columns']
            self.scaling_method = transformers.get('scaling_method', 'standard')
            self.is_fitted = True
            
            self.logger.info(f"Loaded transformers from {load_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading transformers: {str(e)}")
            raise

