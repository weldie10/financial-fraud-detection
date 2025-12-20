"""
Feature Engineering Module

This module provides functionality to create meaningful features
including transaction frequency, velocity, and time-based features.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict
from datetime import datetime, timedelta


class FeatureEngineer:
    """
    A class to engineer features from raw data.
    
    Attributes:
        logger (logging.Logger): Logger instance for tracking operations
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the FeatureEngineer.
        
        Args:
            logger (logging.Logger, optional): Logger instance
        """
        self.logger = logger or self._setup_logger()
    
    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Setup a logger for the FeatureEngineer."""
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
    
    def extract_time_features(
        self, 
        df: pd.DataFrame,
        datetime_column: str,
        prefix: str = ""
    ) -> pd.DataFrame:
        """
        Extract time-based features from datetime column.
        
        Args:
            df (pd.DataFrame): DataFrame with datetime column
            datetime_column (str): Name of datetime column
            prefix (str): Prefix for new feature names
            
        Returns:
            pd.DataFrame: DataFrame with time features added
        """
        try:
            if datetime_column not in df.columns:
                raise ValueError(f"Datetime column '{datetime_column}' not found")
            
            self.logger.info(f"Extracting time features from {datetime_column}")
            df_features = df.copy()
            
            # Ensure datetime type
            if not pd.api.types.is_datetime64_any_dtype(df_features[datetime_column]):
                df_features[datetime_column] = pd.to_datetime(
                    df_features[datetime_column], 
                    errors='coerce'
                )
            
            prefix = f"{prefix}_" if prefix else ""
            
            # Extract time features
            df_features[f'{prefix}hour_of_day'] = df_features[datetime_column].dt.hour
            df_features[f'{prefix}day_of_week'] = df_features[datetime_column].dt.dayofweek
            df_features[f'{prefix}day_of_month'] = df_features[datetime_column].dt.day
            df_features[f'{prefix}month'] = df_features[datetime_column].dt.month
            df_features[f'{prefix}year'] = df_features[datetime_column].dt.year
            df_features[f'{prefix}is_weekend'] = (
                df_features[datetime_column].dt.dayofweek >= 5
            ).astype(int)
            df_features[f'{prefix}is_business_hours'] = (
                (df_features[datetime_column].dt.hour >= 9) & 
                (df_features[datetime_column].dt.hour < 17)
            ).astype(int)
            
            self.logger.info(f"Created {7} time-based features")
            
            return df_features
            
        except Exception as e:
            self.logger.error(f"Error extracting time features: {str(e)}")
            raise
    
    def calculate_time_since_signup(
        self, 
        df: pd.DataFrame,
        signup_column: str,
        reference_column: str,
        output_column: str = "time_since_signup"
    ) -> pd.DataFrame:
        """
        Calculate time duration between signup and reference time.
        
        Args:
            df (pd.DataFrame): DataFrame with signup and reference datetime columns
            signup_column (str): Column name for signup datetime
            reference_column (str): Column name for reference datetime (e.g., purchase_time)
            output_column (str): Name for the output column
            
        Returns:
            pd.DataFrame: DataFrame with time_since_signup feature added
        """
        try:
            if signup_column not in df.columns:
                raise ValueError(f"Signup column '{signup_column}' not found")
            if reference_column not in df.columns:
                raise ValueError(f"Reference column '{reference_column}' not found")
            
            self.logger.info(
                f"Calculating time since signup from {signup_column} to {reference_column}"
            )
            
            df_features = df.copy()
            
            # Ensure datetime types
            df_features[signup_column] = pd.to_datetime(
                df_features[signup_column], 
                errors='coerce'
            )
            df_features[reference_column] = pd.to_datetime(
                df_features[reference_column], 
                errors='coerce'
            )
            
            # Calculate time difference in various units
            time_diff = df_features[reference_column] - df_features[signup_column]
            
            df_features[output_column] = time_diff.dt.total_seconds() / 3600  # hours
            df_features[f'{output_column}_days'] = time_diff.dt.days
            df_features[f'{output_column}_minutes'] = time_diff.dt.total_seconds() / 60
            
            # Handle negative values (data quality issue)
            negative_count = (df_features[output_column] < 0).sum()
            if negative_count > 0:
                self.logger.warning(
                    f"Found {negative_count} records with negative time_since_signup. "
                    f"Setting to 0."
                )
                df_features.loc[df_features[output_column] < 0, output_column] = 0
                df_features.loc[df_features[f'{output_column}_days'] < 0, f'{output_column}_days'] = 0
                df_features.loc[df_features[f'{output_column}_minutes'] < 0, f'{output_column}_minutes'] = 0
            
            self.logger.info(f"Created time_since_signup features")
            
            return df_features
            
        except Exception as e:
            self.logger.error(f"Error calculating time since signup: {str(e)}")
            raise
    
    def calculate_transaction_frequency(
        self, 
        df: pd.DataFrame,
        user_column: str,
        datetime_column: str,
        time_windows: List[str] = ["1H", "24H", "7D", "30D"]
    ) -> pd.DataFrame:
        """
        Calculate transaction frequency per user in different time windows.
        
        Args:
            df (pd.DataFrame): DataFrame with user and datetime information
            user_column (str): Column name for user identifier
            datetime_column (str): Column name for transaction datetime
            time_windows (list[str]): List of time windows (e.g., "1H", "24H", "7D")
            
        Returns:
            pd.DataFrame: DataFrame with transaction frequency features added
        """
        try:
            if user_column not in df.columns:
                raise ValueError(f"User column '{user_column}' not found")
            if datetime_column not in df.columns:
                raise ValueError(f"Datetime column '{datetime_column}' not found")
            
            self.logger.info(
                f"Calculating transaction frequency for {len(time_windows)} time windows"
            )
            
            df_features = df.copy()
            
            # Ensure datetime type
            if not pd.api.types.is_datetime64_any_dtype(df_features[datetime_column]):
                df_features[datetime_column] = pd.to_datetime(
                    df_features[datetime_column], 
                    errors='coerce'
                )
            
            # Sort by user and datetime
            df_features = df_features.sort_values([user_column, datetime_column])
            
            for window in time_windows:
                feature_name = f"txn_freq_{window}"
                self.logger.info(f"Calculating {feature_name}")
                
                # Group by user and calculate rolling count
                df_features[feature_name] = (
                    df_features.groupby(user_column)[datetime_column]
                    .transform(lambda x: x.rolling(window=window, on=x).count())
                )
                
                # Fill NaN with 0 (first transaction in window)
                df_features[feature_name] = df_features[feature_name].fillna(0)
            
            self.logger.info(f"Created {len(time_windows)} transaction frequency features")
            
            return df_features
            
        except Exception as e:
            self.logger.error(f"Error calculating transaction frequency: {str(e)}")
            raise
    
    def calculate_transaction_velocity(
        self, 
        df: pd.DataFrame,
        user_column: str,
        datetime_column: str,
        amount_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate transaction velocity (transactions per time unit).
        
        Args:
            df (pd.DataFrame): DataFrame with user and datetime information
            user_column (str): Column name for user identifier
            datetime_column (str): Column name for transaction datetime
            amount_column (str, optional): Column name for transaction amount
            
        Returns:
            pd.DataFrame: DataFrame with velocity features added
        """
        try:
            if user_column not in df.columns:
                raise ValueError(f"User column '{user_column}' not found")
            if datetime_column not in df.columns:
                raise ValueError(f"Datetime column '{datetime_column}' not found")
            
            self.logger.info("Calculating transaction velocity")
            
            df_features = df.copy()
            
            # Ensure datetime type
            if not pd.api.types.is_datetime64_any_dtype(df_features[datetime_column]):
                df_features[datetime_column] = pd.to_datetime(
                    df_features[datetime_column], 
                    errors='coerce'
                )
            
            # Sort by user and datetime
            df_features = df_features.sort_values([user_column, datetime_column])
            
            # Calculate time between consecutive transactions
            df_features['time_since_last_txn'] = (
                df_features.groupby(user_column)[datetime_column]
                .diff()
                .dt.total_seconds() / 3600  # hours
            )
            
            # Calculate velocity (1 / time_since_last_txn, handling division by zero)
            df_features['txn_velocity'] = np.where(
                df_features['time_since_last_txn'] > 0,
                1 / df_features['time_since_last_txn'],
                0
            )
            
            # If amount column exists, calculate amount velocity
            if amount_column and amount_column in df_features.columns:
                df_features['amount_velocity'] = (
                    df_features[amount_column] / 
                    (df_features['time_since_last_txn'] + 1)  # +1 to avoid division by zero
                )
            
            # Fill NaN values (first transaction for each user)
            df_features['time_since_last_txn'] = df_features['time_since_last_txn'].fillna(0)
            df_features['txn_velocity'] = df_features['txn_velocity'].fillna(0)
            
            self.logger.info("Created transaction velocity features")
            
            return df_features
            
        except Exception as e:
            self.logger.error(f"Error calculating transaction velocity: {str(e)}")
            raise
    
    def engineer_all_features(
        self,
        df: pd.DataFrame,
        user_column: str,
        purchase_datetime: str,
        signup_datetime: Optional[str] = None,
        amount_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            df (pd.DataFrame): Input dataframe
            user_column (str): Column name for user identifier
            purchase_datetime (str): Column name for purchase datetime
            signup_datetime (str, optional): Column name for signup datetime
            amount_column (str, optional): Column name for transaction amount
            
        Returns:
            pd.DataFrame: DataFrame with all engineered features
        """
        try:
            self.logger.info("Starting complete feature engineering pipeline")
            
            df_features = df.copy()
            
            # Extract time features
            df_features = self.extract_time_features(
                df_features, 
                purchase_datetime, 
                prefix="purchase"
            )
            
            # Calculate time since signup if signup datetime provided
            if signup_datetime and signup_datetime in df_features.columns:
                df_features = self.calculate_time_since_signup(
                    df_features,
                    signup_datetime,
                    purchase_datetime
                )
            
            # Calculate transaction frequency
            df_features = self.calculate_transaction_frequency(
                df_features,
                user_column,
                purchase_datetime
            )
            
            # Calculate transaction velocity
            df_features = self.calculate_transaction_velocity(
                df_features,
                user_column,
                purchase_datetime,
                amount_column
            )
            
            self.logger.info(
                f"Feature engineering complete. "
                f"Original features: {len(df.columns)}, "
                f"New features: {len(df_features.columns) - len(df.columns)}"
            )
            
            return df_features
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering pipeline: {str(e)}")
            raise

