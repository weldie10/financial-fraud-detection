"""
Data Preparation Module

This module provides functionality for preparing data for model training,
including stratified train-test split and feature-target separation.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split


class DataPreparator:
    """
    A class to prepare data for model training.
    
    Attributes:
        logger (logging.Logger): Logger instance
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the DataPreparator.
        
        Args:
            logger (logging.Logger, optional): Logger instance
        """
        self.logger = logger or self._setup_logger()
    
    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Setup a logger for the DataPreparator."""
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
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training by splitting into train/test sets.
        
        Args:
            df (pd.DataFrame): Complete dataset
            target_column (str): Name of target column
            test_size (float): Proportion of test set (default 0.2)
            random_state (int): Random state for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        try:
            if target_column not in df.columns:
                raise ValueError(
                    f"Target column '{target_column}' not found. "
                    f"Available columns: {list(df.columns)}"
                )
            
            self.logger.info(f"Preparing data with target column: {target_column}")
            self.logger.info(f"Dataset shape: {df.shape}")
            
            # Separate features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Check class distribution
            class_dist = y.value_counts()
            self.logger.info(f"Class distribution:")
            for cls, count in class_dist.items():
                pct = (count / len(y)) * 100
                self.logger.info(f"  Class {cls}: {count:,} ({pct:.2f}%)")
            
            # Stratified train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y  # Preserve class distribution
            )
            
            self.logger.info(f"\nTrain set: {X_train.shape[0]:,} samples")
            self.logger.info(f"Test set: {X_test.shape[0]:,} samples")
            
            # Verify stratification
            train_dist = y_train.value_counts(normalize=True)
            test_dist = y_test.value_counts(normalize=True)
            
            self.logger.info(f"\nTrain set class distribution:")
            for cls in sorted(train_dist.index):
                self.logger.info(f"  Class {cls}: {train_dist[cls]*100:.2f}%")
            
            self.logger.info(f"Test set class distribution:")
            for cls in sorted(test_dist.index):
                self.logger.info(f"  Class {cls}: {test_dist[cls]*100:.2f}%")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise

