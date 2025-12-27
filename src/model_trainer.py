"""
Model Trainer Module

This module provides functionality to train baseline and ensemble models
for fraud detection with proper error handling and logging.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
from pathlib import Path


class ModelTrainer:
    """
    A class to train baseline and ensemble models for fraud detection.
    
    Attributes:
        logger (logging.Logger): Logger instance
        models (dict): Dictionary of trained models
        model_configs (dict): Model configurations
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the ModelTrainer.
        
        Args:
            logger (logging.Logger, optional): Logger instance
        """
        self.logger = logger or self._setup_logger()
        self.models = {}
        self.model_configs = {}
    
    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Setup a logger for the ModelTrainer."""
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
    
    def train_baseline_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        class_weight: Optional[Dict[int, float]] = None,
        random_state: int = 42,
        **kwargs
    ) -> LogisticRegression:
        """
        Train a Logistic Regression baseline model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            class_weight (dict, optional): Class weights for imbalanced data
            random_state (int): Random state for reproducibility
            **kwargs: Additional arguments for LogisticRegression
            
        Returns:
            LogisticRegression: Trained model
        """
        try:
            self.logger.info("Training Logistic Regression baseline model...")
            
            # Default class_weight if not provided
            if class_weight is None:
                class_weight = 'balanced'
            
            # Create and train model
            model = LogisticRegression(
                class_weight=class_weight,
                random_state=random_state,
                max_iter=1000,
                solver='lbfgs',
                **kwargs
            )
            
            model.fit(X_train, y_train)
            
            self.models['logistic_regression'] = model
            self.model_configs['logistic_regression'] = {
                'class_weight': class_weight,
                'random_state': random_state,
                **kwargs
            }
            
            self.logger.info("Logistic Regression model trained successfully")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error training baseline model: {str(e)}")
            raise
    
    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        class_weight: Optional[Dict[int, float]] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs
    ) -> RandomForestClassifier:
        """
        Train a Random Forest ensemble model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            n_estimators (int): Number of trees
            max_depth (int, optional): Maximum tree depth
            min_samples_split (int): Minimum samples to split
            min_samples_leaf (int): Minimum samples in leaf
            class_weight (dict, optional): Class weights
            random_state (int): Random state
            n_jobs (int): Number of parallel jobs
            **kwargs: Additional arguments for RandomForestClassifier
            
        Returns:
            RandomForestClassifier: Trained model
        """
        try:
            self.logger.info("Training Random Forest ensemble model...")
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                class_weight=class_weight,
                random_state=random_state,
                n_jobs=n_jobs,
                **kwargs
            )
            
            model.fit(X_train, y_train)
            
            self.models['random_forest'] = model
            self.model_configs['random_forest'] = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'class_weight': class_weight,
                'random_state': random_state,
                **kwargs
            }
            
            self.logger.info(
                f"Random Forest model trained successfully "
                f"(n_estimators={n_estimators}, max_depth={max_depth})"
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error training Random Forest model: {str(e)}")
            raise
    
    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        scale_pos_weight: Optional[float] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs
    ) -> XGBClassifier:
        """
        Train an XGBoost ensemble model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            n_estimators (int): Number of boosting rounds
            max_depth (int): Maximum tree depth
            learning_rate (float): Learning rate
            scale_pos_weight (float, optional): Scale for positive class (for imbalance)
            random_state (int): Random state
            n_jobs (int): Number of parallel jobs
            **kwargs: Additional arguments for XGBClassifier
            
        Returns:
            XGBClassifier: Trained model
        """
        try:
            self.logger.info("Training XGBoost ensemble model...")
            
            # Calculate scale_pos_weight if not provided
            if scale_pos_weight is None:
                scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
                self.logger.info(f"Auto-calculated scale_pos_weight: {scale_pos_weight:.2f}")
            
            model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                scale_pos_weight=scale_pos_weight,
                random_state=random_state,
                n_jobs=n_jobs,
                eval_metric='logloss',
                **kwargs
            )
            
            model.fit(X_train, y_train)
            
            self.models['xgboost'] = model
            self.model_configs['xgboost'] = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'scale_pos_weight': scale_pos_weight,
                'random_state': random_state,
                **kwargs
            }
            
            self.logger.info(
                f"XGBoost model trained successfully "
                f"(n_estimators={n_estimators}, max_depth={max_depth}, "
                f"learning_rate={learning_rate})"
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error training XGBoost model: {str(e)}")
            raise
    
    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        class_weight: Optional[Dict[int, float]] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs
    ) -> LGBMClassifier:
        """
        Train a LightGBM ensemble model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            n_estimators (int): Number of boosting rounds
            max_depth (int): Maximum tree depth (-1 for no limit)
            learning_rate (float): Learning rate
            class_weight (dict, optional): Class weights
            random_state (int): Random state
            n_jobs (int): Number of parallel jobs
            **kwargs: Additional arguments for LGBMClassifier
            
        Returns:
            LGBMClassifier: Trained model
        """
        try:
            self.logger.info("Training LightGBM ensemble model...")
            
            model = LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                class_weight=class_weight,
                random_state=random_state,
                n_jobs=n_jobs,
                verbose=-1,
                **kwargs
            )
            
            model.fit(X_train, y_train)
            
            self.models['lightgbm'] = model
            self.model_configs['lightgbm'] = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'class_weight': class_weight,
                'random_state': random_state,
                **kwargs
            }
            
            self.logger.info(
                f"LightGBM model trained successfully "
                f"(n_estimators={n_estimators}, max_depth={max_depth}, "
                f"learning_rate={learning_rate})"
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error training LightGBM model: {str(e)}")
            raise
    
    def save_model(self, model_name: str, filepath: str):
        """
        Save a trained model to disk.
        
        Args:
            model_name (str): Name of the model to save
            filepath (str): Path to save the model
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
            
            save_path = Path(filepath)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save model
            joblib.dump(self.models[model_name], save_path)
            
            # Save config
            config_path = save_path.with_suffix('.config.joblib')
            joblib.dump(self.model_configs[model_name], config_path)
            
            self.logger.info(f"Saved model '{model_name}' to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, model_name: str, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            model_name (str): Name to assign to the loaded model
            filepath (str): Path to load the model from
        """
        try:
            load_path = Path(filepath)
            if not load_path.exists():
                raise FileNotFoundError(f"Model file not found: {load_path}")
            
            # Load model
            model = joblib.load(load_path)
            self.models[model_name] = model
            
            # Load config if available
            config_path = load_path.with_suffix('.config.joblib')
            if config_path.exists():
                self.model_configs[model_name] = joblib.load(config_path)
            
            self.logger.info(f"Loaded model '{model_name}' from {load_path}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

