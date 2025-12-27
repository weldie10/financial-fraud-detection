"""
Hyperparameter Tuning Module

This module provides functionality for hyperparameter tuning using
grid search and random search with cross-validation.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, List
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')


class HyperparameterTuner:
    """
    A class to perform hyperparameter tuning for models.
    
    Attributes:
        logger (logging.Logger): Logger instance
        best_params (dict): Best hyperparameters found
        best_score (float): Best cross-validation score
    """
    
    def __init__(
        self,
        cv: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the HyperparameterTuner.
        
        Args:
            cv (int): Number of cross-validation folds
            random_state (int): Random state for reproducibility
            n_jobs (int): Number of parallel jobs
            logger (logging.Logger, optional): Logger instance
        """
        self.logger = logger or self._setup_logger()
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.best_params = {}
        self.best_score = {}
    
    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Setup a logger for the HyperparameterTuner."""
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
    
    def tune_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict[str, List]] = None,
        scoring: str = 'average_precision',
        n_iter: int = 20
    ) -> Dict[str, Any]:
        """
        Tune Random Forest hyperparameters.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            param_grid (dict, optional): Parameter grid. If None, uses default
            scoring (str): Scoring metric
            n_iter (int): Number of iterations for random search
            
        Returns:
            dict: Best parameters and score
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            
            self.logger.info("Tuning Random Forest hyperparameters...")
            
            # Default parameter grid
            if param_grid is None:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': ['balanced', None]
                }
            
            # Create base model
            base_model = RandomForestClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
            
            # Use RandomizedSearchCV for efficiency
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=n_iter,
                cv=StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state),
                scoring=scoring,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=1
            )
            
            search.fit(X_train, y_train)
            
            self.best_params['random_forest'] = search.best_params_
            self.best_score['random_forest'] = search.best_score_
            
            self.logger.info(f"Best parameters: {search.best_params_}")
            self.logger.info(f"Best {scoring} score: {search.best_score_:.4f}")
            
            return {
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'best_model': search.best_estimator_
            }
            
        except Exception as e:
            self.logger.error(f"Error tuning Random Forest: {str(e)}")
            raise
    
    def tune_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict[str, List]] = None,
        scoring: str = 'average_precision',
        n_iter: int = 20
    ) -> Dict[str, Any]:
        """
        Tune XGBoost hyperparameters.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            param_grid (dict, optional): Parameter grid. If None, uses default
            scoring (str): Scoring metric
            n_iter (int): Number of iterations for random search
            
        Returns:
            dict: Best parameters and score
        """
        try:
            from xgboost import XGBClassifier
            
            self.logger.info("Tuning XGBoost hyperparameters...")
            
            # Calculate scale_pos_weight
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            
            # Default parameter grid
            if param_grid is None:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'scale_pos_weight': [scale_pos_weight, scale_pos_weight * 0.5, scale_pos_weight * 1.5]
                }
            
            # Create base model
            base_model = XGBClassifier(
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                eval_metric='logloss'
            )
            
            # Use RandomizedSearchCV
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=n_iter,
                cv=StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state),
                scoring=scoring,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=1
            )
            
            search.fit(X_train, y_train)
            
            self.best_params['xgboost'] = search.best_params_
            self.best_score['xgboost'] = search.best_score_
            
            self.logger.info(f"Best parameters: {search.best_params_}")
            self.logger.info(f"Best {scoring} score: {search.best_score_:.4f}")
            
            return {
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'best_model': search.best_estimator_
            }
            
        except Exception as e:
            self.logger.error(f"Error tuning XGBoost: {str(e)}")
            raise
    
    def tune_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict[str, List]] = None,
        scoring: str = 'average_precision',
        n_iter: int = 20
    ) -> Dict[str, Any]:
        """
        Tune LightGBM hyperparameters.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            param_grid (dict, optional): Parameter grid. If None, uses default
            scoring (str): Scoring metric
            n_iter (int): Number of iterations for random search
            
        Returns:
            dict: Best parameters and score
        """
        try:
            from lightgbm import LGBMClassifier
            
            self.logger.info("Tuning LightGBM hyperparameters...")
            
            # Default parameter grid
            if param_grid is None:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, 10, -1],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'class_weight': ['balanced', None]
                }
            
            # Create base model
            base_model = LGBMClassifier(
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=-1
            )
            
            # Use RandomizedSearchCV
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=n_iter,
                cv=StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state),
                scoring=scoring,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=1
            )
            
            search.fit(X_train, y_train)
            
            self.best_params['lightgbm'] = search.best_params_
            self.best_score['lightgbm'] = search.best_score_
            
            self.logger.info(f"Best parameters: {search.best_params_}")
            self.logger.info(f"Best {scoring} score: {search.best_score_:.4f}")
            
            return {
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'best_model': search.best_estimator_
            }
            
        except Exception as e:
            self.logger.error(f"Error tuning LightGBM: {str(e)}")
            raise

