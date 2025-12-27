"""
Cross-Validation Module

This module provides functionality for stratified K-Fold cross-validation
to reliably estimate model performance.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    accuracy_score
)
import warnings
warnings.filterwarnings('ignore')


class CrossValidator:
    """
    A class to perform stratified K-Fold cross-validation.
    
    Attributes:
        logger (logging.Logger): Logger instance
        cv_results (dict): Cross-validation results
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the CrossValidator.
        
        Args:
            n_splits (int): Number of folds (k)
            random_state (int): Random state for reproducibility
            logger (logging.Logger, optional): Logger instance
        """
        self.logger = logger or self._setup_logger()
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )
        self.cv_results = {}
    
    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Setup a logger for the CrossValidator."""
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
    
    def cross_validate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Perform stratified K-Fold cross-validation.
        
        Args:
            model: Model with fit and predict_proba methods
            X (pd.DataFrame): Features
            y (pd.Series): Target
            model_name (str): Name of the model
            
        Returns:
            dict: Cross-validation results with mean and std
        """
        try:
            self.logger.info(
                f"Performing {self.n_splits}-Fold Stratified Cross-Validation for {model_name}..."
            )
            
            # Store metrics for each fold
            fold_metrics = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1_score': [],
                'roc_auc': [],
                'pr_auc': []
            }
            
            # Perform cross-validation
            for fold, (train_idx, val_idx) in enumerate(self.cv.split(X, y), 1):
                self.logger.info(f"  Fold {fold}/{self.n_splits}...")
                
                # Split data
                X_train_fold = X.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_train_fold = y.iloc[train_idx]
                y_val_fold = y.iloc[val_idx]
                
                # Train model
                model_copy = self._clone_model(model)
                model_copy.fit(X_train_fold, y_train_fold)
                
                # Predict
                y_pred = model_copy.predict(X_val_fold)
                y_pred_proba = model_copy.predict_proba(X_val_fold)[:, 1]
                
                # Calculate metrics
                fold_metrics['accuracy'].append(
                    accuracy_score(y_val_fold, y_pred)
                )
                fold_metrics['precision'].append(
                    precision_score(y_val_fold, y_pred, zero_division=0)
                )
                fold_metrics['recall'].append(
                    recall_score(y_val_fold, y_pred, zero_division=0)
                )
                fold_metrics['f1_score'].append(
                    f1_score(y_val_fold, y_pred, zero_division=0)
                )
                fold_metrics['roc_auc'].append(
                    roc_auc_score(y_val_fold, y_pred_proba)
                )
                fold_metrics['pr_auc'].append(
                    average_precision_score(y_val_fold, y_pred_proba)
                )
            
            # Calculate mean and std
            results = {
                'model_name': model_name,
                'n_splits': self.n_splits
            }
            
            for metric_name, values in fold_metrics.items():
                results[f'{metric_name}_mean'] = np.mean(values)
                results[f'{metric_name}_std'] = np.std(values)
            
            # Log results
            self.logger.info(f"\n{model_name} Cross-Validation Results:")
            self.logger.info(f"  Accuracy: {results['accuracy_mean']:.4f} (+/- {results['accuracy_std']:.4f})")
            self.logger.info(f"  Precision: {results['precision_mean']:.4f} (+/- {results['precision_std']:.4f})")
            self.logger.info(f"  Recall: {results['recall_mean']:.4f} (+/- {results['recall_std']:.4f})")
            self.logger.info(f"  F1-Score: {results['f1_score_mean']:.4f} (+/- {results['f1_score_std']:.4f})")
            self.logger.info(f"  ROC-AUC: {results['roc_auc_mean']:.4f} (+/- {results['roc_auc_std']:.4f})")
            self.logger.info(f"  PR-AUC: {results['pr_auc_mean']:.4f} (+/- {results['pr_auc_std']:.4f})")
            
            self.cv_results[model_name] = results
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in cross-validation: {str(e)}")
            raise
    
    def _clone_model(self, model: Any) -> Any:
        """
        Create a copy of the model for cross-validation.
        
        Args:
            model: Model to clone
            
        Returns:
            Cloned model
        """
        from sklearn.base import clone
        try:
            return clone(model)
        except Exception:
            # Fallback for models that don't support cloning
            import copy
            return copy.deepcopy(model)
    
    def get_results_summary(self) -> pd.DataFrame:
        """
        Get summary of all cross-validation results.
        
        Returns:
            pd.DataFrame: Summary table
        """
        try:
            if not self.cv_results:
                self.logger.warning("No cross-validation results available")
                return pd.DataFrame()
            
            summary_data = []
            for model_name, results in self.cv_results.items():
                summary_data.append({
                    'Model': model_name,
                    'Accuracy (mean±std)': f"{results['accuracy_mean']:.4f}±{results['accuracy_std']:.4f}",
                    'Precision (mean±std)': f"{results['precision_mean']:.4f}±{results['precision_std']:.4f}",
                    'Recall (mean±std)': f"{results['recall_mean']:.4f}±{results['recall_std']:.4f}",
                    'F1-Score (mean±std)': f"{results['f1_score_mean']:.4f}±{results['f1_score_std']:.4f}",
                    'ROC-AUC (mean±std)': f"{results['roc_auc_mean']:.4f}±{results['roc_auc_std']:.4f}",
                    'PR-AUC (mean±std)': f"{results['pr_auc_mean']:.4f}±{results['pr_auc_std']:.4f}"
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.sort_values('PR-AUC (mean±std)', ascending=False)
            
            return summary_df
            
        except Exception as e:
            self.logger.error(f"Error creating results summary: {str(e)}")
            raise

