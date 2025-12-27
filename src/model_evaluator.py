"""
Model Evaluator Module

This module provides functionality to evaluate models using appropriate metrics
for imbalanced classification problems.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ModelEvaluator:
    """
    A class to evaluate classification models with appropriate metrics.
    
    Attributes:
        logger (logging.Logger): Logger instance
        output_dir (Path): Directory to save evaluation plots
    """
    
    def __init__(
        self,
        output_dir: str = "models/evaluation_outputs",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the ModelEvaluator.
        
        Args:
            output_dir (str): Directory to save evaluation plots
            logger (logging.Logger, optional): Logger instance
        """
        self.logger = logger or self._setup_logger()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Setup a logger for the ModelEvaluator."""
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
    
    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "Model",
        plot: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a model using multiple metrics.
        
        Args:
            model: Trained model with predict and predict_proba methods
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            model_name (str): Name of the model for logging
            plot (bool): Whether to generate plots
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        try:
            self.logger.info(f"Evaluating {model_name}...")
            
            # Get predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'model_name': model_name,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'pr_auc': average_precision_score(y_test, y_pred_proba),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            # Log metrics
            self.logger.info(f"{model_name} Metrics:")
            self.logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            self.logger.info(f"  Precision: {metrics['precision']:.4f}")
            self.logger.info(f"  Recall: {metrics['recall']:.4f}")
            self.logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
            self.logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            self.logger.info(f"  PR-AUC: {metrics['pr_auc']:.4f}")
            
            # Generate plots
            if plot:
                self._plot_confusion_matrix(
                    y_test, y_pred, model_name,
                    metrics['confusion_matrix']
                )
                self._plot_precision_recall_curve(
                    y_test, y_pred_proba, model_name, metrics['pr_auc']
                )
                self._plot_roc_curve(
                    y_test, y_pred_proba, model_name, metrics['roc_auc']
                )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def _plot_confusion_matrix(
        self,
        y_test: pd.Series,
        y_pred: np.ndarray,
        model_name: str,
        cm: list
    ):
        """Plot confusion matrix."""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                ax=ax,
                cbar_kws={'label': 'Count'}
            )
            
            ax.set_xlabel('Predicted', fontsize=12)
            ax.set_ylabel('Actual', fontsize=12)
            ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
            ax.set_xticklabels(['Normal', 'Fraud'])
            ax.set_yticklabels(['Normal', 'Fraud'])
            
            plt.tight_layout()
            plt.savefig(
                self.output_dir / f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png',
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Error plotting confusion matrix: {str(e)}")
    
    def _plot_precision_recall_curve(
        self,
        y_test: pd.Series,
        y_pred_proba: np.ndarray,
        model_name: str,
        pr_auc: float
    ):
        """Plot precision-recall curve."""
        try:
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(recall, precision, linewidth=2, label=f'{model_name} (PR-AUC = {pr_auc:.4f})')
            ax.axhline(y=y_test.mean(), color='r', linestyle='--', label='Baseline (Random)')
            ax.set_xlabel('Recall', fontsize=12)
            ax.set_ylabel('Precision', fontsize=12)
            ax.set_title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(
                self.output_dir / f'{model_name.lower().replace(" ", "_")}_pr_curve.png',
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Error plotting PR curve: {str(e)}")
    
    def _plot_roc_curve(
        self,
        y_test: pd.Series,
        y_pred_proba: np.ndarray,
        model_name: str,
        roc_auc: float
    ):
        """Plot ROC curve."""
        try:
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(fpr, tpr, linewidth=2, label=f'{model_name} (ROC-AUC = {roc_auc:.4f})')
            ax.plot([0, 1], [0, 1], 'r--', label='Baseline (Random)')
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(
                self.output_dir / f'{model_name.lower().replace(" ", "_")}_roc_curve.png',
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Error plotting ROC curve: {str(e)}")
    
    def compare_models(
        self,
        results: Dict[str, Dict[str, Any]],
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compare multiple models side-by-side.
        
        Args:
            results (dict): Dictionary of model evaluation results
            output_file (str, optional): Path to save comparison table
            
        Returns:
            pd.DataFrame: Comparison table
        """
        try:
            self.logger.info("Comparing models...")
            
            # Create comparison dataframe
            comparison_data = []
            for model_name, metrics in results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': metrics.get('accuracy', 0),
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0),
                    'F1-Score': metrics.get('f1_score', 0),
                    'ROC-AUC': metrics.get('roc_auc', 0),
                    'PR-AUC': metrics.get('pr_auc', 0)
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('PR-AUC', ascending=False)
            
            # Save to file
            if output_file:
                output_path = self.output_dir / output_file
                comparison_df.to_csv(output_path, index=False)
                self.logger.info(f"Saved comparison table to {output_path}")
            
            # Log comparison
            self.logger.info("\nModel Comparison:")
            self.logger.info(comparison_df.to_string(index=False))
            
            return comparison_df
            
        except Exception as e:
            self.logger.error(f"Error comparing models: {str(e)}")
            raise

