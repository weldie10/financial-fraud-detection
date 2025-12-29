"""
Model Explainability Module

This module provides functionality for model explainability using SHAP
and built-in feature importance methods.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")


class ModelExplainer:
    """
    A class to explain model predictions using SHAP and feature importance.
    
    Attributes:
        logger (logging.Logger): Logger instance
        output_dir (Path): Directory to save explainability outputs
        shap_explainer: SHAP explainer instance
        feature_names: List of feature names
    """
    
    def __init__(
        self,
        output_dir: str = "models/explainability_outputs",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the ModelExplainer.
        
        Args:
            output_dir (str): Directory to save explainability outputs
            logger (logging.Logger, optional): Logger instance
        """
        self.logger = logger or self._setup_logger()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shap_explainer = None
        self.feature_names = None
        
        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Setup a logger for the ModelExplainer."""
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
    
    def extract_builtin_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Extract built-in feature importance from ensemble model.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names (list[str]): List of feature names
            top_n (int): Number of top features to return
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        try:
            self.logger.info("Extracting built-in feature importance...")
            
            if not hasattr(model, 'feature_importances_'):
                raise ValueError(
                    "Model does not have feature_importances_ attribute. "
                    "This method works for tree-based models (Random Forest, XGBoost, LightGBM)."
                )
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Get top N
            top_features = importance_df.head(top_n)
            
            self.logger.info(f"Top {top_n} features by built-in importance:")
            for idx, row in top_features.iterrows():
                self.logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            return importance_df
            
        except Exception as e:
            self.logger.error(f"Error extracting feature importance: {str(e)}")
            raise
    
    def visualize_builtin_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 10,
        model_name: str = "Model"
    ):
        """
        Visualize built-in feature importance.
        
        Args:
            importance_df (pd.DataFrame): Feature importance dataframe
            top_n (int): Number of top features to visualize
            model_name (str): Name of the model
        """
        try:
            self.logger.info(f"Visualizing top {top_n} features...")
            
            top_features = importance_df.head(top_n)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(top_features)), top_features['importance'], color='steelblue')
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('Feature Importance', fontsize=12)
            ax.set_title(f'Top {top_n} Features - {model_name} (Built-in Importance)', 
                        fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.savefig(
                self.output_dir / f'{model_name.lower().replace(" ", "_")}_builtin_importance.png',
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()
            
            self.logger.info(f"Saved built-in importance plot to {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error visualizing feature importance: {str(e)}")
            raise
    
    def create_shap_explainer(
        self,
        model: Any,
        X_sample: pd.DataFrame,
        model_type: str = "auto"
    ):
        """
        Create SHAP explainer for the model.
        
        Args:
            model: Trained model
            X_sample (pd.DataFrame): Sample data for explainer
            model_type (str): Model type ('tree', 'linear', 'auto')
        """
        try:
            if not SHAP_AVAILABLE:
                raise ImportError(
                    "SHAP is not installed. Install with: pip install shap"
                )
            
            self.logger.info("Creating SHAP explainer...")
            
            self.feature_names = list(X_sample.columns)
            
            # Auto-detect model type
            if model_type == "auto":
                model_class = type(model).__name__.lower()
                if 'randomforest' in model_class or 'xgboost' in model_class or 'lightgbm' in model_class:
                    model_type = "tree"
                elif 'logistic' in model_class or 'linear' in model_class:
                    model_type = "linear"
                else:
                    model_type = "tree"  # Default
            
            # Create appropriate explainer
            if model_type == "tree":
                self.shap_explainer = shap.TreeExplainer(model)
                self.logger.info("Using TreeExplainer for tree-based model")
            elif model_type == "linear":
                self.shap_explainer = shap.LinearExplainer(model, X_sample)
                self.logger.info("Using LinearExplainer for linear model")
            else:
                # Fallback to KernelExplainer (slower but works for any model)
                self.shap_explainer = shap.KernelExplainer(
                    model.predict_proba,
                    X_sample.sample(min(100, len(X_sample)), random_state=42)
                )
                self.logger.info("Using KernelExplainer (slower, but universal)")
            
            self.logger.info("SHAP explainer created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating SHAP explainer: {str(e)}")
            raise
    
    def generate_shap_summary_plot(
        self,
        X_sample: pd.DataFrame,
        max_display: int = 20,
        model_name: str = "Model"
    ):
        """
        Generate SHAP summary plot (global feature importance).
        
        Args:
            X_sample (pd.DataFrame): Sample data to explain
            max_display (int): Maximum number of features to display
            model_name (str): Name of the model
        """
        try:
            if self.shap_explainer is None:
                raise ValueError("SHAP explainer not created. Call create_shap_explainer first.")
            
            self.logger.info("Generating SHAP summary plot...")
            
            # Calculate SHAP values (use sample for large datasets)
            sample_size = min(1000, len(X_sample))
            X_sample_shap = X_sample.sample(sample_size, random_state=42)
            
            shap_values = self.shap_explainer.shap_values(X_sample_shap)
            
            # Handle binary classification (shap_values is a list)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class
            
            # Create summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values,
                X_sample_shap,
                max_display=max_display,
                show=False
            )
            plt.title(f'SHAP Summary Plot - {model_name}', fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(
                self.output_dir / f'{model_name.lower().replace(" ", "_")}_shap_summary.png',
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()
            
            self.logger.info(f"Saved SHAP summary plot to {self.output_dir}")
            
            return shap_values
            
        except Exception as e:
            self.logger.error(f"Error generating SHAP summary plot: {str(e)}")
            raise
    
    def generate_shap_force_plot(
        self,
        X_sample: pd.DataFrame,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        instance_idx: int,
        prediction_type: str,
        model_name: str = "Model"
    ):
        """
        Generate SHAP force plot for individual prediction.
        
        Args:
            X_sample (pd.DataFrame): Sample data
            y_pred (np.ndarray): Model predictions
            y_true (np.ndarray): True labels
            instance_idx (int): Index of instance to explain
            prediction_type (str): Type of prediction ('tp', 'fp', 'fn')
            model_name (str): Name of the model
            
        Returns:
            dict: Prediction details and SHAP values
        """
        try:
            if self.shap_explainer is None:
                raise ValueError("SHAP explainer not created. Call create_shap_explainer first.")
            
            self.logger.info(f"Generating SHAP force plot for {prediction_type.upper()}...")
            
            # Get instance
            instance = X_sample.iloc[instance_idx:instance_idx+1]
            shap_values_instance = self.shap_explainer.shap_values(instance)
            
            # Handle binary classification
            if isinstance(shap_values_instance, list):
                shap_values_instance = shap_values_instance[1]
            
            # Get prediction details
            pred_value = y_pred[instance_idx]
            true_value = y_true.iloc[instance_idx] if hasattr(y_true, 'iloc') else y_true[instance_idx]
            pred_proba = self.shap_explainer.model.predict_proba(instance)[0, 1]
            
            # Get expected value (handle different explainer types)
            if hasattr(self.shap_explainer, 'expected_value'):
                if isinstance(self.shap_explainer.expected_value, (list, np.ndarray)):
                    expected_val = self.shap_explainer.expected_value[1] if len(self.shap_explainer.expected_value) > 1 else self.shap_explainer.expected_value[0]
                else:
                    expected_val = self.shap_explainer.expected_value
            else:
                # Fallback: calculate from model
                expected_val = self.shap_explainer.model.predict_proba(instance)[0, 1]
            
            # Create force plot
            plt.figure(figsize=(14, 4))
            try:
                shap.force_plot(
                    expected_val,
                    shap_values_instance[0],
                    instance.iloc[0],
                    matplotlib=True,
                    show=False
                )
            except Exception:
                # Alternative: use waterfall plot if force plot fails
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values_instance[0],
                        base_values=expected_val,
                        data=instance.iloc[0].values,
                        feature_names=self.feature_names
                    ),
                    show=False
                )
            plt.title(
                f'SHAP Force Plot - {prediction_type.upper()}\n'
                f'Predicted: {pred_value}, Actual: {true_value}, Probability: {pred_proba:.4f}',
                fontsize=12,
                fontweight='bold'
            )
            plt.tight_layout()
            plt.savefig(
                self.output_dir / f'{model_name.lower().replace(" ", "_")}_force_{prediction_type}_{instance_idx}.png',
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()
            
            result = {
                'instance_idx': instance_idx,
                'prediction_type': prediction_type,
                'predicted': pred_value,
                'actual': true_value,
                'probability': pred_proba,
                'shap_values': shap_values_instance[0],
                'feature_values': instance.iloc[0].to_dict()
            }
            
            self.logger.info(
                f"Saved force plot for {prediction_type.upper()} "
                f"(idx={instance_idx}, pred={pred_value}, actual={true_value})"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating force plot: {str(e)}")
            raise
    
    def find_prediction_cases(
        self,
        y_pred: np.ndarray,
        y_true: pd.Series,
        prediction_type: str
    ) -> List[int]:
        """
        Find indices of specific prediction cases.
        
        Args:
            y_pred (np.ndarray): Model predictions
            y_true (pd.Series): True labels
            prediction_type (str): 'tp', 'fp', 'fn', 'tn'
            
        Returns:
            list[int]: List of indices matching the case
        """
        try:
            y_true_array = y_true.values if hasattr(y_true, 'values') else y_true
            
            if prediction_type == 'tp':  # True Positive
                mask = (y_pred == 1) & (y_true_array == 1)
            elif prediction_type == 'fp':  # False Positive
                mask = (y_pred == 1) & (y_true_array == 0)
            elif prediction_type == 'fn':  # False Negative
                mask = (y_pred == 0) & (y_true_array == 1)
            elif prediction_type == 'tn':  # True Negative
                mask = (y_pred == 0) & (y_true_array == 0)
            else:
                raise ValueError(f"Unknown prediction type: {prediction_type}")
            
            indices = np.where(mask)[0].tolist()
            self.logger.info(f"Found {len(indices)} {prediction_type.upper()} cases")
            
            return indices
            
        except Exception as e:
            self.logger.error(f"Error finding prediction cases: {str(e)}")
            raise
    
    def calculate_shap_importance(
        self,
        X_sample: pd.DataFrame,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Calculate SHAP-based feature importance.
        
        Args:
            X_sample (pd.DataFrame): Sample data
            top_n (int): Number of top features to return
            
        Returns:
            pd.DataFrame: SHAP importance dataframe
        """
        try:
            if self.shap_explainer is None:
                raise ValueError("SHAP explainer not created. Call create_shap_explainer first.")
            
            self.logger.info("Calculating SHAP feature importance...")
            
            # Calculate SHAP values
            sample_size = min(1000, len(X_sample))
            X_sample_shap = X_sample.sample(sample_size, random_state=42)
            
            shap_values = self.shap_explainer.shap_values(X_sample_shap)
            
            # Handle binary classification
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Calculate mean absolute SHAP values
            shap_importance = pd.DataFrame({
                'feature': self.feature_names,
                'shap_importance': np.abs(shap_values).mean(axis=0)
            }).sort_values('shap_importance', ascending=False)
            
            top_features = shap_importance.head(top_n)
            
            self.logger.info(f"Top {top_n} features by SHAP importance:")
            for idx, row in top_features.iterrows():
                self.logger.info(f"  {row['feature']}: {row['shap_importance']:.4f}")
            
            return shap_importance
            
        except Exception as e:
            self.logger.error(f"Error calculating SHAP importance: {str(e)}")
            raise
    
    def compare_importance(
        self,
        builtin_importance: pd.DataFrame,
        shap_importance: pd.DataFrame,
        top_n: int = 10,
        model_name: str = "Model"
    ) -> pd.DataFrame:
        """
        Compare built-in and SHAP feature importance.
        
        Args:
            builtin_importance (pd.DataFrame): Built-in importance
            shap_importance (pd.DataFrame): SHAP importance
            top_n (int): Number of top features to compare
            model_name (str): Name of the model
            
        Returns:
            pd.DataFrame: Comparison dataframe
        """
        try:
            self.logger.info("Comparing built-in and SHAP importance...")
            
            # Normalize importances for comparison
            builtin_norm = builtin_importance.copy()
            builtin_norm['importance'] = builtin_norm['importance'] / builtin_norm['importance'].max()
            
            shap_norm = shap_importance.copy()
            shap_norm['shap_importance'] = shap_norm['shap_importance'] / shap_norm['shap_importance'].max()
            
            # Merge
            comparison = builtin_norm.merge(
                shap_norm,
                on='feature',
                how='outer'
            ).fillna(0)
            
            # Get top features by SHAP
            top_shap = shap_importance.head(top_n)['feature'].tolist()
            comparison_top = comparison[comparison['feature'].isin(top_shap)].copy()
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(12, 8))
            x = np.arange(len(comparison_top))
            width = 0.35
            
            ax.bar(x - width/2, comparison_top['importance'], width, 
                  label='Built-in Importance', color='steelblue', alpha=0.8)
            ax.bar(x + width/2, comparison_top['shap_importance'], width,
                  label='SHAP Importance', color='coral', alpha=0.8)
            
            ax.set_xlabel('Features', fontsize=12)
            ax.set_ylabel('Normalized Importance', fontsize=12)
            ax.set_title(f'Feature Importance Comparison - {model_name}', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(comparison_top['feature'], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(
                self.output_dir / f'{model_name.lower().replace(" ", "_")}_importance_comparison.png',
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()
            
            self.logger.info(f"Saved importance comparison to {self.output_dir}")
            
            return comparison_top
            
        except Exception as e:
            self.logger.error(f"Error comparing importance: {str(e)}")
            raise

