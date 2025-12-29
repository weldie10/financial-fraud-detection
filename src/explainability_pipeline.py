"""
Explainability Pipeline Module

This module provides a complete pipeline for model explainability analysis
using SHAP and generating business recommendations.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

from .model_explainer import ModelExplainer
from .business_recommender import BusinessRecommender

try:
    from .model_trainer import ModelTrainer
except ImportError:
    from model_trainer import ModelTrainer


class ExplainabilityPipeline:
    """
    Main explainability pipeline class that orchestrates SHAP analysis.
    
    Attributes:
        logger (logging.Logger): Logger instance
        model_explainer (ModelExplainer): SHAP explainability component
        business_recommender (BusinessRecommender): Recommendations component
    """
    
    def __init__(
        self,
        output_dir: str = "models/explainability_outputs",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the ExplainabilityPipeline.
        
        Args:
            output_dir (str): Directory to save explainability outputs
            logger (logging.Logger, optional): Logger instance
        """
        self.logger = logger or self._setup_logger()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model_explainer = ModelExplainer(
            output_dir=str(self.output_dir),
            logger=self.logger
        )
        self.business_recommender = BusinessRecommender(logger=self.logger)
        
        self.logger.info("Explainability pipeline initialized")
    
    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Setup a logger for the ExplainabilityPipeline."""
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
    
    def explain_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "Model",
        sample_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Complete explainability analysis pipeline.
        
        Args:
            model: Trained model to explain
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            model_name (str): Name of the model
            sample_size (int): Sample size for SHAP analysis
            
        Returns:
            dict: Complete explainability results
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("STARTING MODEL EXPLAINABILITY ANALYSIS")
            self.logger.info("=" * 80)
            
            results = {
                'model_name': model_name,
                'builtin_importance': None,
                'shap_importance': None,
                'comparison': None,
                'top_drivers': None,
                'force_plots': [],
                'recommendations': []
            }
            
            # Step 1: Extract built-in feature importance
            self.logger.info("\n[Step 1/6] Extracting built-in feature importance...")
            try:
                builtin_importance = self.model_explainer.extract_builtin_feature_importance(
                    model,
                    list(X_train.columns),
                    top_n=10
                )
                results['builtin_importance'] = builtin_importance
                
                # Visualize
                self.model_explainer.visualize_builtin_importance(
                    builtin_importance,
                    top_n=10,
                    model_name=model_name
                )
            except Exception as e:
                self.logger.warning(f"Could not extract built-in importance: {str(e)}")
            
            # Step 2: Create SHAP explainer
            self.logger.info("\n[Step 2/6] Creating SHAP explainer...")
            X_sample = X_train.sample(min(sample_size, len(X_train)), random_state=42)
            self.model_explainer.create_shap_explainer(model, X_sample)
            
            # Step 3: Generate SHAP summary plot
            self.logger.info("\n[Step 3/6] Generating SHAP summary plot...")
            X_test_sample = X_test.sample(min(sample_size, len(X_test)), random_state=42)
            shap_values = self.model_explainer.generate_shap_summary_plot(
                X_test_sample,
                max_display=20,
                model_name=model_name
            )
            
            # Step 4: Calculate SHAP importance
            self.logger.info("\n[Step 4/6] Calculating SHAP feature importance...")
            shap_importance = self.model_explainer.calculate_shap_importance(
                X_test_sample,
                top_n=10
            )
            results['shap_importance'] = shap_importance
            
            # Step 5: Compare importances
            if results['builtin_importance'] is not None:
                self.logger.info("\n[Step 5/6] Comparing built-in and SHAP importance...")
                comparison = self.model_explainer.compare_importance(
                    results['builtin_importance'],
                    shap_importance,
                    top_n=10,
                    model_name=model_name
                )
                results['comparison'] = comparison
            
            # Step 6: Analyze top drivers
            self.logger.info("\n[Step 6/6] Analyzing top fraud prediction drivers...")
            top_drivers = self.business_recommender.analyze_top_drivers(
                shap_importance,
                top_n=5
            )
            results['top_drivers'] = top_drivers
            
            # Generate force plots for specific cases
            self.logger.info("\nGenerating SHAP force plots for individual predictions...")
            y_pred = model.predict(X_test)
            
            # Find TP, FP, FN cases
            tp_indices = self.model_explainer.find_prediction_cases(y_pred, y_test, 'tp')
            fp_indices = self.model_explainer.find_prediction_cases(y_pred, y_test, 'fp')
            fn_indices = self.model_explainer.find_prediction_cases(y_pred, y_test, 'fn')
            
            # Generate force plots
            force_plots = []
            
            if tp_indices:
                tp_idx = tp_indices[0]
                tp_plot = self.model_explainer.generate_shap_force_plot(
                    X_test, y_pred, y_test, tp_idx, 'tp', model_name
                )
                force_plots.append(tp_plot)
            
            if fp_indices:
                fp_idx = fp_indices[0]
                fp_plot = self.model_explainer.generate_shap_force_plot(
                    X_test, y_pred, y_test, fp_idx, 'fp', model_name
                )
                force_plots.append(fp_plot)
            
            if fn_indices:
                fn_idx = fn_indices[0]
                fn_plot = self.model_explainer.generate_shap_force_plot(
                    X_test, y_pred, y_test, fn_idx, 'fn', model_name
                )
                force_plots.append(fn_plot)
            
            results['force_plots'] = force_plots
            
            # Generate business recommendations
            self.logger.info("\nGenerating business recommendations...")
            recommendations = self.business_recommender.generate_recommendations(
                top_drivers,
                force_plot_insights=force_plots,
                builtin_importance=results['builtin_importance']
            )
            results['recommendations'] = recommendations
            
            # Save recommendations
            self.business_recommender.save_recommendations(
                recommendations,
                output_file=str(self.output_dir / "business_recommendations.txt")
            )
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("EXPLAINABILITY ANALYSIS COMPLETE")
            self.logger.info("=" * 80)
            self.logger.info(f"Top 5 fraud drivers identified")
            self.logger.info(f"{len(recommendations)} business recommendations generated")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in explainability pipeline: {str(e)}")
            raise

