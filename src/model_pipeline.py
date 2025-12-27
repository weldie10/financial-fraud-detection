"""
Model Pipeline Module

This module provides a complete pipeline for model building, training,
evaluation, and comparison following OOP principles.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

try:
    from .data_preparator import DataPreparator
    from .model_trainer import ModelTrainer
    from .model_evaluator import ModelEvaluator
    from .cross_validator import CrossValidator
    from .hyperparameter_tuner import HyperparameterTuner
except ImportError:
    # Fallback for direct imports
    from data_preparator import DataPreparator
    from model_trainer import ModelTrainer
    from model_evaluator import ModelEvaluator
    from cross_validator import CrossValidator
    from hyperparameter_tuner import HyperparameterTuner


class ModelPipeline:
    """
    Main model pipeline class that orchestrates all model building steps.
    
    Attributes:
        logger (logging.Logger): Logger instance
        data_preparator (DataPreparator): Data preparation component
        model_trainer (ModelTrainer): Model training component
        model_evaluator (ModelEvaluator): Model evaluation component
        cross_validator (CrossValidator): Cross-validation component
        hyperparameter_tuner (HyperparameterTuner): Hyperparameter tuning component
    """
    
    def __init__(
        self,
        models_dir: str = "models",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the ModelPipeline.
        
        Args:
            models_dir (str): Directory to save models
            logger (logging.Logger, optional): Logger instance
        """
        self.logger = logger or self._setup_logger()
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_preparator = DataPreparator(logger=self.logger)
        self.model_trainer = ModelTrainer(logger=self.logger)
        self.model_evaluator = ModelEvaluator(
            output_dir=str(self.models_dir / "evaluation_outputs"),
            logger=self.logger
        )
        self.cross_validator = CrossValidator(logger=self.logger)
        self.hyperparameter_tuner = HyperparameterTuner(logger=self.logger)
        
        self.logger.info("Model pipeline initialized")
    
    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Setup a logger for the ModelPipeline."""
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
    
    def build_and_evaluate_models(
        self,
        df: pd.DataFrame,
        target_column: str,
        perform_cv: bool = True,
        tune_hyperparameters: bool = False,
        ensemble_model: str = "random_forest"
    ) -> Dict[str, Any]:
        """
        Complete model building and evaluation pipeline.
        
        Args:
            df (pd.DataFrame): Preprocessed dataset
            target_column (str): Name of target column
            perform_cv (bool): Whether to perform cross-validation
            tune_hyperparameters (bool): Whether to tune hyperparameters
            ensemble_model (str): Ensemble model to train ('random_forest', 'xgboost', 'lightgbm')
            
        Returns:
            dict: Results dictionary with all models and metrics
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("STARTING MODEL BUILDING AND EVALUATION PIPELINE")
            self.logger.info("=" * 80)
            
            results = {
                'models': {},
                'evaluation_results': {},
                'cv_results': {},
                'best_model': None
            }
            
            # Step 1: Prepare data
            self.logger.info("\n[Step 1/5] Preparing data...")
            X_train, X_test, y_train, y_test = self.data_preparator.prepare_data(
                df,
                target_column=target_column,
                test_size=0.2,
                random_state=42
            )
            
            # Step 2: Train baseline model
            self.logger.info("\n[Step 2/5] Training baseline model (Logistic Regression)...")
            baseline_model = self.model_trainer.train_baseline_model(
                X_train,
                y_train,
                class_weight='balanced',
                random_state=42
            )
            results['models']['logistic_regression'] = baseline_model
            
            # Step 3: Train ensemble model
            self.logger.info(f"\n[Step 3/5] Training ensemble model ({ensemble_model})...")
            
            if tune_hyperparameters:
                self.logger.info("  Performing hyperparameter tuning...")
                if ensemble_model == 'random_forest':
                    tune_results = self.hyperparameter_tuner.tune_random_forest(
                        X_train, y_train
                    )
                    ensemble_model_obj = tune_results['best_model']
                elif ensemble_model == 'xgboost':
                    tune_results = self.hyperparameter_tuner.tune_xgboost(
                        X_train, y_train
                    )
                    ensemble_model_obj = tune_results['best_model']
                elif ensemble_model == 'lightgbm':
                    tune_results = self.hyperparameter_tuner.tune_lightgbm(
                        X_train, y_train
                    )
                    ensemble_model_obj = tune_results['best_model']
                else:
                    raise ValueError(f"Unknown ensemble model: {ensemble_model}")
            else:
                # Train with default parameters
                if ensemble_model == 'random_forest':
                    ensemble_model_obj = self.model_trainer.train_random_forest(
                        X_train, y_train,
                        n_estimators=100,
                        max_depth=10,
                        class_weight='balanced',
                        random_state=42
                    )
                elif ensemble_model == 'xgboost':
                    ensemble_model_obj = self.model_trainer.train_xgboost(
                        X_train, y_train,
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42
                    )
                elif ensemble_model == 'lightgbm':
                    ensemble_model_obj = self.model_trainer.train_lightgbm(
                        X_train, y_train,
                        n_estimators=100,
                        max_depth=5,
                        learning_rate=0.1,
                        class_weight='balanced',
                        random_state=42
                    )
                else:
                    raise ValueError(f"Unknown ensemble model: {ensemble_model}")
            
            results['models'][ensemble_model] = ensemble_model_obj
            
            # Step 4: Evaluate models
            self.logger.info("\n[Step 4/5] Evaluating models...")
            
            # Evaluate baseline
            baseline_results = self.model_evaluator.evaluate_model(
                baseline_model,
                X_test,
                y_test,
                model_name="Logistic Regression",
                plot=True
            )
            results['evaluation_results']['logistic_regression'] = baseline_results
            
            # Evaluate ensemble
            ensemble_results = self.model_evaluator.evaluate_model(
                ensemble_model_obj,
                X_test,
                y_test,
                model_name=ensemble_model.replace('_', ' ').title(),
                plot=True
            )
            results['evaluation_results'][ensemble_model] = ensemble_results
            
            # Step 5: Cross-validation (optional)
            if perform_cv:
                self.logger.info("\n[Step 5/5] Performing cross-validation...")
                
                # CV for baseline
                baseline_cv = self.cross_validator.cross_validate(
                    baseline_model,
                    X_train,
                    y_train,
                    model_name="Logistic Regression"
                )
                results['cv_results']['logistic_regression'] = baseline_cv
                
                # CV for ensemble
                ensemble_cv = self.cross_validator.cross_validate(
                    ensemble_model_obj,
                    X_train,
                    y_train,
                    model_name=ensemble_model.replace('_', ' ').title()
                )
                results['cv_results'][ensemble_model] = ensemble_cv
            
            # Compare models and select best
            self.logger.info("\nComparing models and selecting best...")
            comparison_df = self.model_evaluator.compare_models(
                results['evaluation_results'],
                output_file="model_comparison.csv"
            )
            
            # Select best model based on PR-AUC (most important for imbalanced data)
            best_model_name = comparison_df.iloc[0]['Model'].lower().replace(' ', '_')
            results['best_model'] = {
                'name': best_model_name,
                'model': results['models'][best_model_name],
                'metrics': results['evaluation_results'][best_model_name]
            }
            
            self.logger.info(f"\nBest model selected: {best_model_name}")
            self.logger.info(f"  PR-AUC: {results['best_model']['metrics']['pr_auc']:.4f}")
            self.logger.info(f"  F1-Score: {results['best_model']['metrics']['f1_score']:.4f}")
            
            # Save best model
            self.model_trainer.save_model(
                best_model_name,
                str(self.models_dir / f"{best_model_name}_best_model.joblib")
            )
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("MODEL BUILDING PIPELINE COMPLETE")
            self.logger.info("=" * 80)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in model pipeline: {str(e)}")
            raise

