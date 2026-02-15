"""
Integration tests for complete model training pipeline.

Tests cover:
- End-to-end model training workflow
- Data preparation → Training → Evaluation
- Model persistence
- Cross-validation integration
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data_preparator import DataPreparator
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator
from cross_validator import CrossValidator
from imbalance_handler import ImbalanceHandler
from data_transformer import DataTransformer
from sklearn.linear_model import LogisticRegression


class TestModelPipeline:
    """Integration tests for complete model training pipeline."""
    
    def test_complete_training_pipeline(self, sample_imbalanced_data, temp_data_dir):
        """Test complete model training pipeline."""
        # Prepare data
        preparator = DataPreparator()
        X_train, X_test, y_train, y_test = preparator.prepare_data(
            sample_imbalanced_data,
            target_column='target',
            test_size=0.2,
            random_state=42
        )
        
        # Handle imbalance
        handler = ImbalanceHandler(method="smote", random_state=42)
        X_train_resampled, y_train_resampled, _ = handler.resample(X_train, y_train)
        
        # Transform data
        transformer = DataTransformer()
        X_train_transformed = transformer.fit_transform(X_train_resampled)
        X_test_transformed = transformer.transform(X_test)
        
        # Train model
        trainer = ModelTrainer()
        model = trainer.train_baseline_model(
            X_train_transformed,
            y_train_resampled,
            random_state=42
        )
        
        # Evaluate model
        evaluator = ModelEvaluator(output_dir=str(temp_data_dir))
        metrics = evaluator.evaluate_model(
            model,
            X_test_transformed,
            y_test,
            model_name="TestPipeline",
            plot=False
        )
        
        # Verify pipeline completed successfully
        assert model is not None
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert 'pr_auc' in metrics
        assert metrics['accuracy'] >= 0
        assert metrics['accuracy'] <= 1
    
    def test_pipeline_with_cross_validation(self, sample_imbalanced_data):
        """Test pipeline with cross-validation."""
        # Prepare data
        preparator = DataPreparator()
        X_train, X_test, y_train, y_test = preparator.prepare_data(
            sample_imbalanced_data,
            target_column='target',
            test_size=0.2,
            random_state=42
        )
        
        # Handle imbalance
        handler = ImbalanceHandler(method="smote", random_state=42)
        X_train_resampled, y_train_resampled, _ = handler.resample(X_train, y_train)
        
        # Transform data
        transformer = DataTransformer()
        X_train_transformed = transformer.fit_transform(X_train_resampled)
        
        # Cross-validate
        model = LogisticRegression(random_state=42, max_iter=1000)
        cv = CrossValidator(n_splits=3, random_state=42)
        cv_results = cv.cross_validate(
            model,
            X_train_transformed,
            y_train_resampled,
            model_name="CVTest"
        )
        
        assert 'accuracy_mean' in cv_results
        assert 'pr_auc_mean' in cv_results
        assert cv_results['n_splits'] == 3
    
    def test_pipeline_model_persistence(self, sample_imbalanced_data, temp_data_dir):
        """Test model persistence in pipeline."""
        # Prepare data
        preparator = DataPreparator()
        X_train, X_test, y_train, y_test = preparator.prepare_data(
            sample_imbalanced_data,
            target_column='target',
            test_size=0.2,
            random_state=42
        )
        
        # Handle imbalance
        handler = ImbalanceHandler(method="smote", random_state=42)
        X_train_resampled, y_train_resampled, _ = handler.resample(X_train, y_train)
        
        # Transform data
        transformer = DataTransformer()
        X_train_transformed = transformer.fit_transform(X_train_resampled)
        
        # Train and save model
        trainer = ModelTrainer()
        model = trainer.train_baseline_model(
            X_train_transformed,
            y_train_resampled,
            random_state=42
        )
        
        save_path = temp_data_dir / "saved_model.pkl"
        trainer.save_model('logistic_regression', str(save_path))
        
        # Load and verify
        loaded_model = trainer.load_model(str(save_path))
        assert loaded_model is not None
        
        # Test predictions match
        predictions_original = model.predict(X_train_transformed.iloc[:10])
        predictions_loaded = loaded_model.predict(X_train_transformed.iloc[:10])
        assert np.array_equal(predictions_original, predictions_loaded)
    
    def test_pipeline_multiple_models(self, sample_imbalanced_data, temp_data_dir):
        """Test pipeline with multiple model types."""
        # Prepare data
        preparator = DataPreparator()
        X_train, X_test, y_train, y_test = preparator.prepare_data(
            sample_imbalanced_data,
            target_column='target',
            test_size=0.2,
            random_state=42
        )
        
        # Handle imbalance
        handler = ImbalanceHandler(method="smote", random_state=42)
        X_train_resampled, y_train_resampled, _ = handler.resample(X_train, y_train)
        
        # Transform data
        transformer = DataTransformer()
        X_train_transformed = transformer.fit_transform(X_train_resampled)
        X_test_transformed = transformer.transform(X_test)
        
        # Train multiple models
        trainer = ModelTrainer()
        model1 = trainer.train_baseline_model(X_train_transformed, y_train_resampled, random_state=42)
        model2 = trainer.train_random_forest(X_train_transformed, y_train_resampled, n_estimators=50, random_state=42)
        
        # Evaluate both
        evaluator = ModelEvaluator(output_dir=str(temp_data_dir))
        metrics1 = evaluator.evaluate_model(model1, X_test_transformed, y_test, model_name="LR", plot=False)
        metrics2 = evaluator.evaluate_model(model2, X_test_transformed, y_test, model_name="RF", plot=False)
        
        # Compare models
        results = {
            'LogisticRegression': metrics1,
            'RandomForest': metrics2
        }
        comparison_df = evaluator.compare_models(results)
        
        assert len(comparison_df) == 2
        assert 'Model' in comparison_df.columns
