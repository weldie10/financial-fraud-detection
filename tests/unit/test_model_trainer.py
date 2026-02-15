"""
Comprehensive unit tests for ModelTrainer module.

Tests cover:
- Baseline model training (Logistic Regression)
- Ensemble model training (Random Forest, XGBoost, LightGBM)
- Model persistence (save/load)
- Class weight handling
- Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from model_trainer import ModelTrainer


class TestModelTrainer:
    """Test cases for ModelTrainer class."""
    
    def test_initialization(self):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer()
        assert trainer.logger is not None
        assert trainer.models == {}
        assert trainer.model_configs == {}
    
    def test_train_baseline_model(self, sample_model_data):
        """Test training baseline Logistic Regression model."""
        X_train, y_train = sample_model_data
        trainer = ModelTrainer()
        
        model = trainer.train_baseline_model(
            X_train,
            y_train,
            random_state=42
        )
        
        assert model is not None
        assert 'logistic_regression' in trainer.models
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    def test_train_baseline_model_with_class_weight(self, sample_model_data):
        """Test baseline model with custom class weights."""
        X_train, y_train = sample_model_data
        trainer = ModelTrainer()
        
        class_weight = {0: 1.0, 1: 5.0}
        model = trainer.train_baseline_model(
            X_train,
            y_train,
            class_weight=class_weight,
            random_state=42
        )
        
        assert model is not None
        assert trainer.model_configs['logistic_regression']['class_weight'] == class_weight
    
    def test_train_baseline_model_default_balanced(self, sample_model_data):
        """Test that baseline model defaults to balanced class weights."""
        X_train, y_train = sample_model_data
        trainer = ModelTrainer()
        
        model = trainer.train_baseline_model(X_train, y_train, random_state=42)
        
        # Should use 'balanced' by default
        config = trainer.model_configs['logistic_regression']
        assert config['class_weight'] == 'balanced'
    
    def test_train_random_forest(self, sample_model_data):
        """Test training Random Forest model."""
        X_train, y_train = sample_model_data
        trainer = ModelTrainer()
        
        model = trainer.train_random_forest(
            X_train,
            y_train,
            n_estimators=50,
            max_depth=10,
            random_state=42
        )
        
        assert model is not None
        assert 'random_forest' in trainer.models
        assert model.n_estimators == 50
        assert model.max_depth == 10
    
    def test_train_xgboost(self, sample_model_data):
        """Test training XGBoost model."""
        X_train, y_train = sample_model_data
        trainer = ModelTrainer()
        
        model = trainer.train_xgboost(
            X_train,
            y_train,
            n_estimators=50,
            max_depth=5,
            random_state=42
        )
        
        assert model is not None
        assert 'xgboost' in trainer.models
        assert model.n_estimators == 50
    
    def test_train_lightgbm(self, sample_model_data):
        """Test training LightGBM model."""
        X_train, y_train = sample_model_data
        trainer = ModelTrainer()
        
        model = trainer.train_lightgbm(
            X_train,
            y_train,
            n_estimators=50,
            max_depth=5,
            random_state=42
        )
        
        assert model is not None
        assert 'lightgbm' in trainer.models
        assert model.n_estimators == 50
    
    def test_save_and_load_model(self, sample_model_data, temp_data_dir):
        """Test saving and loading models."""
        X_train, y_train = sample_model_data
        trainer = ModelTrainer()
        
        # Train model
        model = trainer.train_baseline_model(X_train, y_train, random_state=42)
        
        # Save
        save_path = temp_data_dir / "model.pkl"
        trainer.save_model('logistic_regression', str(save_path))
        assert save_path.exists()
        
        # Load
        loaded_model = trainer.load_model(str(save_path))
        assert loaded_model is not None
        
        # Test predictions match
        predictions_original = model.predict(X_train.iloc[:10])
        predictions_loaded = loaded_model.predict(X_train.iloc[:10])
        assert np.array_equal(predictions_original, predictions_loaded)
    
    def test_save_model_not_trained(self, temp_data_dir):
        """Test saving non-existent model raises error."""
        trainer = ModelTrainer()
        
        with pytest.raises(KeyError):
            trainer.save_model('nonexistent_model', str(temp_data_dir / "model.pkl"))
    
    def test_load_model_file_not_found(self):
        """Test loading non-existent model file."""
        trainer = ModelTrainer()
        
        with pytest.raises(FileNotFoundError):
            trainer.load_model("nonexistent_model.pkl")
    
    def test_train_multiple_models(self, sample_model_data):
        """Test training multiple models."""
        X_train, y_train = sample_model_data
        trainer = ModelTrainer()
        
        # Train multiple models
        trainer.train_baseline_model(X_train, y_train, random_state=42)
        trainer.train_random_forest(X_train, y_train, n_estimators=50, random_state=42)
        trainer.train_xgboost(X_train, y_train, n_estimators=50, random_state=42)
        
        assert len(trainer.models) == 3
        assert 'logistic_regression' in trainer.models
        assert 'random_forest' in trainer.models
        assert 'xgboost' in trainer.models
    
    def test_model_predictions(self, sample_model_data):
        """Test that trained models can make predictions."""
        X_train, y_train = sample_model_data
        X_test = X_train.iloc[:10]
        
        trainer = ModelTrainer()
        model = trainer.train_baseline_model(X_train, y_train, random_state=42)
        
        # Test predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        assert len(predictions) == len(X_test)
        assert len(probabilities) == len(X_test)
        assert probabilities.shape[1] == 2  # Binary classification
        assert all(pred in [0, 1] for pred in predictions)
