# Models Directory

This directory contains trained model artifacts and evaluation outputs.

## Structure

```
models/
├── *.joblib          # Saved model files
├── *.config.joblib   # Model configurations
└── evaluation_outputs/  # Evaluation visualizations and metrics
    ├── *_confusion_matrix.png
    ├── *_pr_curve.png
    ├── *_roc_curve.png
    └── model_comparison.csv
```

## Usage

### Saving Models

Models are automatically saved by the `ModelTrainer` class:

```python
from src.model_trainer import ModelTrainer

trainer = ModelTrainer()
model = trainer.train_random_forest(X_train, y_train)
trainer.save_model('random_forest', 'models/random_forest_model.joblib')
```

### Loading Models

```python
from src.model_trainer import ModelTrainer

trainer = ModelTrainer()
model = trainer.load_model('random_forest', 'models/random_forest_model.joblib')
```

## Model Artifacts

- **Model Files** (`.joblib`): Serialized trained models
- **Config Files** (`.config.joblib`): Model hyperparameters and configurations
- **Evaluation Outputs**: Visualizations and metrics from model evaluation

## Best Practices

- Version control model files (use descriptive names with timestamps)
- Document model performance metrics alongside saved models
- Keep model configurations for reproducibility
- Regular cleanup of old model versions

