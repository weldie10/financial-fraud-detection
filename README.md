# Financial Fraud Detection

A comprehensive machine learning project for detecting financial fraud using classification algorithms. Implements complete preprocessing and model training pipelines with OOP-based modules.

## Project Structure

```
├── data/
│   ├── raw/              # Source datasets (immutable)
│   └── processed/        # Transformed datasets and visualizations
├── src/                  # Production modules
│   ├── data_loader.py           # Data loading & validation
│   ├── data_cleaner.py          # Data cleaning
│   ├── geolocation.py           # IP to country mapping
│   ├── feature_engineer.py      # Feature engineering
│   ├── eda.py                   # Exploratory data analysis
│   ├── data_transformer.py      # Scaling & encoding
│   ├── imbalance_handler.py     # SMOTE/undersampling
│   ├── preprocessor.py          # Preprocessing pipeline
│   ├── data_preparator.py       # Train-test split
│   ├── model_trainer.py         # Model training
│   ├── model_evaluator.py       # Model evaluation
│   ├── cross_validator.py       # Cross-validation
│   ├── hyperparameter_tuner.py  # Hyperparameter tuning
│   └── model_pipeline.py        # Complete model pipeline
├── notebooks/            # Interactive analysis
│   ├── eda-fraud-data.ipynb
│   ├── eda-creditcard.ipynb
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb           # Task 2: Model building
│   └── shap-explainability.ipynb
├── models/              # Saved model artifacts
└── scripts/             # Utility scripts
```

## Quick Start

### 1. Setup

```bash
# Clone repository
git clone <repository-url>
cd financial-fraud-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Place your datasets in `data/raw/`:
- `Fraud_Data.csv` - E-commerce fraud data
- `creditcard.csv` - Credit card fraud data
- `IpAddress_to_Country.csv` - IP to country mapping (optional)

### 3. Run Preprocessing (Task 1)

```python
from src.preprocessor import PreprocessingPipeline

pipeline = PreprocessingPipeline()
processed_df, metadata = pipeline.process_fraud_data(
    fraud_data_file="Fraud_Data.csv",
    perform_eda=True,
    handle_imbalance=True
)
```

### 4. Train Models (Task 2)

```python
from src.model_pipeline import ModelPipeline

model_pipeline = ModelPipeline()
results = model_pipeline.build_and_evaluate_models(
    df=processed_df,
    target_column="class",
    perform_cv=True,
    ensemble_model="random_forest"
)
```

## Features

### Task 1: Data Preprocessing
- ✅ Data loading with validation
- ✅ Missing value imputation & duplicate removal
- ✅ IP to country geolocation mapping
- ✅ Feature engineering (temporal, velocity, frequency)
- ✅ Data transformation (scaling, encoding)
- ✅ Class imbalance handling (SMOTE, undersampling)
- ✅ Comprehensive EDA with visualizations

### Task 2: Model Building & Training
- ✅ Stratified train-test split (preserves class distribution)
- ✅ Baseline model: Logistic Regression with class weights
- ✅ Ensemble models: Random Forest, XGBoost, LightGBM
- ✅ Model evaluation: AUC-PR, F1-Score, ROC-AUC, Confusion Matrix
- ✅ Cross-validation: Stratified K-Fold (k=5)
- ✅ Hyperparameter tuning with RandomizedSearchCV
- ✅ Model comparison and selection
- ✅ Model persistence (save/load)

## Datasets

### Credit Card Fraud Detection
- **Source:** [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | [Zenodo](https://zenodo.org/record/7395559)
- **Size:** 284,807 transactions (492 fraudulent, 0.17%)
- **Features:** Time, V1-V28 (PCA), Amount, Class

### Fraud Data (E-commerce)
- **Expected columns:** user_id, signup_time, purchase_time, purchase_value, ip_address, class
- **Note:** Custom dataset structure

### IP to Country Mapping
- **Sources:** MaxMind GeoIP2, IP2Location, DB-IP
- **Format:** lower_bound_ip_address, upper_bound_ip_address, country

## Usage Examples

### Complete Pipeline

```python
# 1. Preprocess data
from src.preprocessor import PreprocessingPipeline
preprocessor = PreprocessingPipeline()
df, _ = preprocessor.process_fraud_data("Fraud_Data.csv")

# 2. Train and evaluate models
from src.model_pipeline import ModelPipeline
modeler = ModelPipeline()
results = modeler.build_and_evaluate_models(df, target_column="class")

# 3. Access best model
best_model = results['best_model']['model']
```

### Individual Modules

```python
from src.data_preparator import DataPreparator
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator

# Prepare data
prep = DataPreparator()
X_train, X_test, y_train, y_test = prep.prepare_data(df, "class")

# Train model
trainer = ModelTrainer()
model = trainer.train_random_forest(X_train, y_train)

# Evaluate
evaluator = ModelEvaluator()
metrics = evaluator.evaluate_model(model, X_test, y_test)
```

## Notebooks

- **`notebooks/modeling.ipynb`** - Complete model building pipeline (Task 2)
- **`notebooks/eda-fraud-data.ipynb`** - EDA for fraud data
- **`notebooks/feature-engineering.ipynb`** - Feature engineering examples

## Requirements

Key dependencies:
- pandas, numpy - Data manipulation
- scikit-learn - Machine learning
- xgboost, lightgbm - Ensemble models
- imbalanced-learn - Class imbalance handling
- matplotlib, seaborn - Visualization
- shap - Model explainability (Task 3)

See `requirements.txt` for complete list.

## Project Status

- ✅ **Task 1:** Data preprocessing pipeline complete
- ✅ **Task 2:** Model building and training complete
- 🔄 **Task 3:** Model explainability (SHAP) - In progress

## License

TBD
