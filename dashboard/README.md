# Fraud Detection Interactive Dashboard

A Streamlit-based web application that provides an interactive interface for fraud analysts, product managers, and business stakeholders to explore predictions, test scenarios, and understand fraud drivers.

## Features

### 🔮 Real-Time Predictions
- **Manual Entry**: Input transaction details and get instant fraud predictions
- **Batch Processing**: Upload CSV files for bulk transaction analysis
- **SHAP Explanations**: Understand why a transaction was flagged as fraud

### 📈 Model Performance
- View model comparison metrics (PR-AUC, F1-Score, ROC-AUC)
- Interactive visualizations of ROC and Precision-Recall curves
- Confusion matrices and performance comparisons

### 🔍 Fraud Drivers Analysis
- Global SHAP summary plots showing feature importance
- Top 5 fraud drivers identification
- Feature importance comparisons (built-in vs SHAP)

### 🧪 Scenario Testing
- Test different transaction scenarios
- Vary transaction parameters (amount, time since signup, etc.)
- Visualize how fraud probability changes with different values
- Understand model sensitivity to different features

## Installation

### Prerequisites

```bash
# Install Streamlit and dashboard dependencies
pip install streamlit plotly

# Or install all requirements including dashboard
pip install -r requirements.txt
```

## Running the Dashboard

### Quick Start

```bash
# From project root
streamlit run dashboard/app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

### With Custom Port

```bash
streamlit run dashboard/app.py --server.port 8502
```

## Usage Guide

### 0. Train a Model (First Time Setup)

**Before using the dashboard, you need to train and save a model.**

#### Quick Training (Recommended)

```bash
# From project root
python scripts/train_model.py
```

This script will:
- Load processed data from `data/processed/` (or create sample data if none exists)
- Train a Random Forest model
- Handle class imbalance with SMOTE
- Transform and scale features
- Evaluate the model
- Save the model to `models/fraud_detection_model.joblib`
- Save the transformer to `models/transformer.joblib`

#### Using Notebooks

1. Open `notebooks/modeling.ipynb`
2. Run all cells to train models
3. Models will be saved automatically to `models/` directory

#### Manual Training

```python
from src.model_trainer import ModelTrainer
import joblib

trainer = ModelTrainer()
model = trainer.train_random_forest(X_train, y_train)
joblib.dump(model, "models/fraud_detection_model.joblib")
```

### 1. Load a Model

1. Ensure you have trained and saved a model (see above)
2. Models should be saved in `models/` directory as `.joblib` files
3. Use the sidebar to select and load a model
4. Optionally load a transformer if you have one saved (automatically loaded if named `transformer.joblib`)

### 2. Make Predictions

**Manual Entry:**
- Go to "🔮 Predictions" page
- Select "📝 Manual Entry"
- Fill in transaction details
- Click "Analyze Transaction"
- View prediction, probability, and SHAP explanation

**Batch Processing:**
- Go to "🔮 Predictions" page
- Select "📁 Upload CSV"
- Upload a CSV file with transaction data
- Click "Predict All"
- Download results as CSV

### 3. Explore Model Performance

- Go to "📈 Model Performance" page
- View model comparison metrics
- Explore ROC and PR curves
- Analyze confusion matrices

### 4. Understand Fraud Drivers

- Go to "🔍 Fraud Drivers" page
- View SHAP summary plots
- See top fraud drivers
- Compare feature importance methods

### 5. Test Scenarios

- Go to "🧪 Scenario Testing" page
- Set base transaction parameters
- Select a feature to vary
- Run scenario analysis
- Visualize how fraud probability changes

## Dashboard Structure

```
dashboard/
├── app.py              # Main Streamlit application
└── README.md           # This file
```

## Data Requirements

### For Predictions

The dashboard expects transaction data with the following features (minimum):
- `Amount`: Transaction amount
- `time_since_signup`: Hours since user signup
- `age`: User age
- `hour_of_day`: Hour of transaction (0-23)
- `day_of_week`: Day of week (0-6)
- `transaction_count_1h`: Number of transactions in last hour

**Note**: If your model uses different features, you may need to modify the input forms in `app.py`.

### Model Files

- **Model**: Saved as `.joblib` file in `models/` directory
- **Transformer**: Optional, saved as `transformer.joblib` in `models/` directory
- **Metrics**: Optional, `model_comparison.csv` in `models/evaluation_outputs/`
- **SHAP Visualizations**: Optional, in `models/explainability_outputs/`

## Customization

### Adding New Features

To add new input fields:

1. Update the transaction form in `show_manual_prediction()`
2. Add the feature to `transaction_data` DataFrame
3. Ensure your model and transformer support the new feature

### Styling

Custom CSS is defined in the `main()` function. Modify the `st.markdown()` call with custom styles to change appearance.

### SHAP Integration

If SHAP is not available, the dashboard will still work but SHAP explanations will be disabled. Install SHAP:

```bash
pip install shap
```

## Troubleshooting

### No Models Found

**Error**: "❌ No models found in models/ directory"

**Solution**: Train a model first using one of these methods:

1. **Quick Training Script** (Easiest):
   ```bash
   python scripts/train_model.py
   ```

2. **Using Notebooks**:
   - Open `notebooks/modeling.ipynb`
   - Run all cells
   - Models will be saved automatically

3. **Manual Training**:
   ```python
   from src.model_trainer import ModelTrainer
   import joblib
   
   trainer = ModelTrainer()
   model = trainer.train_random_forest(X_train, y_train)
   joblib.dump(model, "models/fraud_detection_model.joblib")
   ```

After training, refresh the dashboard page.

### Model Not Loading

- Ensure model file exists in `models/` directory
- Check that model was saved correctly (`.joblib` format)
- Verify model file is not corrupted
- Try reloading the model from the sidebar

### Predictions Failing

- Ensure input data matches model's expected features
- Check if transformer is needed and loaded correctly
- Verify feature names match between input and model

### SHAP Not Working

- Install SHAP: `pip install shap`
- Some models may not support TreeExplainer (will fall back to KernelExplainer)
- Large datasets may be slow with KernelExplainer

### Port Already in Use

```bash
# Use a different port
streamlit run dashboard/app.py --server.port 8502
```

## Deployment

### Local Network Access

```bash
streamlit run dashboard/app.py --server.address 0.0.0.0
```

### Production Deployment

For production deployment, consider:

1. **Streamlit Cloud**: Deploy directly from GitHub
2. **Docker**: Containerize the application
3. **Cloud Platforms**: AWS, GCP, Azure with Streamlit hosting

### Docker Example

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Security Considerations

⚠️ **Important**: This dashboard is designed for internal use. For production:

- Add authentication (Streamlit Authenticator)
- Use HTTPS
- Restrict network access
- Validate all inputs
- Sanitize file uploads
- Implement rate limiting

## Support

For issues or questions:
1. Check the main project README
2. Review model training notebooks
3. Check that all dependencies are installed
4. Verify model files are in correct format

## License

Same as main project license.
