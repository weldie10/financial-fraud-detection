# Financial Fraud Detection

A comprehensive machine learning project for detecting financial fraud using various classification algorithms. This project implements a complete preprocessing pipeline with OOP-based modules for data cleaning, feature engineering, geolocation integration, and class imbalance handling.

## Project Structure

```
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml
├── data/                           # Add this folder to .gitignore
│   ├── raw/                      # Original datasets
│   └── processed/         # Cleaned and feature-engineered data
├── notebooks/
│   ├── __init__.py
│   ├── eda-fraud-data.ipynb
│   ├── eda-creditcard.ipynb
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb
│   ├── shap-explainability.ipynb
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading with validation
│   ├── data_cleaner.py         # Data cleaning operations
│   ├── geolocation.py          # IP to country mapping
│   ├── feature_engineer.py    # Feature engineering pipeline
│   ├── eda.py                 # Exploratory data analysis
│   ├── data_transformer.py    # Scaling and encoding
│   ├── imbalance_handler.py   # Class imbalance handling
│   └── preprocessor.py         # Main preprocessing pipeline
├── tests/
│   ├── __init__.py
├── models/                      # Saved model artifacts
├── scripts/
│   ├── __init__.py
│   └── README.md
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd financial-fraud-detection
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Datasets

This project supports two main datasets for fraud detection:

### 1. Credit Card Fraud Detection Dataset

**Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | [Zenodo](https://zenodo.org/record/7395559)

**Description:** 
- Contains transactions made by European cardholders in September 2013
- **Total transactions:** 284,807
- **Fraudulent transactions:** 492 (0.172% - highly imbalanced)
- Features are anonymized using PCA transformation for confidentiality

**Expected Columns:**
- `Time`: Seconds elapsed between each transaction and the first transaction
- `V1-V28`: Principal components from PCA transformation (anonymized features)
- `Amount`: Transaction amount
- `Class`: Target variable (0 = legitimate, 1 = fraudulent)

**Download Instructions:**
1. Visit [Kaggle Dataset Page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Sign in to your Kaggle account (or create one)
3. Click "Download" to get the dataset
4. Extract and place `creditcard.csv` in `data/raw/` directory

**Alternative Source (Zenodo):**
- Direct download: [Zenodo Record 7395559](https://zenodo.org/record/7395559)
- File size: ~150.8 MB

### 2. Fraud Data Dataset (Custom/Simulated)

**Description:**
This dataset contains transaction-level fraud data with user behavior patterns, IP addresses, and temporal features.

**Expected Columns:**
- `user_id`: Unique identifier for each user
- `purchase_time`: Timestamp of the purchase transaction
- `signup_time`: Timestamp when user signed up
- `ip_address`: IP address of the transaction (format: x.x.x.x)
- `class`: Target variable (0 = legitimate, 1 = fraudulent)
- `purchase_value` (optional): Transaction amount

**Note:** This dataset may need to be generated or obtained from your data source. The preprocessing pipeline is designed to handle this structure.

### 3. IP Address to Country Mapping Dataset

**Description:**
Mapping file that converts IP address ranges to countries for geolocation analysis.

**Expected Columns:**
- `lower_bound_ip_address`: Lower bound of IP address range (format: x.x.x.x)
- `upper_bound_ip_address`: Upper bound of IP address range (format: x.x.x.x)
- `country`: Country name or code

**Sources for IP to Country Mapping:**
- **MaxMind GeoIP2:** [MaxMind GeoLite2](https://dev.maxmind.com/geoip/geolite2-free-geolocation-data) (Free version available)
- **IP2Location:** [IP2Location LITE](https://lite.ip2location.com/) (Free database available)
- **DB-IP:** [DB-IP.com](https://db-ip.com/db/download/ip-to-country-lite) (Free CSV available)

**Download Instructions:**
1. Visit one of the sources above
2. Download the IP-to-country mapping CSV file
3. Ensure it has columns: `lower_bound_ip_address`, `upper_bound_ip_address`, `country`
4. Place the file as `IpAddress_to_Country.csv` in `data/raw/` directory

**Note:** You may need to convert the downloaded format to match the expected column names.

## Usage

### Quick Start - Complete Preprocessing Pipeline

The easiest way to use the preprocessing pipeline is through the `PreprocessingPipeline` class:

```python
from src.preprocessor import PreprocessingPipeline

# Initialize pipeline
pipeline = PreprocessingPipeline(
    data_dir="data/raw",
    output_dir="data/processed"
)

# Process fraud data (complete pipeline)
processed_df, metadata = pipeline.process_fraud_data(
    fraud_data_file="Fraud_Data.csv",
    ip_country_file="IpAddress_to_Country.csv",  # Optional
    target_column="class",
    user_column="user_id",
    purchase_datetime="purchase_time",
    signup_datetime="signup_time",
    ip_column="ip_address",
    perform_eda=True,
    handle_imbalance=True,
    save_processed=True
)

# Process credit card data
processed_df, metadata = pipeline.process_credit_card_data(
    credit_card_file="creditcard.csv",
    target_column="Class",
    perform_eda=True,
    handle_imbalance=True,
    save_processed=True
)
```

### Individual Module Usage

You can also use individual modules for specific tasks:

```python
from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.feature_engineer import FeatureEngineer
from src.geolocation import GeolocationMapper
from src.eda import EDA

# Load data
loader = DataLoader(data_dir="data/raw")
df = loader.load_csv("Fraud_Data.csv")

# Clean data
cleaner = DataCleaner(imputation_strategy="mean")
df_cleaned = cleaner.clean(df)

# Engineer features
feature_engineer = FeatureEngineer()
df_features = feature_engineer.engineer_all_features(
    df_cleaned,
    user_column="user_id",
    purchase_datetime="purchase_time",
    signup_datetime="signup_time"
)

# Perform EDA
eda = EDA(output_dir="notebooks/eda_outputs")
class_stats = eda.analyze_class_distribution(df_features, target_column="class")
```

### Jupyter Notebooks

The project includes several Jupyter notebooks for interactive analysis:

1. **`notebooks/eda-fraud-data.ipynb`** - Exploratory data analysis for fraud data
2. **`notebooks/eda-creditcard.ipynb`** - Exploratory data analysis for credit card data
3. **`notebooks/feature-engineering.ipynb`** - Feature engineering examples
4. **`notebooks/modeling.ipynb`** - Complete preprocessing pipeline demonstration
5. **`notebooks/shap-explainability.ipynb`** - Model explainability (to be implemented)

To run the notebooks:
```bash
jupyter notebook notebooks/
```

## Preprocessing Pipeline Features

The preprocessing pipeline includes:

1. **Data Loading & Validation**
   - CSV file loading with error handling
   - Dataframe structure validation
   - Missing file detection

2. **Data Cleaning**
   - Missing value imputation (mean, median, mode, KNN)
   - Duplicate removal
   - Data type correction
   - Date/time parsing

3. **Geolocation Integration**
   - IP address to integer conversion
   - Range-based country lookup
   - Fraud pattern analysis by country

4. **Feature Engineering**
   - Time-based features (hour, day, month, weekend, business hours)
   - Time since signup calculation
   - Transaction frequency (multiple time windows: 1H, 24H, 7D, 30D)
   - Transaction velocity

5. **Exploratory Data Analysis**
   - Univariate analysis with distributions
   - Bivariate analysis with correlations
   - Class distribution and imbalance metrics
   - Comprehensive summary reports

6. **Data Transformation**
   - Numerical feature scaling (StandardScaler, MinMaxScaler, RobustScaler)
   - Categorical feature encoding (One-Hot Encoding)
   - Transformer persistence for reproducibility

7. **Class Imbalance Handling**
   - SMOTE (Synthetic Minority Oversampling)
   - ADASYN, BorderlineSMOTE
   - Undersampling methods
   - Combined methods (SMOTE-Tomek, SMOTE-ENN)
   - Method justification and statistics

## Project Features

- ✅ **OOP Design**: Modular, reusable classes following single responsibility principle
- ✅ **Error Handling**: Comprehensive try/except blocks with detailed logging
- ✅ **Logging**: Built-in logging for tracking all operations
- ✅ **Type Hints**: Full type annotations for better code clarity
- ✅ **Documentation**: Comprehensive docstrings for all classes and methods
- ✅ **Reproducibility**: Save/load transformers and models
- ✅ **Visualization**: Built-in plotting capabilities for EDA
- ✅ **Flexibility**: Configurable parameters for all operations

## Requirements

See `requirements.txt` for full list of dependencies. Key packages include:

- pandas, numpy - Data manipulation
- scikit-learn - Machine learning and preprocessing
- imbalanced-learn - Class imbalance handling
- matplotlib, seaborn, plotly - Visualization
- jupyter - Notebook support
- pytest - Testing

## Testing

Run tests with:
```bash
pytest tests/ -v
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

TBD

