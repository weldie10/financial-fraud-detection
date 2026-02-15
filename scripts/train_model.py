#!/usr/bin/env python3
"""
Quick Model Training Script

This script trains a simple fraud detection model and saves it for use in the dashboard.
It uses sample data if no data files are found, or loads from processed data if available.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model_trainer import ModelTrainer
from data_preparator import DataPreparator
from data_transformer import DataTransformer

# Try to import ImbalanceHandler, but handle import errors gracefully
try:
    from imbalance_handler import ImbalanceHandler
    IMBALANCE_HANDLER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: ImbalanceHandler not available: {str(e)}")
    print("   Continuing without resampling...")
    IMBALANCE_HANDLER_AVAILABLE = False
    ImbalanceHandler = None


def create_sample_data(n_samples=1000):
    """Create sample data for training if no real data is available."""
    print("📊 Creating sample training data...")
    
    np.random.seed(42)
    
    # Generate synthetic features
    data = {
        'Amount': np.random.exponential(88, n_samples),
        'time_since_signup': np.random.exponential(24, n_samples),
        'age': np.random.randint(18, 80, n_samples),
        'hour_of_day': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'transaction_count_1h': np.random.poisson(1, n_samples),
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create target with some correlation to features
    fraud_prob = 1 / (1 + np.exp(-(
        -2 + 
        0.01 * df['Amount'] + 
        -0.1 * df['time_since_signup'] + 
        0.05 * df['transaction_count_1h'] +
        np.random.normal(0, 0.5, n_samples)
    )))
    df['class'] = (fraud_prob > 0.1).astype(int)
    
    print(f"✅ Created {len(df)} samples ({df['class'].sum()} fraud, {len(df) - df['class'].sum()} normal)")
    return df


def load_processed_data():
    """Try to load processed data from data/processed/."""
    processed_dir = Path("data/processed")
    
    # Look for processed fraud data
    possible_files = [
        "fraud_data_processed.csv",
        "Fraud_Data_processed.csv",
        "processed_fraud_data.csv"
    ]
    
    for filename in possible_files:
        filepath = processed_dir / filename
        if filepath.exists():
            print(f"📂 Loading processed data from {filepath}")
            df = pd.read_csv(filepath)
            
            # Try to find target column
            target_cols = ['class', 'Class', 'target', 'fraud']
            target_col = None
            for col in target_cols:
                if col in df.columns:
                    target_col = col
                    break
            
            if target_col:
                print(f"✅ Loaded {len(df)} records with target column '{target_col}'")
                return df, target_col
            else:
                print(f"⚠️  No target column found. Expected one of: {target_cols}")
    
    return None, None


def main():
    """Main training function."""
    print("=" * 80)
    print("🛡️  Fraud Detection Model Training")
    print("=" * 80)
    print()
    
    # Try to load processed data
    df, target_col = load_processed_data()
    
    # If no data found, create sample data
    if df is None:
        print("⚠️  No processed data found. Creating sample data for demonstration...")
        print("   (For production, use your actual processed data)")
        print()
        df = create_sample_data()
        target_col = 'class'
    
    # Prepare data
    print("\n📋 Preparing data...")
    preparator = DataPreparator()
    
    try:
        X_train, X_test, y_train, y_test = preparator.prepare_data(
            df,
            target_column=target_col,
            test_size=0.2,
            random_state=42
        )
        print(f"✅ Train set: {len(X_train)} samples")
        print(f"✅ Test set: {len(X_test)} samples")
    except Exception as e:
        print(f"❌ Error preparing data: {str(e)}")
        print("   Using simple train-test split...")
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    
    # Handle imbalance
    print("\n⚖️  Handling class imbalance...")
    if IMBALANCE_HANDLER_AVAILABLE:
        try:
            handler = ImbalanceHandler(method="smote", random_state=42)
            X_train_resampled, y_train_resampled, stats = handler.resample(X_train, y_train)
            print(f"✅ Resampled: {len(X_train_resampled)} samples")
        except Exception as e:
            print(f"⚠️  Resampling failed: {str(e)}. Using original data.")
            X_train_resampled, y_train_resampled = X_train, y_train
    else:
        print("⚠️  ImbalanceHandler not available. Using original data.")
        print("   (Install compatible imbalanced-learn version to enable SMOTE)")
        X_train_resampled, y_train_resampled = X_train, y_train
    
    # Transform data
    print("\n🔄 Transforming data...")
    transformer = DataTransformer()
    try:
        X_train_transformed = transformer.fit_transform(X_train_resampled)
        X_test_transformed = transformer.transform(X_test)
        
        # Save transformer
        transformer_path = Path("models/transformer.joblib")
        transformer.save_transformers(str(transformer_path))
        print(f"✅ Transformer saved to {transformer_path}")
    except Exception as e:
        print(f"⚠️  Transformation failed: {str(e)}. Using original data.")
        X_train_transformed = X_train_resampled
        X_test_transformed = X_test
    
    # Train model
    print("\n🤖 Training model...")
    trainer = ModelTrainer()
    
    # Train Random Forest (good default)
    print("   Training Random Forest...")
    model = trainer.train_random_forest(
        X_train_transformed,
        y_train_resampled,
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    # Evaluate
    print("\n📊 Evaluating model...")
    from model_evaluator import ModelEvaluator
    evaluator = ModelEvaluator(output_dir="models/evaluation_outputs")
    metrics = evaluator.evaluate_model(
        model,
        X_test_transformed,
        y_test,
        model_name="RandomForest",
        plot=True
    )
    
    print(f"\n✅ Model Performance:")
    print(f"   PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"   F1-Score: {metrics['f1_score']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    
    # Save model
    print("\n💾 Saving model...")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "fraud_detection_model.joblib"
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")
    
    # Also save using ModelTrainer for consistency
    trainer.models['random_forest'] = model
    trainer.save_model('random_forest', str(model_path))
    
    print("\n" + "=" * 80)
    print("✅ Model training complete!")
    print("=" * 80)
    print(f"\n📊 Model file: {model_path}")
    print(f"📊 Transformer: models/transformer.joblib")
    print(f"\n🚀 You can now use the dashboard:")
    print(f"   streamlit run dashboard/app.py")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
