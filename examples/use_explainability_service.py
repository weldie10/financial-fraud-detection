"""
Example: Using the Production Explainability Service

This script demonstrates how to use the ExplainabilityService
for on-demand SHAP explanations in production.
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from explainability_service import ExplainabilityService


def main():
    """Example usage of ExplainabilityService."""
    
    print("=" * 80)
    print("🛡️  Explainability Service Example")
    print("=" * 80)
    print()
    
    # Initialize service
    print("1️⃣  Initializing Explainability Service...")
    
    model_path = "models/fraud_detection_model.joblib"
    background_data_path = "data/processed"
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        print("   Please train a model first: python scripts/train_model.py")
        return
    
    # Find background data
    bg_files = list(Path(background_data_path).glob("*processed*.csv"))
    if not bg_files:
        print(f"⚠️  No processed data found in {background_data_path}")
        print("   Creating sample background data...")
        # Create sample data
        import numpy as np
        np.random.seed(42)
        bg_data = pd.DataFrame({
            'Amount': np.random.exponential(88, 100),
            'time_since_signup': np.random.exponential(24, 100),
            'age': np.random.randint(18, 80, 100),
            'hour_of_day': np.random.randint(0, 24, 100),
            'day_of_week': np.random.randint(0, 7, 100),
            'transaction_count_1h': np.random.poisson(1, 100),
        })
        bg_data_path = None
    else:
        bg_data_path = str(bg_files[0])
        bg_data = None
    
    try:
        service = ExplainabilityService(
            model_path=model_path,
            background_data_path=bg_data_path,
            background_data=bg_data,
            cache_explainer=True
        )
        
        if not service.is_ready():
            print("❌ Service not ready. Check model and background data.")
            return
        
        print(f"✅ Service initialized (explainer type: {service.explainer_type})")
        print()
        
        # Example 1: Explain a single prediction
        print("2️⃣  Explaining a single transaction...")
        
        transaction = pd.DataFrame({
            'Amount': [150.0],
            'time_since_signup': [0.5],  # Very recent signup (high risk)
            'age': [25],
            'hour_of_day': [3],  # Late night (high risk)
            'day_of_week': [5],  # Weekend
            'transaction_count_1h': [5]  # High frequency (high risk)
        })
        
        explanation = service.explain_prediction(transaction)
        
        print(f"   Prediction: {'🚨 FRAUD' if explanation['prediction'] == 1 else '✅ NORMAL'}")
        print(f"   Fraud Probability: {explanation['fraud_probability']:.2%}")
        print(f"   Normal Probability: {explanation['normal_probability']:.2%}")
        print()
        print("   Top 3 Contributing Features:")
        for i, feature in enumerate(explanation['top_features'][:3], 1):
            print(f"   {i}. {feature['feature']}: {feature['value']:.2f} "
                  f"(SHAP: {feature['shap_value']:+.4f})")
        print()
        
        # Example 2: Explain multiple transactions
        print("3️⃣  Explaining multiple transactions (batch)...")
        
        transactions = pd.DataFrame({
            'Amount': [50.0, 200.0, 1000.0],
            'time_since_signup': [24.0, 0.1, 720.0],
            'age': [35, 22, 45],
            'hour_of_day': [12, 2, 14],
            'day_of_week': [1, 6, 2],
            'transaction_count_1h': [1, 10, 2]
        })
        
        explanations = service.explain_batch(transactions, max_instances=10)
        
        print(f"   Explained {len(explanations)} transactions")
        for i, exp in enumerate(explanations, 1):
            print(f"   Transaction {i}: "
                  f"{'🚨 FRAUD' if exp['prediction'] == 1 else '✅ NORMAL'} "
                  f"({exp['fraud_probability']:.2%})")
        print()
        
        # Example 3: Feature importance summary
        print("4️⃣  Feature importance summary...")
        
        importance_df = service.get_feature_importance_summary(transactions, top_n=5)
        
        print("   Top 5 Features (by mean absolute SHAP value):")
        for i, row in importance_df.iterrows():
            print(f"   {i+1}. {row['feature']}: {row['mean_abs_shap']:.4f}")
        print()
        
        print("=" * 80)
        print("✅ Example completed successfully!")
        print("=" * 80)
        print()
        print("💡 Next steps:")
        print("   - Use ExplainabilityService in your production code")
        print("   - Deploy the REST API: python api/explainability_api.py")
        print("   - Integrate with the dashboard for real-time explanations")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
