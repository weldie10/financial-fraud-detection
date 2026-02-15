"""
Fraud Detection Interactive Dashboard

A Streamlit-based dashboard for fraud analysts, product managers, and business stakeholders
to explore predictions, test scenarios, and understand fraud drivers.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    st.warning("⚠️ SHAP not available. Some features will be limited. Install with: pip install shap")

from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator
from data_transformer import DataTransformer
from model_explainer import ModelExplainer
from explainability_service import ExplainabilityService

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .fraud-alert {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
    }
    .safe-alert {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'transformer' not in st.session_state:
    st.session_state.transformer = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'shap_explainer' not in st.session_state:
    st.session_state.shap_explainer = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = None
if 'explainability_service' not in st.session_state:
    st.session_state.explainability_service = None
if 'background_data' not in st.session_state:
    st.session_state.background_data = None


@st.cache_data
def load_model(model_path: str):
    """Load model with caching."""
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


@st.cache_data
def load_transformer(transformer_path: str):
    """Load transformer with caching."""
    try:
        transformer = DataTransformer()
        transformer.load_transformers(transformer_path)
        return transformer
    except Exception as e:
        st.error(f"Error loading transformer: {str(e)}")
        return None


def load_model_metrics():
    """Load model performance metrics."""
    metrics_dir = Path("models/evaluation_outputs")
    if not metrics_dir.exists():
        return None
    
    # Try to load comparison CSV if available
    comparison_file = metrics_dir / "model_comparison.csv"
    if comparison_file.exists():
        return pd.read_csv(comparison_file)
    return None


def create_shap_explainer(model, X_sample):
    """Create SHAP explainer for the model."""
    if not SHAP_AVAILABLE:
        return None
    
    try:
        # Determine model type
        model_type = type(model).__name__.lower()
        
        if 'randomforest' in model_type or 'xgboost' in model_type or 'lightgbm' in model_type:
            explainer = shap.TreeExplainer(model)
        else:
            # Use KernelExplainer for other models
            explainer = shap.KernelExplainer(model.predict_proba, X_sample.sample(min(100, len(X_sample))))
        
        return explainer
    except Exception as e:
        st.warning(f"Could not create SHAP explainer: {str(e)}")
        return None


def predict_single_transaction(model, transformer, transaction_data):
    """Make prediction for a single transaction."""
    try:
        # Transform data
        if transformer and transformer.is_fitted:
            transaction_transformed = transformer.transform(transaction_data)
        else:
            transaction_transformed = transaction_data
        
        # Predict
        prediction = model.predict(transaction_transformed)[0]
        probability = model.predict_proba(transaction_transformed)[0]
        
        return {
            'prediction': int(prediction),
            'fraud_probability': float(probability[1]),
            'normal_probability': float(probability[0])
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None


def get_shap_explanation(explainer, transaction_data, model):
    """Get SHAP values for a transaction."""
    if not SHAP_AVAILABLE or explainer is None:
        return None
    
    try:
        shap_values = explainer.shap_values(transaction_data)
        
        # Handle binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class (fraud)
        
        return shap_values[0] if len(shap_values.shape) > 1 else shap_values
    except Exception as e:
        st.warning(f"SHAP explanation error: {str(e)}")
        return None


def main():
    # Header
    st.markdown('<div class="main-header">🛡️ Fraud Detection Dashboard</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar - Model Selection
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        models_dir = Path("models")
        model_files = list(models_dir.glob("*.joblib"))
        model_files = [f for f in model_files if "config" not in f.name]
        
        if not model_files:
            st.error("❌ No models found in models/ directory")
            st.markdown("---")
            st.markdown("### 📚 How to Train and Save a Model")
            st.markdown("""
            **Option 1: Use the Modeling Notebook**
            1. Open `notebooks/modeling.ipynb`
            2. Run all cells to train models
            3. Models will be saved automatically to `models/` directory
            
            **Option 2: Use the Training Script**
            ```bash
            python scripts/train_model.py
            ```
            
            **Option 3: Quick Training (Python)**
            ```python
            from src.model_trainer import ModelTrainer
            import joblib
            
            # Train model (use your data)
            trainer = ModelTrainer()
            model = trainer.train_random_forest(X_train, y_train)
            
            # Save model
            joblib.dump(model, "models/fraud_detection_model.joblib")
            ```
            """)
            st.markdown("---")
            st.info("💡 Once you have a model file (.joblib) in the models/ directory, refresh this page.")
            return
        
        selected_model = st.selectbox(
            "Select Model",
            options=[f.name for f in model_files],
            help="Choose a trained model to use for predictions"
        )
        
        if st.button("🔄 Load Model", type="primary"):
            with st.spinner("Loading model..."):
                model_path = models_dir / selected_model
                st.session_state.model = load_model(str(model_path))
                
                # Try to load transformer
                transformer_path = models_dir / "transformer.joblib"
                if transformer_path.exists():
                    st.session_state.transformer = load_transformer(str(transformer_path))
                
                # Load metrics
                st.session_state.model_metrics = load_model_metrics()
                
                # Initialize explainability service
                if SHAP_AVAILABLE and st.session_state.model:
                    try:
                        # Try to load background data for SHAP
                        background_data_path = Path("data/processed")
                        background_files = list(background_data_path.glob("*processed*.csv"))
                        
                        if background_files:
                            # Load a sample of processed data for SHAP background
                            bg_data = pd.read_csv(background_files[0])
                            # Sample for efficiency (max 100 rows)
                            if len(bg_data) > 100:
                                bg_data = bg_data.sample(100, random_state=42)
                            
                            # Remove target column if present
                            target_cols = ['class', 'Class', 'target', 'fraud']
                            for col in target_cols:
                                if col in bg_data.columns:
                                    bg_data = bg_data.drop(columns=[col])
                            
                            # Transform if transformer available
                            if st.session_state.transformer and st.session_state.transformer.is_fitted:
                                try:
                                    bg_data = st.session_state.transformer.transform(bg_data)
                                except:
                                    pass  # Use original if transformation fails
                            
                            st.session_state.background_data = bg_data
                            
                            # Initialize explainability service
                            st.session_state.explainability_service = ExplainabilityService(
                                model=st.session_state.model,
                                background_data=bg_data,
                                cache_explainer=True
                            )
                            
                            if st.session_state.explainability_service.is_ready():
                                st.success("✅ Model and explainability service loaded!")
                            else:
                                st.warning("⚠️ Model loaded, but explainability service not ready")
                        else:
                            st.info("💡 Load background data to enable SHAP explanations")
                            st.session_state.explainability_service = None
                    except Exception as e:
                        st.warning(f"⚠️ Could not initialize explainability service: {str(e)}")
                        st.session_state.explainability_service = None
                else:
                    st.session_state.explainability_service = None
                
                if not SHAP_AVAILABLE:
                    st.info("💡 Install SHAP to enable explanations: pip install shap")
                
                st.success("✅ Model loaded successfully!")
        
        st.markdown("---")
        
        # Model info
        if st.session_state.model:
            st.success("✅ Model Ready")
            model_type = type(st.session_state.model).__name__
            st.caption(f"Type: {model_type}")
        else:
            st.warning("⚠️ No model loaded")
        
        st.markdown("---")
        
        # Navigation
        st.header("📊 Navigation")
        page = st.radio(
            "Select Page",
            ["🏠 Home", "🔮 Predictions", "📈 Model Performance", "🔍 Fraud Drivers", "🧪 Scenario Testing"]
        )
    
    # Main content based on selected page
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔮 Predictions":
        show_predictions_page()
    elif page == "📈 Model Performance":
        show_performance_page()
    elif page == "🔍 Fraud Drivers":
        show_fraud_drivers_page()
    elif page == "🧪 Scenario Testing":
        show_scenario_testing_page()


def show_home_page():
    """Home page with overview."""
    st.header("Welcome to the Fraud Detection Dashboard")
    
    # Check if model is loaded
    model_loaded = st.session_state.model is not None
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if model_loaded:
            st.metric("Model Status", "✅ Ready")
        else:
            st.metric("Model Status", "❌ Not Loaded")
    
    with col2:
        if st.session_state.model_metrics is not None:
            best_metric = st.session_state.model_metrics.iloc[0]
            pr_auc = best_metric.get('PR-AUC', 'N/A')
            if isinstance(pr_auc, (int, float)):
                st.metric("Best Model PR-AUC", f"{pr_auc:.4f}")
            else:
                st.metric("Model Metrics", "Available")
        else:
            st.metric("Model Metrics", "Not Available")
    
    with col3:
        st.metric("SHAP Available", "✅ Yes" if SHAP_AVAILABLE else "❌ No")
    
    # Show warning if no model
    if not model_loaded:
        st.markdown("---")
        st.warning("⚠️ **No model loaded.** Please train and save a model first.")
        
        with st.expander("📚 How to Train a Model", expanded=True):
            st.markdown("""
            **Quick Training (Recommended):**
            ```bash
            python scripts/train_model.py
            ```
            
            This script will:
            - Load processed data (or create sample data)
            - Train a Random Forest model
            - Save the model to `models/fraud_detection_model.joblib`
            - Save the transformer for data preprocessing
            
            **Using Notebooks:**
            1. Open `notebooks/modeling.ipynb`
            2. Run all cells to train models
            3. Models will be saved automatically
            
            **Manual Training (Python):**
            ```python
            from src.model_trainer import ModelTrainer
            import joblib
            
            trainer = ModelTrainer()
            model = trainer.train_random_forest(X_train, y_train)
            joblib.dump(model, "models/fraud_detection_model.joblib")
            ```
            """)
        
        st.markdown("---")
    
    st.subheader("📋 Dashboard Features")
    
    features = [
        "🔮 **Predictions**: Make real-time fraud predictions for individual transactions",
        "📈 **Model Performance**: View model metrics, confusion matrices, and ROC/PR curves",
        "🔍 **Fraud Drivers**: Explore SHAP explanations to understand what drives fraud predictions",
        "🧪 **Scenario Testing**: Test different transaction scenarios and see how predictions change"
    ]
    
    for feature in features:
        st.markdown(f"- {feature}")
    
    st.markdown("---")
    
    st.subheader("🚀 Quick Start")
    
    if model_loaded:
        st.markdown("""
        1. ✅ **Model Loaded**: You're ready to go!
        2. **Make Predictions**: Go to the Predictions page to analyze transactions
        3. **Explore Insights**: Use the Fraud Drivers page to understand model behavior
        4. **Test Scenarios**: Try different transaction values to see prediction changes
        """)
    else:
        st.markdown("""
        1. **Train a Model**: Run `python scripts/train_model.py` or use the modeling notebook
        2. **Load Model**: Use the sidebar to select and load your trained model
        3. **Make Predictions**: Go to the Predictions page to analyze transactions
        4. **Explore Insights**: Use the Fraud Drivers page to understand model behavior
        """)


def show_predictions_page():
    """Page for making predictions."""
    st.header("🔮 Transaction Predictions")
    
    if not st.session_state.model:
        st.warning("⚠️ Please load a model from the sidebar first.")
        return
    
    # Input method selection
    input_method = st.radio(
        "Input Method",
        ["📝 Manual Entry", "📁 Upload CSV"],
        horizontal=True
    )
    
    if input_method == "📝 Manual Entry":
        show_manual_prediction()
    else:
        show_batch_prediction()


def show_manual_prediction():
    """Manual transaction entry and prediction."""
    st.subheader("Enter Transaction Details")
    
    # Create input form
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=0.01)
            time_since_signup = st.number_input("Time Since Signup (hours)", min_value=0.0, value=24.0, step=0.1)
            age = st.number_input("User Age", min_value=18, max_value=100, value=35)
        
        with col2:
            hour_of_day = st.number_input("Hour of Day", min_value=0, max_value=23, value=12)
            day_of_week = st.selectbox("Day of Week", range(7), format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])
            transaction_count_1h = st.number_input("Transactions in Last Hour", min_value=0, value=1)
        
        submitted = st.form_submit_button("🔍 Analyze Transaction", type="primary")
    
    if submitted:
        # Create transaction dataframe
        transaction_data = pd.DataFrame({
            'Amount': [amount],
            'time_since_signup': [time_since_signup],
            'age': [age],
            'hour_of_day': [hour_of_day],
            'day_of_week': [day_of_week],
            'transaction_count_1h': [transaction_count_1h]
        })
        
        # Make prediction
        with st.spinner("Analyzing transaction..."):
            result = predict_single_transaction(
                st.session_state.model,
                st.session_state.transformer,
                transaction_data
            )
        
        if result:
            # Display result
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Prediction", "🚨 FRAUD" if result['prediction'] == 1 else "✅ NORMAL")
            
            with col2:
                st.metric("Fraud Probability", f"{result['fraud_probability']:.2%}")
            
            with col3:
                st.metric("Normal Probability", f"{result['normal_probability']:.2%}")
            
            # Alert box
            if result['prediction'] == 1:
                st.markdown(
                    f'<div class="fraud-alert">'
                    f'<h3>🚨 Fraud Detected</h3>'
                    f'<p>This transaction has a <strong>{result["fraud_probability"]:.2%}</strong> probability of being fraudulent.</p>'
                    f'<p><strong>Recommended Action:</strong> Flag for manual review or block transaction.</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="safe-alert">'
                    f'<h3>✅ Transaction Safe</h3>'
                    f'<p>This transaction appears to be legitimate with a <strong>{result["normal_probability"]:.2%}</strong> probability.</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            # SHAP explanation using explainability service
            if SHAP_AVAILABLE and st.session_state.explainability_service and st.session_state.explainability_service.is_ready():
                st.markdown("---")
                st.subheader("🔍 Explanation")
                
                try:
                    # Transform transaction if transformer available
                    transaction_for_shap = transaction_data.copy()
                    if st.session_state.transformer and st.session_state.transformer.is_fitted:
                        try:
                            transaction_for_shap = st.session_state.transformer.transform(transaction_for_shap)
                        except:
                            pass  # Use original if transformation fails
                    
                    # Get explanation from service
                    explanation = st.session_state.explainability_service.explain_prediction(
                        transaction_for_shap,
                        return_values=True,
                        return_plot_data=True
                    )
                    
                    if explanation and 'top_features' in explanation:
                        # Display top contributing features
                        st.markdown("#### Top Contributing Features")
                        
                        top_features = explanation['top_features']
                        feature_df = pd.DataFrame(top_features)
                        
                        # Create visualization
                        fig = px.bar(
                            feature_df,
                            x='shap_value',
                            y='feature',
                            orientation='h',
                            color='shap_value',
                            color_continuous_scale='RdBu',
                            title="Feature Contributions to Prediction",
                            labels={'shap_value': 'SHAP Value', 'feature': 'Feature'}
                        )
                        fig.add_vline(x=0, line_dash="dash", line_color="gray")
                        fig.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display feature details table
                        with st.expander("📊 Detailed Feature Breakdown"):
                            display_df = feature_df[['feature', 'value', 'shap_value']].copy()
                            display_df.columns = ['Feature', 'Value', 'SHAP Contribution']
                            display_df['SHAP Contribution'] = display_df['SHAP Contribution'].round(4)
                            st.dataframe(display_df, use_container_width=True)
                        
                        # Interpretation
                        st.markdown("#### 💡 Interpretation")
                        positive_features = feature_df[feature_df['shap_value'] > 0]
                        negative_features = feature_df[feature_df['shap_value'] < 0]
                        
                        if len(positive_features) > 0:
                            st.markdown("**Features increasing fraud probability:**")
                            for _, row in positive_features.head(3).iterrows():
                                st.markdown(f"- **{row['feature']}** = {row['value']:.2f} (contribution: +{row['shap_value']:.4f})")
                        
                        if len(negative_features) > 0:
                            st.markdown("**Features decreasing fraud probability:**")
                            for _, row in negative_features.head(3).iterrows():
                                st.markdown(f"- **{row['feature']}** = {row['value']:.2f} (contribution: {row['shap_value']:.4f})")
                
                except Exception as e:
                    st.warning(f"Could not generate explanation: {str(e)}")
                    # Fallback to simple explanation
                    if st.session_state.shap_explainer:
                        shap_values = get_shap_explanation(
                            st.session_state.shap_explainer,
                            transaction_data,
                            st.session_state.model
                        )
                        if shap_values is not None:
                            shap_df = pd.DataFrame({
                                'Feature': transaction_data.columns,
                                'SHAP Value': shap_values,
                            })
                            shap_df = shap_df.sort_values('SHAP Value', key=abs, ascending=False)
                            st.dataframe(shap_df.head(10), use_container_width=True)
            elif SHAP_AVAILABLE:
                st.info("💡 Load background data to enable SHAP explanations")


def show_batch_prediction():
    """Batch prediction from CSV upload."""
    st.subheader("Upload CSV File")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} transactions")
            
            st.dataframe(df.head(10))
            
            if st.button("🔍 Predict All", type="primary"):
                with st.spinner("Processing transactions..."):
                    # Make predictions
                    if st.session_state.transformer and st.session_state.transformer.is_fitted:
                        df_transformed = st.session_state.transformer.transform(df)
                    else:
                        df_transformed = df
                    
                    predictions = st.session_state.model.predict(df_transformed)
                    probabilities = st.session_state.model.predict_proba(df_transformed)[:, 1]
                    
                    # Add predictions to dataframe
                    df['Prediction'] = ['FRAUD' if p == 1 else 'NORMAL' for p in predictions]
                    df['Fraud_Probability'] = probabilities
                    
                    # Display results
                    st.subheader("📊 Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Transactions", len(df))
                    with col2:
                        fraud_count = (predictions == 1).sum()
                        st.metric("Fraud Detected", fraud_count)
                    with col3:
                        st.metric("Fraud Rate", f"{fraud_count/len(df):.2%}")
                    
                    st.dataframe(df)
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "fraud_predictions.csv",
                        "text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")


def show_performance_page():
    """Model performance metrics page."""
    st.header("📈 Model Performance")
    
    if not st.session_state.model:
        st.warning("⚠️ Please load a model from the sidebar first.")
        return
    
    if st.session_state.model_metrics is None:
        st.warning("⚠️ Model metrics not available. Please run model evaluation first.")
        return
    
    # Display metrics table
    st.subheader("Model Comparison Metrics")
    st.dataframe(st.session_state.model_metrics, use_container_width=True)
    
    # Visualizations
    metrics_dir = Path("models/evaluation_outputs")
    
    if (metrics_dir / "pr_curves_comparison.png").exists():
        st.subheader("Precision-Recall Curves")
        st.image(str(metrics_dir / "pr_curves_comparison.png"), use_container_width=True)
    
    if (metrics_dir / "roc_curves_comparison.png").exists():
        st.subheader("ROC Curves")
        st.image(str(metrics_dir / "roc_curves_comparison.png"), use_container_width=True)
    
    if (metrics_dir / "model_comparison_metrics.png").exists():
        st.subheader("Model Comparison")
        st.image(str(metrics_dir / "model_comparison_metrics.png"), use_container_width=True)


def show_fraud_drivers_page():
    """SHAP explanations and fraud drivers."""
    st.header("🔍 Fraud Drivers Analysis")
    
    if not st.session_state.model:
        st.warning("⚠️ Please load a model from the sidebar first.")
        return
    
    if not SHAP_AVAILABLE:
        st.error("❌ SHAP is not available. Please install with: pip install shap")
        return
    
    # Load SHAP visualizations if available
    shap_dir = Path("models/explainability_outputs")
    
    if (shap_dir / "shap_summary_plot.png").exists():
        st.subheader("Global Feature Importance (SHAP)")
        st.image(str(shap_dir / "shap_summary_plot.png"), use_container_width=True)
        st.caption("This plot shows which features are most important for fraud detection across all transactions.")
    
    if (shap_dir / "top5_drivers_analysis.png").exists():
        st.subheader("Top 5 Fraud Drivers")
        st.image(str(shap_dir / "top5_drivers_analysis.png"), use_container_width=True)
    
    if (shap_dir / "feature_importance_comparison.png").exists():
        st.subheader("Feature Importance Comparison")
        st.image(str(shap_dir / "feature_importance_comparison.png"), use_container_width=True)
        st.caption("Comparison between built-in feature importance and SHAP values.")


def show_scenario_testing_page():
    """Scenario testing page."""
    st.header("🧪 Scenario Testing")
    
    if not st.session_state.model:
        st.warning("⚠️ Please load a model from the sidebar first.")
        return
    
    st.subheader("Test Different Transaction Scenarios")
    
    # Base transaction
    st.markdown("#### Base Transaction")
    col1, col2 = st.columns(2)
    
    with col1:
        base_amount = st.number_input("Amount ($)", min_value=0.0, value=100.0, key="base_amount")
        base_time_signup = st.number_input("Time Since Signup (hours)", min_value=0.0, value=24.0, key="base_time")
        base_age = st.number_input("Age", min_value=18, max_value=100, value=35, key="base_age")
    
    with col2:
        base_hour = st.number_input("Hour of Day", min_value=0, max_value=23, value=12, key="base_hour")
        base_day = st.selectbox("Day of Week", range(7), format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x], key="base_day")
        base_txn_count = st.number_input("Transactions in Last Hour", min_value=0, value=1, key="base_txn")
    
    # Scenario variations
    st.markdown("---")
    st.markdown("#### Test Variations")
    
    variation_type = st.selectbox(
        "What would you like to vary?",
        ["Amount", "Time Since Signup", "Transaction Count", "Hour of Day"]
    )
    
    if variation_type == "Amount":
        values = st.slider("Amount Range ($)", 0.0, 1000.0, (0.0, 500.0), 10.0)
        test_values = np.linspace(values[0], values[1], 20)
        var_name = "Amount"
    elif variation_type == "Time Since Signup":
        values = st.slider("Time Since Signup Range (hours)", 0.0, 720.0, (0.0, 168.0), 1.0)
        test_values = np.linspace(values[0], values[1], 20)
        var_name = "time_since_signup"
    elif variation_type == "Transaction Count":
        values = st.slider("Transaction Count Range", 0, 50, (0, 20), 1)
        test_values = np.linspace(values[0], values[1], 20, dtype=int)
        var_name = "transaction_count_1h"
    else:  # Hour of Day
        values = st.slider("Hour Range", 0, 23, (0, 23), 1)
        test_values = np.linspace(values[0], values[1], 24, dtype=int)
        var_name = "hour_of_day"
    
    if st.button("🔍 Run Scenario Analysis", type="primary"):
        # Create test transactions
        results = []
        
        for val in test_values:
            transaction = pd.DataFrame({
                'Amount': [base_amount if var_name != "Amount" else val],
                'time_since_signup': [base_time_signup if var_name != "time_since_signup" else val],
                'age': [base_age],
                'hour_of_day': [base_hour if var_name != "hour_of_day" else int(val)],
                'day_of_week': [base_day],
                'transaction_count_1h': [base_txn_count if var_name != "transaction_count_1h" else int(val)]
            })
            
            result = predict_single_transaction(
                st.session_state.model,
                st.session_state.transformer,
                transaction
            )
            
            if result:
                results.append({
                    var_name: val,
                    'Fraud_Probability': result['fraud_probability'],
                    'Prediction': result['prediction']
                })
        
        results_df = pd.DataFrame(results)
        
        # Visualize
        fig = px.line(
            results_df,
            x=var_name,
            y='Fraud_Probability',
            title=f"Fraud Probability vs {variation_type}",
            labels={'Fraud_Probability': 'Fraud Probability', var_name: variation_type}
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Threshold (0.5)")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary
        st.subheader("📊 Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Min Fraud Prob", f"{results_df['Fraud_Probability'].min():.2%}")
        with col2:
            st.metric("Max Fraud Prob", f"{results_df['Fraud_Probability'].max():.2%}")
        with col3:
            st.metric("Avg Fraud Prob", f"{results_df['Fraud_Probability'].mean():.2%}")


if __name__ == "__main__":
    main()
