"""
Production-Ready Explainability Service

A reusable service for generating SHAP explanations on-demand for new predictions.
Designed for production deployment with caching and performance optimization.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")


class ExplainabilityService:
    """
    Production-ready service for generating SHAP explanations on-demand.
    
    Features:
    - Cached SHAP explainers for performance
    - Real-time explanations for new predictions
    - Support for multiple model types
    - Batch explanation support
    - Memory-efficient for production use
    """
    
    def __init__(
        self,
        model: Any = None,
        model_path: Optional[str] = None,
        background_data: Optional[pd.DataFrame] = None,
        background_data_path: Optional[str] = None,
        cache_explainer: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the ExplainabilityService.
        
        Args:
            model: Trained model (if not loading from path)
            model_path: Path to saved model file
            background_data: Background dataset for SHAP (training data sample)
            background_data_path: Path to saved background data
            cache_explainer: Whether to cache the SHAP explainer
            logger: Logger instance
        """
        self.logger = logger or self._setup_logger()
        self.cache_explainer = cache_explainer
        self.model = None
        self.background_data = None
        self.shap_explainer = None
        self.explainer_type = None
        self.feature_names = None
        self._is_initialized = False
        
        # Load model
        if model is not None:
            self.model = model
        elif model_path:
            self.load_model(model_path)
        else:
            self.logger.warning("No model provided. Call load_model() or set_model() before use.")
        
        # Load background data
        if background_data is not None:
            self.background_data = background_data
        elif background_data_path:
            self.load_background_data(background_data_path)
        
        # Initialize explainer if model and background data are available
        if self.model is not None and self.background_data is not None:
            self._initialize_explainer()
    
    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Setup a logger for the service."""
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
    
    def load_model(self, model_path: str):
        """Load model from file."""
        try:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.model = joblib.load(model_path)
            self.logger.info(f"Loaded model from {model_path}")
            
            # Reinitialize explainer if background data is available
            if self.background_data is not None:
                self._initialize_explainer()
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def set_model(self, model: Any):
        """Set model directly."""
        self.model = model
        self.logger.info("Model set")
        
        # Reinitialize explainer if background data is available
        if self.background_data is not None:
            self._initialize_explainer()
    
    def load_background_data(self, data_path: str, sample_size: int = 100):
        """Load background data for SHAP from file."""
        try:
            path = Path(data_path)
            if not path.exists():
                raise FileNotFoundError(f"Background data file not found: {data_path}")
            
            df = pd.read_csv(data_path)
            
            # Sample for efficiency
            if len(df) > sample_size:
                df = df.sample(sample_size, random_state=42)
            
            self.background_data = df
            self.feature_names = list(df.columns)
            self.logger.info(f"Loaded background data: {len(df)} samples")
            
            # Reinitialize explainer if model is available
            if self.model is not None:
                self._initialize_explainer()
            
        except Exception as e:
            self.logger.error(f"Error loading background data: {str(e)}")
            raise
    
    def set_background_data(self, data: pd.DataFrame, sample_size: int = 100):
        """Set background data directly."""
        if len(data) > sample_size:
            data = data.sample(sample_size, random_state=42)
        
        self.background_data = data
        self.feature_names = list(data.columns)
        self.logger.info(f"Background data set: {len(data)} samples")
        
        # Reinitialize explainer if model is available
        if self.model is not None:
            self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize SHAP explainer based on model type."""
        if not SHAP_AVAILABLE:
            self.logger.warning("SHAP not available. Explanations will not work.")
            return
        
        if self.model is None:
            self.logger.warning("Model not set. Cannot initialize explainer.")
            return
        
        if self.background_data is None:
            self.logger.warning("Background data not set. Cannot initialize explainer.")
            return
        
        try:
            model_type = type(self.model).__name__.lower()
            
            # Determine explainer type
            if 'randomforest' in model_type or 'forest' in model_type:
                self.explainer_type = 'tree'
                self.shap_explainer = shap.TreeExplainer(self.model)
                self.logger.info("Initialized TreeExplainer (fast)")
            
            elif 'xgboost' in model_type or 'xgb' in model_type:
                self.explainer_type = 'tree'
                self.shap_explainer = shap.TreeExplainer(self.model)
                self.logger.info("Initialized TreeExplainer for XGBoost (fast)")
            
            elif 'lightgbm' in model_type or 'lgb' in model_type:
                self.explainer_type = 'tree'
                self.shap_explainer = shap.TreeExplainer(self.model)
                self.logger.info("Initialized TreeExplainer for LightGBM (fast)")
            
            else:
                # Use KernelExplainer for other models (slower but universal)
                self.explainer_type = 'kernel'
                # Sample background data for KernelExplainer (max 100 samples)
                bg_sample = self.background_data.sample(
                    min(100, len(self.background_data)),
                    random_state=42
                )
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict_proba,
                    bg_sample
                )
                self.logger.info("Initialized KernelExplainer (slower but universal)")
            
            self._is_initialized = True
            self.logger.info("SHAP explainer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing SHAP explainer: {str(e)}")
            self.shap_explainer = None
            self._is_initialized = False
    
    def explain_prediction(
        self,
        instance: pd.DataFrame,
        return_values: bool = True,
        return_plot_data: bool = True
    ) -> Dict[str, Any]:
        """
        Explain a single prediction.
        
        Args:
            instance: Single instance to explain (DataFrame with one row)
            return_values: Whether to return SHAP values
            return_plot_data: Whether to return data for plotting
        
        Returns:
            Dictionary with explanation data
        """
        if not self._is_initialized:
            raise ValueError(
                "Explainer not initialized. Ensure model and background data are set."
            )
        
        if len(instance) != 1:
            raise ValueError("Instance must contain exactly one row")
        
        try:
            # Get SHAP values
            shap_values = self.shap_explainer.shap_values(instance)
            
            # Handle binary classification (shap_values is a list)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class (fraud)
            
            # Flatten if needed
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]
            
            # Get prediction
            prediction = self.model.predict(instance)[0]
            probability = self.model.predict_proba(instance)[0]
            
            result = {
                'prediction': int(prediction),
                'fraud_probability': float(probability[1]),
                'normal_probability': float(probability[0])
            }
            
            if return_values:
                # Create feature importance dataframe
                feature_importance = pd.DataFrame({
                    'feature': instance.columns,
                    'value': instance.iloc[0].values,
                    'shap_value': shap_values,
                    'abs_shap_value': np.abs(shap_values)
                }).sort_values('abs_shap_value', ascending=False)
                
                result['shap_values'] = shap_values.tolist()
                result['feature_importance'] = feature_importance.to_dict('records')
                result['top_features'] = feature_importance.head(10).to_dict('records')
            
            if return_plot_data:
                result['plot_data'] = {
                    'features': instance.columns.tolist(),
                    'values': instance.iloc[0].values.tolist(),
                    'shap_values': shap_values.tolist()
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error explaining prediction: {str(e)}")
            raise
    
    def explain_batch(
        self,
        instances: pd.DataFrame,
        max_instances: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Explain multiple predictions (batch processing).
        
        Args:
            instances: DataFrame with multiple instances
            max_instances: Maximum number of instances to explain (for performance)
        
        Returns:
            List of explanation dictionaries
        """
        if not self._is_initialized:
            raise ValueError(
                "Explainer not initialized. Ensure model and background data are set."
            )
        
        # Limit batch size for performance
        if len(instances) > max_instances:
            self.logger.warning(
                f"Batch size {len(instances)} exceeds max {max_instances}. "
                f"Processing first {max_instances} instances."
            )
            instances = instances.head(max_instances)
        
        explanations = []
        for idx, row in instances.iterrows():
            try:
                instance_df = pd.DataFrame([row])
                explanation = self.explain_prediction(instance_df)
                explanation['instance_id'] = idx
                explanations.append(explanation)
            except Exception as e:
                self.logger.warning(f"Error explaining instance {idx}: {str(e)}")
                continue
        
        return explanations
    
    def get_feature_importance_summary(
        self,
        instances: pd.DataFrame,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get summary of feature importance across multiple instances.
        
        Args:
            instances: DataFrame with instances to analyze
            top_n: Number of top features to return
        
        Returns:
            DataFrame with aggregated feature importance
        """
        explanations = self.explain_batch(instances)
        
        # Aggregate SHAP values
        all_shap_values = []
        for exp in explanations:
            if 'shap_values' in exp:
                all_shap_values.append(exp['shap_values'])
        
        if not all_shap_values:
            return pd.DataFrame()
        
        shap_array = np.array(all_shap_values)
        mean_shap = np.mean(np.abs(shap_array), axis=0)
        
        importance_df = pd.DataFrame({
            'feature': instances.columns,
            'mean_abs_shap': mean_shap
        }).sort_values('mean_abs_shap', ascending=False).head(top_n)
        
        return importance_df
    
    def save_explainer(self, filepath: str):
        """Save the explainer for later use (if supported)."""
        try:
            # Note: SHAP explainers can't always be pickled
            # This is a placeholder for future implementation
            self.logger.warning("SHAP explainer serialization not fully supported.")
            self.logger.info("To reuse explainer, save model and background data separately.")
        except Exception as e:
            self.logger.error(f"Error saving explainer: {str(e)}")
    
    def is_ready(self) -> bool:
        """Check if service is ready to explain predictions."""
        return self._is_initialized and self.shap_explainer is not None

