"""
REST API for Explainability Service

A Flask/FastAPI-based API endpoint for on-demand SHAP explanations.
Can be deployed as a microservice for production use.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from explainability_service import ExplainabilityService

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Global service instance
explainability_service = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service_ready': explainability_service.is_ready() if explainability_service else False
    })


@app.route('/initialize', methods=['POST'])
def initialize_service():
    """Initialize the explainability service with model and background data."""
    global explainability_service
    
    try:
        data = request.json
        
        model_path = data.get('model_path')
        background_data_path = data.get('background_data_path')
        
        if not model_path:
            return jsonify({'error': 'model_path is required'}), 400
        
        # Initialize service
        explainability_service = ExplainabilityService(
            model_path=model_path,
            background_data_path=background_data_path,
            cache_explainer=True
        )
        
        if explainability_service.is_ready():
            return jsonify({
                'status': 'initialized',
                'explainer_type': explainability_service.explainer_type
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Service initialized but explainer not ready'
            }), 500
    
    except Exception as e:
        logger.error(f"Error initializing service: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/explain', methods=['POST'])
def explain_prediction():
    """Explain a single prediction."""
    if not explainability_service or not explainability_service.is_ready():
        return jsonify({
            'error': 'Service not initialized. Call /initialize first.'
        }), 400
    
    try:
        data = request.json
        
        # Get transaction data
        transaction_data = data.get('transaction')
        if not transaction_data:
            return jsonify({'error': 'transaction data is required'}), 400
        
        # Convert to DataFrame
        transaction_df = pd.DataFrame([transaction_data])
        
        # Get explanation
        explanation = explainability_service.explain_prediction(
            transaction_df,
            return_values=True,
            return_plot_data=True
        )
        
        return jsonify(explanation)
    
    except Exception as e:
        logger.error(f"Error explaining prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/explain_batch', methods=['POST'])
def explain_batch():
    """Explain multiple predictions (batch)."""
    if not explainability_service or not explainability_service.is_ready():
        return jsonify({
            'error': 'Service not initialized. Call /initialize first.'
        }), 400
    
    try:
        data = request.json
        
        # Get transactions
        transactions = data.get('transactions')
        if not transactions:
            return jsonify({'error': 'transactions data is required'}), 400
        
        # Convert to DataFrame
        transactions_df = pd.DataFrame(transactions)
        
        # Get explanations
        max_instances = data.get('max_instances', 100)
        explanations = explainability_service.explain_batch(
            transactions_df,
            max_instances=max_instances
        )
        
        return jsonify({
            'count': len(explanations),
            'explanations': explanations
        })
    
    except Exception as e:
        logger.error(f"Error explaining batch: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/feature_importance', methods=['POST'])
def get_feature_importance():
    """Get feature importance summary across multiple instances."""
    if not explainability_service or not explainability_service.is_ready():
        return jsonify({
            'error': 'Service not initialized. Call /initialize first.'
        }), 400
    
    try:
        data = request.json
        
        # Get transactions
        transactions = data.get('transactions')
        if not transactions:
            return jsonify({'error': 'transactions data is required'}), 400
        
        # Convert to DataFrame
        transactions_df = pd.DataFrame(transactions)
        
        # Get feature importance
        top_n = data.get('top_n', 10)
        importance_df = explainability_service.get_feature_importance_summary(
            transactions_df,
            top_n=top_n
        )
        
        return jsonify({
            'feature_importance': importance_df.to_dict('records')
        })
    
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Default initialization (can be overridden via /initialize endpoint)
    model_path = Path("models/fraud_detection_model.joblib")
    background_path = Path("data/processed")
    
    if model_path.exists():
        background_files = list(background_path.glob("*processed*.csv"))
        if background_files:
            try:
                explainability_service = ExplainabilityService(
                    model_path=str(model_path),
                    background_data_path=str(background_files[0]),
                    cache_explainer=True
                )
                logger.info("Service initialized with default model and background data")
            except Exception as e:
                logger.warning(f"Could not initialize with defaults: {str(e)}")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
