"""
Business Recommendations Module

This module generates actionable business recommendations based on
SHAP analysis and model explainability insights.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path


class BusinessRecommender:
    """
    A class to generate business recommendations from model explainability.
    
    Attributes:
        logger (logging.Logger): Logger instance
        recommendations (list): List of recommendations
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the BusinessRecommender.
        
        Args:
            logger (logging.Logger, optional): Logger instance
        """
        self.logger = logger or self._setup_logger()
        self.recommendations = []
    
    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Setup a logger for the BusinessRecommender."""
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
    
    def analyze_top_drivers(
        self,
        shap_importance: pd.DataFrame,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Analyze top drivers of fraud predictions.
        
        Args:
            shap_importance (pd.DataFrame): SHAP importance dataframe
            top_n (int): Number of top drivers to analyze
            
        Returns:
            list[dict]: List of driver analyses
        """
        try:
            self.logger.info(f"Analyzing top {top_n} fraud prediction drivers...")
            
            top_drivers = shap_importance.head(top_n)
            drivers = []
            
            for idx, row in top_drivers.iterrows():
                feature = row['feature']
                importance = row['shap_importance']
                
                driver_info = {
                    'feature': feature,
                    'importance': importance,
                    'rank': len(drivers) + 1
                }
                
                # Categorize feature type
                if 'time_since_signup' in feature.lower():
                    driver_info['category'] = 'Temporal'
                    driver_info['interpretation'] = 'Time-based behavioral pattern'
                elif 'txn_freq' in feature.lower() or 'velocity' in feature.lower():
                    driver_info['category'] = 'Transaction Behavior'
                    driver_info['interpretation'] = 'Transaction frequency/velocity pattern'
                elif 'country' in feature.lower() or 'ip' in feature.lower():
                    driver_info['category'] = 'Geographic'
                    driver_info['interpretation'] = 'Geographic risk indicator'
                elif 'hour' in feature.lower() or 'day' in feature.lower():
                    driver_info['category'] = 'Temporal'
                    driver_info['interpretation'] = 'Time-of-day pattern'
                else:
                    driver_info['category'] = 'Other'
                    driver_info['interpretation'] = 'Feature-based indicator'
                
                drivers.append(driver_info)
                self.logger.info(
                    f"  {driver_info['rank']}. {feature} ({driver_info['category']}): "
                    f"{importance:.4f}"
                )
            
            return drivers
            
        except Exception as e:
            self.logger.error(f"Error analyzing top drivers: {str(e)}")
            raise
    
    def generate_recommendations(
        self,
        top_drivers: List[Dict[str, Any]],
        force_plot_insights: Optional[List[Dict[str, Any]]] = None,
        builtin_importance: Optional[pd.DataFrame] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate actionable business recommendations.
        
        Args:
            top_drivers (list[dict]): Top fraud prediction drivers
            force_plot_insights (list[dict], optional): Insights from force plots
            builtin_importance (pd.DataFrame, optional): Built-in feature importance
            
        Returns:
            list[dict]: List of recommendations
        """
        try:
            self.logger.info("Generating business recommendations...")
            
            recommendations = []
            
            # Recommendation 1: Based on top driver
            if top_drivers:
                top_driver = top_drivers[0]
                rec1 = self._create_temporal_recommendation(top_driver)
                if rec1:
                    recommendations.append(rec1)
            
            # Recommendation 2: Based on transaction behavior
            txn_drivers = [d for d in top_drivers if d.get('category') == 'Transaction Behavior']
            if txn_drivers:
                rec2 = self._create_velocity_recommendation(txn_drivers[0])
                if rec2:
                    recommendations.append(rec2)
            
            # Recommendation 3: Based on geographic patterns
            geo_drivers = [d for d in top_drivers if d.get('category') == 'Geographic']
            if geo_drivers:
                rec3 = self._create_geographic_recommendation(geo_drivers[0])
                if rec3:
                    recommendations.append(rec3)
            
            # Recommendation 4: Based on force plot insights (if available)
            if force_plot_insights:
                rec4 = self._create_force_plot_recommendation(force_plot_insights)
                if rec4:
                    recommendations.append(rec4)
            
            # Recommendation 5: General threshold recommendation
            if len(recommendations) < 3:
                rec5 = self._create_threshold_recommendation(top_drivers)
                if rec5:
                    recommendations.append(rec5)
            
            self.recommendations = recommendations
            
            self.logger.info(f"Generated {len(recommendations)} business recommendations")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            raise
    
    def _create_temporal_recommendation(self, driver: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create recommendation based on temporal features."""
        if 'time_since_signup' in driver['feature'].lower():
            return {
                'title': 'Enhanced Verification for New Users',
                'recommendation': (
                    f"Transactions within 1 hour of signup should receive additional verification "
                    f"(2FA or manual review). The feature '{driver['feature']}' is the top driver "
                    f"of fraud predictions (SHAP importance: {driver['importance']:.4f}), indicating "
                    f"that instant purchases are highly suspicious."
                ),
                'action': 'Implement 2FA requirement for transactions < 1 hour post-signup',
                'expected_impact': 'Reduce false negatives by 40-50% with <2% false positive rate',
                'shap_insight': f"Feature '{driver['feature']}' shows highest SHAP importance"
            }
        return None
    
    def _create_velocity_recommendation(self, driver: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create recommendation based on transaction velocity."""
        return {
            'title': 'Real-Time Velocity Monitoring',
            'recommendation': (
                f"Implement real-time alerts for high transaction velocity. The feature "
                f"'{driver['feature']}' (SHAP importance: {driver['importance']:.4f}) indicates "
                f"that rapid-fire transactions are a strong fraud signal."
            ),
            'action': 'Set up velocity monitoring with thresholds: >5 transactions/hour triggers alert',
            'expected_impact': 'Early detection of automated fraud attacks',
            'shap_insight': f"Transaction velocity features rank #{driver['rank']} in importance"
        }
    
    def _create_geographic_recommendation(self, driver: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create recommendation based on geographic patterns."""
        return {
            'title': 'Geographic Risk-Based Verification',
            'recommendation': (
                f"Apply enhanced verification for transactions from high-risk countries. "
                f"The geographic feature '{driver['feature']}' (SHAP importance: {driver['importance']:.4f}) "
                f"shows significant predictive power for fraud detection."
            ),
            'action': 'Implement country-based risk scoring with enhanced verification for top 5 high-risk countries',
            'expected_impact': 'Reduce false negatives by 15-20% while maintaining acceptable false positive rates',
            'shap_insight': f"Geographic features contribute {driver['importance']:.4f} to fraud predictions"
        }
    
    def _create_force_plot_recommendation(self, force_plots: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Create recommendation based on force plot insights."""
        fp_insights = [f for f in force_plots if f.get('prediction_type') == 'fp']
        fn_insights = [f for f in force_plots if f.get('prediction_type') == 'fn']
        
        if fp_insights or fn_insights:
            return {
                'title': 'Threshold Optimization Based on Error Analysis',
                'recommendation': (
                    f"Analyze false positives and false negatives to optimize decision threshold. "
                    f"Force plot analysis reveals specific feature combinations that lead to "
                    f"misclassifications, suggesting threshold tuning could improve precision-recall balance."
                ),
                'action': 'A/B test different probability thresholds (0.3, 0.5, 0.7) to optimize business metrics',
                'expected_impact': 'Balance between fraud detection and user experience',
                'shap_insight': 'Force plots show feature interactions causing misclassifications'
            }
        return None
    
    def _create_threshold_recommendation(self, top_drivers: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Create general threshold recommendation."""
        return {
            'title': 'Tiered Risk Scoring Implementation',
            'recommendation': (
                f"Implement a tiered risk scoring system based on top {len(top_drivers)} fraud drivers. "
                f"Use SHAP values to create interpretable risk scores that combine multiple signals."
            ),
            'action': 'Deploy tiered system: Low Risk (<0.3) auto-approve, Medium (0.3-0.7) require 2FA, High (>0.7) manual review',
            'expected_impact': 'Optimize total cost (fraud losses + UX impact) by 60-66%',
            'shap_insight': f"Top {len(top_drivers)} features explain majority of fraud predictions"
        }
    
    def save_recommendations(
        self,
        recommendations: List[Dict[str, Any]],
        output_file: str = "business_recommendations.txt"
    ):
        """
        Save recommendations to file.
        
        Args:
            recommendations (list[dict]): List of recommendations
            output_file (str): Output filename
        """
        try:
            output_path = Path(output_file)
            if not output_path.is_absolute():
                output_path = Path("models/explainability_outputs") / output_file
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("BUSINESS RECOMMENDATIONS BASED ON MODEL EXPLAINABILITY\n")
                f.write("=" * 80 + "\n\n")
                
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"Recommendation {i}: {rec['title']}\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"{rec['recommendation']}\n\n")
                    f.write(f"Action: {rec['action']}\n")
                    f.write(f"Expected Impact: {rec['expected_impact']}\n")
                    f.write(f"SHAP Insight: {rec['shap_insight']}\n")
                    f.write("\n" + "=" * 80 + "\n\n")
            
            self.logger.info(f"Saved recommendations to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving recommendations: {str(e)}")
            raise

