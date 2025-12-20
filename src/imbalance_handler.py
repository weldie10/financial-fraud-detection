"""
Class Imbalance Handler Module

This module provides functionality to handle class imbalance using
SMOTE, undersampling, or other techniques.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter


class ImbalanceHandler:
    """
    A class to handle class imbalance in datasets.
    
    Attributes:
        logger (logging.Logger): Logger instance
        method (str): Resampling method
        sampler: Resampling sampler instance
        is_fitted: Whether sampler has been fitted
    """
    
    def __init__(
        self,
        method: str = "smote",
        random_state: int = 42,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the ImbalanceHandler.
        
        Args:
            method (str): Resampling method ('smote', 'adasyn', 'borderline_smote',
                        'undersample', 'smote_tomek', 'smote_enn')
            random_state (int): Random state for reproducibility
            logger (logging.Logger, optional): Logger instance
        """
        self.logger = logger or self._setup_logger()
        self.method = method
        self.random_state = random_state
        self.sampler = self._setup_sampler()
        self.is_fitted = False
    
    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Setup a logger for the ImbalanceHandler."""
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
    
    def _setup_sampler(self):
        """Setup the resampling sampler based on method."""
        samplers = {
            'smote': SMOTE(random_state=self.random_state),
            'adasyn': ADASYN(random_state=self.random_state),
            'borderline_smote': BorderlineSMOTE(random_state=self.random_state),
            'undersample': RandomUnderSampler(random_state=self.random_state),
            'smote_tomek': SMOTETomek(random_state=self.random_state),
            'smote_enn': SMOTEENN(random_state=self.random_state)
        }
        
        sampler = samplers.get(self.method.lower())
        if sampler is None:
            self.logger.warning(
                f"Unknown method {self.method}. Using SMOTE. "
                f"Available methods: {list(samplers.keys())}"
            )
            sampler = SMOTE(random_state=self.random_state)
        
        return sampler
    
    def analyze_imbalance(
        self,
        y: pd.Series
    ) -> Dict[str, Any]:
        """
        Analyze class distribution and calculate imbalance metrics.
        
        Args:
            y (pd.Series): Target variable
            
        Returns:
            dict: Dictionary with imbalance statistics
        """
        try:
            class_counts = Counter(y)
            total_samples = len(y)
            
            stats = {
                'class_counts': dict(class_counts),
                'class_proportions': {
                    cls: count / total_samples 
                    for cls, count in class_counts.items()
                },
                'total_samples': total_samples,
                'num_classes': len(class_counts),
                'imbalance_ratio': max(class_counts.values()) / min(class_counts.values())
            }
            
            self.logger.info("Class distribution before resampling:")
            for cls in sorted(stats['class_counts'].keys()):
                count = stats['class_counts'][cls]
                prop = stats['class_proportions'][cls] * 100
                self.logger.info(f"  Class {cls}: {count:,} ({prop:.2f}%)")
            
            self.logger.info(f"Imbalance ratio: {stats['imbalance_ratio']:.2f}")
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error analyzing imbalance: {str(e)}")
            raise
    
    def resample(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fit_only: bool = False
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, any]]:
        """
        Resample the dataset to balance classes.
        
        Args:
            X (pd.DataFrame): Feature dataframe
            y (pd.Series): Target variable
            fit_only (bool): If True, only fit the sampler without transforming
            
        Returns:
            tuple: (X_resampled, y_resampled, stats_dict)
        """
        try:
            self.logger.info(f"Resampling using {self.method} method")
            
            # Analyze imbalance before resampling
            before_stats = self.analyze_imbalance(y)
            
            # Convert to numpy arrays for imbalanced-learn
            X_array = X.values if isinstance(X, pd.DataFrame) else X
            y_array = y.values if isinstance(y, pd.Series) else y
            
            # Fit and transform
            if fit_only:
                X_resampled, y_resampled = self.sampler.fit_resample(X_array, y_array)
                self.is_fitted = True
            else:
                X_resampled, y_resampled = self.sampler.fit_resample(X_array, y_array)
                self.is_fitted = True
            
            # Convert back to pandas
            if isinstance(X, pd.DataFrame):
                X_resampled = pd.DataFrame(
                    X_resampled,
                    columns=X.columns,
                    index=pd.RangeIndex(len(X_resampled))
                )
            else:
                X_resampled = pd.DataFrame(X_resampled)
            
            y_resampled = pd.Series(y_resampled)
            
            # Analyze imbalance after resampling
            after_stats = self.analyze_imbalance(y_resampled)
            
            # Create comparison stats
            comparison_stats = {
                'before': before_stats,
                'after': after_stats,
                'method': self.method,
                'samples_added': len(X_resampled) - len(X),
                'samples_removed': len(X) - len(X_resampled) if len(X_resampled) < len(X) else 0
            }
            
            self.logger.info("Class distribution after resampling:")
            for cls in sorted(after_stats['class_counts'].keys()):
                count = after_stats['class_counts'][cls]
                prop = after_stats['class_proportions'][cls] * 100
                self.logger.info(f"  Class {cls}: {count:,} ({prop:.2f}%)")
            
            self.logger.info(
                f"Resampling complete. "
                f"Original: {len(X):,} samples -> Resampled: {len(X_resampled):,} samples"
            )
            
            return X_resampled, y_resampled, comparison_stats
            
        except Exception as e:
            self.logger.error(f"Error in resampling: {str(e)}")
            raise
    
    def justify_method_choice(
        self,
        imbalance_ratio: float,
        dataset_size: int
    ) -> str:
        """
        Provide justification for method choice based on dataset characteristics.
        
        Args:
            imbalance_ratio (float): Ratio of majority to minority class
            dataset_size (int): Total number of samples
            
        Returns:
            str: Justification text
        """
        justification = f"""
        Method Selection Justification:
        
        Dataset Characteristics:
        - Total samples: {dataset_size:,}
        - Imbalance ratio: {imbalance_ratio:.2f}
        
        Selected Method: {self.method.upper()}
        
        Rationale:
        """
        
        if self.method.lower() == 'smote':
            justification += """
        - SMOTE (Synthetic Minority Oversampling Technique) is selected because:
          * It creates synthetic samples rather than duplicating existing ones
          * Works well for moderate to high imbalance ratios
          * Preserves the original data distribution
          * Suitable for datasets with sufficient samples (>1000)
        """
        elif self.method.lower() == 'undersample':
            justification += """
        - Random Undersampling is selected because:
          * Dataset is very large and imbalance is moderate
          * Reduces computational cost
          * May lose important information from majority class
          * Use when dataset size allows for reduction
        """
        elif self.method.lower() == 'smote_tomek':
            justification += """
        - SMOTE-Tomek is selected because:
          * Combines oversampling (SMOTE) with cleaning (Tomek links)
          * Removes borderline samples that may cause misclassification
          * Better than SMOTE alone for noisy datasets
          * Suitable for moderate to high imbalance
        """
        elif self.method.lower() == 'adasyn':
            justification += """
        - ADASYN (Adaptive Synthetic Sampling) is selected because:
          * Adaptively generates more samples for harder-to-learn examples
          * Better than SMOTE for highly imbalanced datasets
          * Focuses on minority class samples near decision boundary
        """
        else:
            justification += f"""
        - {self.method.upper()} is selected based on dataset characteristics
        """
        
        return justification

