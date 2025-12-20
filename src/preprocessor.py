"""
Main Preprocessing Pipeline Module

This module provides a complete preprocessing pipeline that orchestrates
all data preparation steps including loading, cleaning, feature engineering,
transformation, and handling class imbalance.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path

try:
    from .data_loader import DataLoader
    from .data_cleaner import DataCleaner
    from .geolocation import GeolocationMapper
    from .feature_engineer import FeatureEngineer
    from .data_transformer import DataTransformer
    from .imbalance_handler import ImbalanceHandler
    from .eda import EDA
except ImportError:
    # Fallback for direct imports
    from data_loader import DataLoader
    from data_cleaner import DataCleaner
    from geolocation import GeolocationMapper
    from feature_engineer import FeatureEngineer
    from data_transformer import DataTransformer
    from imbalance_handler import ImbalanceHandler
    from eda import EDA


class PreprocessingPipeline:
    """
    Main preprocessing pipeline class that orchestrates all preprocessing steps.
    
    Attributes:
        logger (logging.Logger): Logger instance
        data_loader (DataLoader): Data loading component
        data_cleaner (DataCleaner): Data cleaning component
        geolocation_mapper (GeolocationMapper): Geolocation mapping component
        feature_engineer (FeatureEngineer): Feature engineering component
        data_transformer (DataTransformer): Data transformation component
        imbalance_handler (ImbalanceHandler): Class imbalance handling component
        eda (EDA): Exploratory data analysis component
    """
    
    def __init__(
        self,
        data_dir: str = "data/raw",
        output_dir: str = "data/processed",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            data_dir (str): Directory containing raw data
            output_dir (str): Directory to save processed data
            logger (logging.Logger, optional): Logger instance
        """
        self.logger = logger or self._setup_logger()
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_loader = DataLoader(data_dir=data_dir, logger=self.logger)
        self.data_cleaner = DataCleaner(logger=self.logger)
        self.geolocation_mapper = GeolocationMapper(logger=self.logger)
        self.feature_engineer = FeatureEngineer(logger=self.logger)
        self.data_transformer = DataTransformer(logger=self.logger)
        self.imbalance_handler = ImbalanceHandler(logger=self.logger)
        self.eda = EDA(output_dir=str(self.output_dir / "eda_outputs"), logger=self.logger)
        
        self.logger.info("Preprocessing pipeline initialized")
    
    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Setup a logger for the PreprocessingPipeline."""
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
    
    def process_fraud_data(
        self,
        fraud_data_file: str = "Fraud_Data.csv",
        ip_country_file: Optional[str] = "IpAddress_to_Country.csv",
        target_column: str = "class",
        user_column: str = "user_id",
        purchase_datetime: str = "purchase_time",
        signup_datetime: str = "signup_time",
        ip_column: str = "ip_address",
        amount_column: Optional[str] = None,
        perform_eda: bool = True,
        handle_imbalance: bool = True,
        save_processed: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Complete preprocessing pipeline for fraud data.
        
        Args:
            fraud_data_file (str): Filename of fraud data CSV
            ip_country_file (str, optional): Filename of IP to country mapping CSV
            target_column (str): Name of target column
            user_column (str): Name of user identifier column
            purchase_datetime (str): Name of purchase datetime column
            signup_datetime (str): Name of signup datetime column
            ip_column (str): Name of IP address column
            amount_column (str, optional): Name of transaction amount column
            perform_eda (bool): Whether to perform EDA
            handle_imbalance (bool): Whether to handle class imbalance
            save_processed (bool): Whether to save processed data
            
        Returns:
            tuple: (processed_dataframe, metadata_dict)
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("STARTING FRAUD DATA PREPROCESSING PIPELINE")
            self.logger.info("=" * 80)
            
            metadata = {
                'steps_completed': [],
                'statistics': {}
            }
            
            # Step 1: Load data
            self.logger.info("\n[Step 1/7] Loading data...")
            fraud_df = self.data_loader.load_csv(fraud_data_file)
            self.data_loader.validate_dataframe(fraud_df, min_rows=1)
            metadata['steps_completed'].append('data_loading')
            metadata['statistics']['initial_shape'] = fraud_df.shape
            self.logger.info(f"Loaded {fraud_df.shape[0]:,} rows, {fraud_df.shape[1]} columns")
            
            # Step 2: Data cleaning
            self.logger.info("\n[Step 2/7] Cleaning data...")
            date_columns = [purchase_datetime]
            if signup_datetime in fraud_df.columns:
                date_columns.append(signup_datetime)
            
            fraud_df = self.data_cleaner.clean(
                fraud_df,
                handle_missing=True,
                remove_dups=True,
                correct_types=True,
                date_columns=date_columns
            )
            metadata['steps_completed'].append('data_cleaning')
            metadata['statistics']['after_cleaning_shape'] = fraud_df.shape
            
            # Step 3: Geolocation integration
            if ip_country_file and ip_column in fraud_df.columns:
                self.logger.info("\n[Step 3/7] Integrating geolocation data...")
                try:
                    self.geolocation_mapper.load_ip_country_mapping(
                        str(self.data_dir / ip_country_file)
                    )
                    fraud_df = self.geolocation_mapper.merge_with_country_data(
                        fraud_df,
                        ip_column=ip_column
                    )
                    
                    # Analyze fraud by country
                    country_stats = self.geolocation_mapper.analyze_fraud_by_country(
                        fraud_df,
                        fraud_col=target_column
                    )
                    metadata['statistics']['country_fraud_stats'] = country_stats.to_dict('records')
                    metadata['steps_completed'].append('geolocation_integration')
                except Exception as e:
                    self.logger.warning(f"Geolocation integration failed: {str(e)}")
                    metadata['steps_completed'].append('geolocation_integration_skipped')
            else:
                self.logger.info("\n[Step 3/7] Skipping geolocation integration")
                metadata['steps_completed'].append('geolocation_integration_skipped')
            
            # Step 4: Feature engineering
            self.logger.info("\n[Step 4/7] Engineering features...")
            fraud_df = self.feature_engineer.engineer_all_features(
                fraud_df,
                user_column=user_column,
                purchase_datetime=purchase_datetime,
                signup_datetime=signup_datetime if signup_datetime in fraud_df.columns else None,
                amount_column=amount_column
            )
            metadata['steps_completed'].append('feature_engineering')
            metadata['statistics']['after_feature_engineering_shape'] = fraud_df.shape
            
            # Step 5: Exploratory Data Analysis
            if perform_eda and target_column in fraud_df.columns:
                self.logger.info("\n[Step 5/7] Performing exploratory data analysis...")
                try:
                    # Class distribution
                    class_stats = self.eda.analyze_class_distribution(
                        fraud_df,
                        target_column,
                        plot=True
                    )
                    metadata['statistics']['class_distribution'] = class_stats
                    
                    # Univariate analysis
                    numeric_cols = fraud_df.select_dtypes(include=[np.number]).columns.tolist()
                    if target_column in numeric_cols:
                        numeric_cols.remove(target_column)
                    if numeric_cols:
                        univariate_stats = self.eda.univariate_analysis(
                            fraud_df,
                            columns=numeric_cols[:10],  # Limit to top 10 for performance
                            plot=True
                        )
                    
                    # Bivariate analysis
                    bivariate_stats = self.eda.bivariate_analysis(
                        fraud_df,
                        target_column,
                        plot=True
                    )
                    metadata['statistics']['bivariate_analysis'] = bivariate_stats.to_dict('records')
                    
                    # Generate summary report
                    report = self.eda.generate_summary_report(
                        fraud_df,
                        target_column,
                        output_file="eda_summary_report.txt"
                    )
                    
                    metadata['steps_completed'].append('eda')
                except Exception as e:
                    self.logger.warning(f"EDA failed: {str(e)}")
                    metadata['steps_completed'].append('eda_skipped')
            else:
                self.logger.info("\n[Step 5/7] Skipping EDA")
                metadata['steps_completed'].append('eda_skipped')
            
            # Step 6: Data transformation
            self.logger.info("\n[Step 6/7] Transforming data (scaling and encoding)...")
            fraud_df = self.data_transformer.fit_transform(
                fraud_df,
                target_column=target_column
            )
            metadata['steps_completed'].append('data_transformation')
            metadata['statistics']['after_transformation_shape'] = fraud_df.shape
            
            # Save transformers
            transformer_path = self.output_dir / "transformers.joblib"
            self.data_transformer.save_transformers(str(transformer_path))
            
            # Step 7: Handle class imbalance (only on training data)
            if handle_imbalance and target_column in fraud_df.columns:
                self.logger.info("\n[Step 7/7] Handling class imbalance...")
                try:
                    X = fraud_df.drop(columns=[target_column])
                    y = fraud_df[target_column]
                    
                    X_resampled, y_resampled, imbalance_stats = self.imbalance_handler.resample(
                        X, y
                    )
                    
                    # Reconstruct dataframe
                    fraud_df = pd.concat([X_resampled, y_resampled], axis=1)
                    
                    metadata['statistics']['imbalance_handling'] = imbalance_stats
                    metadata['steps_completed'].append('imbalance_handling')
                    
                    # Log justification
                    justification = self.imbalance_handler.justify_method_choice(
                        imbalance_stats['before']['imbalance_ratio'],
                        len(X)
                    )
                    self.logger.info(justification)
                    
                except Exception as e:
                    self.logger.warning(f"Imbalance handling failed: {str(e)}")
                    metadata['steps_completed'].append('imbalance_handling_skipped')
            else:
                self.logger.info("\n[Step 7/7] Skipping imbalance handling")
                metadata['steps_completed'].append('imbalance_handling_skipped')
            
            # Save processed data
            if save_processed:
                output_file = self.output_dir / "processed_fraud_data.csv"
                fraud_df.to_csv(output_file, index=False)
                self.logger.info(f"\nSaved processed data to {output_file}")
                metadata['output_file'] = str(output_file)
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("PREPROCESSING PIPELINE COMPLETE")
            self.logger.info("=" * 80)
            self.logger.info(f"Final shape: {fraud_df.shape[0]:,} rows, {fraud_df.shape[1]} columns")
            
            return fraud_df, metadata
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing pipeline: {str(e)}")
            raise
    
    def process_credit_card_data(
        self,
        credit_card_file: str = "creditcard.csv",
        target_column: str = "Class",
        perform_eda: bool = True,
        handle_imbalance: bool = True,
        save_processed: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Complete preprocessing pipeline for credit card data.
        
        Args:
            credit_card_file (str): Filename of credit card data CSV
            target_column (str): Name of target column
            perform_eda (bool): Whether to perform EDA
            handle_imbalance (bool): Whether to handle class imbalance
            save_processed (bool): Whether to save processed data
            
        Returns:
            tuple: (processed_dataframe, metadata_dict)
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("STARTING CREDIT CARD DATA PREPROCESSING PIPELINE")
            self.logger.info("=" * 80)
            
            metadata = {
                'steps_completed': [],
                'statistics': {}
            }
            
            # Step 1: Load data
            self.logger.info("\n[Step 1/5] Loading data...")
            cc_df = self.data_loader.load_csv(credit_card_file)
            self.data_loader.validate_dataframe(cc_df, min_rows=1)
            metadata['steps_completed'].append('data_loading')
            metadata['statistics']['initial_shape'] = cc_df.shape
            
            # Step 2: Data cleaning
            self.logger.info("\n[Step 2/5] Cleaning data...")
            cc_df = self.data_cleaner.clean(
                cc_df,
                handle_missing=True,
                remove_dups=True,
                correct_types=True
            )
            metadata['steps_completed'].append('data_cleaning')
            
            # Step 3: EDA
            if perform_eda and target_column in cc_df.columns:
                self.logger.info("\n[Step 3/5] Performing exploratory data analysis...")
                class_stats = self.eda.analyze_class_distribution(cc_df, target_column, plot=True)
                metadata['statistics']['class_distribution'] = class_stats
                metadata['steps_completed'].append('eda')
            
            # Step 4: Data transformation
            self.logger.info("\n[Step 4/5] Transforming data...")
            cc_df = self.data_transformer.fit_transform(cc_df, target_column=target_column)
            metadata['steps_completed'].append('data_transformation')
            
            # Step 5: Handle class imbalance
            if handle_imbalance and target_column in cc_df.columns:
                self.logger.info("\n[Step 5/5] Handling class imbalance...")
                X = cc_df.drop(columns=[target_column])
                y = cc_df[target_column]
                X_resampled, y_resampled, imbalance_stats = self.imbalance_handler.resample(X, y)
                cc_df = pd.concat([X_resampled, y_resampled], axis=1)
                metadata['statistics']['imbalance_handling'] = imbalance_stats
                metadata['steps_completed'].append('imbalance_handling')
            
            # Save processed data
            if save_processed:
                output_file = self.output_dir / "processed_creditcard_data.csv"
                cc_df.to_csv(output_file, index=False)
                self.logger.info(f"\nSaved processed data to {output_file}")
            
            self.logger.info("\nPREPROCESSING PIPELINE COMPLETE")
            return cc_df, metadata
            
        except Exception as e:
            self.logger.error(f"Error in credit card preprocessing: {str(e)}")
            raise

