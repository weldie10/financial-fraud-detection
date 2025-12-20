"""
Exploratory Data Analysis Module

This module provides functionality for univariate, bivariate,
and class distribution analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path


class EDA:
    """
    A class for exploratory data analysis.
    
    Attributes:
        logger (logging.Logger): Logger instance for tracking operations
        output_dir (Path): Directory to save plots
    """
    
    def __init__(
        self, 
        output_dir: str = "notebooks/eda_outputs",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the EDA class.
        
        Args:
            output_dir (str): Directory to save analysis outputs
            logger (logging.Logger, optional): Logger instance
        """
        self.logger = logger or self._setup_logger()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Setup a logger for the EDA class."""
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
    
    def analyze_class_distribution(
        self, 
        df: pd.DataFrame,
        target_column: str,
        plot: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze class distribution and quantify imbalance.
        
        Args:
            df (pd.DataFrame): DataFrame with target column
            target_column (str): Name of target column
            plot (bool): Whether to create visualization
            
        Returns:
            dict: Dictionary with distribution statistics
        """
        try:
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found")
            
            self.logger.info(f"Analyzing class distribution for {target_column}")
            
            # Calculate distribution
            class_counts = df[target_column].value_counts().sort_index()
            class_proportions = df[target_column].value_counts(normalize=True).sort_index()
            
            # Calculate imbalance ratio
            minority_class = class_counts.min()
            majority_class = class_counts.max()
            imbalance_ratio = majority_class / minority_class if minority_class > 0 else np.inf
            
            stats = {
                'class_counts': class_counts.to_dict(),
                'class_proportions': class_proportions.to_dict(),
                'total_samples': len(df),
                'num_classes': len(class_counts),
                'imbalance_ratio': imbalance_ratio,
                'minority_class': class_counts.idxmin(),
                'majority_class': class_counts.idxmax()
            }
            
            self.logger.info(f"Class distribution:")
            for cls, count in class_counts.items():
                pct = class_proportions[cls] * 100
                self.logger.info(f"  Class {cls}: {count} ({pct:.2f}%)")
            
            self.logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}")
            
            # Create visualization
            if plot:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                # Count plot
                class_counts.plot(kind='bar', ax=axes[0], color=['skyblue', 'salmon'])
                axes[0].set_title('Class Distribution (Counts)', fontsize=14, fontweight='bold')
                axes[0].set_xlabel('Class', fontsize=12)
                axes[0].set_ylabel('Count', fontsize=12)
                axes[0].tick_params(axis='x', rotation=0)
                
                # Proportion plot
                class_proportions.plot(kind='bar', ax=axes[1], color=['skyblue', 'salmon'])
                axes[1].set_title('Class Distribution (Proportions)', fontsize=14, fontweight='bold')
                axes[1].set_xlabel('Class', fontsize=12)
                axes[1].set_ylabel('Proportion', fontsize=12)
                axes[1].tick_params(axis='x', rotation=0)
                
                plt.tight_layout()
                plt.savefig(
                    self.output_dir / 'class_distribution.png',
                    dpi=300,
                    bbox_inches='tight'
                )
                plt.close()
                
                self.logger.info(f"Saved class distribution plot to {self.output_dir}")
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error analyzing class distribution: {str(e)}")
            raise
    
    def univariate_analysis(
        self, 
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        plot: bool = True
    ) -> pd.DataFrame:
        """
        Perform univariate analysis on specified columns.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
            columns (list[str], optional): Columns to analyze. If None, analyzes all numeric columns
            plot (bool): Whether to create visualizations
            
        Returns:
            pd.DataFrame: Summary statistics
        """
        try:
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()
            else:
                # Filter to only existing numeric columns
                columns = [c for c in columns if c in df.columns and 
                          pd.api.types.is_numeric_dtype(df[c])]
            
            if not columns:
                self.logger.warning("No numeric columns found for univariate analysis")
                return pd.DataFrame()
            
            self.logger.info(f"Performing univariate analysis on {len(columns)} columns")
            
            # Calculate summary statistics
            summary_stats = df[columns].describe()
            
            # Add additional statistics
            additional_stats = pd.DataFrame({
                'skewness': df[columns].skew(),
                'kurtosis': df[columns].kurtosis(),
                'missing_count': df[columns].isnull().sum(),
                'missing_pct': (df[columns].isnull().sum() / len(df)) * 100
            })
            
            summary_stats = pd.concat([summary_stats.T, additional_stats], axis=1)
            
            # Create visualizations
            if plot:
                n_cols = min(3, len(columns))
                n_rows = (len(columns) + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
                axes = axes.flatten() if len(columns) > 1 else [axes]
                
                for idx, col in enumerate(columns):
                    ax = axes[idx]
                    df[col].hist(bins=50, ax=ax, edgecolor='black', alpha=0.7)
                    ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
                    ax.set_xlabel(col, fontsize=10)
                    ax.set_ylabel('Frequency', fontsize=10)
                    ax.grid(True, alpha=0.3)
                
                # Hide extra subplots
                for idx in range(len(columns), len(axes)):
                    axes[idx].axis('off')
                
                plt.tight_layout()
                plt.savefig(
                    self.output_dir / 'univariate_distributions.png',
                    dpi=300,
                    bbox_inches='tight'
                )
                plt.close()
                
                self.logger.info(f"Saved univariate analysis plots to {self.output_dir}")
            
            return summary_stats
            
        except Exception as e:
            self.logger.error(f"Error in univariate analysis: {str(e)}")
            raise
    
    def bivariate_analysis(
        self, 
        df: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        plot: bool = True
    ) -> pd.DataFrame:
        """
        Perform bivariate analysis between features and target.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
            target_column (str): Name of target column
            feature_columns (list[str], optional): Feature columns to analyze
            plot (bool): Whether to create visualizations
            
        Returns:
            pd.DataFrame: Correlation and relationship statistics
        """
        try:
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found")
            
            if feature_columns is None:
                feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                feature_columns = [c for c in feature_columns if c != target_column]
            else:
                feature_columns = [c for c in feature_columns if c in df.columns and 
                                 c != target_column]
            
            if not feature_columns:
                self.logger.warning("No feature columns found for bivariate analysis")
                return pd.DataFrame()
            
            self.logger.info(
                f"Performing bivariate analysis: {len(feature_columns)} features vs {target_column}"
            )
            
            # Calculate correlations
            correlations = df[feature_columns + [target_column]].corr()[target_column]
            correlations = correlations.drop(target_column).sort_values(ascending=False)
            
            # Create correlation dataframe
            correlation_df = pd.DataFrame({
                'feature': correlations.index,
                'correlation': correlations.values,
                'abs_correlation': correlations.abs().values
            }).sort_values('abs_correlation', ascending=False)
            
            # Create visualizations
            if plot:
                # Correlation plot
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                
                # Top correlations
                top_n = min(20, len(correlation_df))
                top_corrs = correlation_df.head(top_n)
                
                axes[0].barh(range(len(top_corrs)), top_corrs['correlation'], 
                           color=['red' if x < 0 else 'green' for x in top_corrs['correlation']])
                axes[0].set_yticks(range(len(top_corrs)))
                axes[0].set_yticklabels(top_corrs['feature'])
                axes[0].set_xlabel('Correlation with Target', fontsize=12)
                axes[0].set_title(f'Top {top_n} Feature Correlations with Target', 
                                fontsize=14, fontweight='bold')
                axes[0].grid(True, alpha=0.3)
                
                # Box plots for top features
                if len(feature_columns) > 0:
                    top_features = correlation_df.head(6)['feature'].tolist()
                    n_features = len(top_features)
                    n_cols = min(3, n_features)
                    n_rows = (n_features + n_cols - 1) // n_cols
                    
                    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
                    axes2 = axes2.flatten() if n_features > 1 else [axes2]
                    
                    for idx, feature in enumerate(top_features):
                        ax = axes2[idx]
                        df.boxplot(column=feature, by=target_column, ax=ax)
                        ax.set_title(f'{feature} by {target_column}', fontsize=12, fontweight='bold')
                        ax.set_xlabel(target_column, fontsize=10)
                        ax.set_ylabel(feature, fontsize=10)
                    
                    # Hide extra subplots
                    for idx in range(n_features, len(axes2)):
                        axes2[idx].axis('off')
                    
                    plt.suptitle('')
                    plt.tight_layout()
                    plt.savefig(
                        self.output_dir / 'bivariate_boxplots.png',
                        dpi=300,
                        bbox_inches='tight'
                    )
                    plt.close()
                
                plt.tight_layout()
                plt.savefig(
                    self.output_dir / 'bivariate_correlations.png',
                    dpi=300,
                    bbox_inches='tight'
                )
                plt.close()
                
                self.logger.info(f"Saved bivariate analysis plots to {self.output_dir}")
            
            return correlation_df
            
        except Exception as e:
            self.logger.error(f"Error in bivariate analysis: {str(e)}")
            raise
    
    def generate_summary_report(
        self, 
        df: pd.DataFrame,
        target_column: str,
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive EDA summary report.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
            target_column (str): Name of target column
            output_file (str, optional): Path to save report
            
        Returns:
            str: Report text
        """
        try:
            self.logger.info("Generating comprehensive EDA report")
            
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append("EXPLORATORY DATA ANALYSIS REPORT")
            report_lines.append("=" * 80)
            report_lines.append("")
            
            # Dataset overview
            report_lines.append("DATASET OVERVIEW")
            report_lines.append("-" * 80)
            report_lines.append(f"Total rows: {len(df):,}")
            report_lines.append(f"Total columns: {len(df.columns)}")
            report_lines.append(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            report_lines.append("")
            
            # Class distribution
            class_stats = self.analyze_class_distribution(df, target_column, plot=False)
            report_lines.append("CLASS DISTRIBUTION")
            report_lines.append("-" * 80)
            for cls, count in class_stats['class_counts'].items():
                pct = class_stats['class_proportions'][cls] * 100
                report_lines.append(f"Class {cls}: {count:,} ({pct:.2f}%)")
            report_lines.append(f"Imbalance ratio: {class_stats['imbalance_ratio']:.2f}")
            report_lines.append("")
            
            # Missing values
            missing = df.isnull().sum()
            missing_pct = (missing / len(df)) * 100
            report_lines.append("MISSING VALUES")
            report_lines.append("-" * 80)
            if missing.sum() > 0:
                for col in missing[missing > 0].index:
                    report_lines.append(f"{col}: {missing[col]:,} ({missing_pct[col]:.2f}%)")
            else:
                report_lines.append("No missing values found")
            report_lines.append("")
            
            # Top correlations
            correlations = self.bivariate_analysis(df, target_column, plot=False)
            if not correlations.empty:
                report_lines.append("TOP FEATURE CORRELATIONS WITH TARGET")
                report_lines.append("-" * 80)
                top_10 = correlations.head(10)
                for _, row in top_10.iterrows():
                    report_lines.append(f"{row['feature']}: {row['correlation']:.4f}")
                report_lines.append("")
            
            report = "\n".join(report_lines)
            
            if output_file:
                output_path = self.output_dir / output_file
                with open(output_path, 'w') as f:
                    f.write(report)
                self.logger.info(f"Saved report to {output_path}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating summary report: {str(e)}")
            raise

