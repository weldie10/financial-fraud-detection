"""
Geolocation Integration Module

This module provides functionality to convert IP addresses to integer format
and merge with country data using range-based lookup.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple
from pathlib import Path


class GeolocationMapper:
    """
    A class to map IP addresses to countries using range-based lookup.
    
    Attributes:
        logger (logging.Logger): Logger instance for tracking operations
        ip_country_df (pd.DataFrame): DataFrame with IP ranges and countries
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the GeolocationMapper.
        
        Args:
            logger (logging.Logger, optional): Logger instance
        """
        self.logger = logger or self._setup_logger()
        self.ip_country_df = None
    
    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Setup a logger for the GeolocationMapper."""
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
    
    def ip_to_integer(self, ip_address: str) -> Optional[int]:
        """
        Convert IP address string to integer.
        
        Args:
            ip_address (str): IP address in format 'x.x.x.x'
            
        Returns:
            int: Integer representation of IP address, None if invalid
        """
        try:
            if pd.isna(ip_address) or ip_address == '':
                return None
            
            parts = str(ip_address).split('.')
            if len(parts) != 4:
                return None
            
            # Validate all parts are numeric
            if not all(part.isdigit() for part in parts):
                return None
            
            # Convert to integer: a.b.c.d -> a*256^3 + b*256^2 + c*256 + d
            integer_ip = (
                int(parts[0]) * (256 ** 3) +
                int(parts[1]) * (256 ** 2) +
                int(parts[2]) * 256 +
                int(parts[3])
            )
            
            return integer_ip
            
        except Exception as e:
            self.logger.warning(f"Error converting IP {ip_address} to integer: {str(e)}")
            return None
    
    def load_ip_country_mapping(
        self, 
        filepath: str,
        ip_start_col: str = "lower_bound_ip_address",
        ip_end_col: str = "upper_bound_ip_address",
        country_col: str = "country"
    ) -> pd.DataFrame:
        """
        Load IP to country mapping file.
        
        Args:
            filepath (str): Path to the IP address to country mapping CSV
            ip_start_col (str): Column name for lower bound IP address
            ip_end_col (str): Column name for upper bound IP address
            country_col (str): Column name for country
            
        Returns:
            pd.DataFrame: DataFrame with IP ranges and countries
        """
        try:
            self.logger.info(f"Loading IP to country mapping from {filepath}")
            
            df = pd.read_csv(filepath)
            
            # Validate required columns
            required_cols = [ip_start_col, ip_end_col, country_col]
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                raise ValueError(
                    f"Missing required columns: {missing_cols}. "
                    f"Available columns: {list(df.columns)}"
                )
            
            # Convert IP addresses to integers
            self.logger.info("Converting IP addresses to integers")
            df['ip_start_int'] = df[ip_start_col].apply(self.ip_to_integer)
            df['ip_end_int'] = df[ip_end_col].apply(self.ip_to_integer)
            
            # Remove rows with invalid IP conversions
            invalid_mask = df['ip_start_int'].isna() | df['ip_end_int'].isna()
            if invalid_mask.sum() > 0:
                self.logger.warning(
                    f"Removing {invalid_mask.sum()} rows with invalid IP addresses"
                )
                df = df[~invalid_mask].copy()
            
            # Sort by IP start for efficient lookup
            df = df.sort_values('ip_start_int').reset_index(drop=True)
            
            self.ip_country_df = df
            self.logger.info(
                f"Loaded {len(df)} IP range mappings for {df[country_col].nunique()} countries"
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading IP to country mapping: {str(e)}")
            raise
    
    def find_country_for_ip(self, ip_integer: int) -> Optional[str]:
        """
        Find country for a given IP integer using binary search.
        
        Args:
            ip_integer (int): Integer representation of IP address
            
        Returns:
            str: Country name if found, None otherwise
        """
        try:
            if self.ip_country_df is None:
                raise ValueError("IP to country mapping not loaded. Call load_ip_country_mapping first.")
            
            if ip_integer is None or pd.isna(ip_integer):
                return None
            
            # Binary search for matching range
            left, right = 0, len(self.ip_country_df) - 1
            
            while left <= right:
                mid = (left + right) // 2
                row = self.ip_country_df.iloc[mid]
                
                if row['ip_start_int'] <= ip_integer <= row['ip_end_int']:
                    return row['country']
                elif ip_integer < row['ip_start_int']:
                    right = mid - 1
                else:
                    left = mid + 1
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error finding country for IP {ip_integer}: {str(e)}")
            return None
    
    def merge_with_country_data(
        self, 
        df: pd.DataFrame,
        ip_column: str = "ip_address",
        country_col_name: str = "country"
    ) -> pd.DataFrame:
        """
        Merge fraud data with country data based on IP addresses.
        
        Args:
            df (pd.DataFrame): DataFrame with IP addresses
            ip_column (str): Column name containing IP addresses
            country_col_name (str): Name for the new country column
            
        Returns:
            pd.DataFrame: DataFrame with country information added
        """
        try:
            if self.ip_country_df is None:
                raise ValueError("IP to country mapping not loaded. Call load_ip_country_mapping first.")
            
            if ip_column not in df.columns:
                raise ValueError(f"IP column '{ip_column}' not found in dataframe")
            
            self.logger.info(f"Merging country data for {len(df)} records")
            
            df_merged = df.copy()
            
            # Convert IP addresses to integers
            self.logger.info("Converting IP addresses to integers")
            df_merged['ip_integer'] = df_merged[ip_column].apply(self.ip_to_integer)
            
            # Map to countries
            self.logger.info("Mapping IP addresses to countries")
            df_merged[country_col_name] = df_merged['ip_integer'].apply(
                self.find_country_for_ip
            )
            
            # Report mapping statistics
            mapped_count = df_merged[country_col_name].notna().sum()
            mapping_rate = (mapped_count / len(df_merged)) * 100
            
            self.logger.info(
                f"Successfully mapped {mapped_count} IP addresses to countries "
                f"({mapping_rate:.2f}%)"
            )
            
            # Drop temporary column
            df_merged = df_merged.drop(columns=['ip_integer'])
            
            return df_merged
            
        except Exception as e:
            self.logger.error(f"Error merging country data: {str(e)}")
            raise
    
    def analyze_fraud_by_country(
        self, 
        df: pd.DataFrame,
        country_col: str = "country",
        fraud_col: str = "class"
    ) -> pd.DataFrame:
        """
        Analyze fraud patterns by country.
        
        Args:
            df (pd.DataFrame): DataFrame with country and fraud information
            country_col (str): Column name for country
            fraud_col (str): Column name for fraud indicator
            
        Returns:
            pd.DataFrame: Summary statistics by country
        """
        try:
            if country_col not in df.columns:
                raise ValueError(f"Country column '{country_col}' not found")
            if fraud_col not in df.columns:
                raise ValueError(f"Fraud column '{fraud_col}' not found")
            
            self.logger.info("Analyzing fraud patterns by country")
            
            # Group by country
            country_stats = df.groupby(country_col).agg({
                fraud_col: ['count', 'sum', 'mean']
            }).reset_index()
            
            country_stats.columns = [
                'country', 
                'total_transactions', 
                'fraud_count', 
                'fraud_rate'
            ]
            
            country_stats = country_stats.sort_values(
                'fraud_rate', 
                ascending=False
            )
            
            self.logger.info(
                f"Analyzed {len(country_stats)} countries. "
                f"Top fraud rate: {country_stats.iloc[0]['fraud_rate']:.4f}"
            )
            
            return country_stats
            
        except Exception as e:
            self.logger.error(f"Error analyzing fraud by country: {str(e)}")
            raise

