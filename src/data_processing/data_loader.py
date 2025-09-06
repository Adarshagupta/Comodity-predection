"""
Data loading and preprocessing for commodity forecasting competition.
Handles LME, JPX, US Stock, and Forex market data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading and initial preprocessing of multi-market financial data.
    """
    
    def __init__(self, data_path: str = "data/"):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to data directory
        """
        self.data_path = Path(data_path)
        self.raw_data = {}
        self.processed_data = {}
        
    def load_lme_data(self, file_path: str) -> pd.DataFrame:
        """
        Load London Metal Exchange data.
        
        Args:
            file_path: Path to LME data file
            
        Returns:
            Processed LME DataFrame
        """
        try:
            df = pd.read_csv(self.data_path / file_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            
            # Calculate returns
            price_cols = [col for col in df.columns if 'price' in col.lower()]
            for col in price_cols:
                df[f'{col}_return'] = df[col].pct_change()
                
            logger.info(f"Loaded LME data: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading LME data: {e}")
            return pd.DataFrame()
    
    def load_jpx_data(self, file_path: str) -> pd.DataFrame:
        """
        Load Japan Exchange Group data.
        
        Args:
            file_path: Path to JPX data file
            
        Returns:
            Processed JPX DataFrame
        """
        try:
            df = pd.read_csv(self.data_path / file_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            
            # Calculate returns
            price_cols = [col for col in df.columns if 'price' in col.lower() or 'close' in col.lower()]
            for col in price_cols:
                df[f'{col}_return'] = df[col].pct_change()
                
            logger.info(f"Loaded JPX data: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading JPX data: {e}")
            return pd.DataFrame()
    
    def load_us_stock_data(self, file_path: str) -> pd.DataFrame:
        """
        Load US Stock market data.
        
        Args:
            file_path: Path to US Stock data file
            
        Returns:
            Processed US Stock DataFrame
        """
        try:
            df = pd.read_csv(self.data_path / file_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            
            # Calculate returns
            price_cols = [col for col in df.columns if any(x in col.lower() for x in ['price', 'close', 'open', 'high', 'low'])]
            for col in price_cols:
                df[f'{col}_return'] = df[col].pct_change()
                
            logger.info(f"Loaded US Stock data: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading US Stock data: {e}")
            return pd.DataFrame()
    
    def load_forex_data(self, file_path: str) -> pd.DataFrame:
        """
        Load Forex market data.
        
        Args:
            file_path: Path to Forex data file
            
        Returns:
            Processed Forex DataFrame
        """
        try:
            df = pd.read_csv(self.data_path / file_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            
            # Calculate returns
            rate_cols = [col for col in df.columns if 'rate' in col.lower() or 'usd' in col.lower()]
            for col in rate_cols:
                df[f'{col}_return'] = df[col].pct_change()
                
            logger.info(f"Loaded Forex data: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading Forex data: {e}")
            return pd.DataFrame()
    
    def load_all_data(self, data_files: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """
        Load all market data from specified files.
        
        Args:
            data_files: Dictionary mapping market names to file paths
            
        Returns:
            Dictionary of loaded DataFrames
        """
        data = {}
        
        for market, file_path in data_files.items():
            if market.lower() == 'lme':
                data[market] = self.load_lme_data(file_path)
            elif market.lower() == 'jpx':
                data[market] = self.load_jpx_data(file_path)
            elif market.lower() == 'us_stock':
                data[market] = self.load_us_stock_data(file_path)
            elif market.lower() == 'forex':
                data[market] = self.load_forex_data(file_path)
            else:
                logger.warning(f"Unknown market type: {market}")
                
        self.raw_data = data
        return data
    
    def create_price_difference_series(self, df: pd.DataFrame, asset1: str, asset2: str) -> pd.Series:
        """
        Create price-difference series between two assets.
        
        Args:
            df: DataFrame containing asset prices
            asset1: Name of first asset column
            asset2: Name of second asset column
            
        Returns:
            Price difference series
        """
        if asset1 not in df.columns or asset2 not in df.columns:
            logger.error(f"Assets {asset1} or {asset2} not found in DataFrame")
            return pd.Series()
            
        return df[asset1] - df[asset2]
    
    def align_data_by_date(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Align all market data by common dates.
        
        Args:
            data_dict: Dictionary of market DataFrames
            
        Returns:
            Combined DataFrame with aligned dates
        """
        if not data_dict:
            return pd.DataFrame()
            
        # Find common date range
        date_ranges = []
        for df in data_dict.values():
            if not df.empty:
                date_ranges.append((df.index.min(), df.index.max()))
        
        if not date_ranges:
            return pd.DataFrame()
            
        common_start = max(start for start, _ in date_ranges)
        common_end = min(end for _, end in date_ranges)
        
        # Create date range
        date_range = pd.date_range(start=common_start, end=common_end, freq='D')
        
        # Align all data
        aligned_data = pd.DataFrame(index=date_range)
        
        for market, df in data_dict.items():
            if not df.empty:
                # Add market prefix to column names
                df_copy = df.copy()
                df_copy.columns = [f"{market}_{col}" for col in df_copy.columns]
                
                # Align to common date range
                df_aligned = df_copy.reindex(date_range, method='ffill')
                
                # Merge with main DataFrame
                aligned_data = aligned_data.join(df_aligned, how='left')
        
        logger.info(f"Aligned data shape: {aligned_data.shape}")
        return aligned_data
    
    def clean_data(self, df: pd.DataFrame, 
                   fill_method: str = 'forward',
                   max_missing_pct: float = 0.1) -> pd.DataFrame:
        """
        Clean and handle missing data.
        
        Args:
            df: Input DataFrame
            fill_method: Method for filling missing values ('forward', 'backward', 'mean', 'drop')
            max_missing_pct: Maximum percentage of missing values allowed per column
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning data with shape: {df.shape}")
        
        # Remove columns with too many missing values
        missing_pct = df.isnull().sum() / len(df)
        cols_to_keep = missing_pct[missing_pct <= max_missing_pct].index
        df_clean = df[cols_to_keep].copy()
        
        logger.info(f"Removed {len(df.columns) - len(cols_to_keep)} columns with >{max_missing_pct*100}% missing values")
        
        # Fill missing values
        if fill_method == 'forward':
            df_clean = df_clean.fillna(method='ffill')
        elif fill_method == 'backward':
            df_clean = df_clean.fillna(method='bfill')
        elif fill_method == 'mean':
            df_clean = df_clean.fillna(df_clean.mean())
        elif fill_method == 'drop':
            df_clean = df_clean.dropna()
        
        # Remove any remaining NaN values
        df_clean = df_clean.dropna()
        
        logger.info(f"Cleaned data shape: {df_clean.shape}")
        return df_clean


# Example usage and data file configuration
DEFAULT_DATA_FILES = {
    'lme': 'lme_prices.csv',
    'jpx': 'jpx_data.csv', 
    'us_stock': 'us_stock_data.csv',
    'forex': 'forex_rates.csv'
}
