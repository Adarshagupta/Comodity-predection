"""
Competition-specific data loader for the Mitsui Commodity Prediction Challenge.
Handles the actual competition data format and structure.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompetitionDataLoader:
    """
    Handles loading and preprocessing of the actual competition data format.
    
    Dataset structure:
    - train.csv: Historical financial data with date_id and time series identifiers
    - train_labels.csv: Target variables (target_0 to target_423)
    - target_pairs.csv: Details of target calculations
    - test.csv: Test set with is_scored column
    """
    
    def __init__(self, data_path: str = "data/"):
        """
        Initialize competition data loader.
        
        Args:
            data_path: Path to competition data directory
        """
        self.data_path = Path(data_path)
        self.train_data = None
        self.train_labels = None
        self.target_pairs = None
        self.test_data = None
        
    def load_train_data(self) -> pd.DataFrame:
        """
        Load training data from train.csv.
        
        Returns:
            DataFrame with training data
        """
        try:
            train_file = self.data_path / "train.csv"
            self.train_data = pd.read_csv(train_file)
            
            # Convert date_id to datetime if needed
            if 'date_id' in self.train_data.columns:
                self.train_data['date_id'] = pd.to_datetime(self.train_data['date_id'])
                
            logger.info(f"Loaded training data: {self.train_data.shape}")
            logger.info(f"Date range: {self.train_data['date_id'].min()} to {self.train_data['date_id'].max()}")
            
            # Log time series identifiers
            time_series_cols = [col for col in self.train_data.columns if col != 'date_id']
            logger.info(f"Found {len(time_series_cols)} time series columns")
            
            # Group by market prefix
            market_groups = {}
            for col in time_series_cols:
                if col.startswith('LME_'):
                    market_groups.setdefault('LME', []).append(col)
                elif col.startswith('JPX_'):
                    market_groups.setdefault('JPX', []).append(col)
                elif col.startswith('US_'):
                    market_groups.setdefault('US', []).append(col)
                elif col.startswith('FX_'):
                    market_groups.setdefault('FX', []).append(col)
                    
            for market, cols in market_groups.items():
                logger.info(f"{market}: {len(cols)} columns")
                
            return self.train_data
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return pd.DataFrame()
    
    def load_train_labels(self) -> pd.DataFrame:
        """
        Load training labels from train_labels.csv.
        
        Returns:
            DataFrame with target variables
        """
        try:
            labels_file = self.data_path / "train_labels.csv"
            self.train_labels = pd.read_csv(labels_file)
            
            # Convert date_id to datetime if needed
            if 'date_id' in self.train_labels.columns:
                self.train_labels['date_id'] = pd.to_datetime(self.train_labels['date_id'])
                
            logger.info(f"Loaded training labels: {self.train_labels.shape}")
            
            # Count target columns
            target_cols = [col for col in self.train_labels.columns if col.startswith('target_')]
            logger.info(f"Found {len(target_cols)} target variables")
            
            return self.train_labels
            
        except Exception as e:
            logger.error(f"Error loading training labels: {e}")
            return pd.DataFrame()
    
    def load_target_pairs(self) -> pd.DataFrame:
        """
        Load target pairs metadata from target_pairs.csv.
        
        Returns:
            DataFrame with target pair details
        """
        try:
            pairs_file = self.data_path / "target_pairs.csv"
            self.target_pairs = pd.read_csv(pairs_file)
            
            logger.info(f"Loaded target pairs: {self.target_pairs.shape}")
            logger.info(f"Target details columns: {list(self.target_pairs.columns)}")
            
            # Log summary statistics
            if 'lag' in self.target_pairs.columns:
                logger.info(f"Lag distribution: \n{self.target_pairs['lag'].value_counts().sort_index()}")
                
            return self.target_pairs
            
        except Exception as e:
            logger.error(f"Error loading target pairs: {e}")
            return pd.DataFrame()
    
    def load_test_data(self) -> pd.DataFrame:
        """
        Load test data from test.csv.
        
        Returns:
            DataFrame with test data
        """
        try:
            test_file = self.data_path / "test.csv"
            self.test_data = pd.read_csv(test_file)
            
            # Convert date_id to datetime if needed
            if 'date_id' in self.test_data.columns:
                self.test_data['date_id'] = pd.to_datetime(self.test_data['date_id'])
                
            logger.info(f"Loaded test data: {self.test_data.shape}")
            
            # Check is_scored column
            if 'is_scored' in self.test_data.columns:
                scored_count = self.test_data['is_scored'].sum()
                logger.info(f"Scored rows: {scored_count}/{len(self.test_data)}")
                
            return self.test_data
            
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            return pd.DataFrame()
    
    def load_all_competition_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all competition data files.
        
        Returns:
            Dictionary with all loaded DataFrames
        """
        data = {}
        
        data['train'] = self.load_train_data()
        data['train_labels'] = self.load_train_labels()
        data['target_pairs'] = self.load_target_pairs()
        data['test'] = self.load_test_data()
        
        return data
    
    def merge_train_data_labels(self) -> pd.DataFrame:
        """
        Merge training data with labels on date_id.
        
        Returns:
            Merged DataFrame with features and targets
        """
        if self.train_data is None:
            self.load_train_data()
        if self.train_labels is None:
            self.load_train_labels()
            
        try:
            merged = self.train_data.merge(self.train_labels, on='date_id', how='inner')
            logger.info(f"Merged train data and labels: {merged.shape}")
            
            return merged
            
        except Exception as e:
            logger.error(f"Error merging train data and labels: {e}")
            return pd.DataFrame()
    
    def get_feature_columns(self) -> List[str]:
        """
        Get list of feature columns (time series identifiers).
        
        Returns:
            List of feature column names
        """
        if self.train_data is None:
            self.load_train_data()
            
        feature_cols = [col for col in self.train_data.columns if col != 'date_id']
        return feature_cols
    
    def get_target_columns(self) -> List[str]:
        """
        Get list of target columns.
        
        Returns:
            List of target column names
        """
        if self.train_labels is None:
            self.load_train_labels()
            
        target_cols = [col for col in self.train_labels.columns if col.startswith('target_')]
        return target_cols
    
    def get_market_groups(self) -> Dict[str, List[str]]:
        """
        Group features by market (LME, JPX, US, FX).
        
        Returns:
            Dictionary mapping market names to column lists
        """
        feature_cols = self.get_feature_columns()
        
        market_groups = {
            'LME': [col for col in feature_cols if col.startswith('LME_')],
            'JPX': [col for col in feature_cols if col.startswith('JPX_')],
            'US': [col for col in feature_cols if col.startswith('US_')],
            'FX': [col for col in feature_cols if col.startswith('FX_')]
        }
        
        return market_groups
    
    def create_time_series_features(self, df: pd.DataFrame, 
                                  window_sizes: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Create time series features from the raw data.
        
        Args:
            df: Input DataFrame with time series data
            window_sizes: Window sizes for rolling features
            
        Returns:
            DataFrame with engineered features
        """
        features_df = df.copy()
        
        # Get numeric columns (excluding date_id)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'date_id' in numeric_cols:
            numeric_cols.remove('date_id')
            
        for col in numeric_cols:
            # Returns
            features_df[f'{col}_return'] = df[col].pct_change()
            
            # Log returns
            features_df[f'{col}_log_return'] = np.log(df[col] / df[col].shift(1))
            
            # Moving averages and ratios
            for window in window_sizes:
                ma_col = f'{col}_ma_{window}'
                features_df[ma_col] = df[col].rolling(window).mean()
                features_df[f'{col}_ma_ratio_{window}'] = df[col] / features_df[ma_col]
                
            # Volatility
            for window in [5, 20]:
                features_df[f'{col}_vol_{window}'] = df[col].pct_change().rolling(window).std()
        
        logger.info(f"Created time series features. Shape: {features_df.shape}")
        return features_df
    
    def create_cross_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create cross-market relationship features.
        
        Args:
            df: Input DataFrame with market data
            
        Returns:
            DataFrame with cross-market features
        """
        features_df = df.copy()
        market_groups = self.get_market_groups()
        
        # Calculate average returns per market
        for market, cols in market_groups.items():
            available_cols = [col for col in cols if f'{col}_return' in features_df.columns]
            if len(available_cols) > 1:
                return_cols = [f'{col}_return' for col in available_cols]
                features_df[f'{market}_avg_return'] = features_df[return_cols].mean(axis=1)
                features_df[f'{market}_return_spread'] = features_df[return_cols].std(axis=1)
        
        # Cross-market correlations (rolling)
        markets = list(market_groups.keys())
        for i, market1 in enumerate(markets):
            for market2 in markets[i+1:]:
                if f'{market1}_avg_return' in features_df.columns and f'{market2}_avg_return' in features_df.columns:
                    features_df[f'{market1}_{market2}_corr_20'] = (
                        features_df[f'{market1}_avg_return']
                        .rolling(20)
                        .corr(features_df[f'{market2}_avg_return'])
                    )
        
        logger.info(f"Created cross-market features. Shape: {features_df.shape}")
        return features_df
    
    def prepare_competition_data(self, create_features: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for model training with competition format.
        
        Args:
            create_features: Whether to create engineered features
            
        Returns:
            Tuple of (features_df, targets_df)
        """
        # Load and merge data
        merged_df = self.merge_train_data_labels()
        
        if merged_df.empty:
            logger.error("Failed to load competition data")
            return pd.DataFrame(), pd.DataFrame()
        
        # Sort by date
        merged_df = merged_df.sort_values('date_id')
        
        # Separate features and targets
        feature_cols = self.get_feature_columns()
        target_cols = self.get_target_columns()
        
        # Base features
        features_df = merged_df[['date_id'] + feature_cols].copy()
        targets_df = merged_df[['date_id'] + target_cols].copy()
        
        # Create engineered features if requested
        if create_features:
            features_df = self.create_time_series_features(features_df)
            features_df = self.create_cross_market_features(features_df)
        
        # Set date_id as index
        features_df.set_index('date_id', inplace=True)
        targets_df.set_index('date_id', inplace=True)
        
        # Remove rows with all NaN targets
        valid_mask = ~targets_df.isnull().all(axis=1)
        features_df = features_df[valid_mask]
        targets_df = targets_df[valid_mask]
        
        logger.info(f"Final prepared data - Features: {features_df.shape}, Targets: {targets_df.shape}")
        
        return features_df, targets_df
    
    def get_target_info(self, target_name: str) -> Optional[Dict]:
        """
        Get information about a specific target from target_pairs.csv.
        
        Args:
            target_name: Name of the target (e.g., 'target_0')
            
        Returns:
            Dictionary with target information or None
        """
        if self.target_pairs is None:
            self.load_target_pairs()
            
        target_info = self.target_pairs[self.target_pairs['target'] == target_name]
        
        if target_info.empty:
            return None
            
        return target_info.iloc[0].to_dict()


# Example usage and utility functions
def download_competition_data():
    """
    Download competition data using Kaggle API.
    Note: Requires kaggle.json credentials file.
    """
    try:
        import kaggle
        kaggle.api.competition_download_files(
            'mitsui-commodity-prediction-challenge',
            path='data/',
            unzip=True
        )
        logger.info("Competition data downloaded successfully")
    except Exception as e:
        logger.error(f"Error downloading competition data: {e}")
        logger.info("Please download manually from Kaggle")


def explore_competition_data(data_path: str = "data/"):
    """
    Explore the structure of competition data.
    
    Args:
        data_path: Path to competition data
    """
    loader = CompetitionDataLoader(data_path)
    
    print("📊 COMPETITION DATA EXPLORATION")
    print("=" * 50)
    
    # Load all data
    data = loader.load_all_competition_data()
    
    # Training data exploration
    if not data['train'].empty:
        print(f"\n🔹 TRAINING DATA ({data['train'].shape[0]} rows, {data['train'].shape[1]} columns)")
        print(f"Date range: {data['train']['date_id'].min()} to {data['train']['date_id'].max()}")
        
        # Market breakdown
        market_groups = loader.get_market_groups()
        for market, cols in market_groups.items():
            print(f"  {market}: {len(cols)} time series")
    
    # Training labels exploration
    if not data['train_labels'].empty:
        print(f"\n🔹 TRAINING LABELS ({data['train_labels'].shape[0]} rows, {data['train_labels'].shape[1]} columns)")
        target_cols = loader.get_target_columns()
        print(f"  Targets: {len(target_cols)} (target_0 to target_{len(target_cols)-1})")
    
    # Target pairs exploration
    if not data['target_pairs'].empty:
        print(f"\n🔹 TARGET PAIRS ({data['target_pairs'].shape[0]} rows)")
        if 'lag' in data['target_pairs'].columns:
            print("  Lag distribution:")
            lag_counts = data['target_pairs']['lag'].value_counts().sort_index()
            for lag, count in lag_counts.items():
                print(f"    Lag {lag}: {count} targets")
    
    # Test data exploration
    if not data['test'].empty:
        print(f"\n🔹 TEST DATA ({data['test'].shape[0]} rows, {data['test'].shape[1]} columns)")
        if 'is_scored' in data['test'].columns:
            scored = data['test']['is_scored'].sum()
            print(f"  Scored rows: {scored}/{len(data['test'])}")
    
    print("\n✅ Data exploration complete!")
    return loader
