"""
Feature engineering for commodity price forecasting.
Focuses on price-difference series and cross-market signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats
import ta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates features for commodity return prediction from multi-market data.
    """
    
    def __init__(self):
        self.feature_columns = []
        self.scaler = None
        
    def create_price_difference_features(self, df: pd.DataFrame, 
                                       asset_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Create price-difference series features between asset pairs.
        
        Args:
            df: Input DataFrame with asset prices
            asset_pairs: List of (asset1, asset2) tuples
            
        Returns:
            DataFrame with price difference features
        """
        features = df.copy()
        
        for asset1, asset2 in asset_pairs:
            if asset1 in df.columns and asset2 in df.columns:
                # Price difference
                diff_col = f"{asset1}_{asset2}_diff"
                features[diff_col] = df[asset1] - df[asset2]
                
                # Price ratio
                ratio_col = f"{asset1}_{asset2}_ratio"
                features[ratio_col] = df[asset1] / (df[asset2] + 1e-8)
                
                # Log price difference
                log_diff_col = f"{asset1}_{asset2}_log_diff"
                features[log_diff_col] = np.log(df[asset1] + 1) - np.log(df[asset2] + 1)
                
                # Rolling statistics of differences
                for window in [5, 10, 20]:
                    features[f"{diff_col}_ma_{window}"] = features[diff_col].rolling(window).mean()
                    features[f"{diff_col}_std_{window}"] = features[diff_col].rolling(window).std()
                    features[f"{ratio_col}_ma_{window}"] = features[ratio_col].rolling(window).mean()
                
        logger.info(f"Created price difference features. Shape: {features.shape}")
        return features
    
    def create_technical_indicators(self, df: pd.DataFrame, 
                                  price_columns: List[str]) -> pd.DataFrame:
        """
        Create technical analysis indicators.
        
        Args:
            df: Input DataFrame
            price_columns: List of price column names
            
        Returns:
            DataFrame with technical indicators
        """
        features = df.copy()
        
        for col in price_columns:
            if col not in df.columns:
                continue
                
            # Moving averages
            for window in [5, 10, 20, 50]:
                features[f"{col}_ma_{window}"] = df[col].rolling(window).mean()
                features[f"{col}_ema_{window}"] = df[col].ewm(span=window).mean()
                
            # RSI
            try:
                features[f"{col}_rsi"] = ta.momentum.RSIIndicator(df[col]).rsi()
            except:
                pass
                
            # Bollinger Bands
            try:
                bb = ta.volatility.BollingerBands(df[col])
                features[f"{col}_bb_upper"] = bb.bollinger_hband()
                features[f"{col}_bb_lower"] = bb.bollinger_lband()
                features[f"{col}_bb_width"] = bb.bollinger_wband()
            except:
                pass
                
            # MACD
            try:
                macd = ta.trend.MACD(df[col])
                features[f"{col}_macd"] = macd.macd()
                features[f"{col}_macd_signal"] = macd.macd_signal()
                features[f"{col}_macd_diff"] = macd.macd_diff()
            except:
                pass
        
        logger.info(f"Created technical indicators. Shape: {features.shape}")
        return features
    
    def create_volatility_features(self, df: pd.DataFrame, 
                                 return_columns: List[str]) -> pd.DataFrame:
        """
        Create volatility-based features.
        
        Args:
            df: Input DataFrame
            return_columns: List of return column names
            
        Returns:
            DataFrame with volatility features
        """
        features = df.copy()
        
        for col in return_columns:
            if col not in df.columns:
                continue
                
            # Rolling volatility
            for window in [5, 10, 20, 60]:
                features[f"{col}_vol_{window}"] = df[col].rolling(window).std()
                
            # Realized volatility (sum of squared returns)
            for window in [5, 10, 20]:
                features[f"{col}_realized_vol_{window}"] = np.sqrt(
                    (df[col] ** 2).rolling(window).sum()
                )
                
            # GARCH-like features
            features[f"{col}_abs_return"] = np.abs(df[col])
            features[f"{col}_squared_return"] = df[col] ** 2
            
            # Rolling skewness and kurtosis
            for window in [20, 60]:
                features[f"{col}_skew_{window}"] = df[col].rolling(window).skew()
                features[f"{col}_kurt_{window}"] = df[col].rolling(window).kurt()
        
        logger.info(f"Created volatility features. Shape: {features.shape}")
        return features
    
    def create_cross_market_features(self, df: pd.DataFrame, 
                                   market_prefixes: List[str]) -> pd.DataFrame:
        """
        Create cross-market relationship features.
        
        Args:
            df: Input DataFrame with multi-market data
            market_prefixes: List of market prefixes (e.g., ['lme', 'jpx', 'us_stock', 'forex'])
            
        Returns:
            DataFrame with cross-market features
        """
        features = df.copy()
        
        # Get return columns for each market
        market_returns = {}
        for prefix in market_prefixes:
            return_cols = [col for col in df.columns if col.startswith(prefix) and 'return' in col]
            if return_cols:
                market_returns[prefix] = return_cols
        
        # Create cross-market correlations
        for i, market1 in enumerate(market_prefixes):
            for market2 in market_prefixes[i+1:]:
                if market1 in market_returns and market2 in market_returns:
                    for col1 in market_returns[market1][:3]:  # Limit to avoid too many features
                        for col2 in market_returns[market2][:3]:
                            # Rolling correlation
                            for window in [10, 20, 60]:
                                corr_col = f"{col1}_{col2}_corr_{window}"
                                features[corr_col] = df[col1].rolling(window).corr(df[col2])
        
        # Market regime indicators
        for prefix in market_prefixes:
            if prefix in market_returns:
                # Average market return
                market_cols = market_returns[prefix]
                if len(market_cols) > 1:
                    features[f"{prefix}_avg_return"] = df[market_cols].mean(axis=1)
                    features[f"{prefix}_return_spread"] = df[market_cols].std(axis=1)
        
        logger.info(f"Created cross-market features. Shape: {features.shape}")
        return features
    
    def create_momentum_features(self, df: pd.DataFrame, 
                               price_columns: List[str]) -> pd.DataFrame:
        """
        Create momentum and trend features.
        
        Args:
            df: Input DataFrame
            price_columns: List of price column names
            
        Returns:
            DataFrame with momentum features
        """
        features = df.copy()
        
        for col in price_columns:
            if col not in df.columns:
                continue
                
            # Price momentum (rate of change)
            for period in [1, 3, 5, 10, 20]:
                features[f"{col}_roc_{period}"] = df[col].pct_change(period)
                
            # Price relative to moving averages
            for window in [10, 20, 50]:
                ma_col = f"{col}_ma_{window}"
                if ma_col in features.columns:
                    features[f"{col}_rel_ma_{window}"] = df[col] / features[ma_col] - 1
                    
            # Trend strength
            for window in [10, 20]:
                features[f"{col}_trend_{window}"] = df[col].rolling(window).apply(
                    lambda x: stats.linregress(range(len(x)), x)[0] if len(x) == window else np.nan
                )
        
        logger.info(f"Created momentum features. Shape: {features.shape}")
        return features
    
    def create_lag_features(self, df: pd.DataFrame, 
                          target_columns: List[str],
                          lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """
        Create lagged features.
        
        Args:
            df: Input DataFrame
            target_columns: Columns to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        features = df.copy()
        
        for col in target_columns:
            if col not in df.columns:
                continue
                
            for lag in lags:
                features[f"{col}_lag_{lag}"] = df[col].shift(lag)
        
        logger.info(f"Created lag features. Shape: {features.shape}")
        return features
    
    def create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based seasonal features.
        
        Args:
            df: Input DataFrame with datetime index
            
        Returns:
            DataFrame with seasonal features
        """
        features = df.copy()
        
        # Day of week
        features['day_of_week'] = df.index.dayofweek
        features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # Month
        features['month'] = df.index.month
        features['quarter'] = df.index.quarter
        
        # Day of month
        features['day_of_month'] = df.index.day
        features['is_month_end'] = (df.index == df.index.to_period('M').end_time).astype(int)
        features['is_month_start'] = (df.index.day == 1).astype(int)
        
        # Year
        features['year'] = df.index.year
        
        # Cyclical encoding for continuous time features
        features['day_of_week_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_of_week_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        
        logger.info(f"Created seasonal features. Shape: {features.shape}")
        return features
    
    def build_feature_pipeline(self, df: pd.DataFrame,
                             asset_pairs: List[Tuple[str, str]] = None,
                             market_prefixes: List[str] = None) -> pd.DataFrame:
        """
        Build complete feature engineering pipeline.
        
        Args:
            df: Input DataFrame
            asset_pairs: Asset pairs for price difference features
            market_prefixes: Market prefixes for cross-market features
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting feature engineering pipeline...")
        
        # Start with original data
        features = df.copy()
        
        # Get price and return columns
        price_columns = [col for col in df.columns if any(x in col.lower() 
                        for x in ['price', 'close', 'rate']) and 'return' not in col.lower()]
        return_columns = [col for col in df.columns if 'return' in col.lower()]
        
        # Create seasonal features
        features = self.create_seasonal_features(features)
        
        # Create technical indicators
        if price_columns:
            features = self.create_technical_indicators(features, price_columns)
        
        # Create volatility features
        if return_columns:
            features = self.create_volatility_features(features, return_columns)
        
        # Create momentum features
        if price_columns:
            features = self.create_momentum_features(features, price_columns)
        
        # Create lag features
        important_cols = price_columns + return_columns
        if important_cols:
            features = self.create_lag_features(features, important_cols[:10])  # Limit to avoid too many features
        
        # Create price difference features
        if asset_pairs:
            features = self.create_price_difference_features(features, asset_pairs)
        
        # Create cross-market features
        if market_prefixes:
            features = self.create_cross_market_features(features, market_prefixes)
        
        # Remove infinite and NaN values
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Store feature columns
        self.feature_columns = features.columns.tolist()
        
        logger.info(f"Feature engineering complete. Final shape: {features.shape}")
        logger.info(f"Total features created: {len(self.feature_columns)}")
        
        return features
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'correlation', 
                       max_features: int = 100) -> List[str]:
        """
        Select most important features.
        
        Args:
            X: Feature DataFrame
            y: Target series
            method: Feature selection method ('correlation', 'mutual_info', 'variance')
            max_features: Maximum number of features to select
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Selecting features using {method} method...")
        
        # Remove non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols].copy()
        
        # Remove features with too many NaN values
        valid_cols = X_numeric.columns[X_numeric.isnull().sum() < len(X_numeric) * 0.5].tolist()
        X_clean = X_numeric[valid_cols].fillna(0)
        
        if method == 'correlation':
            # Correlation with target
            correlations = abs(X_clean.corrwith(y)).sort_values(ascending=False)
            selected_features = correlations.head(max_features).index.tolist()
            
        elif method == 'variance':
            # Remove low variance features
            variances = X_clean.var().sort_values(ascending=False)
            selected_features = variances.head(max_features).index.tolist()
            
        else:
            # Default to correlation
            correlations = abs(X_clean.corrwith(y)).sort_values(ascending=False)
            selected_features = correlations.head(max_features).index.tolist()
        
        logger.info(f"Selected {len(selected_features)} features")
        return selected_features
