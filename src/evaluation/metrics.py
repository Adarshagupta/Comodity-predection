"""
Evaluation metrics for commodity forecasting competition.
Implements the Sharpe ratio variant used in the competition.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompetitionMetrics:
    """
    Implements competition evaluation metrics including the Sharpe ratio variant.
    """
    
    @staticmethod
    def spearman_correlation(predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Calculate Spearman rank correlation between predictions and targets.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Spearman correlation coefficient
        """
        # Remove NaN values
        mask = ~(np.isnan(predictions) | np.isnan(targets))
        if mask.sum() < 2:
            return 0.0
            
        pred_clean = predictions[mask]
        target_clean = targets[mask]
        
        # Calculate Spearman correlation
        correlation, _ = stats.spearmanr(pred_clean, target_clean)
        
        return correlation if not np.isnan(correlation) else 0.0
    
    @staticmethod
    def sharpe_ratio_variant(predictions: np.ndarray, targets: np.ndarray, 
                           window_size: Optional[int] = None) -> float:
        """
        Calculate the competition's Sharpe ratio variant metric.
        
        The metric is computed as:
        mean(Spearman correlations) / std(Spearman correlations)
        
        Args:
            predictions: Model predictions array
            targets: Ground truth targets array
            window_size: Window size for rolling correlations (if None, uses single correlation)
            
        Returns:
            Sharpe ratio variant score
        """
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have the same length")
        
        if window_size is None or window_size >= len(predictions):
            # Single correlation calculation
            correlation = CompetitionMetrics.spearman_correlation(predictions, targets)
            return correlation  # Single value, std = 0, so return correlation itself
        
        # Rolling correlations
        correlations = []
        
        for i in range(window_size, len(predictions) + 1):
            window_pred = predictions[i-window_size:i]
            window_target = targets[i-window_size:i]
            
            corr = CompetitionMetrics.spearman_correlation(window_pred, window_target)
            correlations.append(corr)
        
        correlations = np.array(correlations)
        
        # Remove NaN correlations
        valid_correlations = correlations[~np.isnan(correlations)]
        
        if len(valid_correlations) == 0:
            return 0.0
        
        mean_corr = np.mean(valid_correlations)
        std_corr = np.std(valid_correlations)
        
        # Avoid division by zero
        if std_corr == 0:
            return mean_corr
        
        sharpe_variant = mean_corr / std_corr
        
        logger.info(f"Sharpe ratio variant: {sharpe_variant:.4f} "
                   f"(mean: {mean_corr:.4f}, std: {std_corr:.4f})")
        
        return sharpe_variant
    
    @staticmethod
    def multiple_asset_sharpe_variant(predictions_dict: dict, targets_dict: dict) -> float:
        """
        Calculate Sharpe ratio variant for multiple assets.
        
        Args:
            predictions_dict: Dictionary of {asset_name: predictions_array}
            targets_dict: Dictionary of {asset_name: targets_array}
            
        Returns:
            Average Sharpe ratio variant across all assets
        """
        asset_scores = []
        
        for asset in predictions_dict.keys():
            if asset in targets_dict:
                score = CompetitionMetrics.sharpe_ratio_variant(
                    predictions_dict[asset], 
                    targets_dict[asset]
                )
                asset_scores.append(score)
                logger.info(f"Asset {asset} Sharpe variant: {score:.4f}")
        
        if not asset_scores:
            return 0.0
        
        final_score = np.mean(asset_scores)
        logger.info(f"Overall Sharpe variant score: {final_score:.4f}")
        
        return final_score
    
    @staticmethod
    def information_ratio(predictions: np.ndarray, targets: np.ndarray, 
                         benchmark: Optional[np.ndarray] = None) -> float:
        """
        Calculate Information Ratio (similar to Sharpe but relative to benchmark).
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            benchmark: Benchmark predictions (if None, uses zero)
            
        Returns:
            Information ratio
        """
        if benchmark is None:
            benchmark = np.zeros_like(predictions)
        
        # Calculate excess returns
        excess_pred = predictions - benchmark
        
        # Calculate tracking error (std of excess returns)
        correlation = CompetitionMetrics.spearman_correlation(excess_pred, targets)
        
        return correlation
    
    @staticmethod
    def max_drawdown(returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown of a return series.
        
        Args:
            returns: Array of returns
            
        Returns:
            Maximum drawdown value
        """
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return np.min(drawdown)
    
    @staticmethod
    def calmar_ratio(returns: np.ndarray) -> float:
        """
        Calculate Calmar ratio (annual return / max drawdown).
        
        Args:
            returns: Array of returns
            
        Returns:
            Calmar ratio
        """
        annual_return = np.mean(returns) * 252  # Assuming daily returns
        max_dd = CompetitionMetrics.max_drawdown(returns)
        
        if max_dd == 0:
            return np.inf if annual_return > 0 else 0
        
        return annual_return / abs(max_dd)


class ModelValidator:
    """
    Validates model performance using time series cross-validation.
    """
    
    def __init__(self, initial_train_size: int = 252, 
                 step_size: int = 21, 
                 forecast_horizon: int = 1):
        """
        Initialize validator.
        
        Args:
            initial_train_size: Initial training window size (e.g., 252 for 1 year)
            step_size: Step size for rolling window (e.g., 21 for monthly)
            forecast_horizon: Number of periods to forecast
        """
        self.initial_train_size = initial_train_size
        self.step_size = step_size
        self.forecast_horizon = forecast_horizon
    
    def time_series_split(self, data_length: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate time series train/test splits.
        
        Args:
            data_length: Length of the dataset
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        splits = []
        
        for start in range(0, data_length - self.initial_train_size - self.forecast_horizon + 1, 
                          self.step_size):
            
            train_end = start + self.initial_train_size
            test_start = train_end
            test_end = min(test_start + self.forecast_horizon, data_length)
            
            if test_end > test_start:
                train_indices = np.arange(start, train_end)
                test_indices = np.arange(test_start, test_end)
                splits.append((train_indices, test_indices))
        
        logger.info(f"Generated {len(splits)} time series splits")
        return splits
    
    def cross_validate_model(self, model, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Perform time series cross-validation.
        
        Args:
            model: Model with fit/predict methods
            X: Feature DataFrame
            y: Target series
            
        Returns:
            Dictionary with validation results
        """
        splits = self.time_series_split(len(X))
        
        scores = []
        predictions_all = []
        targets_all = []
        
        for i, (train_idx, test_idx) in enumerate(splits):
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Remove NaN values
            train_mask = ~(X_train.isnull().any(axis=1) | y_train.isnull())
            X_train_clean = X_train[train_mask]
            y_train_clean = y_train[train_mask]
            
            test_mask = ~(X_test.isnull().any(axis=1) | y_test.isnull())
            X_test_clean = X_test[test_mask]
            y_test_clean = y_test[test_mask]
            
            if len(X_train_clean) == 0 or len(X_test_clean) == 0:
                continue
            
            try:
                # Train model
                model.fit(X_train_clean, y_train_clean)
                
                # Make predictions
                y_pred = model.predict(X_test_clean)
                
                # Calculate score
                score = CompetitionMetrics.sharpe_ratio_variant(y_pred, y_test_clean.values)
                scores.append(score)
                
                predictions_all.extend(y_pred)
                targets_all.extend(y_test_clean.values)
                
                logger.info(f"Fold {i+1}: Score = {score:.4f}")
                
            except Exception as e:
                logger.warning(f"Error in fold {i+1}: {e}")
                continue
        
        # Calculate overall metrics
        overall_score = CompetitionMetrics.sharpe_ratio_variant(
            np.array(predictions_all), 
            np.array(targets_all)
        )
        
        results = {
            'scores': scores,
            'mean_score': np.mean(scores) if scores else 0,
            'std_score': np.std(scores) if scores else 0,
            'overall_score': overall_score,
            'n_splits': len(scores)
        }
        
        logger.info(f"Cross-validation complete: Mean score = {results['mean_score']:.4f} "
                   f"(±{results['std_score']:.4f})")
        
        return results


# Example usage functions
def evaluate_model_performance(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """
    Comprehensive model evaluation.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        
    Returns:
        Dictionary with various performance metrics
    """
    metrics = {}
    
    # Competition metric
    metrics['sharpe_variant'] = CompetitionMetrics.sharpe_ratio_variant(predictions, targets)
    
    # Basic correlation
    metrics['spearman_corr'] = CompetitionMetrics.spearman_correlation(predictions, targets)
    metrics['pearson_corr'] = np.corrcoef(predictions, targets)[0, 1] if len(predictions) > 1 else 0
    
    # Error metrics
    metrics['mae'] = np.mean(np.abs(predictions - targets))
    metrics['rmse'] = np.sqrt(np.mean((predictions - targets) ** 2))
    
    # Directional accuracy
    pred_direction = np.sign(predictions)
    target_direction = np.sign(targets)
    metrics['directional_accuracy'] = np.mean(pred_direction == target_direction)
    
    return metrics
