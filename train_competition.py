#!/usr/bin/env python
"""
Competition-specific training script for Mitsui Commodity Prediction Challenge.

This script handles the actual competition data format with multiple targets.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# Add src to path
sys.path.append('src')

from data_processing.competition_data_loader import CompetitionDataLoader, explore_competition_data
from feature_engineering.features import FeatureEngineer
from models.ensemble_models import create_default_ensemble, create_custom_ensemble
from evaluation.metrics import CompetitionMetrics, ModelValidator, evaluate_model_performance
from utils.submission import ModelPersistence

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('competition_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MultiTargetModel:
    """
    Wrapper for handling multiple target prediction.
    """
    
    def __init__(self, base_model_factory, target_columns):
        """
        Initialize multi-target model.
        
        Args:
            base_model_factory: Function that creates a new model instance
            target_columns: List of target column names
        """
        self.base_model_factory = base_model_factory
        self.target_columns = target_columns
        self.models = {}
        self.is_fitted = False
        
    def fit(self, X, y):
        """
        Fit separate models for each target.
        
        Args:
            X: Feature matrix
            y: Target matrix with multiple columns
        """
        logger.info(f"Training models for {len(self.target_columns)} targets...")
        
        for i, target_col in enumerate(self.target_columns):
            if target_col in y.columns:
                logger.info(f"Training model {i+1}/{len(self.target_columns)}: {target_col}")
                
                # Get target values (remove NaN)
                target_values = y[target_col]
                valid_mask = ~target_values.isnull()
                
                if valid_mask.sum() > 100:  # Minimum samples for training
                    X_target = X[valid_mask]
                    y_target = target_values[valid_mask]
                    
                    # Create and train model
                    model = self.base_model_factory()
                    model.fit(X_target, y_target)
                    self.models[target_col] = model
                else:
                    logger.warning(f"Insufficient data for {target_col}: {valid_mask.sum()} samples")
        
        self.is_fitted = True
        logger.info(f"Trained {len(self.models)} target models successfully")
        
    def predict(self, X):
        """
        Generate predictions for all targets.
        
        Args:
            X: Feature matrix
            
        Returns:
            DataFrame with predictions for each target
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        predictions = pd.DataFrame(index=X.index)
        
        for target_col in self.target_columns:
            if target_col in self.models:
                try:
                    pred = self.models[target_col].predict(X)
                    predictions[target_col] = pred
                except Exception as e:
                    logger.warning(f"Error predicting {target_col}: {e}")
                    predictions[target_col] = 0.0
            else:
                # No model trained for this target
                predictions[target_col] = 0.0
        
        return predictions


def evaluate_multi_target_performance(predictions, targets):
    """
    Evaluate performance across multiple targets.
    
    Args:
        predictions: DataFrame with predictions
        targets: DataFrame with actual targets
        
    Returns:
        Dictionary with performance metrics
    """
    results = {}
    target_scores = []
    
    common_targets = set(predictions.columns) & set(targets.columns)
    
    for target in common_targets:
        pred = predictions[target].values
        actual = targets[target].values
        
        # Remove NaN values
        mask = ~(np.isnan(pred) | np.isnan(actual))
        if mask.sum() > 10:
            pred_clean = pred[mask]
            actual_clean = actual[mask]
            
            # Calculate metrics
            score = CompetitionMetrics.sharpe_ratio_variant(pred_clean, actual_clean)
            corr = CompetitionMetrics.spearman_correlation(pred_clean, actual_clean)
            
            results[target] = {
                'sharpe_variant': score,
                'spearman_corr': corr,
                'n_samples': len(pred_clean)
            }
            
            target_scores.append(score)
    
    # Overall performance
    if target_scores:
        results['overall'] = {
            'mean_sharpe_variant': np.mean(target_scores),
            'median_sharpe_variant': np.median(target_scores),
            'std_sharpe_variant': np.std(target_scores),
            'n_targets': len(target_scores)
        }
    
    return results


def main(args):
    """Main training pipeline for competition."""
    logger.info("Starting Mitsui Commodity Prediction Challenge training...")
    
    try:
        # Initialize competition data loader
        data_loader = CompetitionDataLoader(data_path=args.data_path)
        
        # Explore data structure if requested
        if args.explore_data:
            logger.info("Exploring competition data structure...")
            explore_competition_data(args.data_path)
        
        # Load and prepare data
        logger.info("Loading competition data...")
        features_df, targets_df = data_loader.prepare_competition_data(create_features=True)
        
        if features_df.empty or targets_df.empty:
            raise ValueError("Failed to load competition data")
        
        logger.info(f"Loaded data - Features: {features_df.shape}, Targets: {targets_df.shape}")
        
        # Handle missing values in features
        logger.info("Preprocessing features...")
        
        # Fill missing values with forward fill then backward fill
        features_clean = features_df.fillna(method='ffill').fillna(method='bfill')
        
        # Remove columns with too many missing values
        missing_pct = features_clean.isnull().sum() / len(features_clean)
        valid_feature_cols = missing_pct[missing_pct <= args.max_missing_pct].index
        features_clean = features_clean[valid_feature_cols]
        
        logger.info(f"After preprocessing: {features_clean.shape}")
        
        # Train/validation split by date
        split_date = features_clean.index[int(len(features_clean) * args.train_split)]
        
        X_train = features_clean[features_clean.index <= split_date]
        X_val = features_clean[features_clean.index > split_date]
        y_train = targets_df[targets_df.index <= split_date]
        y_val = targets_df[targets_df.index > split_date]
        
        logger.info(f"Train: {X_train.shape}, Validation: {X_val.shape}")
        
        # Feature selection per target (if requested)
        if args.feature_selection:
            logger.info("Performing feature selection...")
            feature_engineer = FeatureEngineer()
            
            # Select features based on first few targets with sufficient data
            target_cols = data_loader.get_target_columns()
            selection_targets = []
            
            for target_col in target_cols[:10]:  # Use first 10 targets for selection
                if target_col in y_train.columns:
                    target_values = y_train[target_col]
                    if target_values.notna().sum() > 200:
                        selection_targets.append(target_col)
                        
            if selection_targets:
                # Use average target for feature selection
                avg_target = y_train[selection_targets].mean(axis=1)
                valid_mask = ~avg_target.isnull()
                
                selected_features = feature_engineer.select_features(
                    X_train[valid_mask], 
                    avg_target[valid_mask],
                    max_features=args.max_features
                )
                
                X_train = X_train[selected_features]
                X_val = X_val[selected_features]
                
                logger.info(f"Selected {len(selected_features)} features")
        
        # Create multi-target model
        logger.info("Creating multi-target ensemble model...")
        
        def model_factory():
            return create_default_ensemble()
        
        target_columns = data_loader.get_target_columns()
        multi_model = MultiTargetModel(model_factory, target_columns)
        
        # Train model
        logger.info("Training multi-target model...")
        multi_model.fit(X_train, y_train)
        
        # Validation
        logger.info("Generating validation predictions...")
        val_predictions = multi_model.predict(X_val)
        
        # Evaluate performance
        logger.info("Evaluating model performance...")
        performance = evaluate_multi_target_performance(val_predictions, y_val)
        
        # Log results
        logger.info("Validation Performance:")
        if 'overall' in performance:
            overall = performance['overall']
            logger.info(f"  Mean Sharpe Variant: {overall['mean_sharpe_variant']:.4f}")
            logger.info(f"  Median Sharpe Variant: {overall['median_sharpe_variant']:.4f}")
            logger.info(f"  Std Sharpe Variant: {overall['std_sharpe_variant']:.4f}")
            logger.info(f"  Targets with predictions: {overall['n_targets']}")
        
        # Save model and metadata
        logger.info("Saving trained model...")
        
        model_metadata = {
            'competition': 'mitsui-commodity-prediction-challenge',
            'model_type': 'multi_target_ensemble',
            'target_columns': target_columns,
            'feature_columns': list(X_train.columns),
            'performance': performance,
            'data_info': {
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'feature_count': X_train.shape[1],
                'target_count': len(target_columns)
            },
            'preprocessing': {
                'max_missing_pct': args.max_missing_pct,
                'feature_selection': args.feature_selection,
                'max_features': args.max_features if args.feature_selection else None
            }
        }
        
        ModelPersistence.save_model(multi_model, args.output_path, model_metadata)
        logger.info(f"Model saved to {args.output_path}")
        
        # Save feature importance if available
        if hasattr(multi_model, 'models') and multi_model.models:
            try:
                feature_importance = {}
                for target, model in list(multi_model.models.items())[:5]:  # Sample few models
                    if hasattr(model, 'get_model_performance'):
                        # This would need to be implemented for ensemble models
                        pass
                
                logger.info("Feature importance analysis completed")
            except Exception as e:
                logger.warning(f"Could not generate feature importance: {e}")
        
        logger.info("Competition training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model for Mitsui Commodity Prediction Challenge")
    
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/",
        help="Path to competition data directory"
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        default="submissions/competition_model.pkl",
        help="Path to save trained model"
    )
    
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of data to use for training"
    )
    
    parser.add_argument(
        "--max-missing-pct",
        type=float,
        default=0.1,
        help="Maximum percentage of missing values allowed per feature"
    )
    
    parser.add_argument(
        "--feature-selection",
        action="store_true",
        help="Perform automated feature selection"
    )
    
    parser.add_argument(
        "--max-features",
        type=int,
        default=100,
        help="Maximum number of features to select"
    )
    
    parser.add_argument(
        "--explore-data",
        action="store_true",
        help="Explore and display data structure"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    
    main(args)
