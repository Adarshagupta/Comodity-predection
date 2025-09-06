#!/usr/bin/env python
"""
Main training script for commodity forecasting competition.

This script orchestrates the complete training pipeline:
1. Data loading and preprocessing
2. Feature engineering 
3. Model training and validation
4. Model persistence

Usage:
    python train.py --config configs/model_config.json --data-config configs/data_config.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add src to path
sys.path.append('src')

from data_processing.data_loader import DataLoader
from feature_engineering.features import FeatureEngineer
from models.ensemble_models import create_custom_ensemble, create_default_ensemble
from evaluation.metrics import ModelValidator, evaluate_model_performance
from utils.submission import ModelPersistence

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        return {}


def main(args):
    """Main training pipeline."""
    logger.info("Starting commodity forecasting model training...")
    
    # Load configurations
    model_config = load_config(args.config) if args.config else {}
    data_config = load_config(args.data_config) if args.data_config else {}
    
    try:
        # Initialize components
        data_loader = DataLoader(data_path=args.data_path)
        feature_engineer = FeatureEngineer()
        
        # Load data
        logger.info("Loading and preprocessing data...")
        
        if args.sample_data:
            # Create sample data for demonstration
            raw_data = create_sample_data()
        else:
            # Load real data files
            data_files = data_config.get('data_sources', {})
            file_paths = {market: info['file_path'] for market, info in data_files.items()}
            raw_data = data_loader.load_all_data(file_paths)
        
        # Align and clean data
        aligned_data = data_loader.align_data_by_date(raw_data)
        clean_data = data_loader.clean_data(
            aligned_data, 
            fill_method=data_config.get('data_preprocessing', {}).get('fill_method', 'forward'),
            max_missing_pct=data_config.get('data_preprocessing', {}).get('max_missing_pct', 0.1)
        )
        
        logger.info(f"Clean data shape: {clean_data.shape}")
        
        # Feature engineering
        logger.info("Engineering features...")
        
        fe_config = data_config.get('feature_engineering', {})
        asset_pairs = fe_config.get('asset_pairs', [])
        market_prefixes = fe_config.get('market_prefixes', [])
        
        features = feature_engineer.build_feature_pipeline(
            clean_data,
            asset_pairs=asset_pairs,
            market_prefixes=market_prefixes
        )
        
        logger.info(f"Feature matrix shape: {features.shape}")
        
        # Create target variable
        target_config = data_config.get('target_variables', {}).get('primary', {})
        target_asset = target_config.get('asset', 'lme_copper_price')
        target_horizon = target_config.get('horizon', 1)
        
        if target_asset in clean_data.columns:
            target = clean_data[target_asset].pct_change().shift(-target_horizon)
        else:
            # Fallback to first available price column
            price_cols = [col for col in clean_data.columns if 'price' in col.lower()]
            if price_cols:
                target = clean_data[price_cols[0]].pct_change().shift(-target_horizon)
            else:
                raise ValueError("No suitable target variable found")
        
        # Remove NaN values
        mask = ~(features.isnull().any(axis=1) | target.isnull())
        X = features[mask]
        y = target[mask]
        
        logger.info(f"Training data shape: {X.shape}, Target shape: {y.shape}")
        
        # Feature selection
        selection_config = model_config.get('feature_selection', {})
        max_features = selection_config.get('max_features', 100)
        
        selected_features = feature_engineer.select_features(
            X, y,
            method=selection_config.get('method', 'correlation'),
            max_features=max_features
        )
        
        X_selected = X[selected_features]
        logger.info(f"Selected {len(selected_features)} features")
        
        # Train/validation split
        split_ratio = args.train_split
        split_idx = int(len(X_selected) * split_ratio)
        
        X_train = X_selected.iloc[:split_idx]
        X_val = X_selected.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_val = y.iloc[split_idx:]
        
        logger.info(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
        
        # Model training
        logger.info("Training ensemble model...")
        
        if model_config.get('ensemble_models'):
            # Use custom configuration
            ensemble_model = create_custom_ensemble(model_config['ensemble_models'])
        else:
            # Use default ensemble
            ensemble_model = create_default_ensemble()
        
        # Train model
        ensemble_model.fit(X_train, y_train)
        
        # Evaluation
        logger.info("Evaluating model performance...")
        
        # Validation predictions
        y_pred_train = ensemble_model.predict(X_train)
        y_pred_val = ensemble_model.predict(X_val)
        
        # Calculate metrics
        train_metrics = evaluate_model_performance(y_pred_train, y_train.values)
        val_metrics = evaluate_model_performance(y_pred_val, y_val.values)
        
        logger.info("Training Performance:")
        for metric, value in train_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        logger.info("Validation Performance:")
        for metric, value in val_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Cross-validation (if requested)
        if args.cross_validate:
            logger.info("Performing time series cross-validation...")
            
            validation_config = model_config.get('validation', {})
            validator = ModelValidator(
                initial_train_size=validation_config.get('initial_train_size', 252),
                step_size=validation_config.get('step_size', 21),
                forecast_horizon=validation_config.get('forecast_horizon', 1)
            )
            
            # Use simpler model for CV to save time
            from models.ensemble_models import XGBoostModel
            cv_model = XGBoostModel(n_estimators=100)
            
            cv_results = validator.cross_validate_model(cv_model, X_selected, y)
            
            logger.info("Cross-Validation Results:")
            logger.info(f"  Mean Score: {cv_results['mean_score']:.4f} (±{cv_results['std_score']:.4f})")
            logger.info(f"  Overall Score: {cv_results['overall_score']:.4f}")
        
        # Save model
        logger.info("Saving trained model...")
        
        model_metadata = {
            'selected_features': selected_features,
            'feature_engineer_config': {
                'asset_pairs': asset_pairs,
                'market_prefixes': market_prefixes
            },
            'model_config': model_config,
            'data_config': data_config,
            'performance_metrics': {
                'training': train_metrics,
                'validation': val_metrics
            },
            'data_info': {
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'feature_count': len(selected_features),
                'target_variable': target_asset
            }
        }
        
        ModelPersistence.save_model(
            ensemble_model, 
            args.output_path, 
            model_metadata
        )
        
        logger.info(f"Model saved to {args.output_path}")
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def create_sample_data():
    """Create sample data for demonstration."""
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
    n_days = len(dates)
    
    # LME data
    lme_data = pd.DataFrame({
        'date': dates,
        'copper_price': 8000 + np.cumsum(np.random.randn(n_days) * 50),
        'aluminum_price': 2000 + np.cumsum(np.random.randn(n_days) * 20),
        'zinc_price': 3000 + np.cumsum(np.random.randn(n_days) * 30),
    }).set_index('date')
    
    # JPX data
    jpx_data = pd.DataFrame({
        'date': dates,
        'nikkei_close': 25000 + np.cumsum(np.random.randn(n_days) * 200),
        'topix_close': 1800 + np.cumsum(np.random.randn(n_days) * 15),
    }).set_index('date')
    
    # US Stock data
    us_stock_data = pd.DataFrame({
        'date': dates,
        'sp500_close': 4000 + np.cumsum(np.random.randn(n_days) * 50),
        'nasdaq_close': 12000 + np.cumsum(np.random.randn(n_days) * 150),
    }).set_index('date')
    
    # Forex data
    forex_data = pd.DataFrame({
        'date': dates,
        'usd_jpy_rate': 110 + np.cumsum(np.random.randn(n_days) * 0.5),
        'eur_usd_rate': 1.2 + np.cumsum(np.random.randn(n_days) * 0.01),
    }).set_index('date')
    
    return {
        'lme': lme_data,
        'jpx': jpx_data,
        'us_stock': us_stock_data,
        'forex': forex_data
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train commodity forecasting model")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/model_config.json",
        help="Path to model configuration file"
    )
    
    parser.add_argument(
        "--data-config",
        type=str,
        default="configs/data_config.json", 
        help="Path to data configuration file"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/",
        help="Path to data directory"
    )
    
    parser.add_argument(
        "--output-path",
        type=str, 
        default="submissions/trained_model.pkl",
        help="Path to save trained model"
    )
    
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of data to use for training"
    )
    
    parser.add_argument(
        "--cross-validate",
        action="store_true",
        help="Perform time series cross-validation"
    )
    
    parser.add_argument(
        "--sample-data",
        action="store_true",
        help="Use sample data for demonstration"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    
    main(args)
