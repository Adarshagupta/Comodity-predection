#!/usr/bin/env python
"""
Prediction script for commodity forecasting competition.

This script loads a trained model and generates predictions for new data.

Usage:
    python predict.py --model-path submissions/trained_model.pkl --data-path data/test/
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add src to path
sys.path.append('src')

from data_processing.data_loader import DataLoader
from feature_engineering.features import FeatureEngineer
from utils.submission import ModelPersistence, CompetitionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(args):
    """Main prediction pipeline."""
    logger.info("Starting prediction pipeline...")
    
    try:
        # Load trained model
        logger.info(f"Loading model from {args.model_path}")
        model, metadata = ModelPersistence.load_model(args.model_path)
        
        if model is None:
            raise ValueError("Failed to load model")
        
        logger.info("Model loaded successfully")
        
        # Initialize components
        data_loader = DataLoader(data_path=args.data_path)
        feature_engineer = FeatureEngineer()
        
        # Load test data
        logger.info("Loading test data...")
        
        if args.sample_data:
            # Create sample test data
            test_data = create_sample_test_data()
            aligned_data = data_loader.align_data_by_date(test_data)
        else:
            # Load real test data
            # This would depend on competition data format
            test_files = list(Path(args.data_path).glob("*.csv"))
            if not test_files:
                raise ValueError(f"No test files found in {args.data_path}")
            
            # For now, assume single test file
            test_data = pd.read_csv(test_files[0], index_col=0, parse_dates=True)
            aligned_data = test_data
        
        # Clean data
        clean_data = data_loader.clean_data(aligned_data, fill_method='forward')
        logger.info(f"Test data shape: {clean_data.shape}")
        
        # Engineer features using same configuration
        fe_config = metadata.get('feature_engineer_config', {})
        asset_pairs = fe_config.get('asset_pairs', [])
        market_prefixes = fe_config.get('market_prefixes', [])
        
        features = feature_engineer.build_feature_pipeline(
            clean_data,
            asset_pairs=asset_pairs,
            market_prefixes=market_prefixes
        )
        
        # Select same features used in training
        selected_features = metadata.get('selected_features', [])
        available_features = [f for f in selected_features if f in features.columns]
        
        if len(available_features) < len(selected_features) * 0.8:
            logger.warning(f"Only {len(available_features)}/{len(selected_features)} features available")
        
        X_test = features[available_features]
        
        # Handle missing features by filling with zeros
        for feature in selected_features:
            if feature not in X_test.columns:
                X_test[feature] = 0
        
        # Reorder columns to match training
        X_test = X_test[selected_features]
        
        logger.info(f"Test features shape: {X_test.shape}")
        
        # Make predictions
        logger.info("Generating predictions...")
        predictions = model.predict(X_test)
        
        # Create submission DataFrame
        submission = pd.DataFrame({
            'id': range(len(predictions)),
            'prediction': predictions
        })
        
        # Save predictions
        submission.to_csv(args.output_path, index=False)
        logger.info(f"Predictions saved to {args.output_path}")
        
        # Display summary statistics
        logger.info("Prediction Summary:")
        logger.info(f"  Number of predictions: {len(predictions)}")
        logger.info(f"  Mean prediction: {np.mean(predictions):.6f}")
        logger.info(f"  Std prediction: {np.std(predictions):.6f}")
        logger.info(f"  Min prediction: {np.min(predictions):.6f}")
        logger.info(f"  Max prediction: {np.max(predictions):.6f}")
        
        # Validate submission format
        from utils.submission import CompetitionPipeline
        pipeline = CompetitionPipeline(model, feature_engineer, data_loader)
        is_valid = pipeline.validate_submission(args.output_path)
        
        if is_valid:
            logger.info("✅ Submission validation passed")
        else:
            logger.error("❌ Submission validation failed")
        
        logger.info("Prediction pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


def create_sample_test_data():
    """Create sample test data for demonstration."""
    dates = pd.date_range(start='2025-01-01', end='2025-01-31', freq='D')
    n_days = len(dates)
    
    # LME data
    lme_data = pd.DataFrame({
        'date': dates,
        'copper_price': 8500 + np.cumsum(np.random.randn(n_days) * 30),
        'aluminum_price': 2100 + np.cumsum(np.random.randn(n_days) * 15),
        'zinc_price': 3200 + np.cumsum(np.random.randn(n_days) * 25),
    }).set_index('date')
    
    # JPX data
    jpx_data = pd.DataFrame({
        'date': dates,
        'nikkei_close': 28000 + np.cumsum(np.random.randn(n_days) * 150),
        'topix_close': 2000 + np.cumsum(np.random.randn(n_days) * 12),
    }).set_index('date')
    
    # US Stock data
    us_stock_data = pd.DataFrame({
        'date': dates,
        'sp500_close': 4800 + np.cumsum(np.random.randn(n_days) * 40),
        'nasdaq_close': 13000 + np.cumsum(np.random.randn(n_days) * 120),
    }).set_index('date')
    
    # Forex data
    forex_data = pd.DataFrame({
        'date': dates,
        'usd_jpy_rate': 145 + np.cumsum(np.random.randn(n_days) * 0.3),
        'eur_usd_rate': 1.1 + np.cumsum(np.random.randn(n_days) * 0.008),
    }).set_index('date')
    
    return {
        'lme': lme_data,
        'jpx': jpx_data,
        'us_stock': us_stock_data,
        'forex': forex_data
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions with trained model")
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model file"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/test/",
        help="Path to test data directory"
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        default="submissions/predictions.csv",
        help="Path to save predictions"
    )
    
    parser.add_argument(
        "--sample-data",
        action="store_true",
        help="Use sample test data for demonstration"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    
    main(args)
