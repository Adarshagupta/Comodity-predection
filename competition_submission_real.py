#!/usr/bin/env python
"""
Real competition submission script for Mitsui Commodity Prediction Challenge.

This script integrates with the actual Kaggle evaluation API.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append('src')

from data_processing.competition_data_loader import CompetitionDataLoader
from feature_engineering.features import FeatureEngineer
from utils.submission import ModelPersistence

# Import Kaggle evaluation environment
try:
    import kaggle_evaluation
    KAGGLE_ENV_AVAILABLE = True
except ImportError:
    KAGGLE_ENV_AVAILABLE = False
    print("Warning: Kaggle evaluation environment not available. Running in simulation mode.")


class CompetitionSubmission:
    """
    Handles the competition submission workflow.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize competition submission.
        
        Args:
            model_path: Path to trained model file
        """
        self.model_path = model_path
        self.model = None
        self.metadata = None
        self.data_loader = CompetitionDataLoader()
        
        # Load trained model
        self.load_model()
        
    def load_model(self):
        """Load the trained model and metadata."""
        try:
            self.model, self.metadata = ModelPersistence.load_model(self.model_path)
            print(f"✅ Model loaded from {self.model_path}")
            
            if self.metadata:
                print(f"Model type: {self.metadata.get('model_type', 'Unknown')}")
                print(f"Target count: {self.metadata.get('data_info', {}).get('target_count', 'Unknown')}")
                print(f"Feature count: {self.metadata.get('data_info', {}).get('feature_count', 'Unknown')}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess features to match training format.
        
        Args:
            df: Raw feature DataFrame
            
        Returns:
            Preprocessed feature DataFrame
        """
        try:
            # Create time series features to match training
            processed_df = self.data_loader.create_time_series_features(df)
            processed_df = self.data_loader.create_cross_market_features(processed_df)
            
            # Select features used in training
            if self.metadata and 'feature_columns' in self.metadata:
                feature_columns = self.metadata['feature_columns']
                
                # Keep only features that exist and add missing ones as zeros
                available_features = [col for col in feature_columns if col in processed_df.columns]
                processed_df = processed_df[available_features]
                
                # Add missing features as zeros
                for col in feature_columns:
                    if col not in processed_df.columns:
                        processed_df[col] = 0.0
                
                # Reorder to match training
                processed_df = processed_df[feature_columns]
            
            # Handle missing values
            processed_df = processed_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            return processed_df
            
        except Exception as e:
            print(f"❌ Error preprocessing features: {e}")
            return df
    
    def make_predictions(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for all targets.
        
        Args:
            features_df: Feature DataFrame
            
        Returns:
            DataFrame with predictions for each target
        """
        try:
            # Preprocess features
            X = self.preprocess_features(features_df)
            
            # Generate predictions
            predictions = self.model.predict(X)
            
            # Ensure all target columns are present
            if self.metadata and 'target_columns' in self.metadata:
                target_columns = self.metadata['target_columns']
                
                for col in target_columns:
                    if col not in predictions.columns:
                        predictions[col] = 0.0
                
                predictions = predictions[target_columns]
            
            return predictions
            
        except Exception as e:
            print(f"❌ Error making predictions: {e}")
            # Return zeros as fallback
            target_columns = self.metadata.get('target_columns', [f'target_{i}' for i in range(424)])
            return pd.DataFrame(0.0, index=features_df.index, columns=target_columns)
    
    def run_kaggle_submission(self):
        """
        Run the actual Kaggle submission using the evaluation API.
        """
        if not KAGGLE_ENV_AVAILABLE:
            print("❌ Kaggle evaluation environment not available")
            return
            
        print("🚀 Starting Kaggle competition submission...")
        
        # Initialize Kaggle environment
        env = kaggle_evaluation.make_env()
        
        # Main prediction loop
        for iteration, (test_df, sample_prediction_df) in enumerate(env.iter_test()):
            try:
                print(f"📊 Iteration {iteration + 1}: Processing {len(test_df)} rows")
                
                # Generate predictions
                predictions_df = self.make_predictions(test_df)
                
                # Format for submission (only scored rows)
                if 'is_scored' in test_df.columns:
                    scored_mask = test_df['is_scored']
                    scored_predictions = predictions_df[scored_mask]
                    
                    print(f"  Scored rows: {scored_mask.sum()}/{len(test_df)}")
                else:
                    scored_predictions = predictions_df
                
                # Update sample prediction with our predictions
                submission_df = sample_prediction_df.copy()
                
                # Match predictions to submission format
                for col in scored_predictions.columns:
                    if col in submission_df.columns:
                        submission_df.loc[scored_predictions.index, col] = scored_predictions[col]
                
                # Submit predictions
                env.predict(submission_df)
                
                print(f"  ✅ Iteration {iteration + 1} completed")
                
            except Exception as e:
                print(f"  ❌ Error in iteration {iteration + 1}: {e}")
                # Submit zeros as fallback
                fallback_df = sample_prediction_df.copy()
                fallback_df.iloc[:] = 0.0
                env.predict(fallback_df)
        
        print("🏁 Kaggle submission completed!")
    
    def simulate_submission(self, test_data_path: str = None):
        """
        Simulate the submission process for testing.
        
        Args:
            test_data_path: Path to test data file
        """
        print("🎮 Running submission simulation...")
        
        if test_data_path and Path(test_data_path).exists():
            # Load test data
            test_df = pd.read_csv(test_data_path)
            if 'date_id' in test_df.columns:
                test_df['date_id'] = pd.to_datetime(test_df['date_id'])
                test_df.set_index('date_id', inplace=True)
        else:
            # Create dummy test data
            print("Creating dummy test data for simulation...")
            dates = pd.date_range(start='2025-01-01', periods=90, freq='D')
            
            # Simulate test data structure
            test_df = pd.DataFrame(index=dates)
            
            # Add dummy market data
            market_groups = self.data_loader.get_market_groups()
            for market, cols in market_groups.items():
                for col in cols[:3]:  # Sample few columns per market
                    test_df[col] = 100 + np.cumsum(np.random.randn(len(dates)) * 0.1)
            
            test_df['is_scored'] = True
            test_df.iloc[-30:, test_df.columns.get_loc('is_scored')] = False  # Last 30 days not scored
        
        # Generate predictions
        print(f"Generating predictions for {len(test_df)} rows...")
        predictions_df = self.make_predictions(test_df)
        
        # Create submission file
        submission_path = "submissions/competition_predictions.csv"
        
        # Format submission
        submission_df = pd.DataFrame()
        submission_df['date_id'] = test_df.index
        
        # Add target predictions
        for col in predictions_df.columns:
            submission_df[col] = predictions_df[col].values
        
        # Save submission
        submission_df.to_csv(submission_path, index=False)
        
        print(f"✅ Simulation completed. Predictions saved to {submission_path}")
        print(f"Prediction summary:")
        print(f"  Shape: {predictions_df.shape}")
        print(f"  Mean prediction: {predictions_df.mean().mean():.6f}")
        print(f"  Std prediction: {predictions_df.std().mean():.6f}")
        
        return submission_df


def main():
    """Main submission function."""
    
    # Configuration
    model_path = "submissions/competition_model.pkl"
    test_data_path = "data/test.csv"
    
    try:
        # Initialize submission
        submission = CompetitionSubmission(model_path)
        
        # Run submission
        if KAGGLE_ENV_AVAILABLE:
            # Real Kaggle submission
            submission.run_kaggle_submission()
        else:
            # Simulation mode
            submission.simulate_submission(test_data_path)
        
    except Exception as e:
        print(f"❌ Submission failed: {e}")
        raise


if __name__ == "__main__":
    main()
