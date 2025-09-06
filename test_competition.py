#!/usr/bin/env python
"""
Test script for competition-specific functionality.
Tests the Mitsui Commodity Prediction Challenge components.
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def test_competition_data_loader():
    """Test the competition data loader with simulated data."""
    print("🧪 Testing Competition Data Loader...")
    
    try:
        from data_processing.competition_data_loader import CompetitionDataLoader
        
        # Create temporary sample files
        Path('temp_data').mkdir(exist_ok=True)
        
        # Create sample train.csv
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        train_data = pd.DataFrame({
            'date_id': dates,
            'LME_COPPER_CLOSE': 8000 + np.cumsum(np.random.randn(100) * 50),
            'LME_ALUMINUM_CLOSE': 2000 + np.cumsum(np.random.randn(100) * 20),
            'JPX_NIKKEI_CLOSE': 27000 + np.cumsum(np.random.randn(100) * 200),
            'US_SPX_CLOSE': 4200 + np.cumsum(np.random.randn(100) * 50),
            'FX_USDJPY_CLOSE': 130 + np.cumsum(np.random.randn(100) * 0.5)
        })
        train_data.to_csv('temp_data/train.csv', index=False)
        
        # Create sample train_labels.csv
        train_labels = pd.DataFrame({
            'date_id': dates,
            'target_0': np.random.randn(100) * 0.01,
            'target_1': np.random.randn(100) * 0.01,
            'target_2': np.random.randn(100) * 0.01
        })
        train_labels.to_csv('temp_data/train_labels.csv', index=False)
        
        # Create sample target_pairs.csv
        target_pairs = pd.DataFrame({
            'target': ['target_0', 'target_1', 'target_2'],
            'lag': [1, 1, 1],
            'pair': ['LME_COPPER_CLOSE', 'LME_ALUMINUM_CLOSE', 'LME_COPPER_CLOSE-LME_ALUMINUM_CLOSE']
        })
        target_pairs.to_csv('temp_data/target_pairs.csv', index=False)
        
        # Test data loader
        loader = CompetitionDataLoader('temp_data/')
        
        # Test loading functions
        train_df = loader.load_train_data()
        labels_df = loader.load_train_labels()
        pairs_df = loader.load_target_pairs()
        
        # Test data preparation
        features_df, targets_df = loader.prepare_competition_data()
        
        print(f"  ✅ Train data: {train_df.shape}")
        print(f"  ✅ Labels data: {labels_df.shape}")
        print(f"  ✅ Target pairs: {pairs_df.shape}")
        print(f"  ✅ Prepared features: {features_df.shape}")
        print(f"  ✅ Prepared targets: {targets_df.shape}")
        
        # Cleanup
        import shutil
        shutil.rmtree('temp_data')
        
        return True
        
    except Exception as e:
        print(f"  ❌ Competition data loader test failed: {e}")
        return False

def test_multi_target_model():
    """Test multi-target model functionality."""
    print("\n🧪 Testing Multi-Target Model...")
    
    try:
        # Create sample data
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(200, 10), 
                        columns=[f'feature_{i}' for i in range(10)])
        
        # Multiple targets with some missing values
        y = pd.DataFrame({
            'target_0': np.random.randn(200) * 0.01,
            'target_1': np.random.randn(200) * 0.01,
            'target_2': np.random.randn(200) * 0.01
        })
        
        # Add some missing values
        y.loc[50:60, 'target_1'] = np.nan
        y.loc[150:160, 'target_2'] = np.nan
        
        # Import and test multi-target model
        from train_competition import MultiTargetModel
        from models.ensemble_models import create_default_ensemble
        
        def model_factory():
            from sklearn.linear_model import Ridge
            return Ridge(alpha=1.0)  # Use simple model for testing
        
        # Create multi-target model
        target_columns = ['target_0', 'target_1', 'target_2']
        multi_model = MultiTargetModel(model_factory, target_columns)
        
        # Train
        multi_model.fit(X[:150], y[:150])
        
        # Predict
        predictions = multi_model.predict(X[150:])
        
        print(f"  ✅ Multi-target model trained: {len(multi_model.models)} targets")
        print(f"  ✅ Predictions shape: {predictions.shape}")
        
        # Test evaluation
        from train_competition import evaluate_multi_target_performance
        performance = evaluate_multi_target_performance(predictions, y[150:])
        
        print(f"  ✅ Performance evaluation: {len(performance)} targets evaluated")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Multi-target model test failed: {e}")
        return False

def test_competition_submission():
    """Test competition submission functionality."""
    print("\n🧪 Testing Competition Submission...")
    
    try:
        # Create a dummy model for testing
        from sklearn.linear_model import Ridge
        from utils.submission import ModelPersistence
        
        # Create and save a dummy model
        dummy_model = Ridge(alpha=1.0)
        X_dummy = np.random.randn(100, 10)
        y_dummy = np.random.randn(100)
        dummy_model.fit(X_dummy, y_dummy)
        
        # Create metadata
        metadata = {
            'competition': 'mitsui-commodity-prediction-challenge',
            'model_type': 'test_model',
            'target_columns': [f'target_{i}' for i in range(5)],
            'feature_columns': [f'feature_{i}' for i in range(10)]
        }
        
        # Save model
        test_model_path = 'temp_test_model.pkl'
        ModelPersistence.save_model(dummy_model, test_model_path, metadata)
        
        # Test submission class
        from competition_submission_real import CompetitionSubmission
        
        # Initialize submission (this will load the model)
        submission = CompetitionSubmission(test_model_path)
        
        # Test simulation
        result = submission.simulate_submission()
        
        print(f"  ✅ Competition submission simulation: {result.shape}")
        
        # Cleanup
        if Path(test_model_path).exists():
            os.remove(test_model_path)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Competition submission test failed: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering for competition data."""
    print("\n🧪 Testing Competition Feature Engineering...")
    
    try:
        from data_processing.competition_data_loader import CompetitionDataLoader
        
        # Create sample market data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'date_id': dates,
            'LME_COPPER_CLOSE': 8000 + np.cumsum(np.random.randn(100) * 50),
            'LME_ALUMINUM_CLOSE': 2000 + np.cumsum(np.random.randn(100) * 20),
            'JPX_NIKKEI_CLOSE': 27000 + np.cumsum(np.random.randn(100) * 200),
            'FX_USDJPY_CLOSE': 130 + np.cumsum(np.random.randn(100) * 0.5)
        }).set_index('date_id')
        
        loader = CompetitionDataLoader()
        
        # Test time series features
        ts_features = loader.create_time_series_features(data)
        print(f"  ✅ Time series features: {ts_features.shape}")
        
        # Test cross-market features
        cross_features = loader.create_cross_market_features(ts_features)
        print(f"  ✅ Cross-market features: {cross_features.shape}")
        
        # Test market grouping
        market_groups = loader.get_market_groups()
        print(f"  ✅ Market groups: {len(market_groups)} markets")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Feature engineering test failed: {e}")
        return False

def main():
    """Run all competition tests."""
    print("🏆 TESTING MITSUI COMMODITY PREDICTION CHALLENGE FRAMEWORK")
    print("=" * 70)
    
    tests = [
        test_competition_data_loader,
        test_multi_target_model,
        test_feature_engineering,
        test_competition_submission
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 70)
    print("COMPETITION TEST RESULTS:")
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All competition tests passed! Framework is ready for the Mitsui Challenge.")
        print("\nNext steps for competition:")
        print("1. Download real competition data: kaggle competitions download -c mitsui-commodity-prediction-challenge")
        print("2. Train model: python train_competition.py --data-path data/ --explore-data")
        print("3. Submit: python competition_submission_real.py")
        print("4. Monitor performance during forecasting phase")
    else:
        print("\n❌ Some competition tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main()
