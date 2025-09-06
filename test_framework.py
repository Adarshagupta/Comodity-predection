#!/usr/bin/env python
"""
Quick test script to verify the commodity forecasting framework works correctly.
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from pathlib import Path

# Ensure directories exist
Path('submissions').mkdir(exist_ok=True)

def test_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")
    
    try:
        from data_processing.data_loader import DataLoader
        from feature_engineering.features import FeatureEngineer
        from evaluation.metrics import CompetitionMetrics, evaluate_model_performance
        from utils.submission import ModelPersistence
        print("✅ All modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_processing():
    """Test data loading and processing."""
    print("\nTesting data processing...")
    
    try:
        from data_processing.data_loader import DataLoader
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        sample_data = {
            'lme': pd.DataFrame({
                'copper_price': 8000 + np.cumsum(np.random.randn(len(dates)) * 20)
            }, index=dates),
            'forex': pd.DataFrame({
                'usd_jpy_rate': 140 + np.cumsum(np.random.randn(len(dates)) * 0.5)
            }, index=dates)
        }
        
        data_loader = DataLoader()
        aligned_data = data_loader.align_data_by_date(sample_data)
        clean_data = data_loader.clean_data(aligned_data)
        
        print(f"✅ Data processing successful. Shape: {clean_data.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Data processing error: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering."""
    print("\nTesting feature engineering...")
    
    try:
        from data_processing.data_loader import DataLoader
        from feature_engineering.features import FeatureEngineer
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
        data = pd.DataFrame({
            'lme_copper_price': 8000 + np.cumsum(np.random.randn(len(dates)) * 20),
            'lme_aluminum_price': 2000 + np.cumsum(np.random.randn(len(dates)) * 10),
        }, index=dates)
        
        feature_engineer = FeatureEngineer()
        features = feature_engineer.build_feature_pipeline(
            data,
            asset_pairs=[('lme_copper_price', 'lme_aluminum_price')],
            market_prefixes=['lme']
        )
        
        print(f"✅ Feature engineering successful. Features: {features.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Feature engineering error: {e}")
        return False

def test_metrics():
    """Test evaluation metrics."""
    print("\nTesting evaluation metrics...")
    
    try:
        from evaluation.metrics import CompetitionMetrics, evaluate_model_performance
        
        # Create sample predictions and targets
        np.random.seed(42)
        predictions = np.random.randn(100)
        targets = predictions + np.random.randn(100) * 0.5  # Add some noise
        
        # Test Sharpe ratio variant
        sharpe_score = CompetitionMetrics.sharpe_ratio_variant(predictions, targets)
        
        # Test comprehensive evaluation
        metrics = evaluate_model_performance(predictions, targets)
        
        print(f"✅ Metrics calculation successful. Sharpe variant: {sharpe_score:.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Metrics calculation error: {e}")
        return False

def test_simple_model():
    """Test simple model training."""
    print("\nTesting simple model training...")
    
    try:
        from sklearn.linear_model import Ridge
        from evaluation.metrics import CompetitionMetrics
        
        # Create sample data
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.random.randn(200)
        
        # Train simple model
        model = Ridge(alpha=1.0)
        model.fit(X[:150], y[:150])
        
        # Make predictions
        predictions = model.predict(X[150:])
        targets = y[150:]
        
        # Evaluate
        score = CompetitionMetrics.sharpe_ratio_variant(predictions, targets)
        
        print(f"✅ Simple model training successful. Score: {score:.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Model training error: {e}")
        return False

def test_model_persistence():
    """Test model saving and loading."""
    print("\nTesting model persistence...")
    
    try:
        from utils.submission import ModelPersistence
        from sklearn.linear_model import Ridge
        
        # Create and train a simple model
        model = Ridge(alpha=1.0)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        model.fit(X, y)
        
        # Save model
        test_path = 'submissions/test_model.pkl'
        metadata = {'test': True, 'features': ['a', 'b', 'c']}
        ModelPersistence.save_model(model, test_path, metadata)
        
        # Load model
        loaded_model, loaded_metadata = ModelPersistence.load_model(test_path)
        
        # Test prediction
        test_pred_original = model.predict(X[:5])
        test_pred_loaded = loaded_model.predict(X[:5])
        
        if np.allclose(test_pred_original, test_pred_loaded):
            print("✅ Model persistence successful")
            return True
        else:
            print("❌ Model persistence failed - predictions don't match")
            return False
            
    except Exception as e:
        print(f"❌ Model persistence error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Commodity Forecasting Framework")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_processing,
        test_feature_engineering,
        test_metrics,
        test_simple_model,
        test_model_persistence
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("TEST RESULTS:")
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Framework is ready for use.")
        print("\nNext steps:")
        print("1. Run: python train.py --sample-data")
        print("2. Run: python predict.py --model-path submissions/trained_model.pkl --sample-data")
        print("3. Explore: jupyter notebook notebooks/commodity_forecasting_example.ipynb")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main()
