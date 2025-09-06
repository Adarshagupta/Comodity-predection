#!/usr/bin/env python
"""
Setup script for the commodity forecasting competition project.

This script helps set up the environment and verify all components work correctly.
"""

import subprocess
import sys
import os
from pathlib import Path


def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False
    return True


def create_directories():
    """Create necessary directories."""
    print("Creating directory structure...")
    
    directories = [
        "data",
        "data/raw", 
        "data/processed",
        "data/test",
        "submissions",
        "logs",
        "models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  Created: {directory}")
    
    print("✅ Directory structure created")


def test_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")
    
    modules_to_test = [
        "src.data_processing.data_loader",
        "src.feature_engineering.features", 
        "src.models.ensemble_models",
        "src.evaluation.metrics",
        "src.utils.submission"
    ]
    
    sys.path.append('src')
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            return False
    
    print("✅ All modules imported successfully")
    return True


def run_basic_test():
    """Run a basic end-to-end test."""
    print("Running basic functionality test...")
    
    try:
        # Test data loading
        from src.data_processing.data_loader import DataLoader
        data_loader = DataLoader()
        print("  ✅ DataLoader initialized")
        
        # Test feature engineering
        from src.feature_engineering.features import FeatureEngineer
        feature_engineer = FeatureEngineer()
        print("  ✅ FeatureEngineer initialized")
        
        # Test model creation
        from src.models.ensemble_models import create_default_ensemble
        model = create_default_ensemble()
        print("  ✅ Ensemble model created")
        
        # Test metrics
        from src.evaluation.metrics import CompetitionMetrics
        import numpy as np
        
        # Test Sharpe ratio calculation
        pred = np.random.randn(100)
        target = np.random.randn(100)
        score = CompetitionMetrics.sharpe_ratio_variant(pred, target)
        print(f"  ✅ Sharpe ratio variant: {score:.4f}")
        
        print("✅ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False


def create_sample_data():
    """Create sample data files for testing."""
    print("Creating sample data files...")
    
    import pandas as pd
    import numpy as np
    
    # Generate sample data
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    n_days = len(dates)
    
    # LME sample data
    lme_data = pd.DataFrame({
        'date': dates,
        'copper_price': 8000 + np.cumsum(np.random.randn(n_days) * 50),
        'aluminum_price': 2000 + np.cumsum(np.random.randn(n_days) * 20),
        'zinc_price': 3000 + np.cumsum(np.random.randn(n_days) * 30),
    })
    lme_data.to_csv('data/raw/lme_sample.csv', index=False)
    
    # JPX sample data
    jpx_data = pd.DataFrame({
        'date': dates,
        'nikkei_close': 27000 + np.cumsum(np.random.randn(n_days) * 200),
        'topix_close': 1900 + np.cumsum(np.random.randn(n_days) * 15),
    })
    jpx_data.to_csv('data/raw/jpx_sample.csv', index=False)
    
    print("✅ Sample data files created in data/raw/")


def print_usage_instructions():
    """Print usage instructions."""
    print("\n" + "="*60)
    print("COMMODITY FORECASTING COMPETITION - SETUP COMPLETE")
    print("="*60)
    
    print("\n📋 QUICK START GUIDE:")
    print("\n1. Train a model with sample data:")
    print("   python train.py --sample-data")
    
    print("\n2. Generate predictions:")
    print("   python predict.py --model-path submissions/trained_model.pkl --sample-data")
    
    print("\n3. Run the competition submission:")
    print("   python competition_submission.py")
    
    print("\n4. Explore with Jupyter notebook:")
    print("   jupyter notebook notebooks/commodity_forecasting_example.ipynb")
    
    print("\n📁 PROJECT STRUCTURE:")
    print("   data/           - Data files (add your competition data here)")
    print("   src/            - Source code modules") 
    print("   notebooks/      - Jupyter notebooks for exploration")
    print("   configs/        - Configuration files")
    print("   submissions/    - Model outputs and predictions")
    
    print("\n🔧 CONFIGURATION:")
    print("   Edit configs/model_config.json for model settings")
    print("   Edit configs/data_config.json for data settings")
    
    print("\n📊 COMPETITION REQUIREMENTS:")
    print("   - Runtime: ≤8 hours (training), ≤9 hours (forecasting)")
    print("   - Metric: Sharpe ratio variant (mean Spearman correlation / std dev)")
    print("   - Data: LME, JPX, US Stock, Forex markets")
    
    print("\n🎯 NEXT STEPS:")
    print("   1. Replace sample data with real competition datasets")
    print("   2. Tune hyperparameters using the validation framework")
    print("   3. Experiment with feature engineering techniques")
    print("   4. Test different ensemble configurations")
    
    print("\nGood luck with the competition! 🚀")


def main():
    """Main setup function."""
    print("🚀 Setting up Commodity Forecasting Competition Environment")
    print("="*60)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher required")
        return False
    
    print(f"✅ Python {sys.version}")
    
    # Setup steps
    success = True
    
    success &= install_requirements()
    create_directories()
    success &= test_imports()
    success &= run_basic_test()
    create_sample_data()
    
    if success:
        print("\n🎉 Setup completed successfully!")
        print_usage_instructions()
    else:
        print("\n❌ Setup encountered errors. Please check the output above.")
        return False
    
    return True


if __name__ == "__main__":
    main()
