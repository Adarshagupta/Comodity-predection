#!/usr/bin/env python
"""
Download script for Mitsui Commodity Prediction Challenge data.
"""

import os
import sys
from pathlib import Path

def setup_kaggle_credentials():
    """
    Guide user through Kaggle API setup.
    """
    print("🔑 KAGGLE API SETUP REQUIRED")
    print("=" * 50)
    print()
    print("To download the competition data, you need Kaggle API credentials:")
    print()
    print("1. Go to https://www.kaggle.com/account")
    print("2. Scroll down to 'API' section")
    print("3. Click 'Create New API Token'")
    print("4. Download kaggle.json file")
    print("5. Place it in one of these locations:")
    print(f"   - C:\\Users\\{os.getenv('USERNAME', 'YourUsername')}\\.kaggle\\kaggle.json")
    print("   - Or set KAGGLE_CONFIG_DIR environment variable")
    print()
    print("6. Make sure the file has this format:")
    print('   {"username":"your-username","key":"your-api-key"}')
    print()
    
    # Check if credentials exist
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_file = kaggle_dir / 'kaggle.json'
    
    if kaggle_file.exists():
        print("✅ Found kaggle.json - ready to download!")
        return True
    else:
        print("❌ kaggle.json not found. Please set up credentials first.")
        print(f"Expected location: {kaggle_file}")
        return False

def download_competition_data():
    """
    Download the Mitsui Commodity Prediction Challenge data.
    """
    try:
        import kaggle
        
        print("\n📊 DOWNLOADING COMPETITION DATA")
        print("=" * 50)
        
        # Create data directory
        Path('data').mkdir(exist_ok=True)
        
        # Download competition files
        print("Downloading competition data...")
        kaggle.api.competition_download_files(
            'mitsui-commodity-prediction-challenge',
            path='data/',
            unzip=True
        )
        
        print("✅ Competition data downloaded successfully!")
        
        # List downloaded files
        data_files = list(Path('data').glob('*'))
        print(f"\nDownloaded files ({len(data_files)}):")
        for file in sorted(data_files):
            size = file.stat().st_size / (1024*1024) if file.is_file() else 0
            print(f"  📄 {file.name} ({size:.1f} MB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        
        if "401" in str(e):
            print("\n🔑 Authentication Error:")
            print("Please check your Kaggle API credentials.")
            print("Make sure kaggle.json has correct username and API key.")
        elif "403" in str(e):
            print("\n🚫 Permission Error:")
            print("You may need to:")
            print("1. Accept competition rules on Kaggle website")
            print("2. Join the competition first")
        elif "404" in str(e):
            print("\n🔍 Competition Not Found:")
            print("Please verify the competition name is correct.")
            print("Visit: https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge")
        
        return False

def create_sample_data_files():
    """
    Create sample data files that match the competition format for testing.
    """
    print("\n🎮 CREATING SAMPLE DATA FOR TESTING")
    print("=" * 50)
    
    import pandas as pd
    import numpy as np
    
    # Ensure data directory exists
    Path('data').mkdir(exist_ok=True)
    
    # Create sample data matching competition format
    dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')
    n_days = len(dates)
    
    print(f"Creating sample data for {n_days} days...")
    
    # 1. train.csv - Historic finance data
    train_data = pd.DataFrame({'date_id': dates})
    
    # LME metals
    metals = ['COPPER', 'ALUMINUM', 'ZINC', 'LEAD', 'NICKEL']
    base_prices = {'COPPER': 8000, 'ALUMINUM': 2000, 'ZINC': 3000, 'LEAD': 2200, 'NICKEL': 18000}
    
    for metal in metals:
        base_price = base_prices[metal]
        train_data[f'LME_{metal}_CLOSE'] = base_price + np.cumsum(np.random.randn(n_days) * base_price * 0.02)
        train_data[f'LME_{metal}_VOLUME'] = np.random.exponential(1000, n_days)
    
    # JPX indices
    train_data['JPX_NIKKEI_CLOSE'] = 27000 + np.cumsum(np.random.randn(n_days) * 200)
    train_data['JPX_TOPIX_CLOSE'] = 1900 + np.cumsum(np.random.randn(n_days) * 15)
    
    # US indices
    train_data['US_SPX_CLOSE'] = 4200 + np.cumsum(np.random.randn(n_days) * 50)
    train_data['US_NDX_CLOSE'] = 12500 + np.cumsum(np.random.randn(n_days) * 150)
    train_data['US_DJI_CLOSE'] = 34000 + np.cumsum(np.random.randn(n_days) * 300)
    
    # FX rates
    train_data['FX_USDJPY_CLOSE'] = 130 + np.cumsum(np.random.randn(n_days) * 0.5)
    train_data['FX_EURUSD_CLOSE'] = 1.1 + np.cumsum(np.random.randn(n_days) * 0.01)
    train_data['FX_GBPUSD_CLOSE'] = 1.3 + np.cumsum(np.random.randn(n_days) * 0.01)
    
    train_data.to_csv('data/train.csv', index=False)
    print(f"  ✅ Created train.csv ({train_data.shape})")
    
    # 2. train_labels.csv - Target variables
    train_labels = pd.DataFrame({'date_id': dates})
    
    # Create sample targets (simplified for demo)
    for i in range(50):  # Sample of 50 targets instead of full 424
        if i < 25:
            # Log returns
            base_col = list(train_data.columns)[i % 10 + 1]
            if base_col in train_data.columns:
                returns = train_data[base_col].pct_change()
                train_labels[f'target_{i}'] = np.log(1 + returns).shift(-1)
        else:
            # Price differences
            col1 = list(train_data.columns)[(i % 5) + 1]
            col2 = list(train_data.columns)[(i % 5) + 6]
            if col1 in train_data.columns and col2 in train_data.columns:
                diff = train_data[col1] - train_data[col2]
                train_labels[f'target_{i}'] = diff.pct_change().shift(-1)
    
    train_labels.to_csv('data/train_labels.csv', index=False)
    print(f"  ✅ Created train_labels.csv ({train_labels.shape})")
    
    # 3. target_pairs.csv - Target metadata
    target_pairs = []
    for i in range(50):
        if i < 25:
            target_pairs.append({
                'target': f'target_{i}',
                'lag': 1,
                'pair': list(train_data.columns)[i % 10 + 1]
            })
        else:
            col1 = list(train_data.columns)[(i % 5) + 1]
            col2 = list(train_data.columns)[(i % 5) + 6]
            target_pairs.append({
                'target': f'target_{i}',
                'lag': 1,
                'pair': f'{col1}-{col2}'
            })
    
    target_pairs_df = pd.DataFrame(target_pairs)
    target_pairs_df.to_csv('data/target_pairs.csv', index=False)
    print(f"  ✅ Created target_pairs.csv ({target_pairs_df.shape})")
    
    # 4. test.csv - Test data
    test_dates = pd.date_range(start='2025-01-01', periods=90, freq='D')
    test_data = pd.DataFrame({'date_id': test_dates})
    
    # Add same structure as training data
    for col in train_data.columns[1:]:
        last_value = train_data[col].iloc[-1]
        test_data[col] = last_value + np.cumsum(np.random.randn(90) * last_value * 0.01)
    
    test_data['is_scored'] = True
    test_data.loc[60:, 'is_scored'] = False  # Last 30 days not scored
    
    test_data.to_csv('data/test.csv', index=False)
    print(f"  ✅ Created test.csv ({test_data.shape})")
    
    print(f"\n📁 Sample data created in data/ directory:")
    data_files = list(Path('data').glob('*.csv'))
    for file in sorted(data_files):
        size = file.stat().st_size / 1024
        print(f"  📄 {file.name} ({size:.1f} KB)")
    
    return True

def main():
    """Main function to download or create sample data."""
    print("🏆 MITSUI COMMODITY PREDICTION CHALLENGE - DATA SETUP")
    print("=" * 60)
    
    # Check for Kaggle credentials
    has_credentials = setup_kaggle_credentials()
    
    if has_credentials:
        print("\nAttempting to download real competition data...")
        success = download_competition_data()
        
        if not success:
            print("\nFalling back to sample data creation...")
            create_sample_data_files()
    else:
        print("\nCreating sample data for development...")
        create_sample_data_files()
    
    print("\n🎯 NEXT STEPS:")
    print("1. If using sample data: python train_competition.py --data-path data/ --explore-data")
    print("2. If you have real data: Set up Kaggle credentials and re-run this script")
    print("3. Explore data: jupyter notebook notebooks/mitsui_competition_example.ipynb")
    
    # Test data loading
    print("\n🧪 Testing data loading...")
    try:
        sys.path.append('src')
        from data_processing.competition_data_loader import explore_competition_data
        explore_competition_data('data/')
    except Exception as e:
        print(f"Data loading test failed: {e}")

if __name__ == "__main__":
    main()
