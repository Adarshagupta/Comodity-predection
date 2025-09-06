# Commodity Price Forecasting Competition - Complete Guide

## 🎯 Competition Overview

This project provides a comprehensive framework for the **Commodity Price Forecasting Competition** that challenges participants to predict future commodity returns using historical data from multiple markets:

- **London Metal Exchange (LME)** - Copper, Aluminum, Zinc, Lead, Nickel
- **Japan Exchange Group (JPX)** - Nikkei, TOPIX indices  
- **US Stock Markets** - S&P 500, NASDAQ, Dow Jones
- **Forex Markets** - USD/JPY, EUR/USD, GBP/USD

### Key Competition Details
- **Evaluation Metric**: Sharpe ratio variant (mean Spearman correlation / standard deviation)
- **Runtime Limits**: ≤8 hours (training), ≤9 hours (forecasting phase)
- **Prize Pool**: $100,000 total (1st: $20k, 2nd-3rd: $10k each, 4th-15th: $5k each)
- **Timeline**: July 2025 - January 2026

## 🏗️ Framework Architecture

### Core Components

1. **Data Processing** (`src/data_processing/`)
   - Multi-market data loading and alignment
   - Missing data handling and cleaning
   - Price-difference series creation

2. **Feature Engineering** (`src/feature_engineering/`)
   - Technical indicators (MA, RSI, MACD, Bollinger Bands)
   - Cross-market correlation features
   - Volatility and momentum indicators
   - Seasonal and lag features

3. **Models** (`src/models/`)
   - Ensemble approach with multiple algorithms:
     - XGBoost, LightGBM, CatBoost
     - Random Forest
     - Regularized Linear Models (Ridge, Lasso, Elastic Net)
   - Weighted ensemble combination

4. **Evaluation** (`src/evaluation/`)
   - Competition metric implementation
   - Time series cross-validation
   - Performance analysis tools

5. **Submission** (`src/utils/`)
   - API integration framework
   - Model persistence
   - Prediction pipeline

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run setup script
python setup.py
```

### 2. Train a Model
```bash
# With sample data (for testing)
python train.py --sample-data --cross-validate

# With real competition data
python train.py --data-path data/competition/ --config configs/model_config.json
```

### 3. Generate Predictions
```bash
# Generate predictions with trained model
python predict.py --model-path submissions/trained_model.pkl --data-path data/test/

# For competition submission
python competition_submission.py
```

### 4. Explore with Notebooks
```bash
jupyter notebook notebooks/commodity_forecasting_example.ipynb
```

## 📊 Model Performance Strategy

### Feature Engineering Approach
1. **Price-Difference Series**: Extract robust signals between asset pairs
2. **Cross-Market Features**: Capture correlations between different markets
3. **Technical Indicators**: Traditional financial analysis tools
4. **Volatility Features**: GARCH-like volatility modeling
5. **Temporal Features**: Seasonal patterns and lag relationships

### Ensemble Strategy
- **Diversity**: Combine tree-based, linear, and boosting algorithms
- **Stability**: Weighted averaging reduces overfitting
- **Robustness**: Multiple models handle different market regimes

### Validation Framework
- **Time Series CV**: Respects temporal structure
- **Rolling Window**: Simulates real trading conditions
- **Competition Metric**: Direct optimization of Sharpe ratio variant

## 🔧 Configuration

### Model Configuration (`configs/model_config.json`)
```json
{
  "ensemble_models": [
    {
      "type": "xgboost",
      "params": {
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.05
      }
    }
  ],
  "feature_selection": {
    "max_features": 100,
    "method": "correlation"
  }
}
```

### Data Configuration (`configs/data_config.json`)
```json
{
  "data_sources": {
    "lme": {
      "file_path": "lme_prices.csv",
      "price_columns": ["copper_price", "aluminum_price"]
    }
  },
  "feature_engineering": {
    "asset_pairs": [["lme_copper_price", "lme_aluminum_price"]],
    "technical_indicators": {
      "moving_averages": [5, 10, 20, 50]
    }
  }
}
```

## 📈 Competition Metric Deep Dive

The competition uses a **Sharpe ratio variant**:

```
Score = mean(Spearman_correlations) / std(Spearman_correlations)
```

### Why This Metric?
- **Ranking Accuracy**: Spearman correlation measures rank prediction quality
- **Stability**: Standard deviation penalizes inconsistent performance
- **Risk-Adjusted**: Similar to Sharpe ratio in finance

### Optimization Strategy
1. Focus on **directional accuracy** over exact values
2. Maintain **consistent performance** across time periods
3. Build **robust features** that generalize well

## 🎯 Winning Strategies

### 1. Feature Engineering Excellence
- **Price Spreads**: Commodity-specific relationships
- **Market Regime**: Identify different market conditions
- **Alternative Data**: Economic indicators, sentiment data

### 2. Model Ensembling
- **Diverse Algorithms**: Different learning paradigms
- **Time-Based Ensembles**: Models trained on different periods
- **Stacking**: Meta-models to combine predictions

### 3. Validation Discipline
- **Out-of-Time Testing**: Simulate competition conditions
- **Regime Analysis**: Performance across different market cycles
- **Overfitting Prevention**: Conservative feature selection

### 4. Risk Management
- **Position Sizing**: Scale predictions by confidence
- **Drawdown Control**: Monitor cumulative performance
- **Regime Detection**: Adapt to changing market conditions

## 🔍 Data Requirements

### Expected Data Format
```
LME Data:
date,copper_price,aluminum_price,zinc_price,...

JPX Data:
date,nikkei_close,topix_close,...

US Stock Data:
date,sp500_close,nasdaq_close,dow_close,...

Forex Data:
date,usd_jpy_rate,eur_usd_rate,gbp_usd_rate,...
```

### Data Quality Checks
- ✅ Consistent date ranges across markets
- ✅ No excessive missing values (< 10%)
- ✅ Reasonable price ranges (no obvious errors)
- ✅ Proper datetime indexing

## 🛠️ Advanced Customization

### Custom Models
```python
from src.models.ensemble_models import BaseModel

class CustomModel(BaseModel):
    def __init__(self):
        super().__init__("CustomModel")
        # Your custom implementation
        
    def fit(self, X, y):
        # Training logic
        pass
        
    def predict(self, X):
        # Prediction logic
        pass
```

### Custom Features
```python
from src.feature_engineering.features import FeatureEngineer

class CustomFeatureEngineer(FeatureEngineer):
    def create_custom_features(self, df):
        # Your custom feature logic
        return df
```

## 📋 Submission Checklist

### Before Final Submission
- [ ] **Runtime Test**: Ensure < 8 hours training time
- [ ] **Memory Check**: Monitor RAM usage
- [ ] **Validation Score**: Confirm CV performance
- [ ] **Code Review**: Check for bugs and edge cases
- [ ] **Data Leakage**: Verify no future information used
- [ ] **API Integration**: Test submission format
- [ ] **Backup Strategy**: Multiple model versions ready

### Competition Day
- [ ] **Final Training**: Use all available data
- [ ] **Model Ensemble**: Deploy best combination
- [ ] **Monitoring**: Track real-time performance
- [ ] **Contingency**: Fallback models ready

## 🏆 Success Metrics

### Development Phase
- **Cross-Validation Score**: > 0.1 Sharpe ratio variant
- **Stability**: Consistent performance across folds
- **Generalization**: Good out-of-sample results

### Competition Phase
- **Leaderboard Position**: Top 10% target
- **Risk Management**: Controlled drawdowns
- **Consistency**: Stable daily predictions

## 🤝 Support and Resources

### Documentation
- Code comments and docstrings throughout
- Configuration examples in `configs/`
- Jupyter notebook tutorials in `notebooks/`

### Debugging
- Comprehensive logging in all modules
- Error handling for edge cases
- Validation functions for data integrity

### Community
- Share insights while respecting competition rules
- Collaborate on feature engineering ideas
- Learn from ensemble strategies

---

## 🎉 Final Notes

This framework provides a **production-ready foundation** for the commodity forecasting competition. Key advantages:

1. **Comprehensive**: Covers all aspects from data to submission
2. **Flexible**: Easy to customize and extend
3. **Robust**: Handles edge cases and errors gracefully
4. **Scalable**: Efficient for large datasets
5. **Competition-Ready**: Meets all platform requirements

**Good luck with the competition!** 🚀

The framework is designed to be competitive out-of-the-box while providing the flexibility to implement your unique strategies and insights.
