"""
Ensemble models for commodity return prediction.
Combines multiple algorithms for robust forecasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
try:
    import catboost as cb
except ImportError:
    cb = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseModel:
    """
    Base class for all prediction models.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = None
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit the model."""
        raise NotImplementedError
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        raise NotImplementedError
        
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance if available."""
        return None


class XGBoostModel(BaseModel):
    """
    XGBoost model for commodity return prediction.
    """
    
    def __init__(self, **params):
        super().__init__("XGBoost")
        
        default_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(params)
        
        self.model = xgb.XGBRegressor(**default_params)
        self.scaler = StandardScaler()
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit XGBoost model."""
        X_scaled = self.scaler.fit_transform(X.fillna(0))
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with XGBoost."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_scaled = self.scaler.transform(X.fillna(0))
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance."""
        if not self.is_fitted:
            return None
            
        importance = self.model.feature_importances_
        return pd.Series(importance, index=range(len(importance)))


class LightGBMModel(BaseModel):
    """
    LightGBM model for commodity return prediction.
    """
    
    def __init__(self, **params):
        super().__init__("LightGBM")
        
        default_params = {
            'objective': 'regression',
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1
        }
        default_params.update(params)
        
        self.model = lgb.LGBMRegressor(**default_params)
        self.scaler = StandardScaler()
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit LightGBM model."""
        X_scaled = self.scaler.fit_transform(X.fillna(0))
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with LightGBM."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_scaled = self.scaler.transform(X.fillna(0))
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance."""
        if not self.is_fitted:
            return None
            
        importance = self.model.feature_importances_
        return pd.Series(importance, index=range(len(importance)))


class CatBoostModel(BaseModel):
    """
    CatBoost model for commodity return prediction.
    """
    
    def __init__(self, **params):
        super().__init__("CatBoost")
        
        if cb is None:
            raise ImportError("CatBoost not available")
            
        default_params = {
            'iterations': 100,
            'depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbose': False
        }
        default_params.update(params)
        
        self.model = cb.CatBoostRegressor(**default_params)
        self.scaler = StandardScaler()
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit CatBoost model."""
        X_scaled = self.scaler.fit_transform(X.fillna(0))
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with CatBoost."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_scaled = self.scaler.transform(X.fillna(0))
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance."""
        if not self.is_fitted:
            return None
            
        importance = self.model.feature_importances_
        return pd.Series(importance, index=range(len(importance)))


class RandomForestModel(BaseModel):
    """
    Random Forest model for commodity return prediction.
    """
    
    def __init__(self, **params):
        super().__init__("RandomForest")
        
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(params)
        
        self.model = RandomForestRegressor(**default_params)
        self.scaler = StandardScaler()
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit Random Forest model."""
        X_scaled = self.scaler.fit_transform(X.fillna(0))
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with Random Forest."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_scaled = self.scaler.transform(X.fillna(0))
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance."""
        if not self.is_fitted:
            return None
            
        importance = self.model.feature_importances_
        return pd.Series(importance, index=range(len(importance)))


class LinearModel(BaseModel):
    """
    Linear regression models with regularization.
    """
    
    def __init__(self, model_type: str = "ridge", **params):
        super().__init__(f"Linear_{model_type}")
        
        if model_type == "ridge":
            default_params = {'alpha': 1.0, 'random_state': 42}
            default_params.update(params)
            self.model = Ridge(**default_params)
        elif model_type == "lasso":
            default_params = {'alpha': 1.0, 'random_state': 42}
            default_params.update(params)
            self.model = Lasso(**default_params)
        elif model_type == "elastic":
            default_params = {'alpha': 1.0, 'l1_ratio': 0.5, 'random_state': 42}
            default_params.update(params)
            self.model = ElasticNet(**default_params)
        else:
            self.model = LinearRegression()
            
        self.scaler = StandardScaler()
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit linear model."""
        X_scaled = self.scaler.fit_transform(X.fillna(0))
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with linear model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_scaled = self.scaler.transform(X.fillna(0))
        return self.model.predict(X_scaled)


class EnsembleModel:
    """
    Ensemble model that combines multiple base models.
    """
    
    def __init__(self, models: List[BaseModel], 
                 ensemble_method: str = "average",
                 weights: Optional[List[float]] = None):
        """
        Initialize ensemble model.
        
        Args:
            models: List of base models
            ensemble_method: "average", "weighted", or "stacking"
            weights: Weights for weighted average (if None, uses equal weights)
        """
        self.models = models
        self.ensemble_method = ensemble_method
        self.weights = weights
        self.meta_model = None
        self.is_fitted = False
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        elif len(weights) != len(models):
            raise ValueError("Number of weights must match number of models")
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit all base models."""
        logger.info(f"Training ensemble with {len(self.models)} models...")
        
        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{len(self.models)}: {model.name}")
            try:
                model.fit(X, y)
            except Exception as e:
                logger.error(f"Error training {model.name}: {e}")
                
        # If using stacking, train meta-model
        if self.ensemble_method == "stacking":
            self._train_meta_model(X, y)
            
        self.is_fitted = True
        logger.info("Ensemble training complete")
    
    def _train_meta_model(self, X: pd.DataFrame, y: pd.Series):
        """Train meta-model for stacking."""
        # Generate base model predictions
        base_predictions = []
        
        for model in self.models:
            if model.is_fitted:
                try:
                    pred = model.predict(X)
                    base_predictions.append(pred)
                except Exception as e:
                    logger.warning(f"Error getting predictions from {model.name}: {e}")
                    
        if base_predictions:
            # Stack predictions as features for meta-model
            meta_features = np.column_stack(base_predictions)
            self.meta_model = Ridge(alpha=1.0)
            self.meta_model.fit(meta_features, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
            
        predictions = []
        
        for model in self.models:
            if model.is_fitted:
                try:
                    pred = model.predict(X)
                    predictions.append(pred)
                except Exception as e:
                    logger.warning(f"Error getting predictions from {model.name}: {e}")
        
        if not predictions:
            raise ValueError("No valid predictions from base models")
        
        predictions = np.array(predictions)
        
        if self.ensemble_method == "average":
            return np.mean(predictions, axis=0)
            
        elif self.ensemble_method == "weighted":
            # Only use weights for models that provided predictions
            valid_weights = self.weights[:len(predictions)]
            valid_weights = np.array(valid_weights) / np.sum(valid_weights)
            return np.average(predictions, axis=0, weights=valid_weights)
            
        elif self.ensemble_method == "stacking" and self.meta_model is not None:
            meta_features = predictions.T  # Transpose to get (n_samples, n_models)
            return self.meta_model.predict(meta_features)
            
        else:
            # Fallback to simple average
            return np.mean(predictions, axis=0)
    
    def get_model_performance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Get individual model performance scores."""
        from ..evaluation.metrics import CompetitionMetrics
        
        scores = {}
        
        for model in self.models:
            if model.is_fitted:
                try:
                    pred = model.predict(X)
                    score = CompetitionMetrics.sharpe_ratio_variant(pred, y.values)
                    scores[model.name] = score
                except Exception as e:
                    logger.warning(f"Error evaluating {model.name}: {e}")
                    scores[model.name] = 0.0
                    
        return scores


def create_default_ensemble() -> EnsembleModel:
    """
    Create a default ensemble with commonly used models.
    
    Returns:
        EnsembleModel with default configuration
    """
    models = [
        XGBoostModel(n_estimators=200, max_depth=6, learning_rate=0.05),
        LightGBMModel(n_estimators=200, max_depth=6, learning_rate=0.05),
        RandomForestModel(n_estimators=100, max_depth=10),
        LinearModel("ridge", alpha=1.0),
        LinearModel("lasso", alpha=0.1),
    ]
    
    # Add CatBoost if available
    if cb is not None:
        models.append(CatBoostModel(iterations=200, depth=6, learning_rate=0.05))
    
    ensemble = EnsembleModel(
        models=models,
        ensemble_method="weighted",
        weights=[0.25, 0.25, 0.15, 0.15, 0.1, 0.1] if cb else [0.3, 0.3, 0.2, 0.1, 0.1]
    )
    
    return ensemble


def create_custom_ensemble(model_configs: List[Dict[str, Any]]) -> EnsembleModel:
    """
    Create a custom ensemble from configuration.
    
    Args:
        model_configs: List of model configuration dictionaries
        
    Returns:
        EnsembleModel with custom configuration
    """
    models = []
    
    for config in model_configs:
        model_type = config.pop('type')
        
        if model_type == 'xgboost':
            models.append(XGBoostModel(**config))
        elif model_type == 'lightgbm':
            models.append(LightGBMModel(**config))
        elif model_type == 'catboost' and cb is not None:
            models.append(CatBoostModel(**config))
        elif model_type == 'randomforest':
            models.append(RandomForestModel(**config))
        elif model_type == 'linear':
            models.append(LinearModel(**config))
        else:
            logger.warning(f"Unknown model type: {model_type}")
    
    return EnsembleModel(models=models)
