"""Regression model training and evaluation functions."""

import optuna.integration.lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


class RegressionModel:
    """Base class for regression models."""
    
    def __init__(self, model_type, params=None):
        self.model_type = model_type
        self.params = params or {}
        self.model = None
        self.scaler = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the regression model."""
        raise NotImplementedError
        
    def predict(self, X):
        """Make predictions."""
        raise NotImplementedError
        
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        predictions = self.predict(X_test)
        return self._calculate_metrics(y_test, predictions, X_test.shape[1])
    
    @staticmethod
    def _calculate_metrics(y_true, y_pred, n_features):
        """Calculate regression metrics."""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        adj_r2 = RegressionModel._calculate_adjusted_r2(r2, len(y_true), n_features)
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'adj_r2': adj_r2
        }
    
    @staticmethod
    def _calculate_adjusted_r2(r2, n_samples, n_features):
        """Calculate adjusted R-squared."""
        if n_samples <= n_features + 1:
            return np.nan
        return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)


class LightGBMModel(RegressionModel):
    """LightGBM regression model."""
    
    def train(self, X_train, y_train, feature_names, X_val=None, y_val=None):
        """Train LightGBM model."""
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        
        if X_val is not None and y_val is not None:
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets = [valid_data]
        else:
            valid_sets = None
        
        callbacks = [lgb.log_evaluation(50)]
        
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=1000,
            valid_sets=valid_sets,
            callbacks=callbacks
        )
        return self
    
    def predict(self, X):
        """Make predictions with LightGBM."""
        return self.model.predict(X)


class XGBoostModel(RegressionModel):
    """XGBoost regression model."""
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train XGBoost model."""
        self.model = XGBRegressor(**self.params)
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            self.model.fit(X_train, y_train, eval_set=eval_set, verbose=10)
        else:
            self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        """Make predictions with XGBoost."""
        return self.model.predict(X)


class SVMModel(RegressionModel):
    """SVM regression model."""
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train SVM model with feature scaling."""
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model = SVR(**self.params)
        self.model.fit(X_train_scaled, y_train)
        return self
    
    def predict(self, X):
        """Make predictions with SVM."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class RandomForestModel(RegressionModel):
    """Random Forest regression model."""
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train Random Forest model."""
        self.model = RandomForestRegressor(**self.params)
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        """Make predictions with Random Forest."""
        return self.model.predict(X)


class GradientBoostingModel(RegressionModel):
    """Gradient Boosting regression model."""
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train Gradient Boosting model."""
        self.model = GradientBoostingRegressor(**self.params)
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        """Make predictions with Gradient Boosting."""
        return self.model.predict(X)


class KNNModel(RegressionModel):
    """K-Nearest Neighbors regression model."""
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train KNN model with feature scaling."""
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model = KNeighborsRegressor(**self.params)
        self.model.fit(X_train_scaled, y_train)
        return self
    
    def predict(self, X):
        """Make predictions with KNN."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


# Factory function to create models
def create_model(model_type, params=None):
    """Factory function to create regression models."""
    model_classes = {
        'lightgbm': LightGBMModel,
        'xgboost': XGBoostModel,
        'svm': SVMModel,
        'random_forest': RandomForestModel,
        'gradient_boosting': GradientBoostingModel,
        'knn': KNNModel
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model_classes[model_type](model_type, params)


# Default parameter configurations
DEFAULT_PARAMS = {
    'lightgbm': {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'learning_rate': 0.01,
        'num_leaves': 100,
        'max_depth': 8,
        'min_data_in_leaf': 10,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 3,
        'lambda_l1': 0.2,
        'lambda_l2': 0.2,
        'min_gain_to_split': 0.1,
        'max_bin': 255,
        'extra_trees': True,
        'path_smooth': 0.1,
        'early_stopping_rounds': 100,
        'verbosity': -1,
    },
    'xgboost': {
        'objective': 'reg:squarederror',
        'learning_rate': 0.01,
        'max_depth': 8,
        'min_child_weight': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'gamma': 0.1,
        'reg_alpha': 0.2,
        'reg_lambda': 0.2,
        'n_estimators': 1000,
        'eval_metric': 'rmse',
    },
    'svm': {
        'kernel': 'rbf',
        'C': 1.0,
        'epsilon': 0.1,
        'gamma': 'scale'
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    },
    'knn': {
        'n_neighbors': 5,
        'weights': 'uniform'
    }
}
