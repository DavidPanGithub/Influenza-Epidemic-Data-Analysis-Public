"""Feature selection utilities."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import optuna.integration.lightgbm as lgb


def select_top_features(X, y, feature_names, n_features=10, random_state=42):
    """
    Select top N features using LightGBM feature importance.
    
    Args:
        X: Feature matrix
        y: Target values
        feature_names: List of feature names
        n_features: Number of top features to select
        random_state: Random seed
    
    Returns:
        selected_indices: Indices of selected features
        selected_features: Names of selected features
        importance_df: DataFrame with feature importance scores
    """
    # Split data for feature importance calculation
    X_temp, X_val_temp, y_temp, y_val_temp = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # LightGBM parameters for feature selection
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'seed': random_state,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
    }
    
    # Create datasets
    train_data = lgb.Dataset(X_temp, label=y_temp, feature_name=list(feature_names))
    val_data = lgb.Dataset(X_val_temp, label=y_val_temp, reference=train_data)
    
    # Train model for feature importance
    print("Training model for feature selection...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
    )
    
    # Get feature importance
    importance = model.feature_importance(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
        'rank': range(1, len(feature_names) + 1)
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Select top features
    top_features = importance_df.head(n_features)['feature'].values
    top_importance = importance_df.head(n_features)['importance'].values
    
    # Create mapping from feature name to index
    feature_to_index = {name: idx for idx, name in enumerate(feature_names)}
    selected_indices = [feature_to_index[f] for f in top_features]
    
    # Print results
    print(f"\nSelected top {n_features} features:")
    print("-" * 50)
    for i, (feat, imp) in enumerate(zip(top_features, top_importance), 1):
        print(f"{i:2d}. {feat:<30} Importance: {imp:.6f}")
    
    return selected_indices, top_features, importance_df


def filter_features_by_importance(X, feature_names, selected_indices):
    """
    Filter features matrix and names by selected indices.
    
    Args:
        X: Original feature matrix
        feature_names: Original feature names
        selected_indices: Indices of features to keep
    
    Returns:
        X_filtered: Filtered feature matrix
        filtered_names: Filtered feature names
    """
    X_filtered = X[:, selected_indices]
    filtered_names = [feature_names[i] for i in selected_indices]
    
    return X_filtered, filtered_names
