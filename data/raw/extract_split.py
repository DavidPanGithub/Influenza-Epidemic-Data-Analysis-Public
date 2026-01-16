"""Data extraction and preparation functions - UPDATED FOR PERIOD SPLITTING."""

import pandas as pd
import numpy as np

def load_flu_data():
    """Load and prepare influenza data."""
    flu_df = pd.read_csv("flux_data.csv")
    return flu_df

def load_google_trends_data():
    """Load and prepare Google Trends data."""
    query_df = pd.read_csv("query_trend.csv")
    return query_df

def prepare_google_data_split(cutoff_row=157):
    """
    Prepare Google Trends and influenza data for modeling, split by period.
    
    Args:
        cutoff_row: Row number where border reopened (157 corresponds to May 2023)
    
    Returns:
        dict: Dictionary containing data for both periods
    """
    query_df = load_google_trends_data()
    flu_df = load_flu_data()
    
    # Extract feature names (keywords)
    feature_names = query_df.columns[1:].tolist()
    
    # Extract ALL data
    X_all = query_df.iloc[:, 1:].to_numpy().astype("float32")
    y_all = flu_df['AandB'].to_numpy().astype('float32')
    
    # Split at cutoff_row (May 2023 border reopening)
    X_pre = X_all[:cutoff_row]
    y_pre = y_all[:cutoff_row]
    
    X_post = X_all[cutoff_row:]
    y_post = y_all[cutoff_row:]
    
    # Validation
    if len(X_pre) != len(y_pre) or len(X_post) != len(y_post):
        raise ValueError("X and y have different lengths. Check query_trend.csv and flux_data.csv")
    
    if np.any(np.isinf(X_pre)) or np.any(np.isinf(y_pre)) or np.any(np.isinf(X_post)) or np.any(np.isinf(y_post)):
        raise ValueError("Infinite values found in data")
    
    print(f"Data loaded and split successfully:")
    print(f"  PRE period (rows 0-{cutoff_row-1}): {X_pre.shape[0]} samples")
    print(f"  POST period (rows {cutoff_row}-end): {X_post.shape[0]} samples")
    print(f"  Number of features: {len(feature_names)}")
    
    return {
        'pre': {'X': X_pre, 'y': y_pre, 'feature_names': feature_names},
        'post': {'X': X_post, 'y': y_post, 'feature_names': feature_names},
        'all': {'X': X_all, 'y': y_all, 'feature_names': feature_names}
    }