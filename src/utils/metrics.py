"""Utility functions for metrics calculation and reporting."""

import numpy as np
from typing import Dict, List


def calculate_experiment_statistics(metrics_list: List[Dict]) -> Dict:
    """
    Calculate statistics across multiple experiments.
    
    Args:
        metrics_list: List of dictionaries containing metrics from each experiment
    
    Returns:
        Dictionary containing statistics for each model
    """
    all_models = set()
    for metrics in metrics_list:
        all_models.update(metrics.keys())
    
    stats = {}
    
    for model in all_models:
        # Collect all valid metrics for this model
        model_metrics = []
        for exp_metrics in metrics_list:
            if exp_metrics.get(model) is not None:
                model_metrics.append(exp_metrics[model])
        
        if not model_metrics:
            stats[model] = None
            continue
        
        # Calculate statistics for each metric
        model_stats = {}
        metric_names = model_metrics[0].keys()  # Get metric names from first valid result
        
        for metric in metric_names:
            values = [m[metric] for m in model_metrics if metric in m and not np.isnan(m[metric])]
            if values:
                model_stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        stats[model] = model_stats
    
    return stats


def print_statistics(stats: Dict):
    """
    Print statistics in a readable format.
    
    Args:
        stats: Statistics dictionary from calculate_experiment_statistics
    """
    for model_name, model_stats in stats.items():
        if model_stats is None:
            print(f"\n{model_name.upper()}: No successful experiments")
            continue
        
        print(f"\n{model_name.upper()} Statistics:")
        print("-" * 60)
        
        for metric_name, values in model_stats.items():
            print(f"{metric_name.upper():<10}: {values['mean']:10.4f} ± {values['std']:8.4f} "
                  f"(min: {values['min']:8.4f}, max: {values['max']:8.4f}, n={values['count']})")


def create_results_summary(stats: Dict, n_experiments: int, n_features: int) -> str:
    """
    Create a formatted summary of results.
    
    Args:
        stats: Statistics dictionary
        n_experiments: Number of experiments run
        n_features: Number of features used
    
    Returns:
        Formatted summary string
    """
    summary = []
    summary.append("=" * 80)
    summary.append(f"EXPERIMENT RESULTS SUMMARY")
    summary.append(f"Number of experiments: {n_experiments}")
    summary.append(f"Number of features used: {n_features}")
    summary.append("=" * 80)
    
    # Find best model by R² mean
    best_model = None
    best_r2 = -float('inf')
    
    for model_name, model_stats in stats.items():
        if model_stats and 'r2' in model_stats:
            r2_mean = model_stats['r2']['mean']
            if r2_mean > best_r2:
                best_r2 = r2_mean
                best_model = model_name
    
    if best_model:
        summary.append(f"\nBest performing model: {best_model.upper()} (R² = {best_r2:.4f})")
    
    return "\n".join(summary)
