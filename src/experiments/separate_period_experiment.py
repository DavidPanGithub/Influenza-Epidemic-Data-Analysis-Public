"""Separate period training experiment with visualization."""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from data.raw.extract_split import prepare_google_data_split
from src.models.feature_selection import select_top_features, filter_features_by_importance
from src.models.regression import create_model, DEFAULT_PARAMS
from src.utils.metrics import calculate_experiment_statistics, print_statistics


class SeparatePeriodExperiment:
    """Train and evaluate models separately on PRE and POST periods."""
    
    def __init__(self, n_experiments=10, n_top_features=8, cutoff_row=157, random_seed=42):
        self.n_experiments = n_experiments
        self.n_top_features = n_top_features
        self.cutoff_row = cutoff_row
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Models to evaluate
        self.models_to_evaluate = [
            'lightgbm',
            'xgboost',
            'svm',
            'random_forest',
            'gradient_boosting',
            'knn'
        ]
        
        # Results storage
        self.results = {
            'pre': {'metrics': [], 'feature_importance': None, 'selected_features': None},
            'post': {'metrics': [], 'feature_importance': None, 'selected_features': None}
        }
        
        # Set style for plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def train_and_evaluate_period(self, period_name, X, y, feature_names):
        """
        Train and evaluate models on a specific period.
        """
        print(f"\n{'='*60}")
        print(f"ANALYZING {period_name.upper()} PERIOD")
        print(f"{'='*60}")
        
        print(f"Data shape: {X.shape}")
        print(f"Target statistics:")
        print(f"  Min: {y.min():.1f}")
        print(f"  Max: {y.max():.1f}")
        print(f"  Mean: {y.mean():.1f}")
        print(f"  Std: {y.std():.1f}")
        print(f"  Variance: {np.var(y):.1f}")
        
        # Select top features for this period
        print(f"\nSelecting top {self.n_top_features} features...")
        selected_indices, selected_features, importance_df = select_top_features(
            X, y, feature_names, n_features=self.n_top_features, 
            random_state=self.random_seed
        )
        
        # Store feature importance
        self.results[period_name]['feature_importance'] = importance_df
        self.results[period_name]['selected_features'] = selected_features
        
        # Run multiple experiments
        print(f"\nRunning {self.n_experiments} experiments...")
        period_metrics = []
        
        for i in range(self.n_experiments):
            print(f"\nExperiment {i+1}/{self.n_experiments}")
            metrics = self.run_single_experiment(
                X, y, feature_names, selected_indices, i
            )
            period_metrics.append(metrics)
        
        # Calculate statistics
        print(f"\n{'='*60}")
        print(f"{period_name.upper()} PERIOD RESULTS")
        print(f"{'='*60}")
        
        stats = calculate_experiment_statistics(period_metrics)
        print_statistics(stats)
        
        self.results[period_name]['metrics'] = period_metrics
        self.results[period_name]['stats'] = stats
        
        return stats
    
    def run_single_experiment(self, X, y, feature_names, selected_indices, experiment_num):
        """
        Run a single training and evaluation experiment.
        """
        # Filter features
        X_filtered, feature_names_filtered = filter_features_by_importance(
            X, feature_names, selected_indices
        )
        
        # Split the data (60% train, 20% validation, 20% test)
        # Use fixed seed based on experiment number for reproducibility
        split_seed = self.random_seed + experiment_num * 100
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_filtered, y, test_size=0.4, random_state=split_seed
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=split_seed
        )
        
        metrics_results = {}
        
        # Train and evaluate each model
        for model_type in self.models_to_evaluate:
            try:
                # Get model with default parameters
                model = create_model(model_type, DEFAULT_PARAMS.get(model_type, {}))
                
                # Train model
                if model_type == 'lightgbm':
                    model.train(X_train, y_train, feature_names_filtered, X_val, y_val)
                else:
                    model.train(X_train, y_train, X_val, y_val)
                
                # Evaluate model
                metrics = model.evaluate(X_test, y_test)
                metrics_results[model_type] = metrics
                
                print(f"  {model_type}: R²={metrics['r2']:.4f}, MSE={metrics['mse']:.0f}")
                
            except Exception as e:
                print(f"  {model_type}: Failed - {str(e)[:50]}")
                metrics_results[model_type] = None
        
        return metrics_results
    
    def run_comparison(self):
        """Run the full separate period comparison."""
        print("=" * 80)
        print("SEPARATE PERIOD TRAINING EXPERIMENT")
        print(f"Training models independently on PRE and POST periods (cutoff: row {self.cutoff_row})")
        print("=" * 80)
        
        # Load and split data
        print("\nLoading and splitting data...")
        data_dict = prepare_google_data_split(self.cutoff_row)
        
        # Store data for plotting
        self.data_dict = data_dict
        
        # Train and evaluate on PRE period
        pre_stats = self.train_and_evaluate_period(
            'pre', 
            data_dict['pre']['X'], 
            data_dict['pre']['y'], 
            data_dict['pre']['feature_names']
        )
        
        # Train and evaluate on POST period
        post_stats = self.train_and_evaluate_period(
            'post', 
            data_dict['post']['X'], 
            data_dict['post']['y'], 
            data_dict['post']['feature_names']
        )
        
        # Generate comparison summary
        print("\n" + "=" * 80)
        print("PERFORMANCE COMPARISON SUMMARY")
        print("=" * 80)
        self.generate_comparison_summary()
        
        # Generate diagrams
        print("\nGenerating diagrams...")
        self.generate_diagrams()
        
        return self.results
    
    def generate_comparison_summary(self):
        """Generate a summary comparing PRE and POST period performance."""
        # Create comparison table
        comparison_data = []
        
        for model_name in self.models_to_evaluate:
            # Check if model has results in both periods
            pre_has_results = (model_name in self.results['pre']['stats'] and 
                              self.results['pre']['stats'][model_name] is not None)
            post_has_results = (model_name in self.results['post']['stats'] and 
                               self.results['post']['stats'][model_name] is not None)
            
            if pre_has_results and post_has_results:
                pre_stats = self.results['pre']['stats'][model_name]
                post_stats = self.results['post']['stats'][model_name]
                
                comparison_data.append({
                    'Model': model_name.upper(),
                    'MSE (Pre)': f"{pre_stats['mse']['mean']:,.0f}",
                    'MSE (Post)': f"{post_stats['mse']['mean']:,.0f}",
                    'MAE (Pre)': f"{pre_stats['mae']['mean']:.1f}",
                    'MAE (Post)': f"{post_stats['mae']['mean']:.1f}",
                    'R² (Pre)': f"{pre_stats['r2']['mean']:.4f}",
                    'R² (Post)': f"{post_stats['r2']['mean']:.4f}",
                    'Δ R²': f"{post_stats['r2']['mean'] - pre_stats['r2']['mean']:+.4f}"
                })
            else:
                # Show which period is missing
                status = []
                if not pre_has_results:
                    status.append("No PRE results")
                if not post_has_results:
                    status.append("No POST results")
                
                comparison_data.append({
                    'Model': model_name.upper(),
                    'MSE (Pre)': 'N/A',
                    'MSE (Post)': 'N/A',
                    'MAE (Pre)': 'N/A',
                    'MAE (Post)': 'N/A',
                    'R² (Pre)': 'N/A',
                    'R² (Post)': 'N/A',
                    'Δ R²': f"Missing: {', '.join(status)}"
                })
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            print("\nTable 1: Model Performance Comparison (Separate Training)")
            print("-" * 100)
            print(df_comparison.to_string(index=False))
        
        # Feature comparison
        print("\n\nTable 2: Top Feature Comparison Between Periods")
        print("-" * 60)
        
        pre_features = self.results['pre']['selected_features']
        post_features = self.results['post']['selected_features']
        
        if pre_features is not None and post_features is not None:
            feature_comparison = []
            max_len = max(len(pre_features), len(post_features))
            
            for i in range(max_len):
                pre_feat = pre_features[i] if i < len(pre_features) else ""
                post_feat = post_features[i] if i < len(post_features) else ""
                feature_comparison.append({
                    'Rank': i + 1,
                    'Pre-Period': pre_feat,
                    'Post-Period': post_feat
                })
            
            df_features = pd.DataFrame(feature_comparison)
            print(df_features.to_string(index=False))
        
        # Key observations
        print("\n\nKEY OBSERVATIONS:")
        print("-" * 60)
        
        # Find best model in each period
        for period in ['pre', 'post']:
            best_model = None
            best_r2 = -float('inf')
            
            for model_name in self.models_to_evaluate:
                if (model_name in self.results[period]['stats'] and 
                    self.results[period]['stats'][model_name] is not None):
                    r2 = self.results[period]['stats'][model_name]['r2']['mean']
                    if r2 > best_r2:
                        best_r2 = r2
                        best_model = model_name
            
            if best_model:
                print(f"{period.upper()} Period: {best_model.upper()} performs best (R²={best_r2:.4f})")
        
        # Calculate average improvement
        valid_models = []
        for model_name in self.models_to_evaluate:
            pre_valid = (model_name in self.results['pre']['stats'] and 
                        self.results['pre']['stats'][model_name] is not None)
            post_valid = (model_name in self.results['post']['stats'] and 
                         self.results['post']['stats'][model_name] is not None)
            
            if pre_valid and post_valid:
                valid_models.append(model_name)
        
        if valid_models:
            pre_avg_r2 = np.mean([self.results['pre']['stats'][m]['r2']['mean'] 
                                 for m in valid_models])
            post_avg_r2 = np.mean([self.results['post']['stats'][m]['r2']['mean'] 
                                  for m in valid_models])
            
            print(f"\nOverall (based on {len(valid_models)} models):")
            print(f"  Average R² improvement: {post_avg_r2 - pre_avg_r2:+.4f}")
            print(f"  Models perform {'better' if post_avg_r2 > pre_avg_r2 else 'worse'} in POST period")
    
    def generate_diagrams(self):
        """Generate comprehensive diagrams for the analysis."""
        # Create results directory if it doesn't exist
        os.makedirs('results/figures', exist_ok=True)
        
        print("\nGenerating diagrams in 'results/figures/' directory:")
        
        # 1. Performance Comparison Bar Chart
        print("  1. Performance comparison bar chart...")
        self.plot_performance_comparison()
        
        # 2. R² Improvement Visualization
        print("  2. R² improvement visualization...")
        self.plot_r2_improvement()
        
        # 3. Feature Importance Comparison
        print("  3. Feature importance comparison...")
        self.plot_feature_importance_comparison()
        
        # 4. Data Distribution Comparison
        print("  4. Data distribution comparison...")
        self.plot_data_distribution()
        
        # 5. Model Performance Heatmap
        print("  5. Model performance heatmap...")
        self.plot_performance_heatmap()
        
        print("\n✓ All diagrams saved to 'results/figures/' directory")
    
    def plot_performance_comparison(self):
        """Plot side-by-side performance comparison."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Prepare data
        models = []
        pre_r2 = []
        post_r2 = []
        pre_mse = []
        post_mse = []
        pre_mae = []
        post_mae = []
        
        for model_name in self.models_to_evaluate:
            if (model_name in self.results['pre']['stats'] and 
                model_name in self.results['post']['stats'] and
                self.results['pre']['stats'][model_name] is not None and
                self.results['post']['stats'][model_name] is not None):
                
                models.append(model_name.upper())
                pre_r2.append(self.results['pre']['stats'][model_name]['r2']['mean'])
                post_r2.append(self.results['post']['stats'][model_name]['r2']['mean'])
                pre_mse.append(self.results['pre']['stats'][model_name]['mse']['mean'])
                post_mse.append(self.results['post']['stats'][model_name]['mse']['mean'])
                pre_mae.append(self.results['pre']['stats'][model_name]['mae']['mean'])
                post_mae.append(self.results['post']['stats'][model_name]['mae']['mean'])
        
        x = np.arange(len(models))
        width = 0.35
        
        # R² Plot
        ax = axes[0]
        pre_bars = ax.bar(x - width/2, pre_r2, width, label='Pre-Reopening', 
                         color='lightcoral', alpha=0.8, edgecolor='darkred')
        post_bars = ax.bar(x + width/2, post_r2, width, label='Post-Reopening', 
                          color='lightgreen', alpha=0.8, edgecolor='darkgreen')
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
        ax.set_title('R² Score Comparison\nPRE vs POST Border Reopening', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Add value labels
        for bar in pre_bars + post_bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        # MSE Plot
        ax = axes[1]
        pre_bars = ax.bar(x - width/2, pre_mse, width, label='Pre-Reopening', 
                         color='lightcoral', alpha=0.8, edgecolor='darkred')
        post_bars = ax.bar(x + width/2, post_mse, width, label='Post-Reopening', 
                          color='lightgreen', alpha=0.8, edgecolor='darkgreen')
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('MSE', fontsize=12, fontweight='bold')
        ax.set_title('Mean Squared Error Comparison\nPRE vs POST Border Reopening', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add value labels
        for bar in pre_bars + post_bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:,.0f}', ha='center', va='bottom', fontsize=9)
        
        # MAE Plot
        ax = axes[2]
        pre_bars = ax.bar(x - width/2, pre_mae, width, label='Pre-Reopening', 
                         color='lightcoral', alpha=0.8, edgecolor='darkred')
        post_bars = ax.bar(x + width/2, post_mae, width, label='Post-Reopening', 
                          color='lightgreen', alpha=0.8, edgecolor='darkgreen')
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('MAE', fontsize=12, fontweight='bold')
        ax.set_title('Mean Absolute Error Comparison\nPRE vs POST Border Reopening', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add value labels
        for bar in pre_bars + post_bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('results/figures/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_r2_improvement(self):
        """Plot R² improvement from PRE to POST period."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = []
        improvements = []
        colors = []
        
        for model_name in self.models_to_evaluate:
            if (model_name in self.results['pre']['stats'] and 
                model_name in self.results['post']['stats'] and
                self.results['pre']['stats'][model_name] is not None and
                self.results['post']['stats'][model_name] is not None):
                
                pre_r2 = self.results['pre']['stats'][model_name]['r2']['mean']
                post_r2 = self.results['post']['stats'][model_name]['r2']['mean']
                improvement = post_r2 - pre_r2
                
                models.append(model_name.upper())
                improvements.append(improvement)
                
                # Color based on improvement
                if improvement > 0:
                    colors.append('green')
                else:
                    colors.append('red')
        
        # Sort by improvement
        sorted_idx = np.argsort(improvements)
        models = [models[i] for i in sorted_idx]
        improvements = [improvements[i] for i in sorted_idx]
        colors = [colors[i] for i in sorted_idx]
        
        bars = ax.barh(models, improvements, color=colors, alpha=0.7)
        
        ax.set_xlabel('Δ R² (POST - PRE)', fontsize=12, fontweight='bold')
        ax.set_title('R² Improvement After Border Reopening', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.grid(True, alpha=0.3, linestyle='--', axis='x')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{width:+.2f}', ha='left' if width >= 0 else 'right',
                   va='center', fontsize=10, fontweight='bold')
        
        # Add interpretation
        ax.text(0.05, 0.95, 'Positive: Better after reopening\nNegative: Worse after reopening',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig('results/figures/r2_improvement.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance_comparison(self):
        """Plot feature importance comparison between periods."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        
        for idx, period in enumerate(['pre', 'post']):
            ax = axes[idx]
            
            if self.results[period]['feature_importance'] is not None:
                importance_df = self.results[period]['feature_importance']
                top_features = importance_df.head(10).sort_values('importance', ascending=True)
                
                bars = ax.barh(range(len(top_features)), top_features['importance'], 
                              color='steelblue', alpha=0.7)
                
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(top_features['feature'], fontsize=9)
                ax.set_xlabel('Feature Importance (Gain)', fontsize=10, fontweight='bold')
                ax.set_title(f'Top 10 Features - {period.upper()} Period\n(May 2023 Border Reopening)',
                           fontsize=12, fontweight='bold', pad=10)
                ax.grid(True, alpha=0.3, linestyle='--', axis='x')
                
                # Add value labels
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2,
                           f'{width:,.0f}', ha='left', va='center', fontsize=8)
            
            else:
                ax.text(0.5, 0.5, f'No feature importance data\nfor {period.upper()} period',
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{period.upper()} Period', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/figures/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_data_distribution(self):
        """Plot data distribution comparison between periods."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Get data
        pre_data = self.data_dict['pre']['y']
        post_data = self.data_dict['post']['y']
        
        # Histogram
        ax = axes[0]
        bins = np.linspace(0, max(max(pre_data), max(post_data)), 30)
        
        ax.hist(pre_data, bins=bins, alpha=0.5, label=f'PRE (n={len(pre_data)})', 
               color='lightcoral', edgecolor='darkred')
        ax.hist(post_data, bins=bins, alpha=0.5, label=f'POST (n={len(post_data)})', 
               color='lightgreen', edgecolor='darkgreen')
        
        ax.set_xlabel('Influenza Cases (A+B)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Distribution of Influenza Cases\nPRE vs POST Border Reopening',
                   fontsize=12, fontweight='bold', pad=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Box plot
        ax = axes[1]
        bp = ax.boxplot([pre_data, post_data], labels=['PRE', 'POST'], 
                       patch_artist=True, showfliers=True)
        
        # Customize box colors
        bp['boxes'][0].set_facecolor('lightcoral')
        bp['boxes'][0].set_edgecolor('darkred')
        bp['boxes'][1].set_facecolor('lightgreen')
        bp['boxes'][1].set_edgecolor('darkgreen')
        
        ax.set_ylabel('Influenza Cases (A+B)', fontsize=11, fontweight='bold')
        ax.set_title('Case Distribution Comparison\n(Box Plot)', 
                   fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        stats_text = f"""PRE Period:
Mean: {np.mean(pre_data):.1f}
Std: {np.std(pre_data):.1f}
Max: {np.max(pre_data):.0f}

POST Period:
Mean: {np.mean(post_data):.1f}
Std: {np.std(post_data):.1f}
Max: {np.max(post_data):.0f}"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('results/figures/data_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_heatmap(self):
        """Create a heatmap of model performance metrics."""
        # Prepare data for heatmap
        metrics = ['R²', 'MSE', 'MAE']
        periods = ['PRE', 'POST']
        
        # Create data matrix
        data = []
        for model_name in self.models_to_evaluate:
            if (model_name in self.results['pre']['stats'] and 
                model_name in self.results['post']['stats'] and
                self.results['pre']['stats'][model_name] is not None and
                self.results['post']['stats'][model_name] is not None):
                
                row = [
                    self.results['pre']['stats'][model_name]['r2']['mean'],
                    self.results['post']['stats'][model_name]['r2']['mean'],
                    self.results['pre']['stats'][model_name]['mse']['mean'] / 1000,  # Scale for visualization
                    self.results['post']['stats'][model_name]['mse']['mean'] / 1000,
                    self.results['pre']['stats'][model_name]['mae']['mean'],
                    self.results['post']['stats'][model_name]['mae']['mean']
                ]
                data.append(row)
        
        if not data:
            return
            
        data = np.array(data)
        models = [m.upper() for m in self.models_to_evaluate if m in self.results['pre']['stats']]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(periods) * len(metrics)))
        ax.set_yticks(np.arange(len(models)))
        
        # Create x-axis labels
        x_labels = []
        for period in periods:
            for metric in metrics:
                x_labels.append(f'{period}\n{metric}')
        
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_yticklabels(models, fontsize=10, fontweight='bold')
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Performance Value', rotation=-90, va="bottom", fontsize=10)
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(periods) * len(metrics)):
                if j < 2:  # R² values
                    text = ax.text(j, i, f'{data[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontsize=9, fontweight='bold')
                elif j < 4:  # MSE values (scaled)
                    text = ax.text(j, i, f'{data[i, j]:.0f}k',
                                  ha="center", va="center", color="black", fontsize=9, fontweight='bold')
                else:  # MAE values
                    text = ax.text(j, i, f'{data[i, j]:.0f}',
                                  ha="center", va="center", color="black", fontsize=9, fontweight='bold')
        
        ax.set_title("Model Performance Heatmap\nPRE vs POST Border Reopening", 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('results/figures/performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function to run the separate period experiment."""
    try:
        # Create and run experiment
        experiment = SeparatePeriodExperiment(
            n_experiments=10,
            n_top_features=8,
            cutoff_row=157,  # May 2023 border reopening
            random_seed=42
        )
        
        results = experiment.run_comparison()
        
        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETE!")
        print("Diagrams saved to 'results/figures/' directory")
        print("=" * 80)
        
        return results
        
    except Exception as e:
        print(f"\nError running experiment: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()