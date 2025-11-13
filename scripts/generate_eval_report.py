"""
Generate comprehensive model evaluation report (PDF).
Includes performance metrics, visualizations, and comparisons.
"""

import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report
)


def load_model_results(artifacts_dir: Path) -> dict:
    """Load all model artifacts and results."""
    import sys
    from pathlib import Path
    
    # Add models directory to path for imports
    models_dir = Path(__file__).parent.parent / "models"
    if str(models_dir) not in sys.path:
        sys.path.insert(0, str(models_dir))
    
    results = {}
    
    for model_dir in artifacts_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        model_path = model_dir / "model.pkl"
        
        if not model_path.exists():
            continue
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            results[model_name] = {
                'model': model,
                'path': model_dir
            }
        except Exception as e:
            print(f"Warning: Could not load model from {model_path}: {e}")
            continue
    
    return results


def create_title_page(pdf, title="Model Evaluation Report"):
    """Create title page."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    ax.text(0.5, 0.7, title, 
            ha='center', va='center', fontsize=28, fontweight='bold')
    ax.text(0.5, 0.6, "Crypto Volatility Detection",
            ha='center', va='center', fontsize=18)
    ax.text(0.5, 0.5, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            ha='center', va='center', fontsize=12)
    ax.text(0.5, 0.3, "Milestone 3: Model Training & Evaluation",
            ha='center', va='center', fontsize=14, style='italic')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_metrics_summary_page(pdf, metrics_dict: dict):
    """Create metrics comparison table."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    
    # Title
    fig.text(0.5, 0.95, 'Model Performance Comparison', 
             ha='center', fontsize=16, fontweight='bold')
    
    # Create comparison table
    metrics_df = pd.DataFrame(metrics_dict).T
    
    # Format table
    cell_text = []
    for idx, row in metrics_df.iterrows():
        cell_text.append([
            idx,
            f"{row.get('pr_auc', 0):.4f}",
            f"{row.get('f1_score', 0):.4f}",
            f"{row.get('precision', 0):.4f}",
            f"{row.get('recall', 0):.4f}",
            f"{row.get('roc_auc', 0):.4f}"
        ])
    
    table = ax.table(
        cellText=cell_text,
        colLabels=['Model', 'PR-AUC', 'F1-Score', 'Precision', 'Recall', 'ROC-AUC'],
        cellLoc='center',
        loc='center',
        bbox=[0.1, 0.4, 0.8, 0.4]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best scores
    best_pr_auc = metrics_df['pr_auc'].max() if 'pr_auc' in metrics_df else 0
    for i, (idx, row) in enumerate(metrics_df.iterrows(), 1):
        if row.get('pr_auc', 0) == best_pr_auc:
            table[(i, 1)].set_facecolor('#FFD700')
    
    # Add notes
    fig.text(0.5, 0.25, 
             f"Primary Metric: PR-AUC (best: {best_pr_auc:.4f})",
             ha='center', fontsize=12, style='italic')
    fig.text(0.5, 0.20,
             "PR-AUC is preferred due to class imbalance (spike events are rare)",
             ha='center', fontsize=10)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_pr_curves_page(pdf, test_results: dict):
    """Create PR curves comparison."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model_name, data in test_results.items():
        if 'y_true' in data and 'y_proba' in data:
            precision, recall, _ = precision_recall_curve(data['y_true'], data['y_proba'])
            from sklearn.metrics import average_precision_score
            pr_auc = average_precision_score(data['y_true'], data['y_proba'])
            ax.plot(recall, precision, label=f"{model_name} (AUC={pr_auc:.3f})", linewidth=2)
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_roc_curves_page(pdf, test_results: dict):
    """Create ROC curves comparison."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model_name, data in test_results.items():
        if 'y_true' in data and 'y_proba' in data:
            fpr, tpr, _ = roc_curve(data['y_true'], data['y_proba'])
            from sklearn.metrics import roc_auc_score
            roc_auc = roc_auc_score(data['y_true'], data['y_proba'])
            ax.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})", linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_confusion_matrices_page(pdf, test_results: dict):
    """Create confusion matrices for all models."""
    n_models = len(test_results)
    cols = 2
    rows = (n_models + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(11, 4 * rows))
    fig.suptitle('Confusion Matrices (Test Set)', fontsize=16, fontweight='bold')
    
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (model_name, data) in enumerate(test_results.items()):
        if 'y_true' in data and 'y_pred' in data:
            cm = confusion_matrix(data['y_true'], data['y_pred'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Normal', 'Spike'],
                       yticklabels=['Normal', 'Spike'])
            axes[idx].set_title(model_name, fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def create_feature_importance_page(pdf, models_dict: dict, X_test=None):
    """Create feature importance plots."""
    n_plots = sum(1 for m in models_dict.values() 
                  if hasattr(m['model'], 'coef_') or hasattr(m['model'], 'feature_importances_'))
    
    if n_plots == 0:
        return
    
    # Create figure and subplots - handle axes properly
    cols = 2
    rows = (n_plots + 1) // 2
    fig = plt.figure(figsize=(11, 4 * rows))
    fig.suptitle('Feature Importance', fontsize=16, fontweight='bold')
    
    idx = 0
    for model_name, data in models_dict.items():
        model = data['model']
        
        # Only create subplot if model has feature importance attributes
        if hasattr(model, 'coef_'):
            # Create subplot at correct position
            ax = fig.add_subplot(rows, cols, idx + 1)
            
            # Logistic Regression - use actual feature names
            importance = model.coef_[0]
            # Get feature names from the model's training data if available
            if hasattr(model, 'feature_names_in_'):
                feature_names = list(model.feature_names_in_)
            else:
                # Fallback: try to get from test data columns passed to function
                # This ensures we use the same features as training
                if X_test is not None and hasattr(X_test, 'columns'):
                    feature_names = list(X_test.columns)
                else:
                    # Last resort: use actual feature names (should match train.py)
                    feature_names = ['return_mean_60s', 'return_mean_300s', 
                                   'return_std_300s', 'spread', 'spread_bps']
            
            sorted_idx = np.argsort(np.abs(importance))
            feature_names_arr = np.array(feature_names)[sorted_idx]
            importance_sorted = importance[sorted_idx]
            ax.barh(feature_names_arr, importance_sorted)
            ax.set_title(f'{model_name} (Coefficients)', fontweight='bold')
            ax.set_xlabel('Coefficient Value')
            idx += 1
            
        elif hasattr(model, 'feature_importances_'):
            # Create subplot at correct position
            ax = fig.add_subplot(rows, cols, idx + 1)
            
            # XGBoost - use actual feature names from model
            importance = model.feature_importances_
            # Get feature names from the model's training data if available
            if hasattr(model, 'feature_names_in_'):
                feature_names = list(model.feature_names_in_)
            else:
                # Fallback: try to get from test data columns passed to function
                # This ensures we use the same features as training
                if X_test is not None and hasattr(X_test, 'columns'):
                    feature_names = list(X_test.columns)
                else:
                    # Last resort: use actual feature names (should match train.py)
                    feature_names = ['return_mean_60s', 'return_mean_300s', 
                                   'return_std_300s', 'spread', 'spread_bps']
            
            sorted_idx = np.argsort(importance)
            feature_names_arr = np.array(feature_names)[sorted_idx]
            importance_sorted = importance[sorted_idx]
            ax.barh(feature_names_arr, importance_sorted)
            ax.set_title(f'{model_name} (Importance)', fontweight='bold')
            ax.set_xlabel('Feature Importance')
            idx += 1
    
    # Hide unused subplots
    total_subplots = rows * cols
    for i in range(idx, total_subplots):
        ax_unused = fig.add_subplot(rows, cols, i + 1)
        ax_unused.axis('off')
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def generate_report(features_path: str, artifacts_dir: str, output_path: str):
    """Generate comprehensive evaluation report."""
    print(f"Generating evaluation report...")
    
    # Import prepare_features from train.py and BaselineVolatilityDetector
    import sys
    models_dir = Path(__file__).parent.parent / "models"
    if str(models_dir) not in sys.path:
        sys.path.insert(0, str(models_dir))
    from train import prepare_features
    from baseline import BaselineVolatilityDetector
    
    features_path = Path(features_path)
    artifacts_dir = Path(artifacts_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_parquet(features_path)
    df = df.sort_values('timestamp')
    
    # Time-based split (use last 15% as test)
    test_start_idx = int(len(df) * 0.85)
    test_df = df.iloc[test_start_idx:].copy()
    
    # Load models
    models_dict = load_model_results(artifacts_dir)
    
    if not models_dict:
        print("No models found in artifacts directory")
        return
    
    # Prepare test data using same logic as training
    X_test, y_test = prepare_features(test_df)
    
    # Evaluate all models
    test_results = {}
    metrics_dict = {}
    
    for model_name, data in models_dict.items():
        model = data['model']
        
        # Make predictions
        if hasattr(model, 'predict'):
            # Baseline model needs multiple features (composite z-score)
            if model_name == 'baseline':
                # Use the same feature selection logic as train.py
                baseline_features = BaselineVolatilityDetector.DEFAULT_FEATURES
                available_features = [f for f in baseline_features if f in X_test.columns]
                
                if not available_features:
                    print(f"Warning: None of the required baseline features found.")
                    print(f"  Expected: {baseline_features}")
                    print(f"  Available: {X_test.columns.tolist()}")
                    continue
                
                if len(available_features) < len(baseline_features):
                    missing = set(baseline_features) - set(available_features)
                    print(f"⚠ Warning: Some baseline features missing: {missing}")
                    print(f"   Using {len(available_features)} available features: {available_features}")
                
                X_model = X_test[available_features].copy()
            else:
                X_model = X_test
            
            y_pred = model.predict(X_model)
            
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_model)[:, 1]
            else:
                y_proba = y_pred
            
            # Store results
            test_results[model_name] = {
                'y_true': y_test.values,
                'y_pred': y_pred,
                'y_proba': y_proba
            }
            
            # Compute metrics
            from sklearn.metrics import (precision_recall_fscore_support, 
                                       roc_auc_score, average_precision_score)
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='binary', zero_division=0
            )
            
            try:
                roc_auc = roc_auc_score(y_test, y_proba)
            except:
                roc_auc = 0.0
            
            try:
                pr_auc = average_precision_score(y_test, y_proba)
            except:
                pr_auc = 0.0
            
            metrics_dict[model_name] = {
                'pr_auc': pr_auc,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'roc_auc': roc_auc
            }
    
    # Generate PDF report
    with PdfPages(output_path) as pdf:
        create_title_page(pdf)
        create_metrics_summary_page(pdf, metrics_dict)
        create_pr_curves_page(pdf, test_results)
        create_roc_curves_page(pdf, test_results)
        create_confusion_matrices_page(pdf, test_results)
        create_feature_importance_page(pdf, models_dict, X_test)
    
    print(f"✓ Report saved to: {output_path}")
    
    # Print summary to console
    print("\n=== Model Performance Summary ===")
    for model_name, metrics in metrics_dict.items():
        print(f"\n{model_name}:")
        print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Generate model evaluation report")
    parser.add_argument("--features", default="data/processed/features.parquet",
                       help="Path to features parquet file")
    parser.add_argument("--artifacts", default="models/artifacts",
                       help="Path to model artifacts directory")
    parser.add_argument("--output", default="reports/model_eval.pdf",
                       help="Output PDF path")
    
    args = parser.parse_args()
    
    generate_report(args.features, args.artifacts, args.output)


if __name__ == "__main__":
    main()