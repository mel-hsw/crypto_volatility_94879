"""
Training pipeline for volatility detection models.
Supports baseline (z-score) and ML models (Logistic Regression, XGBoost).
Logs everything to MLflow.
"""

import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)
import matplotlib.pyplot as plt

from baseline import BaselineVolatilityDetector

# Try to import XGBoost (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")


def load_and_split_data(
    features_path: str,
    val_size: float = 0.15,
    test_size: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load features and split into train/val/test with time-based split.
    
    Args:
        features_path: Path to features parquet file
        val_size: Fraction for validation set
        test_size: Fraction for test set
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    df = pd.read_parquet(features_path)
    
    # Validate required columns
    if 'timestamp' not in df.columns:
        raise ValueError(f"Missing required column 'timestamp' in {features_path}. "
                        f"Available columns: {df.columns.tolist()}")
    
    if 'volatility_spike' not in df.columns:
        raise ValueError(
            f"Missing required column 'volatility_spike' in {features_path}.\n"
            f"Available columns: {df.columns.tolist()}\n\n"
            f"This file appears to be unlabeled. Please add labels using:\n"
            f"  python scripts/add_labels.py --features {features_path}\n\n"
            f"Or use a labeled file (e.g., features_labeled.parquet)"
        )
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Time-based split
    n = len(df)
    train_end = int(n * (1 - val_size - test_size))
    val_end = int(n * (1 - test_size))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    return train_df, val_df, test_df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract feature columns and target variable.
    
    Uses reduced feature set (10 features) to minimize multicollinearity:
    - Removed perfectly correlated duplicates (r=1.0):
      * return_std_* (duplicates of log_return_std_*)
      * log_return_mean_* (duplicates of return_mean_*)
    - Keeps diverse time windows (30s, 60s, 300s) for different signals
    - Focuses on log returns (more stable for crypto) and key statistics
    
    Args:
        df: Features dataframe
        
    Returns:
        Tuple of (X, y)
    """
    # Reduced feature set (10 features) - removes perfect correlations
    priority_features = [
        # Volatility features (log returns - more stable for crypto)
        'log_return_std_30s',  # 30-second log return volatility
        'log_return_std_60s',  # 60-second log return volatility (best separation: 0.569)
        'log_return_std_300s', # 300-second log return volatility
        
        # Return statistics
        'return_mean_60s',     # 1-minute return mean
        'return_mean_300s',    # 5-minute return mean
        'return_min_30s',      # Minimum return in 30s (downside risk)
        
        # Spread volatility
        'spread_std_300s',     # 300-second spread volatility
        'spread_mean_60s',     # 60-second spread mean
        
        # Trade intensity
        'tick_count_60s',      # Trading intensity
    ]
    
    # Select available features
    available_cols = [col for col in priority_features if col in df.columns]
    
    # Create derived feature: return range (return_max - return_min)
    if 'return_max_60s' in df.columns and 'return_min_60s' in df.columns:
        df = df.copy()
        df['return_range_60s'] = df['return_max_60s'] - df['return_min_60s']
        if 'return_range_60s' not in available_cols:
            available_cols.append('return_range_60s')
    
    if not available_cols:
        raise ValueError("No matching feature columns found in dataframe. "
                        f"Available columns: {df.columns.tolist()}")
    
    X = df[available_cols].copy()
    y = df['volatility_spike'].copy()
    
    # Fill NaN values with 0
    X = X.fillna(0)
    
    return X, y


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Binary predictions
        y_proba: Prediction probabilities
        
    Returns:
        Dictionary of metrics
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    try:
        roc_auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        roc_auc = 0.0
    
    try:
        pr_auc = average_precision_score(y_true, y_proba)
    except ValueError:
        pr_auc = 0.0
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'accuracy': float((tp + tn) / (tp + tn + fp + fn))
    }


def plot_pr_curve(y_true: np.ndarray, y_proba: np.ndarray, save_path: Path):
    """Generate and save Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR-AUC = {pr_auc:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, save_path: Path):
    """Generate and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC-AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, 
                   threshold: float = 2.0) -> Dict:
    """Train baseline z-score model."""
    print("\n=== Training Baseline Model ===")
    
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    X_test, y_test = prepare_features(test_df)
    
    # Use only volatility feature for baseline
    # Prefer log_return_std_60s (best separation), then fallback to others
    volatility_col = None
    for col_name in ['log_return_std_60s', 'log_return_std_30s', 'log_return_std_300s', 
                     'return_std_60s', 'return_std_30s', 'price_volatility_5min', 'return_std_300s']:
        if col_name in X_train.columns:
            volatility_col = col_name
            break
    
    if volatility_col is None:
        raise ValueError("No volatility column found. Expected log_return_std_60s, log_return_std_30s, log_return_std_300s, return_std_60s, return_std_30s, return_std_300s, or price_volatility_5min")
    
    X_train_base = X_train[[volatility_col]]
    X_val_base = X_val[[volatility_col]]
    X_test_base = X_test[[volatility_col]]
    
    with mlflow.start_run(run_name="baseline_zscore") as run:
        # Log parameters
        mlflow.log_param("model_type", "baseline")
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("train_samples", len(train_df))
        mlflow.log_param("val_samples", len(val_df))
        mlflow.log_param("test_samples", len(test_df))
        
        # Train model
        model = BaselineVolatilityDetector(threshold=threshold)
        model.fit(X_train_base, y_train)
        
        # Validation metrics
        y_val_pred = model.predict(X_val_base)
        y_val_proba = model.predict_proba(X_val_base)[:, 1]
        val_metrics = compute_metrics(y_val.values, y_val_pred, y_val_proba)
        
        # Test metrics
        y_test_pred = model.predict(X_test_base)
        y_test_proba = model.predict_proba(X_test_base)[:, 1]
        test_metrics = compute_metrics(y_test.values, y_test_pred, y_test_proba)
        
        # Log metrics
        for split, metrics in [("val", val_metrics), ("test", test_metrics)]:
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{split}_{metric_name}", value)
        
        # Create plots
        plots_dir = Path("models/artifacts/baseline")
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        plot_pr_curve(y_test.values, y_test_proba, plots_dir / "pr_curve.png")
        plot_roc_curve(y_test.values, y_test_proba, plots_dir / "roc_curve.png")
        
        # Log artifacts - use fluent API which handles HTTP uploads automatically
        artifact_files = [
            ("pr_curve.png", plots_dir / "pr_curve.png"),
            ("roc_curve.png", plots_dir / "roc_curve.png")
        ]
        
        for artifact_name, artifact_path in artifact_files:
            try:
                # Use fluent API - automatically uses HTTP when connected to server
                mlflow.log_artifact(str(artifact_path))
            except (OSError, Exception) as e:
                error_str = str(e)
                if "Read-only file system" in error_str or "/mlflow" in error_str:
                    print(f"⚠ Warning: Could not log artifact {artifact_name} to MLflow server")
                    print(f"   Artifact saved locally at: {artifact_path}")
                    print(f"   Error: {str(e)[:100]}")
                else:
                    raise
        
        # Save model
        model_path = plots_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        try:
            # Use fluent API - automatically uses HTTP when connected to server
            mlflow.log_artifact(str(model_path))
        except (OSError, Exception) as e:
            error_str = str(e)
            if "Read-only file system" in error_str or "/mlflow" in error_str:
                print(f"⚠ Warning: Could not log model artifact to MLflow server")
                print(f"   Model saved locally at: {model_path}")
                print(f"   Error: {str(e)[:100]}")
            else:
                raise
        
        print(f"Validation PR-AUC: {val_metrics['pr_auc']:.4f}")
        print(f"Test PR-AUC: {test_metrics['pr_auc']:.4f}")
        
        return {"model": model, "val_metrics": val_metrics, "test_metrics": test_metrics}


def train_logistic_regression(train_df: pd.DataFrame, val_df: pd.DataFrame, 
                               test_df: pd.DataFrame) -> Dict:
    """Train Logistic Regression model."""
    print("\n=== Training Logistic Regression ===")
    
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    X_test, y_test = prepare_features(test_df)
    
    # Scale features for better convergence
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    with mlflow.start_run(run_name="logistic_regression"):
        # Log parameters
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("train_samples", len(train_df))
        mlflow.log_param("val_samples", len(val_df))
        mlflow.log_param("test_samples", len(test_df))
        mlflow.log_param("features", list(X_train.columns))
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("feature_scaling", "StandardScaler")
        
        # Train model
        model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Validation metrics
        y_val_pred = model.predict(X_val_scaled)
        y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
        val_metrics = compute_metrics(y_val.values, y_val_pred, y_val_proba)
        
        # Test metrics
        y_test_pred = model.predict(X_test_scaled)
        y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
        test_metrics = compute_metrics(y_test.values, y_test_pred, y_test_proba)
        
        # Log metrics
        for split, metrics in [("val", val_metrics), ("test", test_metrics)]:
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{split}_{metric_name}", value)
        
        # Create plots
        plots_dir = Path("models/artifacts/logistic_regression")
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        plot_pr_curve(y_test.values, y_test_proba, plots_dir / "pr_curve.png")
        plot_roc_curve(y_test.values, y_test_proba, plots_dir / "roc_curve.png")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'coefficient': model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)
        
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['feature'], feature_importance['coefficient'])
        plt.xlabel('Coefficient')
        plt.title('Feature Importance (Logistic Regression)')
        plt.tight_layout()
        plt.savefig(plots_dir / "feature_importance.png")
        plt.close()
        
        # Log artifacts - handle potential filesystem errors gracefully
        artifact_files = [
            ("pr_curve.png", plots_dir / "pr_curve.png"),
            ("roc_curve.png", plots_dir / "roc_curve.png"),
            ("feature_importance.png", plots_dir / "feature_importance.png")
        ]
        
        for artifact_name, artifact_path in artifact_files:
            try:
                mlflow.log_artifact(str(artifact_path))
            except OSError as e:
                if "Read-only file system" in str(e) or "/mlflow" in str(e):
                    print(f"⚠ Warning: Could not log artifact {artifact_name} to MLflow server")
                    print(f"   Artifact saved locally at: {artifact_path}")
                else:
                    raise
        
        # Save scaler with model (wrap in Pipeline for easy use)
        from sklearn.pipeline import Pipeline
        model_pipeline = Pipeline([
            ('scaler', scaler),
            ('classifier', model)
        ])
        
        # Log model
        try:
            mlflow.sklearn.log_model(model_pipeline, "model")
        except OSError as e:
            if "Read-only file system" in str(e) or "/mlflow" in str(e):
                print(f"⚠ Warning: Could not log model to MLflow server")
            else:
                raise
        
        # Save locally (save pipeline with scaler)
        model_path = plots_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model_pipeline, f)
        
        print(f"Validation PR-AUC: {val_metrics['pr_auc']:.4f}")
        print(f"Test PR-AUC: {test_metrics['pr_auc']:.4f}")
        
        return {"model": model, "val_metrics": val_metrics, "test_metrics": test_metrics}


def train_xgboost(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
    """Train XGBoost model."""
    if not XGBOOST_AVAILABLE:
        print("XGBoost not available. Skipping.")
        return {}
    
    print("\n=== Training XGBoost ===")
    
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    X_test, y_test = prepare_features(test_df)
    
    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    with mlflow.start_run(run_name="xgboost"):
        # Log parameters
        params = {
            "model_type": "xgboost",
            "max_depth": 5,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "scale_pos_weight": scale_pos_weight,
            "objective": "binary:logistic",
            "eval_metric": "aucpr"
        }
        
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        mlflow.log_param("train_samples", len(train_df))
        mlflow.log_param("val_samples", len(val_df))
        mlflow.log_param("test_samples", len(test_df))
        mlflow.log_param("features", list(X_train.columns))
        
        # Train model
        model = xgb.XGBClassifier(
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100,
            scale_pos_weight=scale_pos_weight,
            objective='binary:logistic',
            eval_metric='aucpr',
            random_state=42
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Validation metrics
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        val_metrics = compute_metrics(y_val.values, y_val_pred, y_val_proba)
        
        # Test metrics
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test.values, y_test_pred, y_test_proba)
        
        # Log metrics
        for split, metrics in [("val", val_metrics), ("test", test_metrics)]:
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{split}_{metric_name}", value)
        
        # Create plots
        plots_dir = Path("models/artifacts/xgboost")
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        plot_pr_curve(y_test.values, y_test_proba, plots_dir / "pr_curve.png")
        plot_roc_curve(y_test.values, y_test_proba, plots_dir / "roc_curve.png")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.xlabel('Importance')
        plt.title('Feature Importance (XGBoost)')
        plt.tight_layout()
        plt.savefig(plots_dir / "feature_importance.png")
        plt.close()
        
        # Log artifacts - handle potential filesystem errors gracefully
        artifact_files = [
            ("pr_curve.png", plots_dir / "pr_curve.png"),
            ("roc_curve.png", plots_dir / "roc_curve.png"),
            ("feature_importance.png", plots_dir / "feature_importance.png")
        ]
        
        for artifact_name, artifact_path in artifact_files:
            try:
                mlflow.log_artifact(str(artifact_path))
            except OSError as e:
                if "Read-only file system" in str(e) or "/mlflow" in str(e):
                    print(f"⚠ Warning: Could not log artifact {artifact_name} to MLflow server")
                    print(f"   Artifact saved locally at: {artifact_path}")
                else:
                    raise
        
        # Log model
        try:
            mlflow.xgboost.log_model(model, "model")
        except OSError as e:
            if "Read-only file system" in str(e) or "/mlflow" in str(e):
                print(f"⚠ Warning: Could not log model to MLflow server")
            else:
                raise
        
        # Save locally
        model_path = plots_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"Validation PR-AUC: {val_metrics['pr_auc']:.4f}")
        print(f"Test PR-AUC: {test_metrics['pr_auc']:.4f}")
        
        return {"model": model, "val_metrics": val_metrics, "test_metrics": test_metrics}


def main():
    parser = argparse.ArgumentParser(description="Train volatility detection models")
    parser.add_argument("--features", default="data/processed/features_labeled.parquet",
                       help="Path to features parquet file")
    parser.add_argument("--models", nargs='+', default=["baseline", "logistic"],
                       choices=["baseline", "logistic", "xgboost"],
                       help="Models to train")
    parser.add_argument("--mlflow-uri", default="http://localhost:5001",
                       help="MLflow tracking URI")
    
    args = parser.parse_args()
    
    # Set MLflow tracking
    # If server URI is provided, try to use it; otherwise use local directory
    if args.mlflow_uri.startswith('http'):
        # Try to connect to MLflow server
        try:
            mlflow.set_tracking_uri(args.mlflow_uri)
            # Test connection with a simple health check or experiment access
            from mlflow.tracking import MlflowClient
            import urllib.request
            # Quick HTTP health check
            health_url = f"{args.mlflow_uri.rstrip('/')}/health"
            try:
                urllib.request.urlopen(health_url, timeout=2)
            except:
                raise ConnectionError("MLflow server health check failed")
            # Try to access MLflow client (this will fail if server is truly not accessible)
            client = MlflowClient(tracking_uri=args.mlflow_uri)
            _ = client._tracking_client  # Access internal client to verify connection
            
            # Verify artifact URI is set correctly (should use server's artifact store)
            # MLflow should automatically use the server's artifact root when connected
            print(f"✓ Connected to MLflow server at {args.mlflow_uri}")
            
            # Ensure we're using the server's artifact store, not a local path
            # This prevents MLflow from trying to write to /mlflow locally
            import os
            # Clear any local MLFLOW_ARTIFACT_URI environment variable that might interfere
            if 'MLFLOW_ARTIFACT_URI' in os.environ:
                artifact_uri = os.environ['MLFLOW_ARTIFACT_URI']
                if artifact_uri.startswith('/mlflow'):
                    del os.environ['MLFLOW_ARTIFACT_URI']
                    print(f"   Cleared local artifact URI: {artifact_uri}")
        except Exception as e:
            # Server not accessible, use local directory
            print(f"⚠ Warning: MLflow server at {args.mlflow_uri} not accessible")
            print(f"   Reason: {str(e)[:80]}")
            print("   Falling back to local MLflow tracking (./mlruns/)")
            # Use local directory instead
            local_mlflow_path = Path("mlruns").absolute()
            local_mlflow_path.mkdir(exist_ok=True, parents=True)
            mlflow.set_tracking_uri(f"file://{local_mlflow_path}")
            print(f"✓ Using local MLflow tracking at: {local_mlflow_path}")
    else:
        # Already a file path
        mlflow.set_tracking_uri(args.mlflow_uri)
    
    mlflow.set_experiment("crypto-volatility-detection")
    
    # Load and split data
    print("Loading data...")
    train_df, val_df, test_df = load_and_split_data(args.features)
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_df)} samples ({train_df['volatility_spike'].mean():.2%} spikes)")
    print(f"  Val:   {len(val_df)} samples ({val_df['volatility_spike'].mean():.2%} spikes)")
    print(f"  Test:  {len(test_df)} samples ({test_df['volatility_spike'].mean():.2%} spikes)")
    
    results = {}
    
    # Train models
    if "baseline" in args.models:
        results['baseline'] = train_baseline(train_df, val_df, test_df)
    
    if "logistic" in args.models:
        results['logistic'] = train_logistic_regression(train_df, val_df, test_df)
    
    if "xgboost" in args.models and XGBOOST_AVAILABLE:
        results['xgboost'] = train_xgboost(train_df, val_df, test_df)
    
    # Summary comparison
    print("\n=== Model Comparison (Test Set) ===")
    print(f"{'Model':<20} {'PR-AUC':<10} {'F1':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 60)
    
    for model_name, result in results.items():
        if result:
            m = result['test_metrics']
            print(f"{model_name:<20} {m['pr_auc']:<10.4f} {m['f1_score']:<10.4f} "
                  f"{m['precision']:<10.4f} {m['recall']:<10.4f}")
    
    # Get the actual tracking URI being used
    actual_uri = mlflow.get_tracking_uri()
    print(f"\nAll models logged to MLflow: {actual_uri}")
    if actual_uri.startswith('http'):
        print(f"View results at: {actual_uri}")
    else:
        print(f"View results with: mlflow ui --backend-store-uri {actual_uri}")


if __name__ == "__main__":
    main()