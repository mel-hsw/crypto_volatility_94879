"""
Comprehensive feature engineering and selection for Random Forest model.
Analyzes feature importance, correlation, and selects optimal feature set.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Dict
import warnings

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
    RFE,
    RFECV,
)
from sklearn.preprocessing import StandardScaler
import joblib


def load_data(features_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load features and extract target."""
    df = pd.read_parquet(features_path)

    if "volatility_spike" not in df.columns:
        raise ValueError("Missing 'volatility_spike' column. Add labels first.")

    # Get all feature columns (exclude metadata AND target variables)
    exclude_cols = [
        # Metadata columns
        "timestamp",
        "product_id",
        "price",
        "best_bid",
        "best_ask",
        "spread",
        "spread_bps",
        "time_since_last_trade",
        "gap_seconds",
        # Target variables (CRITICAL - these cause data leakage!)
        "volatility_spike",  # The actual target
        "label",  # Alternative target name
        "future_volatility",  # What we're predicting (forward-looking!)
        # Processing metadata (not features)
        "chunk_id",
        "time_diff",
        # Derived/duplicate features that shouldn't be used
        "price_return_1min",
        "price_return_5min",
        "price_volatility_5min",
        "bid_ask_spread",
        "bid_ask_spread_bps",  # If these are duplicates
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].copy()
    y = df["volatility_spike"].copy()

    # Fill NaN
    X = X.fillna(0)

    return X, y, feature_cols


def analyze_feature_importance(
    X: pd.DataFrame, y: pd.Series, n_estimators: int = 100, random_state: int = 42
) -> pd.DataFrame:
    """
    Train Random Forest and extract feature importance.

    Returns DataFrame with feature importance scores.
    """
    print("\n=== Random Forest Feature Importance ===")

    # Calculate class weights for imbalance
    class_weight = "balanced"

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )

    rf.fit(X, y)

    # Get feature importance
    importance_df = pd.DataFrame(
        {"feature": X.columns, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)

    print(f"\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))

    return importance_df, rf


def analyze_correlation(X: pd.DataFrame, threshold: float = 0.8) -> Dict:
    """
    Analyze feature correlations and identify highly correlated pairs.

    Returns dictionary with correlation analysis results.
    """
    print("\n=== Correlation Analysis ===")

    corr_matrix = X.corr().abs()

    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                high_corr_pairs.append(
                    {
                        "feature1": corr_matrix.columns[i],
                        "feature2": corr_matrix.columns[j],
                        "correlation": corr_matrix.iloc[i, j],
                    }
                )

    if high_corr_pairs:
        print(f"\nFound {len(high_corr_pairs)} pairs with correlation > {threshold}:")
        high_corr_df = pd.DataFrame(high_corr_pairs).sort_values(
            "correlation", ascending=False
        )
        print(high_corr_df.head(10).to_string(index=False))
    else:
        print(f"\nNo feature pairs with correlation > {threshold}")

    return {"correlation_matrix": corr_matrix, "high_corr_pairs": high_corr_pairs}


def univariate_feature_selection(
    X: pd.DataFrame, y: pd.Series, k: int = 15
) -> List[str]:
    """
    Select top k features using univariate statistical tests.

    Uses f_classif (ANOVA F-value) and mutual information.
    """
    print(f"\n=== Univariate Feature Selection (Top {k}) ===")

    # F-test
    selector_f = SelectKBest(score_func=f_classif, k=k)
    selector_f.fit(X, y)

    f_scores = pd.DataFrame(
        {"feature": X.columns, "f_score": selector_f.scores_}
    ).sort_values("f_score", ascending=False)

    # Mutual Information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({"feature": X.columns, "mi_score": mi_scores}).sort_values(
        "mi_score", ascending=False
    )

    print("\nTop features by F-test:")
    print(f_scores.head(k).to_string(index=False))

    print("\nTop features by Mutual Information:")
    print(mi_df.head(k).to_string(index=False))

    # Combine: take features that appear in top k of either method
    top_f = set(f_scores.head(k)["feature"])
    top_mi = set(mi_df.head(k)["feature"])
    combined_top = list(top_f | top_mi)

    return combined_top, f_scores, mi_df


def recursive_feature_elimination(
    X: pd.DataFrame, y: pd.Series, n_features: int = 15, cv_folds: int = 5
) -> Tuple[List[str], pd.DataFrame]:
    """
    Use RFE with cross-validation to select optimal features.
    """
    print(f"\n=== Recursive Feature Elimination (RFE) ===")

    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # Use RFECV for automatic feature count selection
    rfecv = RFECV(
        estimator=rf,
        step=1,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
        scoring="average_precision",
        n_jobs=-1,
    )

    rfecv.fit(X, y)

    selected_features = X.columns[rfecv.support_].tolist()

    print(f"\nOptimal number of features: {rfecv.n_features_}")
    print(f"Selected features ({len(selected_features)}):")
    for feat in selected_features:
        print(f"  - {feat}")

    # Create ranking DataFrame
    ranking_df = pd.DataFrame(
        {"feature": X.columns, "rank": rfecv.ranking_, "selected": rfecv.support_}
    ).sort_values("rank")

    return selected_features, ranking_df


def evaluate_feature_set(
    X: pd.DataFrame, y: pd.Series, feature_set: List[str], cv_folds: int = 5
) -> Dict:
    """
    Evaluate a feature set using cross-validation with Random Forest.

    Returns dictionary with performance metrics.
    """
    X_subset = X[feature_set].copy()

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    pr_auc_scorer = make_scorer(average_precision_score)

    scores = cross_val_score(rf, X_subset, y, cv=cv, scoring=pr_auc_scorer, n_jobs=-1)

    return {
        "n_features": len(feature_set),
        "mean_pr_auc": scores.mean(),
        "std_pr_auc": scores.std(),
        "features": feature_set,
    }


def compare_feature_sets(
    X: pd.DataFrame, y: pd.Series, feature_sets: Dict[str, List[str]], cv_folds: int = 5
) -> pd.DataFrame:
    """
    Compare multiple feature sets using cross-validation.

    Returns DataFrame with comparison results.
    """
    print("\n=== Comparing Feature Sets ===")

    results = []
    for name, features in feature_sets.items():
        print(f"\nEvaluating: {name} ({len(features)} features)...")
        result = evaluate_feature_set(X, y, features, cv_folds)
        result["feature_set"] = name
        results.append(result)

    comparison_df = pd.DataFrame(results).sort_values("mean_pr_auc", ascending=False)

    print("\n" + "=" * 60)
    print("Feature Set Comparison (Cross-Validated PR-AUC):")
    print("=" * 60)
    print(
        comparison_df[
            ["feature_set", "n_features", "mean_pr_auc", "std_pr_auc"]
        ].to_string(index=False)
    )

    return comparison_df


def create_visualizations(
    importance_df: pd.DataFrame, corr_matrix: pd.DataFrame, output_dir: Path
):
    """Create visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Feature importance plot
    plt.figure(figsize=(10, 8))
    top_n = min(20, len(importance_df))
    top_features = importance_df.head(top_n)
    plt.barh(range(len(top_features)), top_features["importance"])
    plt.yticks(range(len(top_features)), top_features["feature"])
    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_n} Feature Importances (Random Forest)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close()

    # Correlation heatmap (top features)
    top_features_list = importance_df.head(15)["feature"].tolist()
    corr_subset = corr_matrix.loc[top_features_list, top_features_list]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_subset,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
    )
    plt.title("Correlation Matrix (Top 15 Features)")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close()

    print(f"\nVisualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Feature engineering and selection")
    parser.add_argument(
        "--features",
        default="data/processed/features_labeled.parquet",
        help="Path to features parquet file",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/feature_analysis",
        help="Output directory for results",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees for Random Forest",
    )
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds")

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    X, y, feature_cols = load_data(args.features)
    print(f"Loaded {len(X)} samples with {len(feature_cols)} features")
    print(f"Class distribution: {y.value_counts().to_dict()}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Feature Importance Analysis
    importance_df, rf_model = analyze_feature_importance(X, y, args.n_estimators)
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    # 2. Correlation Analysis
    corr_results = analyze_correlation(X, threshold=0.8)

    # 3. Univariate Feature Selection
    univariate_features, f_scores, mi_scores = univariate_feature_selection(X, y, k=15)

    # 4. Recursive Feature Elimination
    rfe_features, rfe_ranking = recursive_feature_elimination(
        X, y, cv_folds=args.cv_folds
    )
    rfe_ranking.to_csv(output_dir / "rfe_ranking.csv", index=False)

    # 5. Compare Different Feature Sets
    feature_sets = {
        "all_features": list(X.columns),
        "top_10_importance": importance_df.head(10)["feature"].tolist(),
        "top_15_importance": importance_df.head(15)["feature"].tolist(),
        "top_20_importance": importance_df.head(20)["feature"].tolist(),
        "univariate_selection": univariate_features,
        "rfe_selection": rfe_features,
    }

    comparison_df = compare_feature_sets(X, y, feature_sets, args.cv_folds)
    comparison_df.to_csv(output_dir / "feature_set_comparison.csv", index=False)

    # 6. Create Visualizations
    create_visualizations(importance_df, corr_results["correlation_matrix"], output_dir)

    # 7. Save recommended feature set
    best_set = comparison_df.iloc[0]
    recommended_features = feature_sets[best_set["feature_set"]]

    print("\n" + "=" * 60)
    print("RECOMMENDED FEATURE SET:")
    print("=" * 60)
    print(f"Feature Set: {best_set['feature_set']}")
    print(f"Number of Features: {best_set['n_features']}")
    print(
        f"Cross-Validated PR-AUC: {best_set['mean_pr_auc']:.4f} ± {best_set['std_pr_auc']:.4f}"
    )
    print(f"\nFeatures:")
    for feat in recommended_features:
        print(f"  - {feat}")

    # Save recommended features
    with open(output_dir / "recommended_features.txt", "w") as f:
        f.write("# Recommended Feature Set\n")
        f.write(
            f"# PR-AUC: {best_set['mean_pr_auc']:.4f} ± {best_set['std_pr_auc']:.4f}\n"
        )
        f.write(f"# Number of features: {best_set['n_features']}\n\n")
        for feat in recommended_features:
            f.write(f"{feat}\n")

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
