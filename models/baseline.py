"""
Baseline volatility detection model using rule-based approach.
This serves as the benchmark for ML models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


class BaselineVolatilityDetector:
    """
    Rule-based baseline model for volatility spike detection.
    Uses z-score threshold on rolling volatility.
    """
    
    def __init__(self, threshold: float = 2.0):
        """
        Args:
            threshold: Z-score threshold for spike detection (default: 2.0)
        """
        self.threshold = threshold
        self.mean_ = None
        self.std_ = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Compute historical statistics for z-score calculation.
        
        Args:
            X: Features dataframe with 'price_volatility_5min' column
            y: Target labels (not used for baseline, but kept for API consistency)
        """
        if 'price_volatility_5min' not in X.columns:
            raise ValueError("X must contain 'price_volatility_5min' column")
        
        # Compute mean and std from training data
        volatility = X['price_volatility_5min'].dropna()
        self.mean_ = volatility.mean()
        self.std_ = volatility.std()
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict volatility spikes using z-score threshold.
        
        Args:
            X: Features dataframe
            
        Returns:
            Binary predictions (1 = spike, 0 = normal)
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        volatility = X['price_volatility_5min'].fillna(self.mean_)
        
        # Compute z-scores
        z_scores = (volatility - self.mean_) / (self.std_ + 1e-8)
        
        # Threshold to binary predictions
        predictions = (z_scores >= self.threshold).astype(int)
        
        return predictions.values
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return probability-like scores based on z-score distance.
        
        Args:
            X: Features dataframe
            
        Returns:
            Array of shape (n_samples, 2) with probabilities for [normal, spike]
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        volatility = X['price_volatility_5min'].fillna(self.mean_)
        z_scores = (volatility - self.mean_) / (self.std_ + 1e-8)
        
        # Convert z-scores to pseudo-probabilities using sigmoid
        spike_proba = 1 / (1 + np.exp(-z_scores))
        normal_proba = 1 - spike_proba
        
        return np.column_stack([normal_proba, spike_proba])
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Features dataframe
            y: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        # Compute metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, average='binary', zero_division=0
        )
        
        try:
            roc_auc = roc_auc_score(y, y_proba)
        except ValueError:
            roc_auc = 0.0
        
        try:
            pr_auc = average_precision_score(y, y_proba)
        except ValueError:
            pr_auc = 0.0
        
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        
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
            'threshold': self.threshold
        }