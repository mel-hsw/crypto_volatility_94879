"""
Add volatility_spike labels to features dataframe.
Computes future volatility and applies threshold to create binary labels.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def add_labels(features_path: str, output_path: str = None, threshold_percentile: int = 90):
    """
    Add volatility_spike labels to features dataframe.
    
    Args:
        features_path: Path to features parquet file (without labels)
        output_path: Path to save labeled features (default: adds '_labeled' suffix)
        threshold_percentile: Percentile to use as threshold (default: 90)
    """
    print(f"Loading features from {features_path}")
    df = pd.read_parquet(features_path)
    
    # Convert timestamp to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"Loaded {len(df)} rows")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Sort by timestamp to ensure correct ordering
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Compute forward-looking volatility (60-second horizon)
    HORIZON_SECONDS = 60
    
    # Calculate rolling forward volatility
    df['price_pct_change'] = df['price'].pct_change()
    
    # Estimate number of ticks in 60 seconds
    time_diff = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
    ticks_per_second = len(df) / time_diff if time_diff > 0 else 1.0
    window_size = max(1, int(ticks_per_second * HORIZON_SECONDS))
    
    print(f"Average ticks per second: {ticks_per_second:.2f}")
    print(f"Window size for {HORIZON_SECONDS}s: {window_size} ticks")
    
    # Compute forward-looking volatility (shifted backwards)
    df['future_volatility'] = df['price_pct_change'].shift(-window_size).rolling(window=window_size).std()
    
    # Drop NaN values at the end
    df_clean = df.dropna(subset=['future_volatility']).copy()
    
    print(f"After computing future volatility: {len(df_clean)} valid rows")
    
    # Calculate threshold
    THRESHOLD = np.percentile(df_clean['future_volatility'], threshold_percentile)
    print(f"\nSelected Threshold: {THRESHOLD:.6f} ({threshold_percentile}th percentile)")
    
    # Create binary labels
    df_clean['volatility_spike'] = (df_clean['future_volatility'] >= THRESHOLD).astype(int)
    
    # Class distribution
    label_counts = df_clean['volatility_spike'].value_counts()
    print(f"\nClass Distribution:")
    print(f"  No Spike (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(df_clean)*100:.1f}%)")
    print(f"  Spike (1):    {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(df_clean)*100:.1f}%)")
    
    # Determine output path
    if output_path is None:
        features_path_obj = Path(features_path)
        output_path = features_path_obj.parent / f"{features_path_obj.stem}_labeled.parquet"
    
    # Save labeled features
    df_clean.to_parquet(output_path, index=False)
    print(f"\n✓ Saved labeled dataset to {output_path}")
    print(f"  Shape: {df_clean.shape}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Add volatility_spike labels to features")
    parser.add_argument("--features", required=True,
                       help="Path to features parquet file (without labels)")
    parser.add_argument("--output", default=None,
                       help="Output path for labeled features (default: adds '_labeled' suffix)")
    parser.add_argument("--threshold-percentile", type=int, default=90,
                       help="Percentile to use as threshold (default: 90)")
    
    args = parser.parse_args()
    
    add_labels(args.features, args.output, args.threshold_percentile)


if __name__ == "__main__":
    main()

