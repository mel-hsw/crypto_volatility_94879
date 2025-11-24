#!/usr/bin/env python3
"""
Consolidate all available feature files into a single dataset with balanced splits.

This script:
1. Loads all available feature parquet files
2. Removes duplicates (based on timestamp)
3. Sorts by timestamp
4. Creates stratified train/val/test splits with balanced spike rates
5. Saves consolidated dataset
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys


def consolidate_features(
    input_files: list,
    output_path: str,
    remove_duplicates: bool = True,
    min_timestamp_gap_seconds: float = 0.1,
) -> pd.DataFrame:
    """
    Consolidate multiple feature files into one dataset.

    Args:
        input_files: List of paths to feature parquet files
        output_path: Path to save consolidated dataset
        remove_duplicates: Whether to remove duplicate timestamps
        min_timestamp_gap_seconds: Minimum gap between timestamps to consider unique

    Returns:
        Consolidated DataFrame
    """
    print("=" * 60)
    print("Consolidating Feature Files")
    print("=" * 60)

    all_dfs = []

    for file_path in input_files:
        path = Path(file_path)
        if not path.exists():
            print(f"⚠ Skipping {path.name}: File not found")
            continue

        try:
            df = pd.read_parquet(path)

            # Validate required columns
            if "volatility_spike" not in df.columns:
                print(f"⚠ Skipping {path.name}: Missing 'volatility_spike' column")
                continue
            if "timestamp" not in df.columns:
                print(f"⚠ Skipping {path.name}: Missing 'timestamp' column")
                continue

            # Convert timestamp to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"])

            print(
                f"✓ Loaded {path.name}: {len(df)} samples, spike_rate: {df['volatility_spike'].mean():.2%}"
            )
            all_dfs.append(df)

        except Exception as e:
            print(f"✗ Error loading {path.name}: {e}")
            continue

    if not all_dfs:
        raise ValueError("No valid feature files found!")

    # Concatenate all dataframes
    print(f"\nConcatenating {len(all_dfs)} files...")
    consolidated = pd.concat(all_dfs, ignore_index=True)
    print(f"Total samples before deduplication: {len(consolidated)}")

    # Remove duplicates based on timestamp
    if remove_duplicates:
        print("\nRemoving duplicates...")
        # Sort by timestamp first
        consolidated = consolidated.sort_values("timestamp").reset_index(drop=True)

        # Remove exact duplicates (same timestamp and all features)
        before_dedup = len(consolidated)
        consolidated = consolidated.drop_duplicates(subset=["timestamp"], keep="first")
        duplicates_removed = before_dedup - len(consolidated)
        print(f"Removed {duplicates_removed} duplicate timestamps")

        # Also remove rows that are too close in time (within min_timestamp_gap_seconds)
        if min_timestamp_gap_seconds > 0:
            time_diff = consolidated["timestamp"].diff().dt.total_seconds()
            too_close = time_diff < min_timestamp_gap_seconds
            too_close[0] = False  # First row has no previous row
            before_close = len(consolidated)
            consolidated = consolidated[~too_close].reset_index(drop=True)
            close_removed = before_close - len(consolidated)
            print(
                f"Removed {close_removed} rows too close in time (< {min_timestamp_gap_seconds}s)"
            )

    # Final sort by timestamp
    consolidated = consolidated.sort_values("timestamp").reset_index(drop=True)

    # Summary statistics
    print(f"\n{'='*60}")
    print("Consolidated Dataset Summary")
    print(f"{'='*60}")
    print(f"Total samples: {len(consolidated)}")
    print(
        f"Time range: {consolidated['timestamp'].min()} to {consolidated['timestamp'].max()}"
    )
    print(
        f"Duration: {(consolidated['timestamp'].max() - consolidated['timestamp'].min()).total_seconds() / 3600:.2f} hours"
    )
    print(f"Overall spike rate: {consolidated['volatility_spike'].mean():.2%}")
    print(f"Spikes: {consolidated['volatility_spike'].sum()}")
    print(f"Non-spikes: {(consolidated['volatility_spike'] == 0).sum()}")

    # Save consolidated dataset
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    consolidated.to_parquet(output_path, index=False)
    print(f"\n✓ Saved consolidated dataset to: {output_path}")

    return consolidated


def create_stratified_splits(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple:
    """
    Create stratified train/val/test splits with balanced spike rates.

    Args:
        df: Consolidated DataFrame
        val_size: Fraction for validation set
        test_size: Fraction for test set
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    print(f"\n{'='*60}")
    print("Creating Stratified Splits")
    print(f"{'='*60}")

    # First split: train vs (val+test)
    train_size = 1 - val_size - test_size
    temp_size = val_size + test_size

    # Use stratified split to maintain spike rate
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        stratify=df["volatility_spike"],
        random_state=random_state,
    )

    # Second split: val vs test
    val_ratio = val_size / temp_size
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio),
        stratify=temp_df["volatility_spike"],
        random_state=random_state,
    )

    # Report split statistics
    print(f"\nSplit Statistics:")
    print(
        f"{'Set':<12} {'Samples':<10} {'Spike Rate':<12} {'Spikes':<10} {'Non-Spikes':<10}"
    )
    print("-" * 60)

    for name, split_df in [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test", test_df),
    ]:
        spike_rate = split_df["volatility_spike"].mean()
        spikes = split_df["volatility_spike"].sum()
        non_spikes = (split_df["volatility_spike"] == 0).sum()
        print(
            f"{name:<12} {len(split_df):<10} {spike_rate:<12.2%} {spikes:<10} {non_spikes:<10}"
        )

    # Verify spike rates are balanced
    train_rate = train_df["volatility_spike"].mean()
    val_rate = val_df["volatility_spike"].mean()
    test_rate = test_df["volatility_spike"].mean()

    max_diff = max(
        abs(train_rate - val_rate),
        abs(train_rate - test_rate),
        abs(val_rate - test_rate),
    )
    print(f"\nMax spike rate difference: {max_diff:.2%}")

    if max_diff < 0.01:  # Less than 1% difference
        print("✓ Splits are well-balanced!")
    else:
        print("⚠ Warning: Spike rates differ significantly across splits")

    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate feature files and create balanced splits"
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=[
            "data/processed/features_replay.parquet",
            "data/processed/features_long_20251124_024939.parquet",
            "data/processed/features_combined.parquet",
            "data/processed/features_all_raw.parquet",
            "data/processed/features.parquet",
        ],
        help="Input feature parquet files to consolidate",
    )
    parser.add_argument(
        "--output",
        default="data/processed/features_consolidated.parquet",
        help="Output path for consolidated dataset",
    )
    parser.add_argument(
        "--splits",
        action="store_true",
        help="Also create stratified train/val/test splits",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Validation set size (default: 0.15)",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.15, help="Test set size (default: 0.15)"
    )
    parser.add_argument(
        "--no-dedup", action="store_true", help="Don't remove duplicate timestamps"
    )

    args = parser.parse_args()

    try:
        # Consolidate files
        consolidated = consolidate_features(
            args.input, args.output, remove_duplicates=not args.no_dedup
        )

        # Create stratified splits if requested
        if args.splits:
            train_df, val_df, test_df = create_stratified_splits(
                consolidated, val_size=args.val_size, test_size=args.test_size
            )

            # Save splits
            output_dir = Path(args.output).parent
            train_path = output_dir / "features_consolidated_train.parquet"
            val_path = output_dir / "features_consolidated_val.parquet"
            test_path = output_dir / "features_consolidated_test.parquet"

            train_df.to_parquet(train_path, index=False)
            val_df.to_parquet(val_path, index=False)
            test_df.to_parquet(test_path, index=False)

            print(f"\n✓ Saved splits:")
            print(f"  Train: {train_path}")
            print(f"  Validation: {val_path}")
            print(f"  Test: {test_path}")

        print(f"\n{'='*60}")
        print("✓ Consolidation complete!")
        print(f"{'='*60}")

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
