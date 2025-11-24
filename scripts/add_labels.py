"""
Add volatility_spike labels to features dataframe.
Computes future volatility and applies threshold to create binary labels.
Uses chunk-aware calculation to avoid cross-dataset volatility calculations.
"""

import argparse
import sys
from pathlib import Path

# Import from featurizer to use the chunk-aware label creation
sys.path.insert(0, str(Path(__file__).parent.parent))
from features.featurizer import add_labels_to_file


def main():
    parser = argparse.ArgumentParser(
        description="Add volatility_spike labels to features. "
        "Only calculates volatility within each data chunk (separated by gaps)."
    )
    parser.add_argument(
        "--features",
        required=True,
        help="Path to features parquet file (without labels)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for labeled features (default: adds '_labeled' suffix)",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=int,
        default=90,
        help="Percentile to use as threshold (default: 90)",
    )
    parser.add_argument(
        "--gap-threshold-seconds",
        type=int,
        default=300,
        help="Time gap (seconds) that indicates a new data chunk (default: 300 = 5 min)",
    )

    args = parser.parse_args()

    add_labels_to_file(
        args.features,
        args.output,
        args.threshold_percentile,
        args.gap_threshold_seconds,
    )


if __name__ == "__main__":
    main()
