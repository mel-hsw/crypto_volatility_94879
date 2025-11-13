"""
Process raw data files through the featurizer pipeline.
This script duplicates what the featurizer does but on file data instead of Kafka streaming.

Reads NDJSON files from data/raw, computes features using FeatureComputer,
and optionally adds volatility_spike labels using the same logic as FeaturePipeline.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path to import features module
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.featurizer import FeatureComputer, FeaturePipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_raw_data(raw_dir: str = None, file_pattern: str = None) -> list:
    """
    Load raw tick data from NDJSON files in the raw data directory.
    
    Args:
        raw_dir: Directory containing raw NDJSON files (default: data/raw)
        file_pattern: Optional glob pattern to filter files (e.g., '*.ndjson')
        
    Returns:
        List of tick dictionaries, sorted by timestamp
    """
    if raw_dir is None:
        raw_dir = Path(__file__).parent.parent / 'data' / 'raw'
    else:
        raw_dir = Path(raw_dir)
    
    if not raw_dir.exists():
        logger.error(f"Raw data directory not found: {raw_dir}")
        return []
    
    # Find all NDJSON files
    if file_pattern:
        files = list(raw_dir.glob(file_pattern))
    else:
        files = list(raw_dir.glob('*.ndjson'))
    
    if not files:
        logger.warning(f"No NDJSON files found in {raw_dir}")
        return []
    
    files = sorted(files)  # Process in order
    logger.info(f"Found {len(files)} file(s) to process")
    
    ticks = []
    
    for filepath in files:
        logger.info(f"Loading {filepath.name}...")
        
        try:
            with open(filepath, 'r') as f:
                file_ticks = 0
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        tick = json.loads(line)
                        ticks.append(tick)
                        file_ticks += 1
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num} in {filepath.name}: {e}")
                        continue
                
                logger.info(f"  Loaded {file_ticks} ticks from {filepath.name}")
        except Exception as e:
            logger.error(f"Error reading {filepath.name}: {e}")
            continue
    
    logger.info(f"Loaded {len(ticks)} total ticks from {len(files)} file(s)")
    
    # Sort by timestamp to ensure correct ordering
    try:
        ticks = sorted(ticks, key=lambda t: pd.to_datetime(t.get('timestamp', t.get('time', 0))))
        logger.info("Sorted ticks by timestamp")
    except Exception as e:
        logger.warning(f"Could not sort ticks by timestamp: {e}")
    
    return ticks


def compute_features_from_ticks(ticks: list, window_sizes: list = [30, 60, 300]) -> pd.DataFrame:
    """
    Compute features from ticks using FeatureComputer (same as featurizer).
    
    Args:
        ticks: List of tick dictionaries
        window_sizes: Window sizes in seconds for feature computation
        
    Returns:
        DataFrame of computed features
    """
    feature_computer = FeatureComputer(window_sizes=window_sizes)
    features_list = []
    
    logger.info(f"Computing features for {len(ticks)} ticks...")
    logger.info(f"Using window sizes: {window_sizes}s")
    
    for i, tick in enumerate(ticks):
        # Add tick to buffer
        feature_computer.add_tick(tick)
        
        # Compute features (same logic as FeaturePipeline.process_message)
        features = feature_computer.compute_features(tick)
        features_list.append(features)
        
        # Progress logging
        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i + 1}/{len(ticks)} ticks ({100*(i+1)/len(ticks):.1f}%)")
    
    logger.info(f"✓ Computed features for {len(features_list)} ticks")
    
    # Convert to DataFrame
    df = pd.DataFrame(features_list)
    
    # Ensure timestamp is datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df


def process_raw_files(
    raw_dir: str = None,
    output_file: str = 'data/processed/features_from_files.parquet',
    window_sizes: list = [30, 60, 300],
    add_labels: bool = True,
    label_threshold_percentile: int = 90,
    label_gap_threshold_seconds: int = 300,
    file_pattern: str = None
) -> pd.DataFrame:
    """
    Process raw data files through the featurizer pipeline.
    
    This function duplicates what FeaturePipeline does but on file data:
    1. Loads NDJSON files from raw_dir
    2. Computes features using FeatureComputer
    3. Optionally adds volatility_spike labels using FeaturePipeline logic
    
    Args:
        raw_dir: Directory containing raw NDJSON files (default: data/raw)
        output_file: Path to output parquet file
        window_sizes: Window sizes in seconds for feature computation
        add_labels: Whether to add volatility_spike labels
        label_threshold_percentile: Percentile to use as threshold for labels
        label_gap_threshold_seconds: Time gap (seconds) that indicates a new data chunk
        file_pattern: Optional glob pattern to filter files
        
    Returns:
        DataFrame with computed features (and labels if add_labels=True)
    """
    # Load raw data
    ticks = load_raw_data(raw_dir, file_pattern)
    
    if not ticks:
        logger.error("No ticks loaded. Exiting.")
        return pd.DataFrame()
    
    # Compute features (same as featurizer)
    df = compute_features_from_ticks(ticks, window_sizes=window_sizes)
    
    logger.info(f"Features computed: {len(df)} rows, {len(df.columns)} columns")
    
    # Add labels if requested (same logic as FeaturePipeline)
    if add_labels:
        logger.info("Adding volatility_spike labels...")
        df = FeaturePipeline._add_labels_to_dataframe(
            df,
            threshold_percentile=label_threshold_percentile,
            gap_threshold_seconds=label_gap_threshold_seconds
        )
        logger.info(f"Labels added: {len(df)} rows")
    
    # Save to parquet file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(output_path, index=False)
    logger.info(f"✓ Saved features to {output_path}")
    logger.info(f"  Shape: {df.shape}")
    
    if add_labels and 'volatility_spike' in df.columns:
        spike_rate = df['volatility_spike'].mean()
        spike_count = df['volatility_spike'].sum()
        logger.info(f"  Spike rate: {spike_rate:.2%} ({spike_count} spikes out of {len(df)} total)")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Process raw data files through featurizer pipeline. '
                    'Duplicates what the featurizer does but on file data instead of Kafka.'
    )
    parser.add_argument('--raw-dir', default='data/raw',
                       help='Directory containing raw NDJSON files (default: data/raw)')
    parser.add_argument('--output', default='data/processed/features_from_files.parquet',
                       help='Output parquet file (default: data/processed/features_from_files.parquet)')
    parser.add_argument('--windows', nargs='+', type=int, default=[30, 60, 300],
                       help='Window sizes in seconds (default: 30 60 300)')
    parser.add_argument('--add-labels', action='store_true', default=True,
                       help='Add volatility_spike labels (default: True)')
    parser.add_argument('--no-labels', dest='add_labels', action='store_false',
                       help='Do not add labels')
    parser.add_argument('--label-threshold-percentile', type=int, default=90,
                       help='Percentile to use as threshold for labels (default: 90)')
    parser.add_argument('--label-gap-threshold-seconds', type=int, default=300,
                       help='Time gap (seconds) that indicates a new data chunk (default: 300 = 5 min)')
    parser.add_argument('--file-pattern', default=None,
                       help='Optional glob pattern to filter files (e.g., "*20251108*.ndjson")')
    
    args = parser.parse_args()
    
    # Process files
    df = process_raw_files(
        raw_dir=args.raw_dir,
        output_file=args.output,
        window_sizes=args.windows,
        add_labels=args.add_labels,
        label_threshold_percentile=args.label_threshold_percentile,
        label_gap_threshold_seconds=args.label_gap_threshold_seconds,
        file_pattern=args.file_pattern
    )
    
    if df.empty:
        logger.error("No data processed. Exiting.")
        sys.exit(1)
    
    logger.info("✓ Processing complete!")


if __name__ == '__main__':
    main()

