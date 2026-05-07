import pandas as pd
import os
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_csv_to_parquet(csv_path, output_dir=None):
    """Converts a single CSV file to Parquet."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.error(f"File not found: {csv_path}")
        return

    if output_dir is None:
        output_dir = csv_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / f"{csv_path.stem}.parquet"
    
    logger.info(f"Converting {csv_path} to {parquet_path}...")
    
    # Load and ensure Datetime is correctly handled
    df = pd.read_csv(csv_path)
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        # It's often better to keep Datetime as a column for Parquet, 
        # but the train script handles both.
    
    # Parquet preserves types and is much faster to load
    df.to_parquet(parquet_path, engine='pyarrow', index=False)
    
    csv_size = csv_path.stat().st_size / (1024 * 1024)
    pq_size = parquet_path.stat().st_size / (1024 * 1024)
    
    logger.info(f"Done! Size: {csv_size:.2f}MB -> {pq_size:.2f}MB ({pq_size/csv_size:.1%})")

def main():
    parser = argparse.ArgumentParser(description="Convert CSV energy data to Parquet.")
    parser.add_argument("path", help="Path to a CSV file or a directory containing CSVs.")
    parser.add_argument("--output_dir", help="Optional output directory for Parquet files.")
    args = parser.parse_args()

    target_path = Path(args.path)
    
    if target_path.is_file():
        if target_path.suffix.lower() == '.csv':
            convert_csv_to_parquet(target_path, args.output_dir)
        else:
            logger.error("Provided file is not a CSV.")
    elif target_path.is_dir():
        csv_files = list(target_path.glob("*.csv"))
        if not csv_files:
            logger.warning(f"No CSV files found in {target_path}")
            return
        
        logger.info(f"Found {len(csv_files)} CSV files. Starting batch conversion...")
        for csv_file in csv_files:
            convert_csv_to_parquet(csv_file, args.output_dir)
    else:
        logger.error(f"Invalid path: {target_path}")

if __name__ == "__main__":
    main()
