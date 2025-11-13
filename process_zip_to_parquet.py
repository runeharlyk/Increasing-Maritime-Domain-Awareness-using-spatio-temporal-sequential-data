import polars as pl
from pathlib import Path
import time
import zipfile
import io
import data_config as config

def clean_dataframe(df):
    if df.is_empty():
        return df
    
    print(f"  Initial rows: {len(df)}")
    
    print("  Applying geographic bounding box filter...")
    north, west, south, east = config.BOUNDING_BOX
    before = len(df)
    df = df.filter(
        (pl.col("Latitude") <= north)
        & (pl.col("Latitude") >= south)
        & (pl.col("Longitude") >= west)
        & (pl.col("Longitude") <= east)
    )
    print(f"  Removed {before - len(df)} rows outside bounding box. Rows now: {len(df)}")
    
    print("  Filtering by vessel class...")
    before = len(df)
    df = df.filter(pl.col("Type of mobile").is_in(config.VESSEL_CLASSES))
    print(f"  Kept {'/'.join(config.VESSEL_CLASSES)} vessels. Rows now: {len(df)}")
    
    print("  Validating MMSI format...")
    before = len(df)
    df = df.filter(pl.col("MMSI").str.len_chars() == config.MMSI_LENGTH)
    df = df.filter(
        pl.col("MMSI").str.slice(0, 3).cast(pl.Int32).is_between(config.MMSI_MID_MIN, config.MMSI_MID_MAX)
    )
    print(f"  Removed {before - len(df)} rows with invalid MMSI. Rows now: {len(df)}")
    
    print("  Parsing timestamps...")
    if "# Timestamp" in df.columns:
        df = df.rename({"# Timestamp": "Timestamp"})
    df = df.with_columns(
        pl.col("Timestamp")
        .str.to_datetime(format="%d/%m/%Y %H:%M:%S", strict=False)
        .alias("Timestamp")
    )
    timestamp_na = df.filter(pl.col("Timestamp").is_null()).height
    if timestamp_na > 0:
        print(f"  Removing {timestamp_na} rows with invalid timestamps")
        df = df.filter(pl.col("Timestamp").is_not_null())
    print(f"  Timestamp range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    
    print("  Removing duplicates...")
    before = len(df)
    df = df.unique(subset=["Timestamp", "MMSI"], keep="first")
    print(f"  Removed {before - len(df)} duplicate rows. Rows now: {len(df)}")
    
    print(f"  Applying track filtering (length > {config.TRACK_MIN_LENGTH}, SOG {config.TRACK_MIN_SOG}-{config.TRACK_MAX_SOG} knots, timespan >= {config.TRACK_MIN_TIMESPAN/3600:.0f} hour)...")
    before = len(df)
    unique_before = df["MMSI"].n_unique()
    track_stats = df.group_by("MMSI").agg([
        pl.len().alias("count"),
        pl.col("SOG").max().alias("max_sog"),
        (pl.col("Timestamp").max() - pl.col("Timestamp").min()).dt.total_seconds().alias("timespan_seconds")
    ])
    valid_tracks = track_stats.filter(
        (pl.col("count") > config.TRACK_MIN_LENGTH) &
        (pl.col("max_sog") >= config.TRACK_MIN_SOG) &
        (pl.col("max_sog") <= config.TRACK_MAX_SOG) &
        (pl.col("timespan_seconds") >= config.TRACK_MIN_TIMESPAN)
    ).select("MMSI")
    df = df.join(valid_tracks, on="MMSI", how="inner")
    df = df.sort(["MMSI", "Timestamp"])
    print(f"  Removed {before - len(df)} rows from invalid tracks. Rows now: {len(df)}")
    print(f"  Unique vessels: {unique_before} -> {df['MMSI'].n_unique()}")
    
    print(f"  Creating segments based on time gaps (>= {config.SEGMENT_TIME_GAP/60:.0f} minutes)...")
    unique_mmsi = df["MMSI"].unique().to_list()
    
    segment_dfs = []
    for mmsi in unique_mmsi:
        mmsi_df = df.filter(pl.col("MMSI") == mmsi)
        
        mmsi_df = mmsi_df.with_columns([
            pl.col("Timestamp").diff().fill_null(pl.duration(seconds=0)).alias("time_diff")
        ])
        
        mmsi_df = mmsi_df.with_columns([
            (pl.col("time_diff").dt.total_seconds() >= config.SEGMENT_TIME_GAP).cast(pl.Int32).alias("segment_marker")
        ])
        
        mmsi_df = mmsi_df.with_columns([
            pl.col("segment_marker").cum_sum().alias("Segment")
        ])
        
        mmsi_df = mmsi_df.drop(["time_diff", "segment_marker"])
        segment_dfs.append(mmsi_df)
    
    df = pl.concat(segment_dfs, how="vertical")
    df = df.sort(["MMSI", "Timestamp"])
    
    print("  Applying segment filtering...")
    before = len(df)
    segment_stats = df.group_by(["MMSI", "Segment"]).agg([
        pl.len().alias("count"),
        pl.col("SOG").max().alias("max_sog"),
        (pl.col("Timestamp").max() - pl.col("Timestamp").min()).dt.total_seconds().alias("timespan_seconds")
    ])
    valid_segments = segment_stats.filter(
        (pl.col("count") > config.TRACK_MIN_LENGTH) &
        (pl.col("max_sog") >= config.TRACK_MIN_SOG) &
        (pl.col("max_sog") <= config.TRACK_MAX_SOG) &
        (pl.col("timespan_seconds") >= config.TRACK_MIN_TIMESPAN)
    ).select(["MMSI", "Segment"])
    df = df.join(valid_segments, on=["MMSI", "Segment"], how="inner")
    print(f"  Removed {before - len(df)} rows from invalid segments. Rows now: {len(df)}")
    
    print("  Converting SOG from knots to m/s...")
    df = df.with_columns((pl.col("SOG") * config.KNOTS_TO_MS).alias("SOG"))
    
    if "Ship type" in df.columns:
        print("  Filtering by ship type...")
        before = len(df)
        df = df.filter(pl.col("Ship type").is_in(config.SHIP_TYPES))
        print(f"  Kept {'/'.join(config.SHIP_TYPES)} only. Removed {before - len(df)} rows. Rows now: {len(df)}")
    
    return df

def process_zip_file(zip_path, output_path):
    print(f"\nProcessing {zip_path.name}...")
    
    dtypes = {
        "MMSI": pl.Utf8,
        "SOG": pl.Float64,
        "COG": pl.Float64,
        "Longitude": pl.Float64,
        "Latitude": pl.Float64,
        "# Timestamp": pl.Utf8,
        "Type of mobile": pl.Utf8,
        "ROT": pl.Float64,
        "Width": pl.Float64,
        "Length": pl.Float64,
        "Ship type": pl.Utf8,
    }
    
    frames = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        csv_files = [name for name in zf.namelist() if name.endswith('.csv')]
        print(f"  Found {len(csv_files)} CSV file(s) in zip")
        
        for csv_name in csv_files:
            try:
                with zf.open(csv_name) as csv_file:
                    csv_bytes = io.BytesIO(csv_file.read())
                    tmp = pl.read_csv(csv_bytes, columns=list(dtypes.keys()), schema_overrides=dtypes, ignore_errors=True)
                    tmp = tmp.with_columns(pl.lit(csv_name).alias("source_file"))
                    frames.append(tmp)
            except Exception as e:
                print(f"  Failed to read {csv_name}: {e}")
    
    if not frames:
        print("  No valid CSV files found")
        return
    
    df = pl.concat(frames, how="vertical")
    print(f"  Loaded {len(frames)} CSV file(s). Total rows: {len(df)}")
    
    df = clean_dataframe(df)
    
    if df.is_empty():
        print("  No data remaining after cleaning")
        return
    
    print(f"\n  Final statistics:")
    print(f"  Unique vessels (MMSI): {df['MMSI'].n_unique()}")
    print(f"  Total datapoints: {len(df)}")
    print(f"  Average points per vessel: {len(df) / df['MMSI'].n_unique():.1f}")
    
    segment_counts = df.select([pl.col('MMSI'), pl.col('Segment')]).unique().height
    print(f"  Total segments: {segment_counts}")
    
    df.write_parquet(output_path, compression=config.COMPRESSION)
    
    file_size_bytes = output_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"  Saved to {output_path.name}")
    print(f"  File size: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")

def main():
    data_dir = Path("data")
    
    zip_files = sorted(data_dir.glob("*.zip"))
    print(f"Found {len(zip_files)} zip file(s) in {data_dir}")
    
    if not zip_files:
        print("No zip files to process")
        return
    
    processed = 0
    skipped = 0
    
    for zip_path in zip_files:
        output_path = data_dir / (zip_path.stem + ".parquet")
        
        if output_path.exists():
            print(f"\nSkipping {zip_path.name} - {output_path.name} already exists")
            skipped += 1
            continue
        
        try:
            process_zip_file(zip_path, output_path)
            processed += 1
        except Exception as e:
            print(f"  ERROR processing {zip_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total zip files: {len(zip_files)}")
    print(f"Processed: {processed}")
    print(f"Skipped (already exists): {skipped}")
    print(f"Failed: {len(zip_files) - processed - skipped}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal time: {end_time - start_time:.2f} seconds")

