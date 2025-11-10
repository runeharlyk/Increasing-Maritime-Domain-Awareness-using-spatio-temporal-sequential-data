import pandas as pd
from pathlib import Path
import time
import data_config as config

def main():
    data_dir = Path("data")
    
    print("Loading CSV files...")
    csv_files = sorted(data_dir.glob("*.csv"))
    
    dtypes = {
        "Type of mobile": "object",
        "MMSI": "object",
        "Latitude": float,
        "Longitude": float,
        "ROT": float,
        "SOG": float,
        "COG": float,
        "Width": float,
        "Length": float,
        "Ship type": "object",
        "# Timestamp": "object",
    }
    
    frames = []
    for path in csv_files:
        try:
            tmp = pd.read_csv(path, usecols=list(dtypes.keys()), dtype=dtypes, low_memory=False)
            tmp["source_file"] = path.name
            frames.append(tmp)
        except Exception as e:
            print(f"Failed to read {path.name}: {e}")
    
    if frames:
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.DataFrame()
    
    print(f"Loaded {len(frames)} file(s). Total rows: {len(df)}")
    
    # Apply bounding box filter (remove geographic errors)
    print("\nApplying geographic bounding box filter...")
    north, west, south, east = config.BOUNDING_BOX
    before = len(df)
    df = df[
        (df["Latitude"] <= north)
        & (df["Latitude"] >= south)
        & (df["Longitude"] >= west)
        & (df["Longitude"] <= east)
    ]
    print(f"Removed {before - len(df)} rows outside bounding box. Rows now: {len(df)}")
    
    # Filter by vessel class
    print("\nFiltering by vessel class...")
    before = len(df)
    df = df[df["Type of mobile"].isin(config.VESSEL_CLASSES)]
    print(f"Kept {'/'.join(config.VESSEL_CLASSES)} vessels. Rows now: {len(df)}")
    
    # MMSI format validation
    print("\nValidating MMSI format...")
    before = len(df)
    df = df[df["MMSI"].str.len() == config.MMSI_LENGTH]  # Adhere to MMSI format
    df = df[df["MMSI"].str[:3].astype(int).between(config.MMSI_MID_MIN, config.MMSI_MID_MAX)]  # Adhere to MID standard
    print(f"Removed {before - len(df)} rows with invalid MMSI. Rows now: {len(df)}")
    
    # Parse timestamp
    print("\nParsing timestamps...")
    if '# Timestamp' in df.columns:
        df = df.rename(columns={"# Timestamp": "Timestamp"})
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"], 
        format="%d/%m/%Y %H:%M:%S", 
        errors="coerce"
    )
    # Remove rows with invalid timestamps
    timestamp_na = df["Timestamp"].isna().sum()
    if timestamp_na > 0:
        print(f"Removing {timestamp_na} rows with invalid timestamps")
        df = df.dropna(subset=["Timestamp"])
    print(f"Timestamp range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    
    # Remove duplicates
    print("\nRemoving duplicates...")
    before = len(df)
    df = df.drop_duplicates(["Timestamp", "MMSI"], keep="first")
    print(f"Removed {before - len(df)} duplicate rows. Rows now: {len(df)}")
    
    # Track filtering function
    def track_filter(g):
        len_filt = len(g) > config.TRACK_MIN_LENGTH
        sog_filt = config.TRACK_MIN_SOG <= g["SOG"].max() <= config.TRACK_MAX_SOG
        time_filt = (
            g["Timestamp"].max() - g["Timestamp"].min()
        ).total_seconds() >= config.TRACK_MIN_TIMESPAN
        return len_filt and sog_filt and time_filt
    
    # Apply track filtering
    print(f"\nApplying track filtering (length > {config.TRACK_MIN_LENGTH}, SOG {config.TRACK_MIN_SOG}-{config.TRACK_MAX_SOG} knots, timespan >= {config.TRACK_MIN_TIMESPAN/3600:.0f} hour)...")
    before = len(df)
    unique_before = df["MMSI"].nunique()
    df = df.groupby("MMSI").filter(track_filter)
    df = df.sort_values(["MMSI", "Timestamp"])
    print(f"Removed {before - len(df)} rows from invalid tracks. Rows now: {len(df)}")
    print(f"Unique vessels: {unique_before} -> {df['MMSI'].nunique()}")
    
    # Divide tracks into segments based on time gaps
    print(f"\nCreating segments based on time gaps (>= {config.SEGMENT_TIME_GAP/60:.0f} minutes)...")
    df["Segment"] = df.groupby("MMSI")["Timestamp"].transform(
        lambda x: (x.diff().dt.total_seconds().fillna(0) >= config.SEGMENT_TIME_GAP).cumsum()
    )
    
    # Apply segment filtering
    print("Applying segment filtering...")
    before = len(df)
    df = df.groupby(["MMSI", "Segment"]).filter(track_filter)
    df = df.reset_index(drop=True)
    print(f"Removed {before - len(df)} rows from invalid segments. Rows now: {len(df)}")
    
    # Convert SOG from knots to m/s
    print("\nConverting SOG from knots to m/s...")
    df["SOG"] = config.KNOTS_TO_MS * df["SOG"]
    
    # Filter by ship type (if available)
    if "Ship type" in df.columns:
        print("\nFiltering by ship type...")
        ship_type_counts = df["Ship type"].value_counts(dropna=False)
        print(f"Available ship types:\n{ship_type_counts}")
        
        before = len(df)
        df = df[df["Ship type"].isin(config.SHIP_TYPES)].reset_index(drop=True)
        print(f"Kept {'/'.join(config.SHIP_TYPES)} only. Removed {before - len(df)} rows. Rows now: {len(df)}")
    
    # Report final statistics
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    mmsi_counts = df["MMSI"].value_counts().sort_values(ascending=False)
    print(f"Unique vessels (MMSI): {mmsi_counts.size}")
    print(f"Total datapoints: {len(df)}")
    print(f"Average points per vessel: {len(df) / mmsi_counts.size:.1f}")
    print(f"\nTop 20 vessels by datapoint count:")
    print(mmsi_counts.head(20))
    
    # Count segments per vessel
    segment_counts = df.groupby("MMSI")["Segment"].nunique()
    print(f"\nTotal segments: {df.groupby(['MMSI', 'Segment']).ngroups}")
    print(f"Average segments per vessel: {segment_counts.mean():.1f}")
    
    # Save to parquet
    out_path = data_dir / config.OUTPUT_FILE
    df.to_parquet(out_path, index=False, engine="pyarrow", compression=config.COMPRESSION)
    
    # Report file size
    file_size_bytes = out_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"\nSaved {len(df)} rows to {out_path}")
    print(f"File size: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")


