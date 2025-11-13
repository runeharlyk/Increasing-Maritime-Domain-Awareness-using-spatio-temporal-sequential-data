import polars as pl
from pathlib import Path
import time
import numpy as np
import data_config as config


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    lon1_rad = np.radians(lon1)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance


def main():
    data_dir = Path("data")

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

    print("Loading CSV files...")
    csv_files = sorted(data_dir.glob("*.csv"))

    frames = []
    for path in csv_files:
        try:
            tmp = pl.read_csv(path, columns=list(dtypes.keys()), schema_overrides=dtypes, ignore_errors=True)
            tmp = tmp.with_columns(pl.lit(path.name).alias("source_file"))
            frames.append(tmp)
        except Exception as e:
            print(f"Failed to read {path.name}: {e}")

    if frames:
        df = pl.concat(frames, how="vertical")
    else:
        df = pl.DataFrame()

    print(f"Loaded {len(frames)} file(s). Total rows: {len(df)}")

    # Apply bounding box filter (remove geographic errors)
    print("\nApplying geographic bounding box filter...")
    north, west, south, east = config.BOUNDING_BOX
    before = len(df)
    df = df.filter(
        (pl.col("Latitude") <= north)
        & (pl.col("Latitude") >= south)
        & (pl.col("Longitude") >= west)
        & (pl.col("Longitude") <= east)
    )
    print(f"Removed {before - len(df)} rows outside bounding box. Rows now: {len(df)}")

    # Filter by vessel class
    print("\nFiltering by vessel class...")
    before = len(df)
    df = df.filter(pl.col("Type of mobile").is_in(config.VESSEL_CLASSES))
    print(f"Kept {'/'.join(config.VESSEL_CLASSES)} vessels. Rows now: {len(df)}")

    # MMSI format validation
    print("\nValidating MMSI format...")
    before = len(df)
    df = df.filter(pl.col("MMSI").str.len_chars() == config.MMSI_LENGTH)  # Adhere to MMSI format
    df = df.filter(
        pl.col("MMSI").str.slice(0, 3).cast(pl.Int32).is_between(config.MMSI_MID_MIN, config.MMSI_MID_MAX)
    )  # Adhere to MID standard
    print(f"Removed {before - len(df)} rows with invalid MMSI. Rows now: {len(df)}")

    # Parse timestamp
    print("\nParsing timestamps...")
    if "# Timestamp" in df.columns:
        df = df.rename({"# Timestamp": "Timestamp"})
    df = df.with_columns(
        pl.col("Timestamp").str.to_datetime(format="%d/%m/%Y %H:%M:%S", strict=False).alias("Timestamp")
    )
    # Remove rows with invalid timestamps
    timestamp_na = df.filter(pl.col("Timestamp").is_null()).height
    if timestamp_na > 0:
        print(f"Removing {timestamp_na} rows with invalid timestamps")
        df = df.filter(pl.col("Timestamp").is_not_null())
    print(f"Timestamp range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")

    # Remove duplicates
    print("\nRemoving duplicates...")
    before = len(df)
    df = df.unique(subset=["Timestamp", "MMSI"], keep="first")
    print(f"Removed {before - len(df)} duplicate rows. Rows now: {len(df)}")

    # Apply track filtering
    print(
        f"\nApplying track filtering (length > {config.TRACK_MIN_LENGTH}, SOG {config.TRACK_MIN_SOG}-{config.TRACK_MAX_SOG} knots, timespan >= {config.TRACK_MIN_TIMESPAN/3600:.0f} hour)..."
    )
    before = len(df)
    unique_before = df["MMSI"].n_unique()
    track_stats = df.group_by("MMSI").agg(
        [
            pl.len().alias("count"),
            pl.col("SOG").max().alias("max_sog"),
            (pl.col("Timestamp").max() - pl.col("Timestamp").min()).dt.total_seconds().alias("timespan_seconds"),
        ]
    )
    valid_tracks = track_stats.filter(
        (pl.col("count") > config.TRACK_MIN_LENGTH)
        & (pl.col("max_sog") >= config.TRACK_MIN_SOG)
        & (pl.col("max_sog") <= config.TRACK_MAX_SOG)
        & (pl.col("timespan_seconds") >= config.TRACK_MIN_TIMESPAN)
    ).select("MMSI")
    df = df.join(valid_tracks, on="MMSI", how="inner")
    df = df.sort(["MMSI", "Timestamp"])
    print(f"Removed {before - len(df)} rows from invalid tracks. Rows now: {len(df)}")
    print(f"Unique vessels: {unique_before} -> {df['MMSI'].n_unique()}")

    # Divide tracks into segments based on time gaps
    print(f"\nCreating segments based on time gaps (>= {config.SEGMENT_TIME_GAP/60:.0f} minutes)...")
    # Process each MMSI separately to avoid window expression issues after join
    unique_mmsi = df["MMSI"].unique().to_list()

    segment_dfs = []
    for mmsi in unique_mmsi:
        mmsi_df = df.filter(pl.col("MMSI") == mmsi)

        # Compute time differences
        mmsi_df = mmsi_df.with_columns(
            [pl.col("Timestamp").diff().fill_null(pl.duration(seconds=0)).alias("time_diff")]
        )

        # Mark new segments (time gap >= configured threshold)
        mmsi_df = mmsi_df.with_columns(
            [(pl.col("time_diff").dt.total_seconds() >= config.SEGMENT_TIME_GAP).cast(pl.Int32).alias("segment_marker")]
        )

        # Create segment IDs
        mmsi_df = mmsi_df.with_columns([pl.col("segment_marker").cum_sum().alias("Segment")])

        mmsi_df = mmsi_df.drop(["time_diff", "segment_marker"])
        segment_dfs.append(mmsi_df)

    df = pl.concat(segment_dfs, how="vertical")
    df = df.sort(["MMSI", "Timestamp"])

    # Apply segment filtering
    print("Applying segment filtering...")
    before = len(df)
    segment_stats = df.group_by(["MMSI", "Segment"]).agg(
        [
            pl.len().alias("count"),
            pl.col("SOG").max().alias("max_sog"),
            (pl.col("Timestamp").max() - pl.col("Timestamp").min()).dt.total_seconds().alias("timespan_seconds"),
        ]
    )
    valid_segments = segment_stats.filter(
        (pl.col("count") > config.TRACK_MIN_LENGTH)
        & (pl.col("max_sog") >= config.TRACK_MIN_SOG)
        & (pl.col("max_sog") <= config.TRACK_MAX_SOG)
        & (pl.col("timespan_seconds") >= config.TRACK_MIN_TIMESPAN)
    ).select(["MMSI", "Segment"])
    df = df.join(valid_segments, on=["MMSI", "Segment"], how="inner")
    print(f"Removed {before - len(df)} rows from invalid segments. Rows now: {len(df)}")

    # Filter by point-to-point speed
    if config.SPEED_ANOMALY_ACTION != "keep":
        print(
            f"\nFiltering by point-to-point speed (max {config.MAX_POINT_TO_POINT_SPEED_KMH} km/h, action: {config.SPEED_ANOMALY_ACTION})..."
        )
        before = len(df)

        segment_dfs = []
        unique_mmsi = df["MMSI"].unique().to_list()

        for mmsi in unique_mmsi:
            mmsi_df = df.filter(pl.col("MMSI") == mmsi)
            segments = mmsi_df["Segment"].unique().to_list()

            for segment in segments:
                seg_df = mmsi_df.filter(pl.col("Segment") == segment).sort("Timestamp")

                if len(seg_df) < 2:
                    segment_dfs.append(seg_df)
                    continue

                if config.SPEED_ANOMALY_ACTION == "drop":
                    changed = True
                    while changed and len(seg_df) >= 2:
                        changed = False

                        lats = seg_df["Latitude"].to_numpy()
                        lons = seg_df["Longitude"].to_numpy()
                        timestamps = seg_df["Timestamp"].to_list()

                        distances_km = haversine_distance(lats[:-1], lons[:-1], lats[1:], lons[1:])
                        time_diffs_hours = np.array(
                            [
                                (timestamps[i + 1] - timestamps[i]).total_seconds() / 3600.0
                                for i in range(len(timestamps) - 1)
                            ]
                        )
                        speeds_kmh = np.where(time_diffs_hours > 0, distances_km / time_diffs_hours, 0)

                        violations = speeds_kmh > config.MAX_POINT_TO_POINT_SPEED_KMH
                        if violations.any():
                            keep_indices = [0] + [i + 1 for i in range(len(violations)) if not violations[i]]
                            seg_df = seg_df[keep_indices]
                            changed = True

                elif config.SPEED_ANOMALY_ACTION == "drop_both":
                    lats = seg_df["Latitude"].to_numpy()
                    lons = seg_df["Longitude"].to_numpy()
                    timestamps = seg_df["Timestamp"].to_list()

                    distances_km = haversine_distance(lats[:-1], lons[:-1], lats[1:], lons[1:])
                    time_diffs_hours = np.array(
                        [
                            (timestamps[i + 1] - timestamps[i]).total_seconds() / 3600.0
                            for i in range(len(timestamps) - 1)
                        ]
                    )
                    speeds_kmh = np.where(time_diffs_hours > 0, distances_km / time_diffs_hours, 0)

                    violations = speeds_kmh > config.MAX_POINT_TO_POINT_SPEED_KMH
                    drop_indices = set()
                    for i in range(len(violations)):
                        if violations[i]:
                            drop_indices.add(i)
                            drop_indices.add(i + 1)
                    keep_indices = [i for i in range(len(seg_df)) if i not in drop_indices]
                    seg_df = seg_df[keep_indices]

                segment_dfs.append(seg_df)

        df = pl.concat(segment_dfs, how="vertical")
        df = df.sort(["MMSI", "Timestamp"])
        print(f"Removed {before - len(df)} rows with excessive point-to-point speed. Rows now: {len(df)}")

    # Convert SOG from knots to m/s
    print("\nConverting SOG from knots to m/s...")
    df = df.with_columns((pl.col("SOG") * config.KNOTS_TO_MS).alias("SOG"))

    # Filter by ship type (if available)
    if "Ship type" in df.columns:
        print("\nFiltering by ship type...")
        ship_type_counts = df.group_by("Ship type").agg(pl.len().alias("count")).sort("count", descending=True)
        print(f"Available ship types: {ship_type_counts['Ship type'].to_list()}")

        before = len(df)
        df = df.filter(pl.col("Ship type").is_in(config.SHIP_TYPES))
        print(f"Kept {'/'.join(config.SHIP_TYPES)} only. Removed {before - len(df)} rows. Rows now: {len(df)}")

    # Report final statistics
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    mmsi_counts = df.group_by("MMSI").agg(pl.len().alias("count")).sort("count", descending=True)
    print(f"Unique vessels (MMSI): {df['MMSI'].n_unique()}")
    print(f"Total datapoints: {len(df)}")
    print(f"Average points per vessel: {len(df) / df['MMSI'].n_unique():.1f}")
    top_vessel = mmsi_counts.row(0)
    print(f"Top vessel: MMSI {top_vessel[0]} with {top_vessel[1]} datapoints")

    # Count segments per vessel
    segment_counts = df.group_by("MMSI").agg(pl.col("Segment").n_unique().alias("segments"))
    print(f"\nTotal segments: {df.select([pl.col('MMSI'), pl.col('Segment')]).unique().height}")
    print(f"Average segments per vessel: {segment_counts['segments'].mean():.1f}")

    # Save to parquet
    out_path = data_dir / config.OUTPUT_FILE
    df.write_parquet(out_path, compression=config.COMPRESSION)

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
