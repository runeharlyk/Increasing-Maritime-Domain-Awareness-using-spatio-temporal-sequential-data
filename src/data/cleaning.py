import polars as pl
import numpy as np
import zipfile
import io
from pathlib import Path
from multiprocessing import Pool

from src.utils.geo import haversine_distance
from src.utils import config


def clean_dataframe(df):
    if df.is_empty():
        return df

    print(f"  Initial rows: {df.height}")

    print("  Filtering by ship type...")
    before = df.height
    df = df.filter(pl.col("Ship type").is_in(config.SHIP_TYPES))
    print(f"  Kept {'/'.join(config.SHIP_TYPES)} only. Removed {before - df.height} rows. Rows now: {df.height}")

    # Filter by vessel class
    print("  Filtering by vessel class...")
    before = df.height
    df = df.filter(pl.col("Type of mobile").is_in(config.VESSEL_CLASSES))
    print(f"  Kept {'/'.join(config.VESSEL_CLASSES)} vessels. Rows now: {df.height}")

    # Apply bounding box filter
    print("  Applying geographic bounding box filter...")
    north, west, south, east = config.BOUNDING_BOX
    before = df.height
    df = df.filter(
        (pl.col("Latitude") <= north)
        & (pl.col("Latitude") >= south)
        & (pl.col("Longitude") >= west)
        & (pl.col("Longitude") <= east)
    )
    print(f"  Removed {before - len(df)} rows outside bounding box. Rows now: {len(df)}")
    print(f"  Removed {before - df.height} rows outside bounding box. Rows now: {df.height}")

    # MMSI format validation
    print("  Validating MMSI format...")
    before = df.height
    df = df.filter(
        (pl.col("MMSI").str.len_chars() == config.MMSI_LENGTH)
        & (pl.col("MMSI").str.slice(0, 3).cast(pl.Int32).is_between(config.MMSI_MID_MIN, config.MMSI_MID_MAX))
    )
    print(f"  Removed {before - df.height} rows with invalid MMSI. Rows now: {df.height}")

    # Parse timestamp
    print("  Parsing timestamps...")
    df = df.rename({"# Timestamp": "Timestamp"})
    df = df.with_columns(
        pl.col("Timestamp").str.to_datetime(format="%d/%m/%Y %H:%M:%S", strict=False).alias("Timestamp")
    )
    timestamp_na = df.filter(pl.col("Timestamp").is_null()).height
    if timestamp_na > 0:
        print(f"  Removing {timestamp_na} rows with invalid timestamps")
        df = df.filter(pl.col("Timestamp").is_not_null())
    print(f"  Timestamp range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")

    # Remove duplicates
    print("  Removing duplicates...")
    before = df.height
    df = df.unique(subset=["Timestamp", "MMSI"], keep="first")
    print(f"  Removed {before - df.height} duplicate rows. Rows now: {df.height}")

    # Apply track filtering
    print(
        f"  Applying track filtering (length > {config.TRACK_MIN_LENGTH}, SOG {config.TRACK_MIN_SOG}-{config.TRACK_MAX_SOG} knots, timespan >= {config.TRACK_MIN_TIMESPAN/3600:.0f} hour)..."
    )
    before = df.height
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
    df = df.join(valid_tracks, on="MMSI", how="inner").sort(["MMSI", "Timestamp"])
    print(f"  Removed {before - df.height} rows from invalid tracks. Rows now: {df.height}")
    print(f"  Unique vessels: {unique_before} -> {df['MMSI'].n_unique()}")

    # Create segments based on time gaps
    print(f"  Creating segments based on time gaps (>= {config.SEGMENT_TIME_GAP/60:.0f} minutes)...")
    unique_mmsi = df["MMSI"].unique().to_list()

    segment_dfs = []
    for mmsi in unique_mmsi:
        mmsi_df = df.filter(pl.col("MMSI") == mmsi)

        mmsi_df = mmsi_df.with_columns(
            [pl.col("Timestamp").diff().fill_null(pl.duration(seconds=0)).alias("time_diff")]
        )

        mmsi_df = mmsi_df.with_columns(
            [(pl.col("time_diff").dt.total_seconds() >= config.SEGMENT_TIME_GAP).cast(pl.Int32).alias("segment_marker")]
        )

        mmsi_df = mmsi_df.with_columns([pl.col("segment_marker").cum_sum().alias("Segment")])

        mmsi_df = mmsi_df.drop(["time_diff", "segment_marker"])
        segment_dfs.append(mmsi_df)

    df = pl.concat(segment_dfs, how="vertical")
    df = df.sort(["MMSI", "Timestamp"])

    # Apply segment filtering
    print("  Applying segment filtering...")
    before = df.height
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
    print(f"  Removed {before - df.height} rows from invalid segments. Rows now: {df.height}")

    # Filter by point-to-point speed
    if config.SPEED_ANOMALY_ACTION != "keep":
        print(
            f"  Filtering by point-to-point speed (max {config.MAX_POINT_TO_POINT_SPEED_KMH} km/h, action: {config.SPEED_ANOMALY_ACTION})..."
        )
        before = df.height

        df = df.group_by(["MMSI", "Segment"], maintain_order=True).apply(filter_segment).sort(["MMSI", "Timestamp"])
        print(f"  Removed {before - df.height} rows with excessive point-to-point speed. Rows now: {df.height}")

    # Convert SOG from knots to m/s
    print("  Converting SOG from knots to m/s...")
    df = df.with_columns((pl.col("SOG") * config.KNOTS_TO_MS).alias("SOG"))

    return df


def filter_segment(seg_df: pl.DataFrame) -> pl.DataFrame:
    seg_df = seg_df.sort("Timestamp")
    if seg_df.height < 2:
        return seg_df

    lats = seg_df["Latitude"].to_numpy()
    lons = seg_df["Longitude"].to_numpy()
    timestamps = seg_df["Timestamp"].to_list()

    if config.SPEED_ANOMALY_ACTION == "drop":
        changed = True
        while changed and seg_df.height >= 2:
            lats = seg_df["Latitude"].to_numpy()
            lons = seg_df["Longitude"].to_numpy()
            timestamps = seg_df["Timestamp"].to_list()

            distances_km = haversine_distance(lats[:-1], lons[:-1], lats[1:], lons[1:])
            time_diffs_hours = np.array(
                [(timestamps[i + 1] - timestamps[i]).total_seconds() / 3600.0 for i in range(len(timestamps) - 1)]
            )
            speeds_kmh = np.where(time_diffs_hours > 0, distances_km / time_diffs_hours, 0)

            violations = speeds_kmh > config.MAX_POINT_TO_POINT_SPEED_KMH
            if not violations.any():
                changed = False
            else:
                keep_indices = [0] + [i + 1 for i in range(len(violations)) if not violations[i]]
                seg_df = seg_df[keep_indices]

        return seg_df

    if config.SPEED_ANOMALY_ACTION == "drop_both":
        distances_km = haversine_distance(lats[:-1], lons[:-1], lats[1:], lons[1:])
        time_diffs_hours = np.array(
            [(timestamps[i + 1] - timestamps[i]).total_seconds() / 3600.0 for i in range(len(timestamps) - 1)]
        )
        speeds_kmh = np.where(time_diffs_hours > 0, distances_km / time_diffs_hours, 0)

        violations = speeds_kmh > config.MAX_POINT_TO_POINT_SPEED_KMH
        if not violations.any():
            return seg_df

        drop_indices = set()
        for i, v in enumerate(violations):
            if v:
                drop_indices.add(i)
                drop_indices.add(i + 1)
        keep_indices = [i for i in range(seg_df.height) if i not in drop_indices]
        return seg_df[keep_indices]

    return seg_df


def process_zip_file(args):
    zip_path, output_path = args
    if not isinstance(output_path, Path):
        output_path = Path(output_path)
    try:
        print(f"\n[Worker] Processing {zip_path.name}...")

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

        cleaned_frames = []
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_files = [name for name in zf.namelist() if name.endswith(".csv")]
            print(f"  [{zip_path.name}] Found {len(csv_files)} CSV file(s) in zip")

            for i, csv_name in enumerate(csv_files, 1):
                try:
                    print(f"  [{zip_path.name}] [{i}/{len(csv_files)}] Processing {csv_name}...")
                    with zf.open(csv_name) as csv_file:
                        csv_bytes = io.BytesIO(csv_file.read())
                        tmp = pl.read_csv(
                            csv_bytes, columns=list(dtypes.keys()), schema_overrides=dtypes, ignore_errors=True
                        )
                        tmp = tmp.with_columns(pl.lit(csv_name).alias("source_file"))

                        cleaned = clean_dataframe(tmp)
                        if not cleaned.is_empty():
                            cleaned_frames.append(cleaned)
                        else:
                            print(f"  [{zip_path.name}] No data remaining after cleaning {csv_name}")
                except Exception as e:
                    print(f"  [{zip_path.name}] Failed to read {csv_name}: {e}")

        if not cleaned_frames:
            print(f"  [{zip_path.name}] No valid data after cleaning all CSV files")
            return {"success": False, "zip": zip_path.name, "error": "No valid data"}

        print(f"  [{zip_path.name}] Concatenating {len(cleaned_frames)} cleaned dataframe(s)...")
        df = pl.concat(cleaned_frames, how="vertical")

        if df.is_empty():
            print(f"  [{zip_path.name}] No data remaining after cleaning")
            return {"success": False, "zip": zip_path.name, "error": "No data after cleaning"}

        print(f"  [{zip_path.name}] Final statistics:")
        print(f"  [{zip_path.name}] Unique vessels (MMSI): {df['MMSI'].n_unique()}")
        print(f"  [{zip_path.name}] Total datapoints: {df.height}")
        print(f"  [{zip_path.name}] Average points per vessel: {df.height / df['MMSI'].n_unique():.1f}")

        segment_counts = df.select([pl.col("MMSI"), pl.col("Segment")]).unique().height
        print(f"  [{zip_path.name}] Total segments: {segment_counts}")

        df.write_parquet(output_path, compression=config.COMPRESSION)

        file_size_bytes = output_path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)
        print(f"  [{zip_path.name}] Saved to {output_path.name}")
        print(f"  [{zip_path.name}] File size: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")

        return {"success": True, "zip": zip_path.name, "output": output_path.name}
    except Exception as e:
        print(f"  [{zip_path.name}] ERROR: {e}")
        import traceback

        traceback.print_exc()
        return {"success": False, "zip": zip_path.name, "error": str(e)}


def process_multiple_zip_files(data_dir, num_workers=4):
    data_dir = Path(data_dir)

    zip_files = sorted(data_dir.glob("*.zip"))
    print(f"Found {len(zip_files)} zip file(s) in {data_dir}")
    print(f"Requested {num_workers} worker(s)")

    if not zip_files:
        print("No zip files to process")
        return {"processed": 0, "skipped": 0, "failed": 0}

    tasks = []
    skipped = 0

    for zip_path in zip_files:
        output_path = data_dir / (zip_path.stem + ".parquet")

        if output_path.exists():
            print(f"Skipping {zip_path.name} - {output_path.name} already exists")
            skipped += 1
            continue

        tasks.append((zip_path, output_path))

    if not tasks:
        print("\nNo files to process (all already exist)")
        return {"processed": 0, "skipped": skipped, "failed": 0}

    num_workers = min(num_workers, len(tasks))

    print(f"\nProcessing {len(tasks)} file(s) with {num_workers} worker(s)...\n")

    with Pool(num_workers) as pool:
        results = pool.map(process_zip_file, tasks)

    processed = sum(1 for r in results if r["success"])
    failed = sum(1 for r in results if not r["success"])

    print(f"Total zip files: {len(zip_files)}")
    print(f"Processed: {processed}")
    print(f"Skipped (already exists): {skipped}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed files:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['zip']}: {r.get('error', 'Unknown error')}")

    return {"processed": processed, "skipped": skipped, "failed": failed}
