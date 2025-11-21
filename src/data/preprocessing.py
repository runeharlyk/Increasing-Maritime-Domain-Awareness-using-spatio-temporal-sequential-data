import polars as pl
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.utils import config

def merge_cross_file_segments(df, max_time_gap_minutes=config.SEGMENT_TIME_GAP / 60):
    print("\nMerging continuous segments across file boundaries...")

    df = df.sort(["MMSI", "FileIndex", "Timestamp"])
    segment_stats = (
        df.group_by(["MMSI", "FileIndex", "Segment"])
        .agg([
            pl.col("Timestamp").min().alias("seg_start"),
            pl.col("Timestamp").max().alias("seg_end"),
        ])
        .sort(["MMSI", "FileIndex", "seg_start"])
    )
    
    segment_stats = segment_stats.with_columns([
        pl.col("seg_end").shift(1).over("MMSI").alias("prev_seg_end"),
        pl.col("FileIndex").shift(1).over("MMSI").alias("prev_file_idx"),
    ])
    
    segment_stats = segment_stats.with_columns([
        ((pl.col("seg_start") - pl.col("prev_seg_end")).dt.total_minutes()).alias("time_gap_minutes"),
    ])
    
    segment_stats = segment_stats.with_columns([
        (
            (pl.col("FileIndex") != pl.col("prev_file_idx")) &
            (pl.col("time_gap_minutes") >= 0) &
            (pl.col("time_gap_minutes") <= max_time_gap_minutes)
        ).fill_null(False).not_().cast(pl.Int32).alias("is_new_segment")
    ])
    
    segment_stats = segment_stats.with_columns([
        pl.col("is_new_segment").cum_sum().over("MMSI").alias("GlobalSegment")
    ])
    
    df = df.join(
        segment_stats.select(["MMSI", "FileIndex", "Segment", "GlobalSegment"]),
        on=["MMSI", "FileIndex", "Segment"],
        how="left"
    )

    original_count = segment_stats.height
    global_count = df.select(["MMSI", "GlobalSegment"]).n_unique()
    merged_count = original_count - global_count

    print(f"  Original (MMSI, FileIndex, Segment) combinations: {original_count}")
    print(f"  Merged (MMSI, GlobalSegment) combinations: {global_count}")
    print(f"  Merged {merged_count} segment pairs across file boundaries")
    print(f"  Time gap threshold: {max_time_gap_minutes} minutes")

    return df


def load_and_prepare_data(data_dir, max_time_gap_minutes=config.SEGMENT_TIME_GAP / 60):
    print("Loading data...")

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    parquet_files = sorted(data_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    print(f"Found {len(parquet_files)} parquet files")

    dfs = []
    for idx, file in enumerate(tqdm(parquet_files, desc="Loading files")):
        try:
            df_temp = pl.read_parquet(file)
            df_temp = df_temp.with_columns(pl.lit(idx).alias("FileIndex"))
            dfs.append(df_temp)
        except Exception as e:
            print(f"Warning: Failed to load {file.name}: {e}")

    if not dfs:
        raise ValueError("No data loaded successfully")

    df = pl.concat(dfs, how="vertical")
    print(f"Loaded {len(df)} rows from {len(dfs)} files")

    required_cols = ["MMSI", "Latitude", "Longitude", "SOG", "COG", "Segment", "Timestamp"]
    missing = [col for col in required_cols if col not in df.columns]
    assert len(missing) == 0, f"Missing required columns: {missing}"

    df = df.sort(["MMSI", "Timestamp"])

    df = merge_cross_file_segments(df, max_time_gap_minutes=max_time_gap_minutes)

    return df


def create_sequences(df, input_hours, output_hours, sampling_rate):
    print(f"\nCreating sequences ({input_hours}h input -> {output_hours}h output)...")

    input_timesteps = int(input_hours * 60 / sampling_rate)
    output_timesteps = int(output_hours * 60 / sampling_rate)
    min_length = input_timesteps + output_timesteps

    feature_cols = ["Latitude", "Longitude", "SOG", "COG"]

    print("Resampling and adding features with Polars...")

    if "GlobalSegment" not in df.columns:
        raise ValueError("'GlobalSegment' column not found. Ensure merge_cross_file_segments() was called in load_and_prepare_data()")

    agg_cols = [
        pl.col("Latitude").first(),
        pl.col("Longitude").first(),
        pl.col("SOG").first(),
        pl.col("COG").first(),
        pl.col("GlobalSegment").first(),
    ]

    df_processed = (
        df.sort(["MMSI", "Timestamp"])
        .group_by_dynamic("Timestamp", every=f"{sampling_rate}m", by="MMSI")
        .agg(agg_cols)
        .drop_nulls(subset=feature_cols)
        .with_columns(
            [
                (2 * np.pi * pl.col("Timestamp").dt.hour() / 24.0).sin().alias("hour_sin"),
                (2 * np.pi * pl.col("Timestamp").dt.hour() / 24.0).cos().alias("hour_cos"),
                (2 * np.pi * pl.col("Timestamp").dt.weekday() / 7.0).sin().alias("day_of_week_sin"),
                (2 * np.pi * pl.col("Timestamp").dt.weekday() / 7.0).cos().alias("day_of_week_cos"),
            ]
        )
        .with_columns(
            [
                (pl.col("COG") * np.pi / 180.0).sin().alias("COG_sin"),
                (pl.col("COG") * np.pi / 180.0).cos().alias("COG_cos"),
            ]
        )
        .with_columns(
            [
                pl.col("SOG").diff().over(["MMSI", "GlobalSegment"]).fill_null(0).alias("SOG_diff"),
                (((pl.col("COG").diff().over(["MMSI", "GlobalSegment"]).fill_null(0) + 180) % 360) - 180).alias(
                    "COG_diff_raw"
                ),
            ]
        )
        .with_columns(
            [
                (pl.col("COG_diff_raw") * np.pi / 180.0).sin().alias("COG_diff_sin"),
                (pl.col("COG_diff_raw") * np.pi / 180.0).cos().alias("COG_diff_cos"),
            ]
        )
    )

    enhanced_features = [
        "Latitude",
        "Longitude",
        "SOG",
        "COG_sin",
        "COG_cos",
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "SOG_diff",
        "COG_diff_sin",
        "COG_diff_cos",
    ]

    sequences = []
    targets = []
    mmsi_labels = []

    stride = 1
    print(f"Creating sequences with stride={stride}")

    group_cols = ["MMSI", "GlobalSegment"]
    print(f"Processing by {group_cols} (merged continuous segments across files)...")

    segment_groups = df_processed.partition_by(group_cols, as_dict=True)

    skipped_irregular = 0
    for group_key, vessel_data in tqdm(segment_groups.items(), desc="Processing segments"):
        n_points = len(vessel_data)

        if n_points < min_length:
            continue

        data_array = vessel_data.select(enhanced_features).to_numpy()
        lat_lon = vessel_data.select(["Latitude", "Longitude"]).to_numpy()
        timestamps = vessel_data.select("Timestamp").to_numpy().flatten()

        mmsi = group_key[0] if isinstance(group_key, tuple) else group_key

        for i in range(0, n_points - min_length + 1, stride):
            input_seq = data_array[i : i + input_timesteps]
            output_seq = lat_lon[i + input_timesteps : i + input_timesteps + output_timesteps]
            seq_timestamps = timestamps[i : i + min_length]

            if np.isnan(input_seq).any() or np.isnan(output_seq).any():
                continue

            time_diffs = np.diff(seq_timestamps).astype("timedelta64[m]").astype(int)
            expected_diff = sampling_rate
            if not np.all(np.abs(time_diffs - expected_diff) <= 1):
                skipped_irregular += 1
                continue

            sequences.append(input_seq)
            targets.append(output_seq.flatten())
            mmsi_labels.append(mmsi)

    if skipped_irregular > 0:
        print(f"  Skipped {skipped_irregular} sequences with irregular time spacing")

    sequences = np.array(sequences)
    targets = np.array(targets)
    mmsi_labels = np.array(mmsi_labels)

    print(f"Created {len(sequences)} sequences from {len(np.unique(mmsi_labels))} unique vessels")
    print(f"  Input shape: {sequences.shape}")
    print(f"  Target shape: {targets.shape}")
    print(f"  Stride: {stride} timesteps ({stride * sampling_rate} minutes)")
    overlap_pct = max(0, (input_timesteps - stride) / input_timesteps * 100)
    print(f"  Sequence overlap: ~{overlap_pct:.1f}%")

    return sequences, targets, mmsi_labels, enhanced_features


def split_by_vessel(sequences, targets, mmsi_labels, train_ratio=0.7, val_ratio=0.15, random_seed=42):
    unique_mmsi = np.unique(mmsi_labels)
    n_vessels = len(unique_mmsi)

    np.random.seed(random_seed)
    shuffled_mmsi = np.random.permutation(unique_mmsi)

    train_end = int(train_ratio * n_vessels)
    val_end = int((train_ratio + val_ratio) * n_vessels)

    train_mmsi = set(shuffled_mmsi[:train_end])
    val_mmsi = set(shuffled_mmsi[train_end:val_end])
    test_mmsi = set(shuffled_mmsi[val_end:])

    print(f"\nVessel-based split:")
    print(f"  Train vessels: {len(train_mmsi)} ({train_ratio*100:.0f}%)")
    print(f"  Val vessels: {len(val_mmsi)} ({val_ratio*100:.0f}%)")
    print(f"  Test vessels: {len(test_mmsi)} ({(1-train_ratio-val_ratio)*100:.0f}%)")

    train_mask = np.array([mmsi in train_mmsi for mmsi in mmsi_labels])
    val_mask = np.array([mmsi in val_mmsi for mmsi in mmsi_labels])
    test_mask = np.array([mmsi in test_mmsi for mmsi in mmsi_labels])

    X_train = sequences[train_mask]
    X_val = sequences[val_mask]
    X_test = sequences[test_mask]
    y_train = targets[train_mask]
    y_val = targets[val_mask]
    y_test = targets[test_mask]

    print(f"  Train sequences: {len(X_train)}")
    print(f"  Val sequences: {len(X_val)}")
    print(f"  Test sequences: {len(X_test)}")

    train_vessels_in_data = set(mmsi_labels[train_mask])
    val_vessels_in_data = set(mmsi_labels[val_mask])
    test_vessels_in_data = set(mmsi_labels[test_mask])

    overlap_train_val = train_vessels_in_data & val_vessels_in_data
    overlap_train_test = train_vessels_in_data & test_vessels_in_data
    overlap_val_test = val_vessels_in_data & test_vessels_in_data

    if overlap_train_val or overlap_train_test or overlap_val_test:
        print(f"  ⚠️  WARNING: Vessel overlap detected!")
        if overlap_train_val:
            print(f"    Train-Val overlap: {len(overlap_train_val)} vessels")
        if overlap_train_test:
            print(f"    Train-Test overlap: {len(overlap_train_test)} vessels")
        if overlap_val_test:
            print(f"    Val-Test overlap: {len(overlap_val_test)} vessels")
    else:
        print(f"  ✅ No vessel overlap - proper split confirmed!")

    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize_data(X_train, X_val, X_test, y_train, y_val, y_test):
    print("\nNormalizing data...")

    n_samples, n_timesteps, n_features = X_train.shape

    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    features_to_normalize = [0, 1, 2, 9]
    features_not_normalized = [3, 4, 5, 6, 7, 8, 10, 11]

    input_scaler = StandardScaler()
    X_train_norm = X_train[:, :, features_to_normalize].reshape(-1, len(features_to_normalize))
    input_scaler.fit(X_train_norm)

    X_train_scaled[:, :, features_to_normalize] = input_scaler.transform(X_train_norm).reshape(
        n_samples, n_timesteps, len(features_to_normalize)
    )
    X_val_scaled[:, :, features_to_normalize] = input_scaler.transform(
        X_val[:, :, features_to_normalize].reshape(-1, len(features_to_normalize))
    ).reshape(X_val.shape[0], n_timesteps, len(features_to_normalize))
    X_test_scaled[:, :, features_to_normalize] = input_scaler.transform(
        X_test[:, :, features_to_normalize].reshape(-1, len(features_to_normalize))
    ).reshape(X_test.shape[0], n_timesteps, len(features_to_normalize))

    output_scaler = StandardScaler()
    output_timesteps = y_train.shape[1] // 2
    y_train_reshaped = y_train.reshape(-1, 2)
    output_scaler.fit(y_train_reshaped)

    y_train_scaled = output_scaler.transform(y_train_reshaped).reshape(y_train.shape[0], -1)
    y_val_scaled = output_scaler.transform(y_val.reshape(-1, 2)).reshape(y_val.shape[0], -1)
    y_test_scaled = output_scaler.transform(y_test.reshape(-1, 2)).reshape(y_test.shape[0], -1)

    print(f"  Input features: {n_features}")
    print(f"  Features normalized (Lat, Lon, SOG, SOG_diff): {features_to_normalize}")
    print(f"  Features NOT normalized (sin/cos): {features_not_normalized}")
    print(f"  Input scaler - mean: {input_scaler.mean_}")
    print(f"  Input scaler - scale: {input_scaler.scale_}")
    print(f"  Output scaler - mean: {output_scaler.mean_}")
    print(f"  Output scaler - scale: {output_scaler.scale_}")

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train_scaled,
        y_val_scaled,
        y_test_scaled,
        input_scaler,
        output_scaler,
    )
